#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""外部平台数据同步引擎 — 从 ES 拉取线索(作品+评论+意图得分)写入 SQLite。

状态机：idle → running → done/stopped/error（镜像 trainer_engine.TrainerEngine）

同步方式二选一：
  - 按天数：range.createTime 最近 N 天，createTime 升序
  - 按条数：match_all 按 createTime 倒序取最近 N 条，到量即停

流程：
  1. 对每个索引发起 scroll 查询（两方式共用 scroll 分页）
  2. 逐页解析文档，写入 works / annotations 表（INSERT OR IGNORE 去重）
  3. 记录同步批次到 sync_batches，进度通过 status() 轮询

ES 访问仅用 Python stdlib urllib，不新增依赖。
"""
import json
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

from common import (ES_BASE_URL, ES_DEFAULT_DAYS, ES_DEFAULT_LIMIT,
                    ES_INDICES, ES_MAX_DAYS, ES_MAX_LIMIT, ES_MIN_DAYS,
                    ES_MIN_LIMIT, ES_PAGE_SIZE, ES_SCROLL_KEEP)

_INDEX_TO_KEY = {v: k for k, v in ES_INDICES.items()}


def es_test_connection():
    """探测 ES 连通性，返回 (ok, message)。"""
    try:
        req = urllib.request.Request(ES_BASE_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ver = (data.get("version") or {}).get("number", "?")
            return True, f"ES 连接正常 (版本 {ver})"
    except Exception as e:
        return False, f"ES 连接失败: {e}"


class EsSyncEngine:
    """单个同步任务的状态机。一次只跑一个同步。线程安全由调用方(API)保证。"""

    def __init__(self):
        self.goal_id = None
        self.goal_name = ""
        self.indices = []
        self.mode = "days"           # days(按天数) / limit(按条数)
        self.days = ES_DEFAULT_DAYS
        self.limit = None            # 按条数同步时生效
        self.sync_id = None
        self.state = "idle"          # idle / running / done / stopped / error
        self.reason = ""
        self.thread = None
        self._stop = None            # threading.Event
        self.logs = deque(maxlen=500)
        self.progress = {
            "phase": "idle", "index": "", "docs_processed": 0,
            "docs_total": None, "percent": 0, "elapsed": 0,
            "works_inserted": 0, "works_skipped": 0,
            "ann_inserted": 0, "ann_skipped": 0,
        }

    # ---- 日志 ----

    def _log(self, msg):
        ts = datetime.now().strftime("%m-%d %H:%M:%S")
        self.logs.append(f"{ts} {msg}")

    # ---- 状态快照（供 /status 轮询） ----

    def status(self):
        p = dict(self.progress)
        return {
            "state": self.state,
            "reason": self.reason,
            "goal_id": self.goal_id,
            "goal_name": self.goal_name,
            "indices": self.indices,
            "mode": self.mode,
            "days": self.days,
            "limit": self.limit,
            "sync_id": self.sync_id,
            "progress": p,
            "logs": list(self.logs)[-200:],
        }

    # ---- 启动 / 停止 ----

    def start(self, goal_id, goal_name, indices, days=None, limit=None):
        """启动同步（后台线程）。

        同步方式二选一：按天数 days(1-30) 或按条数 limit(1-100000)。
        days/limit 只能给一个；都为空时默认按天数 ES_DEFAULT_DAYS。
        """
        import threading
        if self.state in ("running",):
            raise ValueError(f"同步状态为 {self.state}，无法重复开始")
        if not indices:
            raise ValueError("至少选择一个索引")
        if days is not None and limit is not None:
            raise ValueError("同步方式只能选择天数或条数之一")
        if limit is not None:
            if limit < ES_MIN_LIMIT or limit > ES_MAX_LIMIT:
                raise ValueError(f"同步条数必须在 {ES_MIN_LIMIT}-{ES_MAX_LIMIT} 之间")
            self.mode, self.days, self.limit = "limit", None, limit
        else:
            if days is None:
                days = ES_DEFAULT_DAYS
            if days < ES_MIN_DAYS or days > ES_MAX_DAYS:
                raise ValueError(f"同步天数必须在 {ES_MIN_DAYS}-{ES_MAX_DAYS} 之间")
            self.mode, self.days, self.limit = "days", days, None
        self.goal_id = goal_id
        self.goal_name = goal_name
        self.indices = indices
        self.state = "running"
        self.reason = "同步中"
        self._stop = threading.Event()
        self.logs.clear()
        self.progress.update({
            "phase": "querying", "index": "", "docs_processed": 0,
            "docs_total": None, "percent": 0, "elapsed": 0,
            "works_inserted": 0, "works_skipped": 0,
            "ann_inserted": 0, "ann_skipped": 0,
        })
        # 记录同步批次
        import db
        self.sync_id = db.execute(
            "INSERT INTO sync_batches (goal_id, indices, days, sync_limit, state) "
            "VALUES (?, ?, ?, ?, 'running')",
            (goal_id, ",".join(indices), self.days, self.limit))
        if self.mode == "limit":
            self._log(f"同步开始: 索引={indices} 按条数取最近 {self.limit} 条")
        else:
            self._log(f"同步开始: 索引={indices} 按天数取最近 {self.days} 天")
        self.thread = threading.Thread(
            target=self._run, args=(goal_id, indices), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """优雅停止同步。"""
        if self.state == "running":
            self.reason = "正在停止..."
            self._stop.set()
            return True
        raise ValueError(f"当前状态 {self.state} 无法停止")

    # ---- 主流程 ----

    def _run(self, goal_id, indices):
        import db
        counts = {"works_inserted": 0, "works_skipped": 0,
                  "ann_inserted": 0, "ann_skipped": 0}
        start_time = time.time()
        try:
            # 预载已有作品，用于跨索引去重
            conn = db.get_conn()
            try:
                note_cache = {r["note_id"]: r["id"] for r in conn.execute(
                    "SELECT id, note_id FROM works WHERE goal_id = ?", (goal_id,))}
            finally:
                conn.close()
            self._log(f"已存在作品: {len(note_cache)} 条")
            dup_works = set()  # 记录本次已判定为重复的作品ID，避免重复计数

            index_names = [ES_INDICES[k] for k in indices]
            if self.mode == "limit" and len(index_names) > 1:
                # 按条数：多索引合并查询，按 createTime 倒序全局取前 N 条
                path = ",".join(index_names)
                self.progress["index"] = path
                self._log(f"[{path}] 开始按条数拉取最近 {self.limit} 条...")
                self._pull_index(goal_id, path,
                                 note_cache, dup_works, counts, start_time)
            else:
                for key in indices:
                    if self._stop.is_set():
                        break
                    index_name = ES_INDICES[key]
                    self.progress["index"] = index_name
                    if self.mode == "limit":
                        self._log(f"[{index_name}] 开始按条数拉取最近 {self.limit} 条...")
                    else:
                        self._log(f"[{index_name}] 开始拉取最近 {self.days} 天数据...")
                    self._pull_index(goal_id, index_name,
                                     note_cache, dup_works, counts, start_time)

            elapsed = round(time.time() - start_time, 1)
            if self._stop.is_set():
                self.state = "stopped"
                self.reason = "用户手动停止"
                self.progress["phase"] = "stopped"
            else:
                self.state = "done"
                self.reason = f"同步完成: 作品新增{counts['works_inserted']} " \
                              f"评论新增{counts['ann_inserted']}"
                self.progress["phase"] = "done"
            self.progress["elapsed"] = elapsed
            self._log(f"同步结束: {self.reason}, 耗时 {elapsed}s")
            self._finish_batch(self.state, counts)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.state = "error"
            self.reason = f"同步出错: {e}"
            self.progress["phase"] = "error"
            self._log(f"ERROR: {e}")
            self._finish_batch("error", counts, error=str(e))

    def _finish_batch(self, state, counts, error=None):
        import db
        from datetime import timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        db.execute("""
            UPDATE sync_batches SET state = ?, works_inserted = ?, works_skipped = ?,
                ann_inserted = ?, ann_skipped = ?, error = ?, finished_at = ?
            WHERE id = ?
        """, (state, counts["works_inserted"], counts["works_skipped"],
              counts["ann_inserted"], counts["ann_skipped"], error, now,
              self.sync_id))

    # ---- ES 访问 ----

    def _es_request(self, method, path, body=None):
        url = ES_BASE_URL + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Connection", "close")  # 避免长连接被 ES 侧重置
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _clear_scroll(self, scroll_id):
        try:
            self._es_request("DELETE", "/_search/scroll",
                             {"scroll_id": [scroll_id]})
        except Exception:
            pass

    # ---- 拉取（index_path 可为单索引或逗号合并的多索引） ----

    def _pull_index(self, goal_id, index_path,
                    note_cache, dup_works, counts, start_time):
        scroll_id = None
        limit = self.limit
        if limit:
            query = {"match_all": {}}
            sort = [{"createTime": "desc"}, {"_id": "desc"}]
        else:
            gte = int(time.time() * 1000) - self.days * 86400000
            query = {"range": {"createTime": {"gte": gte}}}
            sort = [{"createTime": "asc"}, {"_id": "asc"}]
        self.progress["phase"] = "querying"
        self.progress["docs_processed"] = 0
        try:
            resp = self._es_request("POST", f"/{index_path}/_search?scroll={ES_SCROLL_KEEP}", {
                "query": query,
                "size": ES_PAGE_SIZE,
                "sort": sort,
                "track_total_hits": True,
            })
            total = (resp.get("hits", {}).get("total") or {}).get("value")
            if limit:
                total = min(total, limit) if total else limit
            self.progress["docs_total"] = total
            self._log(f"[{index_path}] 命中文档: {total}")

            while True:
                if self._stop.is_set():
                    break
                hits = resp.get("hits", {}).get("hits", [])
                if not hits:
                    break
                scroll_id = resp.get("_scroll_id", scroll_id)
                if limit:
                    remaining = limit - self.progress["docs_processed"]
                    if remaining <= 0:
                        break
                    if len(hits) > remaining:
                        hits = hits[:remaining]
                self._log(f"[{index_path}] 拉取到 {len(hits)} 条, "
                          f"累计 {self.progress['docs_processed'] + len(hits)}/{total}")
                self._write_page(goal_id, self.sync_id, hits,
                                 note_cache, dup_works, counts)
                self.progress["docs_processed"] += len(hits)
                self.progress["elapsed"] = round(time.time() - start_time, 1)
                self.progress["percent"] = int(
                    self.progress["docs_processed"] / total * 100) if total else 0
                self.progress["phase"] = "writing"
                if limit and self.progress["docs_processed"] >= limit:
                    break
                resp = self._es_request("POST", "/_search/scroll", {
                    "scroll": ES_SCROLL_KEEP, "scroll_id": scroll_id})
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ES 请求失败: {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"ES 无法连接: {e.reason}") from e
        finally:
            if scroll_id:
                self._clear_scroll(scroll_id)

    def _write_page(self, goal_id, sync_id, hits, note_cache,
                    dup_works, counts):
        """把一页 ES 文档写入 works/annotations（单事务）。"""
        import db
        conn = db.get_conn()
        try:
            for hit in hits:
                if self._stop and self._stop.is_set():
                    break
                src = hit.get("_source") or {}
                note_id = src.get("noteId")
                comment_id = src.get("commentId")
                if not note_id or not comment_id:
                    continue
                src_index = _INDEX_TO_KEY.get(hit.get("_index"), "")

                work_id = note_cache.get(str(note_id))
                if work_id is None:
                    cur = conn.execute("""
                        INSERT OR IGNORE INTO works
                            (goal_id, note_id, note_title, content, platform,
                             author_nickname, author_id, note_time, note_type,
                             note_cover, note_video, note_url, topics,
                             source_index, sync_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (goal_id, str(note_id), src.get("noteTitle"),
                          src.get("noteDesc"), src.get("platform"),
                          src.get("authorNickname"), str(src.get("authorId") or ""),
                          src.get("noteTime"), src.get("noteType"),
                          src.get("noteCover"), src.get("noteVideo"),
                          None, src.get("noteTopics"), src_index, sync_id))
                    if cur.rowcount == 1:
                        work_id = cur.lastrowid
                        note_cache[str(note_id)] = work_id
                        counts["works_inserted"] += 1
                    else:
                        row = conn.execute(
                            "SELECT id FROM works WHERE goal_id = ? AND note_id = ?",
                            (goal_id, str(note_id))).fetchone()
                        work_id = row[0] if row else None
                        note_cache[str(note_id)] = work_id
                        if str(note_id) not in dup_works:
                            dup_works.add(str(note_id))
                            counts["works_skipped"] += 1
                else:
                    if str(note_id) not in dup_works:
                        dup_works.add(str(note_id))
                        counts["works_skipped"] += 1

                if work_id is None:
                    continue
                cur = conn.execute("""
                    INSERT OR IGNORE INTO annotations
                        (goal_id, work_id, comment_id, comment, raw_score,
                         comment_create_time, comment_user_name, source_index, sync_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (goal_id, work_id, str(comment_id), src.get("commentContent"),
                      src.get("intentScore"), src.get("commentCreateTime"),
                      src.get("commentUserName"), src_index, sync_id))
                if cur.rowcount == 1:
                    counts["ann_inserted"] += 1
                else:
                    counts["ann_skipped"] += 1
            self.progress.update({
                "works_inserted": counts["works_inserted"],
                "works_skipped": counts["works_skipped"],
                "ann_inserted": counts["ann_inserted"],
                "ann_skipped": counts["ann_skipped"],
            })
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
