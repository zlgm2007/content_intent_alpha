#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""自动标注引擎 — 用已训练 ONNX 模型批量预测未标注评论，全部写库。

状态机：idle → running → done/stopped/error（镜像 trainer_engine.TrainerEngine）

流程：
  1. 加载指定目标、指定已训练模型的 ONNX 推理器
  2. 取该目标全部未标注评论（手工 comments.pending + 外部 annotations.pending）
  3. 分批推理全部写库：confidence >= threshold 写模型得分，低置信写 score=0（无意图）
  4. 所有评论标记为已标注（status='labeled'，auto_labeled=1），移出待标注；进度经 status() 轮询，支持 stop
"""
import os
import time
from collections import deque
from datetime import datetime

from common import REPO_DIR

SCORE_THRESHOLD = 3
BATCH_SIZE = 32


def _score_to_label(score):
    return "has_intent" if score >= SCORE_THRESHOLD else "no_intent"


class AutoLabelEngine:
    """单个自动标注任务的状态机。一次只跑一个。线程安全由调用方(API)保证。"""

    def __init__(self):
        self.goal_id = None
        self.goal_name = ""
        self.model_id = None
        self.threshold = 0.8
        self.max_count = 0
        self.state = "idle"          # idle / running / done / stopped / error
        self.reason = ""
        self.thread = None
        self._stop = None            # threading.Event
        self.logs = deque(maxlen=500)
        self.progress = {
            "phase": "idle", "total": 0, "processed": 0,
            "labeled": 0, "low_conf": 0, "percent": 0, "elapsed": 0,
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
            "model_id": self.model_id,
            "threshold": self.threshold,
            "max_count": self.max_count,
            "progress": p,
            "logs": list(self.logs)[-200:],
        }

    # ---- 启动 / 停止 ----

    def start(self, goal_id, goal_name, model_id, threshold=0.8, max_count=0):
        """启动自动标注（后台线程）。"""
        import threading
        import db
        if self.state in ("running",):
            raise ValueError(f"自动标注状态为 {self.state}，无法重复开始")
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("置信度阈值必须在 0.5-1.0 之间")
        if max_count < 0:
            raise ValueError("条数上限不能为负数")
        model = db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if not model:
            raise ValueError("模型不存在")
        if model["status"] != "trained":
            raise ValueError(f"模型状态为 {model['status']}，需为 trained")
        if model["goal_id"] != goal_id:
            raise ValueError("模型不属于该意图目标")
        model_dir = os.path.join(REPO_DIR, model["tokenizer_dir"])
        if not os.path.isdir(model_dir):
            raise ValueError(f"模型目录不存在: {model_dir}")
        onnx = os.path.join(model_dir, "model_int8.onnx")
        if not os.path.isfile(onnx):
            onnx = os.path.join(model_dir, "model.onnx")
        if not os.path.isfile(onnx):
            raise ValueError("模型目录缺少 ONNX 文件")

        self.goal_id = goal_id
        self.goal_name = goal_name
        self.model_id = model_id
        self.threshold = threshold
        self.max_count = max_count
        self.state = "running"
        self.reason = "自动标注中"
        self._stop = threading.Event()
        self.logs.clear()
        self.progress.update({
            "phase": "loading", "total": 0, "processed": 0,
            "labeled": 0, "low_conf": 0, "percent": 0, "elapsed": 0,
        })
        self._log(f"自动标注开始: 模型=#{model_id} 阈值={threshold} "
                  f"上限={max_count if max_count else '全部'}")
        self.thread = threading.Thread(
            target=self._run,
            args=(goal_id, model_id, threshold, max_count), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """优雅停止自动标注。"""
        if self.state == "running":
            self.reason = "正在停止..."
            self._stop.set()
            return True
        raise ValueError(f"当前状态 {self.state} 无法停止")

    # ---- 主流程 ----

    def _run(self, goal_id, model_id, threshold, max_count):
        import db
        start_time = time.time()
        try:
            model = db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
            model_dir = os.path.join(REPO_DIR, model["tokenizer_dir"])
            from inference import IntentInferencer
            inferencer = IntentInferencer(model_dir)
            self.progress["phase"] = "querying"
            self._log("正在加载未标注评论...")

            rows = db.query_all("""
                SELECT * FROM (
                    SELECT cm.id AS id, 'manual' AS source,
                           cm.comment AS comment, ct.text AS content_text
                    FROM comments cm JOIN contents ct ON cm.content_id = ct.id
                    WHERE cm.goal_id = ? AND cm.status = 'pending'
                    UNION ALL
                    SELECT a.id AS id, 'es' AS source,
                           COALESCE(a.comment, '') AS comment,
                           COALESCE(w.content, w.note_title, '') AS content_text
                    FROM annotations a JOIN works w ON a.work_id = w.id
                    WHERE a.goal_id = ? AND a.status = 'pending'
                ) ORDER BY id DESC
            """, (goal_id, goal_id))
            if max_count > 0:
                rows = rows[:max_count]
            total = len(rows)
            self.progress["total"] = total
            if total == 0:
                self._finish("done", "没有未标注评论", start_time)
                return
            self._log(f"未标注评论: {total} 条，开始推理...")

            labeled = 0
            low_conf = 0
            processed = 0
            for i in range(0, total, BATCH_SIZE):
                if self._stop.is_set():
                    break
                chunk = rows[i:i + BATCH_SIZE]
                pairs = [(r["content_text"] or "", r["comment"] or "")
                         for r in chunk]
                preds = inferencer.predict_batch(pairs, batch_size=BATCH_SIZE)
                results = []
                for r, pr in zip(chunk, preds):
                    # 全部写库：高置信写模型得分，低置信记 0 分（无意图），避免重复标记
                    results.append((r["source"], r["id"],
                                    pr["score"] if pr["confidence"] >= threshold else 0))
                    if pr["confidence"] < threshold:
                        low_conf += 1
                labeled += len(results)
                processed += len(chunk)
                self._write_results(results)
                self.progress.update({
                    "phase": "writing", "processed": processed,
                    "labeled": labeled, "low_conf": low_conf,
                    "elapsed": round(time.time() - start_time, 1),
                    "percent": int(processed / total * 100) if total else 0,
                })
                self._log(f"已处理 {processed}/{total} 标注 {labeled} 低置信记0 {low_conf}")

            if self._stop.is_set():
                self._finish("stopped", f"用户手动停止，已标注 {labeled} 条", start_time)
            else:
                self._finish(
                    "done",
                    f"完成: 标注 {labeled} 条（低置信记0分 {low_conf} 条）", start_time)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.state = "error"
            self.reason = f"自动标注出错: {e}"
            self.progress["phase"] = "error"
            self._log(f"ERROR: {e}")

    def _finish(self, state, reason, start_time):
        self.state = state
        self.reason = reason
        self.progress["phase"] = state
        self.progress["elapsed"] = round(time.time() - start_time, 1)
        self._log(f"自动标注结束: {reason}, 耗时 {self.progress['elapsed']}s")

    def _write_results(self, results):
        """把全部标注结果按来源分表写库（每批一个事务）；低置信已映射为 score=0。"""
        if not results:
            return
        import db
        manual = [h for h in results if h[0] == "manual"]
        es = [h for h in results if h[0] == "es"]
        conn = db.get_conn()
        try:
            if manual:
                conn.executemany(
                    "UPDATE comments SET score = ?, label = ?, status = 'labeled', "
                    "auto_labeled = 1 WHERE id = ? AND status = 'pending'",
                    [(s, _score_to_label(s), i) for (_, i, s) in manual])
            if es:
                conn.executemany(
                    "UPDATE annotations SET score = ?, label = ?, status = 'labeled', "
                    "auto_labeled = 1 WHERE id = ? AND status = 'pending'",
                    [(s, _score_to_label(s), i) for (_, i, s) in es])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
