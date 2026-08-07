#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""外部数据同步 API — /sync/api/*

功能：
  - status:           同步进度/日志（前端轮询）
  - history:          同步批次历史（分页）
  - works:            已同步作品列表（分页 + 标注统计）
  - work_detail:      单个外部作品 + 其标注评论（供作品列表合并视图）
  - annotations:      已同步评论列表（分页，支持按状态筛选）
  - test_connection:  探测 ES 连通性
  - start:            启动同步 {goal_id, indices, days 或 limit}（二选一）
  - stop:             停止同步
  - label:            标注同步评论（打分 0-5）
  - skip:             跳过同步评论
  - work_delete:      删除外部作品（级联删除其标注）
"""
import json

import db
from common import ES_DEFAULT_DAYS, ES_INDICES
from es_sync_engine import EsSyncEngine, es_test_connection

_engine = EsSyncEngine()
SCORE_THRESHOLD = 3


def _json(obj):
    return obj, None


class SyncAPI:
    """外部数据同步 API。"""

    def __init__(self):
        self.engine = _engine

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def _get_status(self, q):
        return _json(self.engine.status())

    def _get_history(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "20")), 100)
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        rows = db.query_all("""
            SELECT * FROM sync_batches WHERE goal_id = ?
            ORDER BY id DESC LIMIT ? OFFSET ?
        """, (goal_id, size, offset))
        total = db.query_one(
            "SELECT COUNT(*) AS n FROM sync_batches WHERE goal_id = ?",
            (goal_id,))["n"]
        return _json({"items": rows, "total": total, "page": page, "size": size})

    def _get_works(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "20")), 100)
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        rows = db.query_all("""
            SELECT w.*,
                   (SELECT COUNT(*) FROM annotations a
                    WHERE a.work_id = w.id) AS ann_count,
                   (SELECT COUNT(*) FROM annotations a
                    WHERE a.work_id = w.id AND a.status = 'labeled') AS labeled_count,
                   (SELECT COUNT(*) FROM annotations a
                    WHERE a.work_id = w.id AND a.status = 'pending') AS pending_count
            FROM works w
            WHERE w.goal_id = ?
            ORDER BY w.id DESC
            LIMIT ? OFFSET ?
        """, (goal_id, size, offset))
        total = db.query_one(
            "SELECT COUNT(*) AS n FROM works WHERE goal_id = ?",
            (goal_id,))["n"]
        return _json({"works": rows, "total": total, "page": page, "size": size})

    def _get_annotations(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "20")), 100)
        status = self._q(q, "status", "")
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        where = "a.goal_id = ?"
        params = [goal_id]
        if status:
            where += " AND a.status = ?"
            params.append(status)
        rows = db.query_all(f"""
            SELECT a.id, a.comment, a.raw_score, a.score, a.label, a.status,
                   a.source_index, a.comment_user_name,
                   w.note_id, w.note_title, w.content AS work_content
            FROM annotations a
            JOIN works w ON a.work_id = w.id
            WHERE {where}
            ORDER BY a.id DESC
            LIMIT ? OFFSET ?
        """, params + [size, offset])
        total = db.query_one(f"""
            SELECT COUNT(*) AS n FROM annotations a
            JOIN works w ON a.work_id = w.id
            WHERE {where}
        """, params)["n"]
        return _json({"items": rows, "total": total, "page": page, "size": size})

    def _get_work_detail(self, q):
        work_id = int(self._q(q, "id", "0"))
        if work_id <= 0:
            raise ValueError("缺少 id")
        work = db.query_one("SELECT * FROM works WHERE id = ?", (work_id,))
        if not work:
            raise ValueError("作品不存在")
        anns = db.query_all("""
            SELECT id, comment_id, comment, raw_score, score, label, status,
                   source_index, comment_user_name, comment_create_time
            FROM annotations WHERE work_id = ? ORDER BY id DESC
        """, (work_id,))
        return _json({"work": work, "annotations": anns})

    def _get_test_connection(self, q):
        ok, msg = es_test_connection()
        return _json({"ok": ok, "message": msg})

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_start(self, body):
        goal_id = int(body.get("goal_id", 0))
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        goal = db.query_one("SELECT * FROM intent_goals WHERE id = ?", (goal_id,))
        if not goal:
            raise ValueError("意图目标不存在")

        # 同步方式：days/limit 只能给一个，都为空默认按天数
        days_raw = body.get("days")
        limit_raw = body.get("limit")
        if days_raw is not None and limit_raw is not None:
            raise ValueError("同步方式只能选择天数或条数之一")
        days, limit = None, None
        if limit_raw is not None:
            limit = int(limit_raw)
        else:
            days = int(days_raw) if days_raw is not None else ES_DEFAULT_DAYS

        raw = body.get("indices") or "both"
        if isinstance(raw, str):
            raw = raw.strip().lower()
        if raw == "both":
            indices = list(ES_INDICES.keys())
        elif raw in ES_INDICES:
            indices = [raw]
        else:
            raise ValueError("indices 需为 ge3 / lt3 / both")

        self.engine.start(goal_id, goal["name"], indices, days=days, limit=limit)
        return _json({"ok": True, "goal_id": goal_id, "indices": indices,
                      "days": days, "limit": limit})

    def _post_stop(self, body):
        self.engine.stop()
        return _json({"ok": True})

    def _post_label(self, body):
        ann_id = int(body.get("id", 0))
        score = int(body.get("score", -1))
        if ann_id <= 0:
            raise ValueError("缺少评论 id")
        if score < 0 or score > 5:
            raise ValueError("分数必须为 0-5")
        label = "has_intent" if score >= SCORE_THRESHOLD else "no_intent"
        db.execute(
            "UPDATE annotations SET score = ?, label = ?, status = 'labeled' "
            "WHERE id = ?", (score, label, ann_id))
        row = db.query_one("SELECT * FROM annotations WHERE id = ?", (ann_id,))
        return _json({"ok": True, "annotation": row})

    def _post_skip(self, body):
        ann_id = int(body.get("id", 0))
        if ann_id <= 0:
            raise ValueError("缺少评论 id")
        db.execute(
            "UPDATE annotations SET status = 'skipped' WHERE id = ?", (ann_id,))
        return _json({"ok": True})

    def _post_unlabel(self, body):
        ann_id = int(body.get("id", 0))
        if ann_id <= 0:
            raise ValueError("缺少评论 id")
        db.execute(
            "UPDATE annotations SET score = NULL, label = NULL, "
            "status = 'pending' WHERE id = ?", (ann_id,))
        return _json({"ok": True})

    def _post_work_delete(self, body):
        work_id = int(body.get("id", 0))
        if work_id <= 0:
            raise ValueError("缺少 id")
        db.execute("DELETE FROM works WHERE id = ?", (work_id,))
        return _json({"ok": True})

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "status": _get_status,
        "history": _get_history,
        "works": _get_works,
        "work_detail": _get_work_detail,
        "annotations": _get_annotations,
        "test_connection": _get_test_connection,
    }

    _POST = {
        "start": _post_start,
        "stop": _post_stop,
        "label": _post_label,
        "skip": _post_skip,
        "unlabel": _post_unlabel,
        "work_delete": _post_work_delete,
    }
