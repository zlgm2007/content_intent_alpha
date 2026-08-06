#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""意图目标管理 API — /intent/api/*

功能：
  - list:   列出所有意图目标 + 统计数据
  - get:    获取单个意图目标详情
  - create: 创建意图目标
  - update: 更新意图目标
  - delete: 删除意图目标（级联删除关联数据）
"""
import json

import db


def _json(obj):
    return obj, None


class IntentAPI:
    """意图目标管理 API。"""

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def _get_list(self, q):
        rows = db.query_all("""
            SELECT g.*,
                   (SELECT COUNT(*) FROM contents WHERE goal_id = g.id)       AS content_count,
                   (SELECT COUNT(*) FROM comments WHERE goal_id = g.id)       AS comment_count,
                   (SELECT COUNT(*) FROM comments WHERE goal_id = g.id
                    AND status = 'labeled')                                    AS labeled_count,
                   (SELECT COUNT(*) FROM models WHERE goal_id = g.id
                    AND status = 'trained')                                    AS model_count
            FROM intent_goals g
            ORDER BY g.id DESC
        """)
        return _json({"goals": rows})

    def _get_detail(self, q):
        goal_id = int(self._q(q, "id", "0"))
        if goal_id <= 0:
            raise ValueError("缺少 id")
        goal = db.query_one("SELECT * FROM intent_goals WHERE id = ?", (goal_id,))
        if not goal:
            raise ValueError("意图目标不存在")
        stats = db.query_one("""
            SELECT
                (SELECT COUNT(*) FROM contents WHERE goal_id = ?)       AS content_count,
                (SELECT COUNT(*) FROM comments WHERE goal_id = ?)       AS comment_count,
                (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'labeled') AS labeled_count,
                (SELECT COUNT(*) FROM models WHERE goal_id = ? AND status = 'trained')   AS model_count
        """, (goal_id, goal_id, goal_id, goal_id))
        goal.update(stats)
        return _json(goal)

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_create(self, body):
        name = str(body.get("name", "")).strip()
        if not name:
            raise ValueError("意图名称不能为空")
        desc = str(body.get("description", "")).strip()
        # 检查重名
        existing = db.query_one("SELECT id FROM intent_goals WHERE name = ?", (name,))
        if existing:
            raise ValueError(f"意图目标已存在: {name}")
        gid = db.execute(
            "INSERT INTO intent_goals (name, description) VALUES (?, ?)",
            (name, desc))
        goal = db.query_one("SELECT * FROM intent_goals WHERE id = ?", (gid,))
        return _json({"ok": True, "goal": goal})

    def _post_update(self, body):
        goal_id = int(body.get("id", 0))
        if goal_id <= 0:
            raise ValueError("缺少 id")
        name = str(body.get("name", "")).strip()
        desc = str(body.get("description", "")).strip()
        if name:
            dup = db.query_one(
                "SELECT id FROM intent_goals WHERE name = ? AND id != ?",
                (name, goal_id))
            if dup:
                raise ValueError(f"意图名称已被使用: {name}")
            db.execute(
                "UPDATE intent_goals SET name = ?, description = ? WHERE id = ?",
                (name, desc, goal_id))
        else:
            db.execute(
                "UPDATE intent_goals SET description = ? WHERE id = ?",
                (desc, goal_id))
        goal = db.query_one("SELECT * FROM intent_goals WHERE id = ?", (goal_id,))
        return _json({"ok": True, "goal": goal})

    def _post_delete(self, body):
        goal_id = int(body.get("id", 0))
        if goal_id <= 0:
            raise ValueError("缺少 id")
        db.execute("DELETE FROM intent_goals WHERE id = ?", (goal_id,))
        return _json({"ok": True})

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "list": _get_list,
        "detail": _get_detail,
    }

    _POST = {
        "create": _post_create,
        "update": _post_update,
        "delete": _post_delete,
    }
