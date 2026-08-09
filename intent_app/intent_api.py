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
                   (SELECT COUNT(*) FROM contents WHERE goal_id = g.id) AS content_count,
                   (SELECT COUNT(*) FROM works WHERE goal_id = g.id) AS es_work_count,
                   (SELECT COUNT(*) FROM comments WHERE goal_id = g.id) AS comment_count,
                   (SELECT COUNT(*) FROM annotations WHERE goal_id = g.id) AS es_comment_count,
                   (SELECT COUNT(*) FROM comments WHERE goal_id = g.id
                    AND status = 'labeled') AS labeled_count,
                   (SELECT COUNT(*) FROM annotations WHERE goal_id = g.id
                    AND status = 'labeled') AS es_labeled_count,
                   (SELECT COUNT(*) FROM models WHERE goal_id = g.id
                    AND status = 'trained') AS model_count
            FROM intent_goals g
            ORDER BY g.id DESC
        """)
        # 合并总数 = 手工 + 外部(ES)
        for r in rows:
            r["work_total"] = r["content_count"] + r["es_work_count"]
            r["comment_total"] = r["comment_count"] + r["es_comment_count"]
            r["labeled_total"] = r["labeled_count"] + r["es_labeled_count"]
            r["score_definitions"] = json.loads(r["score_definitions"]) if r.get("score_definitions") else None
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
                (SELECT COUNT(*) FROM contents WHERE goal_id = ?) AS content_count,
                (SELECT COUNT(*) FROM works WHERE goal_id = ?) AS es_work_count,
                (SELECT COUNT(*) FROM comments WHERE goal_id = ?) AS comment_count,
                (SELECT COUNT(*) FROM annotations WHERE goal_id = ?) AS es_comment_count,
                (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'labeled') AS labeled_count,
                (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'labeled') AS es_labeled_count,
                (SELECT COUNT(*) FROM models WHERE goal_id = ? AND status = 'trained') AS model_count
        """, (goal_id, goal_id, goal_id, goal_id, goal_id, goal_id, goal_id))
        stats["work_total"] = stats["content_count"] + stats["es_work_count"]
        stats["comment_total"] = stats["comment_count"] + stats["es_comment_count"]
        stats["labeled_total"] = stats["labeled_count"] + stats["es_labeled_count"]
        goal.update(stats)
        goal["score_definitions"] = json.loads(goal["score_definitions"]) if goal.get("score_definitions") else None
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
        # 分数定义
        score_defs = body.get("score_definitions")
        defs_json = None
        if score_defs:
            if not isinstance(score_defs, list) or len(score_defs) < 2:
                raise ValueError("score_definitions 至少需要 2 个分数定义")
            defs_json = json.dumps(score_defs, ensure_ascii=False)
        # 意图阈值
        threshold = body.get("intent_threshold")
        # 检查重名
        existing = db.query_one("SELECT id FROM intent_goals WHERE name = ?", (name,))
        if existing:
            raise ValueError(f"意图目标已存在: {name}")
        gid = db.execute(
            "INSERT INTO intent_goals (name, description, score_definitions, intent_threshold) VALUES (?, ?, ?, ?)",
            (name, desc, defs_json, threshold))
        goal = db.query_one("SELECT * FROM intent_goals WHERE id = ?", (gid,))
        return _json({"ok": True, "goal": goal})

    def _post_update(self, body):
        goal_id = int(body.get("id", 0))
        if goal_id <= 0:
            raise ValueError("缺少 id")
        name = str(body.get("name", "")).strip()
        desc = str(body.get("description", "")).strip()

        # 构建动态 SET 子句
        sets = []
        params = []
        if name:
            dup = db.query_one(
                "SELECT id FROM intent_goals WHERE name = ? AND id != ?",
                (name, goal_id))
            if dup:
                raise ValueError(f"意图名称已被使用: {name}")
            sets.append("name = ?")
            params.append(name)
        sets.append("description = ?")
        params.append(desc)

        # 分数定义
        if "score_definitions" in body:
            score_defs = body.get("score_definitions")
            if score_defs:
                if not isinstance(score_defs, list) or len(score_defs) < 2:
                    raise ValueError("score_definitions 至少需要 2 个分数定义")
                sets.append("score_definitions = ?")
                params.append(json.dumps(score_defs, ensure_ascii=False))
            else:
                sets.append("score_definitions = NULL")
        # 意图阈值
        if "intent_threshold" in body:
            threshold = body.get("intent_threshold")
            sets.append("intent_threshold = ?")
            params.append(threshold)

        params.append(goal_id)
        db.execute(
            f"UPDATE intent_goals SET {', '.join(sets)} WHERE id = ?",
            tuple(params))
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
