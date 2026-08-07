#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""自动标注 API — /autolabel/api/*

功能：
  - status: 自动标注进度/日志（前端轮询）
  - start:  启动自动标注 {goal_id, model_id, threshold, max_count}
  - stop:   停止自动标注
"""
import db
from autolabel_engine import AutoLabelEngine

_engine = AutoLabelEngine()


def _json(obj):
    return obj, None


class AutoLabelAPI:
    """自动标注 API。"""

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
        model_id = int(body.get("model_id", 0))
        if model_id <= 0:
            raise ValueError("缺少 model_id")
        threshold = float(body.get("threshold", 0.8))
        max_count = int(body.get("max_count", 0))
        self.engine.start(goal_id, goal["name"], model_id,
                          threshold=threshold, max_count=max_count)
        return _json({"ok": True, "goal_id": goal_id, "model_id": model_id,
                      "threshold": threshold, "max_count": max_count})

    def _post_stop(self, body):
        self.engine.stop()
        return _json({"ok": True})

    # ---- 路由表 ----

    _GET = {
        "status": _get_status,
    }

    _POST = {
        "start": _post_start,
        "stop": _post_stop,
    }
