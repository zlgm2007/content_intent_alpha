#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""训练 API — /train/api/*

功能：
  - status:     训练状态/进度/日志（前端轮询）
  - models:     列出某意图目标的已训练模型
  - data_stats: 某意图目标的标注数据统计
  - start:      启动训练
  - stop:       停止训练
  - predict:    实时推理（单条）
"""
import json
import os

import db
from common import MODELS_DIR, REPO_DIR
from trainer_engine import TrainerEngine

_engine = TrainerEngine()


def _json(obj):
    return obj, None


class TrainerAPI:
    """模型训练 API。"""

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

    def _get_models(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        rows = db.query_all("""
            SELECT * FROM models WHERE goal_id = ? ORDER BY id DESC
        """, (goal_id,))
        for r in rows:
            r["params"] = json.loads(r["params"]) if r.get("params") else {}
            r["onnx_exists"] = os.path.isfile(
                os.path.join(REPO_DIR, r["onnx_path"])) if r.get("onnx_path") else False
        return _json({"models": rows})

    def _get_data_stats(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        total = db.query_one(
            "SELECT COUNT(*) AS n FROM comments WHERE goal_id = ?", (goal_id,))["n"]
        labeled = db.query_one(
            "SELECT COUNT(*) AS n FROM comments WHERE goal_id = ? AND status = 'labeled'",
            (goal_id,))["n"]
        score_dist = db.query_all("""
            SELECT score, COUNT(*) AS count
            FROM comments
            WHERE goal_id = ? AND status = 'labeled'
            GROUP BY score ORDER BY score
        """, (goal_id,))
        has_intent = sum(r["count"] for r in score_dist if r["score"] >= 3)
        no_intent = sum(r["count"] for r in score_dist if r["score"] < 3)
        return _json({
            "total_comments": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "has_intent": has_intent,
            "no_intent": no_intent,
            "score_distribution": {str(r["score"]): r["count"] for r in score_dist},
            "ready": labeled >= 20,
        })

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

        # 检查数据量
        labeled_count = db.query_one(
            "SELECT COUNT(*) AS n FROM comments WHERE goal_id = ? AND status = 'labeled'",
            (goal_id,))["n"]
        if labeled_count < 20:
            raise ValueError(f"标注数据不足: {labeled_count} 条，至少需要 20 条")

        hyperparams = body.get("hyperparams") or {}
        self.engine.start(goal_id, goal["name"], hyperparams)
        return _json({"ok": True, "goal_id": goal_id, "goal_name": goal["name"]})

    def _post_stop(self, body):
        self.engine.stop()
        return _json({"ok": True})

    def _post_predict(self, body):
        """实时推理：输入作品内容 + 评论，返回意图评分。"""
        model_id = int(body.get("model_id", 0))
        content = str(body.get("content", "")).strip()
        comment = str(body.get("comment", "")).strip()
        if model_id <= 0:
            raise ValueError("缺少 model_id")
        if not content or not comment:
            raise ValueError("作品内容和评论不能为空")

        model = db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if not model:
            raise ValueError("模型不存在")

        model_dir = os.path.join(REPO_DIR, model["tokenizer_dir"])
        if not os.path.isdir(model_dir):
            raise ValueError(f"模型目录不存在: {model_dir}")

        from inference import IntentInferencer
        inferencer = IntentInferencer(model_dir)
        result = inferencer.predict(content, comment)
        result["model_id"] = model_id
        return _json({"ok": True, "result": result})

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "status": _get_status,
        "models": _get_models,
        "data_stats": _get_data_stats,
    }

    _POST = {
        "start": _post_start,
        "stop": _post_stop,
        "predict": _post_predict,
    }
