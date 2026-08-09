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
from common import MODELS_DIR, REPO_DIR, get_goal_score_config
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
        # 合并 手工(comments) + 外部(annotations) 统计
        total = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ?) "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ?) AS n",
            (goal_id, goal_id))["n"]
        labeled = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'labeled') "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'labeled') AS n",
            (goal_id, goal_id))["n"]
        auto_labeled = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND auto_labeled = 1) "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND auto_labeled = 1) AS n",
            (goal_id, goal_id))["n"]
        unlabeled = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'pending') "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'pending') AS n",
            (goal_id, goal_id))["n"]
        score_dist = db.query_all("""
            SELECT score, SUM(cnt) AS count FROM (
                SELECT score, COUNT(*) AS cnt FROM comments
                WHERE goal_id = ? AND status = 'labeled' GROUP BY score
                UNION ALL
                SELECT score, COUNT(*) AS cnt FROM annotations
                WHERE goal_id = ? AND status = 'labeled' GROUP BY score
            ) GROUP BY score ORDER BY score
        """, (goal_id, goal_id))
        # 动态阈值
        score_cfg = get_goal_score_config(goal_id)
        threshold = score_cfg["threshold"]
        has_intent = sum(r["count"] for r in score_dist if r["score"] >= threshold)
        no_intent = sum(r["count"] for r in score_dist if r["score"] < threshold)
        # ES 原始得分可作伪标签的条数（只读训练用）
        raw_available = db.query_one(
            "SELECT COUNT(*) AS n FROM annotations "
            "WHERE goal_id = ? AND raw_score IS NOT NULL", (goal_id,))["n"]
        return _json({
            "total_comments": total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "auto_labeled": auto_labeled,
            "has_intent": has_intent,
            "no_intent": no_intent,
            "score_distribution": {str(r["score"]): r["count"] for r in score_dist},
            "ready": labeled >= 20,
            "raw_available": raw_available,
            "ready_raw": raw_available >= 20,
            "max_score": score_cfg["max_score"],
            "num_labels": score_cfg["num_labels"],
            "threshold": threshold,
            "score_definitions": score_cfg["score_definitions"],
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

        label_source = body.get("label_source", "labeled")
        if label_source not in ("labeled", "raw"):
            raise ValueError("label_source 必须为 labeled 或 raw")

        # 检查数据量：labeled=人工标注合并，raw=ES 原始得分伪标签
        if label_source == "raw":
            data_count = db.query_one(
                "SELECT COUNT(*) AS n FROM annotations "
                "WHERE goal_id = ? AND raw_score IS NOT NULL", (goal_id,))["n"]
        else:
            data_count = db.query_one(
                "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'labeled') "
                "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'labeled') AS n",
                (goal_id, goal_id))["n"]
        if data_count < 20:
            raise ValueError(f"标注数据不足: {data_count} 条，至少需要 20 条")

        hyperparams = body.get("hyperparams") or {}
        self.engine.start(goal_id, goal["name"], hyperparams, label_source)
        return _json({"ok": True, "goal_id": goal_id, "goal_name": goal["name"],
                      "label_source": label_source})

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
