#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""数据批跑 API — /batch/api/*

功能：
  - models:  列出可用的已训练模型（所有意图目标）
  - run:     对已标注数据批量推理，输出识别率报告
  - results: 查看批跑结果明细（分页）
  - report:  查看批跑统计报告
"""
import json
import os

import db
from common import REPO_DIR

SCORE_THRESHOLD = 3


def _json(obj):
    return obj, None


class BatchAPI:
    """数据批跑 API。"""

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def _get_models(self, q):
        goal_id = self._q(q, "goal_id", "")
        if goal_id:
            goal_id = int(goal_id)
            rows = db.query_all(
                "SELECT * FROM models WHERE goal_id = ? AND status = 'trained' ORDER BY id DESC",
                (goal_id,))
        else:
            rows = db.query_all(
                "SELECT * FROM models WHERE status = 'trained' ORDER BY id DESC")
        goals = {g["id"]: g["name"] for g in db.query_all("SELECT * FROM intent_goals")}
        for r in rows:
            r["params"] = json.loads(r["params"]) if r.get("params") else {}
            r["goal_name"] = goals.get(r["goal_id"], "?")
            r["onnx_exists"] = os.path.isfile(
                os.path.join(REPO_DIR, r["onnx_path"])) if r.get("onnx_path") else False
        return _json({"models": rows})

    def _get_results(self, q):
        model_id = int(self._q(q, "model_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "50")), 500)
        if model_id <= 0:
            raise ValueError("缺少 model_id")
        offset = (page - 1) * size
        rows = db.query_all("""
            SELECT br.*, cm.comment, cm.score AS true_score,
                   ct.text AS content_text
            FROM batch_results br
            JOIN comments cm ON br.comment_id = cm.id
            JOIN contents ct ON cm.content_id = ct.id
            WHERE br.model_id = ?
            ORDER BY cm.id
            LIMIT ? OFFSET ?
        """, (model_id, size, offset))
        total = db.query_one(
            "SELECT COUNT(*) AS n FROM batch_results WHERE model_id = ?",
            (model_id,))["n"]
        return _json({"items": rows, "total": total, "page": page, "size": size})

    def _get_report(self, q):
        model_id = int(self._q(q, "model_id", "0"))
        if model_id <= 0:
            raise ValueError("缺少 model_id")
        rows = db.query_all(
            "SELECT * FROM batch_results WHERE model_id = ?", (model_id,))
        if not rows:
            return _json({"error": "暂无批跑结果"})

        total = len(rows)
        correct = sum(1 for r in rows if r["is_correct"])
        accuracy = correct / total if total > 0 else 0

        # 混淆矩阵 (6x6)
        confusion = [[0] * 6 for _ in range(6)]
        for r in rows:
            true_s = r["comment_id"]  # 这里用 stored true score
            # 需要从 comments 表获取 true_score，这里 rows 已经 join 了
        # 重新查询带 true_score 的完整数据
        full_rows = db.query_all("""
            SELECT br.predicted_score, br.is_correct, cm.score AS true_score
            FROM batch_results br
            JOIN comments cm ON br.comment_id = cm.id
            WHERE br.model_id = ?
        """, (model_id,))

        confusion = [[0] * 6 for _ in range(6)]
        for r in full_rows:
            t = r["true_score"]
            p = r["predicted_score"]
            if t is not None and p is not None and 0 <= t <= 5 and 0 <= p <= 5:
                confusion[t][p] += 1

        # 二分类指标 (>=3 = has_intent)
        tp = sum(1 for r in full_rows
                 if r["true_score"] >= SCORE_THRESHOLD and r["predicted_score"] >= SCORE_THRESHOLD)
        fp = sum(1 for r in full_rows
                 if r["true_score"] < SCORE_THRESHOLD and r["predicted_score"] >= SCORE_THRESHOLD)
        fn = sum(1 for r in full_rows
                 if r["true_score"] >= SCORE_THRESHOLD and r["predicted_score"] < SCORE_THRESHOLD)
        tn = sum(1 for r in full_rows
                 if r["true_score"] < SCORE_THRESHOLD and r["predicted_score"] < SCORE_THRESHOLD)

        binary_acc = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 按分数段统计
        score_stats = {}
        for score in range(6):
            score_rows = [r for r in full_rows if r["true_score"] == score]
            if score_rows:
                s_correct = sum(1 for r in score_rows if r["predicted_score"] == score)
                score_stats[str(score)] = {
                    "total": len(score_rows),
                    "correct": s_correct,
                    "accuracy": round(s_correct / len(score_rows), 4),
                }

        model = db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        goal = db.query_one(
            "SELECT name FROM intent_goals WHERE id = ?",
            (model["goal_id"],)) if model else None

        return _json({
            "model_id": model_id,
            "model_info": {
                "goal_name": goal["name"] if goal else "?",
                "accuracy": model["accuracy"] if model else None,
                "created_at": model["created_at"] if model else None,
            },
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "confusion_matrix": confusion,
            "binary": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "accuracy": round(binary_acc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            "score_stats": score_stats,
        })

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_run(self, body):
        """批量推理。"""
        model_id = int(body.get("model_id", 0))
        if model_id <= 0:
            raise ValueError("缺少 model_id")

        model = db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if not model:
            raise ValueError("模型不存在")

        model_dir = os.path.join(REPO_DIR, model["tokenizer_dir"])
        if not os.path.isdir(model_dir):
            raise ValueError(f"模型目录不存在: {model_dir}")

        # 加载已标注数据
        rows = db.query_all("""
            SELECT cm.id, cm.comment, cm.score, ct.text AS content_text
            FROM comments cm
            JOIN contents ct ON cm.content_id = ct.id
            WHERE cm.goal_id = ? AND cm.status = 'labeled'
            ORDER BY cm.id
        """, (model["goal_id"],))

        if not rows:
            raise ValueError("没有已标注数据可批跑")

        # 推理
        from inference import IntentInferencer
        inferencer = IntentInferencer(model_dir)
        pairs = [(r["content_text"], r["comment"]) for r in rows]
        predictions = inferencer.predict_batch(pairs, batch_size=32)

        # 清除旧结果
        db.execute("DELETE FROM batch_results WHERE model_id = ?", (model_id,))

        # 写入结果
        conn = db.get_conn()
        try:
            for row, pred in zip(rows, predictions):
                true_score = row["score"]
                pred_score = pred["score"]
                is_correct = (true_score == pred_score)
                pred_label = "has_intent" if pred_score >= SCORE_THRESHOLD else "no_intent"
                conn.execute("""
                    INSERT INTO batch_results (model_id, comment_id, predicted_score,
                                               predicted_label, confidence, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (model_id, row["id"], pred_score, pred_label,
                      pred["confidence"], is_correct))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        # 计算快速报告
        total = len(rows)
        correct = sum(1 for r, p in zip(rows, predictions)
                      if r["score"] == p["score"])
        accuracy = correct / total if total > 0 else 0

        return _json({
            "ok": True,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "message": f"批跑完成: {total} 条, 正确 {correct} 条, 准确率 {accuracy:.2%}",
        })

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "models": _get_models,
        "results": _get_results,
        "report": _get_report,
    }

    _POST = {
        "run": _post_run,
    }
