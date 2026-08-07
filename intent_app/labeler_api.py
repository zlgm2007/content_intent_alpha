#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""数据标注 API — /lbl/api/*

功能：
  - contents:        作品列表（分页）
  - content_detail:  单个作品 + 关联评论
  - content_create:  创建作品
  - content_delete:  删除作品
  - comment_add:     添加单条评论
  - comment_batch:   批量添加评论
  - comment_label:   标注评论（打分 0-5）
  - comment_unlabel: 取消标注
  - comment_delete:  删除评论
  - labeled_list:    已标注评论列表（分页，供训练/批跑用）
  - stats:           标注统计
  - import_data:     批量导入（JSON 格式）
"""
import json

import db


def _json(obj):
    return obj, None

# 0-2 分 → no_intent, 3-5 分 → has_intent
SCORE_THRESHOLD = 3


def _score_to_label(score):
    if score is None:
        return None
    return "has_intent" if score >= SCORE_THRESHOLD else "no_intent"


class LabelerAPI:
    """数据标注 API。"""

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def _get_contents(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "20")), 100)
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        # 合并 手工录入(contents) + 外部同步(works) 作品，来源用 source 区分
        rows = db.query_all("""
            SELECT * FROM (
                SELECT c.id AS id, 'manual' AS source,
                       c.text AS text, c.note_id AS note_id,
                       c.note_title AS note_title, c.platform AS platform,
                       c.source_index AS source_index, c.created_at AS created_at,
                       (SELECT COUNT(*) FROM comments cm WHERE cm.content_id = c.id) AS comment_count,
                       (SELECT COUNT(*) FROM comments cm WHERE cm.content_id = c.id
                        AND cm.status = 'labeled') AS labeled_count
                FROM contents c WHERE c.goal_id = ?
                UNION ALL
                SELECT w.id AS id, 'es' AS source,
                       COALESCE(w.content, w.note_title, '') AS text,
                       w.note_id AS note_id, w.note_title AS note_title,
                       w.platform AS platform, w.source_index AS source_index,
                       w.created_at AS created_at,
                       (SELECT COUNT(*) FROM annotations a WHERE a.work_id = w.id) AS comment_count,
                       (SELECT COUNT(*) FROM annotations a WHERE a.work_id = w.id
                        AND a.status = 'labeled') AS labeled_count
                FROM works w WHERE w.goal_id = ?
            )
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """, (goal_id, goal_id, size, offset))
        total = db.query_one("""
            SELECT (SELECT COUNT(*) FROM contents WHERE goal_id = ?)
                 + (SELECT COUNT(*) FROM works WHERE goal_id = ?) AS n
        """, (goal_id, goal_id))["n"]
        return _json({"contents": rows, "total": total, "page": page, "size": size})

    def _get_content_detail(self, q):
        content_id = int(self._q(q, "id", "0"))
        if content_id <= 0:
            raise ValueError("缺少 id")
        content = db.query_one("SELECT * FROM contents WHERE id = ?", (content_id,))
        if not content:
            raise ValueError("作品不存在")
        comments = db.query_all(
            "SELECT * FROM comments WHERE content_id = ? ORDER BY id",
            (content_id,))
        return _json({"content": content, "comments": comments})

    def _get_labeled_list(self, q):
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "50")), 500)
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        rows = db.query_all("""
            SELECT * FROM (
                SELECT cm.id AS id, cm.comment AS comment, cm.score AS score,
                       cm.label AS label, ct.text AS content_text,
                       ct.id AS content_id, 'manual' AS source
                FROM comments cm JOIN contents ct ON cm.content_id = ct.id
                WHERE cm.goal_id = ? AND cm.status = 'labeled'
                UNION ALL
                SELECT a.id AS id, COALESCE(a.comment, '') AS comment,
                       a.score AS score, a.label AS label,
                       COALESCE(w.content, w.note_title, '') AS content_text,
                       w.id AS content_id, 'es' AS source
                FROM annotations a JOIN works w ON a.work_id = w.id
                WHERE a.goal_id = ? AND a.status = 'labeled'
            ) ORDER BY id DESC LIMIT ? OFFSET ?
        """, (goal_id, goal_id, size, offset))
        total = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'labeled') "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'labeled') AS n",
            (goal_id, goal_id))["n"]
        return _json({"items": rows, "total": total, "page": page, "size": size})

    def _get_stats(self, q):
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
        unlabeled = db.query_one(
            "SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'pending') "
            "+ (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'pending') AS n",
            (goal_id, goal_id))["n"]
        # 按分数分布（手工 + 外部合并）
        score_dist = db.query_all("""
            SELECT score, SUM(cnt) AS count FROM (
                SELECT score, COUNT(*) AS cnt FROM comments
                WHERE goal_id = ? AND status = 'labeled' GROUP BY score
                UNION ALL
                SELECT score, COUNT(*) AS cnt FROM annotations
                WHERE goal_id = ? AND status = 'labeled' GROUP BY score
            ) GROUP BY score ORDER BY score
        """, (goal_id, goal_id))
        return _json({
            "total": total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "score_distribution": {str(r["score"]): r["count"] for r in score_dist},
        })

    def _get_unlabeled(self, q):
        """获取未标注评论（分页，供快速标注用）。合并手工(comments) + 外部(annotations)。"""
        goal_id = int(self._q(q, "goal_id", "0"))
        page = int(self._q(q, "page", "1"))
        size = min(int(self._q(q, "size", "20")), 100)
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        offset = (page - 1) * size
        rows = db.query_all("""
            SELECT * FROM (
                SELECT cm.id AS id, 'manual' AS source,
                       cm.comment AS comment, cm.raw_score AS raw_score,
                       ct.text AS content_text
                FROM comments cm JOIN contents ct ON cm.content_id = ct.id
                WHERE cm.goal_id = ? AND cm.status = 'pending'
                UNION ALL
                SELECT a.id AS id, 'es' AS source,
                       COALESCE(a.comment, '') AS comment, a.raw_score AS raw_score,
                       COALESCE(w.content, w.note_title, '') AS content_text
                FROM annotations a JOIN works w ON a.work_id = w.id
                WHERE a.goal_id = ? AND a.status = 'pending'
            ) ORDER BY id DESC LIMIT ? OFFSET ?
        """, (goal_id, goal_id, size, offset))
        total = db.query_one("""
            SELECT (SELECT COUNT(*) FROM comments WHERE goal_id = ? AND status = 'pending')
                 + (SELECT COUNT(*) FROM annotations WHERE goal_id = ? AND status = 'pending') AS n
        """, (goal_id, goal_id))["n"]
        return _json({"items": rows, "total": total, "page": page, "size": size})

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_content_create(self, body):
        goal_id = int(body.get("goal_id", 0))
        text = str(body.get("text", "")).strip()
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        if not text:
            raise ValueError("作品内容不能为空")
        metadata = body.get("metadata")
        meta_str = json.dumps(metadata, ensure_ascii=False) if metadata else None
        cid = db.execute(
            "INSERT INTO contents (goal_id, text, metadata) VALUES (?, ?, ?)",
            (goal_id, text, meta_str))
        content = db.query_one("SELECT * FROM contents WHERE id = ?", (cid,))
        return _json({"ok": True, "content": content})

    def _post_content_delete(self, body):
        content_id = int(body.get("id", 0))
        if content_id <= 0:
            raise ValueError("缺少 id")
        db.execute("DELETE FROM contents WHERE id = ?", (content_id,))
        return _json({"ok": True})

    def _post_comment_add(self, body):
        goal_id = int(body.get("goal_id", 0))
        content_id = int(body.get("content_id", 0))
        comment = str(body.get("comment", "")).strip()
        if goal_id <= 0 or content_id <= 0:
            raise ValueError("缺少 goal_id 或 content_id")
        if not comment:
            raise ValueError("评论内容不能为空")
        cid = db.execute(
            "INSERT INTO comments (goal_id, content_id, comment) VALUES (?, ?, ?)",
            (goal_id, content_id, comment))
        row = db.query_one("SELECT * FROM comments WHERE id = ?", (cid,))
        return _json({"ok": True, "comment": row})

    def _post_comment_batch(self, body):
        """批量添加评论。body: {content_id, comments: ["...", "..."]}"""
        goal_id = int(body.get("goal_id", 0))
        content_id = int(body.get("content_id", 0))
        comments = body.get("comments", [])
        if goal_id <= 0 or content_id <= 0:
            raise ValueError("缺少 goal_id 或 content_id")
        if not comments or not isinstance(comments, list):
            raise ValueError("comments 必须为非空列表")
        params = [(goal_id, content_id, str(c).strip())
                  for c in comments if str(c).strip()]
        if not params:
            raise ValueError("没有有效评论")
        count = db.executemany(
            "INSERT INTO comments (goal_id, content_id, comment) VALUES (?, ?, ?)",
            params)
        return _json({"ok": True, "imported": count})

    def _post_comment_label(self, body):
        comment_id = int(body.get("id", 0))
        score = int(body.get("score", -1))
        if comment_id <= 0:
            raise ValueError("缺少评论 id")
        if score < 0 or score > 5:
            raise ValueError("分数必须为 0-5")
        label = _score_to_label(score)
        db.execute(
            "UPDATE comments SET score = ?, label = ?, status = 'labeled' WHERE id = ?",
            (score, label, comment_id))
        row = db.query_one("SELECT * FROM comments WHERE id = ?", (comment_id,))
        return _json({"ok": True, "comment": row})

    def _post_comment_unlabel(self, body):
        comment_id = int(body.get("id", 0))
        if comment_id <= 0:
            raise ValueError("缺少评论 id")
        db.execute(
            "UPDATE comments SET score = NULL, label = NULL, status = 'pending' WHERE id = ?",
            (comment_id,))
        return _json({"ok": True})

    def _post_comment_delete(self, body):
        comment_id = int(body.get("id", 0))
        if comment_id <= 0:
            raise ValueError("缺少评论 id")
        db.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
        return _json({"ok": True})

    def _post_import_data(self, body):
        """批量导入数据（JSON 格式）。

        格式：
        {
          "goal_id": 1,
          "items": [
            {
              "content": "作品内容",
              "comments": [
                {"text": "评论1", "score": 5},
                {"text": "评论2", "score": 0},
                {"text": "评论3"}   // 未标注
              ]
            }
          ]
        }
        """
        goal_id = int(body.get("goal_id", 0))
        items = body.get("items", [])
        if goal_id <= 0:
            raise ValueError("缺少 goal_id")
        if not items or not isinstance(items, list):
            raise ValueError("items 必须为非空列表")

        content_count = 0
        comment_count = 0
        conn = db.get_conn()
        try:
            for item in items:
                content_text = str(item.get("content", "")).strip()
                if not content_text:
                    continue
                cur = conn.execute(
                    "INSERT INTO contents (goal_id, text) VALUES (?, ?)",
                    (goal_id, content_text))
                content_id = cur.lastrowid
                content_count += 1
                for c in item.get("comments", []):
                    comment_text = str(c.get("text", "")).strip()
                    if not comment_text:
                        continue
                    score = c.get("score")
                    if score is not None:
                        score = int(score)
                        if score < 0 or score > 5:
                            score = None
                    if score is not None:
                        conn.execute(
                            "INSERT INTO comments (goal_id, content_id, comment, score, label, status) "
                            "VALUES (?, ?, ?, ?, ?, 'labeled')",
                            (goal_id, content_id, comment_text, score, _score_to_label(score)))
                    else:
                        conn.execute(
                            "INSERT INTO comments (goal_id, content_id, comment) VALUES (?, ?, ?)",
                            (goal_id, content_id, comment_text))
                    comment_count += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return _json({"ok": True, "contents": content_count, "comments": comment_count})

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "contents": _get_contents,
        "content_detail": _get_content_detail,
        "labeled_list": _get_labeled_list,
        "unlabeled": _get_unlabeled,
        "stats": _get_stats,
    }

    _POST = {
        "content_create": _post_content_create,
        "content_delete": _post_content_delete,
        "comment_add": _post_comment_add,
        "comment_batch": _post_comment_batch,
        "comment_label": _post_comment_label,
        "comment_unlabel": _post_comment_unlabel,
        "comment_delete": _post_comment_delete,
        "import_data": _post_import_data,
    }
