#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""SQLite 数据库管理 — 建表、索引、连接工厂。

设计要点：
  - WAL 模式：读写不互斥，百万级数据下性能差距巨大
  - 每次请求创建独立连接（sqlite3 连接开销极小，且 ThreadingHTTPServer 多线程）
  - 索引覆盖高频查询路径：按 goal_id 筛选、按 status 筛选、分页
"""
import json
import os
import sqlite3
import threading

from common import DB_PATH, DATA_DIR, ensure_dirs

_init_lock = threading.Lock()
_initialized = False

SCHEMA = """
-- 意图目标表
CREATE TABLE IF NOT EXISTS intent_goals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 作品内容表
CREATE TABLE IF NOT EXISTS contents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id     INTEGER NOT NULL REFERENCES intent_goals(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    metadata    TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 评论表（含标注状态）
CREATE TABLE IF NOT EXISTS comments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id     INTEGER NOT NULL REFERENCES intent_goals(id) ON DELETE CASCADE,
    content_id  INTEGER NOT NULL REFERENCES contents(id) ON DELETE CASCADE,
    comment     TEXT NOT NULL,
    score       INTEGER,                          -- NULL=未标注, 0-5=已标注
    label       TEXT,                             -- NULL=未标注, has_intent / no_intent
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending / labeled / skipped
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 模型记录表
CREATE TABLE IF NOT EXISTS models (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id           INTEGER NOT NULL REFERENCES intent_goals(id) ON DELETE CASCADE,
    onnx_path         TEXT NOT NULL,
    tokenizer_dir     TEXT,
    accuracy          REAL,
    f1_score          REAL,
    num_train_samples INTEGER,
    status            TEXT NOT NULL DEFAULT 'trained',  -- training / trained / failed
    params            TEXT,                              -- JSON: lr, batch_size, epochs, max_length...
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 批跑结果表
CREATE TABLE IF NOT EXISTS batch_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    comment_id      INTEGER NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    predicted_score INTEGER,
    predicted_label TEXT,
    confidence      REAL,
    is_correct      BOOLEAN,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_contents_goal ON contents(goal_id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_goal ON comments(goal_id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_content ON comments(content_id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_status ON comments(goal_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_comments_score ON comments(goal_id, score)",
    "CREATE INDEX IF NOT EXISTS idx_models_goal ON models(goal_id)",
    "CREATE INDEX IF NOT EXISTS idx_batch_model ON batch_results(model_id)",
    "CREATE INDEX IF NOT EXISTS idx_batch_comment ON batch_results(comment_id)",
]


def init_db():
    """初始化数据库：建表 + 索引 + WAL 模式。线程安全，只执行一次。"""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        ensure_dirs()
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            for idx in INDEXES:
                conn.execute(idx)
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def get_conn():
    """返回一个新的 SQLite 连接（WAL + foreign_keys）。

    每次 API 调用创建独立连接，sqlite3 连接开销极小。
    """
    if not _initialized:
        init_db()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row  # 结果以 dict-like 方式访问
    return conn


def query_one(sql, params=()):
    """执行查询，返回单行 dict 或 None。"""
    conn = get_conn()
    try:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def query_all(sql, params=()):
    """执行查询，返回 list[dict]。"""
    conn = get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def execute(sql, params=()):
    """执行写操作，返回 lastrowid。"""
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def executemany(sql, params_list):
    """批量执行写操作（事务），返回受影响行数。"""
    conn = get_conn()
    try:
        cur = conn.executemany(sql, params_list)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
