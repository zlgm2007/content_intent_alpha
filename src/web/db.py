"""Web 管理台 SQLite 存储。

表：
- import_tasks    ES 导入任务
- es_docs         导入的 ES 文档（去重后的核心样本）
- sync_state      各索引增量同步位点
- lexicon_versions 词典版本记录
- train_tasks     训练任务（供后续训练中心使用）
- settings        key-value 设置
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.web.config import WEB_DB_PATH

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS import_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'pending',
    total INTEGER DEFAULT 0,
    message TEXT DEFAULT '',
    params TEXT,
    created_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS es_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_index TEXT,
    comment_content TEXT,
    note_desc TEXT,
    note_title TEXT,
    note_topics TEXT,
    intent_content TEXT,
    intent_brand TEXT,
    intent_category TEXT,
    intent_product TEXT,
    intent_score INTEGER,
    update_time TEXT,
    import_batch TEXT
);
CREATE INDEX IF NOT EXISTS idx_es_docs_score ON es_docs(intent_score);
CREATE INDEX IF NOT EXISTS idx_es_docs_brand ON es_docs(intent_brand);
CREATE INDEX IF NOT EXISTS idx_es_docs_category ON es_docs(intent_category);
CREATE TABLE IF NOT EXISTS sync_state (
    index_name TEXT PRIMARY KEY,
    last_update_time TEXT
);
CREATE TABLE IF NOT EXISTS lexicon_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    count INTEGER,
    source TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS train_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    params TEXT,
    created_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT,
    path TEXT,
    created_at TEXT
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(WEB_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(WEB_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock:
        conn = get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()


# ---------- import_tasks ----------

def create_import_task(params: dict) -> int:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO import_tasks(status, params, created_at) VALUES(?,?,?)",
            ("pending", json.dumps(params, ensure_ascii=False), _now()),
        )
        conn.commit()
        task_id = cur.lastrowid
        conn.close()
    return task_id


def update_import_task(task_id: int, status: str = None, total: int = None,
                       message: str = None, finished: bool = False):
    with _lock:
        conn = get_conn()
        fields, vals = [], []
        if status is not None:
            fields.append("status=?"); vals.append(status)
        if total is not None:
            fields.append("total=?"); vals.append(total)
        if message is not None:
            fields.append("message=?"); vals.append(message)
        if finished:
            fields.append("finished_at=?"); vals.append(_now())
        if fields:
            vals.append(task_id)
            conn.execute(f"UPDATE import_tasks SET {','.join(fields)} WHERE id=?", vals)
            conn.commit()
        conn.close()


def get_import_task(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM import_tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_import_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM import_tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- es_docs ----------

def insert_es_docs(rows: List[dict], batch: str):
    if not rows:
        return
    with _lock:
        conn = get_conn()
        conn.executemany(
            """INSERT INTO es_docs(source_index, comment_content, note_desc, note_title,
               note_topics, intent_content, intent_brand, intent_category, intent_product,
               intent_score, update_time, import_batch)
               VALUES(:source_index, :comment_content, :note_desc, :note_title,
               :note_topics, :intent_content, :intent_brand, :intent_category, :intent_product,
               :intent_score, :update_time, :import_batch)""",
            rows,
        )
        conn.commit()
        conn.close()


def count_es_docs() -> int:
    conn = get_conn()
    row = conn.execute("SELECT COUNT(*) c FROM es_docs").fetchone()
    conn.close()
    return row["c"]


def count_es_docs_by_score() -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT intent_score, COUNT(*) c FROM es_docs GROUP BY intent_score ORDER BY intent_score"
    ).fetchall()
    conn.close()
    return [{"score": r["intent_score"], "count": r["c"]} for r in rows]


def fetch_ner_samples(limit: int = 0) -> List[Dict[str, Any]]:
    """返回正文+评论非空、且至少一个实体非空的样本（正文与评论都参与 NER）。"""
    sql = (
        "SELECT note_title, note_desc, comment_content, "
        "intent_brand, intent_category, intent_product FROM es_docs "
        "WHERE (note_desc IS NOT NULL AND note_desc != '' "
        "   OR comment_content IS NOT NULL AND comment_content != '') "
        "AND (intent_brand IS NOT NULL OR intent_category IS NOT NULL OR intent_product IS NOT NULL)"
    )
    if limit:
        sql += f" ORDER BY RANDOM() LIMIT {int(limit)}"
    conn = get_conn()
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_intent_samples(limit: int = 0) -> List[Dict[str, Any]]:
    """返回评论非空（intent_score 可为空）、且带上正文上下文的样本。"""
    sql = (
        "SELECT note_title, note_desc, comment_content, intent_score FROM es_docs "
        "WHERE comment_content IS NOT NULL AND comment_content != ''"
    )
    if limit:
        sql += f" ORDER BY RANDOM() LIMIT {int(limit)}"
    conn = get_conn()
    rows = conn.execute(sql).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- sync_state ----------

def get_sync_state(index_name: str) -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT last_update_time FROM sync_state WHERE index_name=?", (index_name,)).fetchone()
    conn.close()
    return row["last_update_time"] if row else None


def set_sync_state(index_name: str, last_update_time: str):
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO sync_state(index_name, last_update_time) VALUES(?,?) "
            "ON CONFLICT(index_name) DO UPDATE SET last_update_time=excluded.last_update_time",
            (index_name, last_update_time),
        )
        conn.commit()
        conn.close()


def clear_all_data():
    """清空已导入文档与增量位点（用于全量重导）。"""
    with _lock:
        conn = get_conn()
        conn.execute("DELETE FROM es_docs")
        conn.execute("DELETE FROM sync_state")
        conn.commit()
        conn.close()


def mark_interrupted_running_tasks():
    """进程重启后，把遗留的 running 训练任务标记为 interrupted（线程已随进程终止）。"""
    with _lock:
        conn = get_conn()
        conn.execute(
            "UPDATE train_tasks SET status='interrupted', message='服务重启，训练被中断', finished_at=? WHERE status='running'",
            (_now(),),
        )
        conn.commit()
        conn.close()


# ---------- lexicon_versions ----------

def add_lexicon_version(ltype: str, count: int, source: str):
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO lexicon_versions(type, count, source, created_at) VALUES(?,?,?,?)",
            (ltype, count, source, _now()),
        )
        conn.commit()
        conn.close()


def list_lexicon_versions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM lexicon_versions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- train_tasks ----------

def create_train_task(task_type: str, params: dict) -> int:
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO train_tasks(task_type, status, params, created_at) VALUES(?,?,?,?)",
            (task_type, "pending", json.dumps(params, ensure_ascii=False), _now()),
        )
        conn.commit()
        task_id = cur.lastrowid
        conn.close()
    return task_id


def update_train_task(task_id: int, status: str = None, progress: float = None,
                      message: str = None, finished: bool = False):
    with _lock:
        conn = get_conn()
        fields, vals = [], []
        if status is not None:
            fields.append("status=?"); vals.append(status)
        if progress is not None:
            fields.append("progress=?"); vals.append(progress)
        if message is not None:
            fields.append("message=?"); vals.append(message)
        if finished:
            fields.append("finished_at=?"); vals.append(_now())
        if fields:
            vals.append(task_id)
            conn.execute(f"UPDATE train_tasks SET {','.join(fields)} WHERE id=?", vals)
            conn.commit()
        conn.close()


def get_train_task(task_id: int) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM train_tasks WHERE id=?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_train_tasks(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM train_tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_running_train_task() -> Optional[Dict[str, Any]]:
    """查询当前正在运行的训练任务（全局只允许一个）。

    同时把 status='running' 且属于历史遗留（进程重启后仍在 running）的任务排除：
    这类任务由 mark_interrupted_running_tasks() 在启动时统一标记为 interrupted。
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM train_tasks WHERE status='running' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- model_versions ----------

def add_model_version(model_type: str, path: str):
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO model_versions(model_type, path, created_at) VALUES(?,?,?)",
            (model_type, path, _now()),
        )
        conn.commit()
        conn.close()


def list_model_versions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM model_versions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
