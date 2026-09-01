"""ES 数据接入 API：配置、连接测试、导入任务、词典聚合。"""
from __future__ import annotations

import threading
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.web import db
from src.web.config import get_es_config, save_es_config
from src.web.es_client import ESClient
from src.web.es_importer import run_import
from src.web.lexicon_service import refresh_lexicon_from_es

router = APIRouter(prefix="/api/es", tags=["es"])


class ESConfigIn(BaseModel):
    hosts: List[str]
    indices: List[str]
    scroll_size: int = 1000
    slices: int = 4
    only_core: bool = True


class ImportIn(BaseModel):
    indices: Optional[List[str]] = None
    only_core: bool = True
    incremental: bool = True
    max_docs: int = 0


@router.get("/config")
def get_config():
    return get_es_config()


@router.put("/config")
def update_config(cfg: ESConfigIn):
    full = get_es_config()
    full["hosts"] = cfg.hosts
    full["indices"] = cfg.indices
    full["scroll_size"] = cfg.scroll_size
    full["slices"] = cfg.slices
    full["only_core"] = cfg.only_core
    save_es_config(full)
    return {"ok": True, "config": full}


@router.post("/test")
def test_connection():
    cfg = get_es_config()
    try:
        client = ESClient(cfg["hosts"])
        info = client.ping()
        counts = {}
        for idx in cfg["indices"]:
            counts[idx] = client.count(idx)
        return {
            "ok": True,
            "cluster": info.get("cluster_name"),
            "version": info.get("version", {}).get("number"),
            "counts": counts,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@router.post("/import")
def start_import(body: ImportIn):
    task_id = db.create_import_task(body.model_dump())
    t = threading.Thread(target=run_import, args=(task_id, body.model_dump()), daemon=True)
    t.start()
    return {"task_id": task_id}


@router.get("/import/tasks")
def list_tasks():
    return db.list_import_tasks(20)


@router.get("/import/tasks/{task_id}")
def get_task(task_id: int):
    return db.get_import_task(task_id)


@router.post("/lexicon/refresh")
def refresh_lexicon():
    """从 ES 聚合四词典并合并到本地文件（同步执行，聚合量大时可能较慢）。"""
    result = refresh_lexicon_from_es()
    return {"ok": True, "result": result}


@router.post("/clear")
def clear_data():
    """清空已导入文档与增量位点（用于全量重导）。"""
    db.clear_all_data()
    return {"ok": True, "message": "已清空导入数据与增量位点"}
