"""系统与总览 API。"""
from __future__ import annotations

import os

from fastapi import APIRouter

from src.web import db
from src.web.config import get_es_config, load_yaml
from src.web.es_client import ESClient

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    es_ok = False
    es_info = None
    try:
        es_info = ESClient(get_es_config()["hosts"]).ping()
        es_ok = True
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "es_ok": es_ok,
        "es_cluster": (es_info or {}).get("cluster_name"),
        "es_version": (es_info or {}).get("version", {}).get("number"),
    }


@router.get("/intent/config")
def intent_config():
    """返回当前目标意图的配置（意图名、打分范围、ES 打分字段）。

    换一类意图时：改 configs/intent.yaml 的 model.intent_name，以及
    configs/es.yaml 的 fields.intent_score（指向新意图的 ES 打分字段），
    重新导入数据 → 数据准备 → 训练 → 导出 ONNX 即可，训练/推理链路无需改代码。
    """
    cfg = load_yaml("intent.yaml")
    model = cfg.get("model", {})
    score_field = get_es_config()["fields"].get("intent_score", "intentScore")
    return {
        "name": model.get("intent_name", "购物"),
        "num_labels": model.get("num_labels", 6),
        "es_score_field": score_field,
        "hint": "换意图：改 intent.yaml 的 intent_name + es.yaml 的 fields.intent_score，再重导数据/重训",
    }


@router.get("/stats")
def stats():
    score_dist = db.count_es_docs_by_score()
    lexicon = {}
    lexicon_dir = os.path.join("data", "ner", "lexicon")
    for f in ["brand", "category", "product", "model"]:
        p = os.path.join(lexicon_dir, f + ".txt")
        n = 0
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fp:
                n = sum(1 for ln in fp if ln.strip() and not ln.strip().startswith("#"))
        lexicon[f] = n
    return {
        "es_docs": db.count_es_docs(),
        "score_dist": score_dist,
        "lexicon": lexicon,
        "recent_imports": db.list_import_tasks(5),
    }
