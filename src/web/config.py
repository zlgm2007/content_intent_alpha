"""Web 管理台配置加载。

提供项目根目录、各路径、ES 配置的读取。
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_DIR = os.path.join(BASE_DIR, "configs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEB_DB_PATH = os.path.join(DATA_DIR, "web.db")

# ES 配置默认值
DEFAULT_ES_CONFIG: Dict[str, Any] = {
    "hosts": ["http://10.104.214.120:9200"],
    "indices": ["lead_clue_score_ge3", "lead_clue_score_lt3"],
    "fields": {
        "text_comment": "commentContent",
        "text_desc": "noteDesc",
        "text_title": "noteTitle",
        "text_topics": "noteTopics",
        "intent_content": "intentContent",
        "intent_brand": "intentBrand",
        "intent_category": "intentCategory",
        "intent_product": "intentProduct",
        "intent_score": "intentScore",
        "update_time": "updateTime",
    },
    "scroll_size": 1000,
    "scroll_timeout": "2m",
    "slices": 4,
    "only_core": True,
}


def load_yaml(name: str) -> Dict[str, Any]:
    """加载 configs 下的 yaml，缺失返回空 dict。"""
    path = os.path.join(CONFIG_DIR, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_es_config() -> Dict[str, Any]:
    """读取 configs/es.yaml，覆盖默认值。"""
    cfg = DEFAULT_ES_CONFIG.copy()
    user = load_yaml("es.yaml").get("es", {})
    cfg.update(user)
    return cfg


def save_es_config(cfg: Dict[str, Any]):
    """写回 configs/es.yaml。"""
    path = os.path.join(CONFIG_DIR, "es.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"es": cfg}, f, allow_unicode=True, sort_keys=False)
