"""模型管理服务：ONNX 导出、备份、版本记录。"""
from __future__ import annotations

import os
import shutil
from datetime import datetime

from src.web import db
from src.web.config import BASE_DIR, load_yaml


def _backup_if_exists(onnx_path: str):
    """若目标 ONNX 已存在，备份到 backup 目录。"""
    if os.path.exists(onnx_path):
        backup_dir = os.path.join(os.path.dirname(onnx_path), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(onnx_path).replace(".onnx", f"_{ts}.onnx")
        shutil.copy2(onnx_path, os.path.join(backup_dir, name))


def export_ner_onnx() -> dict:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    cfg = load_yaml("ner.yaml")
    model_dir = cfg["train"]["output_dir"]
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise RuntimeError("未找到 NER 模型，请先训练")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=len(cfg["model"]["labels"])
    )
    from src.common.onnx_utils import export_model

    onnx_path = cfg["export"]["onnx_path"]
    _backup_if_exists(onnx_path)
    export_model(
        model, tokenizer,
        onnx_path=onnx_path,
        opset=cfg["export"]["opset"],
        max_length=cfg["model"]["max_length"],
        quantize=cfg["export"]["quantize"],
    )
    db.add_model_version("ner", onnx_path)
    return {"ok": True, "path": onnx_path}


def export_intent_onnx() -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_yaml("intent.yaml")
    model_dir = cfg["train"]["output_dir"]
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise RuntimeError("未找到意图模型，请先训练")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=cfg["model"]["num_labels"]
    )
    from src.common.onnx_utils import export_model

    onnx_path = cfg["export"]["onnx_path"]
    _backup_if_exists(onnx_path)
    export_model(
        model, tokenizer,
        onnx_path=onnx_path,
        opset=cfg["export"]["opset"],
        max_length=cfg["model"]["max_length"],
        quantize=cfg["export"]["quantize"],
    )
    db.add_model_version("intent", onnx_path)
    return {"ok": True, "path": onnx_path}


def list_models() -> dict:
    """返回当前模型文件状态。"""
    result = {}
    for mtype, rel in [("ner", "models/ner/ner.onnx"), ("intent", "models/intent/intent.onnx")]:
        p = os.path.join(BASE_DIR, rel)
        result[mtype] = {
            "onnx": rel,
            "exists": os.path.exists(p),
            "size_mb": round(os.path.getsize(p) / 1024 / 1024, 2) if os.path.exists(p) else 0,
        }
    result["versions"] = db.list_model_versions(20)
    return result
