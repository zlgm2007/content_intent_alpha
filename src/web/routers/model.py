"""模型管理 API：ONNX 导出、下载、版本列表。"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.web import model_service
from src.web.config import BASE_DIR

router = APIRouter(prefix="/api/model", tags=["model"])

_ONNX_MAP = {
    "ner": ("models/ner/ner.onnx", "ner.onnx"),
    "intent": ("models/intent/intent.onnx", "intent.onnx"),
}


@router.post("/export/{model_type}")
def export_model(model_type: str):
    if model_type not in _ONNX_MAP:
        raise HTTPException(400, f"未知模型类型: {model_type}")
    try:
        if model_type == "ner":
            return model_service.export_ner_onnx()
        return model_service.export_intent_onnx()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@router.get("")
def list_models():
    return model_service.list_models()


@router.get("/download/{model_type}")
def download(model_type: str):
    if model_type not in _ONNX_MAP:
        raise HTTPException(400, f"未知模型类型: {model_type}")
    rel, filename = _ONNX_MAP[model_type]
    path = os.path.join(BASE_DIR, rel)
    if not os.path.exists(path):
        raise HTTPException(404, f"ONNX 不存在，请先导出: {rel}")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")
