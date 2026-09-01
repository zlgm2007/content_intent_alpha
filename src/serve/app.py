"""FastAPI 统一推理服务。

同时挂载 NER 与意图两个 ONNX 模型，输入正文+评论，输出统一 JSON。

启动（注意避开管理台默认的 8000 端口）：
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.ner.inference import NERInference
from src.intent.inference import IntentInference

app = FastAPI(title="content_intent_alpha 推理服务", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ner_engine = None
intent_engine = None

try:
    ner_engine = NERInference(os.path.join(BASE_DIR, "configs/ner.yaml"))
    print("[serve] NER 模型加载成功")
except Exception as e:  # noqa: BLE001
    print(f"[serve] NER 模型加载失败（可先训练并导出 ONNX）: {e}")

try:
    intent_engine = IntentInference(os.path.join(BASE_DIR, "configs/intent.yaml"))
    print("[serve] 意图模型加载成功")
except Exception as e:  # noqa: BLE001
    print(f"[serve] 意图模型加载失败（可先训练并导出 ONNX）: {e}")


class PredictRequest(BaseModel):
    title: str = ""      # 标题（可选）
    note: str = ""       # 正文
    comment: str = ""    # 评论


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    source: str = "正文"  # 实体来源段落（标题/正文/评论）


class IntentResult(BaseModel):
    name: str
    score: int
    prob: float
    level: str


class PredictResponse(BaseModel):
    title: str = ""
    note: str = ""
    comment: str = ""
    entities: list[Entity] = []
    intents: list[IntentResult] = []


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ner": ner_engine is not None,
        "intent": intent_engine is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    entities = []
    intents = []

    if ner_engine is not None:
        entities = ner_engine.predict_note_comment(req.title, req.note, req.comment)
    if intent_engine is not None:
        intents = [intent_engine.predict_note_comment(req.title, req.note, req.comment)]

    return PredictResponse(
        title=req.title, note=req.note, comment=req.comment,
        entities=entities, intents=intents,
    )
