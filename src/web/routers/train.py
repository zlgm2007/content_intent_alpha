"""训练中心 API：异步训练 NER / 意图模型，SSE 实时推送进度与日志。"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.web import db, train_service

router = APIRouter(prefix="/api/train", tags=["train"])


class TrainIn(BaseModel):
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5


def _start(task_type: str, body: TrainIn) -> dict:
    """启动训练并立即返回任务状态。

    若全局已有任务在跑，本次请求会被拒绝（status=failed + 明确 message），
    前端可据此直接提示用户，而不必等 SSE 推送。
    """
    task_id = train_service.start_training(task_type, body.model_dump())
    task = db.get_train_task(task_id) or {}
    return {
        "task_id": task_id,
        "ok": task.get("status") != "failed",
        "status": task.get("status", ""),
        "message": task.get("message", ""),
    }


@router.post("/ner")
def train_ner(body: TrainIn):
    return _start("ner", body)


@router.post("/intent")
def train_intent(body: TrainIn):
    return _start("intent", body)


@router.get("/tasks")
def list_tasks():
    return db.list_train_tasks(20)


@router.get("/tasks/{task_id}")
def get_task(task_id: int):
    return db.get_train_task(task_id)


@router.get("/tasks/{task_id}/stream")
async def stream(task_id: int):
    """SSE 端点：实时推送训练进度、日志、指标。"""
    async def gen():
        last_seq = -1
        while True:
            events, last_seq = train_service.get_events_after(task_id, last_seq)
            for ev in events:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

            task = db.get_train_task(task_id)
            if task and task["status"] in ("done", "failed", "stopped", "interrupted"):
                if not any(e["type"] in ("done", "failed", "stopped") for e in events):
                    yield f"data: {json.dumps({'type': 'done', 'status': task['status'], 'message': task['message']}, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/tasks/{task_id}/stop")
def stop_task(task_id: int):
    """停止正在运行的训练任务（优雅停止，保留当前最优模型）。"""
    stopped = train_service.stop_training(task_id)
    return {"ok": stopped, "task_id": task_id}
