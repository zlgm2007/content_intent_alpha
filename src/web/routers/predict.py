"""在线试测 API：加载已训练的模型，实时识别实体与意图得分。

模型缓存策略（重要）
--------------------
模型实例是模块级单例，避免每次请求都重新加载 ONNX。但单例必须能失效，否则有两个
真实缺陷：

1. 训练中心训完 → 立刻去在线试测看效果，看到的还是**旧模型**，必须重启后端才生效。
   （ONNX 文件在磁盘上已被替换，但 onnxruntime 会话还持有旧图）
2. 若首次请求时 ONNX 尚未导出（还没训过），`_failed` 一旦置 True 就**永不重试**，
   之后即使训完并导出，仍然一直提示「模型未就绪」，同样只能靠重启解决。

这里按 ONNX 文件的 **mtime** 做自动失效：文件首次出现 / 被重训替换 / 被重新导出，
mtime 就会变，下一次请求自动重新加载。用 mtime 而不是在训练回调里清缓存，是为了
同时覆盖「用脚本直接重训、完全不走 API」的场景（如 scripts/retrain_ner.py）。

另提供 POST /api/predict/reload 手动强制重载，以及 GET /api/predict/status 查看状态。
"""
from __future__ import annotations

import os
import time

from fastapi import APIRouter
from pydantic import BaseModel

from src.web.config import BASE_DIR

router = APIRouter(prefix="/api/predict", tags=["predict"])


class _ModelCache:
    """按 ONNX 文件 mtime 自动失效的模型单例。"""

    def __init__(self, name: str, onnx_rel: str, config_rel: str):
        self.name = name
        self.onnx_path = os.path.join(BASE_DIR, onnx_rel)
        self.config_path = os.path.join(BASE_DIR, config_rel)
        self._inst = None
        self._mtime = 0.0      # 最近一次加载尝试时的文件 mtime（0 = 文件不存在）
        self._failed = False

    @property
    def loaded(self) -> bool:
        return self._inst is not None

    def _cur_mtime(self) -> float:
        try:
            return os.path.getmtime(self.onnx_path)
        except OSError:
            return 0.0

    def invalidate(self):
        """强制丢弃缓存实例，下次 get() 重新加载。"""
        self._inst = None
        self._mtime = 0.0
        self._failed = False

    def get(self, loader):
        cur = self._cur_mtime()
        if cur != self._mtime:
            # 文件首次出现 / 被替换 / 被删除 → 丢弃旧实例重试。
            # 这同时修复了「加载失败后被永久记住」的问题。
            self._inst = None
            self._failed = False
            self._mtime = cur
        if self._inst is None and not self._failed:
            try:
                self._inst = loader(self.config_path)
            except Exception:  # noqa: BLE001
                self._failed = True
        return self._inst

    def status(self) -> dict:
        mt = self._cur_mtime()
        return {
            "loaded": self.loaded,
            "onnx": self.onnx_path,
            "exists": mt > 0,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)) if mt else None,
        }


def _new_ner(config_path: str):
    from src.ner.inference import NERInference

    return NERInference(config_path)


def _new_intent(config_path: str):
    from src.intent.inference import IntentInference

    return IntentInference(config_path)


_NER = _ModelCache("ner", "models/ner/ner.onnx", "configs/ner.yaml")
_INTENT = _ModelCache("intent", "models/intent/intent.onnx", "configs/intent.yaml")


class PredictIn(BaseModel):
    title: str = ""      # 标题（可选）
    note: str = ""       # 正文（笔记描述）
    comment: str = ""    # 评论


@router.post("")
def predict(body: PredictIn):
    entities = []
    intents = []
    errors = []

    ner = _NER.get(_new_ner)
    if ner is not None:
        entities = ner.predict_note_comment(body.title, body.note, body.comment)
    else:
        errors.append("NER 模型未就绪（需先训练并导出 ONNX）")

    intent = _INTENT.get(_new_intent)
    if intent is not None:
        intents = [intent.predict_note_comment(body.title, body.note, body.comment)]
    else:
        errors.append("意图模型未就绪（需先训练并导出 ONNX）")

    return {
        "title": body.title,
        "note": body.note,
        "comment": body.comment,
        "entities": entities,
        "intents": intents,
        "errors": errors,
    }


@router.get("/status")
def predict_status():
    """查看两个模型的加载状态与 ONNX 时间戳（排查「训完没生效」用）。"""
    return {"ner": _NER.status(), "intent": _INTENT.status()}


@router.post("/reload")
def reload_models():
    """强制丢弃缓存并重新加载模型（重训后想立即生效时可手动调用）。"""
    _NER.invalidate()
    _INTENT.invalidate()
    ner = _NER.get(_new_ner)
    intent = _INTENT.get(_new_intent)
    return {
        "ner": "loaded" if ner else "unavailable",
        "intent": "loaded" if intent else "unavailable",
    }
