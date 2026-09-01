"""数据准备 API：从已导入数据生成训练集。"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.web import data_prep_service as dps

router = APIRouter(prefix="/api/data", tags=["data"])


class NERDataIn(BaseModel):
    limit: int = 0               # 0 = 全部
    min_entities: int = 1        # 至少标注到几个实体才保留
    max_per_pattern: int = 5     # 每种「实体组合」最多保留几条（0=不去重，生成全量）
    rebuild_type_map: bool = False  # 强制重建实体类型归一化映射
    # 标注方式：
    #   "gold" 只标该样本 ES gold 字段里出现的字符串（旧，实测漏标 51.1%）
    #   "dict" 词典全量一致标注（新，同词每次出现都标、类型全局固定）
    annotator: str = "dict"
    # 留出比例（在模式去重之前切分）。>0 时额外产出 holdout.conll 用于评估泛化。
    holdout_ratio: float = 0.1


class IntentDataIn(BaseModel):
    limit: int = 0
    balance: bool = True
    max_per_class: int = 20000
    # 随机留出比例（在平衡之前切分）。强烈建议 >0，否则 0/2/4 类会被 100% 纳入
    # 训练、留出集退化成 3 分类，无法评估泛化能力。
    holdout_ratio: float = 0.1


@router.post("/ner/prepare")
def prepare_ner(body: NERDataIn):
    return dps.build_ner_data(
        limit=body.limit,
        min_entities=body.min_entities,
        rebuild_type_map=body.rebuild_type_map,
        max_per_pattern=body.max_per_pattern,
        annotator=body.annotator,
        holdout_ratio=body.holdout_ratio,
    )


@router.post("/intent/prepare")
def prepare_intent(body: IntentDataIn):
    return dps.build_intent_data(
        body.limit, body.balance, body.max_per_class, body.holdout_ratio
    )
