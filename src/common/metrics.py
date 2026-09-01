"""评估指标工具。

- NER：实体级 F1（严格匹配 (type, start, end)）
- 意图：Accuracy / AUC / 分数命中率
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple


def bio_to_entities(tags: Sequence[str], text: str) -> List[dict]:
    """把 BIO 标签序列解码为实体列表。

    Args:
        tags: 与 token 对齐的 BIO 标签序列。
        text: 原始文本（用于回填实体文本，可选）。

    Returns:
        [{"type": "BRAND", "start": int, "end": int, "text": str}, ...]
    """
    entities = []
    current_type = None
    start = -1
    for i, tag in enumerate(tags):
        if tag == "O":
            if current_type is not None:
                entities.append(_build(current_type, start, i, text))
                current_type = None
        elif tag.startswith("B-"):
            if current_type is not None:
                entities.append(_build(current_type, start, i, text))
            current_type = tag[2:]
            start = i
        elif tag.startswith("I-"):
            t = tag[2:]
            if current_type is None or t != current_type:
                # 孤立 I 标签，容错当作新实体开始
                if current_type is not None:
                    entities.append(_build(current_type, start, i, text))
                current_type = t
                start = i
    if current_type is not None:
        entities.append(_build(current_type, start, len(tags), text))
    return entities


def _build(etype: str, start: int, end: int, text: str) -> dict:
    return {
        "type": etype,
        "start": start,
        "end": end,
        "text": text[start:end] if text else "",
    }


def compute_ner_f1(
    pred_entities: Sequence[dict], gold_entities: Sequence[dict]
) -> Tuple[float, float, float]:
    """实体级精确 F1，按 (type, start, end) 严格匹配。"""
    pred_set = {(e["type"], e["start"], e["end"]) for e in pred_entities}
    gold_set = {(e["type"], e["start"], e["end"]) for e in gold_entities}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_per_type_f1(
    pred_entities: Sequence[dict], gold_entities: Sequence[dict]
) -> dict:
    """按实体类型分别计算 F1，便于定位"型号最难"等问题。"""
    types = {e["type"] for e in pred_entities} | {e["type"] for e in gold_entities}
    result = {}
    for t in sorted(types):
        p = [e for e in pred_entities if e["type"] == t]
        g = [e for e in gold_entities if e["type"] == t]
        _, _, f1 = compute_ner_f1(p, g)
        result[t] = f1
    return result


def compute_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if len(preds) == 0:
        return 0.0
    return sum(1 for p, l in zip(preds, labels) if p == l) / len(preds)


def score_hit_rate(
    pred_scores: Sequence[int], gold_scores: Sequence[int], tolerance: int = 1
) -> float:
    """分数命中率：预测分数与人工标注分数差 <= tolerance 视为命中。"""
    if len(pred_scores) == 0:
        return 0.0
    hits = sum(1 for p, g in zip(pred_scores, gold_scores) if abs(p - g) <= tolerance)
    return hits / len(pred_scores)


def per_class_metrics(preds: Sequence[int], labels: Sequence[int]) -> dict:
    """二分类的 per-class precision / recall / f1。"""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, l in zip(preds, labels):
        if p == l:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    out = {}
    for c in sorted(set(preds) | set(labels)):
        pr = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        out[c] = {"precision": pr, "recall": rc, "f1": f1}
    return out
