"""词典聚合服务：从 ES 聚合品牌/品类/商品/型号，清洗后合并到本地词典。

- BRAND  <- intentBrand 去重
- CATEGORY <- intentCategory 去重
- PRODUCT <- intentProduct 去重（过滤混入的品类值）
- MODEL   <- 从 intentProduct 里正则提取"字母数字混合"型号片段
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Set

from src.web import db
from src.web.config import DATA_DIR, get_es_config
from src.web.es_client import ESClient

LEXICON_DIR = os.path.join(DATA_DIR, "ner", "lexicon")

# 型号：字母数字混合（P20 / GT2 / S24 Ultra / 4090 ...）
MODEL_RE = re.compile(r"[A-Za-z]{1,8}\d{1,5}[A-Za-z0-9]{0,12}|\d{2,6}[A-Za-z]{1,8}[A-Za-z0-9]{0,12}")

# 明显不是实体的噪声
_BAD_PATTERNS = [re.compile(r"^[\d\W]+$"), re.compile(r"^.$")]  # 纯数字符号 / 单字符
_SEPARATORS = re.compile(r"[、，,;；/|]+")
# 型号里不该出现的单位/容量词（ml/g/kg/km 等）
_UNIT_RE = re.compile(r"\d+\s*(ml|mg|g|kg|km|cm|mm|l|hz|w|v)\b", re.I)


def _clean_keys(k) -> list:
    """拆分复合值（顿号/逗号等分隔）并清洗，返回多个干净词条。"""
    if not k:
        return []
    k = str(k).strip()
    if k.lower() == "null":
        return []
    out = []
    for p in _SEPARATORS.split(k):
        p = p.strip()
        if not p or p.lower() == "null":
            continue
        if any(pat.match(p) for pat in _BAD_PATTERNS):
            continue
        out.append(p)
    return out


def aggregate_lexicon(client: ESClient, indices: List[str], fields: dict, top_n: int = 5000):
    """从 ES 聚合三字段，返回 {BRAND:set, CATEGORY:set, PRODUCT:set}。"""
    result = {"BRAND": set(), "CATEGORY": set(), "PRODUCT": set()}
    for idx in indices:
        for term in client.terms_agg(idx, fields["intent_brand"], top_n):
            result["BRAND"].update(_clean_keys(term["key"]))
        for term in client.terms_agg(idx, fields["intent_category"], top_n):
            result["CATEGORY"].update(_clean_keys(term["key"]))
        for term in client.terms_agg(idx, fields["intent_product"], top_n):
            result["PRODUCT"].update(_clean_keys(term["key"]))
    return result


def extract_models(products: Set[str], categories: Set[str]) -> Set[str]:
    """从商品集合提取型号：含字母数字混合的片段；纯品类词不提取。"""
    models: Set[str] = set()
    for p in products:
        if p in categories:
            continue
        for m in MODEL_RE.findall(p):
            m = m.strip()
            if len(m) >= 2 and not m.isdigit() and not _UNIT_RE.search(m):
                models.add(m)
    return models


def _read_existing(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out


def _write_lexicon(path: str, words: Set[str], header: str):
    words = sorted(w for w in words if w and w.strip())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n")
        for w in words:
            f.write(w + "\n")


def refresh_lexicon_from_es(top_n: int = 5000) -> Dict[str, int]:
    """从 ES 聚合词典并合并到本地文件。返回各类型词条数。"""
    cfg = get_es_config()
    client = ESClient(cfg["hosts"])
    indices = cfg["indices"]
    fields = cfg["fields"]

    agg = aggregate_lexicon(client, indices, fields, top_n)

    # 商品词典：过滤混入的品类值
    products = {p for p in agg["PRODUCT"] if p not in agg["CATEGORY"]}
    # 型号：从商品提取 + 原 model 词典保留
    models = extract_models(agg["PRODUCT"], agg["CATEGORY"])

    targets = {
        "BRAND": ("brand.txt", agg["BRAND"], "品牌词典（ES 聚合）"),
        "CATEGORY": ("category.txt", agg["CATEGORY"], "品类词典（ES 聚合）"),
        "PRODUCT": ("product.txt", products, "商品词典（ES 聚合，已过滤品类）"),
        "MODEL": ("model.txt", models, "型号词典（从商品提取）"),
    }

    result = {}
    for ltype, (fname, new_words, header) in targets.items():
        if ltype == "MODEL":
            # 型号提取不完整，保留历史手动型号（合并）
            existing = _read_existing(os.path.join(LEXICON_DIR, fname))
            merged = existing | new_words
        else:
            # 品牌/品类/商品：聚合为权威全量，覆盖（避免残留旧噪声）
            merged = new_words
        _write_lexicon(os.path.join(LEXICON_DIR, fname), merged, header)
        result[ltype] = len(merged)
        db.add_lexicon_version(ltype, len(merged), "es_aggregate")

    return result
