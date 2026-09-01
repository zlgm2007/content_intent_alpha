"""意图数据预处理：整理正负样本为 JSONL 标注文件。

支持两种方式：
1. 用户分别准备 data/intent/raw/positive.txt（含目标意图）与
   data/intent/raw/negative.txt（不含目标意图），脚本合并输出 JSONL。
2. 只有 data/intent/raw/all.txt 时，用关键词规则自动打标（供人工校验）。

用法：
    python -m src.intent.preprocess --config configs/intent.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import yaml

# 目标意图的默认正样本关键词（当前为「购物」意图；换意图时需替换为对应意图的关键词）
# 注：本脚本是 CLI 遗留的辅助打标工具，Web 管理台实际走 data_prep_service（用 ES 打分真值），不依赖此关键词表。
DEFAULT_KEYWORDS = [
    "买", "购买", "下单", "多少钱", "价格", "优惠", "促销", "折扣",
    "有货", "库存", "现货", "入手", "想要", "想买", "推荐", "链接",
]


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_from_pos_neg(raw_dir: str, out_file: str) -> int:
    """从 positive.txt / negative.txt 合并生成标注文件。"""
    pos = read_lines(os.path.join(raw_dir, "positive.txt"))
    neg = read_lines(os.path.join(raw_dir, "negative.txt"))
    if not pos and not neg:
        return 0

    rows = [{"text": t, "label": 1} for t in pos] + [{"text": t, "label": 0} for t in neg]
    write_jsonl(out_file, rows)
    print(f"[preprocess] 正样本 {len(pos)} 条, 负样本 {len(neg)} 条 -> {out_file}")
    return len(rows)


def build_from_keywords(raw_dir: str, out_file: str, keywords: List[str]) -> int:
    """从 all.txt 用关键词规则自动打标。"""
    texts = read_lines(os.path.join(raw_dir, "all.txt"))
    rows = []
    for t in texts:
        label = 1 if any(k in t for k in keywords) else 0
        rows.append({"text": t, "label": label})
    write_jsonl(out_file, rows)
    pos = sum(1 for r in rows if r["label"] == 1)
    print(f"[preprocess] 关键词打标: 共 {len(rows)} 条 (正 {pos}) -> {out_file}")
    print("[preprocess] 请人工校验后再训练。")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"] if "raw_dir" in cfg["data"] else "data/intent/raw"
    out_file = cfg["data"]["labeled_file"]

    n = build_from_pos_neg(raw_dir, out_file)
    if n == 0:
        build_from_keywords(raw_dir, out_file, DEFAULT_KEYWORDS)


if __name__ == "__main__":
    main()
