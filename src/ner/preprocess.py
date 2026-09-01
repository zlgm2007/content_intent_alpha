"""NER 数据预处理：词典自动打标 + BIO 编码。

用品牌/品类/商品/型号四个词典对原始语料做最长匹配自动打标，
输出 CONLL 格式文件供人工校验与训练使用。

用法：
    python -m src.ner.preprocess --config configs/ner.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import yaml

# 实体类型 -> 词典文件名
TYPE_FILES = {
    "BRAND": "brand.txt",
    "CATEGORY": "category.txt",
    "PRODUCT": "product.txt",
    "MODEL": "model.txt",
}
# 重叠时的优先级（商品 > 型号 > 品牌 > 品类）
PRIORITY = ["PRODUCT", "MODEL", "BRAND", "CATEGORY"]


def load_lexicon(lexicon_dir: str) -> Dict[str, List[str]]:
    """加载四个词典，按长度降序（最长匹配优先）。"""
    lexicon: Dict[str, List[str]] = {}
    for etype, fname in TYPE_FILES.items():
        path = os.path.join(lexicon_dir, fname)
        words: List[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#"):
                        words.append(w)
        lexicon[etype] = sorted(words, key=len, reverse=True)
    return lexicon


def annotate_text(text: str, lexicon: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """对单条文本做词典匹配打标，返回 (tokens, tags)。字符级切分。"""
    tokens = list(text)
    n = len(tokens)
    tags = ["O"] * n
    occupied = [False] * n  # 是否已被更高优先级实体占用

    for etype in PRIORITY:
        for word in lexicon.get(etype, []):
            if not word:
                continue
            start = 0
            while True:
                idx = text.find(word, start)
                if idx == -1:
                    break
                seg = occupied[idx : idx + len(word)]
                if not any(seg):
                    for j in range(len(word)):
                        pos = idx + j
                        tags[pos] = ("B-" if j == 0 else "I-") + etype
                        occupied[pos] = True
                start = idx + 1
    return tokens, tags


def tokens_to_conll(tokens: List[str], tags: List[str]) -> str:
    lines = ["\t".join([t, g]) for t, g in zip(tokens, tags)]
    return "\n".join(lines) + "\n\n"


def preprocess(raw_dir: str, lexicon_dir: str, out_file: str):
    lexicon = load_lexicon(lexicon_dir)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    total_sent = 0
    total_entities = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        files = sorted(
            p for p in os.listdir(raw_dir) if p.endswith(".txt")
        )
        for fname in files:
            path = os.path.join(raw_dir, fname)
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    text = line.strip()
                    if not text:
                        continue
                    tokens, tags = annotate_text(text, lexicon)
                    total_sent += 1
                    total_entities += sum(1 for t in tags if t.startswith("B-"))
                    fout.write(tokens_to_conll(tokens, tags))

    print(f"[preprocess] 处理完成: {total_sent} 句, 自动标注实体 {total_entities} 个 -> {out_file}")
    print("[preprocess] 请人工校验标注后再训练。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ner.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocess(
        raw_dir=cfg["data"]["raw_dir"],
        lexicon_dir=cfg["data"]["lexicon_dir"],
        out_file=cfg["data"]["annotated_file"],
    )


if __name__ == "__main__":
    main()
