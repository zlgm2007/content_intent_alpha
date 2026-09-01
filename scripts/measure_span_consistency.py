"""量化「span 不一致」：词典词在语料中出现了 N 次，但只被标注了 M 次。

这是方案 1（词典全量一致标注）要解决的核心问题。

背景澄清
--------
data_prep_service.build_entity_type_map 的注释里写着「64.9% 实体存在同词异标」，
那是**类型归一化之前**的历史数字。当前 conll 已经过归一化，实测同词异标仅 3.0%。
所以「同词异标」基本已解决，真正遗留的是**漏标**：
ES gold 字段里没写的实体，即使正文里明明白白出现了，也一个都不标。

后果：模型学到「该词可标可不标」，推理时给出 0.49 vs 0.44 这类五五开的概率，
产生碎片实体（"小米" 切出 "小" 或 "米1"）。

用法：
    python scripts/measure_span_consistency.py [--docs 3000]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.data_prep_service import (
    DictAnnotator,
    NER_OUT,
    _collect_entity_stats,
    load_dict_entity_map,
)


def read_conll_docs(path: str, max_docs: int = 0):
    """从 conll 还原 doc 文本，并同时抽取该 doc 内已有的 gold 实体。

    返回 (docs, gold_hits)。gold_hits 统计的是**这批文档内部**的标注次数，
    与 dict_hits 严格同口径，无需按全量折算（折算会引入数量级误差）。
    """
    docs, gold_hits = [], Counter()
    cur_chars: list = []
    cur_type, cur_tok = "", []

    def flush_doc():
        nonlocal cur_chars, cur_type, cur_tok
        if cur_type:
            gold_hits["".join(cur_tok)] += 1
        if cur_chars:
            docs.append("".join(cur_chars))
        cur_chars, cur_type, cur_tok = [], "", []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                flush_doc()
                if max_docs and len(docs) >= max_docs:
                    break
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            cur_chars.append(ch)
            if tag.startswith("B-"):
                if cur_type:
                    gold_hits["".join(cur_tok)] += 1
                cur_type, cur_tok = tag[2:], [ch]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_tok.append(ch)
            else:
                if cur_type:
                    gold_hits["".join(cur_tok)] += 1
                cur_type, cur_tok = "", []
    flush_doc()
    return docs, gold_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000, help="抽样文档数，0=全部")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    wmap = load_dict_entity_map()
    if not wmap:
        print("错误：未找到 dict_entity_map.json，请先运行 build_dict_entity_map()")
        return
    ann = DictAnnotator(wmap)

    docs, gold_hits = read_conll_docs(NER_OUT, args.docs)
    print(f"抽样文档数: {len(docs)}")

    # 词典全量标注下的出现次数（与 gold_hits 同批文档，口径一致）
    dict_hits: Counter = Counter()
    for d in docs:
        _, tags = ann.annotate(d)
        cur_type, cur_chars = "", []
        for ch, tg in zip(d, tags):
            if tg.startswith("B-"):
                if cur_type:
                    dict_hits["".join(cur_chars)] += 1
                cur_type, cur_chars = tg[2:], [ch]
            elif tg.startswith("I-") and cur_type == tg[2:]:
                cur_chars.append(ch)
            else:
                if cur_type:
                    dict_hits["".join(cur_chars)] += 1
                cur_type, cur_chars = "", []
        if cur_type:
            dict_hits["".join(cur_chars)] += 1

    rows = []
    for w, n_dict in dict_hits.items():
        rows.append((w, n_dict, gold_hits.get(w, 0)))
    rows.sort(key=lambda x: -x[1])

    total_dict = sum(r[1] for r in rows)
    total_gold = sum(r[2] for r in rows)
    # 只统计"词典也认为该出现"的部分里，gold 覆盖了多少
    covered = sum(min(r[1], r[2]) for r in rows)
    print(f"本批文档中，词典词共出现 {total_dict} 次")
    print(f"现有 conll 在同一批文档上标注了 {total_gold} 次")
    print(f"其中能对应的约 {covered} 次 -> 标注率 {100*covered/max(1,total_dict):.1f}%")
    print(f"=> 漏标约 {100 - 100*covered/max(1,total_dict):.1f}%\n")

    print(f"=== 漏标最严重的 TOP{args.top} ===")
    print(f"{'实体':<20}{'出现':>7}{'已标':>9}{'漏标率':>9}")
    for w, n_dict, n_gold in rows[: args.top]:
        miss = 100 * (1 - n_gold / n_dict) if n_dict else 0
        print(f"{w[:18]:<20}{n_dict:>7}{n_gold:>9.0f}{miss:>8.1f}%")


if __name__ == "__main__":
    main()
