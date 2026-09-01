"""词典冲突分析：决定「同词多类型」时该选哪个类型。

背景（方案 1：词典全量一致标注）
--------------------------------
原标注逻辑 `_align` 只标注「该样本 ES gold 字段里出现的字符串」，导致：
  - 同一词在不同样本里时标时不标（span 不一致）：「汽车」出现 5324 次只标 41.3%
  - 同一词在不同样本里标成不同类型（同词异标）：64.9% 实体存在类型冲突
模型学到的是模糊边界 + 不完整召回，推理时产生 0.49 vs 0.44 五五开和碎片实体。

改为词典全量标注后，每个词典词在**每次出现**时都被标注，且类型固定。
这就带来一个新问题：一个词可能同时属于多个词典（如「小米」既是品牌也是商品）。
本脚本量化冲突规模，并给出基于 ES gold 统计的消解方案。

用法：
    python scripts/analyze_lexicon_conflict.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.config import DATA_DIR
from src.web.data_prep_service import ENTITY_TYPE_MAP_FILE, NER_OUT

LEX_DIR = os.path.join(DATA_DIR, "ner", "lexicon")
TYPES = ["BRAND", "CATEGORY", "PRODUCT", "MODEL"]
FILEMAP = {
    "BRAND": "brand.txt",
    "CATEGORY": "category.txt",
    "PRODUCT": "product.txt",
    "MODEL": "model.txt",
}


def load_lexicon() -> Dict[str, Set[str]]:
    lex = {}
    for t in TYPES:
        path = os.path.join(LEX_DIR, FILEMAP[t])
        words = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#"):
                        words.add(w)
        lex[t] = words
    return lex


def collect_entities_from_conll(path: str) -> Dict[str, Counter]:
    """从 conll（ES gold 标注结果）统计每个实体文本被标成各类型的次数。

    注意：conll 行格式是 `字\\tTAG`（制表符），必须用 split('\\t')，
    用 split() 会被含空格的 token 破坏（踩过的坑）。
    """
    stat: Dict[str, Counter] = defaultdict(Counter)
    cur_type, cur_chars = "", []

    def flush():
        nonlocal cur_type, cur_chars
        if cur_type and cur_chars:
            stat["".join(cur_chars)][cur_type] += 1
        cur_type, cur_chars = "", []

    if not os.path.exists(path):
        return stat
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                flush()
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            if tag.startswith("B-"):
                flush()
                cur_type, cur_chars = tag[2:], [ch]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_chars.append(ch)
            else:
                flush()
    flush()
    return stat


def main():
    lex = load_lexicon()
    print("=== 词典规模 ===")
    for t in TYPES:
        print(f"  {t:<9} {len(lex[t]):>6}")
    all_words = set().union(*[lex[t] for t in TYPES])
    print(f"  {'去重合计':<9} {len(all_words):>6}")

    print("\n=== 跨词典冲突 ===")
    owner: Dict[str, List[str]] = defaultdict(list)
    for w in all_words:
        ts = [t for t in TYPES if w in lex[t]]
        owner[w] = ts
    conflict = {w: ts for w, ts in owner.items() if len(ts) > 1}
    print(f"  冲突词数: {len(conflict)} / {len(all_words)}  ({100*len(conflict)/max(1,len(all_words)):.1f}%)")
    pair_cnt = Counter()
    for w, ts in conflict.items():
        pair_cnt[tuple(sorted(ts))] += 1
    print("  冲突组合 TOP:")
    for pair, n in pair_cnt.most_common(12):
        print(f"    {' + '.join(pair):<32} {n:>5}")
        ex = [w for w, ts in conflict.items() if tuple(sorted(ts)) == pair][:6]
        print(f"      例: {', '.join(ex)}")

    print("\n=== 用 ES gold 统计消解冲突 ===")
    stat = collect_entities_from_conll(NER_OUT)
    hit = {w: stat[w] for w in all_words if w in stat}
    print(f"  词典词中有 ES gold 标注记录的: {len(hit)} / {len(all_words)} "
          f"({100*len(hit)/max(1,len(all_words)):.1f}%)")

    resolved, unresolved = {}, []
    for w, ts in owner.items():
        c = stat.get(w)
        if c and sum(c.values()) > 0:
            # 只在候选类型里选，避免选到词典未收录的类型
            cands = [(t, c.get(t, 0)) for t in ts]
            best_t, best_n = max(cands, key=lambda x: (x[1], -TYPES.index(x[0])))
            if best_n > 0:
                resolved[w] = best_t
                continue
        unresolved.append((w, ts))

    print(f"  由 ES gold 统计消解: {len(resolved)}")
    print(f"  无 gold 记录、需靠默认优先级: {len(unresolved)}")

    if unresolved:
        print("\n  未消解样例（按词长排序，长词风险更高）:")
        for w, ts in sorted(unresolved, key=lambda x: -len(x[0]))[:15]:
            print(f"    {w:<20} {ts}")

    # 默认优先级下的结果（仅用于未消解的词）
    PRIORITY = ["BRAND", "PRODUCT", "MODEL", "CATEGORY"]
    print(f"\n  默认优先级: {' > '.join(PRIORITY)}")
    fb = Counter()
    for w, ts in unresolved:
        for t in PRIORITY:
            if t in ts:
                fb[t] += 1
                break
    print(f"  未消解词按优先级归类: {dict(fb)}")

    final = dict(resolved)
    for w, ts in unresolved:
        for t in PRIORITY:
            if t in ts:
                final[w] = t
                break
    dist = Counter(final.values())
    print(f"\n=== 最终「词 -> 规范类型」映射 ===")
    print(f"  总计 {len(final)} 词")
    for t in TYPES:
        print(f"    {t:<9} {dist.get(t, 0):>6}")

    # 一致性校验：与 ES gold 主类型的一致率
    agree = disagree = 0
    for w, t in final.items():
        c = stat.get(w)
        if c and sum(c.values()) >= 5:
            gt = c.most_common(1)[0][0]
            if gt == t:
                agree += 1
            else:
                disagree += 1
    if agree + disagree:
        print(f"\n  与 ES gold 主类型一致率(出现>=5次): {agree}/{agree+disagree} "
              f"= {100*agree/(agree+disagree):.1f}%")


if __name__ == "__main__":
    main()
