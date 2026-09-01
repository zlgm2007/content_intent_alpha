"""ES 标签冲突根因分析（只读，绝不写 ES / 不写 web.db）。

背景
----
评估时发现「同一条文本在 ES 里被打成不同分数」，重复组冲突率 41.7%。
一度怀疑是外部程序标注质量差，本脚本用于把冲突拆开，看它到底是什么。

结论（2026-09-01 实测，151,121 条评论非空样本）
----------------------------------------------
41.7% 的冲突率其实是两种性质完全不同的东西混在一起：

  1. 跨索引对立（结构性，1,960 组，冲突率 100%）
     `lead_clue_score_ge3` 只含 3/4/5 分，`lead_clue_score_lt3` 只含 0/1/2 分，
     两个索引分数区间【完全互斥、零重叠】。同一条文本只要同时落进两个索引，
     标签必然对立。这不是噪声，是数据整合问题。

  2. 同索引内噪声（真实标注噪声，3,350 组，冲突率 7.6%，即 255 组）
     这才是外部程序自身的标注不一致。

为什么不能靠「取 update_time 最新的那条」修复
--------------------------------------------
实测跨索引组里 ge3 更新占 51.3%、lt3 更新占 48.7%，几乎五五开，
中位时间差仅 0.1 天，两索引时间范围完全相同（2026-07-10 ~ 2026-08-31）。
说明外部程序是【并行写入两套判定】，不是「新版本覆盖旧版本」。
没有时间方向性，取最新等于随机挑。

去重时的系统性偏差（重要）
--------------------------
`db.fetch_intent_samples` 的 SQL 无 ORDER BY，返回 DB 自然顺序；
ge3 索引先导入（id 更小）→ 去重时 ge3 记录先占位 → lt3 的低分记录被当重复丢弃。
实测 1,960 组跨索引样本，保留方向 **100% 是 high(3-5)**，零例外。
即：外部程序判定为「低意向」的样本，被我们按「高意向」训练了。

影响规模：训练集 991 条（1.21%）、留出集 198 条（1.42%）。
即使全部剔除，留出集准确率上限变动仅 ±1.42pp —— 收益有限，未单独为此重训。

用法
----
    python scripts/analyze_es_label_conflict.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.web.data_prep_service import compose_note_comment  # noqa: E402

GE3 = "lead_clue_score_ge3"
LT3 = "lead_clue_score_lt3"


def _load_rows(db_path: Path):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, source_index, note_title, note_desc, comment_content, "
        "intent_score, update_time, import_batch FROM es_docs "
        "WHERE comment_content IS NOT NULL AND comment_content != ''"
    ).fetchall()
    con.close()
    return rows


def _key(r) -> str:
    return compose_note_comment(r["note_title"], r["note_desc"], r["comment_content"])


def _ts(r):
    try:
        return int(r["update_time"])
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="ES 标签冲突根因分析（只读）")
    ap.add_argument("--db", default=str(ROOT / "data" / "web.db"))
    args = ap.parse_args()

    rows = _load_rows(Path(args.db))
    print(f"评论非空样本: {len(rows)}")

    # ---- Q1 各索引分数分布 ----
    print("\n" + "=" * 70)
    print("【Q1】各索引的 intent_score 分布")
    print("=" * 70)
    for idx in (GE3, LT3):
        dist = Counter(
            r["intent_score"] for r in rows if r["source_index"] == idx
        )
        tot = sum(dist.values())
        print(f"\n{idx}  合计 {tot}")
        for s in range(6):
            c = dist.get(s, 0)
            bar = "#" * int(40 * c / tot) if tot else ""
            print(f"   分数 {s}: {c:7d}  {100*c/tot:5.1f}%  {bar}")

    # ---- 分组 ----
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        k = _key(r)
        if k:
            groups[k].append(r)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n去重后文本数: {len(groups)}")
    print(f"重复文本组: {len(dup)}  涉及样本: {sum(len(v) for v in dup.values())}")

    cross = {k: v for k, v in dup.items() if len({r["source_index"] for r in v}) > 1}
    within = {k: v for k, v in dup.items() if len({r["source_index"] for r in v}) == 1}

    # ---- Q2/Q3 冲突分解 ----
    print("\n" + "=" * 70)
    print("【Q2/Q3】冲突分解：跨索引 vs 同索引")
    print("=" * 70)
    for name, gs in (("跨索引(ge3+lt3)", cross), ("同索引内", within)):
        conf = sum(
            1
            for v in gs.values()
            if len({r["intent_score"] for r in v if r["intent_score"] is not None}) > 1
        )
        tot = len(gs)
        rate = 100 * conf / tot if tot else 0.0
        print(f"  {name:18s}: 共 {tot:5d}  冲突 {conf:5d}  冲突率 {rate:5.1f}%")

    # ---- Q4 分数配对 ----
    print("\n" + "=" * 70)
    print("【Q4】跨索引组的分数配对 (ge3分, lt3分)")
    print("=" * 70)
    pairs: Counter = Counter()
    for v in cross.values():
        ge = [
            r["intent_score"]
            for r in v
            if r["source_index"] == GE3 and r["intent_score"] is not None
        ]
        lt = [
            r["intent_score"]
            for r in v
            if r["source_index"] == LT3 and r["intent_score"] is not None
        ]
        if ge and lt:
            pairs[(ge[0], lt[0])] += 1
    for (g, l), c in sorted(pairs.items(), key=lambda x: -x[1])[:12]:
        print(f"  ge3={g}  lt3={l}   {c:5d} 组")
    print(f"  共 {len(pairs)} 种配对，合计 {sum(pairs.values())} 组")

    # ---- Q5 时间方向性 ----
    print("\n" + "=" * 70)
    print("【Q5】跨索引组里哪边 update_time 更新？（能否靠'取最新'修复）")
    print("=" * 70)
    ge_newer = lt_newer = equal = skipped = 0
    diffs = []
    for v in cross.values():
        ge = [x for x in (_ts(r) for r in v if r["source_index"] == GE3) if x]
        lt = [x for x in (_ts(r) for r in v if r["source_index"] == LT3) if x]
        if not ge or not lt:
            skipped += 1
            continue
        tg, tl = max(ge), max(lt)
        diffs.append(tl - tg)
        if tg > tl:
            ge_newer += 1
        elif tl > tg:
            lt_newer += 1
        else:
            equal += 1
    tot = ge_newer + lt_newer + equal
    if tot:
        print(f"  可比较: {tot}  (缺时间戳跳过: {skipped})")
        print(f"    ge3(高分段) 更新: {ge_newer:5d}  ({100*ge_newer/tot:5.1f}%)")
        print(f"    lt3(低分段) 更新: {lt_newer:5d}  ({100*lt_newer/tot:5.1f}%)")
        print(f"    完全相同:        {equal:5d}")
    pos = [d for d in diffs if d > 0]
    neg = [d for d in diffs if d < 0]
    if pos:
        print(f"  lt3 比 ge3 晚: {len(pos)} 组  中位晚 {sorted(pos)[len(pos)//2]/86400000:.1f} 天")
    if neg:
        print(f"  ge3 比 lt3 晚: {len(neg)} 组  中位晚 {sorted(neg)[len(neg)//2]/86400000:.1f} 天")
    if pos and neg and abs(len(pos) - len(neg)) / len(diffs) < 0.15:
        print("  >> 两边几乎五五开：无时间方向性，【不能】靠取最新修复")

    # ---- Q6 去重时的保留方向（系统性偏差）----
    print("\n" + "=" * 70)
    print("【Q6】跨索引样本去重后保留成了哪一边？（系统性偏差）")
    print("=" * 70)
    train_p = ROOT / "data" / "intent" / "labeled" / "train.jsonl"
    hold_p = ROOT / "data" / "intent" / "labeled" / "holdout.jsonl"
    if not (train_p.exists() and hold_p.exists()):
        print("  (缺少 train.jsonl / holdout.jsonl，跳过)")
        return 0

    tr, hd = {}, {}
    with train_p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tr[d["text"]] = d["label"]
    with hold_p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            hd[d["text"]] = d["label"]

    hit_tr = hit_hd = miss = 0
    side_tr: Counter = Counter()
    side_hd: Counter = Counter()
    for k in cross:
        if k in tr:
            hit_tr += 1
            side_tr["high(3-5)" if tr[k] >= 3 else "low(0-2)"] += 1
        elif k in hd:
            hit_hd += 1
            side_hd["high(3-5)" if hd[k] >= 3 else "low(0-2)"] += 1
        else:
            miss += 1

    print(f"  命中训练集: {hit_tr:5d}  ({100*hit_tr/len(tr):.2f}% of 训练集)  保留方向 {dict(side_tr)}")
    print(f"  命中留出集: {hit_hd:5d}  ({100*hit_hd/len(hd):.2f}% of 留出集)  保留方向 {dict(side_hd)}")
    print(f"  两边都没有: {miss}")
    if side_tr and len(side_tr) == 1:
        print("  >> 保留方向 100% 偏向一侧：外部判定为低意向的样本被按高意向训练")
    print(f"\n  即使全部剔除，留出集准确率上限变动 ±{100*hit_hd/len(hd):.2f}pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
