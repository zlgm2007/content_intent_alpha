"""用人工金标准评测集算出「不受脏标签污染」的真实指标。

配套 build_gold_annotation_set.py 使用：先人工标注，再用本脚本评测。

为什么不能直接拿留出集算
------------------------
留出集的真值是 ES 外部程序的打分，而这套打分已被证实有系统性错误（低分档 18.3%
含明确购买信号却被判无意向）。在它上面算出的 0.7196，既不是模型的真实水平，
也无法区分「模型错」和「标签错」。

人工标注之后，可以同时得到三个数字，把这件事彻底拆开：

  1. 标签一致率  = 人工 vs 外部标签     → 外部标签有多脏
  2. 真实准确率  = 模型 vs 人工         → 模型真实水平（★ 这才是要的数）
  3. 脏标签准确率 = 模型 vs 外部标签    → 之前一直在看的 0.7196

关键设计
--------
**P0 层是分层等量抽样（每档 42 条），不是按比例抽样**，所以直接算平均会高估小类、
低估大类，得到的整体准确率是有偏的。必须按留出集的真实分布做后分层加权还原：

    acc_weighted = Σ_c (N_c / N_total) × acc_c

P1 层（高置信分歧样本）是**非随机**筛选出来的（专门挑模型和标签打架的），
所以它上面的准确率**没有代表性**，只能用来回答「分歧时谁对」：
    模型对了多少 / 外部标签对了多少 / 两个都错多少
绝不能拿 P1 层的准确率当成模型的整体水平。

用法
----
    /opt/miniconda3/bin/python3 scripts/eval_on_gold.py
    /opt/miniconda3/bin/python3 scripts/eval_on_gold.py --file 路径.xlsx
    /opt/miniconda3/bin/python3 scripts/eval_on_gold.py --cm        # 打印混淆矩阵
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
from openpyxl import load_workbook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOLD = os.path.join("data", "intent", "labeled", "gold_annotation.xlsx")
HOLDOUT = os.path.join("data", "intent", "labeled", "holdout.jsonl")

# 需与 build_gold_annotation_set.py 保持一致；P1 层是专挑两者相差 >= 该值的样本
P1_MIN_GAP = 2

# 留出集的真实分布，用于后分层加权还原整体准确率
TRUE_DIST = None  # 运行时从留出集读

HEADER_GOLD = "★你的打分(0-5)"


def load_true_dist():
    """读留出集真实分布。P0 层是分层等量抽样，必须加权还原。"""
    if not os.path.exists(HOLDOUT):
        return None
    c = collections.Counter()
    for line in open(HOLDOUT, encoding="utf-8"):
        c[json.loads(line)["label"]] += 1
    return c


def read_gold(path: str):
    """读回已填写的行。返回 [(优先级, 外部标签, 模型预测, 人工打分, 备注)]"""
    wb = load_workbook(path, data_only=True)
    if "待标注" not in wb.sheetnames:
        raise SystemExit(f"{path} 里找不到「待标注」sheet")
    ws = wb["待标注"]
    head = [c.value for c in ws[1]]
    try:
        i_pri = head.index("优先级")
        i_ext = head.index("外部标签")
        i_pred = head.index("模型预测")
        i_gold = head.index(HEADER_GOLD)
        i_cmt = head.index("评论原文（★打分看这列）")
        i_note = head.index("备注")
    except ValueError as e:
        raise SystemExit(f"表头缺列: {e}\n实际表头: {head}")

    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        g = r[i_gold]
        if g is None or str(g).strip() == "":
            continue
        try:
            g = int(float(g))
        except (TypeError, ValueError):
            continue
        if not 0 <= g <= 5:
            continue
        rows.append({
            "pri": r[i_pri], "ext": int(r[i_ext]), "pred": int(r[i_pred]),
            "gold": g, "comment": r[i_cmt] or "", "note": r[i_note] or "",
        })
    return rows


def acc(a, b):
    return float(np.mean(np.array(a) == np.array(b))) if a else float("nan")


def with_ci(p, n):
    """二项分布 95% 置信区间（正态近似）。"""
    if n == 0:
        return "n/a"
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return f"{p:.4f} ±{1.96 * se:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=GOLD)
    ap.add_argument("--cm", action="store_true", help="打印混淆矩阵")
    ap.add_argument("--show", type=int, default=0, help="展示 N 条三方不一致的样本")
    args = ap.parse_args()

    rows = read_gold(args.file)
    if not rows:
        print(f"{args.file} 里还没有填任何打分。")
        print(f"请打开表格，填「{HEADER_GOLD}」这一列后再运行。")
        return
    print(f"已标注 {len(rows)} 条\n")

    dist = load_true_dist()
    total = sum(dist.values()) if dist else 0

    for pri, label in (("P0", "P0 层（分层等量，可无偏估计）"), ("P1", "P1 层（高置信分歧，仅用于诊断）")):
        sub = [r for r in rows if r["pri"] == pri]
        if not sub:
            continue
        print("=" * 68)
        print(f"{label}  已标 {len(sub)} 条")
        print("=" * 68)

        ext = [r["ext"] for r in sub]
        pred = [r["pred"] for r in sub]
        gold = [r["gold"] for r in sub]
        n = len(sub)

        print(f"  外部标签 vs 人工   一致率 = {acc(ext, gold):.4f}   << 外部标签有多脏")
        print(f"  模型     vs 人工   准确率 = {with_ci(acc(pred, gold), n)}   << 模型真实水平")
        if pri == "P1":
            # 这一层是专挑两者相差 >= P1_MIN_GAP 的样本，两者必然不相等，恒为 0
            print(f"  模型     vs 外部   准确率 = {acc(pred, ext):.4f}   "
                  f"（恒为 0：本层专挑两者相差≥{P1_MIN_GAP}分的样本，无参考价值）")
        else:
            print(f"  模型     vs 外部   准确率 = {acc(pred, ext):.4f}   << 之前一直在看的数")
        print(f"  MAE(模型,人工) = {np.mean(np.abs(np.array(pred) - np.array(gold))):.4f}"
              f"   ±1容差 = {np.mean(np.abs(np.array(pred) - np.array(gold)) <= 1):.4f}")

        if pri == "P0":
            # 后分层加权还原
            wsum, acc_w, mae_w = 0.0, 0.0, 0.0
            per_class = {}
            for c in range(6):
                s = [r for r in sub if r["ext"] == c]
                if not s:
                    continue
                a = acc([r["pred"] for r in s], [r["gold"] for r in s])
                per_class[c] = (a, len(s))
                if dist:
                    w = dist[c] / total
                    acc_w += w * a
                    mae_w += w * float(np.mean(np.abs(
                        np.array([r["pred"] for r in s]) - np.array([r["gold"] for r in s]))))
                    wsum += w
            print("\n  【按真实分布加权还原的整体准确率】")
            print(f"    整体准确率 ≈ {acc_w:.4f}   MAE ≈ {mae_w:.4f}   (覆盖权重 {wsum:.3f})")
            print(f"    对比：在脏标签上算出的六分类准确率 0.7196")
            print("\n  分档准确率（P0 层内，未加权）:")
            for c, (a, m) in sorted(per_class.items()):
                print(f"    {c} 分: {a:.4f}  ({m} 条)")

            # 二分口径
            b = [1 if v >= 3 else 0 for v in gold]
            p = [1 if v >= 3 else 0 for v in pred]
            e = [1 if v >= 3 else 0 for v in ext]
            tp = sum(1 for x, z in zip(p, b) if x == 1 and z == 1)
            fp = sum(1 for x, z in zip(p, b) if x == 1 and z == 0)
            fn = sum(1 for x, z in zip(p, b) if x == 0 and z == 1)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            print(f"\n  二分（有无意向，以人工为准）:")
            print(f"    模型   P/R/F1 = {tp/(tp+fp) if tp+fp else 0:.4f} / "
                  f"{tp/(tp+fn) if tp+fn else 0:.4f} / {f1:.4f}   (TP={tp} FP={fp} FN={fn})")
            tp2 = sum(1 for x, z in zip(e, b) if x == 1 and z == 1)
            fp2 = sum(1 for x, z in zip(e, b) if x == 1 and z == 0)
            fn2 = sum(1 for x, z in zip(e, b) if x == 0 and z == 1)
            f2 = 2 * tp2 / (2 * tp2 + fp2 + fn2) if (2 * tp2 + fp2 + fn2) else 0.0
            print(f"    外部标签 F1 = {f2:.4f}   (TP={tp2} FP={fp2} FN={fn2})")
        else:
            # P1 层：回答「分歧时谁对」
            m_ok = sum(1 for r in sub if r["pred"] == r["gold"])
            e_ok = sum(1 for r in sub if r["ext"] == r["gold"])
            neither = sum(1 for r in sub if r["pred"] != r["gold"] and r["ext"] != r["gold"])
            n = len(sub)
            print("\n  【分歧时谁对】(模型 vs 外部标签，以人工为准)")
            print(f"    模型对          {m_ok:>4} ({m_ok/n*100:>5.1f}%)")
            print(f"    外部标签对      {e_ok:>4} ({e_ok/n*100:>5.1f}%)")
            print(f"    两个都错        {neither:>4} ({neither/n*100:>5.1f}%)")
            if m_ok > e_ok:
                print(f"    >> 模型比外部标签更准，比值 {m_ok/max(e_ok,1):.2f} : 1")
            elif e_ok > m_ok:
                print(f"    >> 外部标签比模型更准，比值 {e_ok/max(m_ok,1):.2f} : 1")
            else:
                print("    >> 两者持平")
        print()

    if args.cm:
        sub = [r for r in rows if r["pri"] == "P0"] or rows
        gold = [r["gold"] for r in sub]
        pred = [r["pred"] for r in sub]
        print("【混淆矩阵（P0 层，模型预测 vs 人工标注）】")
        cm = np.zeros((6, 6), int)
        for g, p in zip(gold, pred):
            cm[g, p] += 1
        print("真值\\预测" + "".join(f"{i:>7}" for i in range(6)) + "    召回")
        for i in range(6):
            t = cm[i].sum()
            r = f"{cm[i,i]/t:.3f}" if t else "  -  "
            print(f"     {i}   " + "".join(f"{v:>7}" for v in cm[i]) + f"   {r}")
        print("精准率   " + "".join(
            f"{(cm[i,i]/cm[:,i].sum()):>7.3f}" if cm[:, i].sum() else "      -"
            for i in range(6)))

    if args.show:
        odd = [r for r in rows
               if len({r["gold"], r["ext"], r["pred"]}) == 3][: args.show]
        print(f"\n【三方互不相同的样本 {len(odd)} 条】")
        for r in odd:
            print(f"  人工{r['gold']} 外部{r['ext']} 模型{r['pred']}  {str(r['comment'])[:70]}")


if __name__ == "__main__":
    main()
