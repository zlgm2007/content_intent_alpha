"""意图模型量化评估（含「已见 / 未见」拆分）。

为什么必须拆分
--------------
`db.fetch_intent_samples()` 从 es_docs 随机取样，而意图训练集也是从同一张表构建的
（92,904 条）。随机取样必然混入大量训过的样本，指标会变成「记忆 + 泛化」的混合值。
--holdout 按「样本原文是否出现在 train.jsonl 中」拆成两组，只有「未见」那一组
才代表真实泛化能力。

为什么还要看分布
----------------
意图训练集做过类别平衡（1/3/5 类各截断到 20,000 条），与 es_docs 的真实分布
差异很大（真实分布里 3 分占 36.6%，训练集里只有 21.5%）。
这会导致模型的预测分布偏离真实分布，因此除准确率外必须输出：
  - 预测分布 vs 真值分布（看系统性偏移）
  - 逐类别 P/R/F1（看是哪一类被压/被抬）
  - 二分「是否有意图(>=3)」的 P/R/F1（业务最关心的一档）
  - 相邻容差准确率 / MAE（0-5 是有序分数，差 1 分和差 4 分后果完全不同）

用法：
    PYTHONPATH=. python scripts/eval_intent.py --limit 1000 --holdout
    PYTHONPATH=. python scripts/eval_intent.py --limit 500 --show 20   # 看错判样例
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import yaml  # noqa: E402

from src.common.compose import compose_note_comment  # noqa: E402
from src.intent.inference import IntentInference  # noqa: E402
from src.web import db  # noqa: E402

# 业务口径：LEVEL_MAP 中 0/1/2 = 无意图，3/4/5 = 有意图
HAS_INTENT_THRESHOLD = 3


def load_train_texts(path: str) -> set[str]:
    """读取意图训练集的原文集合，用于 --holdout 判定样本是否被训过。"""
    texts: set[str] = set()
    if not os.path.exists(path):
        print(f"  [警告] 训练集不存在: {path}（--holdout 将全部判为未见）")
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.add(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return texts


def compute_metrics(pairs: list[tuple[int, int]]) -> dict:
    """pairs = [(gold, pred), ...]"""
    if not pairs:
        return {}

    n = len(pairs)
    exact = sum(1 for g, p in pairs if g == p)
    within1 = sum(1 for g, p in pairs if abs(g - p) <= 1)
    mae = sum(abs(g - p) for g, p in pairs) / n

    # 二分：是否有意图
    tp = sum(1 for g, p in pairs if g >= HAS_INTENT_THRESHOLD and p >= HAS_INTENT_THRESHOLD)
    fp = sum(1 for g, p in pairs if g < HAS_INTENT_THRESHOLD and p >= HAS_INTENT_THRESHOLD)
    fn = sum(1 for g, p in pairs if g >= HAS_INTENT_THRESHOLD and p < HAS_INTENT_THRESHOLD)
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0

    # 逐类别
    per_class = {}
    for c in range(6):
        tpc = sum(1 for g, p in pairs if p == c and g == c)
        npc = sum(1 for g, p in pairs if p == c)
        ngc = sum(1 for g, p in pairs if g == c)
        pc = tpc / npc if npc else 0.0
        rc = tpc / ngc if ngc else 0.0
        per_class[c] = {
            "P": pc, "R": rc,
            "F1": 2 * pc * rc / (pc + rc) if pc + rc else 0.0,
            "n_pred": npc, "n_gold": ngc,
        }

    return {
        "n": n,
        "acc": exact / n,
        "acc_within1": within1 / n,
        "mae": mae,
        "bin_P": p_, "bin_R": r_, "bin_F1": f1,
        "bin_tp": tp, "bin_fp": fp, "bin_fn": fn,
        "per_class": per_class,
        "gold_dist": Counter(g for g, _ in pairs),
        "pred_dist": Counter(p for _, p in pairs),
        "confusion": Counter((g, p) for g, p in pairs),
    }


def report(tag: str, m: dict, show_cm: bool = False):
    if not m:
        print(f"\n===== {tag} =====\n  无有效样本")
        return
    print(f"\n===== {tag} =====")
    print(f"  样本={m['n']}")
    print(f"  准确率(精确匹配)={m['acc']:.4f}   相邻容差(±1)={m['acc_within1']:.4f}   MAE={m['mae']:.4f}")
    print(f"  [二分·是否有意图>={HAS_INTENT_THRESHOLD}] P={m['bin_P']:.4f} R={m['bin_R']:.4f} "
          f"F1={m['bin_F1']:.4f}  (TP={m['bin_tp']} FP={m['bin_fp']} FN={m['bin_fn']})")

    print(f"\n  {'分数':>4} {'真值数':>7} {'预测数':>7} {'P':>8} {'R':>8} {'F1':>8}   真值占比→预测占比")
    for c in range(6):
        pc = m["per_class"][c]
        gr = m["gold_dist"].get(c, 0) / m["n"] * 100
        pr = m["pred_dist"].get(c, 0) / m["n"] * 100
        flag = "  ⚠" if abs(gr - pr) >= 5 else ""
        print(f"  {c:>4} {pc['n_gold']:>7} {pc['n_pred']:>7} {pc['P']:>8.4f} {pc['R']:>8.4f} "
              f"{pc['F1']:>8.4f}   {gr:5.1f}% → {pr:5.1f}%{flag}")

    if show_cm:
        print("\n  混淆矩阵（行=真值，列=预测）")
        print(f"  {'':>6}" + "".join(f"{c:>7}" for c in range(6)))
        for g in range(6):
            row = "".join(f"{m['confusion'].get((g, p), 0):>7}" for p in range(6))
            print(f"  {g:>6}{row}")


def check_holdout_validity(train_texts: set[str]) -> None:
    """检查 holdout 拆分是否有效，发现「整类被训练集吃光」时报警。

    意图数据准备对 1/3/5 类做了截断（各 2 万），但 0/2/4 类数量本就不足 2 万，
    于是被 **100% 纳入训练**。后果：「未见」组里几乎只剩 1/3/5 三类，
    类别数从 6 降到 3，任务被人为简化 —— 此时「未见组准确率反而更高」
    是纯粹的选择偏差假象，绝非泛化能力更强。
    """
    if not train_texts:
        return
    import sqlite3

    from src.web.config import BASE_DIR

    conn = sqlite3.connect(os.path.join(BASE_DIR, "data/web.db"))
    total: Counter[int] = Counter()
    hit: Counter[int] = Counter()
    for title, desc, comment, score in conn.execute(
        "SELECT note_title, note_desc, comment_content, intent_score FROM es_docs"
    ):
        if score is None:
            continue
        t = compose_note_comment(title or "", desc or "", comment or "")
        total[int(score)] += 1
        if t in train_texts:
            hit[int(score)] += 1
    conn.close()

    print("\n  [holdout 有效性检查] 各分数进入训练集的比例：")
    bad = []
    for c in sorted(total):
        r = hit[c] / total[c] if total[c] else 0.0
        mark = ""
        if r >= 0.95:
            mark = "  ⚠ 该类几乎无留出样本"
            bad.append(c)
        print(f"    {c} 分: {hit[c]:>6,}/{total[c]:>6,} = {r * 100:5.1f}%{mark}")

    if bad:
        print(f"\n  ⚠⚠ 结论：{bad} 这些类别被训练集全部（或近乎全部）占用，"
              f"「未见」组缺失这些类别，任务被人为简化。")
        print("     此时对比「已见 vs 未见」得到的任何结论都是选择偏差，不可采信。")
        print("     要拿到可信的泛化指标，必须改用「随机划分后重训」。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/intent.yaml")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--holdout", action="store_true", help="按训练集是否见过拆分（强烈建议）")
    ap.add_argument("--train-file", default="data/intent/labeled/train.jsonl")
    ap.add_argument("--holdout-file", default=None,
                    help="直接评估留出集 jsonl（数据准备时预留，与训练集零重叠）。"
                         "推荐用这个，比 --holdout 更干净：后者按'是否出现在训练集'事后拆分，"
                         "会因类别平衡导致 0/2/4 类被 100% 吃光、留出集退化成 3 分类。")
    ap.add_argument("--cm", action="store_true", help="打印混淆矩阵")
    ap.add_argument("--show", type=int, default=0, help="打印 N 个错判样例")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 直接评估预留留出集：这是唯一能保证六类齐全、且与训练集零重叠的评估方式
    if args.holdout_file:
        import json as _json
        rows = [_json.loads(l) for l in open(args.holdout_file, "r", encoding="utf-8")]
        if args.limit:
            rows = rows[: args.limit]
        print(f"留出集 {len(rows):,} 条 -> {args.holdout_file}", flush=True)
        from collections import Counter as _C
        dist = _C(int(r["label"]) for r in rows)
        print("  分布:", dict(sorted(dist.items())), flush=True)
        missing = [c for c in range(6) if dist.get(c, 0) == 0]
        if missing:
            print(f"  ⚠️ 缺少分值 {missing}，六分类已退化，评估结论不可采信！", flush=True)
        engine = IntentInference(args.config, onnx_path=args.onnx)
        pairs = []
        for r in rows:
            pairs.append((int(r["label"]), int(engine.predict(r["text"])["score"])))
        report("留出集（真·未见过）", compute_metrics(pairs), show_cm=args.cm)
        return

    print(f"加载 {args.limit} 条真实样本…", flush=True)
    samples = db.fetch_intent_samples(args.limit)
    print(f"  实际取到 {len(samples)} 条", flush=True)

    engine = IntentInference(args.config, onnx_path=args.onnx)
    print("模型已加载", flush=True)

    train_texts = load_train_texts(args.train_file) if args.holdout else set()
    if args.holdout:
        print(f"  训练集 {len(train_texts):,} 条", flush=True)
        check_holdout_validity(train_texts)

    buckets: dict[str, list[tuple[int, int]]] = {"已见·训练集内": [], "未见·训练集外": []}
    wrong: list[tuple[str, int, int, str]] = []

    skipped = 0
    for s in samples:
        g = s.get("intent_score")
        if g is None:
            skipped += 1
            continue
        g = int(g)
        text = compose_note_comment(
            s.get("note_title") or "", s.get("note_desc") or "", s.get("comment_content") or ""
        )
        pred = engine.predict(text)
        p = int(pred["score"])
        key = "已见·训练集内" if (args.holdout and text in train_texts) else "未见·训练集外"
        buckets[key].append((g, p))
        if args.show and g != p and len(wrong) < args.show:
            wrong.append((key, g, p, text[:90]))

    if skipped:
        print(f"  跳过 {skipped} 条无 intent_score 的样本", flush=True)

    for key in ("已见·训练集内", "未见·训练集外"):
        if args.holdout:
            report(f"{key}({len(buckets[key])})", compute_metrics(buckets[key]), show_cm=args.cm)
        else:
            all_pairs = buckets["已见·训练集内"] + buckets["未见·训练集外"]
            report("全部样本", compute_metrics(all_pairs), show_cm=args.cm)
            break

    if args.holdout and buckets["已见·训练集内"] and buckets["未见·训练集外"]:
        m1 = compute_metrics(buckets["已见·训练集内"])
        m2 = compute_metrics(buckets["未见·训练集外"])
        print("\n===== 泛化落差（已见 → 未见）=====")
        for k, label in (("acc", "准确率"), ("acc_within1", "±1容差准确率"),
                         ("bin_F1", "二分F1(有意图)")):
            d = m2[k] - m1[k]
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
            print(f"  {label:14s}: {m1[k]:.4f} → {m2[k]:.4f}  {arrow}{abs(d):.4f} "
                  f"({d / m1[k] * 100:+.1f}%)" if m1[k] else f"  {label}: n/a")
        print(f"  MAE            : {m1['mae']:.4f} → {m2['mae']:.4f}（越低越好）")

    if wrong:
        print(f"\n----- 错判样例（前 {len(wrong)} 条）-----")
        for key, g, p, t in wrong:
            print(f"  [{key}] 真值={g} 预测={p}  文本={t!r}")


if __name__ == "__main__":
    main()
