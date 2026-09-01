"""意图分数先验偏移修正（离线搜索强度，不重训、不改模型）。

问题
----
意图训练集做过类别平衡（1/3/5 类各截断到 20,000 条），但 es_docs 的真实分布
差异很大：

    分数   训练集占比   真实占比
     0      8.2%        5.1%
     1     21.5%       14.5%
     2     17.7%       10.9%
     3     21.5%       36.6%   <- 真实最多，训练里被压到 1/3
     4      9.5%        5.8%
     5     21.5%       27.1%

后果：模型预测分布被拉向被过度代表的 1/2/4 类，而真实占 36.6% 的 3 分被低估
（实测预测占比仅 32.4%）。这是典型的 **prior shift（先验偏移）**。

方法
----
对 logits 加一个与类别先验相关的偏置（Saerens-Latinne-Decaestecker 的简化版）：

    adjusted_logits[c] = logits[c] + alpha * log(P_real[c] / P_train[c])

alpha 为强度：0 = 不修正，1 = 完全修正。扫一遍 alpha 找最优。
只需缓存一次 logits，扫描是纯计算，很快。

为什么不用「迭代式 SLR」
------------------------
SLR 会迭代到后验分布完全匹配目标先验，但它假设「条件概率 P(x|y) 不变」，
在类别被非随机截断（1/3/5 是随机截断，0/2/4 是全量）时这个假设部分失效。
所以这里用可调 alpha 的简化版，用数据决定修多少。

用法：
    PYTHONPATH=. python scripts/tune_intent_prior.py --limit 1500
    PYTHONPATH=. python scripts/tune_intent_prior.py --limit 1500 --holdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.common.compose import compose_note_comment  # noqa: E402
from src.common.onnx_utils import OnnxRunner  # noqa: E402
from src.web import db  # noqa: E402

HAS_INTENT = 3
NUM_CLASSES = 6


def train_prior(path: str) -> np.ndarray:
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c[json.loads(line)["label"]] += 1
    tot = sum(c.values())
    return np.array([c[i] / tot for i in range(NUM_CLASSES)], dtype=np.float64)


def real_prior() -> np.ndarray:
    c = Counter()
    for (s,) in db.get_conn().execute("SELECT intent_score FROM es_docs"):
        if s is not None:
            c[int(s)] += 1
    tot = sum(c.values())
    return np.array([c[i] / tot for i in range(NUM_CLASSES)], dtype=np.float64)


def metrics(golds: list[int], preds: list[int]) -> dict:
    n = len(golds)
    if not n:
        return {}
    mae = sum(abs(g - p) for g, p in zip(golds, preds)) / n
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / n
    acc1 = sum(1 for g, p in zip(golds, preds) if abs(g - p) <= 1) / n
    tp = sum(1 for g, p in zip(golds, preds) if g >= HAS_INTENT and p >= HAS_INTENT)
    fp = sum(1 for g, p in zip(golds, preds) if g < HAS_INTENT and p >= HAS_INTENT)
    fn = sum(1 for g, p in zip(golds, preds) if g >= HAS_INTENT and p < HAS_INTENT)
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    # 严重错判：真值无意图(0-1)却判成强烈意图(5)，或反之
    severe = sum(1 for g, p in zip(golds, preds) if abs(g - p) >= 4)
    return {
        "acc": acc, "acc1": acc1, "mae": mae,
        "P": p_, "R": r_, "F1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
        "severe": severe,
    }


def sensitivity(L: np.ndarray, golds: np.ndarray, bias: np.ndarray, n: int = 1200, rounds: int = 20):
    """真实业务先验未知时的风险分析。

    背景：es_docs 由两个索引拼成，且**索引按意图分数完全切分**——
      lead_clue_score_ge3 (105,181 条) 只有 3/4/5 分
      lead_clue_score_lt3 ( 45,969 条) 只有 0/1/2 分
    所以 es_docs 里「有意图」占 69.6% 是**导入比例决定的**，不一定是真实业务分布。
    线上若跑的是这两个索引，它成立；若跑新内容，先验未知。

    做法：固定各组内部的分值构成，只改变「有意图组 : 无意图组」的混合比例，
    重采样出不同的假设业务分布，看 alpha=1.0 相对 alpha=0 是赢是亏。
    """
    pos_idx = np.where(golds >= HAS_INTENT)[0]   # 3/4/5
    neg_idx = np.where(golds < HAS_INTENT)[0]    # 0/1/2
    if len(pos_idx) < 50 or len(neg_idx) < 50:
        print("样本不足以做敏感性分析")
        return
    rng = np.random.default_rng(42)

    cur_pos = len(pos_idx) / (len(pos_idx) + len(neg_idx))
    print(f"\nes_docs 的有意图占比 = {cur_pos * 100:.1f}（导入比例产物，非自然分布）")
    print("\n假设线上「有意图」占比不同时，alpha=1.0 相对不修正的得失：")
    print(f"  {'线上有意图占比':>14}{'准确率(0→1)':>22}{'MAE(0→1)':>20}"
          f"{'二分F1(0→1)':>22}{'结论':>8}")

    rows = []
    for ratio in [0.30, 0.40, 0.50, 0.60, 0.696, 0.80, 0.90]:
        accs = {0.0: [], 1.0: []}
        maes = {0.0: [], 1.0: []}
        f1s = {0.0: [], 1.0: []}
        n_pos = int(n * ratio)
        n_neg = n - n_pos
        for _ in range(rounds):
            sel = np.concatenate([
                rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=len(pos_idx) < n_pos),
                rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=len(neg_idx) < n_neg),
            ])
            g = list(golds[sel])
            for a in (0.0, 1.0):
                p = list((L[sel] + a * bias).argmax(axis=1))
                m = metrics(g, p)
                accs[a].append(m["acc"])
                maes[a].append(m["mae"])
                f1s[a].append(m["F1"])
        a0 = (np.mean(accs[0.0]), np.mean(maes[0.0]), np.mean(f1s[0.0]))
        a1 = (np.mean(accs[1.0]), np.mean(maes[1.0]), np.mean(f1s[1.0]))
        verdict = "修正更优" if a1[2] > a0[2] else "修正有害"
        print(f"  {ratio * 100:>13.1f}%"
              f"{a0[0]:>10.4f}→{a1[0]:<10.4f}"
              f"{a0[1]:>9.4f}→{a1[1]:<9.4f}"
              f"{a0[2]:>10.4f}→{a1[2]:<10.4f}{verdict:>8}")
        rows.append((ratio, a0, a1))

    win = [r[0] for r in rows if r[2][2] > r[1][2]]
    if win:
        print(f"\n  => alpha=1.0 在「有意图占比 ≥ {min(win) * 100:.0f}%」时优于不修正。")
        print(f"     若线上真实占比低于这个值，应保持 alpha 较小或关闭修正。")
    else:
        print("\n  => 在测试的所有比例下 alpha=1.0 都不优于不修正，建议关闭。")
    print("     注：修正的本质是「把预测拉向多数类」，业务先验越极端收益越大，反之有害。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/intent.yaml")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--train-file", default="data/intent/labeled/train.jsonl")
    ap.add_argument("--holdout", action="store_true", help="额外报告「未见」子集上的效果")
    ap.add_argument("--save", action="store_true",
                    help="把偏置写进配置的 inference.prior_file（推理端据此修正）")
    ap.add_argument("--sensitivity", action="store_true",
                    help="敏感性分析：真实业务先验未知时，alpha=1.0 在什么范围内是安全的")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pt, pr = train_prior(args.train_file), real_prior()

    if args.save:
        icfg = cfg.get("inference") or {}
        out = icfg.get("prior_file") or "configs/intent_prior.json"
        bias_save = np.log(np.where(pt > 0, pr, 1e-9) / np.where(pt > 0, pt, 1.0))
        bias_save = np.where(pt > 0, bias_save, 0.0)
        payload = {
            "bias": [round(float(x), 6) for x in bias_save],
            "train_prior": [round(float(x), 6) for x in pt],
            "real_prior": [round(float(x), 6) for x in pr],
            "note": "log(P_real/P_train)，由 scripts/tune_intent_prior.py 生成；"
                    "配合 configs/intent.yaml 的 inference.prior_correction_alpha 使用",
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"偏置已写入 {out}: {payload['bias']}")
        return
    print("先验分布对比（训练集 vs 真实）：")
    print(f"  {'分数':>4}{'训练集':>10}{'真实':>10}{'比值':>9}")
    for i in range(NUM_CLASSES):
        ratio = pr[i] / pt[i] if pt[i] else 0
        print(f"  {i:>4}{pt[i] * 100:>9.1f}%{pr[i] * 100:>9.1f}%{ratio:>9.2f}")

    bias = np.log(np.where(pt > 0, pr, 1e-9) / np.where(pt > 0, pt, 1.0))
    bias = np.where(pt > 0, bias, 0.0)
    print("\n修正偏置 log(P_real/P_train):", np.round(bias, 4).tolist())

    print(f"\n加载 {args.limit} 条样本并缓存 logits…", flush=True)
    samples = db.fetch_intent_samples(args.limit)
    tok = AutoTokenizer.from_pretrained(cfg["train"]["output_dir"])
    runner = OnnxRunner(cfg["export"]["onnx_path"])

    golds, logits_all, unseen_flags = [], [], []
    train_texts = set()
    if args.holdout:
        with open(args.train_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    train_texts.add(json.loads(line)["text"])

    for s in samples:
        g = s.get("intent_score")
        if g is None:
            continue
        text = compose_note_comment(
            s.get("note_title") or "", s.get("note_desc") or "", s.get("comment_content") or ""
        )
        enc = tok(text, max_length=cfg["model"]["max_length"], padding="max_length",
                  truncation=True, return_tensors="np")
        logits_all.append(runner.run(enc["input_ids"], enc["attention_mask"])[0])
        golds.append(int(g))
        unseen_flags.append(text not in train_texts)

    L = np.array(logits_all, dtype=np.float64)
    golds_arr = np.array(golds)
    print(f"  有效样本 {len(golds)} 条", flush=True)

    if args.sensitivity:
        sensitivity(L, golds_arr, bias)
        return

    idx_unseen = np.array(unseen_flags)
    subsets = [("全部", np.ones(len(golds), dtype=bool))]
    if args.holdout and idx_unseen.any():
        subsets.append((f"未见({int(idx_unseen.sum())})", idx_unseen))

    for name, mask in subsets:
        print(f"\n{'=' * 78}\n子集：{name}（{int(mask.sum())} 条）")
        print(f"  {'alpha':>6}{'准确率':>9}{'±1':>9}{'MAE':>9}"
              f"{'二分P':>9}{'二分R':>9}{'二分F1':>9}{'严重错判':>9}")
        best_alpha, best_f1 = 0.0, -1.0
        rows = []
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
            preds = (L + alpha * bias).argmax(axis=1)
            m = metrics(list(golds_arr[mask]), list(preds[mask]))
            rows.append((alpha, m))
            print(f"  {alpha:>6.1f}{m['acc']:>9.4f}{m['acc1']:>9.4f}{m['mae']:>9.4f}"
                  f"{m['P']:>9.4f}{m['R']:>9.4f}{m['F1']:>9.4f}{m['severe']:>9}")
            if m["F1"] > best_f1:
                best_f1, best_alpha = m["F1"], alpha
        base = rows[0][1]
        bm = [r for r in rows if r[0] == best_alpha][0][1]
        print(f"\n  最优 alpha = {best_alpha}（按二分F1）")
        print(f"    准确率 {base['acc']:.4f} → {bm['acc']:.4f} ({(bm['acc'] - base['acc']) * 100:+.2f}pp)")
        print(f"    MAE    {base['mae']:.4f} → {bm['mae']:.4f} ({bm['mae'] - base['mae']:+.4f}，越低越好)")
        print(f"    二分F1 {base['F1']:.4f} → {bm['F1']:.4f} ({(bm['F1'] - base['F1']) * 100:+.2f}pp)")
        print(f"    严重错判 {base['severe']} → {bm['severe']}")

    print("\n注：alpha=0 即当前线上行为。若最优 alpha=0，说明先验偏移修正无效，不应启用。")


if __name__ == "__main__":
    main()
