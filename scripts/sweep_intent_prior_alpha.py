"""意图先验修正 alpha 扫描：一次算 logits，离线扫所有 alpha。

为什么需要这个脚本
------------------
训练集做过类别平衡（1/3/5 各截断到 20,000），与真实分布不一致，
模型系统性低估 3/5 分（实测 3 分预测占比 35.6% vs 真值 39.0%，
二分 FN=1321 远大于 FP=852）。configs/intent.yaml 里有现成的修正机制
`logits += alpha * log(P_real/P_train)`，但 alpha 默认 0.0（关闭）。

scripts/tune_intent_prior.py 只做「不同有意图占比」的敏感性分析，没有 alpha 扫描。
若直接改配置再跑评测，每个 alpha 要重跑一遍全量推理（约 4 分钟），扫 10 个值就是 40 分钟。
这里把 logits 先算完缓存到 npz，之后扫任意多组 alpha 都是毫秒级。

方法论
------
**在留出集上调 alpha 属于用测试数据调参，数字会偏乐观。**
所以默认把留出集按奇偶一分为二：
  - tune 半区：用来选 alpha
  - eval  半区：用选出的 alpha 报告真实效果（未参与选择）
用 --no-split 可关闭（仅用于观察趋势，不要据此定参数）。

缓存：logits 存在 /tmp/intent_logits_cache.npz，以 (留出集路径 + 模型 mtime) 为键，
       输入没变就直接用。改了模型会自动失效重算。

用法：
    PYTHONPATH=. python scripts/sweep_intent_prior_alpha.py
    PYTHONPATH=. python scripts/sweep_intent_prior_alpha.py --no-split   # 只看趋势
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.intent.inference import IntentInference  # noqa: E402

CACHE = "/tmp/intent_logits_cache.npz"
HOLDOUT = "data/intent/labeled/holdout.jsonl"
PRIOR_FILE = "configs/intent_prior.json"
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def load_logits(rows, cfg="configs/intent.yaml", batch=64):
    """算全量 logits，带缓存。返回 (N, 6) float64。"""
    onnx = "models/intent/intent.onnx"
    key = f"{HOLDOUT}|{os.path.getmtime(onnx)}|{len(rows)}"
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if str(z["key"]) == key:
            print(f"复用缓存 logits {z['logits'].shape} -> {CACHE}")
            return z["logits"]
    print(f"计算 logits（{len(rows)} 条，约需数分钟）…", flush=True)

    inf = IntentInference(cfg)
    enc_all = []
    for i in range(0, len(rows), batch):
        chunk = [r["text"] for r in rows[i : i + batch]]
        enc = inf.tokenizer(
            chunk, max_length=inf.max_length, padding="max_length",
            truncation=True, return_tensors="np",
        )
        enc_all.append(enc)
        if (i // batch) % 20 == 0:
            print(f"  tokenize {i}/{len(rows)}", flush=True)

    outs = []
    done = 0
    for enc in enc_all:
        lg = inf.runner.run(enc["input_ids"], enc["attention_mask"])
        outs.append(np.asarray(lg, dtype=np.float64))
        done += len(enc["input_ids"])
        if done % (batch * 20) < batch:
            print(f"  推理 {done}/{len(rows)}", flush=True)

    logits = np.concatenate(outs, axis=0)
    np.savez(CACHE, key=key, logits=logits)
    print(f"已缓存 -> {CACHE}")
    return logits


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float((y_true == y_pred).mean())
    tol = float((np.abs(y_true - y_pred) <= 1).mean())
    mae = float(np.abs(y_true - y_pred).mean())
    hi_t, hi_p = y_true >= 3, y_pred >= 3
    tp = int((hi_t & hi_p).sum())
    fp = int((~hi_t & hi_p).sum())
    fn = int((hi_t & ~hi_p).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    # 严重错判：跨侧且差距 >= 3（0<->5, 0<->4, 1<->5）
    severe = int((np.abs(y_true - y_pred) >= 3).sum())
    return dict(acc=acc, tol=tol, mae=mae, bin_p=p, bin_r=r, bin_f1=f1,
                tp=tp, fp=fp, fn=fn, severe=severe)


def per_class_f1(y_true, y_pred):
    out = {}
    for c in range(6):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=HOLDOUT)
    ap.add_argument("--no-split", action="store_true",
                    help="不做 tune/eval 切分，在全量上扫描（数字偏乐观，仅看趋势）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    print(f"留出集 {len(rows)} 条，分布 {dict(zip(*np.unique(y, return_counts=True)))}")

    bias = np.array(json.load(open(PRIOR_FILE, encoding="utf-8"))["bias"], dtype=np.float64)
    print(f"先验偏置: {np.round(bias, 3).tolist()}")

    logits = load_logits(rows)
    if logits.shape[0] != len(rows):
        print(f"缓存维度不符({logits.shape[0]} != {len(rows)})，删除重算")
        os.remove(CACHE)
        logits = load_logits(rows)

    if args.no_split:
        tune_idx = eval_idx = np.arange(len(rows))
    else:
        tune_idx = np.arange(0, len(rows), 2)
        eval_idx = np.arange(1, len(rows), 2)

    def sweep(idx, title):
        print(f"\n{'='*78}\n{title}  (n={len(idx)})\n{'='*78}")
        print(f"{'alpha':>6}{'准确率':>9}{'±1容差':>9}{'MAE':>8}"
              f"{'二分F1':>9}{'FP':>6}{'FN':>6}{'严重错判':>9}")
        best = (None, -1.0)
        for a in ALPHAS:
            pred = np.argmax(logits[idx] + a * bias, axis=-1)
            m = metrics(y[idx], pred)
            print(f"{a:>6.1f}{m['acc']:>9.4f}{m['tol']:>9.4f}{m['mae']:>8.4f}"
                  f"{m['bin_f1']:>9.4f}{m['fp']:>6}{m['fn']:>6}{m['severe']:>9}")
            if m["bin_f1"] > best[1]:
                best = (a, m["bin_f1"])
        return best[0]

    best_a = sweep(tune_idx, "【tune 半区】用来选 alpha")

    if args.no_split:
        print("\n(--no-split：未做独立验证，数字偏乐观)")
        return 0

    print(f"\n>>> tune 半区最优 alpha = {best_a}（按二分 F1）")

    for a in (0.0, best_a):
        pred = np.argmax(logits[eval_idx] + a * bias, axis=-1)
        m = metrics(y[eval_idx], pred)
        f1s = per_class_f1(y[eval_idx], pred)
        tag = "当前(alpha=0)" if a == 0.0 else f"候选(alpha={best_a})"
        print(f"\n--- eval 半区 · {tag} ---")
        print(f"  准确率 {m['acc']:.4f}  ±1容差 {m['tol']:.4f}  MAE {m['mae']:.4f}")
        print(f"  二分 P {m['bin_p']:.4f} R {m['bin_r']:.4f} F1 {m['bin_f1']:.4f} "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        print(f"  严重错判(|差|>=3) {m['severe']}")
        print("  分类型 F1: " + "  ".join(f"{c}:{v:.3f}" for c, v in f1s.items()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
