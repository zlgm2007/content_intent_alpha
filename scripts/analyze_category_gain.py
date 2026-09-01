"""量化「品类级映射」能带来多少 NER 收益（离线模拟，不改任何模型）。

背景
----
未见样本的漏召 top 是 `苹果`/`手机`/`空调`（粗粒度品类），误报 top 却是
`iphone17`/`p20max`/`t90`（细粒度型号）。模型没错，是**粒度错配**：
严格口径下这算 1 次漏召 + 1 次误报，两边同时扣分。

思路
----
ES 的 intent_product/intent_brand 与 intent_category 天然共现，可统计出
「iphone17 → 手机(486次)」这类映射。给模型预测出的细粒度实体补一层品类，
即可同时消掉漏召和误报。

公平性（关键）
--------------
映射**只用训练集中出现过的行**构建。若用全部行，评估样本自身的
(iphone17, 手机) 标签会直接进入映射表，等于把答案喂给评估 → 结果虚高。

用法：
    PYTHONPATH=. python scripts/analyze_category_gain.py --limit 700
    PYTHONPATH=. python scripts/analyze_category_gain.py --limit 700 --min-count 5 --show 15
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import yaml  # noqa: E402

from src.common.compose import compose_note_comment  # noqa: E402
from src.ner.inference import NERInference  # noqa: E402
from src.web import db  # noqa: E402

NULLS = {"", "null", "none", "nan"}


def norm(s: str) -> str:
    return (s or "").strip().replace("\u3000", "").lower()


def load_train_texts(path: str) -> set[str]:
    """conll 行格式是 token\\tTAG，必须按制表符切（用 split() 会被空格 token 破坏）。"""
    texts: set[str] = set()
    cur: list[str] = []
    if not os.path.exists(path):
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    texts.add("".join(cur))
                    cur = []
                continue
            cur.append(line.split("\t")[0])
    if cur:
        texts.add("".join(cur))
    return texts


def contain(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 2:
        return False
    return a in b or b in a


def build_maps(rows_seen: list[dict], min_count: int) -> dict[str, str]:
    """从『训练集内的行』统计 实体->主品类 映射。"""
    counter: dict[str, Counter] = defaultdict(Counter)
    for r in rows_seen:
        cats = {norm(r.get("intent_category") or "")}
        cats = {c for c in cats if c and c not in NULLS}
        if not cats:
            continue
        for f in ("intent_product", "intent_brand"):
            k = norm(r.get(f) or "")
            if k and k not in NULLS:
                for c in cats:
                    counter[k][c] += 1
    out: dict[str, str] = {}
    for k, c in counter.items():
        cat, n = c.most_common(1)[0]
        if n >= min_count and cat != k:
            out[k] = cat
    return out


def score(pairs: list[tuple[set[str], set[str]]]) -> dict:
    """pairs = [(gold_set, pred_set), ...]，返回严格 / 包含两套口径。"""
    tp = n_pred = n_gold = tp_l_g = tp_l_p = 0
    for G, P in pairs:
        n_gold += len(G)
        n_pred += len(P)
        tp += len(G & P)
        tp_l_g += sum(1 for g in G if any(contain(g, p) for p in P))
        tp_l_p += sum(1 for p in P if any(contain(p, g) for g in G))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    pl = tp_l_p / n_pred if n_pred else 0.0
    rl = tp_l_g / n_gold if n_gold else 0.0
    return {
        "n_pred": n_pred, "n_gold": n_gold, "tp": tp,
        "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
        "P_len": pl, "R_len": rl, "F1_len": 2 * pl * rl / (pl + rl) if pl + rl else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ner.yaml")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--limit", type=int, default=700, help="未见样本评估条数")
    ap.add_argument("--min-count", type=int, default=3, help="映射最小共现次数")
    ap.add_argument("--mode", choices=["add", "replace", "both"], default="both",
                    help="add=保留型号并补品类；replace=用品类替换型号；both=两种都跑")
    ap.add_argument("--train-conll", default="data/ner/annotated/train.conll")
    ap.add_argument("--show", type=int, default=0, help="打印 N 条映射生效的样例")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("加载全部 es_docs 并划分已见/未见…", flush=True)
    seen_texts = load_train_texts(args.train_conll)
    all_rows = db.fetch_ner_samples(0)  # 0 = 全部
    rows_seen, pool_unseen = [], []
    for r in all_rows:
        t = compose_note_comment(
            r.get("note_title") or "", r.get("note_desc") or "", r.get("comment_content") or ""
        )
        (rows_seen if t in seen_texts else pool_unseen).append(r)
    print(f"  训练集 {len(seen_texts):,} 条 | 已见行 {len(rows_seen):,} | 未见行 {len(pool_unseen):,}",
          flush=True)

    cmap = build_maps(rows_seen, args.min_count)
    print(f"  构建映射(仅用已见行, 最小共现 {args.min_count})：{len(cmap):,} 条", flush=True)
    for w in ("iphone17", "iphone", "p20max", "t90", "石头", "格力"):
        if w in cmap:
            print(f"    {w} -> {cmap[w]}")

    import random

    random.shuffle(pool_unseen)
    samples = pool_unseen[: args.limit]
    print(f"\n在 {len(samples)} 条未见样本上评估…", flush=True)

    engine = NERInference(args.config, onnx_path=args.onnx)

    modes = ["add", "replace"] if args.mode == "both" else [args.mode]
    results: dict[str, list] = {m: [] for m in modes}
    base_pairs, shown = [], 0
    for s in samples:
        G = {norm(str(s.get(k) or "")) for k in ("intent_brand", "intent_category", "intent_product")}
        G = {g for g in G if g and g not in NULLS}
        if not G:
            continue
        ents = engine.predict_note_comment(
            s.get("note_title") or "", s.get("note_desc") or "", s.get("comment_content") or ""
        )
        P = {norm(e["text"]) for e in ents}
        P = {p for p in P if p and p not in NULLS}

        src = {p for p in P if p in cmap}
        mapped = {cmap[p] for p in src}

        base_pairs.append((G, P))
        for m in modes:
            results[m].append((G, (P | mapped) if m == "add" else ((P - src) | mapped)))

        if args.show and shown < args.show and mapped and (G & (mapped - P)):
            shown += 1
            print(f"\n  [{shown}] 预测={sorted(P)}")
            print(f"      映射={ {k: cmap[k] for k in sorted(src)} }")
            print(f"      真值={sorted(G)}  新增命中={sorted(G & (mapped - P))}")

    b = score(base_pairs)
    print(f"\n{'=' * 74}")
    print(f"未见样本 {len(base_pairs)} 条")
    print(f"{'':20}{'严格P':>9}{'严格R':>9}{'严格F1':>9} |{'包含P':>9}{'包含R':>9}{'包含F1':>9}"
          f"{'预测数':>9}{'命中':>8}")
    print(f"{'现状(仅模型预测)':<18}{b['P']:>9.4f}{b['R']:>9.4f}{b['F1']:>9.4f} |"
          f"{b['P_len']:>9.4f}{b['R_len']:>9.4f}{b['F1_len']:>9.4f}"
          f"{b['n_pred']:>9}{b['tp']:>8}")

    labels = {"add": "+品类(补充,保留型号)", "replace": "+品类(替换型号)"}
    for m in modes:
        a = score(results[m])
        d, dl = a["F1"] - b["F1"], a["F1_len"] - b["F1_len"]
        dp = (d / b["F1"] * 100) if b["F1"] else 0
        dpl = (dl / b["F1_len"] * 100) if b["F1_len"] else 0
        print(f"{labels[m]:<18}{a['P']:>9.4f}{a['R']:>9.4f}{a['F1']:>9.4f} |"
              f"{a['P_len']:>9.4f}{a['R_len']:>9.4f}{a['F1_len']:>9.4f}"
              f"{a['n_pred']:>9}{a['tp']:>8}"
              f"   严格F1 {d:+.4f}({dp:+.1f}%) 包含F1 {dl:+.4f}({dpl:+.1f}%)")

    print("\n注：映射只用训练集内的行构建，未泄漏评估样本真值。")


if __name__ == "__main__":
    main()
