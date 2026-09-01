"""NER 量化评估 / A-B 对比（note 级别，指标严格落在 [0,1]）。

指标口径（修正旧脚本的统计错误）：
  旧脚本把「真值」按 set 去重，却把「预测」按 (标题/正文/评论) 分段重复计数，
  同一个实体在三个段落各命中一次 → 命中数 3 > 真值 1，导致 R 出现 1.46 这种 >1 的值。
  本脚本在 note 级别对预测文本去重后再比对，P / R / F1 恒在 [0,1]。

用法：
    # 评估当前线上模型
    PYTHONPATH=. python scripts/eval_ner_ab.py --limit 400

    # A/B：老模型 vs 新模型
    PYTHONPATH=. python scripts/eval_ner_ab.py --limit 400 \
        --onnx models/ner/backup/ner_20260901_095847.onnx --tag 老模型
    PYTHONPATH=. python scripts/eval_ner_ab.py --limit 400 --tag 新模型

    # 一次跑完 A/B（自动对比 backup 里最新一份与当前模型）
    PYTHONPATH=. python scripts/eval_ner_ab.py --limit 400 --ab

    # 打印未命中/误报样例（排查 badcase）
    PYTHONPATH=. python scripts/eval_ner_ab.py --limit 100 --show 20
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import yaml  # noqa: E402

from src.ner.inference import NERInference  # noqa: E402
from src.ner.postprocess import load_entity_vocab  # noqa: E402
from src.web import db  # noqa: E402

GOLD_FIELDS = ("intent_brand", "intent_category", "intent_product")
NULLS = {"", "null", "none", "nan"}


def norm(s: str) -> str:
    """实体文本归一化：去空白、小写、去全角空格。"""
    return (s or "").strip().replace("\u3000", "").lower()


def md5_of(path: str, chunk: int = 1 << 20) -> str:
    """文件 md5，用于在 backup 目录里识别真正的旧模型。"""
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_train_texts(path: str) -> set[str]:
    """从 conll 还原训练集原文集合，用于把评估样本拆成「已见 / 未见」。

    坑：conll 行格式是 `token\\tTAG`，必须用制表符切分。若用 split()（按任意空白切），
    空格 token 的行会退化成只有标签 → 还原出的文本里会混入 "O"/"B-BRAND" 等标签字符，
    导致与原文永远匹配不上，重叠率被严重低估（实测 39.5% → 5%）。
    """
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


def sample_text(s: dict) -> str:
    """按与训练/推理完全相同的规则还原样本输入文本。"""
    from src.common.compose import compose_note_comment

    return compose_note_comment(
        s.get("note_title") or "", s.get("note_desc") or "", s.get("comment_content") or ""
    )


def gold_set(sample: dict) -> set[str]:
    out = set()
    for k in GOLD_FIELDS:
        v = norm(str(sample.get(k) or ""))
        if v and v not in NULLS:
            out.add(v)
    return out


def _contain(a: str, b: str) -> bool:
    """包含匹配：相等 / a 包含 b / b 包含 a（长度>=2 才允许子串，避免单字乱匹配）。"""
    if a == b:
        return True
    if min(len(a), len(b)) < 2:
        return False
    return a in b or b in a


def predict_all(engine: NERInference, samples: list[dict]) -> list[tuple[set[str], dict[str, float]]]:
    """跑一遍预测并缓存 note 级别的 (真值集合, {实体文本: 置信度})，供阈值扫描复用。"""
    cache = []
    for s in samples:
        G = gold_set(s)
        if not G:
            continue
        ents = engine.predict_note_comment(
            s.get("note_title") or "",
            s.get("note_desc") or "",
            s.get("comment_content") or "",
        )
        best: dict[str, float] = {}
        for e in ents:
            t = norm(e["text"])
            if not t or t in NULLS:
                continue
            if t not in best or e.get("prob", 0.0) > best[t]:
                best[t] = float(e.get("prob", 0.0))
        cache.append((G, best))
    return cache


def metrics_from_cache(cache, thresh: float = 0.0) -> dict:
    """按给定置信度阈值算指标（预测已按 note 去重）。"""
    tp = n_pred = n_gold = tp_l_gold = tp_l_pred = 0
    for G, P in cache:
        preds = {t for t, p in P.items() if p >= thresh}
        if not preds:
            n_gold += len(G)
            continue
        tp += len(G & preds)
        n_pred += len(preds)
        n_gold += len(G)
        tp_l_gold += sum(1 for g in G if any(_contain(g, p) for p in preds))
        tp_l_pred += sum(1 for p in preds if any(_contain(p, g) for g in G))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    pl = tp_l_pred / n_pred if n_pred else 0.0
    rl = tp_l_gold / n_gold if n_gold else 0.0
    return {
        "n_notes": len(cache), "n_pred": n_pred, "n_gold": n_gold, "tp": tp,
        "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
        "P_len": pl, "R_len": rl, "F1_len": 2 * pl * rl / (pl + rl) if pl + rl else 0.0,
    }


def sweep(cache):
    """扫描置信度阈值，寻找更好的 P/R 平衡点。"""
    print("\n----- 置信度阈值扫描（当前模型）-----")
    print(f"  {'阈值':>6} {'预测数':>7} {'严格P':>8} {'严格R':>8} {'严格F1':>8} "
          f"{'包含P':>8} {'包含R':>8} {'包含F1':>8}")
    best = None
    for i in range(0, 10):
        th = round(i * 0.05, 2)
        m = metrics_from_cache(cache, th)
        print(f"  {th:>6.2f} {m['n_pred']:>7} {m['P']:>8.4f} {m['R']:>8.4f} {m['F1']:>8.4f} "
              f"{m['P_len']:>8.4f} {m['R_len']:>8.4f} {m['F1_len']:>8.4f}")
        if best is None or m["F1"] > best[1]["F1"]:
            best = (th, m)
    print(f"\n  最优阈值(严格F1) = {best[0]:.2f}  →  P={best[1]['P']:.4f} R={best[1]['R']:.4f} "
          f"F1={best[1]['F1']:.4f} | 包含F1={best[1]['F1_len']:.4f}")
    return best


def evaluate(engine: NERInference, samples: list[dict], vocab: set[str], show: int = 0):
    """note 级别 micro 平均；预测按文本去重。

    同时给出两套口径：
      strict    —— 文本完全相等才算命中（会被「粒度不同」误判，如真值「手机」vs 预测「iphone17」）
      lenient   —— 包含匹配也算命中（真值/预测互为子串），用于判断「到底找没找到这个东西」
    """
    tp = n_pred = n_gold = 0
    tp_l_gold = tp_l_pred = 0
    n_with_pred = 0
    frag_counter: Counter[str] = Counter()
    miss_counter: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    shown = 0

    for s in samples:
        G = gold_set(s)
        if not G:
            continue
        ents = engine.predict_note_comment(
            s.get("note_title") or "",
            s.get("note_desc") or "",
            s.get("comment_content") or "",
        )

        # 关键：note 级别去重，消除「同一实体在三段各命中一次」的重复计数
        pred_texts = {norm(e["text"]) for e in ents if norm(e["text"])}
        pred_texts = {t for t in pred_texts if t not in NULLS}

        hit = G & pred_texts
        tp += len(hit)
        n_pred += len(pred_texts)
        n_gold += len(G)
        if pred_texts:
            n_with_pred += 1

        # 包含匹配口径
        tp_l_gold += sum(1 for g in G if any(_contain(g, p) for p in pred_texts))
        tp_l_pred += sum(1 for p in pred_texts if any(_contain(p, g) for g in G))

        # 碎片统计：单字且不在训练实体白名单 → 碎片
        for e in ents:
            t = norm(e["text"])
            if len(t) == 1 and t not in vocab:
                frag_counter[t] += 1

        miss_counter.update(G - pred_texts)
        fp_counter.update(pred_texts - G)

        if show and shown < show and (G - pred_texts or pred_texts - G):
            shown += 1
            print(f"\n  [{shown}] 标题={str(s.get('note_title') or '')[:30]!r}")
            print(f"      正文={str(s.get('note_desc') or '')[:60]!r}")
            print(f"      评论={str(s.get('comment_content') or '')[:40]!r}")
            print(f"      真值={sorted(G)}")
            print(f"      预测={sorted(pred_texts)}")
            print(f"      漏召={sorted(G - pred_texts)}  误报={sorted(pred_texts - G)}")

    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    pl = tp_l_pred / n_pred if n_pred else 0.0
    rl = tp_l_gold / n_gold if n_gold else 0.0
    f1l = 2 * pl * rl / (pl + rl) if pl + rl else 0.0
    return {
        "n_notes": len([s for s in samples if gold_set(s)]),
        "n_pred": n_pred,
        "n_gold": n_gold,
        "tp": tp,
        "P": p,
        "R": r,
        "F1": f1,
        "P_len": pl,
        "R_len": rl,
        "F1_len": f1l,
        "coverage": n_with_pred / max(1, len([s for s in samples if gold_set(s)])),
        "frag_hits": sum(frag_counter.values()),
        "frag_top": frag_counter.most_common(15),
        "miss_top": miss_counter.most_common(15),
        "fp_top": fp_counter.most_common(15),
    }


def report(tag: str, m: dict, show: int):
    print(f"\n===== {tag} =====")
    print(f"  样本={m['n_notes']}  真值实体={m['n_gold']}  预测实体(去重后)={m['n_pred']}  命中={m['tp']}")
    print(f"  [严格相等] P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
    print(f"  [包含匹配] P={m['P_len']:.4f}  R={m['R_len']:.4f}  F1={m['F1_len']:.4f}  "
          f"有预测覆盖率={m['coverage']:.4f}")
    print(f"  碎片实体(单字且不在白名单)={m['frag_hits']}  top={m['frag_top'][:10]}")
    print(f"  漏召 top: {m['miss_top'][:8]}")
    print(f"  误报 top: {m['fp_top'][:8]}")
    if show:
        print("  （样例见上方）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ner.yaml")
    ap.add_argument("--onnx", default=None, help="指定 ONNX 路径，默认用配置里的 export.onnx_path")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ab", action="store_true", help="对比 backup 里最新一份与当前模型")
    ap.add_argument("--show", type=int, default=0, help="打印 N 个漏召/误报样例")
    ap.add_argument("--sweep", action="store_true", help="扫描置信度阈值找更优 P/R 平衡点")
    ap.add_argument("--holdout", action="store_true",
                    help="按「训练集是否见过」拆分评估，量化记忆 vs 泛化（强烈建议开启）")
    ap.add_argument("--train-conll", default="data/ner/annotated/train.conll",
                    help="训练集 conll，用于 --holdout 判定样本是否被训过")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vocab_file = (cfg.get("data") or {}).get("entity_vocab_file")
    vocab = load_entity_vocab(vocab_file)

    print(f"加载 {args.limit} 条真实样本…", flush=True)
    samples = db.fetch_ner_samples(args.limit)
    print(f"  实际取到 {len(samples)} 条", flush=True)

    targets: list[tuple[str, str]] = []
    if args.ab:
        cur = args.onnx or cfg["export"]["onnx_path"]
        cur_md5 = md5_of(cur)
        # 按 mtime 倒序取 backup，按 md5 去重，并剔除与当前模型字节相同的那份
        # （导出时 backup 目录也会写入新模型，只按路径/文件名区分会拿错「老模型」）
        seen = {cur_md5}
        old = None
        for b in sorted(glob.glob("models/ner/backup/*.onnx"), key=os.path.getmtime, reverse=True):
            if os.path.abspath(b) == os.path.abspath(cur):
                continue
            m = md5_of(b)
            if m in seen:
                continue
            seen.add(m)
            old = b
            break
        if old:
            targets.append((f"老模型({os.path.basename(old)})", old))
        targets.append(("新模型(当前)", cur))
    else:
        targets.append((args.tag or (args.onnx or cfg["export"]["onnx_path"]),
                        args.onnx or cfg["export"]["onnx_path"]))

    # 分组：默认整体评估；--holdout 拆成「已见 / 未见」两组
    # 训练集覆盖率约 43%（52k/120k），随机取样必然混入 4 成训过的样本，
    # 不拆分的话指标是「记忆 + 泛化」的混合值，会系统性高估真实效果。
    if args.holdout:
        train_texts = load_train_texts(args.train_conll)
        seen, unseen = [], []
        for s in samples:
            (seen if sample_text(s) in train_texts else unseen).append(s)
        print(f"  训练集 {len(train_texts):,} 条 → 已见 {len(seen)} 条 / 未见 {len(unseen)} 条", flush=True)
        groups = [(f"已见·训练集内({len(seen)})", seen), (f"未见·训练集外({len(unseen)})", unseen)]
    else:
        groups = [("全部", samples)]

    results = {}
    for tag, onnx in targets:
        print(f"\n--- 加载模型: {onnx} ---", flush=True)
        engine = NERInference(args.config, onnx_path=onnx)
        if args.show:
            print(f"\n--- {tag} 差异样例 ---")
        for gname, gsamples in groups:
            m = evaluate(engine, gsamples, vocab, show=args.show)
            full_tag = f"{tag} | {gname}" if len(groups) > 1 else tag
            report(full_tag, m, args.show)
            results[full_tag] = m

        if args.sweep:
            sweep(predict_all(engine, samples))

    # A/B 结论：按分组配对比较（有分组时 results 是 模型数×分组数）
    if len(targets) == 2 and results:
        pairs = []
        if len(groups) > 1:
            for gname, _ in groups:
                k1 = f"{targets[0][0]} | {gname}"
                k2 = f"{targets[1][0]} | {gname}"
                if k1 in results and k2 in results:
                    pairs.append((k1, k2))
        else:
            pairs.append((targets[0][0], targets[1][0]))

        for t1, t2 in pairs:
            m1, m2 = results[t1], results[t2]
            print(f"\n===== A/B 结论（{t1} → {t2}）=====")
            for k, label in (("P", "精确率"), ("R", "召回率"), ("F1", "F1"),
                             ("P_len", "精确率(包含)"), ("R_len", "召回率(包含)"), ("F1_len", "F1(包含)")):
                d = m2[k] - m1[k]
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
                if m1[k]:
                    print(f"  {label:14s}: {m1[k]:.4f} → {m2[k]:.4f}  {arrow}{abs(d):.4f} "
                          f"({d / m1[k] * 100:+.1f}%)")
                else:
                    print(f"  {label:14s}: n/a → {m2[k]:.4f}")
            print(f"  碎片: {m1['frag_hits']} → {m2['frag_hits']}")
            print(f"  预测量: {m1['n_pred']} → {m2['n_pred']}")


if __name__ == "__main__":
    main()
