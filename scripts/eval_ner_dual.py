"""NER 双轨评测：ES 真值轨（A）+ 词典自洽轨（B）。

为什么必须双轨
--------------
ES 真值是外部程序的初始化输出，本身不完整：实测词典词在语料中出现 25,958 次，
现有标注只覆盖了 12,699 次（漏标 51.1%）。也就是说，「模型标出来了但真值没有」
里很大一部分其实是**真值漏记**，不是模型误报（此前实测 69.5% 的"误报"命中词典）。

只用 A 轨会得出「改进反而变差」的错误结论；只用 B 轨会自说自话。
两轨一起看才能判断改进到底是召回提升还是真的变差：

  A 轨 gold：真值 = _align(text, ES 品牌/品类/商品字段)
     -> 看相对旧模型的**变化方向**，绝对值受真值不全压制
  B 轨 dict：真值 = 词典全量一致标注
     -> 看模型在**未见文本**上复现「一致 + 完整」标准的能力

两轨都在留出集上跑（训练时没见过的文档）。

额外指标
--------
- 类型一致性：同一个实体文本被预测成多种类型的比例（越低越好）
- 碎片率：单字实体占比、非词典实体占比

基线（务必用这个锚点对比，不要用旧数字）
----------------------------------------
2026-09-01 实测，旧模型（词典重训前）`backup/ner_20260901_095855.onnx`：

    全量留出集 14,135 条（主锚点）：
      A 轨 P 75.7% / R 80.7% / F1 78.1%
      B 轨 P 74.7% / R 38.9% / F1 51.1%
      B 轨分类型 F1：CATEGORY 61.6% > BRAND 49.6% > MODEL 24.1% > PRODUCT 19.4%
      一致性摇摆 6.9%（374/5438），碎片 0.2%，词典外 11.9%

    留出集前 2000 条（与全量相差 <1pp，说明 2000 条已具代表性，快速回归可用）：
      A 轨 P 76.4% / R 81.0% / F1 78.7%
      B 轨 P 75.3% / R 39.6% / F1 51.9%
      B 轨分类型 F1：CATEGORY 61.7% > BRAND 51.9% > MODEL 27.9% > PRODUCT 19.2%
      一致性摇摆 5.0%（68/1353），碎片 0.2%，词典外 11.6%

    ⚠️ 更早期测出过「A 轨 71.3% / B 轨 48.1%」，那组数字有 bug 已被修正：
    聚合时 span 元组漏带 doc_id，start 是文档内相对偏移，跨文档高频实体
    （不同文档第 5 个字都是「空调」）被误去重，系统性压低了 P/R/F1。
    现在 spans_from_tags/spans_from_preds 都会带上 doc_id。

    B 轨 P 高 R 低的形态说明旧模型不是标错，而是**不敢标、标不全**，
    继承了训练数据 51.1% 的漏标。重训的成败看 B 轨 R 能否大幅抬升且 P 不明显掉。

复现方式（显式指定模型，不要依赖「当前 onnx 是哪个版本」）：
    # 旧模型基线
    python scripts/eval_ner_dual.py --limit 2000 \
        --onnx models/ner/backup/ner_20260901_095855.onnx
    # 新模型（训练导出后）
    python scripts/eval_ner_dual.py --limit 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ner.inference import NERInference
from src.web.config import BASE_DIR
from src.web.data_prep_service import (
    DictAnnotator,
    NER_HOLDOUT_JSONL,
    _align,
    build_dict_entity_map,
    load_dict_entity_map,
    load_entity_type_map,
)


def spans_from_tags(text: str, tags: List[str], doc_id: int) -> Set[Tuple[int, int, str, str]]:
    """从字符级 BIO 标签还原实体集合 {(doc_id, start, type, surface)}。

    doc_id 必须带上：start 是**文档内**相对偏移，不同文档第 5 个字都是「空调」
    会被当成同一实体。早期版本漏了 doc_id，跨文档高频实体被误去重，
    系统性压低 P/R/F1（基线数字就是这么测出来的，修正后需重测基线）。
    """
    out, cur_type, cur, st = set(), "", "", 0
    for i, (ch, tg) in enumerate(zip(text, tags)):
        if tg.startswith("B-"):
            if cur_type:
                out.add((doc_id, st, cur_type, cur))
            cur_type, cur, st = tg[2:], ch, i
        elif tg.startswith("I-") and cur_type == tg[2:]:
            cur += ch
        else:
            if cur_type:
                out.add((doc_id, st, cur_type, cur))
            cur_type, cur = "", ""
    if cur_type:
        out.add((doc_id, st, cur_type, cur))
    return out


def spans_from_preds(preds: List[dict], doc_id: int) -> Set[Tuple[int, int, str, str]]:
    return {(doc_id, p["start"], p["type"], p["text"]) for p in preds}


def prf(gold: Set, pred: Set) -> Tuple[float, float, float, int]:
    hit = len(gold & pred)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="0=全部留出集")
    ap.add_argument("--config", default="configs/ner.yaml")
    ap.add_argument("--holdout", default=NER_HOLDOUT_JSONL)
    ap.add_argument("--show", type=int, default=10, help="展示几个差例")
    ap.add_argument(
        "--onnx", default=None,
        help="指定要评测的 ONNX。留空用 configs/ner.yaml 的 export.onnx_path（即当前线上模型）。"
             "对比新旧模型时显式指定：旧模型在 models/ner/backup/ 下按时间戳保存，"
             "不要依赖「当前 onnx 是哪个版本」这种隐式状态。",
    )
    args = ap.parse_args()

    if not os.path.exists(args.holdout):
        print(f"错误：找不到留出集 {args.holdout}")
        print("请先用 annotator=dict、holdout_ratio>0 重新生成 NER 数据。")
        return

    rows = [json.loads(l) for l in open(args.holdout, "r", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"留出集样本: {len(rows)}")

    wmap = load_dict_entity_map() or build_dict_entity_map()
    annot = DictAnnotator(wmap)
    type_map = load_entity_type_map()

    inf = NERInference(args.config, onnx_path=args.onnx)
    print(f"评测模型: {args.onnx or 'configs/ner.yaml -> export.onnx_path'}")

    agg = {"A": [set(), set()], "B": [set(), set()]}
    per_type = {"A": defaultdict(lambda: [set(), set()]), "B": defaultdict(lambda: [set(), set()])}
    # 一致性：实体文本 -> 被预测成的类型集合
    text2types: Dict[str, Set[str]] = defaultdict(set)
    n_pred_ent = 0
    n_frag = 0          # 单字实体
    n_oov = 0           # 不在词典里的实体
    badcases = []

    for i, r in enumerate(rows):
        text = r["text"]
        if not text:
            continue
        # A 轨真值
        _, gold_tags = _align(
            text, r.get("intent_brand"), r.get("intent_category"),
            r.get("intent_product"), type_map,
        )
        # B 轨真值
        _, dict_tags = annot.annotate(text)
        G = spans_from_tags(text, gold_tags, i)
        D = spans_from_tags(text, dict_tags, i)

        preds = inf.predict(text)
        P = spans_from_preds(preds, i)

        for k, GS in (("A", G), ("B", D)):
            agg[k][0] |= GS
            agg[k][1] |= P
            for sp in GS:
                per_type[k][sp[2]][0].add(sp)
            for sp in P:
                per_type[k][sp[2]][1].add(sp)

        for _d, _st, t, s in P:
            text2types[s].add(t)
            n_pred_ent += 1
            if len(s) == 1:
                n_frag += 1
            if s not in wmap:
                n_oov += 1

        if len(badcases) < args.show and D and (len(D & P) / len(D) < 0.5):
            badcases.append((text, sorted(D - P), sorted(P - D)))

        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(rows)}", flush=True)

    print(f"\n预测实体总数: {n_pred_ent}")

    print("\n" + "=" * 62)
    print(f"{'轨道':<34}{'P':>9}{'R':>9}{'F1':>9}{'命中':>8}")
    print("=" * 62)
    labels = {"A": "A 轨 · ES 真值（真值不全，仅看变化）",
              "B": "B 轨 · 词典全量一致标注（自洽标准）"}
    for k in ("A", "B"):
        p, r, f, hit = prf(agg[k][0], agg[k][1])
        print(f"{labels[k]:<34}{p*100:>8.1f}%{r*100:>8.1f}%{f*100:>8.1f}%{hit:>8}")

    print("\n--- B 轨 分类型 ---")
    print(f"{'类型':<12}{'P':>9}{'R':>9}{'F1':>9}")
    for t in sorted(per_type["B"].keys()):
        p, r, f, _ = prf(per_type["B"][t][0], per_type["B"][t][1])
        print(f"{t:<12}{p*100:>8.1f}%{r*100:>8.1f}%{f*100:>8.1f}%")

    multi = {w: ts for w, ts in text2types.items() if len(ts) > 1}
    print(f"\n--- 一致性 / 碎片 ---")
    print(f"被预测成多种类型的实体文本: {len(multi)} / {len(text2types)} "
          f"({100*len(multi)/max(1,len(text2types)):.1f}%)")
    print(f"单字碎片实体占比: {100*n_frag/max(1,n_pred_ent):.1f}%")
    print(f"词典外实体占比  : {100*n_oov/max(1,n_pred_ent):.1f}%  "
          f"（泛化能力，不是越低越好）")
    if multi:
        top = sorted(multi.items(), key=lambda x: -len(x[1]))[:8]
        print("  摇摆最严重的词:", ", ".join(f"{w}({'/'.join(sorted(t))})" for w, t in top))

    if badcases:
        print(f"\n--- B 轨差例（召回<50%）---")
        for text, miss, extra in badcases[: args.show]:
            print(f"\n  文本: {text[:70]}")
            print(f"    漏标: {[m[3] for m in miss][:6]}")
            print(f"    多标: {[e[3] for e in extra][:6]}")


if __name__ == "__main__":
    main()
