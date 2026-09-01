"""新旧 NER 模型逐实体对比：专门看「新模型多标了什么、少标了什么」。

为什么需要这个脚本
------------------
双轨评测只能证明「新模型符合词典标准」（B 轨 F1 94.8%）。
但词典本身是从 ES gold 派生的，若词典里混了垃圾词，模型标得越一致、
越完整，错得也就越整齐。B 轨是自洽标准，自己不能证明自己正确。

所以必须单独回答：**新模型比旧模型多标出来的那近一倍实体，到底是不是对的？**

做法：同一批留出集文本分别过新旧两个 ONNX，逐实体做差集：
  - NEW_ONLY：新标了、旧没标 -> 关注点。若在词典中且语义合理，说明是「补全」；
    若是垃圾词，说明词典被污染、整套标注标准要重新审视。
  - OLD_ONLY：旧标了、新没标 -> 回归检查。若丢的是真实体，说明重训有副作用。

用法：
    python scripts/compare_ner_models.py --limit 800
    python scripts/compare_ner_models.py --old models/ner/backup/ner_20260901_095855.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ner.inference import NERInference  # noqa: E402
from src.web.data_prep_service import (  # noqa: E402
    NER_HOLDOUT_JSONL,
    build_dict_entity_map,
    load_dict_entity_map,
)

OLD_ONNX = os.path.join("models", "ner", "backup", "ner_20260901_095855.onnx")


def spans(preds: List[dict], doc_id: int) -> Set[Tuple[int, int, str, str]]:
    return {(doc_id, p["start"], p["type"], p["text"]) for p in preds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--old", default=OLD_ONNX, help="旧模型 ONNX")
    ap.add_argument("--new", default=None, help="新模型 ONNX（默认用 configs 里的线上模型）")
    ap.add_argument("--holdout", default=NER_HOLDOUT_JSONL)
    ap.add_argument("--top", type=int, default=40, help="展示高频差集词数量")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"留出集样本: {len(rows)}")

    wmap = load_dict_entity_map() or build_dict_entity_map()

    cfg = "configs/ner.yaml"
    old_inf = NERInference(cfg, onnx_path=args.old)
    new_inf = NERInference(cfg, onnx_path=args.new)

    OLD: Set[Tuple[int, int, str, str]] = set()
    NEW: Set[Tuple[int, int, str, str]] = set()
    for i, r in enumerate(rows):
        text = r.get("text") or ""
        if not text:
            continue
        OLD |= spans(old_inf.predict(text), i)
        NEW |= spans(new_inf.predict(text), i)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(rows)}", flush=True)

    new_only = NEW - OLD
    old_only = OLD - NEW
    both = NEW & OLD

    print(f"\n旧模型实体: {len(OLD)}   新模型实体: {len(NEW)}")
    print(f"  两者都标: {len(both)}")
    print(f"  仅新模型: {len(new_only)}   ({100*len(new_only)/max(1,len(NEW)):.1f}% of 新模型输出)")
    print(f"  仅旧模型: {len(old_only)}   ({100*len(old_only)/max(1,len(OLD)):.1f}% of 旧模型输出)")

    def analyze(name: str, S: Set[Tuple[int, int, str, str]]):
        if not S:
            return
        in_dict = sum(1 for _d, _s, _t, w in S if w in wmap)
        print(f"\n{'='*66}\n{name}  共 {len(S)} 个")
        print(f"{'='*66}")
        print(f"  命中词典: {in_dict} ({100*in_dict/len(S):.1f}%)   词典外: {len(S)-in_dict} "
              f"({100*(len(S)-in_dict)/len(S):.1f}%)")

        c: Counter = Counter()
        tdist: Dict[str, Counter] = {}
        for _d, _s, t, w in S:
            c[w] += 1
            tdist.setdefault(w, Counter())[t] += 1
        print(f"\n  出现最多的 {min(args.top, len(c))} 个（格式：词 ×次数 [类型] 词典内/外）")
        for w, n in c.most_common(args.top):
            ts = "/".join(sorted(tdist[w]))
            mark = "典内" if w in wmap else "典外"
            dtype = wmap.get(w)
            extra = f"  (词典类型={dtype})" if dtype and dtype not in tdist[w] else ""
            print(f"    {w:<24} ×{n:<4} [{ts}] {mark}{extra}")

    analyze("【A】新模型多标出来的实体（重点审查：是补全还是垃圾）", new_only)
    analyze("【B】新模型丢掉的实体（回归检查）", old_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
