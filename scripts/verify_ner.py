"""NER badcase 回归验证。

把需求 6 排查过程中积累的 6 条关键用例固化下来，随时可跑，防止后续改模型/改解码
又把碎片实体放回去。每条用例声明「必须出现」和「必须不出现」的实体，逐条断言。

用法：
    PYTHONPATH=. python scripts/verify_ner.py
    PYTHONPATH=. python scripts/verify_ner.py --onnx models/ner/backup/xxx.onnx  # 验证某个历史模型
    PYTHONPATH=. python scripts/verify_ner.py --raw                              # 看过滤前的原始解码

退出码：全部通过 0，有失败 1（可直接接 CI）。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 每条用例：
#   must     —— 必须识别出的实体文本（至少出现一次）
#   must_not —— 绝不能出现的文本（碎片实体）
#   empty    —— 为 True 时要求不输出任何实体
CASES = [
    {
        "name": "用户badcase(碎片实体)",
        "title": "小米11 双十一大促",
        "desc": "小米11 双十一大促",
        "comment": "这个活动哪里有",
        # 原始 badcase：模型吐出 一/小/米11 三个碎片
        "must": {"小米11"},
        "must_not": {"一", "小", "米11"},
    },
    {
        "name": "开箱+求购壳",
        "title": "iPhone 17 开箱",
        "desc": "iPhone 17 开箱 用了半个月的感受",
        "comment": "壳有吗 我想要个手机壳",
        "must": {"手机壳"},
        "must_not": set(),
    },
    {
        "name": "售后吐槽",
        "title": "小米空调售后",
        "desc": "小米空调售后是真行 又是被小米服务惊艳到了",
        "comment": "线上买的还是线下买的？",
        "must": {"小米空调", "小米"},
        "must_not": set(),
    },
    {
        "name": "单字合法实体(白名单)",
        "title": "",
        "desc": "这双鞋真的很好穿 推荐买",
        "comment": "什么牌子的鞋",
        # 「鞋」是训练数据里的合法单字实体，不能被长度过滤误杀
        "must": {"鞋"},
        "must_not": set(),
    },
    {
        "name": "多实体家电",
        "title": "",
        "desc": "格力的空调和海尔的冰箱哪个值得买",
        "comment": "求推荐洗衣机",
        "must": {"洗衣机"},
        "must_not": set(),
    },
    {
        "name": "无实体闲聊",
        "title": "",
        "desc": "今天天气真好",
        "comment": "是啊",
        "must": set(),
        "must_not": set(),
        "empty": True,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ner.yaml")
    ap.add_argument("--onnx", default=None, help="指定 ONNX，默认用配置里的 export.onnx_path")
    ap.add_argument("--raw", action="store_true", help="输出过滤前的原始解码（排查用，会跳过断言）")
    args = ap.parse_args()

    from src.ner.inference import NERInference

    eng = NERInference(args.config, onnx_path=args.onnx)

    if args.raw:
        for c in CASES:
            ents = eng.predict_note_comment(c["title"], c["desc"], c["comment"], raw=True)
            print(f"{c['name']} (raw): {[(e['text'], e['type'], e['source']) for e in ents]}")
        return 0

    print(f"{'用例':<22} {'结果':<6} 实体")
    print("-" * 92)
    failed = []
    for c in CASES:
        ents = eng.predict_note_comment(c["title"], c["desc"], c["comment"])
        texts = {e["text"] for e in ents}
        shown = sorted((e["text"], e["type"], e["source"]) for e in ents)

        problems = []
        for t in c["must"]:
            if t not in texts:
                problems.append(f"缺少{t!r}")
        for t in c["must_not"]:
            if t in texts:
                problems.append(f"出现碎片{t!r}")
        if c.get("empty") and texts:
            problems.append(f"应无实体却输出{shown}")

        ok = not problems
        if not ok:
            failed.append((c["name"], problems))
        print(f"{c['name']:<22} {'PASS' if ok else 'FAIL':<6} {shown}")
        for p in problems:
            print(f"{'':<22} {'':<6}   -> {p}")

    print("-" * 92)
    if failed:
        print(f"\n{len(failed)}/{len(CASES)} 条用例失败")
        return 1
    print(f"\n全部 {len(CASES)} 条用例通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
