"""生成示例数据，让工程骨架可以端到端跑通。

生成：
1. data/ner/annotated/train.conll —— NER 字符级 BIO 标注示例
2. data/intent/labeled/train.jsonl —— 意图正负样本

用法：
    python scripts/make_sample_data.py
"""
from __future__ import annotations

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (文本, [(实体文本, 实体类型), ...])
NER_SAMPLES = [
    ("我想买一台小米14 Pro手机", [("小米", "BRAND"), ("14 Pro", "MODEL"), ("手机", "CATEGORY")]),
    ("华为Mate60 Pro多少钱", [("华为", "BRAND"), ("Mate60 Pro", "MODEL")]),
    ("推荐一款苹果的耳机", [("苹果", "BRAND"), ("耳机", "CATEGORY")]),
    ("这个AirPods Pro价格怎么样", [("AirPods Pro", "PRODUCT")]),
    ("三星Galaxy S24 Ultra有货吗", [("三星", "BRAND"), ("Galaxy S24 Ultra", "MODEL")]),
    ("李宁的跑步鞋怎么样", [("李宁", "BRAND"), ("跑步鞋", "CATEGORY")]),
    ("想入手联想小新笔记本", [("联想", "BRAND"), ("小新", "PRODUCT"), ("笔记本", "CATEGORY")]),
    ("小米14 256G版本多少钱", [("小米", "BRAND"), ("14", "MODEL")]),
    ("OPPO Find X7 拍照怎么样", [("OPPO", "BRAND"), ("Find X7", "MODEL")]),
    ("戴尔XPS 13的屏幕素质", [("戴尔", "BRAND"), ("XPS 13", "MODEL")]),
    ("这个键盘手感不错", [("键盘", "CATEGORY")]),
    ("帮我看看荣耀的手表", [("荣耀", "BRAND"), ("手表", "CATEGORY")]),
    ("海尔冰箱价格多少", [("海尔", "BRAND"), ("冰箱", "CATEGORY")]),
    ("耐克运动鞋有活动吗", [("耐克", "BRAND"), ("运动鞋", "CATEGORY")]),
    ("想买索尼的相机", [("索尼", "BRAND"), ("相机", "CATEGORY")]),
]

INTENT_POS = [
    "我想买一台手机",
    "这个多少钱",
    "有优惠活动吗",
    "帮我下单买这个",
    "这款有没有现货",
    "现在入手划算吗",
    "推荐一款性价比高的",
    "能不能便宜点",
    "我要买两件",
    "有没有折扣券",
    "想入手一台笔记本",
    "这个可以包邮吗",
]

INTENT_NEG = [
    "今天天气真不错",
    "帮我查一下物流到哪了",
    "这个软件怎么使用",
    "你叫什么名字",
    "给我讲个笑话",
    "帮我设置一个闹钟",
    "翻译一下这句话",
    "今天的新闻有什么",
    "帮我写一封邮件",
    "这首歌是谁唱的",
]


def text_to_bio(text: str, entities) -> list:
    """把 (text, entities) 转成字符级 BIO 标签。"""
    tags = ["O"] * len(text)
    for etext, etype in entities:
        idx = text.find(etext)
        if idx == -1:
            print(f"[warn] 找不到实体 '{etext}' 于 '{text}'，跳过")
            continue
        for j in range(len(etext)):
            tags[idx + j] = ("B-" if j == 0 else "I-") + etype
    return list(zip(list(text), tags))


def make_ner():
    out = os.path.join(BASE_DIR, "data/ner/annotated/train.conll")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lines = []
    for text, entities in NER_SAMPLES:
        for tok, tag in text_to_bio(text, entities):
            lines.append(f"{tok}\t{tag}")
        lines.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sample] NER 标注写入 {len(NER_SAMPLES)} 句 -> {out}")


def make_intent():
    out = os.path.join(BASE_DIR, "data/intent/labeled/train.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = [{"text": t, "label": 1} for t in INTENT_POS] + [
        {"text": t, "label": 0} for t in INTENT_NEG
    ]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sample] 意图样本写入 正{len(INTENT_POS)}/负{len(INTENT_NEG)} -> {out}")


if __name__ == "__main__":
    make_ner()
    make_intent()
    print("示例数据生成完成，可开始训练。")
