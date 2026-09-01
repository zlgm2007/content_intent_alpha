"""重训 NER 流水线（词典全量一致标注数据版）。

流程：
  1. 通过训练中心启动 NER 训练（走 /api/train/ner，自动落库 + MPS 加速）
  2. 轮询任务状态直到 done / failed
  3. 导出 ONNX
  4. 基于新标注重建实体白名单（单字实体的合法集）
  5. 验证：用户 badcase + 400 条真实样本量化评估

前置：本脚本【不】做数据准备，需先生成好 data/ner/annotated/train.conll。
  想用词典全量标注（推荐）先跑：
    PYTHONPATH=. python -c "from src.web import data_prep_service as d; \
print(d.build_ner_data(limit=0, min_entities=1, max_per_pattern=5, annotator='dict', holdout_ratio=0.1))"
  想回到 ES 真值对齐：annotator='gold'。

注意：第 4 步白名单从 train.conll 重建，换了标注标准后**必须**重跑本脚本，
否则会用旧标准的单字白名单过滤新模型的输出。

用法：
    PYTHONPATH=. python scripts/retrain_ner.py
"""
from __future__ import annotations

import os
import sys
import time

import requests
import yaml

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE = "http://localhost:8000"
# 轮询上限要大于实际训练时长，否则脚本会在训练完之前退出、跳过 ONNX 导出。
# 实测 MPS 上约 81 step/min：80,718 条 / batch32 × 3 epoch = 7,567 step ≈ 93 分钟。
POLL_TIMEOUT_MIN = 150


def log(msg: str):
    print(f"\n===== {msg} =====", flush=True)


def main():
    # ---------- 1. 启动训练 ----------
    log("[1/5] 启动 NER 训练（类型归一化后的新数据）")
    r = requests.post(
        f"{BASE}/api/train/ner",
        json={"epochs": 3, "batch_size": 32, "learning_rate": 2e-5},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # 后端并发保护：已有任务在跑时 ok=false，此时不应继续轮询
    if data.get("ok") is False:
        print(f"启动被拒绝: {data.get('message')}", flush=True)
        sys.exit(1)
    task_id = data["task_id"]
    print(f"task_id = {task_id}", flush=True)

    # ---------- 2. 轮询 ----------
    log("[2/5] 等待训练完成")
    deadline = time.time() + POLL_TIMEOUT_MIN * 60
    last_msg = ""
    while time.time() < deadline:
        t = requests.get(f"{BASE}/api/train/tasks/{task_id}", timeout=10).json()
        msg = t.get("message") or ""
        if msg != last_msg:
            print(
                f"  [{time.strftime('%H:%M:%S')}] status={t['status']} "
                f"progress={t.get('progress')} {msg}",
                flush=True,
            )
            last_msg = msg
        if t["status"] in ("done", "failed", "stopped", "interrupted"):
            break
        time.sleep(10)

    t = requests.get(f"{BASE}/api/train/tasks/{task_id}", timeout=10).json()
    print("最终状态: " + str(t), flush=True)
    if t["status"] != "done":
        print("训练未成功完成，终止后续步骤", flush=True)
        sys.exit(1)

    # ---------- 3. 导出 ONNX ----------
    log("[3/5] 导出 ONNX")
    print(requests.post(f"{BASE}/api/model/export/ner", timeout=600).text, flush=True)

    # ---------- 4. 重建实体白名单 ----------
    log("[4/5] 重建实体白名单")
    from src.ner.postprocess import build_entity_vocab, save_entity_vocab
    from src.web.data_prep_service import NER_OUT

    vocab = build_entity_vocab(NER_OUT, min_count=2)
    save_entity_vocab(vocab, "data/ner/annotated/entity_vocab.json")
    singles = sorted(w for w in vocab if len(w) == 1)
    print(f"实体表 {len(vocab)} 条，单字实体 {len(singles)} 个: {singles}", flush=True)

    # ---------- 5. 验证 ----------
    log("[5/5] 验证")
    from collections import Counter

    from src.ner.inference import NERInference
    from src.web import db

    eng = NERInference("configs/ner.yaml")

    print("\n--- badcase 验证 ---", flush=True)
    for name, t_, d_, c_ in [
        ("用户badcase", "小米11 双十一大促", "小米11 双十一大促", "这个活动哪里有"),
        ("开箱+求购壳", "iPhone 17 开箱", "iPhone 17 开箱 用了半个月的感受", "壳有吗 我想要个手机壳"),
        ("售后吐槽", "小米空调售后", "小米空调售后是真行 又是被小米服务惊艳到了", "线上买的还是线下买的？"),
        ("单字合法实体", "", "这双鞋真的很好穿 推荐买", "什么牌子的鞋"),
        ("多实体家电", "", "格力的空调和海尔的冰箱哪个值得买", "求推荐洗衣机"),
        ("无实体闲聊", "", "今天天气真好", "是啊"),
    ]:
        ents = eng.predict_note_comment(t_, d_, c_)
        print(f"  {name}: {[(e['text'], e['type'], e['source'], e['prob']) for e in ents]}", flush=True)

    print("\n--- 400 条真实样本量化评估 ---", flush=True)
    # 注意：这里不再自己算指标。旧实现把「真值」按 set 去重、却把「预测」按
    # (标题/正文/评论) 分段重复计数，导致命中数 > 真值数（R=1.46 这种 >1 的值）。
    # 统一走 scripts/eval_ner_ab.py 的 note 级去重口径，指标恒在 [0,1]。
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_ner_ab import evaluate, load_entity_vocab, report

    with open("configs/ner.yaml", "r", encoding="utf-8") as f:
        _vocab_file = (yaml.safe_load(f).get("data") or {}).get("entity_vocab_file")
    vocab = load_entity_vocab(_vocab_file)
    samples = db.fetch_ner_samples(400)
    report("重训后(400条真实样本)", evaluate(eng, samples, vocab), 0)
    print("\n  提示：与旧模型对比请跑 "
          "`python scripts/eval_ner_ab.py --limit 400 --ab`", flush=True)

    print("\n重训流水线全部完成", flush=True)


if __name__ == "__main__":
    main()
