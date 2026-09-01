"""NER 重训收尾流水线：等训练完成 → 导出 ONNX → 重建白名单 → badcase 验证 → 双轨评测。

为什么需要这个脚本（而不是只跑 scripts/retrain_ner.py）
-------------------------------------------------------
`retrain_ner.py` 是**客户端轮询**模式：训练实际在服务端（uvicorn 后台线程）跑。
客户端进程一旦被杀（沙箱超时、终端关闭、后台任务被清理），**训练仍会继续**，
但收尾三步 —— 导出 ONNX、重建实体白名单、badcase 验证 —— 就永远不会执行。

2026-09-01 实测踩了这个坑：客户端 12:21 启动训练后即死，日志停在 393 字节再无更新，
但服务端训练一直跑到 13:0x。若只盯任务状态会误以为「训练完就完事了」，
结果线上用的还是旧 ONNX。

判据：**不要靠 pgrep/ps 判断进程存活**（本沙箱禁用进程枚举，pgrep 静默返回空）。
只看服务端任务状态 + ONNX mtime 是否真的变了。

本脚本幂等，可重复执行：
  - 训练已完成 → 直接进入收尾
  - ONNX 已导出 → 仍会重新导出（确保与最新 checkpoint 一致）
  - 白名单每次从 train.conll 重建（换了标注标准后必须重跑，否则用旧标准过滤新模型输出）

用法：
    python scripts/finish_ner_pipeline.py              # 等待 + 收尾 + 评测
    python scripts/finish_ner_pipeline.py --no-wait    # 训练已完成，直接收尾
    python scripts/finish_ner_pipeline.py --task-id 24 # 指定要等的任务
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import requests
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://localhost:8000"
PY = sys.executable


def log(msg: str):
    print(f"\n===== {msg} =====", flush=True)


def poll_task(task_id: int, timeout_min: int = 150) -> dict:
    log(f"等待训练任务 {task_id} 完成")
    deadline = time.time() + timeout_min * 60
    last = ""
    while time.time() < deadline:
        t = requests.get(f"{BASE}/api/train/tasks/{task_id}", timeout=10).json()
        msg = t.get("message") or ""
        if msg != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {t['status']} | {msg}", flush=True)
            last = msg
        if t["status"] in ("done", "failed", "stopped", "interrupted"):
            return t
        time.sleep(10)
    raise TimeoutError(f"任务 {task_id} 在 {timeout_min} 分钟内未结束")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=24)
    ap.add_argument("--no-wait", action="store_true", help="训练已完成，跳过等待")
    args = ap.parse_args()

    # ---------- 1. 等训练 ----------
    if not args.no_wait:
        t = poll_task(args.task_id)
        print("最终状态: " + str(t), flush=True)
        if t["status"] != "done":
            print("训练未成功完成，终止收尾流程", flush=True)
            return 1

    # ---------- 2. 导出 ONNX ----------
    log("[1/4] 导出 ONNX")
    before = os.path.getmtime("models/ner/ner.onnx")
    r = requests.post(f"{BASE}/api/model/export/ner", timeout=900)
    print(r.text[:500], flush=True)
    after = os.path.getmtime("models/ner/ner.onnx")
    if after <= before:
        print("⚠️ ONNX mtime 未变化，导出可能未生效，终止", flush=True)
        return 1
    print(f"ONNX 已更新 (mtime {time.strftime('%H:%M:%S', time.localtime(after))})", flush=True)

    # ---------- 3. 重建实体白名单 ----------
    log("[2/4] 重建实体白名单（必须从新的 train.conll 重建）")
    from src.ner.postprocess import build_entity_vocab, save_entity_vocab
    from src.web.data_prep_service import NER_OUT

    with open("configs/ner.yaml", "r", encoding="utf-8") as f:
        vocab_file = (yaml.safe_load(f).get("data") or {}).get("entity_vocab_file")
    vocab = build_entity_vocab(NER_OUT, min_count=2)
    save_entity_vocab(vocab, vocab_file)
    singles = sorted(w for w in vocab if len(w) == 1)
    print(f"实体表 {len(vocab)} 条，单字实体 {len(singles)} 个: {singles}", flush=True)

    # ---------- 4. badcase 验证 ----------
    log("[3/4] badcase 验证")
    from src.ner.inference import NERInference

    eng = NERInference("configs/ner.yaml")
    cases = [
        ("用户badcase", "小米11 双十一大促", "小米11 双十一大促", "这个活动哪里有"),
        ("开箱+求购壳", "iPhone 17 开箱", "iPhone 17 开箱 用了半个月的感受", "壳有吗 我想要个手机壳"),
        ("售后吐槽", "小米空调售后", "小米空调售后是真行 又是被小米服务惊艳到了", "线上买的还是线下买的？"),
        ("单字合法实体", "", "这双鞋真的很好穿 推荐买", "什么牌子的鞋"),
        ("多实体家电", "", "格力的空调和海尔的冰箱哪个值得买", "求推荐洗衣机"),
        ("无实体闲聊", "", "今天天气真好", "是啊"),
    ]
    for name, ti, de, co in cases:
        ents = eng.predict_note_comment(ti, de, co)
        print(f"  {name}: {[(e['text'], e['type'], e['source']) for e in ents]}", flush=True)

    # ---------- 5. 双轨评测 ----------
    log("[4/4] 双轨评测")
    for tag, lim in (("2000", "2000"), ("full", "0")):
        out = f"/tmp/ner_eval_new_{tag}.log"
        print(f"\n--- 留出集 {tag} ---", flush=True)
        with open(out, "w", encoding="utf-8") as fh:
            subprocess.run(
                [PY, "scripts/eval_ner_dual.py", "--limit", lim,
                 "--show", "5" if tag == "2000" else "0"],
                stdout=fh, stderr=subprocess.STDOUT, check=False,
            )
        with open(out, encoding="utf-8") as fh:
            print(fh.read(), flush=True)

    print("\n收尾流水线全部完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
