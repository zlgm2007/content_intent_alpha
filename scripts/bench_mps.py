"""MPS vs CPU 训练吞吐基准（不落盘、不保存模型）。

用真实 intent 数据 + 与 train.py 相同的模型/优化器，跑若干 step 计时，
输出 CPU / MPS 的 it/s 与加速比，用于估算完整训练时长。
"""
from __future__ import annotations

import time
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.dataset import IntentDataset, read_intent_jsonl

N_STEPS = 30  # MPS 计时步数
N_CPU = 8     # CPU 计时步数（CPU 慢，少跑几步）


def build(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained"])
    texts, labels = read_intent_jsonl(cfg["data"]["labeled_file"])
    n = len(texts)
    ds = IntentDataset(texts, labels, tokenizer, cfg["model"]["max_length"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["pretrained"], num_labels=cfg["model"]["num_labels"]
    )
    return model, loader


def bench(device, model, loader, n_steps):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.00002, weight_decay=0.01)
    it = iter(loader)
    # 预热 2 步（含首次 kernel 编译）
    for _ in range(2):
        b = next(it)
        input_ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        lab = b["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=mask, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        b = next(it)
        input_ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        lab = b["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=mask, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return n_steps / dt


def main():
    cfg = yaml.safe_load(open("configs/intent.yaml"))
    model, loader = build(cfg)
    n = len(loader.dataset)

    # CPU
    cpu_ips = bench(torch.device("cpu"), model, loader, N_CPU)
    # MPS
    if torch.backends.mps.is_available():
        mps_ips = bench(torch.device("mps"), model, loader, N_STEPS)
        speedup = mps_ips / cpu_ips
    else:
        mps_ips = None
        speedup = 0.0

    total_batches = n / cfg["train"]["batch_size"]
    per_epoch_batches = total_batches  # 近似
    print("=" * 50)
    print(f"样本数={n}  batch_size={cfg['train']['batch_size']}  max_length={cfg['model']['max_length']}")
    print(f"每 epoch 批次数≈{per_epoch_batches:.0f}")
    print(f"CPU  it/s = {cpu_ips:.2f}")
    if mps_ips:
        print(f"MPS  it/s = {mps_ips:.2f}   加速比 = {speedup:.2f}x")
        cpu_epoch = per_epoch_batches / cpu_ips
        mps_epoch = per_epoch_batches / mps_ips
        print(f"单 epoch：CPU≈{cpu_epoch/60:.1f}min  MPS≈{mps_epoch/60:.1f}min")
        print(f"3 epoch：CPU≈{3*cpu_epoch/60:.1f}min  MPS≈{3*mps_epoch/60:.1f}min")
    print("=" * 50)


if __name__ == "__main__":
    main()
