"""意图模型训练：6 分类（0–5 分直接打分）。

用 intentScore 真值训练，模型直接输出 0–5 分，无需概率阈值映射。

用法：
    python -m src.intent.train --config configs/intent.yaml
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.dataset import IntentDataset, read_intent_jsonl
from src.common.device import get_device
from src.common.metrics import compute_accuracy, per_class_metrics, score_hit_rate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg, tokenizer):
    texts, labels = read_intent_jsonl(cfg["data"]["labeled_file"])

    val_file = cfg["data"].get("val_file")
    if val_file and os.path.exists(val_file):
        val_texts, val_labels = read_intent_jsonl(val_file)
        train_texts, train_labels = texts, labels
    else:
        n = len(texts)
        split = int(n * cfg["data"]["train_split"])
        idx = list(range(n))
        random.shuffle(idx)
        train_texts = [texts[i] for i in idx[:split]]
        train_labels = [labels[i] for i in idx[:split]]
        val_texts = [texts[i] for i in idx[split:]]
        val_labels = [labels[i] for i in idx[split:]]

    train_ds = IntentDataset(train_texts, train_labels, tokenizer, cfg["model"]["max_length"])
    val_ds = IntentDataset(val_texts, val_labels, tokenizer, cfg["model"]["max_length"])
    return train_ds, val_ds


def predict(model, dataloader, device):
    """返回验证集的 (preds, labels)，preds 为 0–5 分。"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            p = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(p)
            labels.extend(batch["labels"].cpu().tolist())
    return np.array(preds), np.array(labels)


def train(cfg, progress_callback: Optional[Callable[[float, str, Optional[float]], None]] = None, stop_event=None):
    set_seed(cfg["train"]["seed"])
    device = get_device()

    num_labels = cfg["model"]["num_labels"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["pretrained"], num_labels=num_labels
    ).to(device)

    train_ds, val_ds = build_datasets(cfg, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["train"]["learning_rate"],
        total_steps=total_steps, pct_start=cfg["train"]["warmup_ratio"],
    )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    total_epochs = cfg["train"]["epochs"]
    total_batches = len(train_loader)
    best_acc = 0.0
    for epoch in range(total_epochs):
        if stop_event is not None and stop_event.is_set():
            break
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            if stop_event is not None and stop_event.is_set():
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if progress_callback and (batch_idx + 1) % 10 == 0:
                prog = (epoch * total_batches + batch_idx + 1) / (total_epochs * total_batches)
                progress_callback(
                    prog,
                    f"epoch {epoch+1}/{total_epochs} batch {batch_idx+1}/{total_batches} loss={loss.item():.4f}",
                    loss.item(),
                )

        if stop_event is not None and stop_event.is_set():
            break

        preds, y_true = predict(model, val_loader, device)
        acc = compute_accuracy(preds, y_true)
        hit = score_hit_rate(preds, y_true, tolerance=1)
        pcm = per_class_metrics(preds, y_true)
        msg = (f"[epoch {epoch+1}/{total_epochs}] loss={total_loss/len(train_loader):.4f} "
               f"acc={acc:.4f} 分数命中率(±1)={hit:.4f}")
        print(msg)
        print(f"  per-class f1: {pcm}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(cfg["train"]["output_dir"])
            tokenizer.save_pretrained(cfg["train"]["output_dir"])
            print(f"  已保存最优模型 (acc={best_acc:.4f})")

        if progress_callback:
            progress_callback(
                (epoch + 1) / total_epochs,
                f"epoch {epoch+1}/{total_epochs} acc={acc:.4f} 命中率={hit:.4f}",
            )

    print(f"训练完成，最优 acc={best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
