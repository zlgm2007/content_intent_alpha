"""NER 模型训练：基于 albert_chinese_small 微调 token 分类。

用法：
    python -m src.ner.train --config configs/ner.yaml
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
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.common.dataset import NERDataset, read_conll
from src.common.device import get_device
from src.common.metrics import bio_to_entities, compute_ner_f1, compute_per_type_f1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg, tokenizer, label2id):
    tokens_list, tags_list = read_conll(cfg["data"]["annotated_file"])

    # 有独立验证集则使用，否则按比例切分
    val_file = cfg["data"].get("val_file")
    if val_file and os.path.exists(val_file):
        val_tokens, val_tags = read_conll(val_file)
        train_tokens, train_tags = tokens_list, tags_list
    else:
        n = len(tokens_list)
        split = int(n * cfg["data"]["train_split"])
        idx = list(range(n))
        random.shuffle(idx)
        train_tokens = [tokens_list[i] for i in idx[:split]]
        train_tags = [tags_list[i] for i in idx[:split]]
        val_tokens = [tokens_list[i] for i in idx[split:]]
        val_tags = [tags_list[i] for i in idx[split:]]

    train_ds = NERDataset(train_tokens, train_tags, tokenizer, label2id, cfg["model"]["max_length"])
    val_ds = NERDataset(val_tokens, val_tags, tokenizer, label2id, cfg["model"]["max_length"])
    return train_ds, val_ds


def evaluate(model, dataloader, id2label, device):
    model.eval()
    all_pred_entities = []
    all_gold_entities = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)

            for i in range(input_ids.size(0)):
                valid_mask = labels[i] != -100
                p_ids = preds[i][valid_mask].cpu().tolist()
                g_ids = labels[i][valid_mask].cpu().tolist()
                p_tags = [id2label[x] for x in p_ids]
                g_tags = [id2label[x] for x in g_ids]
                all_pred_entities.extend(bio_to_entities(p_tags, ""))
                all_gold_entities.extend(bio_to_entities(g_tags, ""))

    p, r, f1 = compute_ner_f1(all_pred_entities, all_gold_entities)
    per_type = compute_per_type_f1(all_pred_entities, all_gold_entities)
    return p, r, f1, per_type


def train(cfg, progress_callback: Optional[Callable[[float, str, Optional[float]], None]] = None, stop_event=None):
    set_seed(cfg["train"]["seed"])
    device = get_device()

    labels = cfg["model"]["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"]["pretrained"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_ds, val_ds = build_dataset(cfg, tokenizer, label2id)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["train"]["learning_rate"],
        total_steps=total_steps, pct_start=cfg["train"]["warmup_ratio"],
    )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    best_f1 = 0.0
    total_epochs = cfg["train"]["epochs"]
    total_batches = len(train_loader)
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

        p, r, f1, per_type = evaluate(model, val_loader, id2label, device)
        print(f"[epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} "
              f"P={p:.4f} R={r:.4f} F1={f1:.4f}")
        print(f"  分类型 F1: {per_type}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(cfg["train"]["output_dir"])
            tokenizer.save_pretrained(cfg["train"]["output_dir"])
            print(f"  已保存最优模型 (F1={best_f1:.4f}) -> {cfg['train']['output_dir']}")

        if progress_callback:
            progress_callback(
                (epoch + 1) / total_epochs,
                f"epoch {epoch+1}/{total_epochs} F1={f1:.4f} P={p:.4f} R={r:.4f}",
            )

    print(f"训练完成，最优 F1={best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ner.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
