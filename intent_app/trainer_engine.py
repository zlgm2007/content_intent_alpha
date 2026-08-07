#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""训练引擎 — 后台线程运行 RoBERTa-wwm-ext 微调，支持指标/ETA/控制/导出。

状态机：idle → running → done/stopped/error；ML 依赖延迟导入（仅训练时才 import）。
"""
import json
import os
import time
from collections import deque
from datetime import datetime

from common import MODELS_DIR, NUM_LABELS, PRETRAINED_MODEL


class TrainerEngine:
    """单个训练任务的状态机。一次只跑一个训练。线程安全由调用方(API)保证。"""

    def __init__(self):
        self.goal_id = None
        self.goal_name = ""
        self.state = "idle"          # idle / running / done / stopped / error
        self.reason = ""
        self.thread = None
        self._stop = None            # threading.Event, start 时创建
        self.max_steps = 0
        # 监控数据
        self.metrics = deque(maxlen=5000)   # 每步 {step, epoch, loss, avg_loss, lr, elapsed}
        self.val_points = []                # 每 epoch {step, epoch, acc}
        self.logs = deque(maxlen=500)
        self.progress = {
            "step": 0, "epoch": 0, "total_epochs": 0,
            "loss": 0, "avg_loss": 0, "lr": 0, "elapsed": 0,
            "eta_sec": None, "steps_per_sec": 0,
            "val_acc": None, "val_step": 0,
            "train_samples": 0, "val_samples": 0,
        }
        self.final_acc = None
        self.last_model_dir = None

    # ---- 日志捕获 ----

    def _log(self, msg):
        """简易日志（不依赖 loguru，避免额外依赖）。"""
        ts = datetime.now().strftime("%m-%d %H:%M:%S")
        self.logs.append(f"{ts} {msg}")

    # ---- 状态快照（供 /status 轮询） ----

    def status(self):
        p = dict(self.progress)
        return {
            "state": self.state,
            "reason": self.reason,
            "goal_id": self.goal_id,
            "goal_name": self.goal_name,
            "max_steps": self.max_steps,
            "progress": p,
            "metrics": list(self.metrics)[-300:],
            "val_points": self.val_points[-200:],
            "logs": list(self.logs)[-200:],
            "final_acc": self.final_acc,
            "last_model_dir": self.last_model_dir,
        }

    # ---- 训练控制 ----

    def start(self, goal_id, goal_name, hyperparams, label_source="labeled"):
        """启动训练（后台线程）。

        label_source: 'labeled' 用人工标注（手工+外部已标注）；
                      'raw' 用 ES 原始得分 raw_score 作伪标签（只读，不改数据）。
        """
        import threading
        if self.state in ("running",):
            raise ValueError(f"训练状态为 {self.state}，无法重复开始")
        if label_source not in ("labeled", "raw"):
            raise ValueError("label_source 必须为 labeled 或 raw")
        self.goal_id = goal_id
        self.goal_name = goal_name
        self.state = "running"
        self.reason = "训练中"
        self._stop = threading.Event()
        self.metrics.clear()
        self.val_points.clear()
        self.logs.clear()
        self.final_acc = None
        self.last_model_dir = None
        self.thread = threading.Thread(
            target=self._run, args=(goal_id, goal_name, hyperparams, label_source),
            daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """优雅停训。"""
        if self.state == "running":
            self.reason = "正在停止..."
            self._stop.set()
            return True
        raise ValueError(f"当前状态 {self.state} 无法停止")

    # ---- 训练主循环 ----

    def _run(self, goal_id, goal_name, hyperparams, label_source="labeled"):
        """训练线程主体。延迟导入所有 ML 依赖。"""
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                      get_linear_schedule_with_warmup)
            from torch.optim import AdamW
        except ImportError as e:
            self.state = "error"
            self.reason = f"缺少 ML 依赖: {e}. 请安装: pip install torch transformers optimum onnxruntime"
            self._log(f"ERROR: {self.reason}")
            return

        import db

        # ---- 超参数 ----
        lr = float(hyperparams.get("learning_rate", 2e-5))
        batch_size = int(hyperparams.get("batch_size", 16))
        epochs = int(hyperparams.get("epochs", 3))
        max_length = int(hyperparams.get("max_length", 256))
        warmup_ratio = float(hyperparams.get("warmup_ratio", 0.1))
        weight_decay = float(hyperparams.get("weight_decay", 0.01))
        val_ratio = float(hyperparams.get("val_ratio", 0.2))

        try:
            # ---- 1. 加载标注数据 ----
            # label_source='raw'：只读 ES 原始得分 raw_score 作伪标签，不改数据
            if label_source == "raw":
                self._log("正在加载 ES 原始得分(raw_score)作为训练标签...")
                rows = db.query_all("""
                    SELECT a.comment AS comment, a.raw_score AS score,
                           COALESCE(w.content, w.note_title, '') AS content_text
                    FROM annotations a JOIN works w ON a.work_id = w.id
                    WHERE a.goal_id = ? AND a.raw_score IS NOT NULL
                    ORDER BY a.id
                """, (goal_id,))
            else:
                self._log("正在加载标注数据（人工标注，手工+外部合并）...")
                rows = db.query_all("""
                    SELECT * FROM (
                        SELECT cm.comment AS comment, cm.score AS score,
                               ct.text AS content_text
                        FROM comments cm JOIN contents ct ON cm.content_id = ct.id
                        WHERE cm.goal_id = ? AND cm.status = 'labeled'
                        UNION ALL
                        SELECT COALESCE(a.comment, '') AS comment, a.score AS score,
                               COALESCE(w.content, w.note_title, '') AS content_text
                        FROM annotations a JOIN works w ON a.work_id = w.id
                        WHERE a.goal_id = ? AND a.status = 'labeled'
                    ) ORDER BY comment
                """, (goal_id, goal_id))

            if len(rows) < 20:
                raise ValueError(f"标注数据不足: {len(rows)} 条，至少需要 20 条")

            # 检查类别分布
            score_counts = {}
            for r in rows:
                s = r["score"]
                score_counts[s] = score_counts.get(s, 0) + 1
            if len(score_counts) < 2:
                raise ValueError(f"标注数据只有 {len(score_counts)} 种分数，至少需要 2 种")

            src_label = "ES原始得分(raw_score)" if label_source == "raw" else "人工标注"
            self._log(f"训练标签来源: {src_label}, 数据: {len(rows)} 条, "
                      f"分数分布: {score_counts}")

            # ---- 2. 切分训练/验证 ----
            import random
            random.seed(42)
            random.shuffle(rows)
            val_size = max(1, int(len(rows) * val_ratio))
            val_data = rows[:val_size]
            train_data = rows[val_size:]

            self.progress["train_samples"] = len(train_data)
            self.progress["val_samples"] = len(val_data)
            self.progress["total_epochs"] = epochs

            self._log(f"训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")

            # ---- 3. 加载 tokenizer + model ----
            self._log(f"正在加载预训练模型: {PRETRAINED_MODEL}...")
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(
                PRETRAINED_MODEL, num_labels=NUM_LABELS)

            # ---- 4. 创建 Dataset + DataLoader ----
            train_ds = _IntentDataset(train_data, tokenizer, max_length)
            val_ds = _IntentDataset(val_data, tokenizer, max_length)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # ---- 5. 优化器 + 调度器 ----
            device = torch.device("mps" if torch.backends.mps.is_available()
                                  else "cuda" if torch.cuda.is_available()
                                  else "cpu")
            model = model.to(device)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped = [
                {"params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                 "weight_decay": weight_decay},
                {"params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped, lr=lr)
            total_steps = len(train_loader) * epochs
            warmup_steps = int(total_steps * warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)
            self.max_steps = total_steps

            self._log(f"设备: {device}, 总步数: {total_steps}, 预热: {warmup_steps}")

            # ---- 6. 训练循环 ----
            start_time = time.time()
            global_step = 0
            best_val_acc = 0.0
            best_model_state = None

            for epoch in range(epochs):
                if self._stop.is_set():
                    break
                model.train()
                epoch_loss = 0.0
                epoch_steps = 0

                for batch in train_loader:
                    if self._stop.is_set():
                        break

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch["token_type_ids"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    global_step += 1
                    epoch_steps += 1
                    epoch_loss += loss.item()

                    elapsed = time.time() - start_time
                    sps = global_step / max(0.001, elapsed)
                    remaining = max(0, total_steps - global_step)
                    eta = round(remaining / sps, 1) if remaining > 0 else 0

                    self.progress.update({
                        "step": global_step, "epoch": epoch + 1,
                        "loss": round(loss.item(), 4),
                        "avg_loss": round(epoch_loss / epoch_steps, 4),
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": round(elapsed, 1),
                        "eta_sec": eta,
                        "steps_per_sec": round(sps, 3),
                    })
                    self.metrics.append({
                        "step": global_step, "epoch": epoch + 1,
                        "loss": round(loss.item(), 4),
                        "avg_loss": round(epoch_loss / epoch_steps, 4),
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed": round(elapsed, 1),
                    })

                    if global_step % 50 == 0:
                        self._log(
                            f"epoch {epoch+1}/{epochs} step {global_step}/{total_steps} "
                            f"loss {self.progress['avg_loss']:.4f} "
                            f"lr {self.progress['lr']:.2e}")

                # ---- Epoch 结束，验证 ----
                if self._stop.is_set():
                    break
                val_acc = self._evaluate(model, val_loader, device)
                self.val_points.append({
                    "step": global_step, "epoch": epoch + 1, "acc": round(val_acc, 4)})
                self.progress["val_acc"] = round(val_acc, 4)
                self.progress["val_step"] = global_step
                self._log(f"epoch {epoch+1}/{epochs} 验证准确率: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}
                    self._log(f"  -> 新最佳模型 (acc={best_val_acc:.4f})")

            # ---- 7. 训练结束 ----
            if self._stop.is_set():
                self.state = "stopped"
                self.reason = "用户手动停止"
            else:
                self.state = "done"
                self.reason = f"训练完成, 最佳验证准确率: {best_val_acc:.4f}"

            self.final_acc = round(best_val_acc, 4)

            # ---- 8. 导出 ONNX + 量化 ----
            if best_model_state:
                self._log("正在导出 ONNX 模型...")
                model.load_state_dict(best_model_state)
                model.eval()
                model_dir = self._export_onnx(
                    model, tokenizer, goal_id, goal_name,
                    best_val_acc, hyperparams)
                self.last_model_dir = model_dir
                self._log(f"ONNX 模型已保存: {model_dir}")

                # 保存到数据库
                rel_path = os.path.relpath(model_dir)
                num_samples = len(train_data) + len(val_data)
                params_json = json.dumps({
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "max_length": max_length,
                    "warmup_ratio": warmup_ratio,
                    "weight_decay": weight_decay,
                    "val_ratio": val_ratio,
                    "pretrained_model": PRETRAINED_MODEL,
                    "label_source": label_source,
                }, ensure_ascii=False)
                # 可读命名：{来源}_{epochs}ep_acc_{acc 4位小数}，来源 raw=ES原始得分 / manual=人工标注
                src = "manual" if label_source == "labeled" else "raw"
                model_name = f"{src}_{epochs}ep_acc_{best_val_acc:.4f}"
                model_id = db.execute("""
                    INSERT INTO models (goal_id, name, onnx_path, tokenizer_dir, accuracy,
                                        num_train_samples, status, params)
                    VALUES (?, ?, ?, ?, ?, ?, 'trained', ?)
                """, (goal_id, model_name,
                      os.path.join(rel_path, "model_int8.onnx"),
                      rel_path,
                      best_val_acc,
                      num_samples,
                      params_json))
                self._log(f"模型记录已保存到数据库 (id={model_id}, name={model_name})")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.state = "error"
            self.reason = f"训练出错: {e}"
            self._log(f"ERROR: {e}")

    def _evaluate(self, model, val_loader, device):
        """在验证集上评估准确率。"""
        import torch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        model.train()
        return correct / max(1, total)

    def _export_onnx(self, model, tokenizer, goal_id, goal_name, acc, hyperparams):
        """导出 ONNX 模型 + INT8 量化。"""
        import torch

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_"
                            for c in goal_name)
        model_dir = os.path.join(MODELS_DIR, f"goal_{goal_id}_{safe_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # 保存 tokenizer
        tokenizer.save_pretrained(model_dir)

        # 导出 ONNX (FP32)
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model, export=True, provider="CPUExecutionProvider")
            ort_model.save_pretrained(model_dir)
        except Exception:
            # fallback: 手动导出
            self._log("Optimum 导出失败，使用手动 torch.onnx 导出...")
            self._manual_export_onnx(model, tokenizer, model_dir)

        onnx_path = os.path.join(model_dir, "model.onnx")
        int8_path = os.path.join(model_dir, "model_int8.onnx")

        # INT8 动态量化（torch 导出图带冲突 value_info，先清空；per_channel 质量优于 per_tensor）
        if os.path.isfile(onnx_path):
            try:
                import onnx
                from onnxruntime.quantization import quantize_dynamic, QuantType
                m = onnx.load(onnx_path)
                del m.graph.value_info[:]
                clean = onnx_path + ".clean"
                onnx.save(m, clean, save_as_external_data=True,
                          all_tensors_to_one_file=True,
                          location=os.path.basename(clean) + ".data")
                quantize_dynamic(clean, int8_path, weight_type=QuantType.QInt8, per_channel=True,
                                 extra_options={"MatMulConstBOnly": False})
                os.remove(clean); os.remove(clean + ".data")
                self._log("INT8 量化完成")
            except Exception as e:
                self._log(f"INT8 量化失败，使用 FP32: {e}")
                import shutil; shutil.copy2(onnx_path, int8_path)

        # 保存训练配置
        config = {
            "goal_id": goal_id,
            "goal_name": goal_name,
            "accuracy": acc,
            "pretrained_model": PRETRAINED_MODEL,
            "num_labels": NUM_LABELS,
            "hyperparams": hyperparams,
            "exported_at": timestamp,
        }
        with open(os.path.join(model_dir, "training_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return model_dir

    def _manual_export_onnx(self, model, tokenizer, model_dir):
        """手动导出 ONNX（fallback）。"""
        import torch
        model.eval()
        model.cpu()
        # 创建 dummy input
        dummy = tokenizer("测试内容", "测试评论", max_length=256,
                          truncation=True, padding="max_length", return_tensors="pt")
        onnx_path = os.path.join(model_dir, "model.onnx")
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )


class _IntentDataset:
    """PyTorch Dataset for intent classification (sentence pair)."""

    def __init__(self, data, tokenizer, max_length=256):
        # data: list of dict with content_text, comment, score
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch
        row = self.data[idx]
        content = row["content_text"] or ""
        comment = row["comment"] or ""
        score = row["score"]
        encoding = self.tokenizer(
            content, comment,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(score, dtype=torch.long),
        }
