#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""ONNX 推理引擎 — 加载量化后的 ONNX 模型进行意图评分推理。

ML 依赖（onnxruntime / transformers / numpy）均为延迟导入。
"""
import os


class IntentInferencer:
    """加载 ONNX 模型 + tokenizer，进行意图分类推理。"""

    def __init__(self, model_dir, max_length=256):
        """
        Args:
            model_dir: 模型目录（包含 model_int8.onnx + tokenizer 文件）
            max_length: token 最大长度
        """
        self.model_dir = model_dir
        self.max_length = max_length
        self._session = None
        self._tokenizer = None
        self._input_names = None

    def _ensure_loaded(self):
        """延迟加载模型和 tokenizer。"""
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer
        import numpy as np

        # 优先使用 INT8 量化模型
        onnx_path = os.path.join(self.model_dir, "model_int8.onnx")
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.join(self.model_dir, "model.onnx")
        if not os.path.isfile(onnx_path):
            raise ValueError(f"ONNX 模型不存在: {onnx_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._np = np

    def predict(self, content, comment):
        """单条推理。

        Returns:
            dict: {
                "score": int (0~max_score, 由模型分类数决定),
                "confidence": float,
                "probabilities": list[float] (长度=num_labels)
            }
        """
        self._ensure_loaded()
        np = self._np

        encoding = self._tokenizer(
            content, comment,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np")

        feed = {}
        for name in self._input_names:
            if name in encoding:
                feed[name] = encoding[name]
            elif name == "token_type_ids" and "token_type_ids" in encoding:
                feed[name] = encoding["token_type_ids"]

        outputs = self._session.run(None, feed)
        logits = outputs[0]  # shape: (1, num_labels)

        # softmax
        logits = logits[0].astype(np.float64)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        score = int(np.argmax(probs))
        confidence = float(probs[score])

        return {
            "score": score,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
        }

    def predict_batch(self, pairs, batch_size=32):
        """批量推理。

        Args:
            pairs: list of (content, comment)
            batch_size: 批大小

        Returns:
            list of dict (同 predict 返回格式)
        """
        self._ensure_loaded()
        np = self._np
        results = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            contents = [p[0] for p in batch]
            comments = [p[1] for p in batch]

            encoding = self._tokenizer(
                contents, comments,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="np")

            feed = {}
            for name in self._input_names:
                if name in encoding:
                    feed[name] = encoding[name]

            outputs = self._session.run(None, feed)
            logits = outputs[0].astype(np.float64)  # (batch, num_labels)

            for row in logits:
                exp = np.exp(row - np.max(row))
                probs = exp / np.sum(exp)
                score = int(np.argmax(probs))
                confidence = float(probs[score])
                results.append({
                    "score": score,
                    "confidence": round(confidence, 4),
                    "probabilities": [round(float(p), 4) for p in probs],
                })

        return results
