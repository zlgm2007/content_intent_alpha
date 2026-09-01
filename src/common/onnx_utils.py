"""ONNX 导出与加载工具。

- export_model：把 PyTorch 模型导出为 ONNX（动态 batch / 序列长度）。
- quantize_onnx：可选 int8 动态量化。
- OnnxRunner：onnxruntime 推理封装。
"""
from __future__ import annotations

import os
from typing import List

import numpy as np


def export_model(
    model,
    tokenizer,
    onnx_path: str,
    opset: int = 14,
    dynamic_axes: bool = True,
    max_length: int = 128,
    quantize: bool = False,
):
    """导出 HuggingFace 分类/序列标注模型为 ONNX。

    输入：input_ids (int64), attention_mask (int64)
    输出：logits (float32)
    """
    import torch

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model.eval()

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

    dynamic_axes_dict = None
    input_names = ["input_ids", "attention_mask"]
    if dynamic_axes:
        dynamic_axes_dict = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,   # 走 legacy 导出，权重内联为单文件 onnx（避免 external data）
        )
    print(f"[onnx] 已导出: {onnx_path}")

    if quantize:
        quantize_onnx(onnx_path, onnx_path.replace(".onnx", ".quant.onnx"))
    return onnx_path


def quantize_onnx(src_path: str, dst_path: str):
    """int8 动态量化，进一步压缩模型体积。"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(src_path, dst_path, weight_type=QuantType.QInt8)
    print(f"[onnx] 已量化: {dst_path}")


class OnnxRunner:
    """onnxruntime 推理封装。"""

    def __init__(self, onnx_path: str, use_gpu: bool = False):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]

    def run(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """返回 logits。"""
        return self.session.run(None, {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        })[0]
