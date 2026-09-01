"""导出 NER 模型为 ONNX。

用法：
    python -m src.ner.export_onnx --config configs/ner.yaml
"""
from __future__ import annotations

import argparse

import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.common.onnx_utils import export_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ner.yaml")
    parser.add_argument("--model_dir", default=None, help="训练产物目录，默认取 config 的 output_dir")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = args.model_dir or cfg["train"]["output_dir"]
    num_labels = len(cfg["model"]["labels"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)

    export_model(
        model,
        tokenizer,
        onnx_path=cfg["export"]["onnx_path"],
        opset=cfg["export"]["opset"],
        max_length=cfg["model"]["max_length"],
        quantize=cfg["export"]["quantize"],
    )


if __name__ == "__main__":
    main()
