"""导出意图模型为 ONNX。

用法：
    python -m src.intent.export_onnx --config configs/intent.yaml
"""
from __future__ import annotations

import argparse

import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.onnx_utils import export_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    parser.add_argument("--model_dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir = args.model_dir or cfg["train"]["output_dir"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=cfg["model"]["num_labels"]
    )

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
