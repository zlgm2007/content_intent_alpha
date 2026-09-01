"""意图推理：加载 ONNX 模型，直接输出 0–5 分（6 分类）。

支持两种输入方式：
- 纯文本：predict(text)
- 正文+评论：predict_note_comment(title, desc, comment)，内部拼接后推理

用法：
    python -m src.intent.inference --config configs/intent.yaml --text "我想买一台手机"
    python -m src.intent.inference --config configs/intent.yaml --desc "iPhone 17 开箱" --comment "壳有吗"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml
from transformers import AutoTokenizer

from src.common.compose import compose_note_comment
from src.common.onnx_utils import OnnxRunner

# 分数 -> 语义标签
LEVEL_MAP = {
    0: "无意图", 1: "无意图", 2: "无意图",
    3: "有倾向", 4: "中度意图", 5: "强烈意图",
}


class IntentInference:
    def __init__(self, config_path: str, onnx_path: str = None, model_dir: str = None):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.max_length = cfg["model"]["max_length"]
        self.intent_name = cfg["model"]["intent_name"]
        self.num_labels = cfg["model"]["num_labels"]

        # 先验偏移修正：训练集做过类别平衡，与真实分布不一致，需在 logits 上补一个偏置。
        # 详见 configs/intent.yaml 的 inference 段。alpha=0 或文件缺失时自动关闭，不影响原行为。
        inf_cfg = cfg.get("inference") or {}
        self.prior_alpha = float(inf_cfg.get("prior_correction_alpha") or 0.0)
        self.prior_bias = None
        if self.prior_alpha > 0:
            pf = inf_cfg.get("prior_file")
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    b = json.load(f).get("bias")
                if b and len(b) == self.num_labels:
                    self.prior_bias = np.array(b, dtype=np.float64)
                else:
                    print(f"[intent] 先验文件维度不符({len(b) if b else 0} != {self.num_labels})，关闭修正")
            except (OSError, ValueError, TypeError) as e:
                print(f"[intent] 未加载先验文件({pf}): {e}，关闭先验修正")
                self.prior_bias = None

        model_dir = model_dir or cfg["train"]["output_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.runner = OnnxRunner(onnx_path or cfg["export"]["onnx_path"])

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        logits = self.runner.run(enc["input_ids"], enc["attention_mask"])  # (1, num_labels)
        lg = logits[0].astype(np.float64)
        # 先验偏移修正（训练集类别平衡 → 真实分布），见 configs/intent.yaml
        if self.prior_bias is not None:
            lg = lg + self.prior_alpha * self.prior_bias
        # 数值稳定 softmax：必须先减去最大值。直接 np.exp(logits) 在 logits 较大时
        # （float32 约 88 以上即 inf）会溢出成 inf/inf = NaN，prob 字段静默变成 nan。
        # 与 src/ner/inference.py 的写法保持一致。
        exp = np.exp(lg - lg.max())
        probs = exp / exp.sum()
        score = int(np.argmax(lg, axis=-1))
        return {
            "name": self.intent_name,
            "score": score,
            "prob": round(float(probs[score]), 4),
            "level": LEVEL_MAP.get(score, "未知"),
        }

    def predict_note_comment(self, title: str = "", desc: str = "", comment: str = "") -> dict:
        """正文+评论 联合推理：拼接后打分。"""
        return self.predict(compose_note_comment(title, desc, comment))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--desc", default="")
    parser.add_argument("--comment", default="")
    args = parser.parse_args()

    engine = IntentInference(args.config)
    if args.text:
        print(engine.predict(args.text))
    else:
        print(engine.predict_note_comment(args.title, args.desc, args.comment))


if __name__ == "__main__":
    main()
