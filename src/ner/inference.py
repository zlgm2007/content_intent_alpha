"""NER 推理：加载 ONNX 模型 + tokenizer，输入文本输出实体列表。

支持两种输入方式：
- 纯文本：predict(text)
- 正文+评论：predict_note_comment(title, desc, comment)，拼接后推理，
  实体附带 source 字段标注来源段落（标题/正文/评论）。

对齐说明：使用 tokenizer 的 offset_mapping 做字符级精确对齐，
正确处理英文/数字子词（如 iPhone -> iphone + ##17）。

解码后处理：见 src/ner/postprocess.py。模型输出的 token 级预测会经过
「严格 BIO 分段 → 段内类型多数票 → 词典/规则过滤」三步，用于消除
"一"、"小"、"米11" 这类碎片实体（详见 postprocess 模块文档）。

用法：
    python -m src.ner.inference --config configs/ner.yaml --text "我想买小米14 Pro 手机"
    python -m src.ner.inference --config configs/ner.yaml --desc "iPhone 17 开箱" --comment "壳有吗"
    python -m src.ner.inference --config configs/ner.yaml --desc "..." --raw   # 查看过滤前的原始解码
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml
from transformers import AutoTokenizer

from src.common.compose import compose_note_comment, locate_source
from src.common.onnx_utils import OnnxRunner
from src.ner.postprocess import NERPostProcessor, load_entity_vocab


class NERInference:
    def __init__(self, config_path: str, onnx_path: str = None, model_dir: str = None):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.max_length = cfg["model"]["max_length"]
        self.id2label = {i: l for i, l in enumerate(cfg["model"]["labels"])}
        self.label2id = {l: i for i, l in enumerate(cfg["model"]["labels"])}

        model_dir = model_dir or cfg["train"]["output_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.runner = OnnxRunner(onnx_path or cfg["export"]["onnx_path"])

        # 解码后处理（严格 BIO + 段内多数票 + 词典过滤）
        inf_cfg = cfg.get("inference") or {}
        data_cfg = cfg.get("data") or {}
        vocab_file = inf_cfg.get("entity_vocab_file") or data_cfg.get("entity_vocab_file")
        self.post = NERPostProcessor(
            id2label=self.id2label,
            lexicon_dir=inf_cfg.get("lexicon_dir") or data_cfg.get("lexicon_dir"),
            min_confidence=inf_cfg.get("min_confidence", 0.0),
            isolated_i_threshold=inf_cfg.get("isolated_i_threshold", 0.9),
            enable_lexicon_filter=inf_cfg.get("enable_lexicon_filter", True),
            min_entity_len=inf_cfg.get("min_entity_len", 1),
            drop_numeric_only=inf_cfg.get("drop_numeric_only", True),
            entity_vocab=load_entity_vocab(vocab_file),
        )

    def predict(self, text: str, raw: bool = False) -> list:
        """识别文本中的实体。返回 [{type, start, end, text, prob}, ...]，偏移为字符级。

        raw=True 时跳过过滤，返回模型原始解码结果（用于排查 badcase）。
        """
        if not text or not text.strip():
            return []
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offsets = enc["offset_mapping"][0]  # (L, 2)
        logits = self.runner.run(enc["input_ids"], enc["attention_mask"])  # (1, L, num_labels)
        pred_ids = np.argmax(logits, axis=-1)[0]  # (L,)
        seq_len = int(enc["attention_mask"][0].sum())

        # 每个 token 预测标签的 softmax 概率（供孤立 I- 判定与实体置信度使用）
        lg = logits[0]  # (L, num_labels)
        exp = np.exp(lg - lg.max(axis=-1, keepdims=True))
        probs = (exp / exp.sum(axis=-1, keepdims=True)).max(axis=-1)  # (L,)

        return self.post.decode_and_filter(
            text, offsets, pred_ids, probs, seq_len, return_filtered=not raw
        )

    def predict_note_comment(
        self, title: str = "", desc: str = "", comment: str = "", raw: bool = False
    ) -> list:
        """正文+评论 联合识别，实体附带 source 字段（标题/正文/评论）。"""
        text = compose_note_comment(title, desc, comment)
        entities = self.predict(text, raw=raw)
        for e in entities:
            e["source"] = locate_source(text, e["start"])
        return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ner.yaml")
    parser.add_argument("--text", default=None)
    parser.add_argument("--title", default="")
    parser.add_argument("--desc", default="")
    parser.add_argument("--comment", default="")
    parser.add_argument("--raw", action="store_true", help="输出过滤前的原始解码结果（排查用）")
    args = parser.parse_args()

    engine = NERInference(args.config)
    if args.text:
        ents = engine.predict(args.text, raw=args.raw)
    else:
        ents = engine.predict_note_comment(args.title, args.desc, args.comment, raw=args.raw)
    print(json.dumps(ents, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
