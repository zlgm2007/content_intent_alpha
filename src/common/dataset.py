"""数据加载与 token 对齐。

- NERDataset：读 CONLL 格式标注，做 token 级标签对齐（处理 BERT 子词切分）。
- IntentDataset：读 JSONL 意图标注，做文本分类。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """序列标注数据集。

    输入为字符级 tokens/tags 列表，例如：
        tokens = ["我", "想", "买", "小", "米", "1", "4"]
        tags   = ["O", "O", "O", "B-BRAND", "I-BRAND", "B-MODEL", "I-MODEL"]

    内部把整句交给 tokenizer 做子词切分，用 offset_mapping 把字符级标签
    对齐到子词 token 上（与推理端保持一致，正确处理英文/数字子词）。
    """

    def __init__(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
        tokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.ignore_label_id = -100
        self.samples = self._align(tokens_list, tags_list)

    @staticmethod
    def _pick_label(seg_tags: List[str]) -> str:
        """从子词覆盖的字符标签区间中挑一个代表标签：B- 优先，其次非 O，否则 O。"""
        for t in seg_tags:
            if t.startswith("B-"):
                return t
        for t in seg_tags:
            if t != "O":
                return t
        return "O"

    def _align(self, tokens_list, tags_list):
        """整句 tokenize + offset_mapping 对齐字符级标签到子词 token。"""
        aligned = []
        for tokens, tags in zip(tokens_list, tags_list):
            text = "".join(tokens)
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            offsets = enc["offset_mapping"]

            label_ids = []
            for sid, (s, e) in zip(input_ids, offsets):
                if s == e:  # padding / CLS / SEP 等特殊 token
                    label_ids.append(self.ignore_label_id)
                    continue
                seg = tags[s:e]
                if not seg:
                    label_ids.append(self.ignore_label_id)
                    continue
                label = self._pick_label(seg)
                label_ids.append(self.label2id.get(label, self.label2id["O"]))

            aligned.append((input_ids, attention_mask, label_ids))
        return aligned

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attention_mask, label_ids = self.samples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class IntentDataset(Dataset):
    """意图分类数据集。labels 为 0/1。"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """读取 CONLL 格式标注文件（每行 `token\\tlabel`，空行分隔句子）。

    也支持 BIOES/BIO 标签。返回 (tokens_list, tags_list)。
    """
    tokens_list: List[List[str]] = []
    tags_list: List[List[str]] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_tokens:
                    tokens_list.append(cur_tokens)
                    tags_list.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cur_tokens.append(parts[0])
            cur_tags.append(parts[1])
    if cur_tokens:
        tokens_list.append(cur_tokens)
        tags_list.append(cur_tags)
    return tokens_list, tags_list


def read_intent_jsonl(path: str) -> Tuple[List[str], List[int]]:
    """读取意图标注文件（JSONL），每行 {"text": str, "label": 0/1}。"""
    import json

    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(int(obj["label"]))
    return texts, labels
