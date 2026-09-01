"""NER 解码后处理：严格 BIO 分段 + 段内类型多数票 + 词典/规则过滤。

## 为什么需要这个模块（由 badcase 驱动）

输入「小米11 双十一大促」（标题与正文相同）时，原解码输出了
`一`(PRODUCT)、`小`(BRAND)、`米11`(PRODUCT) 三个碎片实体。
token 级诊断后确认根因在**解码逻辑**，不在训练数据（训练数据里
单字实体仅占 0.0%~0.7%，且没有 "一"/"小" 这类标注）：

1. **孤立 I- 标签被当成新实体起点**
   "双十一大促" 的 "一" 被预测为 `I-PRODUCT:0.53 vs O:0.44`（极勉强），
   前一个 token "十" 是 O —— 这是非法 BIO 转移。原代码却直接
   `cur_type = t; cur_start = off_start`，把它开成了一个单字实体。

2. **实体内部类型冲突被机械切分**
   "小"=B-BRAND(0.49)、"米"=I-PRODUCT(0.56)、"11"=I-PRODUCT(0.65)，
   原代码遇到 I-PRODUCT 与 cur_type=BRAND 不匹配就 flush，
   把完整的 "小米11" 切成了 "小"(BRAND) + "米11"(PRODUCT)。
   改为**段内多数票**：BRAND 1 票 vs PRODUCT 2 票 → 整段判为 PRODUCT。

3. **缺少噪声过滤**
   单字、纯数字、不在词典中的碎片直接输出。项目已有 1.5 万条词典
   （brand/category/product/model），可用于校验。

因此解码改为「先分段 → 段内投票定类型 → 再过滤」三步。
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Set

# 段落标记字符：实体不允许跨越【标题】/【正文】/【评论】边界
BOUNDARY_CHARS = "【】"

# 纯数字或纯符号（不含任何中英文字符）
_NON_WORD_RE = re.compile(r"^[\d\W_]+$", re.UNICODE)

# 词典文件名 -> 实体类型
LEXICON_FILES = {
    "BRAND": "brand.txt",
    "CATEGORY": "category.txt",
    "PRODUCT": "product.txt",
    "MODEL": "model.txt",
}


def load_lexicon(lexicon_dir: Optional[str]) -> Dict[str, Set[str]]:
    """加载词典目录，返回 {实体类型: 词条集合}。目录不存在时返回空集合。"""
    lex: Dict[str, Set[str]] = {t: set() for t in LEXICON_FILES}
    if not lexicon_dir or not os.path.isdir(lexicon_dir):
        return lex
    for etype, fname in LEXICON_FILES.items():
        path = os.path.join(lexicon_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    lex[etype].add(w)
    return lex


def build_entity_vocab(conll_path: str, min_count: int = 1) -> Set[str]:
    """从训练 CONLL 标注中提取出现过的实体文本，用于单字实体白名单。

    为什么需要：单字实体不能一刀切丢弃。ES 真值里确实存在合法的单字品类
    （"鞋" 出现 1629 次、"酒" 202 次、"床" 56 次），而 badcase 里的
    "一"、"小" 在训练数据中从未作为实体出现过。用「模型见过的实体集合」
    做白名单，可以精准区分二者。
    """
    vocab: Counter = Counter()
    cur_type: Optional[str] = None
    cur_chars: List[str] = []

    with open(conll_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_type:
                    vocab["".join(cur_chars)] += 1
                cur_type, cur_chars = None, []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ch, tag = parts[0], parts[1]
            if tag.startswith("B-"):
                if cur_type:
                    vocab["".join(cur_chars)] += 1
                cur_type, cur_chars = tag[2:], [ch]
            elif tag.startswith("I-") and cur_type == tag[2:]:
                cur_chars.append(ch)
            else:
                if cur_type:
                    vocab["".join(cur_chars)] += 1
                cur_type, cur_chars = None, []
    if cur_type:
        vocab["".join(cur_chars)] += 1

    return {w for w, c in vocab.items() if c >= min_count and w}


def save_entity_vocab(vocab: Set[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(vocab), f, ensure_ascii=False)


def load_entity_vocab(path: Optional[str]) -> Set[str]:
    """加载实体白名单缓存文件；不存在时返回空集（降级为仅词典校验）。"""
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


class NERPostProcessor:
    """把 token 级预测解码为实体，并过滤碎片噪声。

    参数：
        id2label: label id -> BIO 标签
        lexicon_dir: 词典目录（None/不存在时自动跳过词典校验）
        min_confidence: 实体平均置信度下限，0 表示不过滤
        isolated_i_threshold: 孤立 I- 标签被当作 B- 开新段的概率门槛
            （默认 0.9，即只有非常高置信才允许，其余按噪声丢弃）
        enable_lexicon_filter: 是否启用词典校验
        min_entity_len: 实体最小字符数（去重空格后）
        drop_numeric_only: 是否丢弃未命中词典的纯数字/纯符号实体
    """

    def __init__(
        self,
        id2label: Dict[int, str],
        lexicon_dir: Optional[str] = None,
        min_confidence: float = 0.0,
        isolated_i_threshold: float = 0.9,
        enable_lexicon_filter: bool = True,
        min_entity_len: int = 1,
        drop_numeric_only: bool = True,
        entity_vocab: Optional[Set[str]] = None,
    ):
        self.id2label = id2label
        self.min_confidence = float(min_confidence or 0.0)
        self.isolated_i_threshold = float(isolated_i_threshold)
        self.enable_lexicon_filter = bool(enable_lexicon_filter)
        self.min_entity_len = int(min_entity_len)
        self.drop_numeric_only = bool(drop_numeric_only)
        # 训练数据中真实出现过的实体集合（单字实体的白名单）
        self.entity_vocab = entity_vocab or set()

        self.lexicon = load_lexicon(lexicon_dir) if self.enable_lexicon_filter else {}
        self._lex_any: Set[str] = set()
        for words in self.lexicon.values():
            self._lex_any |= words
        self._lex_lower = {w.lower() for w in self._lex_any}

    # ---------- 解码 ----------

    def decode(
        self,
        text: str,
        offsets,
        pred_ids,
        probs=None,
        seq_len: Optional[int] = None,
    ) -> List[dict]:
        """从 token 级预测解码实体（严格 BIO + 段内多数票）。

        offsets:  (L, 2) 字符偏移
        pred_ids: (L,)   预测 label id
        probs:    (L,)   每个 token 预测标签的概率，None 时视为 1.0
        seq_len:  有效长度（含 CLS/SEP），None 时用 pred_ids 全长
        """
        n = len(pred_ids) if seq_len is None else int(seq_len)
        entities: List[dict] = []
        seg: Optional[dict] = None

        def flush():
            nonlocal seg
            if seg is None:
                return
            # 段内类型由多数票决定：解决 B-BRAND + I-PRODUCT 被切碎的问题
            etype = seg["votes"].most_common(1)[0][0]
            s, e = seg["start"], seg["end"]
            if e > s:
                plist = seg["probs"]
                avg = sum(plist) / len(plist) if plist else 1.0
                entities.append({
                    "type": etype,
                    "start": s,
                    "end": e,
                    "text": text[s:e],
                    "prob": round(float(avg), 4),
                })
            seg = None

        for i in range(1, n - 1):
            s, e = int(offsets[i][0]), int(offsets[i][1])
            if e <= s:  # 特殊 token，无对应字符
                continue
            ch = text[s:e]
            # 段落标记：强制断开，防止实体跨段
            if any(c in BOUNDARY_CHARS for c in ch):
                flush()
                continue

            tag = self.id2label[int(pred_ids[i])]
            p = float(probs[i]) if probs is not None else 1.0

            if tag == "O":
                flush()
                continue

            kind, etype = tag[0], tag[2:]
            if kind == "B":
                flush()
                seg = {"votes": Counter({etype: 1}), "start": s, "end": e, "probs": [p]}
            elif kind == "I":
                if seg is None:
                    # 孤立 I-：只有置信度足够高才当作 B- 开新段，否则按噪声丢弃
                    if p >= self.isolated_i_threshold:
                        seg = {"votes": Counter({etype: 1}), "start": s, "end": e, "probs": [p]}
                    continue
                # 类型冲突不切分，只记票；最终类型在 flush 时按多数票决定
                seg["votes"][etype] += 1
                seg["end"] = e
                seg["probs"].append(p)

        flush()
        return entities

    # ---------- 过滤 ----------

    def _in_lexicon(self, word: str) -> bool:
        """是否命中词典。未启用词典或词典为空时返回 True（避免误杀）。"""
        if not self.enable_lexicon_filter or not self._lex_any:
            return True
        return word in self._lex_any or word.lower() in self._lex_lower

    def is_valid(self, ent: dict) -> bool:
        """判断一个候选实体是否为有效实体（过滤碎片噪声）。"""
        s = (ent.get("text") or "").strip()
        if not s:
            return False
        if len(s) < self.min_entity_len:
            return False
        if any(c in BOUNDARY_CHARS for c in s):
            return False
        if self.min_confidence > 0 and ent.get("prob", 1.0) < self.min_confidence:
            return False

        in_lex = self._in_lexicon(s)
        # 单字实体必须命中词典，或在训练数据中作为实体出现过。
        # "鞋"/"酒"/"床" 等是 ES 真值里的合法单字品类（训练数据中共约 2000 次），
        # 不能一刀切丢弃；而 "一"/"小" 从未作为实体出现过，属于碎片。
        if len(s) <= 1 and not (in_lex or s in self.entity_vocab):
            return False
        # 纯数字/纯符号（如 "11"）必须命中词典，否则丢弃
        if self.drop_numeric_only and not in_lex and _NON_WORD_RE.match(s):
            return False
        return True

    def filter(self, entities: List[dict]) -> List[dict]:
        return [e for e in entities if self.is_valid(e)]

    # ---------- 一步到位 ----------

    def decode_and_filter(
        self,
        text: str,
        offsets,
        pred_ids,
        probs=None,
        seq_len: Optional[int] = None,
        return_filtered: bool = True,
    ) -> List[dict]:
        ents = self.decode(text, offsets, pred_ids, probs, seq_len)
        if not return_filtered:
            return ents
        return self.filter(ents)
