"""正文 + 评论 拼接工具。

业务背景：内容分为「正文」（笔记描述）与「评论」（用户评论）两部分，
购买意图主要体现在评论中，但评论需要结合正文上下文才能理解（例如评论
"壳有吗" 只有结合正文 "iPhone 17 开箱" 才能判断出购买意图）。

因此模型输入统一拼接为（连续拼接，不使用换行，避免换行符被 tokenizer 吞掉
导致 NER 字符对齐错位）：
    【标题】{title}【正文】{desc}【评论】{comment}

训练与推理共用同一拼接格式与截断长度，保证特征一致性。
"""
from __future__ import annotations

from typing import Optional

# 各字段截断长度（字符级；评论是意图判断核心，优先保留，正文作为上下文可截断）
TITLE_LIMIT = 40
DESC_LIMIT = 150
COMMENT_LIMIT = 140

# 段落标记（推理层据此给实体标注来源段落）
MARK_TITLE = "【标题】"
MARK_DESC = "【正文】"
MARK_COMMENT = "【评论】"

# 标记 -> 中文标签（用于实体来源展示）
MARK_LABELS = [
    (MARK_TITLE, "标题"),
    (MARK_DESC, "正文"),
    (MARK_COMMENT, "评论"),
]


def _trim(s: Optional[str], limit: int) -> str:
    s = (s or "").strip()
    return s[:limit] if len(s) > limit else s


def compose_note_comment(
    title: str = "",
    desc: str = "",
    comment: str = "",
    title_limit: int = TITLE_LIMIT,
    desc_limit: int = DESC_LIMIT,
    comment_limit: int = COMMENT_LIMIT,
) -> str:
    """把标题/正文/评论拼接为统一输入文本。空字段自动省略。"""
    parts: list = []
    if title:
        parts.append(MARK_TITLE + _trim(title, title_limit))
    if desc:
        parts.append(MARK_DESC + _trim(desc, desc_limit))
    if comment:
        parts.append(MARK_COMMENT + _trim(comment, comment_limit))
    return "".join(parts)


def locate_source(text: str, pos: int) -> str:
    """根据字符位置判断落在哪一段（标题/正文/评论）。"""
    marks = sorted((text.find(mark), label) for mark, label in MARK_LABELS)
    source = "正文"
    for idx, label in marks:
        if idx >= 0 and pos >= idx:
            source = label
        elif idx > pos:
            break
    return source
