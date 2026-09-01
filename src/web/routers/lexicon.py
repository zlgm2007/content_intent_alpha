"""词典管理 API：四词典的增删查。"""
from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.web.config import DATA_DIR

router = APIRouter(prefix="/api/lexicon", tags=["lexicon"])

LEXICON_DIR = os.path.join(DATA_DIR, "ner", "lexicon")
TYPES = {"brand", "category", "product", "model"}


def _path(ltype: str) -> str:
    return os.path.join(LEXICON_DIR, f"{ltype}.txt")


def _read(ltype: str) -> List[str]:
    if ltype not in TYPES:
        raise HTTPException(400, f"未知词典类型: {ltype}")
    p = _path(ltype)
    if not os.path.exists(p):
        return []
    words = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return words


def _write(ltype: str, words: List[str]):
    p = _path(ltype)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    words = sorted(set(w.strip() for w in words if w and w.strip()))
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# {ltype} 词典\n")
        for w in words:
            f.write(w + "\n")


class AddWords(BaseModel):
    words: List[str]


@router.get("")
def list_lexicon():
    return {t: _read(t) for t in sorted(TYPES)}


@router.get("/{ltype}")
def get_lexicon(ltype: str):
    return {"type": ltype, "words": _read(ltype)}


@router.post("/{ltype}")
def add_words(ltype: str, body: AddWords):
    if ltype not in TYPES:
        raise HTTPException(400, f"未知词典类型: {ltype}")
    merged = _read(ltype) + body.words
    _write(ltype, merged)
    return {"type": ltype, "count": len(_read(ltype))}


@router.delete("/{ltype}")
def remove_word(ltype: str, word: str):
    if ltype not in TYPES:
        raise HTTPException(400, f"未知词典类型: {ltype}")
    words = [w for w in _read(ltype) if w != word]
    _write(ltype, words)
    return {"type": ltype, "count": len(words)}
