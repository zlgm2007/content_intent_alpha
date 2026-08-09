#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""intent_app 公共常量与路径工具.

集中定义仓库根、数据/模型目录，以及安全路径拼接。
所有模块统一从这里取路径，避免各模块各自推算目录导致不一致。
"""
import os
import sys
import json

# 仓库根 = intent_app/common.py 的上两级
REPO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

# 主项目关键目录
DATA_DIR = os.path.join(REPO_DIR, "intent_data")       # 数据根
DB_PATH = os.path.join(DATA_DIR, "intent.db")          # SQLite 数据库文件
MODELS_DIR = os.path.join(REPO_DIR, "models")           # ONNX 模型输出目录

# 预训练模型标识
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
NUM_LABELS = 6  # 默认 0-5 分；实际 num_labels 由各 goal 的 score_definitions 决定

# 默认分数定义（score_definitions 为 NULL 时使用）
DEFAULT_SCORE_DEFINITIONS = [
    {"score": 0, "label": "无关", "desc": "评论与意图完全无关"},
    {"score": 1, "label": "弱关联", "desc": "提到相关话题但无意图"},
    {"score": 2, "label": "有关注", "desc": "对相关内容有兴趣但未明确表达"},
    {"score": 3, "label": "有倾向", "desc": "有一定意图倾向"},
    {"score": 4, "label": "明确意向", "desc": "明确表达意图"},
    {"score": 5, "label": "强烈意向", "desc": "意图非常强烈"},
]
DEFAULT_THRESHOLD = 3  # 默认阈值：>=3 为有意图


def get_goal_score_config(goal_id):
    """获取某意图目标的分数配置。

    Returns:
        dict: {
            "max_score": int,          # 最大分数值
            "num_labels": int,         # 分类数 = max_score + 1
            "threshold": int,          # 意图阈值，>= 此值为有意图
            "score_definitions": list, # 分数定义列表 [{score, label, desc}]
        }
    """
    import db
    goal = db.query_one(
        "SELECT score_definitions, intent_threshold FROM intent_goals WHERE id = ?",
        (goal_id,))
    if not goal:
        return {"max_score": 5, "num_labels": 6,
                "threshold": DEFAULT_THRESHOLD,
                "score_definitions": DEFAULT_SCORE_DEFINITIONS}

    defs = None
    raw = goal.get("score_definitions")
    if raw:
        try:
            defs = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

    if defs and isinstance(defs, list) and len(defs) >= 2:
        max_score = max(d["score"] for d in defs if isinstance(d.get("score"), int))
        num_labels = max_score + 1
    else:
        defs = DEFAULT_SCORE_DEFINITIONS
        max_score = 5
        num_labels = 6

    threshold = goal.get("intent_threshold")
    if threshold is None:
        threshold = (max_score + 1) // 2  # ceil(max_score / 2)

    return {"max_score": max_score, "num_labels": num_labels,
            "threshold": threshold, "score_definitions": defs}

# 外部平台数据同步（Elasticsearch）
ES_BASE_URL = os.environ.get("ES_BASE_URL", "http://10.104.214.120:9200")
ES_INDICES = {"ge3": "lead_clue_score_ge3", "lt3": "lead_clue_score_lt3"}
ES_DEFAULT_DAYS = 1        # 按天数同步：默认最近 N 天
ES_MIN_DAYS = 1            # 最少 1 天
ES_MAX_DAYS = 30           # 最多 30 天
ES_DEFAULT_LIMIT = 10000   # 按条数同步：默认最近 N 条
ES_MIN_LIMIT = 1           # 最少 1 条
ES_MAX_LIMIT = 100000      # 最多 10 万条
ES_SCROLL_KEEP = "1m"      # scroll 上下文保留时间
ES_PAGE_SIZE = 1000        # 每次 scroll 拉取条数


def ensure_dirs():
    """确保关键目录存在。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def ensure_sys_paths():
    """把仓库根加入 sys.path，供跨目录 import。"""
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)


def safe_join(root, *parts):
    """把 parts 安全拼接到 root 下，防路径穿越；非法时抛 ValueError。"""
    root_abs = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root_abs, *parts))
    if p != root_abs and not p.startswith(root_abs + os.sep):
        raise ValueError("非法路径")
    return p
