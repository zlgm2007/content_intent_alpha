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

# 仓库根 = intent_app/common.py 的上两级
REPO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

# 主项目关键目录
DATA_DIR = os.path.join(REPO_DIR, "intent_data")       # 数据根
DB_PATH = os.path.join(DATA_DIR, "intent.db")          # SQLite 数据库文件
MODELS_DIR = os.path.join(REPO_DIR, "models")           # ONNX 模型输出目录

# 预训练模型标识
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
NUM_LABELS = 6  # 0-5 分，6 分类

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
