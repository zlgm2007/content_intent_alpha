#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""便捷启动脚本 — python run.py [--port 8800]"""
import os
import sys

# 把 intent_app 加入 sys.path
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "intent_app"))
sys.path.insert(0, REPO)

from intent_app.server import main

if __name__ == "__main__":
    main()
