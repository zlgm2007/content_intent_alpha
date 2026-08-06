#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""内容意图工作台 — 单端口 Web 应用入口.

集成四个功能，顶部菜单切换（前端 shell 用 iframe 内嵌）：
  意图目标管理(IntentAPI)  -> /intent/api/*
  数据标注(LabelerAPI)      -> /lbl/api/*
  模型训练(TrainerAPI)      -> /train/api/*
  数据批跑(BatchAPI)        -> /batch/api/*
  静态页面                  -> / 与 /static/*

用法: python intent_app/server.py [--port 8800]
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common import ensure_sys_paths, ensure_dirs

ensure_sys_paths()
ensure_dirs()

from db import init_db  # noqa: E402
from intent_api import IntentAPI  # noqa: E402
from labeler_api import LabelerAPI  # noqa: E402
from batch_api import BatchAPI  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 50 * 1024 * 1024  # 50MB，批量导入评论数据


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 控制台安静

    # ---- 发送 / 读取 ----

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("请求体过大或为空")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    # ---- 静态文件 ----

    def _serve_static(self, rel):
        root = os.path.abspath(STATIC_DIR)
        path = os.path.abspath(os.path.join(root, rel))
        if path != root and not path.startswith(root + os.sep):
            raise ValueError("非法路径")
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ctype = "text/html; charset=utf-8"
        if rel.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif rel.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        elif rel.endswith(".png"):
            ctype = "image/png"
        elif rel.endswith(".svg"):
            ctype = "image/svg+xml"
        with open(path, "rb") as f:
            self._send_bytes(f.read(), ctype)

    # ---- 按命名空间分发 ----

    def _dispatch(self, ns, sub, q, body=None, is_post=False):
        api = getattr(self.server, ns, None)
        if api is None:
            self._send_json({"error": "not found"}, 404)
            return
        if is_post:
            payload, ctype = api.handle_post(sub, body)
        else:
            payload, ctype = api.handle_get(sub, q)
        if ctype is None:
            self._send_json(payload)
        else:
            self._send_bytes(payload, ctype)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path.startswith("/intent/api/"):
                self._dispatch("intent", path[len("/intent/api/"):], q)
            elif path.startswith("/lbl/api/"):
                self._dispatch("lbl", path[len("/lbl/api/"):], q)
            elif path.startswith("/train/api/"):
                self._dispatch("trainer", path[len("/train/api/"):], q)
            elif path.startswith("/batch/api/"):
                self._dispatch("batch", path[len("/batch/api/"):], q)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path.startswith("/intent/api/"):
                self._dispatch("intent", path[len("/intent/api/"):], None, body, True)
            elif path.startswith("/lbl/api/"):
                self._dispatch("lbl", path[len("/lbl/api/"):], None, body, True)
            elif path.startswith("/train/api/"):
                self._dispatch("trainer", path[len("/train/api/"):], None, body, True)
            elif path.startswith("/batch/api/"):
                self._dispatch("batch", path[len("/batch/api/"):], None, body, True)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="内容意图工作台(意图管理 / 标注 / 训练 / 批跑)")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    init_db()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.intent = IntentAPI()
    server.lbl = LabelerAPI()
    server.batch = BatchAPI()
    try:
        from trainer_api import TrainerAPI
        server.trainer = TrainerAPI()
    except Exception as e:
        print(f"[警告] 训练模块暂不可用: {e}")
        server.trainer = None

    url = f"http://127.0.0.1:{args.port}"
    print("=" * 56)
    print("  内容意图工作台 已启动")
    print(f"  打开浏览器: {url}")
    print("  功能: 意图管理 / 数据标注 / 模型训练 / 数据批跑")
    print("  按 Ctrl+C 停止")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    main()
