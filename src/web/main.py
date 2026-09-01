"""content_intent_alpha Web 管理台后端入口。

启动：
    uvicorn src.web.main:app --host 0.0.0.0 --port 8000

若 frontend/dist 已构建，将自动挂载前端页面（访问 http://localhost:8000 即可）。
开发期也可用 `cd frontend && npm run dev` 单独起前端（默认 5173 端口，已配代理）。
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.web import db
from src.web.config import BASE_DIR
from src.web.routers import data, es, lexicon, model, predict, system, train

app = FastAPI(title="content_intent_alpha 管理台", version="1.0.0")

# 开发期允许前端跨域访问（Vite dev server）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
db.init_db()
# 进程重启后，把遗留的 running 训练任务标记为中断（避免僵尸任务）
db.mark_interrupted_running_tasks()

app.include_router(system.router)
app.include_router(es.router)
app.include_router(lexicon.router)
app.include_router(data.router)
app.include_router(train.router)
app.include_router(model.router)
app.include_router(predict.router)


@app.get("/api")
def api_root():
    return {"app": "content_intent_alpha 管理台", "docs": "/docs", "health": "/api/health"}


# 前端静态资源（若已 build）
DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(os.path.join(DIST_DIR, "index.html")):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(DIST_DIR, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("docs"):
            raise HTTPException(404)
        file = os.path.join(DIST_DIR, full_path)
        if full_path and os.path.isfile(file):
            return FileResponse(file)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
