# content_intent_alpha

内容意图识别工具 — 对用户查看内容后的评论进行意图分类打分（0-5分）。

## 解决什么问题

用户设定意图目标（如"购物意向"），输入作品内容和用户评论，工具对每条评论计算意图强度：

- **5分**：意图强烈（如"求链接！马上买！"）
- **3-4分**：存在意图（如"多少钱？有货吗"）
- **0-2分**：无意图/无价值内容（如"哈哈哈"、"第一个"）

评论的意图依赖作品上下文。例如"多少钱"在购物视频下是购物意向，在搞笑视频下可能只是调侃。因此模型采用**句对分类**：`[CLS] 作品内容 [SEP] 评论 [SEP]`。

## 核心功能

| 功能 | 说明 |
|------|------|
| 意图目标管理 | 创建/管理意图目标（购物意向、咨询意向、吐槽等），每个意图独立训练模型 |
| 数据标注 | 录入作品内容 + 评论，对评论打 0-5 分，支持批量导入 |
| 模型训练 | RoBERTa-wwm-ext 微调训练，后台线程执行，自动导出 ONNX + INT8 量化 |
| 数据批跑 | 加载 ONNX 模型对已标注数据批量推理，输出识别率报告 |

## 技术方案

### 模型

- **预训练基座**：RoBERTa-wwm-ext（HFL，`hfl/chinese-roberta-wwm-ext`）
- **任务**：句对分类（sentence-pair classification），6分类（0-5分）
- **导出**：Hugging Face Optimum → ONNX → INT8 动态量化
- **推理**：ONNX Runtime（CPU，~15ms/条，INT8 模型 ~100MB）

### 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| 前端 | 纯 HTML + CSS + Vanilla JS | 零构建、零依赖、iframe 标签页切换 |
| 后端 | Python stdlib `http.server` | ThreadingHTTPServer 单端口分发，零框架依赖 |
| ML 训练 | PyTorch + Transformers | Hugging Face 生态 |
| 模型导出 | Optimum + ONNX Runtime | 一键导出、量化、跨平台推理 |
| 数据存储 | SQLite (Python stdlib sqlite3) | 零配置、本地文件、百万级数据 |
| 异步任务 | threading.Thread (daemon) | 训练任务后台线程 + 状态机 |

> Web 层架构参考 captcha_alpha/captcha_app，不使用 Vue / FastAPI / 任何 web 框架。

### 分词说明

不需要手动分词（不需要 jieba）。RoBERTa-wwm-ext 自带 tokenizer，直接传原始文本：

```python
inputs = tokenizer("作品内容", "评论内容", max_length=256, truncation=True, padding="max_length")
```

"全词掩码"(wwm) 是预训练阶段的策略，推理时不涉及。tokenizer 自动完成：加 [CLS]/[SEP]、中文字符切分、映射 ID、生成 mask、截断填充。

## 工程目录结构

```
content_intent_alpha/
├── intent_app/                    # 轻量 Web 应用
│   ├── server.py                 # 入口：ThreadingHTTPServer + 路由分发
│   ├── common.py                 # 公共常量：REPO_DIR / safe_join / 路径工具
│   ├── intent_api.py             # 意图目标管理 API -> /intent/api/*
│   ├── labeler_api.py            # 数据标注 API   -> /lbl/api/*
│   ├── trainer_api.py            # 模型训练 API   -> /train/api/*
│   ├── batch_api.py              # 数据批跑 API   -> /batch/api/*
│   ├── trainer_engine.py         # 后台训练引擎（线程 + 状态机）
│   └── static/
│       ├── index.html            # Shell 页面（iframe 标签页切换）
│       ├── intent.html           # 意图目标管理页
│       ├── labeler.html          # 数据标注页
│       ├── trainer.html          # 模型训练页
│       └── batch.html            # 数据批跑页
├── intent_data/                  # 数据目录
│   └── intent.db                 # SQLite 数据库
├── models/                       # 训练产出的 ONNX 模型
├── docs/
│   └── technical_proposal.html   # 技术方案文档
├── requirements.txt
├── LICENSE
└── README.md
```

## API 路由

单端口，URL 前缀分发：

| 前缀 | 模块 | 功能 |
|------|------|------|
| `/intent/api/*` | IntentAPI | 意图目标 CRUD |
| `/lbl/api/*` | LabelerAPI | 作品录入 + 评论标注 |
| `/train/api/*` | TrainerAPI | 训练准备/启动/停止/状态/导出 |
| `/batch/api/*` | BatchAPI | 批量推理 + 结果报告 |
| `/static/*` | 静态文件 | HTML/CSS/JS |

各 API 模块统一接口：`handle_get(path, q)` / `handle_post(path, body)` 返回 `(payload, content_type)`。

## 数据库

SQLite 单文件（`intent_data/intent.db`），5 张表：

| 表 | 说明 | 数据量级 |
|----|------|---------|
| `intent_goals` | 意图目标 | 极小 |
| `contents` | 作品内容 | 上万条 |
| `comments` | 评论（含标注状态） | 百万级 |
| `models` | 模型记录 | 小 |
| `batch_results` | 批跑结果 | 中等 |

百万级数据优化：WAL 模式 + 索引 + 事务批量写入 + 分页查询 + FTS5 全文搜索（可选）。

## 训练参数

| 参数 | 推荐值 |
|------|--------|
| 学习率 | 2e-5 |
| Batch Size | 16-32 |
| Epochs | 3-5 |
| Max Length | 256（作品180 + 评论76 token）|
| Dropout | 0.3 |
| 优化器 | AdamW |
| 学习率调度 | 线性预热 + 衰减 |

最低标注数据量建议：500 条/意图目标（正负各半）。

## 启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 服务
python intent_app/server.py --port 8800

# 浏览器打开
# http://127.0.0.1:8800
```

## 实施路线

1. **工程搭建 + 意图目标管理** — server 骨架 + SQLite 初始化 + 意图管理页
2. **数据标注模块** — 作品录入 + 评论标注界面 + 批量导入
3. **模型训练引擎** — RoBERTa 微调 + 后台线程 + ONNX 导出
4. **批跑验证模块** — ONNX 推理 + 识别率报告
5. **实时推理 + 优化** — 实时评分 + 模型切换

## License

MIT
