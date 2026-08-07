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
| 意图目标管理 | 创建/管理意图目标（购物意向、咨询意向、吐槽等），每个意图独立统计与训练；列表展示作品/评论/已标注合并总数 |
| 数据标注 | 作品列表（合并手工录入 + 外部同步，来源徽标区分）、快速标注（合并手工+外部未标注评论，0-5 一键打分/跳过）、批量导入 JSON |
| 外部数据同步 | 从 Elasticsearch 线索索引拉取外部平台作品 + 评论 + 原始意图得分（`raw_score`），按天数（1-30）/ 按条数（1-10 万）拉取，进度/停止/历史 |
| 自动标注 | 基于已训练模型对该目标全部未标注评论（手工+外部）批量推理，置信度 ≥ 阈值写为已标注并标记 `auto_labeled`，低置信度保持未标注 |
| 模型训练 | RoBERTa-wwm-ext 微调，读源合并手工 + 外部已标注数据，后台线程执行，自动导出 ONNX + INT8 量化 |
| 数据批跑 | 加载 ONNX 模型对已标注数据批量推理，输出识别率 / 混淆矩阵 / 二分类指标报告 |

### 标注数据双通道

数据分为两类来源，存储与展示相互隔离，查询合并：

| 来源 | 作品表 | 评论表 | 说明 |
|------|--------|--------|------|
| 手工录入 | `contents` | `comments` | 人工粘贴的作品 + 评论标注 |
| 外部同步 (ES) | `works` | `annotations` | 从 ES 线索索引拉取，`raw_score` 存 ES 原始得分、`score` 存人工标注 |

作品列表、快速标注、顶部统计、意图目标列表均按 `source`（`manual`/`es`）合并展示两类数据；训练与自动标注的读源同样覆盖两表。

## 技术方案

### 模型

- **预训练基座**：RoBERTa-wwm-ext（HFL，`hfl/chinese-roberta-wwm-ext`）
- **任务**：句对分类（sentence-pair classification），6分类（0-5分）
- **导出**：Hugging Face Optimum → ONNX（缺失时自动回退手动 `torch.onnx` 导出）→ INT8 动态量化
- **推理**：ONNX Runtime（CPU，~15ms/条，INT8 模型 ~100MB）

### 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| 前端 | 纯 HTML + CSS + Vanilla JS | 零构建、零依赖、iframe 标签页切换 |
| 后端 | Python stdlib `http.server` | ThreadingHTTPServer 单端口分发，零框架依赖 |
| ES 访问 | Python stdlib `urllib.request` | 零新增依赖，scroll 分页拉取 |
| ML 训练 | PyTorch + Transformers | Hugging Face 生态 |
| 模型导出 | Optimum + ONNX Runtime | 一键导出、量化、跨平台推理 |
| 数据存储 | SQLite (Python stdlib sqlite3) | 零配置、本地文件、百万级数据 |
| 异步任务 | threading.Thread (daemon) | 训练/同步/自动标注 后台线程 + 状态机 |

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
│   ├── server.py                 # 入口：ThreadingHTTPServer + 6 组路由分发
│   ├── common.py                 # 公共常量：路径 / ES 配置 / 模型标识
│   ├── intent_api.py             # 意图目标管理 API -> /intent/api/*
│   ├── labeler_api.py            # 数据标注 API   -> /lbl/api/*
│   ├── trainer_api.py            # 模型训练 API   -> /train/api/*
│   ├── batch_api.py              # 数据批跑 API   -> /batch/api/*
│   ├── es_sync_api.py            # 外部数据同步 API -> /sync/api/*
│   ├── autolabel_api.py          # 自动标注 API   -> /autolabel/api/*
│   ├── trainer_engine.py         # 后台训练引擎（线程 + 状态机）
│   ├── es_sync_engine.py         # 外部数据同步引擎（ES scroll 拉取）
│   ├── autolabel_engine.py       # 自动标注引擎（批量推理写库）
│   ├── inference.py              # ONNX 推理引擎（单条/批量）
│   ├── db.py                     # SQLite 建表 / 迁移 / 连接工厂
│   └── static/
│       ├── index.html            # Shell 页面（iframe 标签页切换）
│       ├── intent.html           # 意图目标管理页
│       ├── labeler.html          # 数据标注页（作品/外部/自动标注/快速/导入）
│       ├── labeler.css           # 标注页样式
│       ├── labeler.js            # 标注页脚本（公共 + 自动标注）
│       ├── labeler_ext.js        # 标注页脚本（外部数据同步）
│       ├── trainer.html          # 模型训练页
│       └── batch.html            # 数据批跑页
├── intent_data/                  # 数据目录
│   └── intent.db                 # SQLite 数据库（WAL）
├── models/                       # 训练产出的 ONNX 模型
├── sql/                          # 数据库变更文档脚本（按分支存放）
├── docs/
│   ├── technical_proposal.html   # 技术方案文档
│   └── devlog/                   # 研发日记（.gitignore，不入库）
├── requirements.txt
├── LICENSE
└── README.md
```

## API 路由

单端口，URL 前缀分发：

| 前缀 | 模块 | 功能 |
|------|------|------|
| `/intent/api/*` | IntentAPI | 意图目标 CRUD + 合并统计 |
| `/lbl/api/*` | LabelerAPI | 作品录入 + 评论标注 + 批量导入 + 快速标注 |
| `/train/api/*` | TrainerAPI | 训练准备/启动/停止/状态/导出/实时推理 |
| `/batch/api/*` | BatchAPI | 批量推理 + 结果报告 |
| `/sync/api/*` | SyncAPI | 外部数据同步（启动/停止/进度/历史/作品/标注） |
| `/autolabel/api/*` | AutoLabelAPI | 自动标注（启动/停止/状态） |
| `/static/*` | 静态文件 | HTML/CSS/JS |

各 API 模块统一接口：`handle_get(path, q)` / `handle_post(path, body)` 返回 `(payload, content_type)`。

## 数据库

SQLite 单文件（`intent_data/intent.db`，WAL 模式），8 张表：

| 表 | 说明 | 数据量级 |
|----|------|---------|
| `intent_goals` | 意图目标 | 极小 |
| `contents` | 手工录入的作品 | 上万条 |
| `comments` | 手工作品的评论（含标注/`auto_labeled`） | 百万级 |
| `works` | ES 同步的外部作品 | 大 |
| `annotations` | ES 同步评论 + 人工标注（`raw_score`/`auto_labeled`） | 百万级 |
| `sync_batches` | 外部数据同步批次历史 | 小 |
| `models` | 模型记录 | 小 |
| `batch_results` | 批跑结果 | 中等 |

百万级数据优化：WAL 模式 + 索引（goal/status/score 高频路径）+ 事务批量写入 + 分页查询 + FTS5 全文搜索（可选）。

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

- **读源**：该目标全部已标注数据（手工 `comments` + 外部 `annotations` 合并）。
- **最低种子**：≥20 条已标注且 ≥2 种分数（代码校验），建议 500 条/意图目标（正负各半）以获得可用精度。
- **自动标注**：6 分类 argmax 置信度，阈值默认 0.8、范围 0.5-1.0；命中写 `status='labeled'` + `auto_labeled=1`，低置信保持 `pending`。

## 启动方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Web 服务
python intent_app/server.py --port 8800

# 3. 浏览器打开
# http://127.0.0.1:8800
```

### 外部数据同步（可选）

需能访问 ES 集群（默认 `http://10.104.214.120:9200`，可用环境变量覆盖）：

```bash
ES_BASE_URL=http://your-es:9200 python intent_app/server.py --port 8800
```

### 预训练模型下载

首次训练会下载 `hfl/chinese-roberta-wwm-ext`（约 400MB）。无法直连 `huggingface.co` 的环境（如国内网络）需使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python intent_app/server.py --port 8800
```

权重会缓存到 `~/.cache/huggingface/hub`，此后不再依赖网络。

## 实施路线

1. **工程搭建 + 意图目标管理** — server 骨架 + SQLite 初始化 + 意图管理页
2. **数据标注模块** — 作品录入 + 评论标注界面 + 批量导入
3. **模型训练引擎** — RoBERTa 微调 + 后台线程 + ONNX 导出
4. **批跑验证模块** — ONNX 推理 + 识别率报告
5. **外部数据同步** — ES 线索索引拉取（作品/评论/原始得分），快速标注与统计纳入外部数据
6. **监督模型自训练自动标注** — 训练读源合并手工+外部，自动标注引擎 + 「自动标注」页签（阈值/进度/可停止）

后续方向：实时推理 + 模型切换；`optimum[onnxruntime]` 正规 ONNX 导出与真 INT8 量化；FTS5 全文检索。

## License

MIT
