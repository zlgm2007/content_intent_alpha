# content_intent_alpha

中文内容理解工程，集成 **命名实体识别（NER）** 与 **意图强度打分** 两项能力。输入一篇笔记（标题 + 正文）及其评论，输出其中的品牌 / 品类 / 商品 / 型号实体，以及目标意图（当前为「购物」）的 0–5 分强度。

两个模型均基于 `albert_chinese_small` 微调，导出为 **ONNX**，由 `onnxruntime` 做 CPU 推理；配套一个 FastAPI + Vue3 的 Web 管理台，覆盖「ES 导入 → 数据准备 → 训练 → 导出 → 在线试测」全流程。

## 能力一览

| 能力   | 任务类型       | 标签 / 输出                                    | 产物                          |
| ---- | ---------- | ------------------------------------------ | --------------------------- |
| 实体识别 | 序列标注（BIO）  | `BRAND` / `CATEGORY` / `PRODUCT` / `MODEL` | `models/ner/ner.onnx`       |
| 意图打分 | 文本分类（6 分类） | 0–5 分，映射为 无意图 / 有倾向 / 中度意图 / 强烈意图          | `models/intent/intent.onnx` |

分数语义：`0–2` 无意图，`3` 有倾向，`4` 中度意图，`5` 强烈意图（见 `src/intent/inference.py` 的 `LEVEL_MAP`）。

两者都是「标题 + 正文 + 评论」拼接后推理，拼接规则统一走 `src/common/compose.py`，保证训练与线上特征一致。

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+，实测 3.12.8 可用）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 一键启动（等服务就绪后自动打开浏览器）
./start.sh
```

浏览器打开 <http://localhost:8000> 即为管理台。详细的启动方式、验证步骤与故障排查见下方「服务启动与验证」。

## 服务启动与验证

### 环境要求

| 组件     | 要求                  | 备注                       |
| ------ | ------------------- | ------------------------ |
| Python | 3.10+（实测 3.12.8）    | 项目已自带 `.venv`            |
| Node.js | 18+（实测 22.22）       | **仅改前端时需要**              |
| 内存     | ≥ 2GB 可用            | 两个 ONNX 常驻约 1.2GB        |
| GPU    | 不需要                 | 推理纯 CPU；训练在有 MPS 时自动启用   |

### 启动方式

**方式一：一键脚本（推荐）**

```bash
./start.sh              # 8000 端口，就绪后自动打开浏览器
./start.sh 8080         # 指定端口
./start.sh --no-open    # 只启动，不开浏览器
./start.sh --help       # 查看用法
```

脚本依次完成：优先选用 `.venv/bin/python3` → 检测端口占用（**列出占用进程，不盲目 kill**）→ 轮询 `/api/health` 等待就绪（最长 40 秒）→ 打开浏览器。前台运行，`Ctrl+C` 自动回收 uvicorn 进程，不留残留。

**方式二：手动启动**

```bash
source .venv/bin/activate
uvicorn src.web.main:app --host 0.0.0.0 --port 8000

# 或不用激活 venv，直接调用
.venv/bin/uvicorn src.web.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ **`uvicorn` 不会自动打开浏览器**。日志停在 `Uvicorn running on http://0.0.0.0:8000` 就说明启动成功了，需要自己访问 <http://localhost:8000> —— 这是最常见的「启动后没反应」误解。

**方式三：独立推理服务（无管理台，仅 `/health` 与 `/predict`）**

```bash
# 注意避开管理台的 8000 端口
uvicorn src.serve.app:app --host 0.0.0.0 --port 8001
```

面向生产的轻量部署，不含数据库与前端。实测（2026-09-01）：

```bash
curl -s http://localhost:8001/health
# {"status":"ok","ner":true,"intent":true}

curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "客厅改造分享", "note": "换了个扫地机真香", "comment": "什么牌子多少钱"}'
# 返回实体与意图，结果与管理台 /api/predict 一致
```

> 两个服务的模型加载时机不同：
>
> - **独立服务**：在**启动时同步加载**两个模型（日志出现 `[serve] NER 模型加载成功` 即就绪），因此启动比管理台慢一些；`/health` 的 `ner` / `intent` 反映的是**模型是否加载成功**，模型缺失时这两项为 `false`，服务仍会启动但 `/predict` 不可用。
> - **管理台**：模型**懒加载**，服务秒起，第一次调 `/api/predict` 才加载，状态查 `/api/predict/status`。

### 验证服务已就绪

依次检查下面四项，全通过即服务正常：

```bash
# 1) 健康检查 —— 应返回 {"status":"ok","es_ok":false,...}
#    es_ok 为 false 是正常的（ES 在内网，本地连不上），不影响其余功能
curl -s http://localhost:8000/api/health

# 2) 前端页面 —— 应输出 <title>内容意图识别 · 管理台</title>
curl -s http://localhost:8000/ | grep -o "<title>.*</title>"

# 3) 静态资源 —— 两个 200 说明前端产物挂载正常
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/assets/index-Brp9TB6G.js
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/assets/index-z61fX0Ie.css

# 4) 端到端预测 —— 返回实体与意图得分即模型可用
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "客厅改造分享", "note": "换了个扫地机真香", "comment": "什么牌子多少钱"}'
```

> 资源文件名带构建哈希，重新 `npm run build` 后会变。若第 3 步返回 404，用
> `grep -o 'src="[^"]*"' frontend/dist/index.html` 取当前实际文件名。

一键自检（只列出非 200 的端点）：

```bash
for ep in / /api/health /api/stats /api/intent/config /api/predict/status \
          /api/lexicon /api/model /api/es/config /api/train/tasks /api; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8000$ep")
  [ "$code" = "200" ] || echo "FAIL $ep ($code)"
done; echo "自检完成"
```

**已验证项**（2026-09-01，Python 3.12.8 / Node 22.22）：

| 检查项                     | 结果                          |
| ----------------------- | --------------------------- |
| 10 个只读 API 端点           | 全部 200，最慢 96ms               |
| 首页与 JS/CSS 资源           | 200（JS 1.06MB / CSS 361KB）  |
| 8 个子路由直接刷新              | 全部返回首页，SPA fallback 正常      |
| 22 处前端 API 调用 vs 后端路由   | 路径全部对应，含 SSE 训练日志流          |
| 端到端预测                   | 通过，见下方「在线试测请求示例」            |

### 停止与重启

```bash
# 停止（start.sh 前台运行时直接 Ctrl+C 即可）
kill $(lsof -ti tcp:8000)

# 重启：先停再起
kill $(lsof -ti tcp:8000) && sleep 2 && ./start.sh

# 查看是否在运行
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

### 前端开发

已构建产物在 `frontend/dist`，后端启动时自动挂载，**日常使用只需启动后端**。只有改前端时才需要下面几步：

```bash
cd frontend
npm install          # 首次需要（node_modules 未入库）
npm run dev          # 开发模式，5173 端口，已配好代理到后端

npm run build        # 改完必须重新构建，否则线上还是旧页面
```

> 改了 `frontend/src` 一定要 `npm run build`。这是「改了代码但页面没变」的唯一原因。

## 目录结构

```
content_intent_alpha/
├── configs/                  # 训练 / 推理 / 数据源配置
│   ├── es.yaml               # ES 连接、索引、字段映射
│   ├── ner.yaml              # NER 标签体系、训练超参、推理解码后处理
│   ├── intent.yaml           # 意图模型配置 + 先验修正说明
│   └── intent_prior.json     # 先验偏移向量（由脚本生成）
├── data/
│   ├── web.db                # 管理台 SQLite 库（es_docs 等 7 张表）
│   ├── ner/
│   │   ├── lexicon/          # brand / category / product / model 词典 + blocklist
│   │   ├── raw/              # 原始语料
│   │   └── annotated/        # BIO conll 标注、实体表、留出集
│   └── intent/
│       ├── raw/              # 原始语料
│       └── labeled/          # 6 分类 jsonl 标注、留出集、金标准标注集
├── src/
│   ├── common/               # 公共：拼接、数据集、ONNX 工具、指标、设备选择
│   ├── ner/                  # 预处理 / 训练 / 导出 / 后处理 / 推理
│   ├── intent/               # 预处理 / 训练 / 校准 / 导出 / 推理
│   ├── web/                  # 管理台后端（FastAPI）
│   │   ├── main.py           # 入口，同时挂载前端静态资源
│   │   ├── db.py             # SQLite 表结构与读写
│   │   ├── routers/          # system / es / lexicon / data / train / model / predict
│   │   └── *_service.py      # 导入、词典、数据准备、训练、模型管理
│   └── serve/                # 独立推理服务（仅 /health 与 /predict，不含管理台）
├── frontend/                 # Vue3 + Vite + Element Plus 管理台
│   ├── src/                  # 页面源码（改动需重新 build）
│   └── dist/                 # 构建产物，后端直接挂载（入库，开箱即用）
├── models/
│   ├── ner/                  # ner.onnx + tokenizer + config
│   └── intent/               # intent.onnx + tokenizer + config
├── scripts/                  # 评估 / 分析 / 调参 / 流水线脚本
├── tests/
├── start.sh                  # 一键启动（等就绪后自动开浏览器）
└── requirements.txt
```

## Web 管理台

| 页面      | 路由           | 功能                                     |
| ------- | ------------ | -------------------------------------- |
| 总览      | `/`          | 数据量、意图分数分布、词典统计                        |
| ES 数据导入 | `/es-import` | 配置连接、测试连通、分片导入、增量同步、清空重导               |
| 词典管理    | `/lexicon`   | 品牌 / 品类 / 商品 / 型号词典增删改查，从 ES 聚合重建      |
| 数据准备    | `/data-prep` | 由 `es_docs` 生成 NER conll 与意图 jsonl 训练集 |
| 训练中心    | `/train`     | 后台线程异步训练，SSE 实时推日志与进度，可中断              |
| 模型管理    | `/model`     | 导出 ONNX、版本记录、下载                        |
| 在线试测    | `/test`      | 实时输入文本，查看实体与意图得分                       |

### 主要 API

| 分组             | 接口                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------- |
| `/api`         | `GET /health`、`GET /stats`、`GET /intent/config`                                                 |
| `/api/es`      | `GET / PUT /config`、`POST /test`、`POST /import`、`GET /import/tasks[/{id}]`、`POST /lexicon/refresh`、`POST /clear` |
| `/api/lexicon` | `GET /`、`GET / POST / DELETE /{ltype}`                                                          |
| `/api/data`    | `POST /ner/prepare`、`POST /intent/prepare`                                                      |
| `/api/train`   | `POST /ner`、`POST /intent`、`GET /tasks[/{id}]`、`GET /tasks/{id}/stream`、`POST /tasks/{id}/stop` |
| `/api/model`   | `GET /`、`POST /export/{model_type}`、`GET /download/{model_type}`                                |
| `/api/predict` | `POST /`、`GET /status`、`POST /reload`                                                           |

完整交互式文档见 <http://localhost:8000/docs>（服务启动后访问）。

### 在线试测请求示例

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "客厅改造分享", "note": "换了个扫地机真香", "comment": "什么牌子多少钱"}'
```

实测返回（2026-09-01）：

```json
{
  "title": "客厅改造分享",
  "note": "换了个扫地机真香",
  "comment": "什么牌子多少钱",
  "entities": [
    {"type": "CATEGORY", "start": 17, "end": 20, "text": "扫地机", "prob": 0.9953, "source": "正文"}
  ],
  "intents": [
    {"name": "购物", "score": 5, "prob": 0.9625, "level": "强烈意图"}
  ],
  "errors": []
}
```

`start` / `end` 是实体在**拼接后文本**中的字符偏移（不是原始 note 的偏移），拼接规则见 `src/common/compose.py`。

模型按 ONNX 文件 **mtime** 自动失效重载：训练完重新导出后，下一次请求即生效，无需重启后端。也可手动 `POST /api/predict/reload`。

## 常见问题

**服务启动了但「没反应」**  
`uvicorn` 不会自动打开浏览器，日志停在 `Uvicorn running on ...` 就是成功了，手动访问 <http://localhost:8000>。详见「服务启动与验证」。

**首次预测特别慢（约 20 秒）**  
模型是懒加载的：第一次调 `/api/predict` 时才加载两个 ONNX 与 tokenizer，之后每次约几十毫秒。`GET /api/predict/status` 可查看是否已加载。

**端口 8000 被占用 / 改了前端页面没变化**  
分别见「停止与重启」与「前端开发」两节。

**设置了 HTTP(S)\_PROXY 时访问不了 localhost**  
代理会拦截本地请求（表现为 502 `upstream connect failed`）。给 localhost 加例外：

```bash
export no_proxy=localhost,127.0.0.1
# 或单次绕过：curl --noproxy '*' http://localhost:8000/api/health
```

**`/api/health` 返回 `es_ok: false`**  
正常现象。ES 是内网地址 `10.104.214.120:9200`，本地连不上时管理台其余功能不受影响，只有「ES 数据导入」页不可用。

## 数据源

`configs/es.yaml` 默认指向内网 ES `http://10.104.214.120:9200`，两个索引 `lead_clue_score_ge3`（3–5 分）与 `lead_clue_score_lt3`（0–2 分），经 `src/web/es_importer.py` 用 `search_after` 严格增量导入，位点存 `sync_state` 表，保证不重不漏。

当前 `data/web.db` 已导入 **151,150** 条文档，分数分布：

| 分数 | 0     | 1      | 2      | 3      | 4     | 5      |
| -- | ----- | ------ | ------ | ------ | ----- | ------ |
| 条数 | 7,680 | 21,871 | 16,418 | 55,332 | 8,835 | 41,014 |

> 注意：两个索引按分数**完全切分**，因此「有意图占比 69.6%」是导入比例决定的，不是自然分布——这一点直接影响意图模型的先验修正是否该开启，详见 `configs/intent.yaml` 中的敏感性分析。

## 训练数据格式

**NER**（`data/ner/annotated/train.conll`）：标准 BIO 两列，字符与标签用 Tab 分隔，句子间空行分隔。由词典自动打标 + 实体对齐生成。

```
正	O
文	O
小	B-BRAND
米	I-BRAND
```

**意图**（`data/intent/labeled/train.jsonl`）：每行一个样本，正文与评论已按 `compose_note_comment` 拼接。

```json
{"text": "【正文】不买沙发的客厅…【评论】扫地机怎么上下水", "label": 0}
```

当前训练集规模：意图 81,602 条（留出集 13,970 条），NER 81,020 句（金标准版 53,825 句，留出集 14,160 句）。

## 命令行方式

管理台之外，全流程也可命令行执行：

```bash
# NER：训练 → 导出 ONNX
python -m src.ner.train --config configs/ner.yaml
python -m src.ner.export_onnx --config configs/ner.yaml

# 意图：训练 → 导出 ONNX
python -m src.intent.train --config configs/intent.yaml
python -m src.intent.export_onnx --config configs/intent.yaml

# 单条推理
python -m src.intent.inference --config configs/intent.yaml --desc "iPhone 17 开箱" --comment "壳有吗"

# 独立推理服务（不含管理台，只有 /health 与 /predict）
# 注意避开管理台的 8000 端口
uvicorn src.serve.app:app --host 0.0.0.0 --port 8001
```

批量串行训练可用 `bash scripts/orchestrate_training.sh`（NER → 意图 → 导出 → 验证，全程走训练中心 API）。

## 换一个意图目标

系统是针对单一意图的强度打分器，换目标（如「购物」→「售后」）只需：

1. 改 `configs/intent.yaml` 的 `model.intent_name`；
2. 改 `configs/es.yaml` 的 `fields.intent_score` 为新意图在 ES 里的打分字段；
3. 重新导入 ES 数据 → 数据准备 → 训练 → 导出 ONNX。

训练、推理、导出链路与前端文案全部复用（前端通过 `GET /api/intent/config` 读取意图名）。

## 意图先验偏移修正

训练集做过类别平衡（1/3/5 类各截断到 20,000 条），与 ES 真实分布差异显著（3 分真实占 36.6%，训练里仅 21.5%），导致模型系统性低判 3/5 分。`configs/intent.yaml` 的 `inference.prior_correction_alpha` 通过在 logits 上叠加 `alpha * log(P_real / P_train)` 做修正，偏置向量存于 `configs/intent_prior.json`。

- `alpha = 0` 关闭修正，当前配置 **0.8**；
- 留出集实测（13,970 条）：准确率 0.7258 → 0.7367，MAE 0.5775 → 0.5585；
- **仅在「线上有意图(3–5分)占比 ≥ 60%」时开启才划算**，换成未筛选的新内容必须把 alpha 改回 0.0 并重扫。

相关脚本：`scripts/tune_intent_prior.py`（生成/搜索）、`scripts/sweep_intent_prior_alpha.py`（全量 alpha 扫描）。

## 辅助脚本

`scripts/` 下均为离线分析工具，不改动线上模型：

| 脚本                                                          | 用途                                    |
| ----------------------------------------------------------- | ------------------------------------- |
| `eval_ner_ab.py` / `eval_ner_dual.py`                       | NER 量化评估、A-B 对比、ES 真值轨 + 词典自洽轨双轨评测    |
| `eval_intent.py`                                            | 意图模型评估（含已见/未见拆分）                      |
| `eval_on_gold.py`                                           | 用人工金标准集评估，规避脏标签污染                     |
| `build_gold_annotation_set.py`                              | 生成人工标注集 Excel                         |
| `tune_intent_prior.py`                                      | 意图先验偏移向量生成与敏感性分析                      |
| `sweep_intent_prior_alpha.py`                               | 全量 alpha 扫描（tune/eval 半区，避免调参乐观偏差）    |
| `retrain_ner.py` / `finish_ner_pipeline.py`                 | NER 重训与收尾流水线（导出 → 白名单 → badcase → 评测） |
| `verify_ner.py` / `compare_ner_models.py`                   | badcase 回归、新旧模型逐实体差异                  |
| `analyze_lexicon_conflict.py`                               | 词典同词多类型的冲突消解决策                        |
| `analyze_es_label_conflict.py` / `analyze_category_gain.py` | ES 标签冲突根因分析、品类级映射收益量化                 |
| `measure_span_consistency.py`                               | 词典词「出现 N 次但只标注 M 次」的 span 一致性度量       |
| `make_sample_data.py`                                       | 生成示例数据，端到端跑通骨架                        |
| `bench_mps.py`                                              | MPS vs CPU 训练吞吐基准                     |

## 模型产物

`models/{ner,intent}/` 下各含 `*.onnx`（推理用）、`model.safetensors`（PyTorch 权重，供继续训练/重新导出）、`config.json`、`tokenizer.json`、`tokenizer_config.json`。

通过管理台或 `model_service.py` 导出 ONNX 时，会把当前模型先复制到同名目录下的 `backup/` 子目录（按时间戳命名）后再覆盖，避免训练回退时无模型可用。

## 技术栈

- **训练**：PyTorch + HuggingFace `transformers`（底座 `voidful/albert_chinese_small`）
- **部署**：`torch.onnx` 导出 + `onnxruntime` 推理
- **后端**：FastAPI + Uvicorn + SQLite（stdlib `sqlite3`）
- **前端**：Vue 3 + Vite + Element Plus
- **其他**：`pyahocorasick`（词典多模式匹配）、pandas / numpy、scikit-learn（校准）
