#!/bin/bash
# 串行训练编排：启动后端 -> NER 训练 -> 意图训练 -> 导出 ONNX -> 验证预测
# 全部通过训练中心 API 走，自动记录到 train_tasks 并走 MPS 加速。
set -u
# 定位到脚本所在目录的上一级（项目根目录），避免硬编码路径导致跑到别的项目里
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "项目目录: $PROJECT_DIR"
export HF_ENDPOINT=https://hf-mirror.com   # 保险起见显式设置，src/common/__init__.py 也会兜底
# 优先用项目虚拟环境，回退到 PATH 里的 python3
if [ -x "$PROJECT_DIR/.venv/bin/python3" ]; then
  PY="$PROJECT_DIR/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi
echo "Python: $($PY --version 2>&1)"

jget() { $PY -c "import sys,json;d=json.load(sys.stdin);print(d.get('$1',''))"; }

echo "===== [1/5] 启动后端 ====="
$PY -m uvicorn src.web.main:app --host 0.0.0.0 --port 8000 > /tmp/intent_backend.log 2>&1 &
UVPID=$!
echo "uvicorn pid=$UVPID"
for i in $(seq 1 40); do
  curl -sf http://localhost:8000/api/health > /dev/null 2>&1 && { echo "后端就绪"; break; }
  sleep 1
done
grep -E "\[device\]" /tmp/intent_backend.log || true

poll_task() {
  local tid="$1" label="$2"
  while true; do
    local st status prog msg
    st=$(curl -s "http://localhost:8000/api/train/tasks/$tid")
    status=$(echo "$st" | jget status)
    prog=$(echo "$st" | jget progress)
    msg=$(echo "$st" | jget message)
    echo "  [$label] status=$status progress=$prog msg=$msg"
    case "$status" in
      done|failed|stopped|interrupted) echo "$st"; return ;;
    esac
    sleep 10
  done
}

echo "===== [2/5] 启动 NER 训练 ====="
NER_ID=$(curl -s -X POST http://localhost:8000/api/train/ner \
  -H "Content-Type: application/json" \
  -d '{"epochs":3,"batch_size":32,"learning_rate":2e-5}' | jget task_id)
echo "NER task_id=$NER_ID"
poll_task "$NER_ID" "NER"

echo "===== [3/5] 启动意图训练 ====="
INT_ID=$(curl -s -X POST http://localhost:8000/api/train/intent \
  -H "Content-Type: application/json" \
  -d '{"epochs":3,"batch_size":32,"learning_rate":2e-5}' | jget task_id)
echo "INTENT task_id=$INT_ID"
poll_task "$INT_ID" "INTENT"

echo "===== [4/5] 导出 ONNX ====="
echo -n "NER 导出: "; curl -s -X POST http://localhost:8000/api/model/export/ner; echo ""
echo -n "意图导出: "; curl -s -X POST http://localhost:8000/api/model/export/intent; echo ""

echo "===== [5/5] 验证在线试测（正文+评论）====="
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"","note":"小米空调售后是真行 又是被小米服务惊艳到了","comment":"线上买的还是线下买的？"}'
echo ""
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"","note":"iPhone 17 开箱 用了半个月的感受","comment":"壳有吗 我想要个手机壳"}'
echo ""

echo "===== MPS 使用情况 ====="
grep -E "\[device\]" /tmp/intent_backend.log || echo "(未捕获到 device 日志)"
echo "===== 全部完成，后端 pid=$UVPID ====="
