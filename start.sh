#!/bin/bash
# 启动 content_intent_alpha 管理台，并在服务就绪后自动打开浏览器。
#
# 用法：
#   ./start.sh              # 默认 8000 端口
#   ./start.sh 8080         # 指定端口
#   ./start.sh --no-open    # 不自动打开浏览器
#
# 停止：Ctrl+C
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8000
AUTO_OPEN=1
for arg in "$@"; do
  case "$arg" in
    --no-open) AUTO_OPEN=0 ;;
    -h|--help)
      awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
      exit 0 ;;
    [0-9]*)    PORT="$arg" ;;
    *) echo "[warn] 忽略未知参数: $arg" ;;
  esac
done

# 优先使用项目虚拟环境
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
  PY="$SCRIPT_DIR/.venv/bin/python3"
else
  PY="$(command -v python3)"
  echo "[warn] 未找到 .venv，使用系统 python3：$PY"
fi

# 端口被占用时不要盲目覆盖，先提示
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
  echo "[warn] 端口 $PORT 已被占用："
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -n +2
  echo ""
  echo "若确认要重启该服务，先执行：kill \$(lsof -ti tcp:$PORT)"
  echo "或换一个端口启动：$0 8080"
  if [ "$AUTO_OPEN" = "1" ]; then
    echo ""
    echo "服务已在运行，直接打开页面：http://localhost:$PORT"
    ( sleep 1; open "http://localhost:$PORT" ) 2>/dev/null
  fi
  exit 0
fi

echo "项目目录: $SCRIPT_DIR"
echo "Python:   $($PY --version 2>&1)"
echo "端口:     $PORT"
echo ""

$PY -m uvicorn src.web.main:app --host 0.0.0.0 --port "$PORT" &
UVPID=$!

# 退出时杀掉 uvicorn，避免残留后台进程
cleanup() {
  echo ""
  echo "正在停止服务 (pid=$UVPID) ..."
  kill "$UVPID" 2>/dev/null
  wait "$UVPID" 2>/dev/null
  echo "已停止"
}
trap cleanup EXIT INT TERM

# 等待服务就绪（最多 40 秒）
READY=0
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
    READY=1
    break
  fi
  # 进程已退出说明启动失败
  kill -0 "$UVPID" 2>/dev/null || break
  sleep 1
done

if [ "$READY" != "1" ]; then
  echo "[error] 服务启动失败或未能就绪，请检查上方 uvicorn 输出"
  exit 1
fi

echo ""
echo "=============================================="
echo "  服务已就绪：http://localhost:$PORT"
echo "  API 文档：  http://localhost:$PORT/docs"
echo "  停止服务：  Ctrl+C"
echo "=============================================="

if [ "$AUTO_OPEN" = "1" ]; then
  ( sleep 1; open "http://localhost:$PORT" ) 2>/dev/null
fi

wait "$UVPID"
