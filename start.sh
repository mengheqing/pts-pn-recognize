#!/bin/bash

PROJECT_DIR="/root/pts-pn-recognize"
PORT=10086
LOG_FILE="${PROJECT_DIR}/server.log"

cd "$PROJECT_DIR" || exit 1

echo "=== $(date) ==="

echo "[1/5] 拉取最新代码..."
git pull

echo "[2/5] 查找并停止旧进程..."
PID=$(lsof -ti :${PORT} 2>/dev/null)
if [ -n "$PID" ]; then
    kill -9 "$PID"
    echo "已停止进程 $PID"
else
    echo "端口 ${PORT} 无运行进程"
fi

echo "[3/5] 激活虚拟环境..."
. venv/bin/activate

echo "[4/5] 启动服务..."
nohup python app.py >> "$LOG_FILE" 2>&1 &
echo "已启动，PID: $!"

echo "[5/5] 退出虚拟环境..."
deactivate 2>/dev/null || true

echo "等待服务就绪..."
sleep 2

echo "服务启动成功，访问 https://39.96.60.35:${PORT}"
