#!/usr/bin/env bash
# start-ui.sh —— 启动 Trata 网页 + API 代理（scripts/serve-ui.py）
#
# 用法:
#   bash scripts/start-ui.sh               # 后台启动（默认）
#   bash scripts/start-ui.sh --foreground  # 前台运行
#   bash scripts/start-ui.sh --status
#   bash scripts/start-ui.sh --stop
#
# 环境变量:
#   HOST      监听地址   默认 0.0.0.0（局域网可访问；只要本机用改成 127.0.0.1）
#   PORT      网页/API   默认 8001
#   BACKEND   llama-server 默认 http://127.0.0.1:8080
#   LOG       日志       默认 model/trata-ui.log
#   PIDFILE   PID 文件   默认 model/trata-ui.pid
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
BACKEND="${BACKEND:-http://127.0.0.1:8080}"
LOG="${LOG:-$ROOT/model/trata-ui.log}"
PIDFILE="${PIDFILE:-$ROOT/model/trata-ui.pid}"
UI="$ROOT/ui/strata/index.html"
PY="$ROOT/scripts/serve-ui.py"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

case "${1:-}" in
  --stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      kill -0 "$PID" 2>/dev/null && { say "停止 PID $PID"; kill -TERM "$PID"; sleep 1; }
      rm -f "$PIDFILE"
    else
      PIDS=$(pgrep -f "serve-ui.py" || true)
      [ -n "$PIDS" ] && { say "停止 PID $PIDS"; kill -TERM $PIDS; }
    fi
    say "已停止"
    exit 0
    ;;
  --status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "进程: $(cat "$PIDFILE") 运行中"
    elif pgrep -f "serve-ui.py" >/dev/null 2>&1; then
      echo "进程: 运行中（PID=$(pgrep -f serve-ui.py | head -1)）"
    else
      echo "进程: 未运行"
    fi
    echo -n "UI:   "
    curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "无响应"
    echo
    echo -n "后端: "
    curl -s --noproxy '*' --max-time 3 "$BACKEND/health" 2>/dev/null || echo "无响应"
    echo
    exit 0
    ;;
esac

command -v python3 >/dev/null 2>&1 || { echo "需要 python3"; exit 1; }
[ -f "$PY" ] || { echo "找不到 $PY"; exit 1; }
[ -f "$UI" ] || { echo "找不到 $UI"; exit 1; }

ARGS=(--host "$HOST" --port "$PORT" --backend "$BACKEND")

if [ "${1:-}" = "--foreground" ]; then
  exec python3 -I "$PY" "${ARGS[@]}"
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  say "已经在运行（PID $(cat "$PIDFILE")），先 --stop"
  exit 1
fi

nohup python3 -I "$PY" "${ARGS[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 1

if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  say "启动失败，日志:"
  tail -20 "$LOG"
  exit 1
fi

IP=$(ip -4 addr show scope global 2>/dev/null | grep -oP 'inet \K[\d.]+' | head -1)
say "PID $(cat "$PIDFILE")，日志 $LOG"
say "网页:   http://${IP:-127.0.0.1}:$PORT/"
say "API:    http://${IP:-127.0.0.1}:$PORT/v1"
say "监控:   http://${IP:-127.0.0.1}:$PORT/monitor"
[ "$HOST" = "0.0.0.0" ] && say "注意: 对外监听且无鉴权，不要暴露到公网（见 docs/08）"
