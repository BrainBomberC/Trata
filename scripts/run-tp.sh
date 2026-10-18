#!/usr/bin/env bash
# run-tp.sh —— 回退配置：3 卡张量并行 + 1M 上下文（无草稿，46.6–47.7 tok/s）
#
# 生产配置请用 scripts/run-draft.sh（层切分 + LiLiCorr 草稿，54–55 tok/s）。
# 本脚本保留是为了对照、以及在"不想用草稿"时快速回退。
#
# 用法:
#   bash scripts/run-tp.sh                  # 前台运行（Ctrl-C 停止）
#   bash scripts/run-tp.sh --daemon         # 后台运行
#   bash scripts/run-tp.sh --status
#   bash scripts/run-tp.sh --stop
#
# 常用环境变量:
#   MODEL   模型路径      默认原版 IQ2_XXS（TP+1M 用 -ub 64 刚好装下）
#   LLAMA_BIN  llama-server 路径
#   NCCL_DIR  NCCL 目录（含 lib/libnccl.so.2）
#   CTX      上下文        默认 1048576
#   UB       物理批大小    默认 64（1M 下 256 会 OOM）
#   TS       切分比例      默认 1.02,1.02,0.96（按显存余量微调）
#   HOST/PORT 监听地址     默认 127.0.0.1:8080
#   THREADS  CPU 线程      默认 nproc-1
#   NGL       GPU 层数     默认 all
#   CTK/CTV  KV 类型       默认 q4_0（1M 用；200K 可改 q8_0）
#   NP       并发槽位      默认 1；4 路并发用 NP=4（聚合吞吐约 83 tok/s）
#   MM       mmproj 路径   默认 model/mmproj-BF16.gguf（必须是 Bolt 仓库那份）
#   NO_MMPROJ_OFFLOAD=0    允许视觉编码器上 GPU（默认放 CPU）
#   EXTRA    追加参数，例如 EXTRA="-lv 3 --metrics"
#   LOG      后台日志      默认 model/bolt-serve.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LLAMA_BIN:-$ROOT/llama.cpp/build-nccl/bin/llama-server}"
ORIG="$ROOT/model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf"

MODEL="${MODEL:-$ORIG}"

SM="${SM:-tensor}"
CTX="${CTX:-1048576}"
UB="${UB:-64}"
TS="${TS:-1.02,1.02,0.96}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
NGL="${NGL:-all}"
NP="${NP:-1}"
CTK="${CTK:-q4_0}"
CTV="${CTV:-q4_0}"
CPUS=$(nproc)
THREADS="${THREADS:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}"
MM="${MM:-$ROOT/model/mmproj-BF16.gguf}"
EXTRA="${EXTRA:-}"
LOG="${LOG:-$ROOT/model/bolt-serve.log}"
PIDFILE="${PIDFILE:-$ROOT/model/bolt-serve.pid}"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- NCCL 自动探测
if [ -z "${NCCL_DIR:-}" ]; then
  for d in "$HOME"/nccl-venv/lib/python*/site-packages/nvidia/nccl; do
    [ -f "$d/lib/libnccl.so.2" ] && { NCCL_DIR="$d"; break; }
  done
fi

# ---- 子命令
case "${1:-}" in
  --status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "进程: $(cat "$PIDFILE") 运行中"
    elif pgrep -f "llama-server.*$(basename "$MODEL")" >/dev/null 2>&1; then
      echo "进程: 运行中（无 PID 文件，PID=$(pgrep -f "llama-server.*$(basename "$MODEL")" | head -1)）"
    else
      echo "进程: 未运行"
    fi
    echo -n "健康: "
    curl -s --noproxy '*' --max-time 3 "http://$HOST:$PORT/health" 2>/dev/null || echo "无响应"
    echo
    ss -ltn 2>/dev/null | grep -E ":$PORT\b" || true
    exit 0
    ;;
  --stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      if kill -0 "$PID" 2>/dev/null; then
        say "停止 PID $PID …"
        kill -TERM "$PID"
        for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$PID" 2>/dev/null && { say "强制结束"; kill -KILL "$PID"; }
      fi
      rm -f "$PIDFILE"
    else
      PIDS=$(pgrep -f "llama-server.*$(basename "$MODEL")" || true)
      if [ -n "$PIDS" ]; then say "停止 PID $PIDS …"; kill -TERM $PIDS; sleep 3; fi
    fi
    say "已停止"
    exit 0
    ;;
esac

# ---- 校验
[ -x "$BIN" ] || { echo "找不到可执行的 llama-server: $BIN"; echo "先编译，或设置 LLAMA_BIN="; exit 1; }
[ -f "$MODEL" ] || { echo "找不到模型: $MODEL"; echo "先运行 scripts/download-model.sh"; exit 1; }

if [ "$SM" = "tensor" ]; then
  [ -n "${NCCL_DIR:-}" ] && [ -f "$NCCL_DIR/lib/libnccl.so.2" ] || {
    echo "启用 -sm tensor 需要 NCCL（libnccl.so.2）。"
    echo "安装: python3 -m venv ~/nccl-venv && ~/nccl-venv/bin/pip install nvidia-nccl-cu12"
    echo "或设置 NCCL_DIR=/path/to/nvidia/nccl"
    exit 1
  }
fi

ARGS=(-m "$MODEL" -ngl "$NGL" -sm "$SM" -ts "$TS" -fa on
      -ctk "$CTK" -ctv "$CTV" -c "$CTX" -ub "$UB"
      -t "$THREADS" -np "$NP" -fit off
      --host "$HOST" --port "$PORT")

# 超过模型原生 262144 时放开元数据上限
if [ "$CTX" -gt 262144 ] 2>/dev/null; then
  ARGS+=(--override-kv "nemotron_h_moe.context_length=int:$CTX")
fi

if [ -n "$MM" ] && [ -f "$MM" ]; then
  ARGS+=(-mm "$MM")
  [ "${NO_MMPROJ_OFFLOAD:-1}" = "1" ] && ARGS+=(--no-mmproj-offload)
fi

# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

export LD_LIBRARY_PATH="$([ -n "${NCCL_DIR:-}" ] && echo "$NCCL_DIR/lib:")$(dirname "$BIN"):${LD_LIBRARY_PATH:-}"
export LLAMA_SM_TENSOR_ALLOW_ARCH=1

say "模型: $MODEL"
say "切分: -sm $SM -ts $TS  上下文: $CTX  ub: $UB  KV: $CTK/$CTV  槽位: $NP"
say "监听: http://$HOST:$PORT"
[ "$SM" = "tensor" ] && say "NCCL: $NCCL_DIR"

if [ "${1:-}" = "--daemon" ]; then
  mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
  nohup "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  say "后台 PID $(cat "$PIDFILE")，日志 $LOG"
  say "等待就绪（冷缓存可能 3–8 分钟）…"
  for i in $(seq 1 120); do
    sleep 5
    if curl -s --noproxy '*' --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      say "就绪（$((i*5))s）"
      exit 0
    fi
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      say "进程已退出，日志末尾:"
      tail -40 "$LOG"
      exit 1
    fi
  done
  say "10 分钟仍未就绪，看日志: tail -f $LOG"
  exit 1
fi

exec "$BIN" "${ARGS[@]}"
