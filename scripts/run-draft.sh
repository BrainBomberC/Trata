#!/usr/bin/env bash
# run-draft.sh —— 启动生产配置：层切分 3 卡 + LiLiCorr(DFlash) 草稿 + 1M 上下文
#
# 用法:
#   bash scripts/run-draft.sh                  # 前台运行（Ctrl-C 停止）
#   bash scripts/run-draft.sh --daemon         # 后台运行并等待 /health
#   bash scripts/run-draft.sh --status
#   bash scripts/run-draft.sh --stop
#
# 常用环境变量:
#   MODEL    主模型     默认 model/bolt74b/NVIDIA-…-IQ2_XXS.gguf
#   DRAFT    草稿模型   默认 model/draft/dflash-…-NVFP4.gguf（找不到再试 …-LiLiCorr.gguf）
#   MM       mmproj     默认 model/mmproj-BF16.gguf（必须是 Bolt 仓库那份，519 张量）
#   LLAMA_BIN           默认 llama.cpp/build-nccl/bin/llama-server
#   CTX      上下文     默认 1048576
#   UB       物理批     默认 64（1M + 草稿下 128 也可，256 会 OOM）
#   TS       切分比例   默认 1,1,1
#   SM       切分方式   默认 layer（改成 tensor 需要 NCCL，且实测更慢）
#   NMAX     草稿候选数 默认 4（1M 下最优；200K/400K 用 3 更快）
#   NMIN     至少猜几个 默认 0
#   SPEC     草稿类型   默认 draft-dflash；设为 none 可完全关闭草稿
#   NGLD     草稿层数   默认 all；0 = 草稿放 CPU（实测 21.9 tok/s，仅对照）
#   CTK/CTV  KV 类型    默认 q4_0（200K/400K 可改 q8_0）
#   NP       并发槽位   默认 1
#   HOST/PORT           默认 127.0.0.1:8080
#   THREADS  CPU 线程   默认 nproc-1
#   EXTRA    追加参数   例如 EXTRA="-lv 4"
#   LOG/PIDFILE         后台日志与 PID 文件
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${LLAMA_BIN:-$ROOT/llama.cpp/build-nccl/bin/llama-server}"
MODEL="${MODEL:-$ROOT/model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf}"

if [ -n "${DRAFT:-}" ]; then
  :
elif [ -f "$ROOT/model/draft/dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf" ]; then
  DRAFT="$ROOT/model/draft/dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf"
else
  DRAFT="$ROOT/model/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf"
fi

MM="${MM-$ROOT/model/mmproj-BF16.gguf}"     # 显式 MM= 空字符串可禁用视觉
SM="${SM:-layer}"
CTX="${CTX:-1048576}"
UB="${UB:-64}"
TS="${TS:-1,1,1}"
NMAX="${NMAX:-4}"
NMIN="${NMIN:-0}"
SPEC="${SPEC:-draft-dflash}"
NGLD="${NGLD:-all}"
CTK="${CTK:-q4_0}"
CTV="${CTV:-q4_0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
NP="${NP:-1}"
CPUS=$(nproc)
THREADS="${THREADS:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}"
EXTRA="${EXTRA:-}"
LOG="${LOG:-$ROOT/model/bolt-serve.log}"
PIDFILE="${PIDFILE:-$ROOT/model/bolt-serve.pid}"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- NCCL 自动探测（二进制链接了 libnccl；层切分也要能找到它）
if [ -z "${NCCL_DIR:-}" ]; then
  for d in "$HOME"/nccl-venv/lib/python*/site-packages/nvidia/nccl \
           "$HOME"/.venv/lib/python*/site-packages/nvidia/nccl; do
    [ -f "$d/lib/libnccl.so.2" ] && { NCCL_DIR="$d"; break; }
  done
fi

# ---- 子命令
case "${1:-}" in
  --status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "进程: $(cat "$PIDFILE") 运行中"
    elif pgrep -f "llama-server.*$(basename "$MODEL")" >/dev/null 2>&1; then
      echo "进程: 运行中（PID=$(pgrep -f "llama-server.*$(basename "$MODEL")" | head -1)）"
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
[ -f "$MODEL" ] || { echo "找不到主模型: $MODEL"; echo "先运行 bash scripts/download-model.sh"; exit 1; }

if [ "$SPEC" != "none" ]; then
  [ -f "$DRAFT" ] || { echo "找不到草稿模型: $DRAFT"; echo "先运行 PRESET=draft bash scripts/download-model.sh"; exit 1; }
fi

[ -n "$MM" ] && [ ! -f "$MM" ] && say "[warn] 找不到投影器 $MM，视觉功能将不可用（不影响文字）"

ARGS=(-m "$MODEL" -ngl all -sm "$SM" -ts "$TS" -fa on
      -ctk "$CTK" -ctv "$CTV" -c "$CTX" -ub "$UB"
      -t "$THREADS" -np "$NP" -fit off
      --host "$HOST" --port "$PORT")

if [ "$SPEC" != "none" ]; then
  ARGS+=(-md "$DRAFT" -ngld "$NGLD"
         --spec-type "$SPEC" --spec-draft-n-max "$NMAX" --spec-draft-n-min "$NMIN")
fi

# 超过模型原生 262144 时放开元数据上限
if [ "$CTX" -gt 262144 ] 2>/dev/null; then
  ARGS+=(--override-kv "nemotron_h_moe.context_length=int:$CTX")
fi

if [ -n "$MM" ] && [ -f "$MM" ]; then
  ARGS+=(-mm "$MM" --no-mmproj-offload)   # 视觉编码器放 CPU，不抢显存
fi

# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

export LD_LIBRARY_PATH="$([ -n "${NCCL_DIR:-}" ] && echo "$NCCL_DIR/lib:")$(dirname "$BIN"):${LD_LIBRARY_PATH:-}"
export LLAMA_SM_TENSOR_ALLOW_ARCH=1

say "模型:   $MODEL"
say "草稿:   $([ "$SPEC" = none ] && echo 关闭 || echo "$DRAFT")"
say "切分:   -sm $SM -ts $TS   上下文: $CTX   ub: $UB   KV: $CTK/$CTV   槽位: $NP"
say "投机:   ${SPEC} n_max=$NMAX n_min=$NMIN   草稿设备层: $NGLD"
say "视觉:   $([ -n "$MM" ] && [ -f "$MM" ] && echo "$MM" || echo 关闭)"
say "监听:   http://$HOST:$PORT"
[ -n "${NCCL_DIR:-}" ] && say "NCCL:   $NCCL_DIR"

if [ "${1:-}" = "--daemon" ]; then
  mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
  nohup "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  say "后台 PID $(cat "$PIDFILE")，日志 $LOG"
  say "等待就绪（冷缓存可能 3–40 分钟）…"
  for i in $(seq 1 480); do
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
  say "40 分钟仍未就绪，看日志: tail -f $LOG"
  exit 1
fi

exec "$BIN" "${ARGS[@]}"
