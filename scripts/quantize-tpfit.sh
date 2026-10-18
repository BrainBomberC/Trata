#!/usr/bin/env bash
# quantize-tpfit.sh —— 生成 19.58 GiB 的 -tpfit.gguf（3 卡 TP + 1M 上下文专用）
#
# 配方（共 240 条逐张量规则，其余默认 Q6_K）:
#   40 × ssm_in.weight   -> q3_K
#   40 × ssm_out.weight  -> q3_K
#   40 × ffn_down_exps   -> iq2_xxs   （源文件已经是 iq2_xxs，补丁 0003 直接跳过）
#   40 × ffn_up_exps     -> iq2_xxs
#   40 × ffn_down_shexp  -> iq2_xxs
#   40 × ffn_up_shexp    -> iq2_xxs
#
# 用法:
#   bash scripts/quantize-tpfit.sh
#   IN=/path/to/xxx.gguf OUT=/path/to/xxx-tpfit.gguf bash scripts/quantize-tpfit.sh
#   QUANTIZE=/path/to/llama-quantize bash scripts/quantize-tpfit.sh
#   --dry-run     只打印命令，不执行
#
# 环境变量:
#   IN         输入模型    默认 model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf
#   OUT        输出模型    默认同名加 -tpfit
#   QUANTIZE   llama-quantize 路径
#   THREADS    量化线程数  默认 nproc-1
#   FORCE=1    覆盖已有输出
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS"
IN="${IN:-$ROOT/model/bolt74b/$BASE.gguf}"
OUT="${OUT:-$ROOT/model/bolt74b/$BASE-tpfit.gguf}"
QUANTIZE="${QUANTIZE:-$ROOT/llama.cpp/build-nccl/bin/llama-quantize}"
CPUS=$(nproc)
THREADS="${THREADS:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}"
FORCE="${FORCE:-0}"
TYPES="${TYPES:-$ROOT/model/bolt74b/.tpfit-types.txt}"
LOG="${LOG:-$ROOT/benchmarks/logs/requant-tpfit-$(date +%Y%m%d-%H%M%S).log}"
EXPECT_BYTES=21025816576
EXPECT_IN_BYTES=22018326464
# 旧版下载（元数据多 64 字节）张量布局相同，规则与输出一致
LEGACY_IN_BYTES=22018326528

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say() { echo "[$(date '+%F %T')] $*"; }

[ -f "$IN" ] || { echo "找不到输入模型: $IN"; echo "先运行 scripts/download-model.sh"; exit 1; }
[ -x "$QUANTIZE" ] || { echo "找不到可执行的 llama-quantize: $QUANTIZE"; echo "先按 docs/03 编译，或设置 QUANTIZE="; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "需要 python3 生成类型清单"; exit 1; }

IN_SIZE=$(stat -c%s "$IN")
if [ "$IN_SIZE" != "$EXPECT_IN_BYTES" ] && [ "$IN_SIZE" != "$LEGACY_IN_BYTES" ]; then
  say "[warn] 输入大小 $IN_SIZE，预期 $EXPECT_IN_BYTES（官方 IQ2_XXS）"
  say "       如果不是这份文件，专家可能不是 iq2_xxs，本配方会失败或质量不可控"
fi

if [ -f "$OUT" ] && [ "$FORCE" != "1" ]; then
  echo "输出已存在: $OUT"
  echo "要覆盖请 FORCE=1 bash scripts/quantize-tpfit.sh"
  exit 1
fi

# 生成 240 条规则
mkdir -p "$(dirname "$TYPES")"
if ! python3 -I "$ROOT/scripts/gen-expert-types.py" "$IN" "$TYPES" --tpfit; then
  echo "类型清单生成失败（预期 240 条）"
  exit 1
fi

N_IQ=$(grep -c '=iq2_xxs$' "$TYPES" || true)
N_Q3=$(grep -c '=q3_K$' "$TYPES" || true)
if [ "$N_IQ" != "160" ] || [ "$N_Q3" != "80" ]; then
  echo "规则数不对: iq2_xxs=$N_IQ（应 160），q3_K=$N_Q3（应 80）"
  exit 1
fi
say "规则检查通过: iq2_xxs=$N_IQ, q3_K=$N_Q3"

# 磁盘：输入已在，输出约 19.6 GiB
mkdir -p "$(dirname "$OUT")"
FREE_KB=$(df -Pk "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
NEED_KB=$(( EXPECT_BYTES / 1024 + 1048576 ))
if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt "$NEED_KB" ] && [ "$FORCE" != "1" ]; then
  say "[FAIL] 磁盘不足：需要约 $((NEED_KB/1048576)) GiB，可用 $((FREE_KB/1048576)) GiB"
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
CMD=("$QUANTIZE"
  --allow-requantize
  --tensor-type-file "$TYPES"
  --token-embedding-type Q6_K
  -t "$THREADS"
  "$IN" "$OUT" Q6_K)

say "输入: $IN"
say "输出: $OUT"
say "规则: $TYPES"
say "命令: ${CMD[*]}"

if [ "$DRY" = "1" ]; then
  exit 0
fi

say "开始量化（本机约 12 分钟）…"
START=$(date +%s)
"${CMD[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

if [ "$RC" != "0" ]; then
  say "[FAIL] llama-quantize 退出码 $RC，日志: $LOG"
  exit "$RC"
fi

if [ ! -f "$OUT" ]; then
  say "[FAIL] 没有生成输出文件，日志: $LOG"
  exit 1
fi

GOT=$(stat -c%s "$OUT")
say "完成，用时 ${ELAPSED}s，输出 $GOT 字节（$(awk -v n="$GOT" 'BEGIN{printf "%.2f", n/1073741824}') GiB）"
if [ "$GOT" != "$EXPECT_BYTES" ]; then
  say "[warn] 大小与参考值 $EXPECT_BYTES 不同。"
  say "       若规则数是 240/160/80，通常是 llama.cpp 版本差异；先做加载测试。"
else
  say "[ok] 大小与参考构建一致"
fi

say "下一步: LLAMA_BIN=$ROOT/llama.cpp/build-nccl/bin/llama-server bash scripts/run-tp.sh"
