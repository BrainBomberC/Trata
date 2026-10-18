#!/usr/bin/env bash
# download-model.sh —— 从 ModelScope 分块并行下载 Bolt 74B GGUF，支持断点续传与 SHA256 校验
#
# 用法:
#   bash scripts/download-model.sh                 # 下载默认的 IQ2_XXS
#   bash scripts/download-model.sh --check         # 只取远端大小，不下载
#   FILE=mmproj-BF16.gguf DEST=model bash scripts/download-model.sh
#
# 环境变量:
#   REPO        仓库         默认 unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF
#   FILE        文件名       默认 NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf
#   DEST        保存目录     默认 <仓库>/model/bolt74b
#   PARTS       分块数       默认 4
#   EXPECT_SHA  期望 SHA256  默认主模型已知值；换 FILE 时自动置空；显式传空跳过校验
#   PROXY       HF 兜底代理  默认空，例如 http://127.0.0.1:7897
#   FORCE       1 = 跳过磁盘空间检查
#
# 为什么不用 curl --retry 续传:
#   curl 重试会从该次请求的起点重新下载并追加，导致分块内部重复数据。
#   本脚本每一轮都从 part 文件当前实际字节数继续请求剩余 Range。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF}"
DEFAULT_FILE="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf"
FILE="${FILE:-$DEFAULT_FILE}"
DEST="${DEST:-$ROOT/model/bolt74b}"
PARTS="${PARTS:-4}"
PROXY="${PROXY:-}"
FORCE="${FORCE:-0}"
DEFAULT_SHA="30c4eac4cdd63a7c73fcf29254a9fd3f40ab7ef09b555c1dd6653140cc94108f"

if [ "${EXPECT_SHA+x}" = "x" ]; then
  :                                   # 用户显式设置（包括空字符串）
elif [ "$FILE" = "$DEFAULT_FILE" ] && [ "$REPO" = "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF" ]; then
  EXPECT_SHA="$DEFAULT_SHA"
else
  EXPECT_SHA=""                        # 其它文件不知道 SHA，跳过
fi

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

MS_BASE="https://modelscope.cn/api/v1/models/$REPO/repo?Revision=master&FilePath="
HF_BASE="https://huggingface.co/$REPO/resolve/main/"
LOG="$ROOT/model/download.log"
FINAL="$DEST/$FILE"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for cmd in curl sha256sum awk stat; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "缺少命令: $cmd"; exit 1; }
done
mkdir -p "$DEST" "$(dirname "$LOG")"

get_size() {
  local i s
  for i in 1 2 3 4 5; do
    s=$(curl -s --noproxy '*' -L -I -r 0-0 --max-time 60 "$1" 2>/dev/null | tr -d '\r' \
        | awk -F'/' '/[Cc]ontent-[Rr]ange/ {print $2}' | head -1)
    if [ -n "$s" ] && [ "$s" -gt 1000000 ] 2>/dev/null; then
      echo "$s"; return 0
    fi
    sleep 2
  done
  return 1
}

SIZE=$(get_size "${MS_BASE}$FILE")
if [ -z "${SIZE:-}" ]; then
  say "[warn] ModelScope 取不到大小（$REPO/$FILE），尝试 HF"
  if [ -n "$PROXY" ]; then
    https_proxy="$PROXY" http_proxy="$PROXY" \
      curl -fL -C - --retry 10 --retry-delay 10 --retry-all-errors \
           -o "$FINAL" "${HF_BASE}$FILE" || exit 1
  else
    curl -fL -C - --retry 10 --retry-delay 10 --retry-all-errors \
         -o "$FINAL" "${HF_BASE}$FILE" || exit 1
  fi
  say "[ok] $FILE $(stat -c%s "$FINAL") 字节"
  [ -n "$EXPECT_SHA" ] && say "请手工校验: sha256sum '$FINAL'"
  exit 0
fi

say "远端 $FILE = $SIZE 字节（$(awk -v n="$SIZE" 'BEGIN{printf "%.2f", n/1073741824}') GiB）"
[ "$CHECK_ONLY" = "1" ] && exit 0

# 磁盘需要约 2 倍空间：分块 + 拼接后的最终文件
if [ "$FORCE" != "1" ]; then
  FREE_KB=$(df -Pk "$DEST" | awk 'NR==2 {print $4}')
  NEED_KB=$(( SIZE / 512 ))             # 2 × SIZE 字节 = SIZE/512 KB，再留 2%
  NEED_KB=$(( NEED_KB + NEED_KB / 50 ))
  if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt "$NEED_KB" ]; then
    say "[FAIL] 磁盘空间不足：需要约 $((NEED_KB/1048576)) GiB，可用 $((FREE_KB/1048576)) GiB"
    say "       清理后重试，或 FORCE=1 跳过检查（不推荐）"
    exit 1
  fi
fi

verify_sha() {
  [ -z "$EXPECT_SHA" ] && return 0
  say "校验 SHA256（约 1–2 分钟）…"
  local got
  got=$(sha256sum "$FINAL" | awk '{print $1}')
  if [ "$got" = "$EXPECT_SHA" ]; then
    say "[sha256 OK] $got"; return 0
  fi
  say "[sha256 FAIL] 期望 $EXPECT_SHA"
  say "              实际 $got"
  return 1
}

if [ -f "$FINAL" ] && [ "$(stat -c%s "$FINAL")" = "$SIZE" ]; then
  say "[skip] 文件已存在且大小正确"
  verify_sha && exit 0
  say "[warn] 内容校验失败，删除后重下"
  rm -f "$FINAL"
fi

CH="$DEST/.chunks-$(basename "$FILE" .gguf)"
mkdir -p "$CH"
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))

dl_part() {
  local i=$1 start end total part have now try=0
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
  total=$(( end - start + 1 ))
  part="$CH/part$i"
  have=$(stat -c%s "$part" 2>/dev/null || echo 0)
  if [ "$have" -gt "$total" ]; then
    truncate -s "$total" "$part"; have=$total
    say "  part$i 超长，已截断到 $total"
  fi
  say "  part$i 从 $((have/1048576)) / $((total/1048576)) MiB 继续"
  while [ "$have" -lt "$total" ]; do
    try=$(( try + 1 ))
    if [ "$try" -gt 80 ]; then
      say "  part$i FAIL 重试上限"; return 1
    fi
    curl -fsS --noproxy '*' -L --max-time 3600 \
         --speed-time 120 --speed-limit 102400 \
         -r $(( start + have ))-$end "$MS_BASE$FILE" >> "$part"
    now=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$now" -gt "$total" ]; then
      truncate -s "$total" "$part"; have=$total
      say "  part$i 超长，已截断"
    elif [ "$now" -le "$have" ]; then
      sleep 3
    else
      have=$now
    fi
  done
  say "  part$i OK $have 字节"
}

say "=== 下载 $FILE（$PARTS 路）==="
for i in $(seq 0 $(( PARTS - 1 ))); do dl_part "$i" & done
wait

total=0
for i in $(seq 0 $(( PARTS - 1 ))); do
  p="$CH/part$i"
  if [ ! -f "$p" ]; then say "[FAIL] part$i 缺失，可重跑续传"; exit 1; fi
  total=$(( total + $(stat -c%s "$p") ))
done
if [ "$total" != "$SIZE" ]; then
  say "[FAIL] 合计 $total ≠ $SIZE，可重跑续传"; exit 1
fi

say "拼接 → $FINAL"
: > "$FINAL.tmp"
for i in $(seq 0 $(( PARTS - 1 ))); do cat "$CH/part$i" >> "$FINAL.tmp"; done
if [ "$(stat -c%s "$FINAL.tmp")" != "$SIZE" ]; then
  say "[FAIL] 拼接后大小不符"; rm -f "$FINAL.tmp"; exit 1
fi
mv "$FINAL.tmp" "$FINAL"
rm -rf "$CH"
say "[ok] $FILE 完成（$(du -h "$FINAL" | awk '{print $1}')）"

if ! verify_sha; then
  say "文件已保留在 $FINAL，但内容不正确。删除后重跑本脚本。"
  exit 1
fi
