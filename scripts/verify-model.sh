#!/usr/bin/env bash
# verify-model.sh —— 校验下载的模型文件（大小 + SHA256）
#
# 用法:
#   bash scripts/verify-model.sh            # 校验主模型
#   bash scripts/verify-model.sh --all      # 主模型 + mmproj
#   FILE=xxx.gguf EXPECT_SHA=<sha> bash scripts/verify-model.sh
#
# 环境变量:
#   DIR   模型目录   默认 <仓库>/model/bolt74b
#   FILE  文件名     默认主模型
#   EXPECT_SHA 期望值；换 FILE 时请显式提供，否则跳过 SHA
#
# 退出码: 0 = 通过，3 = 缺失/大小或 SHA 不符
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${DIR:-$ROOT/model/bolt74b}"
DEFAULT_FILE="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf"
DEFAULT_SHA="30c4eac4cdd63a7c73fcf29254a9fd3f40ab7ef09b555c1dd6653140cc94108f"
DEFAULT_SIZE=22018326464
MM_FILE="mmproj-BF16.gguf"
MM_SHA="acad03175d80ce7caea6854795e9ecc978e86d4b4caae0125efba5f970216f4f"
MM_SIZE=1733692672

if [ -n "${FILE:-}" ]; then
  if [ "${EXPECT_SHA+x}" != "x" ]; then EXPECT_SHA=""; fi
  if [ "${EXPECT_SIZE+x}" != "x" ]; then EXPECT_SIZE=""; fi
else
  FILE="$DEFAULT_FILE"
  EXPECT_SHA="${EXPECT_SHA-$DEFAULT_SHA}"
  EXPECT_SIZE="${EXPECT_SIZE-$DEFAULT_SIZE}"
fi

check_one() {
  local file="$1" sha="$2" size="$3" path="$DIR/$1" got
  echo "== $file"
  if [ ! -f "$path" ]; then
    echo "  缺失: $path"
    return 1
  fi
  got=$(stat -c%s "$path")
  echo "  大小: $got 字节（$(awk -v n="$got" 'BEGIN{printf "%.2f", n/1073741824}') GiB）"
  if [ -n "$size" ] && [ "$got" != "$size" ]; then
    echo "  大小不符: 期望 $size"
    return 1
  fi
  if [ -z "$sha" ]; then
    echo "  跳过 SHA（未提供期望值）"
    return 0
  fi
  echo "  计算 SHA256（约 1–2 分钟）…"
  got=$(sha256sum "$path" | awk '{print $1}')
  if [ "$got" = "$sha" ]; then
    echo "  SHA256 OK: $got"
    return 0
  fi
  echo "  SHA256 FAIL"
  echo "    期望 $sha"
  echo "    实际 $got"
  return 1
}

rc=0
check_one "$FILE" "$EXPECT_SHA" "$EXPECT_SIZE" || rc=3

if [ "${1:-}" = "--all" ]; then
  check_one "$MM_FILE" "$MM_SHA" "$MM_SIZE" || rc=3
fi

[ "$rc" = "0" ] && echo "结果: 校验通过" || echo "结果: 未通过（exit $rc）"
exit "$rc"
