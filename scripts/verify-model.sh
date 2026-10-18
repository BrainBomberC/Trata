#!/usr/bin/env bash
# verify-model.sh —— 校验下载的模型文件（大小 + SHA256）
#
# 用法:
#   bash scripts/verify-model.sh            # 校验主模型
#   bash scripts/verify-model.sh --draft    # 校验草稿模型
#   bash scripts/verify-model.sh --mmproj   # 校验视觉投影器（并检查张量数）
#   bash scripts/verify-model.sh --all      # 三个都校验
#
# 环境变量:
#   DIR        模型目录  默认 <仓库>/model/bolt74b
#   FILE       文件名    默认主模型
#   EXPECT_SHA 期望值；换 FILE 时请显式提供，否则跳过 SHA
#
# 退出码: 0 = 通过（可能有 64 字节变体警告），3 = 缺失/大小或 SHA 不符
#
# 关于 64 字节变体:
#   托管方更新元数据后，文件可能比文档记录的大/小 64 字节，张量清单完全一致。
#   本脚本对"大小差 ≤64 字节"只给警告，不判失败；差更多才失败。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${DIR:-$ROOT/model/bolt74b}"
DEFAULT_FILE="NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf"
DEFAULT_SHA="30c4eac4cdd63a7c73fcf29254a9fd3f40ab7ef09b555c1dd6653140cc94108f"
DEFAULT_SIZE=22018326464

DRAFT_FILE="dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf"
DRAFT_SHA="080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194"
DRAFT_SIZE=998242560
DRAFT_DIR="$ROOT/model/draft"

MM_FILE="mmproj-BF16.gguf"
MM_SHA="acad03175d80ce7caea6854795e9ecc978e86d4b4caae0125efba5f970216f4f"
MM_SIZE=1733692672
MM_DIR="$ROOT/model"

if [ -n "${FILE:-}" ]; then
  if [ "${EXPECT_SHA+x}" != "x" ]; then EXPECT_SHA=""; fi
  if [ "${EXPECT_SIZE+x}" != "x" ]; then EXPECT_SIZE=""; fi
else
  FILE="$DEFAULT_FILE"
  EXPECT_SHA="${EXPECT_SHA-$DEFAULT_SHA}"
  EXPECT_SIZE="${EXPECT_SIZE-$DEFAULT_SIZE}"
fi

# 返回 0 通过 / 1 失败；64 字节变体设置 VARIANT=1
VARIANT=0
check_one() {
  local dir="$1" file="$2" sha="$3" size="$4" path got
  path="$dir/$file"
  echo "== $file"
  if [ ! -f "$path" ]; then
    echo "  缺失: $path"
    return 1
  fi
  got=$(stat -c%s "$path")
  echo "  大小: $got 字节（$(awk -v n="$got" 'BEGIN{printf "%.2f", n/1073741824}') GiB）"
  if [ -n "$size" ] && [ "$got" != "$size" ]; then
    local diff=$(( got > size ? got - size : size - got ))
    if [ "$diff" -le 64 ]; then
      echo "  [warn] 与文档记录差 $diff 字节（已知的元数据变体，见 docs/05 §4.2）"
      VARIANT=1
    else
      echo "  大小不符: 期望 $size（差 $diff 字节）"
      return 1
    fi
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
  if [ "$VARIANT" = "1" ]; then
    echo "  [warn] SHA 与文档不同（大小只差 ≤64 字节，见 docs/05 §4.2）"
    echo "         实际 $got"
    echo "         若张量清单一致即可使用；否则删除重下。"
    return 0
  fi
  echo "  SHA256 FAIL"
  echo "    期望 $sha"
  echo "    实际 $got"
  return 1
}

check_mm_tensors() {
  local path="$1" n
  command -v python3 >/dev/null 2>&1 || return 0
  n=$(python3 -I - "$path" <<'EOF'
import struct,sys
with open(sys.argv[1],'rb') as f:
    f.read(4); struct.unpack('<I',f.read(4))
    print(struct.unpack('<Q',f.read(8))[0])
EOF
)
  echo "  张量数: $n（Bolt 正确版应为 519）"
  if [ "$n" != "519" ]; then
    echo "  [FAIL] 这不是 Bolt 仓库那份投影器，图像会出错（见 model/README.md）"
    return 1
  fi
}

MODE="${1:-}"
rc=0
case "$MODE" in
  ""|--main)
    check_one "$DIR" "$FILE" "$EXPECT_SHA" "$EXPECT_SIZE" || rc=3
    ;;
  --draft)
    check_one "$DRAFT_DIR" "$DRAFT_FILE" "$DRAFT_SHA" "$DRAFT_SIZE" || rc=3
    ;;
  --mmproj)
    if check_one "$MM_DIR" "$MM_FILE" "$MM_SHA" "$MM_SIZE"; then
      check_mm_tensors "$MM_DIR/$MM_FILE" || rc=3
    else
      rc=3
    fi
    ;;
  --all)
    check_one "$DIR" "$FILE" "$EXPECT_SHA" "$EXPECT_SIZE" || rc=3
    check_one "$DRAFT_DIR" "$DRAFT_FILE" "$DRAFT_SHA" "$DRAFT_SIZE" || rc=3
    if check_one "$MM_DIR" "$MM_FILE" "$MM_SHA" "$MM_SIZE"; then
      check_mm_tensors "$MM_DIR/$MM_FILE" || rc=3
    else
      rc=3
    fi
    ;;
  -h|--help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "未知参数: $MODE（可用: --all / --draft / --mmproj / --help）" >&2
    exit 2
    ;;
esac

[ "$rc" = "0" ] && echo "结果: 校验通过" || echo "结果: 未通过（exit $rc）"
exit "$rc"
