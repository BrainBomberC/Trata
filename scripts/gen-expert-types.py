#!/usr/bin/env python3
"""gen-expert-types.py —— 从 GGUF 头部生成逐张量类型清单，供 llama-quantize --tensor-type-file 使用。

三种模式：

  （默认）保留专家原类型
      这些 AD-/UD- 量化是混合精度的，专家张量里混着 iq3_s / iq3_xxs / q4_K / iq4_xs。
      只把「非专家」部分降精度时，用清单把专家钉回原类型，避免默认类型统一改写。
      输出：名字里含 'exps' 或 'shexp' 的张量（稀疏专家 + 共享专家）。

  --all
      连非专家张量也按原类型列出。

  --tpfit
      本项目生成 19.58 GiB tpfit 的固定配方：
        40 × ssm_in.weight   -> q3_K
        40 × ssm_out.weight  -> q3_K
        160 × 专家张量        -> iq2_xxs（80 个 *_exps + 80 个 *_shexp；
                                 源文件里本来就是，配合 0003 补丁直接跳过）
      其余张量不列出，由 llama-quantize 的默认目标 Q6_K 处理。
      共 240 条规则。

用法:
    python3 -I scripts/gen-expert-types.py <模型.gguf> <输出清单.txt> [--all | --tpfit]

输出格式（llama-quantize --tensor-type-file 要求）:
    blk.1.ffn_down_exps.weight=q4_K
    blk.1.ffn_up_exps.weight=iq4_xs
    ...
"""
import struct
import sys

# GGUF KV 值类型的字节宽度（type 9=数组 单独处理）
KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: 1, 9: 1, 10: 8, 11: 8, 12: 8}
# ggml_type -> llama-quantize 认识的类型名（必须与 ggml_type_name 一致）
TYPE_NAME = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
    10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K", 15: "q8_K",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
    22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4", 40: "nvfp4",
    41: "q1_0", 42: "q2_0",
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    path, out = sys.argv[1], sys.argv[2]
    opts = sys.argv[3:]
    want_all = "--all" in opts
    want_tpfit = "--tpfit" in opts
    if want_all and want_tpfit:
        print("--all 与 --tpfit 不能同时使用")
        return 2

    with open(path, "rb") as f:
        head = f.read(16 * 1024 * 1024)          # 头部（含 tokenizer）远小于 16 MiB
    if head[:4] != b"GGUF":
        print("不是 GGUF 文件")
        return 1
    n_tensors, n_kv = struct.unpack_from("<QQ", head, 8)
    o = 24

    def read_str():
        nonlocal o
        n, = struct.unpack_from("<Q", head, o)
        o += 8
        s = head[o:o + n].decode("utf-8", "replace")
        o += n
        return s

    def skip_value(t):
        nonlocal o
        if t == 8:
            read_str()
        elif t == 9:
            at, = struct.unpack_from("<I", head, o); o += 4
            n, = struct.unpack_from("<Q", head, o); o += 8
            if at == 8:
                for _ in range(n):
                    read_str()
            else:
                o += n * KV_SIZE[at]
        else:
            o += KV_SIZE[t]

    for _ in range(n_kv):
        read_str()
        t, = struct.unpack_from("<I", head, o); o += 4
        skip_value(t)

    lines, counts = [], {}
    for _ in range(n_tensors):
        name = read_str()
        nd, = struct.unpack_from("<I", head, o); o += 4
        o += 8 * nd
        tt, = struct.unpack_from("<I", head, o); o += 4
        o += 8                                      # offset
        tname = TYPE_NAME.get(tt, str(tt))
        if want_tpfit:
            # tpfit 配方：SSM 投影降到 q3_K，专家保持 iq2_xxs，其余交给默认 Q6_K
            if name.endswith("ssm_in.weight") or name.endswith("ssm_out.weight"):
                target = "q3_K"
            elif "exps" in name or "shexp" in name:
                target = "iq2_xxs"
            else:
                continue
            lines.append("%s=%s" % (name, target))
            counts[target] = counts.get(target, 0) + 1
        elif want_all or "exps" in name or "shexp" in name:
            lines.append("%s=%s" % (name, tname))
            counts[tname] = counts.get(tname, 0) + 1

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("写出 %d 条规则 -> %s" % (len(lines), out))
    print("类型分布:", counts)
    if want_tpfit and len(lines) != 240:
        print("警告: tpfit 配方预期 240 条，实际 %d 条。"
              "请确认输入是 NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf" % len(lines))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
