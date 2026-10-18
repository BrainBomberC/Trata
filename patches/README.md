# patches/ —— 让 llama.cpp 跑起 Bolt 74B 的 5 个补丁

补丁分**两组**，对应两个不同的上游基线。**不要混用**——它们不是"依次叠加的 5 个补丁"。

| 组 | 补丁 | 上游基线 | 用途 |
|---|---|---|---|
| **生产** | `0005-lilicorr-dflash.patch` | `5aeac37a35b44fde4bdffaec9159f2fcb253b622` | 层切分 + LiLiCorr 草稿 + 1M（54–55 tok/s） |
| 回退 | `0001`–`0004` | `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414` | 旧版 TP×3 引擎（无 LiLiCorr） |

为什么分成两组：`5aeac37`（上游 master，llama.cpp 0.7.0 / ggml 0.27）已经包含
meta 后端、逐层专家数支持、`nemotron_omni` 视觉，**只差 0005 里的四处改动**；
而 `69f201a` 是更早的快照，需要 0001–0004 才能用。

---

## 1. 生产组：0005（对 `5aeac37`）

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 5aeac37a35b44fde4bdffaec9159f2fcb253b622
git apply --check ../patches/0005-lilicorr-dflash.patch
git apply ../patches/0005-lilicorr-dflash.patch
```

| 文件 | 改动 | 为什么必须 |
|---|---|---|
| `src/models/mamba-base.cpp` | prefill 的大 ubatch 保持 `K=1`（只有验证批次用 `n_rs_seq+1`） | 开草稿后 `K` 会变成 9，预填充的 SSD kernel 退化——这是"预填只慢 2 倍"而不是更慢的关键 |
| `src/llama-arch.cpp` | 把 `NEMOTRON_H` / `NEMOTRON_H_MOE` 从 `-sm tensor` 的禁用名单移除 | TP 模式需要（层切分不需要，但补丁无害） |
| `src/llama-model.cpp` | SSM 整块权重/状态、`output.weight`、DFlash 草稿全部 MIRRORED；`n_head_kv==0` 的层跳过注意力粒度计算 | LiLiCorr 要在**全词表**上做 top-k；Mamba 层没有 KV 头，直接除零 |
| `ggml/src/ggml-backend-meta.cpp` | `GGML_OP_ARANGE` 返回 MIRRORED | 草稿构造候选位置时用到，meta 后端原本不认识这个算子 |

改动合计：**4 个文件，+50 / −5 行**。补丁里没有调试输出，可直接用于生产。

> 层切分（`-sm layer`，生产默认）本身不经过 meta 后端，所以 `llama-model.cpp`、
> `ggml-backend-meta.cpp`、`llama-arch.cpp` 的改动只有回到 TP 回退配置时才生效；
> `mamba-base.cpp` 的改动对两种切分都生效，是草稿预填充性能的关键。

---

## 2. 回退组：0001–0004（对 `69f201a`）

```
69f201a2051ea9b9e9b50c3cc56afd9e2ae64414
model : support MiniCPM-V 4.7 (#29416)
```

| 补丁 | 改的文件 | 作用 | 不打会怎样 |
|---|---:|---|---|
| `0001-nemotron-per-layer-expert-count.patch` | 1 | 按**每层路由器张量的实际形状**读取专家数（240/304/368/432/496） | 模型加载失败：`expected 4096,496 got 4096,240` |
| `0002-tensor-parallel.patch` | 11 | 给 Mamba2/SSM 层实现张量并行；`LLAMA_SM_TENSOR_ALLOW_ARCH=1` 显式开门 | `-sm tensor` 被架构门禁拒绝 |
| `0003-quantize-same-type.patch` | 1 | 重量化时跳过"目标类型 == 源类型"的张量 | tpfit 配方会重算 160 个专家张量，且 IQ 系列缺 imatrix 时报错 |
| `0004-vision-nemotron-omni.patch` | 9（含 1 个新文件） | 识别 `nemotron_omni` 视觉投影器 | `-mm` 报 unknown architecture |

合计：**21 个文件被修改 + 1 个文件新增**。

```bash
git checkout 69f201a2051ea9b9e9b50c3cc56afd9e2ae64414
for p in ../patches/000[1-4]-*.patch; do
  git apply --check "$p" && git apply "$p"
done
```

补丁修改的文件互不重叠，应用顺序无所谓；按编号只是为了看日志方便。
`0004` 会新建 `tools/mtmd/models/nemotron-omni.cpp`，用 `git apply` 时会自动创建。

验证：

```bash
# 应该输出 22（21 改 + 1 新增）
git status --short | wc -l
```

---

## 3. 每个补丁在讲什么（回退组）

详细原理、失败日志与代码位置见 **[docs/04-补丁说明.md](../docs/04-补丁说明.md)**。
这里只给最短版本：

### 0001 逐层专家数

GGUF 元数据只有一个标量 `expert_count = 496`，但 40 个 MoE 层实际是
240×12、304×6、368×9、432×10、496×3。补丁改为在建模和前向图里读
`ffn_gate_inp.weight` 的 `ne[1]`。

### 0002 张量并行（主体）

- `llama-arch.cpp`：增加 `LLAMA_SM_TENSOR_ALLOW_ARCH` 门禁；
- `llama-model.cpp`：把 `ssm_in` 的融合布局 `[z|x|dt]` 按 5 段切分，
  满足 `d_inner = 8192 = 8×1024`、`n_head = 128`、`head_dim = 64`；
- `ggml-backend-meta.cpp`：新增 `handle_ssm_scan`（`s/x/dt/A/B/C` 各在不同轴）；
- `ssm-scan.cu`、`mamba-base.cpp`：输出拆成 `y` 与 `state`，断言 `K == 1`；
- 诊断开关：`GGML_TP_NO_REDUCE`、`GGML_TP_AR_STATS`、`GGML_META_*`（默认关闭）。

### 0003 同类型跳过

`llama_model_quantize_impl` 里发现 `target == current` 就直接复制原字节。
这让 tpfit 的 160 个 `iq2_xxs` 专家保持原样，不需要 imatrix。

### 0004 视觉

新增 `tools/mtmd/models/nemotron-omni.cpp` 并注册架构，处理
`mmproj-BF16.gguf` 的预处理参数与投影图。文本推理不需要它。

---

## 4. 边界

1. **实验性**：没有上游的测试覆盖，不保证跨模型/跨架构正确。
2. **两个基线**：0001–0004 只对 `69f201a` 验证，0005 只对 `5aeac37` 验证。
   升级上游前先 `git apply --check`；冲突就说明基线变了，需要重新对齐。
3. **回退组的 ubatch 限制**：SSM 状态回写断言 `K == 1`。TP + `-np 4` 实测可跑到
   82.9 tok/s 聚合，但多个序列合进同一 ubatch 时会触发断言。
   生产组（0005）没有这个限制。
4. **默认关闭**：不设 `LLAMA_SM_TENSOR_ALLOW_ARCH=1` 时行为与上游一致。
5. **许可**：补丁是 llama.cpp（MIT）的衍生作品，见
   [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)。
6. **上游优先**：如果未来上游正式支持 LiLiCorr 的层切分/TP 组合，
   应改用上游实现，而不是继续维护这些补丁。
