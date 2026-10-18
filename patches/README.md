# patches/ —— 让 llama.cpp 跑起 Bolt 74B 的 4 个补丁

这四个补丁把上游 llama.cpp 缺的四件事补上。它们**只对下面这个 commit 验证过**：

```
69f201a2051ea9b9e9b50c3cc56afd9e2ae64414
model : support MiniCPM-V 4.7 (#29416)
```

上游之后的改动会让补丁冲突。升级前先 `git apply --check`。

| 补丁 | 改的文件 | 作用 | 不打会怎样 |
|---|---:|---|---|
| `0001-nemotron-per-layer-expert-count.patch` | 1 | 按**每层路由器张量的实际形状**读取专家数（240/304/368/432/496） | 模型加载失败：`expected 4096,496 got 4096,240` |
| `0002-tensor-parallel.patch` | 11 | 给 Mamba2/SSM 层实现张量并行；`LLAMA_SM_TENSOR_ALLOW_ARCH=1` 显式开门 | `-sm tensor` 被架构门禁拒绝 |
| `0003-quantize-same-type.patch` | 1 | 重量化时跳过"目标类型 == 源类型"的张量 | tpfit 配方会重算 160 个专家张量，且 IQ 系列缺 imatrix 时报错 |
| `0004-vision-nemotron-omni.patch` | 9（含 1 个新文件） | 识别 `nemotron_omni` 视觉投影器 | `-mm mmproj-BF16.gguf` 报 unknown architecture |

合计：**21 个文件被修改 + 1 个文件新增**。

---

## 1. 应用

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 69f201a2051ea9b9e9b50c3cc56afd9e2ae64414

for p in ../patches/*.patch; do
  git apply --check "$p" && git apply "$p"
done
```

没有 git 时也可以用：

```bash
for p in ../patches/*.patch; do patch -p1 < "$p"; done
```

补丁修改的文件互不重叠，应用顺序无所谓；按编号只是为了看日志方便。
`0004` 会新建 `tools/mtmd/models/nemotron-omni.cpp`，用 `git apply` 时会自动创建。

---

## 2. 验证

```bash
# 应该输出 22（21 改 + 1 新增）
git status --short | wc -l
```

更严格的逐字节验证：

```bash
cd ..
git -C llama.cpp stash list        # 确认没有别的未提交改动
sha256sum llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu
```

本仓库发布前，4 个补丁在**干净的** `69f201a` 上依次应用，
21 个文件与新增文件全部逐字节比对通过，补丁本身无 CRLF。

---

## 3. 每个补丁在讲什么

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
2. **ubatch 单序列**：SSM 状态回写断言 `K == 1`。TP + `-np 4` 实测可跑到
   82.9 tok/s 聚合，但多个序列合进同一 ubatch 时会触发断言。
3. **默认关闭**：不设 `LLAMA_SM_TENSOR_ALLOW_ARCH=1` 时行为与上游一致。
4. **许可**：补丁是 llama.cpp（MIT）的衍生作品，见
   [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)。
5. **上游优先**：如果未来上游正式支持 `nemotron_h_moe` 的张量并行，
   应改用上游实现，而不是继续维护这些补丁。
