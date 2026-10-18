# 在 3 张 RTX 3060 上跑 74B 模型：一次把「不可能」拆成算术的工程实录

> 一台 2016 年的 Xeon 工作站 + 三张 12 GB 消费级显卡 + 46.9 GiB 内存 + 一块 43 MB/s 的 USB2 硬盘。
> 目标模型：NVIDIA **Nemotron-3.5-Bolt-74B-A9B**（`nemotron_h_moe`，74B 总参数、9B 激活）。
>
> 结果：**层切分 40.3 tok/s（200K 上下文）**，**自研张量并行移植 46.4 tok/s**，**4 路并发聚合 82.9 tok/s**。
> 本文把每一步的算术、改动、数据、以及"为什么到不了 70-100"都摊开写。

---

## 目录与内容地图

| 章节 | 你会看到什么 | 适合谁 |
|---|---|---|
| **一、第一性原理** | 自回归解码到底在花什么时间；层切分 vs 张量并行的带宽算术；显存/内存/磁盘三级存储的判据；这个模型为什么特殊（88 层 = 40 SSM + 40 MoE + 8 注意力） | 想理解"为什么快/为什么慢"的人 |
| **二、适用范围与局限** | 这套方法对哪些硬件、哪些模型有效；单流为什么卡在 46 tok/s；量化换速度的边界；明确列出"本机做不到 70-100"的证据 | 想判断"我的机器能不能复刻"的人 |
| **三、从 0 到 1 的部署流程** | 九步：测机器 → 选量化 → 编译引擎 → 打补丁 → 下载校验 → 量化 → 启动 → 测速 → 做成服务；**每一步要改哪个文件、哪个环境变量** | 想动手复刻的人 |
| **四、实测数据总表** | 所有量化/切分/参数组合的速度、预填充、PPL；kernel 级成本拆解；负结果（哪些优化试过但没用） | 想评估收益/避免踩坑的人 |
| **五、如何使用** | 网页控制台、OpenAI 兼容 API、并发、监控指标从哪来 | 想直接用的人 |

**一句话总结**：这不是"调参调到快"，而是先用算术判断"哪些路线必然失败"，把搜索空间从"所有可能"
压缩到"两三条能走的路"，再逐条打通。

---

# 一、第一性原理

## 1.1 自回归解码在花什么时间

大模型生成一个 token 是串行的：上一次的输出是下一次的输入。每一步要做的计算是：

```
h = W · x          # x 是上一层的激活（一个向量，batch=1）
```

**这一步是"内存密集型"而不是"计算密集型"**：矩阵 W 有几十亿个参数，向量 x 只有一个。
GPU 要把 W 的全部字节从显存读进计算单元，而乘加运算只有 `2 × |W|` 次。
在 360 GB/s 显存的 RTX 3060 上，读 1 GB 权重需要 **≈2.8 ms**，而对应的乘加只要几十微秒。

所以第一性原理的第一条：

> **decode 速度 ≈ 每生成一个 token 必须读取的权重字节数 ÷ 有效的显存带宽。**

实测校准：本机 IQ2 量化、层切分时，每 token 要读 **4.88 GiB** 权重，实测 40.27 tok/s，
反推有效带宽 = 4.88 GiB × 40.27 / s ≈ **197 GB/s**（单卡理论 360 GB/s 的 55%，这是 batch=1 时 GEMV
内核的现实效率）。

## 1.2 MoE 让"每 token 读多少"变得可控

Bolt 74B 是**混合专家（MoE）**模型：总共 74B 参数，但每个 token 只激活 9B。

- 稠密权重（SSM 投影、注意力、共享专家、LM head…）**每个 token 都要全读**；
- 专家权重（每层 240–496 个专家）**只读 top-9**。

所以每 token 的流量 = 稠密部分 + 9/专家数 × 专家部分。

对本模型（IQ2_XXS，每 token 4.88 GiB）：

| 组成 | 每 token 字节 |
|---|---:|
| SSM 输入投影（40 层，`ssm_in`） | 1.22 GiB |
| SSM 输出投影（`ssm_out`） | 0.54 GiB |
| LM head / 词嵌入 | 0.41 GiB |
| 共享专家 + latent + 注意力 | 1.11 GiB |
| 被选中的 9 个专家 × 40 层 | ≈0.36 GiB |
| **合计** | **≈4.88 GiB** |

**结论 1**：74B 的模型，每 token 实际只流 4.88 GiB —— 这是它能跑起来的根本原因。
120B 那个模型（专家集合 46.4 GiB）没有这个性质，见 2.3。

## 1.3 显存、内存、磁盘：三级存储的判据

权重可以放在三个地方，速度差三个数量级：

| 位置 | 本机实测有效带宽 | 22 GB 权重每 token 耗时 |
|---|---:|---:|
| 显存（VRAM，三卡合计 34.4 GiB） | 197 GB/s（单卡串行）/ 约 3×（并行读） | 25 ms（层切分） |
| 内存（DDR4，46.9 GiB） | 44.68 GB/s（顺序读） | ≈113 ms → 8.8 tok/s |
| 磁盘（USB2 转接盒） | **43.8 MB/s** | ≈8 分钟 → 0.002 tok/s |

**关键判据（本项目最重要的一条经验）**：

> **模型文件必须比物理内存小 6–8 GiB。**
> 否则 mmap 的权重页会被内核驱逐，每生成一个 token 都要从磁盘补页——不是"慢一点"，是**慢两个数量级**。

实测证据：

| 模型文件 | 大小 | 与 46.9 GiB 内存的关系 | 实测 |
|---|---:|---|---:|
| IQ2_XXS | 22.0 GB | 余量 24 GiB | **40.27 tok/s** |
| IQ4_XS | 43.4 GB | 余量 3.5 GiB | **11–14 tok/s**（推理时 sda 持续读 27–30 MB/s = 在掉盘） |
| 120B UD-IQ3_XXS | 57.3 GB | **超出内存 10 GiB** | **0.04 tok/s**（5 个 token 用了 126 秒） |

> 注意：mmap 的模型页记在 `Cached` 而不是 `used`。`free` 显示"used 只有 4–7 GiB"是记账假象，
> 要看 `MemFree`（只剩几百 MB）和 `Cached`（40 GiB）。

## 1.4 多卡并行：流水线 vs 并行

有两张卡以上时，怎么切模型决定了速度上限。

**层切分（layer split，llama.cpp 默认）** = 流水线：

```
token → [卡0: 第0-29层] → [卡1: 第30-58层] → [卡2: 第59-88层] → 输出
```

同一时刻**只有一张卡在工作**，另外两张在等。所以：

> 有效带宽 = **单卡**带宽（197 GB/s），另外两张卡的带宽完全浪费。

实测佐证：层切分解码时三卡占用率约 28% / 32% / 26%，**加起来 ~100%** ——机器不是在算，是在排队。

**张量并行（tensor split）** = 并行：

```
每一层的 W 被切成 3 份 → 三张卡同时算各自那份 → all-reduce 求和 → 下一层
```

> 有效带宽 ≈ **三张卡之和**，代价是每一层一次跨卡归约（all-reduce）。

对 27B 模型的公开实测（同一台机器）：层切分 35.2 s vs 张量并行 20.5 s（同 700 token），**+72%**。

**但张量并行不是免费的**，本项目的实测代价：

| 代价 | 实测 |
|---|---|
| 每 token 多出 80 次 all-reduce | **+4.4~6.0 ms**（NCCL ring，3 卡，每次 ~55–87 µs） |
| 预填充变慢 | 1489 → **940 tok/s（-37%）**：分割后的矩阵更窄，GEMM 效率下降 |
| 显存更紧 | 每卡还要放 all-reduce 缓冲与更碎的 compute buffer |

## 1.5 这个模型的特殊之处

`nemotron_h_moe` 不是标准的 Transformer。88 层是三种层交替：

| 层类型 | 数量 | 结构 |
|---|---:|---|
| **Mamba2 / SSM 层** | 40 | `ssm_in`(4096→18560) → conv1d → SSM scan → 分组 RMSNorm → `ssm_out`(8192→4096) |
| **MoE 层** | 40 | latent 投影 + 路由（240–496 个专家，top-9）+ 共享专家 |
| **全注意力层** | 8 | 普通 GQA 注意力 |

两个"坑"直接决定了部署难度：

1. **每层专家数不同**（240×12、304×6、368×9、432×10、496×3），而 GGUF 元数据只有一个标量
   `expert_count=496`。llama.cpp 0.6.0 会直接报
   `tensor 'blk.1.ffn_gate_inp.weight' has wrong shape; expected 4096,496 got 4096,240`。
   → 必须从上游 master 回补"逐层读路由器张量形状"的补丁（见 3.4）。

2. **SSM 层是纯顺序递归**：状态张量 `[d_state=128, head_dim=64, n_head=128]` 每层每 token 都要读写。
   张量并行时按"头"切分本可行，但前提是**切分必须按 SSM 组对齐**（`d_inner=8192 = 8 组 × 1024 通道`），
   否则 conv、scan、分组 norm 都会跨卡，切分推断直接断言失败。这是 TP 移植最难的一块（见 3.5）。

## 1.6 为什么先算再动手

模型下载要几小时（本机 11.6 MB/s），量化要几十分钟，一次完整的"下载→加载→测速"要半天。
所以本项目的第一件事不是下载，而是**把每 token 流量和带宽算清楚**：

```
每 token 字节 = 稠密权重 + 9/专家数 × 专家权重
理论 tok/s   = 有效带宽 ÷ 每 token 字节
```

这条公式在开工第一天就把 120B 判了死刑：专家集合 46.4 GiB > 可用内存 42 GiB，
每 token 必然掉 USB2 盘，理论 0.05 tok/s —— 与后来实测的 0.04 tok/s 完全吻合。

---

# 二、适用范围与局限

## 2.1 对哪些硬件有效

| 条件 | 要求 | 说明 |
|---|---|---|
| GPU 数量 | ≥2 张才谈得上并行；**≥3 张**才有明显收益 | 单卡没有切分问题 |
| 单卡显存 | 能放下 `模型/N + KV cache + compute buffer` | 本机 12 GB × 3，N=3 时每卡约 7.5 GB |
| **KV 头数** | 每张卡至少分到 **1 个 KV 头**，否则 KV 缓存只能每卡镜像 | 本模型只有 **2 个 KV 头** → 3 卡下 KV 镜像，1M 上下文每卡多 2.3 GiB（这是 TP+1M 必须换瘦身模型的原因） |
| NVLink / P2P | **不需要** | 本机三张 3060 无 NVLink、PCIe Gen3 x8/x4/x8，张量并行照样工作 |
| 内存 | **模型文件 ≤ 物理内存 − 6~8 GiB** | 这是硬门槛，比显卡更关键 |
| 磁盘 | 随便多慢都能跑（只影响冷启动） | 本机 43.8 MB/s 的 USB2，冷启动 17–40 分钟 |
| CPU | 只需要能喂得动 PCIe | 老 Xeon 也够 |

**经验区间**：消费级多卡（2–4 张 12–24 GB）+ 中等内存（32–64 GiB），适合跑 **20–80B 的 2–4 bit MoE**。
再大就要内存 > 64 GiB，或者更快的内存带宽。

## 2.2 对哪些模型有效

| 模型类型 | 是否适用 | 原因 |
|---|---|---|
| 普通 Transformer / GQA | ✅ 张量并行是成熟功能，直接 `-sm tensor` | 上游已支持 |
| MoE（DeepSeek、Qwen-MoE、OLMoE…） | ✅ | 每 token 只读 top-k 专家，流量天然小 |
| 混合 SSM（Mamba2、Nemotron-H、Jamba…） | ⚠️ 上游**门禁关闭**；本仓库为 `nemotron_h_moe` 做了移植 | 见 3.5 |
| 单卡能装下的模型 | ❌ 不需要切分，切了只会更慢 | 通信是纯开销 |
| 对预填充延迟极敏感的长文本场景 | ⚠️ 张量并行预填充慢 26–37% | 层切分更适合"一次读 200K 冷提示" |
| 需要 70+ tok/s 单流的 70B 模型 | ❌ 本机做不到（见 2.4） | 带宽 + 通信两道墙 |

## 2.3 为什么 120B 不可行（一个算术结论）

| 项 | 数值 |
|---|---:|
| 120B 的专家集合 | **46.4 GiB** |
| 本机可用内存（留出系统） | ≈42 GiB |
| 结论 | **必然缺页，每 token 掉 USB2 盘** |
| 理论速度 | 0.05 tok/s |
| **实测** | **0.04 tok/s**（5 token / 126 s） |

不是"调一调还能救"，是算术上不可能。遇到这种结论要**立刻停手**，别再花时间下载/量化。

## 2.4 单流为什么卡在 46 tok/s（以及为什么到不了 70–100）

用 kernel 级 profiler（`nsys --cuda-graph-trace=node`）测出的每 token 成本：

| 类别 | 占比 | 换算到实测 21.6 ms/token |
|---|---:|---:|
| matmul / GEMV（读权重） | **52.7%** | ≈11.4 ms |
| **NCCL all-reduce** | **27.6%** | ≈6.0 ms |
| elementwise（norm/broadcast/quantize/cpy） | 14.3% | ≈3.1 ms |
| SSM scan / conv | 3.3% | ≈0.7 ms |

三块墙：

1. **带宽墙**：11.4 ms 是每卡 ~1.3 GB 权重 ÷ ~114 GB/s。分割后矩阵变窄，GEMV 效率从层切分的 211 GB/s
   降到 114 GB/s。**降量化精度救不了**（实测 IQ2 与瘦身版速度完全一样）。
2. **通信墙**：80 次 all-reduce/ token，每次 55–87 µs。换 NCCL 算法/协议（Tree、LL128）、
   换实现（不走 NCCL 走蝴蝶拷贝）实测**都更慢**（43.2 / 45.3 / 36.4 vs 46.1）。
3. **固定开销**：1800 多个小 kernel，每个 2.7 µs。

**被实测否定的三种加速思路**（不要再走）：

| 思路 | 实测结果 |
|---|---|
| kernel 融合（跳过 608 个 ADD / 289 个 MUL / 160 个 SCALE 测成本） | 45.78 → 46.6 / 46.04 / 45.97，**全在噪声内**。**已验证跳过确实生效**：跳过 ADD 后模型输出变成乱码（`"\n\n\n01\n\n0"`），但速度不变 → 这些小 kernel 根本不在关键路径上，被 matmul 带宽与屏障等待掩盖了 |
| 把整个 token 捕获成单张 CUDA graph（含 NCCL，节点 4601） | 能跑通，但 **44.0 tok/s（开归约）/ 56.1（关归约）**，比不捕获**慢 4%** |
| 投机解码（ngram） | 40.27 → 38.7 / 39.2 / 35.7，**无收益**（本模型无 MTP head） |

**唯一有效的横向扩展是批处理**：4 路并发时聚合 **82.9 tok/s**（每路 24.3）。
因为固定开销（通信 + 小 kernel）被 4 个序列摊薄了。

## 2.5 量化不是越狠越好

| 量化 | 文件 | 每 token | 实测 | 备注 |
|---|---:|---:|---:|---|
| IQ2_XXS | 22.0 GB | 4.88 GiB | **40.27** | 本项目采用 |
| IQ4_XS | 43.4 GB | 7.65 GiB | 11–14 | 超出内存余量 → 掉盘 |
| IQ4 稠密降精度 | 41.1 GB | 5.35 GiB | 未跑完 | 仍超内存余量 |

**质量代价**：IQ2_XXS 的 wikitext-2 PPL = **5.7589 ± 0.0357**（2048 上下文）。
2 bit 对 74B 模型是"能用"的水平，但不是"无损"。如果内存够，IQ3_XXS + 稠密降精度
（每 token 4.63 GiB）理论上能到 45–60 tok/s 且质量更好——本机受内存限制没走完。

---

# 三、从 0 到 1 的部署流程

下面每一步都标注**要改什么文件 / 什么环境**。完整脚本在 `scripts/`。

## 步骤 0：先测机器（10 分钟）

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv     # 几张卡、多少显存
free -g                                                   # 物理内存（判据：模型要比它小 6–8 GiB）
lsblk -o NAME,TRAN,SIZE,MODEL                             # 模型盘是不是 USB（决定冷启动时间）
uname -r; dpkg -l | grep nvidia-driver                    # 内核/驱动（本项目全程不动它们）
```

记下三件事：**可用显存总和**、**物理内存**、**单卡有效带宽**（可用 `scripts/pciebench/` 里的微基准）。

## 步骤 1：选量化（决定后面所有事）

规则：
1. 先按"内存 − 6~8 GiB"卡死文件上限；
2. 再按"单卡显存 ≥ 模型/N + KV + compute buffer"选卡数；
3. 最后在满足前两条的量化里选**精度最高**的。

本机：内存上限 ≈ 22–24 GB → 候选只有 IQ2_XXS(22.0) / IQ3_XXS(36.3，超了)；
三卡显存 34.4 GiB，22 GB 权重 + KV 全进显存，余量充足 → **IQ2_XXS**。

## 步骤 2：编译引擎（需要 CUDA toolkit）

```bash
# 系统没有 cmake/ninja，用现成 venv 里的
export PATH="/path/to/venv/bin:$PATH"

cd llama.cpp
cmake -B build-nccl -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-nccl -j 8 --target llama-server llama-quantize llama-perplexity
```

**必须验证 NCCL 真的链上了**（否则张量并行会退化成主机搬运，慢 4 倍）：

```bash
grep -o GGML_USE_NCCL build-nccl/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make
ldd build-nccl/bin/llama-server | grep nccl     # 必须出现 libnccl.so.2
```

NCCL 不在系统里？可以从 Python wheel 里取：

```bash
export LD_LIBRARY_PATH="/path/to/site-packages/nvidia/nccl/lib:$PWD/build-nccl/bin:$LD_LIBRARY_PATH"
```

**环境改动**：只需 `PATH`（cmake）和 `LD_LIBRARY_PATH`（NCCL）。**不动**驱动、内核、apt、DKMS。

## 步骤 3：确认引擎认不认这个架构（下载前！）

用 7.5 MB 的元数据探针，几秒钟验证：

```bash
bash scripts/probe-model-arch.sh model/probe/xxx-00001-of-00003.gguf
```

如果报 `unsupported model architecture` 或张量形状不匹配，**先打补丁再下载**，别等几十 GB 下完。

## 步骤 4：打补丁（本模型必须，普通模型可跳过）

### 4.1 逐层专家数（必打）

`Bolt 的 40 个 MoE 层专家数不同`，而 GGUF 只有标量。改 `src/models/nemotron-h.cpp` 两处（从上游 master 0.7.0 回补）：

```cpp
// 1) 创建张量时，用该层路由器张量的实际形状，而不是标量 n_expert
const auto * router_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i).str().c_str());
const int64_t n_expert_layer = router_meta ? router_meta->ne[1] : n_expert;

// 2) 前向图里同样改用张量形状
build_moe_ffn(..., model.layers[il].ffn_up_exps->ne[2], ...);
```

不打会直接报 `check_tensor_dims: tensor 'blk.1.ffn_gate_inp.weight' has wrong shape; expected 4096,496 got 4096,240`。

### 4.2 张量并行（可选，想更快才打）

上游 llama.cpp 的 `llm_arch_supports_sm_tensor()` 对 `NEMOTRON_H_MOE` 返回 `false`。本仓库的移植改了 7 个文件：

| 文件 | 改动 |
|---|---|
| `src/llama-arch.cpp` | 增加环境变量开关 `LLAMA_SM_TENSOR_ALLOW_ARCH=1`（默认仍走官方门禁） |
| `src/llama-model.cpp` | **Mamba2 层的切分规格**（见下） |
| `ggml/src/ggml-backend-meta.cpp` | 新增 `handle_ssm_scan`：SSM scan 各源的头在不同轴上（s=axis2、x=axis1、dt=axis0、B/C=axis1） |
| `src/models/mamba-base.cpp` | `build_mamba2_layer` 改用 `ggml_ssm_scan_y` + `ggml_ssm_scan_state` 两个输出 |
| `ggml/src/ggml.c` | `ggml_ssm_scan` 的 mode=2 输出形状按序列而非按槽位 |
| `ggml/include/ggml.h` / `ggml-backend*.h` | 新增 scan 拆分与整段捕获的接口声明 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 后端捕获接口实现 + 若干诊断开关 |

**核心设计（按 SSM 组对齐）**：

```
d_inner = 8192 = 8 组 × 1024 通道；n_head = 128 头；每组 16 头
切分粒度必须是 1024 的倍数（缓存用 131072 元素 = 16 头 × 8192）
```

融合投影 `ssm_in`（4096 → 18560）的布局是 `[z(8192) | x(10240) | dt(128)]`，
必须切成 **5 段**（z / x / B / C / dt），每段各自的粒度不同；否则视图会跨卡，
切分推断会报 `GGML_ASSERT(ret.axis != UNKNOWN) failed`。

## 步骤 5：下载并校验（几小时，11.6 MB/s）

```bash
EXPECT_SHA=<官方 sha256> PARTS=4 \
  bash scripts/fetch-bolt-quant.sh <repo> <文件名>
grep '\[sha256 ✅\]' <日志>      # 必须看到这一行
```

**踩坑**：下载中断后盲目续传过一次，分块大小全对、68 个采样点也全一致，但整体 SHA 不符——
最后只能整份重下。**永远以官方 SHA256 为准**。

## 步骤 6：（可选）重量化

目的是"专家保持原精度，只降稠密张量"，缩小每 token 流量：

```bash
python3 -I scripts/gen-expert-types.py model/bolt74b/<in>.gguf /tmp/types.txt
./build-nccl/bin/llama-quantize --allow-requantize \
  --tensor-type-file /tmp/types.txt --token-embedding-type Q6_K \
  <in>.gguf <out>.gguf Q4_K
```

注意：`--tensor-type-file` 必须放在位置参数**之前**；IQ 系列目标类型需要 imatrix，
若目标类型与现有类型相同则应跳过检查（本仓库打了补丁）。

## 步骤 7：启动

### 层切分（默认，简单稳定）

```bash
llama-server -m model/bolt74b/...-IQ2_XXS.gguf \
  -ngl all -sm layer -ts 1,1,1 -fa on \
  -ctk q8_0 -ctv q8_0 -c 204800 -ub 512 -t 17 -np 1 -fit off \
  --host 127.0.0.1 --port 8080
```

### 张量并行 + 1M 上下文（**当前生产配置**，需要 4.2 的补丁）

```bash
export LD_LIBRARY_PATH="<nccl>/lib:<llama.cpp>/build-nccl/bin:$LD_LIBRARY_PATH"
export LLAMA_SM_TENSOR_ALLOW_ARCH=1

llama-server \
  -m model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf \
  -ngl all -sm tensor -ts 1,1,1 -fa on \
  -ctk q4_0 -ctv q4_0 -c 1048576 -ub 128 \
  -t 17 -np 1 -fit off \
  -mm model/mmproj-BF16.gguf --no-mmproj-offload \
  --override-kv nemotron_h_moe.context_length=int:1048576 \
  --metrics --host 127.0.0.1 --port 8080 -lv 3
```

实测：**decode 45.8–46.4 tok/s，三卡同时 95/96/95% 满载**；prefill ≈ 706 tok/s（10K 提示）；
显存 11.1 / 11.3 / 11.8 GB。对比层切分 @1M：37.5 tok/s、三卡各约 33%（流水线）。

**为什么必须用 `-tpfit` 这份模型**（两个硬约束）：

1. **KV 头太少**：注意力层只有 **2 个 KV 头**（32 个 Q 头，GQA 16:1）。2 个 KV 头分不到 3 张卡上，
   张量并行只能把 KV 缓存**每卡镜像一份** —— 1M 上下文下每卡多 **2.3 GiB**。
2. 于是每卡预算 = 权重 7.5 GB + 镜像 KV 2.3 GiB + 计算缓冲 1.2 GiB ≈ 11 GB。
   原版 IQ2（22.0 GB）在 `-ub 128` 下差约 1 GiB 装不下；`-tpfit`（19.58 GB）正好放得下。

**其它约束与回退**：

| 情况 | 配置 | 实测 |
|---|---|---:|
| 显存不够、要原版 IQ2 的质量 | `-sm tensor -c 262144`（原生上下文） | 46.8 tok/s |
| 不需要三卡并行、要最快预填充 | `-sm layer` + IQ2 | 37.5 @1M / 40.3 @200K，prefill 1489 |
| 更小 ubatch | `-ub 64`（计算缓冲再省 ~0.3 GB） | 速度基本不变 |

> `-ub 256` 在 1M + TP 下会因 KQ mask 变大而 OOM（实测 device 2 申请 1292 MiB 失败）。

### 参数说明（每一条都有原因）

| 参数 | 为什么 |
|---|---|
| `-ngl all` | 所有权重进 GPU（22 GB / 3 卡装得下） |
| `-sm layer` / `-sm tensor` | 切分方式，见 1.4 |
| `-fa on` | FlashAttention，长上下文必需 |
| `-ctk/-ctv q8_0` | KV cache 量化；200K 上下文下 q8 只占 850 MiB。（**1M 用 q4_0**，否则装不下） |
| `-c 204800` | 目标上下文长度（模型原生 262144，可 `--override-kv` 放开到 1M，但质量未验证）。**当前生产用 1048576** |
| `-ub 512` | 物理批大小，影响 compute buffer。**TP + 1M 时最多 128**（256 会 OOM） |
| `-fit off` | **必须**：否则会先打开模型两次去"拟合显存"，白等一轮 |
| `-np 1` | 单槽；要多路并发改成 4（见第五章） |

### 用 RAM 和 GPU 一起装模型（当显存不够时）

这是本文的另一个核心技巧。llama.cpp 的权重加载有三级：

1. **全部进显存**：`-ngl all`，最快（本机 40.3 tok/s）。
2. **专家缓存**：`-cmoe --moe-cache-mib N` —— 稠密权重进显存，专家按需从**内存**换入。
   实测 35.67 tok/s（比全显存慢 12%）。
3. **按层放 CPU**：`-ncmoe N` —— 把 N 层专家留在内存。
   实测 `-ncmoe 8` = 35.0 tok/s。

**为什么本机最后没用它们**：三卡显存 34.4 GiB > 模型 22 GB，全进显存还有 4.6/2.8/2.4 GiB 余量，
没必要付那 12% 的代价。**但显存不够 24 GB 的机器，`-cmoe` 就是那条"能用内存补显存"的路**——
前提仍然是 1.3 的判据：文件 ≤ 内存 − 6~8 GiB。此外注意：

- `-cmoe` 与张量并行**互斥**（llama.cpp 直接 throw），只能配合层切分；
- `--moe-cache-mib` 开太大（如 22000）会把某张卡显存吃光，decode 反而掉到 3 tok/s；
- 专家页在内存里也会被驱逐，需要 `scripts/drop-dense-cache.py` 精准丢稠密张量的页缓存。

## 步骤 8：测速与质量

```bash
# decode（连发 4 轮看是否收敛）
curl -s --noproxy '*' http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"...","n_predict":512,"temperature":0}' | jq '.timings'

# 预填充（20K 提示）
#   看 timings.prompt_per_second
# PPL
./build-nccl/bin/llama-perplexity -m <model> -f /tmp/wikitext-2-test.txt \
  -c 2048 -ngl all -sm layer -ts 1,1,1 -fa on -ctk q8_0 -ctv q8_0 -fit off -t 17
```

**必测三项**：decode tok/s、200K 预填充耗时、PPL。只看 decode 会漏掉预填充回退（TP 慢 37%）。

## 步骤 9：做成服务（对外 8001）

```bash
# 模型服务（仅本机）
cat > ~/.config/systemd/user/bolt-serve.service <<'EOF'
[Unit]
Description=Bolt 74B inference
After=network.target
[Service]
Type=simple
Environment=LD_LIBRARY_PATH=<nccl>/lib:<llama.cpp>/build-nccl/bin
MemorySwapMax=0
ExecStart=<llama-server> -m <model.gguf> -ngl all -sm layer -ts 1,1,1 -fa on \
  -ctk q8_0 -ctv q8_0 -c 8192 -ub 512 -t 17 -np 1 -fit off --metrics --port 8080
Restart=on-failure
[Install]
WantedBy=default.target
EOF

# 网页 + OpenAI 兼容层（对外 0.0.0.0:8001）
cat > ~/.config/systemd/user/bolt-strata-ui.service <<'EOF'
[Unit]
Description=Bolt web UI (Strata-compatible)
After=network.target
[Service]
Type=simple
ExecStart=/usr/bin/python3 -I <repo>/scripts/serve-ui.py --host 0.0.0.0 --port 8001
Restart=always
[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now bolt-serve.service bolt-strata-ui.service
sudo loginctl enable-linger $USER      # 不登录也开机自启
```

---

# 四、实测数据总表

## 4.1 解码 / 预填充 / 质量

同口径：`-fa on -ctk/-ctv q8_0 -t 17`，单流，greedy。

| 模型 / 配置 | decode tok/s | 200K prefill | PPL(wikitext-2) |
|---|---:|---:|---:|
| **Bolt IQ2_XXS，全进显存，层切分** | **40.27** | ≈2.3 分钟 | **5.7589 ± 0.0357** |
| Bolt IQ2 + `-cmoe --moe-cache-mib 19000` | 35.67 | ≈8.7 分钟 | — |
| Bolt IQ2 + `-ncmoe 8` | 35.0 | — | — |
| Bolt IQ2 + ngram 投机 | 38.7 / 39.2 / 35.7 | — | — |
| Bolt IQ4_XS（43.4 GB，掉盘） | 3.03 → 11 → 14（随缓存策略） | 不可用 | — |
| **Bolt IQ2，张量并行 3 卡** | **46.37** | 940 tok/s（-37%） | 未测 |
| **Bolt tpfit，张量并行 + 1M 上下文（当前生产）** | **45.8–46.4** | 706 tok/s（10K 提示） | 未测 |
| Bolt IQ2，层切分 + 1M 上下文 | 37.5 | — | 未测 |
| Bolt IQ2，张量并行 + 原生 262K | 46.8 | — | 未测 |
| Bolt 瘦身版（tpfit），层切分 / TP | 35.8 / **46.1** | — | — |
| Bolt TP，2 卡 | 45.4 | — | — |
| Bolt TP，4 路并发（聚合） | **82.9**（每路 24.3） | — | — |
| Nemotron 120B（57 GiB，严重掉盘） | **0.04** | ≈34 天 | — |

## 4.2 张量并行的成本拆解（kernel profiler）

每 token 每张卡（3 卡并行，比例为实测，绝对值被追踪开销放大）：

| 类别 | 占比 | 调用/token/卡 | 平均 |
|---|---:|---:|---:|
| matmul / GEMV | 52.7% | 538 | 33.9 µs |
| **NCCL all-reduce** | **27.6%** | 81–110 | 86.8 µs |
| elementwise | 14.3% | 1831 | 2.7 µs |
| SSM scan / conv | 3.3% | 165 | 6.8 µs |

单卡 kernel 时间 0.282 / 0.292 / 0.257 s（第三张卡少干 12%，每次屏障另外两张都要等它）。

## 4.3 负结果（试过、无效，别重复）

| 手段 | 结果 |
|---|---|
| 跳过 608 个 ADD / 289 个 MUL / 160 个 SCALE | 45.78 → 46.6 / 46.04 / 45.97（噪声内） |
| 整 token 单 CUDA graph 捕获（4601 节点，含 NCCL） | 44.0（开归约）/ 56.1（关）→ **比不捕获慢 4%** |
| `NCCL_PROTO=LL128` / `NCCL_ALGO=Tree` / `Tree+Simple` | 45.3 / 43.2 / 42.7（默认 46.1 最好） |
| 不用 NCCL，走蝴蝶归约 | 36.4 |
| `NCCL_P2P_DISABLE=1` | 45.9（无变化） |
| 4 槽并发（层切分） | 33.2 聚合（比单流还低） |
| 降量化（每 token −18%）后跑 TP | 46.37 → 46.1（**完全没变**，已在开销墙） |

## 4.4 硬件实测上限

| 项 | 实测 |
|---|---:|
| 三卡可用显存合计 | 34.42 GiB |
| PCIe（Gen3） | x8 / x4 / x8；H2D 5.98 / 3.12 / 5.98 GB/s |
| 单卡 decode 有效带宽 | ≈197–211 GB/s（batch=1，GEMV 现实效率） |
| 主机内存顺序读 | 44.68 GB/s |
| 模型盘 | USB2，43.8 MB/s |
| 权重分布（IQ2 全显存） | 5.53 / 7.43 / 7.49 GiB + CPU 映射 0.54 GiB |
| 200K KV（q8_0） | 850 MiB |
| SSM 状态 | 164.68 MiB |
| compute buffer | 3.2 GiB（`-ub 256` 可压） |

---

# 五、如何使用

## 5.1 网页控制台

浏览器打开 `http://<机器IP>:8001/`：外观与交互与 [Strata](https://github.com/Niko1221/Strata) 完全一致。

- **Chat**：流式对话、思考过程折叠、Markdown、对话列表（存在浏览器本地）
- **Monitor**：实时 **Decode tok/s** 与 **Prefill tok/s**、每卡 GPU 利用率/显存/温度/功耗、CPU、内存、磁盘、
  请求表（每条请求的 prompt tokens / 复用数 / 输出数 / prefill 速率 / decode 速率 / 耗时）
- **About**：模型名、上下文长度、KV 类型、GPU/CPU/内存型号、OpenAI base URL
- **Images / MCP / 运行配置**：本机没有这些能力，对应的卡片会自动隐藏

## 5.2 OpenAI 兼容 API（任何电脑、任何客户端）

```bash
curl -N http://<IP>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","stream":true,
       "messages":[{"role":"user","content":"你好"}],
       "max_tokens":256,"temperature":0.8}'
```

- 基址：`http://<IP>:8001/v1`
- `model` 随便填（单模型服务）
- 支持 `stream`、`max_tokens`、`temperature`、`top_p`、`top_k`、`seed`
- `reasoning_effort: "none" | "low" | "medium" | "high"` 会被翻译成该模型的
  `chat_template_kwargs.enable_thinking`（关掉=不输出思考过程）

## 5.3 并发（提高聚合吞吐）

```bash
# 4 个槽位、总上下文 16K（每槽 4K）：聚合 ~83 tok/s
NP=4 CTX=16384 bash scripts/run-tp.sh
```

注意：这是"4 个人同时聊天"的体验（每路 ~24 tok/s）；单人流式仍然是 46。

## 5.4 实时参数是怎么算的

- **decode tok/s**：优先用 llama-server `/metrics` 的 `llamacpp:predicted_tokens_seconds`（引擎自算），
  请求结束时用流末 `timings` 校正；
- **prefill tok/s**：`prompt_seconds_total` 的差值（只算真正新读的 token，缓存命中计入 `reused` 列）；
- **GPU/CPU/内存**：每秒 `nvidia-smi` + `/proc` 采样，画成曲线（保留 90 秒窗口）。

## 5.5 常用运维命令

```bash
bash scripts/run-tp.sh --daemon          # 启动模型（后台，等就绪）
bash scripts/run-tp.sh --status
bash scripts/run-tp.sh --stop
bash scripts/start-ui.sh                 # 启动网页/API 代理
bash scripts/start-ui.sh --stop
journalctl --user -u bolt-serve.service -n 100    # systemd 模型日志
journalctl --user -u bolt-strata-ui.service -n 50 # systemd 网页日志
```

---

## 附：这个项目留下的三条可复用经验

1. **先算后做。** 每 token 流量 ÷ 带宽 = 速度上限。它能在下载之前就否掉 120B，省下十几个小时。
2. **判据比调参重要。** "文件 ≤ 内存 − 6~8 GiB" 这一条就决定了用哪个量化、走哪条路线。
3. **瓶颈会转移。** 层切分是带宽墙；上了张量并行就变成通信墙 + 固定开销墙。
   每次提速后都要重新做一次成本拆解，否则会对着已经消失的墙优化。

---

## 附：当前生产配置速查（2026-10-18）

```
模型   model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf
切分   -sm tensor -ts 1,1,1   (环境变量 LLAMA_SM_TENSOR_ALLOW_ARCH=1)
上下文 -c 1048576  -ctk q4_0 -ctv q4_0  --override-kv nemotron_h_moe.context_length=int:1048576
批     -ub 128   (256 会 OOM：1M 下 KQ mask 变大)
视觉   -mm model/mmproj-BF16.gguf --no-mmproj-offload   (编码器放 CPU，省显存)
实测   45.8–46.4 tok/s，三卡 95/96/95%，prefill ≈706 tok/s，显存 11.1/11.3/11.8 GB
```

**为什么是这份模型**：注意力层只有 2 个 KV 头，3 卡无法按头切分 → KV 缓存每卡镜像（1M 下 2.3 GiB）。
每卡预算被压到 ~11 GB，原版 IQ2 差约 1 GiB，`-tpfit` 刚好放得下。

**回退**：

| 想要 | 换什么 | 实测 |
|---|---|---:|
| 更好的稠密精度 | 原版 IQ2 + `-sm layer`（@1M） | 37.5 tok/s |
| 同样的 TP 但要原版模型 | `-c 262144`（原生上下文） | 46.8 tok/s |
| 更快的预填充 | `-sm layer`（prefill 1489 vs 706） | 40.3 @200K |
