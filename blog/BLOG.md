# 在 3 张 RTX 3060 上跑 74B 模型：一次把「不可能」拆成算术的工程实录

> 一台 2016 年的 Xeon 工作站 + 三张 12 GB 消费级显卡 + 46.9 GiB 内存 + 一块 43 MB/s 的 USB2 硬盘。
> 目标模型：NVIDIA **Nemotron-3.5-Bolt-74B-A9B**（`nemotron_h_moe`，74B 总参数、9B 激活）。
>
> 结果：**层切分 + NVIDIA DFlash 草稿模型 @1M 上下文 = 54.0–55.0 tok/s**（当前生产）；
> 回退配置（自研张量并行移植 + 1M、无草稿）46.6–47.7 tok/s；4 路并发聚合 82.9 tok/s。
> 本文把每一步的算术、改动、数据、以及"为什么到不了 70-100"都摊开写。

---

## 目录与内容地图

| 章节 | 你会看到什么 | 适合谁 |
|---|---|---|
| **一、第一性原理** | 自回归解码到底在花什么时间；层切分 vs 张量并行的带宽算术；投机解码为什么能摊薄权重读取；显存/内存/磁盘三级存储的判据；这个模型为什么特殊（88 层 = 40 SSM + 40 MoE + 8 注意力） | 想理解"为什么快/为什么慢"的人 |
| **二、适用范围与局限** | 这套方法对哪些硬件、哪些模型有效；单流不用草稿为什么卡在 46–47、草稿如何绕到 54–55；量化换速度的边界；明确列出"本机做不到 70-100"的证据 | 想判断"我的机器能不能复刻"的人 |
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

## 1.7 投机解码：把"每 token 读一遍权重"摊开

decode 的本质是：**生成 1 个 token，就要把稠密权重完整读一遍**。谁也没法让这次读取变快，
但可以让它**一次多产出几个 token**。

投机解码（speculative decoding）的做法：

1. 一个很小的**草稿模型**先连续猜 `n_max` 个 token；
2. 大模型把这些猜测拼成一个批次，**一次前向**同时算出每个位置"如果是自己会输出什么"；
3. 从第一个位置开始比对，相同就接受，第一个不同就用大模型的结果替换并停止。

```
加速比 ≈ (1 + 接受长度) / 1.2
```

本机用 NVIDIA 官方的 **DFlash** 草稿（1.11 B，`block_size=8`，只在目标第
`[2,19,36,52,69,86]` 层读取隐藏状态），1M 上下文 + `n_max=4` 实测：

| 项 | 数值 |
|---|---:|
| 接受长度 | 3.42 |
| 无草稿（层切分 @1M） | 40–47 tok/s |
| 有草稿（层切分 @1M） | **54.0–55.0 tok/s** |
| 温度 0.7 | 49–52 tok/s |

**为什么它能无损**：验证不是"让小模型打分"，而是大模型用自己的分布重新算一遍。
greedy 下被接受的 token 与不用草稿时**逐字节相同**；采样模式下 rejection sampling
保证输出分布不变。实测同一段代码提示，开/关草稿输出完全一致。

**为什么必须配层切分**：验证批次从 1 个位置变成 `n_max+1 = 5` 个，
张量并行的 all-reduce 数据量也变成 5 倍，正好吃掉草稿的收益；
层切分没有集合通信，批次变宽只是激活张量变宽。实测 TP + 草稿 @32K 只有 45.1 tok/s，
TP + 草稿 @1M 直接 OOM。详见 [docs/12](../docs/12-草稿模型与投机解码.md)。

**代价**：草稿也要把整个 prompt 过一遍，长 prompt 的预填充约慢一倍
（19K：1269 → 616 tok/s；1M：约 17 分钟 → 30 分钟以上）。
所以"一次读 200K 文档、只回几十个字"的场景，回退到无草稿的 TP 配置反而总时间更短。

---

# 二、适用范围与局限

## 2.1 对哪些硬件有效

| 条件 | 要求 | 说明 |
|---|---|---|
| GPU 数量 | ≥2 张才谈得上并行；**≥3 张**才有明显收益 | 单卡没有切分问题 |
| 单卡显存 | 能放下 `模型/N + KV cache + compute buffer` | 本机 12 GB × 3，N=3 时每卡约 7.5 GB |
| **KV 头数** | 每张卡至少分到 **1 个 KV 头**，否则 KV 缓存只能每卡镜像 | 本模型只有 **2 个 KV 头** → 3 卡下 KV 镜像，1M 上下文每卡多 2.25 GiB（这是生产改用层切分、并因此能放下草稿的原因） |
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

## 2.4 纯张量行为什么卡在 46 tok/s（以及为什么到不了 70–100）

> 这一节分析的是**回退配置**（TP×3、无草稿）。生产配置换了一条路：
> 层切分 + DFlash 草稿把单流做到 **54–55 tok/s**（见 1.7 与 4.1），
> 但下面这些"墙"依然解释了为什么单纯靠 TP 到不了 70–100。

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
| 投机解码（ngram，无草稿模型） | 40.27 → 38.7 / 39.2 / 35.7，**无收益**（ngram 只能匹配原文重复，本模型也没有 MTP head） |
| 投机解码（Nano-4B 通用草稿，`draft-simple`） | 未采用：该模式不设置 `n_rs_seq`，混合 Mamba 目标的状态回滚快照无效 |
| 投机解码（DFlash 专用草稿） | **层切分无草稿约 40–47 → 54–55 tok/s**；相对 TP 回退 46.6–47.7 约 +15%。需要另编一版支持 LiLiCorr 的引擎，见步骤 4 |

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

用 7.5 MB 的元数据探针，几秒钟验证：仓库里的脚本只读 GGUF 头部
（元数据 + 张量名/形状/类型），不需要权重。

```bash
python3 -I scripts/gen-expert-types.py model/probe/<第一个分片>.gguf /tmp/probe.txt --all
head -40 /tmp/probe.txt
```

确认架构名是 `nemotron_h_moe`、88 层、每层专家数在 240–496 之间，以及注意力层只有 2 个 KV 头。
架构名或张量形状不对，**先打补丁再下载**，别等几十 GB 下完。

## 步骤 4：打补丁（本模型必须，普通模型可跳过）

### 4.1 逐层专家数（生产基线已自带，不用打）

`Bolt 的 40 个 MoE 层专家数不同`（240×12、304×6、368×9、432×10、496×3），
而 GGUF 元数据只有一个标量。生产基线 `5aeac37` 的上游代码**已经按每层路由器张量的
实际形状读取专家数**，所以这一项不需要任何补丁。

只有回到回退组（`69f201a`）时才要打 `0001-nemotron-per-layer-expert-count.patch`，
它做的就是上游现在做的事：

```cpp
// 1) 创建张量时，用该层路由器张量的实际形状，而不是标量 n_expert
const auto * router_meta = ml.get_tensor_meta(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i).str().c_str());
const int64_t n_expert_layer = router_meta ? router_meta->ne[1] : n_expert;

// 2) 前向图里同样改用张量形状
build_moe_ffn(..., model.layers[il].ffn_up_exps->ne[2], ...);
```

不打会直接报 `check_tensor_dims: tensor 'blk.1.ffn_gate_inp.weight' has wrong shape; expected 4096,496 got 4096,240`。
**生产组看到这条报错，说明你 checkout 错了基线**，不是少打了补丁。

### 4.2 张量并行（回退组，0001–0004；基线 `69f201a`）

> 注意：这组补丁和下一节的 0005 是**两套基线**，不能打在同一个 checkout 上。
> 生产走 4.3，只有需要回退到 TP 时才用这组。

上游 llama.cpp 的 `llm_arch_supports_sm_tensor()` 对 `NEMOTRON_H_MOE` 返回 `false`。
本组四个补丁合计改动 **21 个文件 + 新增 1 个**；其中负责张量并行的
`0002-tensor-parallel.patch` 改动 **11 个文件**（下表只列关键几处）：

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

### 4.3 生产组：LiLiCorr / DFlash 草稿（`0005-lilicorr-dflash.patch`，基线 `5aeac37`）

上游 master 已经能加载 `dflash` 架构，但混合 Mamba + 层切分下仍缺四块拼图：

| 文件 | 改动作用 |
|---|---|
| `src/models/mamba-base.cpp` | 草稿验证时 `K=1` 的预填充路径：`K_rs = n_rs_seq (>0)+1`，`K = (n_seq_tokens > K_rs) ? 1 : K_rs` |
| `src/llama-arch.cpp` | 去掉 `NEMOTRON_H` / `NEMOTRON_H_MOE` 的 TP 门禁（只有回退组需要） |
| `src/llama-model.cpp` | SSM 权重/状态、`output.weight`、DFlash 相关张量标记为 MIRRORED；`n_head_kv>0` 才计算注意力粒度 |
| `ggml/src/ggml-backend-meta.cpp` | `GGML_OP_ARANGE` 也走 MIRRORED，否则草稿的 meta 后端图切分失败 |

打完在干净树上应该是：

```bash
git status --short | wc -l      # 4（生产组）
```

完整的逐条说明见 [docs/04](../docs/04-补丁说明.md)。

## 步骤 5：下载并校验（几小时，11.6 MB/s）

```bash
# 三个文件各下一次：主模型 / DFlash 草稿 / Bolt 仓库的视觉投影器
bash scripts/download-model.sh
PRESET=draft  bash scripts/download-model.sh
PRESET=mmproj bash scripts/download-model.sh

# 最后统一校验（大小 + SHA + 投影器张量数）
bash scripts/verify-model.sh --all
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

### 生产：层切分 + DFlash 草稿 + 1M（当前生产配置，需要 0005 补丁）

```bash
llama-server \
  -m model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf \
  -ngl all -sm layer -ts 1,1,1 -fa on \
  -ctk q4_0 -ctv q4_0 -c 1048576 -ub 64 \
  -t 17 -np 1 -fit off \
  -md model/draft/dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf \
  -ngld all --spec-type draft-dflash \
  --spec-draft-n-max 4 --spec-draft-n-min 0 \
  -mm model/mmproj-BF16.gguf --no-mmproj-offload \
  --override-kv nemotron_h_moe.context_length=int:1048576 \
  --metrics --host 127.0.0.1 --port 8080 -lv 3
```

实测：**decode 54.0–55.0 tok/s（接受长度 3.42）**；温度 0.7 时 49–52。
预填充被草稿拖慢约一倍：19K 提示 616 tok/s（无草稿 1269），1M 约 30 分钟以上。

要点：

- 主模型用**原版 IQ2**，不需要 `-tpfit`；
- `-sm layer` 是前提。TP + 草稿 @32K 只有 45.1 tok/s，TP + 草稿 @1M 直接 OOM；
- `-ub 64`：1M 下 128/256 会 OOM（验证批次本身就比普通 decode 宽）；
- 四行草稿参数缺一不可：`-md` / `-ngld` / `--spec-type` / `--spec-draft-n-max`。

### 回退：张量并行 + 1M，无草稿（需要 0001–0004 补丁）

```bash
export LD_LIBRARY_PATH="<nccl>/lib:<llama.cpp>/build-nccl/bin:$LD_LIBRARY_PATH"
export LLAMA_SM_TENSOR_ALLOW_ARCH=1

llama-server \
  -m model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf \
  -ngl all -sm tensor -ts 1.02,1.02,0.96 -fa on \
  -ctk q4_0 -ctv q4_0 -c 1048576 -ub 64 \
  -t 17 -np 1 -fit off \
  -mm model/mmproj-BF16.gguf --no-mmproj-offload \
  --override-kv nemotron_h_moe.context_length=int:1048576 \
  --metrics --host 127.0.0.1 --port 8080 -lv 3
```

实测：**decode 46.6–47.7 tok/s，三卡同时满载**；预填充比生产快一倍（长 prompt 场景的优势）。

**为什么 TP 的显存更紧**：注意力层只有 **2 个 KV 头**（32 个 Q 头，GQA 16:1）。
2 个 KV 头分不到 3 张卡上，张量并行只能把 KV 缓存**每卡镜像一份** ——
1M 上下文下每卡多 **2.3 GiB**，而层切分平均只有 **0.75 GiB**。
省下来的 1.5 GiB 正好够放草稿（约 0.93 GiB），这是"草稿必须配层切分"的内存原因。

**其它对照**：

| 情况 | 配置 | 实测 |
|---|---|---:|
| 200K 上下文、要更好的 KV 精度 | 生产 + `-c 204800 -ctk q8_0 -ctv q8_0` + `n_max=3` | 55.5 tok/s |
| 超长 prompt、只回少量 token | 回退 `-sm tensor -c 1048576` | 46.6–47.7，prefill 快一倍 |
| 显存不够 | `-sm layer` 无草稿 / 降 `-c` 到 262144 / KV 换 q4_0 | 40–47 |
| 关掉草稿做 A/B | 同生产参数去掉四行 `-md`/`--spec-*` | 约 40 tok/s |

> 1M + TP 下 `-ub 128/256` 会 OOM（compute buffer 与 KQ mask 变大），`-ub 64` 是安全值。

### 参数说明（每一条都有原因）

| 参数 | 为什么 |
|---|---|
| `-ngl all` | 所有权重进 GPU（22 GB / 3 卡装得下） |
| `-sm layer` / `-sm tensor` | 切分方式，见 1.4。**生产用 layer，回退用 tensor** |
| `-fa on` | FlashAttention，长上下文必需 |
| `-ctk/-ctv q4_0` | 1M 上下文的 KV 量化。200K/400K 建议换 `q8_0`（更准，也放得下） |
| `-c 1048576` | 目标上下文（模型原生 262144，用 `--override-kv` 放开到 1M） |
| `-ub 64` | 物理批大小，影响 compute buffer。**1M 下 128/256 会 OOM**；200K 可试 256 |
| `-fit off` | **必须**：否则会先打开模型两次去"拟合显存"，白等一轮 |
| `-np 1` | 单槽；要多路并发改成 4（见第五章） |
| `-md <草稿.gguf>` | 草稿模型路径。没有它就没有投机解码 |
| `-ngld all` | 草稿全部放 GPU；`0`（放 CPU）实测掉到 21.9 tok/s |
| `--spec-type draft-dflash` | 指定 DFlash 协议，不是通用 `draft-simple` |
| `--spec-draft-n-max 4` | 每步最多起草 4 个；200K/400K 用 3 更快（见 docs/12） |
| `--spec-draft-n-min 0` | 允许草稿放弃，避免短回复被拖长 |

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
- 专家页在内存里也会被驱逐，需要用 `vmtouch`/`posix_fadvise` 之类的工具
  精准丢掉稠密张量的页缓存（本机是临时脚本，未随仓库发布）。

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

仓库的 `systemd/` 目录里已经有三份可直接改路径的单元：

| 单元 | 作用 |
|---|---|
| `bolt-serve.service` | **生产**：层切分 + DFlash 草稿 + 1M |
| `bolt-serve-tp.service` | **回退**：TP×3 + 1M，无草稿（与生产互斥） |
| `bolt-strata-ui.service` | Trata 网页 + API 代理（8001） |

```bash
cp systemd/*.service ~/.config/systemd/user/
# 先按文件头注释把 <仓库路径> / <llama-server> / <NCCL> 改成自己的绝对路径
systemctl --user daemon-reload
systemctl --user enable --now bolt-serve.service bolt-strata-ui.service
sudo loginctl enable-linger $USER      # 不登录也开机自启
```

两个模型单元用 `Conflicts=` 互斥，切换时先 `systemctl --user stop` 再 start 另一个。
详见 [systemd/README.md](../systemd/README.md)。

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
| **Bolt IQ2，层切分 + DFlash 草稿 + 1M（当前生产）** | **54.0–55.0**（n_max=4，接受长度 3.42） | 约 460 tok/s（200K，413 s） | 未测 |
| Bolt 同上，温度 0.7 | 49–52 | — | — |
| Bolt 同上，1M + n_max=3 | 48.5–50.6 | — | — |
| Bolt 同上，200K + q8_0 KV + n_max=3 | **55.5** | — | — |
| Bolt 同上，400K + q8_0 KV + n_max=3 | 55.2 | — | — |
| Bolt 层切分 + 草稿放 CPU（`-ngld 0`） | 21.9 | — | — |
| Bolt TP + 草稿 @32K（对照，勿用） | 45.1 | — | — |
| Bolt TP + 草稿 @1M（对照，勿用） | **OOM** | — | — |
| **Bolt IQ2，张量并行 3 卡（回退）** | **46.37** | 940 tok/s（-37%） | 未测 |
| **Bolt IQ2，张量并行 + 1M 上下文（回退）** | **46.6–47.7** | prefill 约为草稿配置的 2 倍 | 未测 |
| Bolt IQ2，层切分 + 1M 上下文（无草稿对照） | 37.5–40.3 | — | 未测 |
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
| TP + DFlash 草稿 @32K | 45.1（验证批次 5 倍放大 all-reduce，草稿收益被吃光） |
| TP + DFlash 草稿 @1M | 启动 OOM（CUDA0 分配草稿 944 MB 失败） |
| 草稿放 CPU（`-ngld 0`） | 21.9（草稿一步变成主机内存瓶颈，比不用草稿还慢） |
| 1M 下 `n_max=3` | 48.5–50.6（比 `n_max=4` 的 54–55 慢：摊薄不足） |
| Nano-4B 通用草稿（`draft-simple`） | 未采用（状态回滚快照无效，混合 Mamba 会算错） |

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

浏览器打开 `http://<机器IP>:8001/`：界面是 **Trata**，在 [Strata](https://github.com/Niko1221/Strata)（MIT）
基础上二次开发——灰紫主题、品牌、进度动效、Monitor 数据对接和本地后端代理都按本项目重做。

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
# 回退配置实测：4 个槽位、总上下文 16K（每槽 4K）：聚合 ~83 tok/s
NP=4 CTX=16384 bash scripts/run-tp.sh

# 生产配置同理，把 run-tp.sh 换成 run-draft.sh（并发数据未单独测）
NP=4 CTX=16384 bash scripts/run-draft.sh
```

注意：这是"4 个人同时聊天"的体验（每路 ~24 tok/s，回退配置）；单人流式为回退 46 / 生产 54–55。
4 路并发时草稿的接受率会变化，生产配置建议自己压测后再定 `NP`。

## 5.4 实时参数是怎么算的

- **decode tok/s**：优先用 llama-server `/metrics` 的 `llamacpp:predicted_tokens_seconds`（引擎自算），
  请求结束时用流末 `timings` 校正；
- **prefill tok/s**：`prompt_seconds_total` 的差值（只算真正新读的 token，缓存命中计入 `reused` 列）；
- **GPU/CPU/内存**：每秒 `nvidia-smi` + `/proc` 采样，画成曲线（保留 90 秒窗口）。

## 5.5 常用运维命令

```bash
bash scripts/run-draft.sh --daemon       # 生产：层切分 + 草稿 + 1M（后台，等就绪）
bash scripts/run-draft.sh --status
bash scripts/run-draft.sh --stop

bash scripts/run-tp.sh --daemon          # 回退：TP×3 + 1M，无草稿（先 stop 上一个）
bash scripts/run-tp.sh --status
bash scripts/run-tp.sh --stop

bash scripts/start-ui.sh                 # 启动网页/API 代理
bash scripts/start-ui.sh --stop
journalctl --user -u bolt-serve.service -n 100    # systemd 模型日志（生产）
journalctl --user -u bolt-serve-tp.service -n 100 # systemd 模型日志（回退）
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
模型   model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf      (原版 IQ2，不需要 tpfit)
草稿   model/draft/dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf  (998 MB)
视觉   model/mmproj-BF16.gguf --no-mmproj-offload                      (编码器放 CPU，省显存)
切分   -sm layer -ts 1,1,1
投机   -ngld all --spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-n-min 0
上下文 -c 1048576  -ctk q4_0 -ctv q4_0  --override-kv nemotron_h_moe.context_length=int:1048576
批     -ub 64   (1M 下 128/256 会 OOM)
引擎   上游 5aeac37 + 补丁 0005
实测   54.0–55.0 tok/s（接受长度 3.42）；温度 0.7 时 49–52；
       200K prefill 约 460 tok/s，1M prefill 30 分钟以上
```

**为什么是层切分**：注意力层只有 2 个 KV 头，TP 下 KV 每卡镜像（1M 下 2.25 GiB），
层切分只有约 0.75 GiB —— 省下的 1.5 GiB 正好放草稿（0.93 GiB）。
而且草稿每步要验证 5 个位置，TP 的 all-reduce 也放大 5 倍，收益会归零。

**回退**（TP×3 + 1M，无草稿，引擎是 `69f201a` + 0001–0004）：

| 想要 | 换什么 | 实测 |
|---|---|---:|
| 更快的长 prompt 预填充 | `bash scripts/run-tp.sh --daemon` | 46.6–47.7 tok/s，prefill 约为生产的 2 倍 |
| 更好的 KV 精度 @200K | 生产 + `-c 204800 -ctk q8_0 -ctv q8_0` + `n_max=3` | 55.5 tok/s |
| TP + 原生 262K（不追 1M） | `-c 262144` | 46.8 tok/s |
| A/B 对照：关草稿 | 生产参数去掉 `-md` / `--spec-*` | 约 40 tok/s |
