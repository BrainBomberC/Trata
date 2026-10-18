# 三张 RTX 3060 跑 74B 大模型：从第一性原理到可复刻部署

> **模型**：NVIDIA **Nemotron-3.5-Bolt-74B-A9B**（74B 总参数 / 9B 激活，`nemotron_h_moe` 架构）
> **硬件**：3 × RTX 3060 12 GB（无 NVLink）+ 46.9 GiB 内存 + 普通 SATA/USB 硬盘
> **结果**：张量并行 3 卡 + 1M 上下文，**decode 45.8–46.4 tok/s**，三卡同时 95%+ 负载；4 路并发聚合 **82.9 tok/s**

这是一个**完整可复刻的部署工程**，不是一份"跑分截图"。仓库里有：

- 从上游 llama.cpp 生成、可逐字节验证的 **4 个补丁**（让 `nemotron_h_moe` 支持张量并行、逐层专家数、视觉投影器）
- 模型**下载地址 + 校验值**，以及把 22 GB 官方量化"裁剪"成 19.58 GiB 的**精确配方**
- 全部启动参数、网页/API 服务、实测数据与负结果
- 面向小白的**第一性原理**讲解：每一步为什么这样做，不这样做会怎样

> **本仓库不包含模型文件。** 模型有 20+ GB，且有自己的许可证。你按第 3 步自己下载、按第 4 步自己量化即可复刻。
> 这正是重点：**给你网址和方法，而不是给你一个网盘链接。**

---

## 目录

- [1. 先看结论：这套配置能做什么、不能做什么](#1-先看结论这套配置能做什么不能做什么)
- [2. 第一性原理：五条算术](#2-第一性原理五条算术)
- [3. 复刻路线（照着敲）](#3-复刻路线照着敲)
- [4. 目录地图](#4-目录地图)
- [5. 三个常见误解](#5-三个常见误解)
- [6. 实测结果](#6-实测结果)
- [7. 这台机器上的真实约束](#7-这台机器上的真实约束)
- [8. 详细文档](#8-详细文档)
- [9. 许可与致谢](#9-许可与致谢)

---

## 1. 先看结论：这套配置能做什么、不能做什么

| 项目 | 实测 |
|---|---:|
| 单人流式生成（当前生产配置） | **45.8–46.4 tok/s** |
| 4 人同时聊天（聚合吞吐） | **82.9 tok/s**（每人约 24 tok/s） |
| 200K 冷提示预填充（层切分） | 约 **2.3 分钟** |
| 10K 提示预填充（张量并行） | 约 **706 tok/s** |
| 1M 上下文 | 能装下（KV 缓存 q4_0，每卡镜像 2.3 GiB） |
| 模型质量（IQ2_XXS，wikitext-2 PPL） | **5.7589 ± 0.0357** |

**能做的事**：本地长期在线的中文/英文对话、长文档问答（200K–1M）、代码辅助、4 人以内的小团队共用。

**做不到的事（本机算术上不可能，不是没调好）**：

- 单流 70–100 tok/s：带宽墙 + 80 次/ token 的跨卡通信墙（见 [docs/09](docs/09-实测数据.md)）
- 跑 120B 模型：权重 57 GiB > 内存 46.9 GiB，每 token 掉盘，实测 **0.04 tok/s**
- 无损质量：2 bit 量化的 PPL 明显高于原始 BF16，适合"能用"，不适合"评测刷榜"

---

## 2. 第一性原理：五条算术

这一节是整个项目的地基。看懂了，后面所有参数你都能自己推。

### ① 生成速度由"读多少字节"决定，不是由"算多少次"决定

大模型生成是逐 token 串行的。每生成一个 token，GPU 要把这一层用到的**全部权重**从显存读一遍，
而真正的乘加运算相对很少：

```
每 token 时间 ≈ 必须读取的权重字节数 ÷ 有效显存带宽
```

RTX 3060 的理论显存带宽是 360 GB/s；batch=1 时实际 GEMV 内核只有约 **197–211 GB/s**（55%–59%）。
这就是为什么"换更快的显卡"比"换更快的 CPU"有用得多。

### ② MoE 让 74B 模型每 token 只读 4.88 GiB

Bolt 74B 是混合专家模型：总共 74B 参数，但每个 token 只激活 **9B**。

| 每 token 要读的部分 | 字节 |
|---|---:|
| SSM 输入/输出投影（40 层） | 1.76 GiB |
| LM head + 词嵌入 | 0.41 GiB |
| 共享专家 + latent + 注意力 | 1.11 GiB |
| 被选中的 9 个专家 × 40 层 | ≈0.36 GiB |
| **合计（IQ2_XXS）** | **≈4.88 GiB** |

4.88 GiB ÷ 197 GB/s ≈ 26 ms/token，对应约 38 tok/s，与实测层切分 40.3 tok/s 吻合。
**74B 不等于每 token 读 74B 的权重**——这是普通显卡能跑它的根本原因。

### ③ 显存 > 内存 > 磁盘，差三个数量级

| 权重放在哪 | 本机实测带宽 | 22 GB 模型每 token 耗时 |
|---|---:|---:|
| 显存 | 197 GB/s | ≈25 ms |
| 内存 | 44.7 GB/s | ≈113 ms（8.8 tok/s） |
| USB2 硬盘 | 43.8 MB/s | ≈8 分钟（0.002 tok/s） |

由此得到本项目最重要的一条判据：

> **模型文件必须比物理内存小 6–8 GiB。**

否则 mmap 的权重页会被内核驱逐，每生成一个 token 都要从磁盘补页——不是"慢一点"，是**慢两个数量级**。
实测：22 GB 模型 = 40 tok/s；43.4 GB 模型 = 11–14 tok/s；57 GB 模型 = 0.04 tok/s。

### ④ 多卡有两种切法：流水线 vs 真并行

**层切分（`-sm layer`，llama.cpp 默认）**：

```
token → [卡0: 第 0–29 层] → [卡1: 第 30–58 层] → [卡2: 第 59–87 层] → 输出
```

同一时刻只有一张卡在算，另外两张在等。三卡占用率约 28%/32%/26%，加起来 ~100%。
**有效带宽 = 单卡带宽**，另外两张卡的带宽被浪费。

**张量并行（`-sm tensor -ts 1,1,1`）**：

```
每一层的权重切成 3 份 → 三张卡同时算 → all-reduce 求和 → 下一层
```

**有效带宽 ≈ 三张卡之和**，代价是每层一次跨卡通信。本模型 88 层，每 token 约 **80 次 all-reduce**，
实测耗时 **+4.4–6.0 ms**（约占总时间 27.6%）。

### ⑤ 量化只解决"字节数"，解决不了"墙"

量化把每个权重从 16 bit 压到 2–6 bit，直接减少每 token 要读的字节。
但它有边界：

- 压到 2 bit 后，权重已经很小，**matmul 只占 52.7%**，通信和固定开销占 47.3%；
- 继续降精度（每 token 再少 18%）实测速度**完全不变**（46.37 → 46.1）——说明已经撞到通信墙。

所以"无脑量化"不是优化，**先判断当前瓶颈在哪**才是。

> 更完整的推导、KV 缓存公式、选型决策树 → **[docs/01-第一性原理.md](docs/01-第一性原理.md)** 和 **[docs/02-选型计算.md](docs/02-选型计算.md)**

---

## 3. 复刻路线（照着敲）

下面假设你把仓库放在 `~/Nemotron3Super`，Linux + NVIDIA 驱动 + CUDA toolkit 已装好。
**每一步都有对应的详细文档；第一次做建议边看文档边敲。**

### 第 0 步：确认你的机器够不够

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g
```

三条硬门槛：

1. **≥3 张 12 GB 显卡**（2 张也能跑，但收益小；见 docs/02）
2. **模型文件 ≤ 物理内存 − 6 GiB**
3. **每张卡能放下** `模型/N + KV 缓存 + 计算缓冲`（本配置约 11 GB/卡）

### 第 1 步：拿到本项目

```bash
git clone <你的仓库地址> Nemotron3Super
cd Nemotron3Super
```

### 第 2 步：编译带 NCCL 的 llama.cpp，并打补丁

```bash
# 上游固定版本（补丁只对这个 commit 保证逐字节一致）
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 69f201a2051ea9b9e9b50c3cc56afd9e2ae64414

# 依次应用 4 个补丁
for p in ../patches/*.patch; do
  git apply --check "$p" && git apply "$p"
done

# 编译（需要 CUDA toolkit 与 NCCL）
cmake -B build-nccl -DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-nccl -j 8 --target llama-server llama-quantize llama-perplexity

# 必须看到 libnccl.so.2，否则张量并行会退化
ldd build-nccl/bin/llama-server | grep nccl
cd ..
```

> 没有 NCCL？可以从 Python wheel 里取：`pip install nvidia-nccl-cu12`，然后把
> `site-packages/nvidia/nccl/lib` 加进 `LD_LIBRARY_PATH`。
> 编译踩坑（g++ 版本、CURL、NCCL 路径）→ [docs/03-编译引擎.md](docs/03-编译引擎.md)

### 第 3 步：下载官方模型（约 22 GB）

```bash
bash scripts/download-model.sh
```

默认从 ModelScope 下载 `unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` 的
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf`，4 路并行、断点续传、SHA256 校验。
下载完会在 `model/bolt74b/` 下。

需要视觉能力时再下载投影器：

```bash
FILE=mmproj-BF16.gguf DEST=model bash scripts/download-model.sh
```

### 第 4 步：把 22 GB 压成 19.58 GiB（tpfit）

```bash
bash scripts/quantize-tpfit.sh
```

它做的事（精确配方，不是"随手量化"）：

| 张量 | 数量 | 目标类型 |
|---|---:|---|
| `ssm_in.weight` / `ssm_out.weight` | 40 + 40 | `q3_K` |
| 稀疏专家 `ffn_*_exps` / 共享专家 `ffn_*_shexp` | 40 × 4 | `iq2_xxs`（保持原精度） |
| 其余稠密张量 | — | `q6_K`（默认目标） |

输出：`model/bolt74b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf`，**21,025,816,576 字节**。
在 3 卡 1M 上下文下，原版差约 1 GiB 装不下，tpfit 刚好放得下。

> 为什么不是"剪枝"？→ [docs/06-裁剪与量化.md](docs/06-裁剪与量化.md)

### 第 5 步：启动

```bash
# 生产配置：张量并行 3 卡 + 1M 上下文
LLAMA_BIN=$PWD/llama.cpp/build-nccl/bin/llama-server \
bash scripts/run-tp.sh
```

看到 `server is listening on 127.0.0.1:8080` 就好了。首次加载取决于磁盘速度（本机 USB2 要 20–40 分钟；
NVMe 几分钟；页缓存热了约 30 秒）。

### 第 6 步：网页 + OpenAI 兼容 API

```bash
bash scripts/start-ui.sh
```

打开 `http://<机器IP>:8001/`。API 基址 `http://<机器IP>:8001/v1`，`model` 字段随便填。

```bash
curl -N http://<机器IP>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","stream":true,
       "messages":[{"role":"user","content":"用一句话解释张量并行"}],
       "max_tokens":256,"temperature":0}'
```

也实现了 **Anthropic Messages API**，可直接把 Claude Code 指过来：

```bash
export ANTHROPIC_BASE_URL=http://<机器IP>:8001
export ANTHROPIC_AUTH_TOKEN=not-checked
claude
```

---

## 4. 目录地图

| 路径 | 内容 |
|---|---|
| `docs/` | **从小白到复刻的完整教程**（先读 `docs/00-阅读指南.md`） |
| `patches/` | 4 个补丁 + 说明；对上游 `69f201a` 逐字节验证通过 |
| `scripts/` | 下载、校验、量化、启动、网页服务；见 `scripts/README.md` |
| `ui/strata/` | 网页前端 Trata（基于 Strata，MIT） |
| `model/` | 模型放这里（**.gitignore 已排除**）；说明见 `model/README.md` |
| `environment/` | 运行时依赖与编译环境（已去掉个人路径） |
| `benchmarks/logs/` | 关键实测原始日志：tpfit 量化、TP 1M、层切分报告 |
| `blog/BLOG.md` | 长文技术实录（原理、边界、数据、负结果） |
| `systemd/` | 两个用户级服务单元模板（开机自启） |

---

## 5. 三个常见误解

### 误解 1："我把参数改成了 `tensor=3`"

llama.cpp **没有** `tensor=3` 这个参数。正确的写法是：

```bash
-sm tensor -ts 1,1,1
```

- `-sm tensor`：切分方式选"张量并行"（split mode = tensor）
- `-ts 1,1,1`：三张卡各分 1 份（tensor split）

另外还要设 `LLAMA_SM_TENSOR_ALLOW_ARCH=1`——上游对 `nemotron_h_moe` 默认关闭张量并行，
这个环境变量是补丁加的安全开关。**三者缺一不可。**

### 误解 2："我剪裁了模型"

本项目的 `-tpfit.gguf` 不是剪枝（没有删神经元、没有改结构），而是**混合精度重量化**：
专家保持 IQ2_XXS，稠密张量重新分配到 Q6_K/Q3_K，使文件从 22.0 GB 降到 19.58 GiB。
模型结构和参数量完全没变，变的是每个权重的存储精度。

### 误解 3："模型叫 netromn3.5 79ba9b"

正确名称是 **NVIDIA Nemotron-3.5-Bolt-74B-A9B**：

- `74B` = 总参数 74B
- `A9B` = 每个 token 激活 9B
- `79ba9b` 很可能是把 "74B-A9B" 看串了

---

## 6. 实测结果

同口径：`-fa on`、单流、greedy、`-t 17`。

| 模型 / 配置 | decode tok/s | 200K prefill | PPL |
|---|---:|---:|---:|
| **IQ2_XXS，全进显存，层切分** | **40.27** | ≈2.3 分钟 | **5.7589 ± 0.0357** |
| IQ2 + `-cmoe --moe-cache-mib 19000` | 35.67 | ≈8.7 分钟 | — |
| IQ2 + `-ncmoe 8` | 35.0 | — | — |
| IQ2 + ngram 投机解码 | 38.7 / 39.2 / 35.7 | — | — |
| **IQ2，张量并行 3 卡** | **46.37** | 940 tok/s | — |
| **tpfit，张量并行 + 1M（生产）** | **45.8–46.4** | 706 tok/s（10K） | — |
| IQ2，层切分 + 1M | 37.5 | — | — |
| IQ2，张量并行 + 原生 262K | 46.8 | — | — |
| tpfit，层切分 | 35.8 | — | — |
| 张量并行，2 卡 | 45.4 | — | — |
| **4 路并发（聚合）** | **82.9**（每路 24.3） | — | — |
| 120B（57 GiB，严重掉盘） | **0.04** | ≈34 天 | — |

**被实测否定、不要重复的优化**：kernel 融合（噪声内）、整 token CUDA graph（慢 4%）、
NCCL LL128/Tree（43.2–45.3）、蝴蝶归约（36.4）、`NCCL_P2P_DISABLE`（无变化）、
层切分 4 槽并发（33.2，比单流还低）。

> 完整表格与 kernel 级拆解 → [docs/09-实测数据.md](docs/09-实测数据.md)

---

## 7. 这台机器上的真实约束

| 约束 | 数值 | 后果 |
|---|---|---|
| 内存 | 46.9 GiB | 模型必须 ≤ ~40 GiB；22 GB 安全，43 GB 掉盘 |
| 单卡显存 | 12 GB | 1M + TP 时每卡预算约 11 GB，原版 IQ2 差 1 GiB |
| KV 头数 | **2** | 3 卡无法按头切 KV，缓存每卡镜像一份 |
| PCIe | Gen3 x8/x4/x8 | all-reduce 约 55–87 µs，80 次/token ≈ 6 ms |
| 模型盘 | USB2 43.8 MB/s | 冷启动 20–40 分钟；热缓存 30 秒 |
| `-ub` | **≤128** | 1M 上下文下 `-ub 256` 会让 KQ mask 变大，device 2 OOM |

**回退方案**（都不需要重新下载模型）：

| 你想要 | 换什么 | 实测 |
|---|---|---:|
| 更好的稠密精度 | 原版 IQ2 + `-sm layer` + 1M | 37.5 tok/s |
| TP 但用原版模型 | `-c 262144`（原生上下文） | 46.8 tok/s |
| 更快的预填充 | `-sm layer` | 1489 tok/s |
| 显存更紧 | `-ub 64` | 速度基本不变 |

---

## 8. 详细文档

| 文档 | 回答什么问题 |
|---|---|
| [docs/00-阅读指南.md](docs/00-阅读指南.md) | 我该按什么顺序读？需要什么基础？ |
| [docs/01-第一性原理.md](docs/01-第一性原理.md) | 为什么 decode 是带宽瓶颈？公式怎么用？ |
| [docs/02-选型计算.md](docs/02-选型计算.md) | 我的机器能跑多大模型？选哪个量化？ |
| [docs/03-编译引擎.md](docs/03-编译引擎.md) | 怎么装 CUDA/NCCL、怎么编译、怎么验证 |
| [docs/04-补丁说明.md](docs/04-补丁说明.md) | 4 个补丁各改了什么、为什么必须 |
| [docs/05-下载与校验.md](docs/05-下载与校验.md) | 模型从哪下、SHA 怎么对、断点续传怎么用 |
| [docs/06-裁剪与量化.md](docs/06-裁剪与量化.md) | tpfit 的精确配方、每个张量为什么这么选 |
| [docs/07-启动与调参.md](docs/07-启动与调参.md) | 每个启动参数的含义与边界 |
| [docs/08-网页与API.md](docs/08-网页与API.md) | 网页、OpenAI/Anthropic API、局域网访问 |
| [docs/09-实测数据.md](docs/09-实测数据.md) | 全部数据、profiler 拆解、负结果 |
| [docs/10-排错.md](docs/10-排错.md) | 报错信息 → 原因 → 处理 |
| [docs/11-术语表.md](docs/11-术语表.md) | MoE、SSM、TP、KV、GQA…都是什么 |

---

## 9. 许可与致谢

- 本仓库代码与文档：[MIT](LICENSE)
- 推理引擎：[llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）；补丁基于上游
  `69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`
- 网页前端：[Strata](https://github.com/Niko1221/Strata)（MIT），改名为 Trata 并调整主题
- 字体：Outfit（SIL Open Font License 1.1，见 `ui/strata/web/fonts/OFL.txt`）
- 模型：**NVIDIA Nemotron-3.5-Bolt-74B-A9B**，版权与使用条件归 NVIDIA；
  下载与商用前请阅读模型页面的原始许可证。本仓库**不再分发**模型权重。

详见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
