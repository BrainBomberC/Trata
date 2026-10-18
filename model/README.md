# model/ —— 模型制品与运行日志

这个目录存放**所有体积巨大的模型文件**和每一次实测留下的日志。它不进 git（见根目录 `.gitignore`），
只在本机存在；别人复刻时这一层需要自己重新下载/量化生成。

> **一句话说明当前生产用什么**：`bolt74b/…-IQ2_XXS.gguf`（主模型）
> + `draft/…-NVFP4.gguf`（LiLiCorr 草稿）+ `mmproj-BF16.gguf`（Bolt 仓库那份投影器）。

## 目录结构

| 路径 | 作用 |
|---|---|
| `bolt74b/` | **主模型目录**：Nemotron-3.5-Bolt-74B-A9B 的各种量化版本 |
| `draft/` | **草稿模型**（投机解码）：LiLiCorr(DFlash) 与 Nano-4B，见下 |
| `mmproj-BF16.gguf` | 视觉投影器。**必须来自 Bolt 仓库**（519 张量），见下面的警告 |
| `probe/` | 探针文件：只含 GGUF 头部的 7.5 MB 分片，用来在下载完整模型前验证引擎认不认这个架构 |
| `UD-IQ3_XXS/` | Nemotron-3.5-Super-VL-120B-A12B 的三分片量化（120B 阶段遗留，本机内存装不下，仅作历史与对照） |
| `*.log` / `*.report.txt` | 每次启动、测试、量化的原始日志与摘要报告 |

## model/bolt74b/ —— 主模型

| 文件 | 大小 | 说明 |
|---|---:|---|
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf` | 22.0 GB | **当前生产用**。层切分 + LiLiCorr 草稿 @1M = **54–55 tok/s**；无草稿时层切分 40.27、TP×3 46.6–47.7 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf` | 21.0 GB | 为 TP+1M 做的"瘦身版"（Q8_0→Q6_K、NVFP4→Q3_K）。**只有 TP+1M 无草稿时用**；层切分下比原版慢（35.8 vs 40.27） |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-denseQ6K.gguf` | 21.6 GB | 实验产物：稠密 Q8_0→Q6_K（**精度略降**，每 token 少读 ~8%） |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-AD-IQ4_XS.gguf` | 46.6 GB | IQ4_XS 原始件。**43.4 GiB 文件 > 本机内存余量，每 token 掉盘，实测 11–14 tok/s**，已弃用 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-AD-IQ4_XS-mini.gguf` | 40.2 GB | IQ4_XS 专家 + Q4_K 稠密。5.58 GiB/token；**1M+草稿下 12 GiB 专家必须放 RAM**，预估 25–30 tok/s（未实测） |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-AD-IQ4_XS-q3exps.gguf` | 35.9 GB | Q3_K 专家 + Q8/Q6 稠密。7.67 GiB/token；**1M+草稿下约 8 GiB 专家放 RAM**，预估 28–35 tok/s（未实测） |

> **选型判据**（贯穿整个项目）：**模型文件必须比机器物理内存小 6–8 GiB**，否则 mmap 的页会被内核驱逐，
> 每生成一个 token 都要从磁盘补页。20.5 GiB 的文件在 46.9 GiB 内存上没有任何问题；43.4 GiB 的文件立刻掉到 11–14 tok/s。

### 官方文件的两种版本（差 64 字节）

ModelScope 元数据里的主模型是 **22,018,326,464 字节 / SHA `30c4eac4…108f`**。
本机实测文件是 **22,018,326,528 字节 / SHA `91d63363…af75`**：843 个张量名、形状、类型完全一致，
只是 GGUF 元数据里某个字符串长度不同（差 64 字节），推理行为完全相同。

**处理原则**：以你下载时托管页面给出的 SHA 为准；只要张量清单一致、大小差几十字节，可以正常使用。
校验脚本会把这种情况单独提示，不会当成文件损坏。

## model/draft/ —— 草稿模型（投机解码）

| 文件 | 大小 | 说明 |
|---|---:|---|
| `dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf` | 998,242,560 | **生产在用**（上游原名）。NVIDIA 官方 DFlash 草稿（1.11B，`block_size=8`，`target_layers=[2,19,36,52,69,86]`），SHA `080e1ea4…0194`；接受长度 3.42（n_max=4）。本机另有一份同内容文件改名为 `…-24GB-LiLiCorr.gguf` |
| `NVIDIA-Nemotron-3-Nano-4B-UD-IQ2_XXS.gguf` | 2,182,496,672 | 通用 4B 草稿（走 `draft-simple`），**未采用**：这种模式不会设置 `n_rs_seq`，混合 Mamba 目标的回滚快照不成立 |

> 文档里叫它 **LiLiCorr 草稿**或 **DFlash 草稿**，指的是同一个东西；
> 上游仓库里的文件名是 `dflash-…-NVFP4.gguf`。启动时用 `-md` 指向它、`--spec-type draft-dflash` 开启。

## 视觉投影器：认仓库，不认文件名

| 来源 | 张量数 | 用不用 |
|---|---:|---|
| **Bolt 仓库** `unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` 的 `mmproj-BF16.gguf` | **519** | ✅ 正确 |
| 120B/Super 仓库里同名文件 | 393 | ❌ 能加载、不报错，但图像特征全错 |

两份文件大小只差几 KB，肉眼几乎分不出来。**唯一可靠的判据是来源仓库和 SHA**：

- Bolt 版：1,733,692,672 字节，SHA `acad0317…6f4f`（ModelScope 元数据）；
  本机那份是 1,733,692,736 字节 / SHA `0ee891ea…26b0`，同样只差 64 字节。
- 用 `PRESET=mmproj bash scripts/download-model.sh` 下载会自动校验，装错直接失败。

检查张量数：

```bash
python3 - <<'EOF' model/mmproj-BF16.gguf
import struct,sys
with open(sys.argv[1],'rb') as f:
    f.read(4); struct.unpack('<I',f.read(4))
    print(struct.unpack('<Q',f.read(8))[0], 'tensors')
EOF
```

## 每 token 读取量（量化选型的关键）

投机解码每步只读一遍稠密权重、摊到 3.4 个 token 上，所以**每 token 字节数就是速度上限**：

| 变体 | 文件 | 稠密/token | 专家/token | 合计 | 1M+草稿显存（预算 ≈25.4 GiB 权重） |
|---|---:|---:|---:|---:|---|
| IQ2_XXS（生产） | 20.5 GiB | 4.62 | 0.41 | **5.02** | ✅ 余 4.9 GiB |
| IQ2-denseQ6K | 20.1 | 4.23 | 0.41 | 4.64 | ✅ |
| IQ4-mini | 37.4 | 4.69 | 0.89 | 5.58 | ❌ 需 ~12 GiB 专家放 RAM |
| IQ4-q3exps | 33.5 | 6.99 | 0.68 | 7.67 | ❌ 需 ~8 GiB 放 RAM |
| IQ4_XS 完整 | 43.4 | 6.99 | 0.94 | 7.93 | ❌ 且超过内存安全线（掉盘） |

## model/probe/ —— 架构探针

三个分片一共 7.5 MB，只包含 GGUF 头部（张量名、形状、量化类型，不含权重）。
这是早期实验留下的快照，用来在不下载完整模型的情况下检查张量命名和量化类型。

仓库里的脚本只读取 GGUF 头部，可以直接拿它查看探针：

```bash
python3 -I scripts/gen-expert-types.py model/probe/<第一个分片>.gguf /tmp/probe-types.txt --all
head -20 /tmp/probe-types.txt
```

（探针文件本身属于本机制品，不随仓库发布。）

## model/UD-IQ3_XXS/ —— 120B 阶段遗留

Nemotron-3.5-Super-VL-120B-A12B 的 UD-IQ3_XXS 三分片（合计 57.3 GiB）。
结论已经写死在 `docs/09-实测数据.md`：**57 GiB 权重 + 46.9 GiB 内存 = 每个 token 都要从磁盘读数 GiB，
实测 0.04 tok/s**，这条路在本机不成立。保留它只是为了对照与复盘。

## 日志文件命名

| 前缀 | 来源 |
|---|---|
| `serve-YYYYMMDD-HHMMSS.log` | `llama-server` 每次启动的完整日志（加载耗时、每卡权重、graph 布局、每次请求的 timings） |
| `tp-YYYYMMDD-HHMMSS.log` | 张量并行实验日志（手动启动 `-sm tensor` 留下的原始日志） |
| `baseline-*.log` / `*.report.txt` | 120B 基线与自动测试的原始日志与摘要 |
| `bolt-*.log` / `*.report.txt` | Bolt 74B 各次启动/测试 |
| `lilicorr-*.log` / `lilicorr-*.report.txt` | 草稿模型（DFlash）专项实测：`smoke`、`200k`、`400k`、`1m`、`nmax-4`、`tp-spec-*`（TP+草稿移植）、`corr-*`（正确性对比） |
| `requant-*.log` | `llama-quantize` 重量化过程的日志 |
| `tpfit` / `ncmoe-test` / `spec` / `1m-ctx` 等 | 对应专项实验 |

排查问题时优先看**最新一份 `serve-*.log`**：里面有每卡权重分配、compute buffer 大小、
`MoE cache size`、每个请求的 `prompt eval` / `eval` 速率、以及投机解码的
`draft acceptance` / `mean acc len`。
