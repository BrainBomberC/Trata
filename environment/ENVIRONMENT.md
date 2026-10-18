# 运行与编译环境

推理引擎是 C++ 二进制，**运行时不需要 Python 环境**。
Python 只用于：下载脚本之外的辅助工具（并发生成类型清单、Trata 代理）。

---

## 一、运行时

| 组件 | 要求 | 本项目实测 |
|---|---|---|
| 系统 | Linux x86_64 | Ubuntu 24.04.4 |
| GPU | NVIDIA，sm_86 或更新 | 3 × RTX 3060 12 GB |
| 驱动 | 支持 CUDA 12.x | 595.58.03 |
| NCCL | 2.x，含 `libnccl.so.2` | 来自 `nvidia-nccl-cu12` wheel |
| llama-server（生产） | 上游 `5aeac37` + 补丁 0005，支持 LiLiCorr | `llama.cpp/build-nccl/bin/` |
| llama-server（回退） | 上游 `69f201a` + 补丁 0001–0004，支持 TP | 同一路径（二选一） |

启动前：

```bash
export NCCL_DIR="$HOME/nccl-venv/lib/python3.12/site-packages/nvidia/nccl"
export LD_LIBRARY_PATH="$NCCL_DIR/lib:$PWD/llama.cpp/build-nccl/bin:$LD_LIBRARY_PATH"
# 只有 -sm tensor 需要：
export LLAMA_SM_TENSOR_ALLOW_ARCH=1
```

`scripts/run-draft.sh`（生产）和 `scripts/run-tp.sh`（回退）都会自动探测 NCCL、
设置 `LD_LIBRARY_PATH`，并导出 `LLAMA_SM_TENSOR_ALLOW_ARCH=1`
（只有 `-sm tensor` 真正用到，层切分设了也无害）。

### 1.1 为什么 NCCL 这么关键

**回退配置**（张量并行）每层都要 all-reduce。没有 NCCL 时退化成主机内存搬运：

| 配置 | 200K prefill |
|---|---:|
| 有 NCCL | 781 tok/s |
| 无 NCCL | 185 tok/s |

**生产配置**（层切分）不做集合通信，但二进制仍链接了 NCCL，
所以启动时也要能找到 `libnccl.so.2`。

编译时 `-DGGML_CUDA_NCCL=ON`，运行时 `ldd ... | grep nccl` 必须有输出。

### 1.2 草稿模型

生产配置额外需要 `model/draft/dflash-…-NVFP4.gguf`（998 MB）。
它只放 GPU 才有效（`-ngld all`）；放 CPU 实测 21.9 tok/s，低于不用草稿。
草稿不需要单独的 CUDA 环境，和主模型共用同一个 `llama-server`。

---

## 二、编译环境

只有需要重新编译时才用得到。

| 组件 | 要求 | 实测 |
|---|---|---|
| CUDA toolkit | 12.x | 12.0.140 |
| g++ | 12 或 13 | 13.3.0（CUDA 12.0 需 g++-12 作 host compiler） |
| cmake | ≥ 3.20 | 4.4.3 |
| ninja | 可选 | 1.13.2 |
| 磁盘 | 约 10 GB | — |

```bash
sudo apt install build-essential cmake ninja-build g++-12
```

三个实测坑：

1. **CUDA 12.0 + GCC 13**：`crt/host_config.h` 会拒绝，必须
   `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`。
2. **没有 libcurl 开发头**：不要开 `LLAMA_CURL`；也因此不能用 `-hf` 自动下载，
   改用 `scripts/download-model.sh`。
3. **NCCL 必须显式指定**：
   `-DNCCL_INCLUDE_DIR=.../include -DNCCL_LIBRARY=.../lib/libnccl.so.2`。

完整命令见 [docs/03-编译引擎](../docs/03-编译引擎.md)。

---

## 三、网络（实测，决定下载策略）

本机测试时间：2026-10。

| 源 | 速度 | 说明 |
|---|---:|---|
| ModelScope 直连（单流） | 6.7–7.4 MB/s | ✅ 首选；必须 `--noproxy '*'` |
| ModelScope 4 路并行 | ≈11 MB/s | 22 GB 约 33 分钟 |
| HF 走代理 | 0.10–1.19 MB/s | 兜底 |
| hf-mirror.com | ≈0.34 MB/s | 比走代理还慢 |

如果机器设了全局 `http_proxy`，访问 ModelScope 可能被劫持或变慢。
`download-model.sh` 默认对 ModelScope 加 `--noproxy '*'`。

---

## 四、磁盘（实测）

| 用途 | 实测速度 | 含义 |
|---|---:|---|
| 模型盘（USB2 转接） | 43.8 MB/s | 22 GB 加载约 8 分钟；运行中掉盘会掉到个位数 tok/s |
| 系统盘（NVMe） | 数百 MB/s | 建议把模型放这里 |
| SATA SSD | 约 46 MB/s（本机实测） | 比预期慢，注意 QLC 缓存耗尽 |

**结论**：

- 模型必须放在**内存能被页缓存容纳 + 磁盘不会成为每 token 路径**的地方；
- 下载/量化需要 2 倍文件大小的空间。

---

## 五、机器红线（建议）

这些不是技术限制，而是避免把能跑的环境弄坏：

- 不为了本项目升级 NVIDIA 驱动、内核、firmware；
- 不在推理机上跑 `apt upgrade`；
- 只用用户级二进制和 venv；
- 停别人的服务前先确认；
- `MemorySwapMax=0`：宁可 OOM 报错，也不要静默换页到 0.04 tok/s。

---

## 六、最小检查清单

```bash
nvidia-smi
nvcc --version            # 仅编译需要
ldconfig -p | grep nccl   # 或 find ~/nccl-venv -name libnccl.so.2
ldd llama.cpp/build-nccl/bin/llama-server | grep nccl
free -g
df -h .
```
