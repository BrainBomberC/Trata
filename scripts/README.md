# scripts/ —— 可执行脚本

所有脚本都**从仓库根目录**运行，路径自动按脚本位置推导，不写死个人目录。
依赖：`bash`、`curl`、`python3`（标准库即可）。

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `download-model.sh` | 从 ModelScope 分块并行下载 + 断点续传 + SHA 校验 | `bash scripts/download-model.sh` |
| `verify-model.sh` | 校验三个文件的大小、SHA 与投影器张量数 | `bash scripts/verify-model.sh --all` |
| `gen-expert-types.py` | 从 GGUF 头生成逐张量类型清单 | `python3 -I scripts/gen-expert-types.py in.gguf out.txt --tpfit` |
| `quantize-tpfit.sh` | 生成 19.58 GiB 的 `-tpfit.gguf`（仅回退组需要） | `bash scripts/quantize-tpfit.sh` |
| **`run-draft.sh`** | **生产启动：层切分 + LiLiCorr 草稿 + 1M** | `bash scripts/run-draft.sh --daemon` |
| `run-tp.sh` | 回退启动：3 卡张量并行 + 1M，无草稿 | `bash scripts/run-tp.sh --daemon` |
| `start-ui.sh` | 启动 Trata 网页 / API 代理 | `bash scripts/start-ui.sh` |
| `serve-ui.py` | 后端代理本体（由 start-ui.sh 调用） | `python3 -I scripts/serve-ui.py` |

启动服务器的两个脚本**互斥**（都监听 8080），不要同时运行。

---

## 1. `download-model.sh`

```bash
bash scripts/download-model.sh                  # 主模型 IQ2_XXS → model/bolt74b/
PRESET=draft  bash scripts/download-model.sh    # DFlash 草稿 → model/draft/
PRESET=mmproj bash scripts/download-model.sh    # Bolt 视觉投影器 → model/

bash scripts/download-model.sh --check          # 只取远端大小
PARTS=8 bash scripts/download-model.sh
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `PRESET` | `main` | `main` / `draft` / `mmproj`，会填好 REPO/FILE/DEST/EXPECT_SHA |
| `REPO` | `unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` | ModelScope 仓库 |
| `FILE` | 随 PRESET | 文件名 |
| `DEST` | 随 PRESET | 主模型 `model/bolt74b`、草稿 `model/draft`、投影器 `model` |
| `PARTS` | `4` | 并发分块数 |
| `EXPECT_SHA` | 随 PRESET | 换文件时自动置空；显式传空跳过 |
| `PROXY` | 空 | ModelScope 失败时 HF 兜底所用代理 |
| `FORCE` | `0` | `1` 跳过磁盘空间检查 |

要点：

- 远端大小用 `Content-Range` 取得，不写死；
- 每轮从 part 文件**当前字节数**续传，不用 `curl --retry`（会重复追加）；
- 拼接后校验 SHA，失败会提示"可能是 64 字节元数据变体"；
- 需要约 2 倍文件大小的磁盘空间；
- `PRESET=mmproj` 的 SHA 是 Bolt 仓库那份；从 120B 仓库下同名文件会直接失败。

日志：`model/download.log`。

---

## 2. `verify-model.sh`

```bash
bash scripts/verify-model.sh              # 主模型
bash scripts/verify-model.sh --draft      # 草稿
bash scripts/verify-model.sh --mmproj     # 投影器（额外检查 519 张量）
bash scripts/verify-model.sh --all        # 三个都查
```

退出码：`0` 通过，`3` 缺失/大小或 SHA 不符。
大小只差 ≤64 字节时给警告而不判失败（已知的元数据变体）。

---

## 3. `gen-expert-types.py`

从 GGUF 头部读取每个张量的实际类型，生成 `llama-quantize --tensor-type-file` 需要的清单。

```bash
# 默认：只列专家张量，保留原类型
python3 -I scripts/gen-expert-types.py model.gguf /tmp/types.txt

# 全部张量按原类型列出
python3 -I scripts/gen-expert-types.py model.gguf /tmp/types.txt --all

# 本项目 tpfit 配方：240 条
python3 -I scripts/gen-expert-types.py model.gguf /tmp/tpfit.txt --tpfit
```

`--tpfit` 输出：

```
blk.0.ssm_in.weight=q3_K
blk.0.ssm_out.weight=q3_K
blk.1.ffn_down_exps.weight=iq2_xxs
...
类型分布: {'q3_K': 80, 'iq2_xxs': 160}
```

如果规则数不是 240，脚本返回 3 并警告——通常意味着输入不是官方 IQ2_XXS。

---

## 4. `quantize-tpfit.sh`

**只有要走 TP 回退、且 1M + 原版 IQ2 装不下时才需要。**
生产配置（层切分 + 草稿）直接用原版 IQ2，不需要 tpfit。

```bash
bash scripts/quantize-tpfit.sh              # 正常执行（约 12 分钟）
bash scripts/quantize-tpfit.sh --dry-run    # 只打印命令
FORCE=1 bash scripts/quantize-tpfit.sh      # 覆盖已有输出
```

| 变量 | 默认 |
|---|---|
| `IN` | `model/bolt74b/NVIDIA-...-IQ2_XXS.gguf` |
| `OUT` | `model/bolt74b/NVIDIA-...-IQ2_XXS-tpfit.gguf` |
| `QUANTIZE` | `llama.cpp/build-nccl/bin/llama-quantize` |
| `THREADS` | `nproc - 1` |
| `TYPES` | `model/bolt74b/.tpfit-types.txt` |
| `LOG` | `benchmarks/logs/requant-tpfit-<时间>.log` |

脚本会：

1. 检查输入大小是否为官方 22,018,326,464 字节（旧版多 64 字节也接受）；
2. 生成并核对 240 条规则（160 iq2_xxs + 80 q3_K）；
3. 检查磁盘；
4. 执行量化并保留日志；
5. 对比输出大小（参考 21,025,816,576 字节）。

**不要**手工改配方里的类型：专家必须保持 `iq2_xxs` 才能免 imatrix；
SSM 降到 `q3_K` 是 tpfit 省出 1 GiB 的来源。

---

## 5. `run-draft.sh`（生产）

```bash
bash scripts/run-draft.sh --daemon     # 后台 + 等就绪（最多 40 分钟）
bash scripts/run-draft.sh              # 前台
bash scripts/run-draft.sh --status
bash scripts/run-draft.sh --stop
```

默认配置：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | `model/bolt74b/...-IQ2_XXS.gguf` | 原版，不需要 tpfit |
| `DRAFT` | `model/draft/dflash-...-NVFP4.gguf` | 找不到时再试 `...-LiLiCorr.gguf` |
| `MM` | `model/mmproj-BF16.gguf` | 设为空字符串可禁用视觉 |
| `LLAMA_BIN` | `llama.cpp/build-nccl/bin/llama-server` | 需支持 LiLiCorr |
| `SM` | `layer` | 改成 `tensor` 需要 NCCL，且实测更慢 |
| `TS` | `1,1,1` | 三卡等分 |
| `CTX` | `1048576` | 超过 262144 自动加 `--override-kv` |
| `UB` | `64` | 1M 下 256 会 OOM |
| `NMAX` | `4` | 1M 最优；200K/400K 用 3 |
| `NMIN` | `0` | 允许草稿放弃 |
| `SPEC` | `draft-dflash` | 设为 `none` 关闭草稿做对照 |
| `NGLD` | `all` | 草稿层数；`0` = 放 CPU（实测 21.9 tok/s） |
| `CTK` / `CTV` | `q4_0` | 200K/400K 建议 `q8_0` |
| `NP` | `1` | 并发槽位 |
| `HOST` / `PORT` | `127.0.0.1:8080` | |
| `THREADS` | `nproc - 1` | |
| `EXTRA` | 空 | 例如 `EXTRA="--metrics -lv 3"` |

示例：

```bash
# 200K + 高质量 KV + 更短的草稿
CTX=204800 CTK=q8_0 CTV=q8_0 NMAX=3 bash scripts/run-draft.sh --daemon

# 关掉草稿做 A/B
SPEC=none bash scripts/run-draft.sh --daemon

# TP + 草稿（对照实验，1M 会 OOM）
SM=tensor CTX=32768 bash scripts/run-draft.sh
```

后台日志：`model/bolt-serve.log`；PID：`model/bolt-serve.pid`。

---

## 6. `run-tp.sh`（回退）

```bash
bash scripts/run-tp.sh --daemon     # 后台 + 等就绪
bash scripts/run-tp.sh              # 前台
bash scripts/run-tp.sh --status
bash scripts/run-tp.sh --stop
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | 原版 `IQ2_XXS.gguf` | TP + 1M 用 `-ub 64` 刚好装下，不需要 tpfit |
| `LLAMA_BIN` | `llama.cpp/build-nccl/bin/llama-server` | 二进制 |
| `NCCL_DIR` | 自动探测 `~/nccl-venv/...` | 含 `lib/libnccl.so.2` |
| `SM` | `tensor` | 张量并行 |
| `CTX` | `1048576` | 超过 262144 自动加 `--override-kv` |
| `UB` | `64` | 1M + TP 下 128/256 会 OOM |
| `TS` | `1.02,1.02,0.96` | 按显存余量微调 |
| `CTK` / `CTV` | `q4_0` | 200K 建议 `q8_0` |
| `NP` | `1` | 并发槽位；`NP=4` 聚合约 83 tok/s |
| `HOST` / `PORT` | `127.0.0.1:8080` | |
| `THREADS` | `nproc - 1` | |
| `MM` | `model/mmproj-BF16.gguf` | 设为空可禁用 |
| `EXTRA` | 空 | 例如 `EXTRA="--metrics -lv 3"` |

示例：

```bash
# 回退：TP×3 + 1M
bash scripts/run-tp.sh --daemon

# 原生 262K（更快，KV 更小）
CTX=262144 CTK=q8_0 CTV=q8_0 UB=256 bash scripts/run-tp.sh

# 4 路并发（总上下文 16K，每槽 4K）
NP=4 CTX=16384 bash scripts/run-tp.sh
```

后台日志：`model/bolt-serve.log`；PID：`model/bolt-serve.pid`。
**注意**：和 `run-draft.sh` 用同一个 PID/日志文件，切换前先 `--stop`。

---

## 7. `start-ui.sh`

```bash
bash scripts/start-ui.sh               # 后台，默认 0.0.0.0:8001
bash scripts/start-ui.sh --foreground
bash scripts/start-ui.sh --status
bash scripts/start-ui.sh --stop
```

| 变量 | 默认 |
|---|---|
| `HOST` | `0.0.0.0` |
| `PORT` | `8001` |
| `BACKEND` | `http://127.0.0.1:8080` |
| `LOG` | `model/trata-ui.log` |
| `PIDFILE` | `model/trata-ui.pid` |

只要本机访问就 `HOST=127.0.0.1 bash scripts/start-ui.sh`。
**默认没有鉴权，不要暴露到公网。**

---

## 8. `serve-ui.py`

由 `start-ui.sh` 调用，一般不需要手工运行：

```bash
python3 -I scripts/serve-ui.py --host 0.0.0.0 --port 8001 \
  --backend http://127.0.0.1:8080
```

它提供：

- 托管 `ui/strata/` 静态文件（Trata 界面）；
- Strata 形状的 `/health`、`/metrics`（JSON）、`/settings`、`/v1/status`、`/api/requests`；
- 代理 `/v1/chat/completions`、`/v1/completions`、`/v1/messages`、`/v1/models` 等；
- 实时统计 decode / prefill / 首 token 时间，供 Monitor 页显示。

只用 Python 标准库，无第三方依赖。

---

## 9. 脚本的约定

1. **不写死个人路径**：全部基于 `$(dirname "$0")/..`；
2. **可覆盖**：所有关键参数都有环境变量；
3. **失败要响**：校验失败返回非 0，不静默继续；
4. **不碰模型权重**：除了下载和量化，不改动 GGUF；
5. **不联网做隐式操作**：只有 `download-model.sh` 联网，其余全本地；
6. **两个启动脚本互斥**：都占用 8080，切换前先 `--stop`。
