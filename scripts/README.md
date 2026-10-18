# scripts/ —— 可执行脚本

所有脚本都**从仓库根目录**运行，路径自动按脚本位置推导，不写死个人目录。
依赖：`bash`、`curl`、`python3`（标准库即可）。

| 脚本 | 作用 | 典型用法 |
|---|---|---|
| `download-model.sh` | 从 ModelScope 分块并行下载 + 断点续传 + SHA 校验 | `bash scripts/download-model.sh` |
| `verify-model.sh` | 校验文件大小与 SHA256 | `bash scripts/verify-model.sh --all` |
| `gen-expert-types.py` | 从 GGUF 头生成逐张量类型清单 | `python3 -I scripts/gen-expert-types.py in.gguf out.txt --tpfit` |
| `quantize-tpfit.sh` | 生成 19.58 GiB 的 `-tpfit.gguf` | `bash scripts/quantize-tpfit.sh` |
| `run-tp.sh` | 启动 3 卡张量并行 + 1M 上下文 | `bash scripts/run-tp.sh --daemon` |
| `start-ui.sh` | 启动 Trata 网页 / API 代理 | `bash scripts/start-ui.sh` |
| `serve-ui.py` | 后端代理本体（由 start-ui.sh 调用） | `python3 -I scripts/serve-ui.py` |

---

## 1. `download-model.sh`

```bash
bash scripts/download-model.sh                 # 默认 IQ2_XXS → model/bolt74b/
bash scripts/download-model.sh --check         # 只取远端大小
FILE=mmproj-BF16.gguf DEST=model bash scripts/download-model.sh
PARTS=8 bash scripts/download-model.sh
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `REPO` | `unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` | ModelScope 仓库 |
| `FILE` | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf` | 文件名 |
| `DEST` | `model/bolt74b` | 保存目录 |
| `PARTS` | `4` | 并发分块数 |
| `EXPECT_SHA` | 主模型已知 SHA | 换文件时自动置空；显式传空跳过 |
| `PROXY` | 空 | ModelScope 失败时 HF 兜底所用代理 |
| `FORCE` | `0` | `1` 跳过磁盘空间检查 |

要点：

- 远端大小用 `Content-Range` 取得，不写死；
- 每轮从 part 文件**当前字节数**续传，不用 `curl --retry`（会重复追加）；
- 拼接后校验 SHA，失败会明确提示重下；
- 需要约 2 倍文件大小的磁盘空间。

日志：`model/download.log`。

---

## 2. `verify-model.sh`

```bash
bash scripts/verify-model.sh          # 主模型
bash scripts/verify-model.sh --all    # 主模型 + mmproj
```

退出码：`0` 通过，`3` 缺失/大小或 SHA 不符。

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

## 5. `run-tp.sh`

```bash
bash scripts/run-tp.sh --daemon     # 后台 + 等就绪
bash scripts/run-tp.sh              # 前台
bash scripts/run-tp.sh --status
bash scripts/run-tp.sh --stop
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | 优先 `-tpfit.gguf` | 没有则用原版 IQ2 |
| `LLAMA_BIN` | `llama.cpp/build-nccl/bin/llama-server` | 二进制 |
| `NCCL_DIR` | 自动探测 `~/nccl-venv/...` | 含 `lib/libnccl.so.2` |
| `SM` | `tensor` | 改成 `layer` 即层切分 |
| `CTX` | `1048576` | 超过 262144 自动加 `--override-kv` |
| `UB` | `128` | 1M 下 256 会 OOM |
| `TS` | `1,1,1` | 三卡等分 |
| `CTK` / `CTV` | `q4_0` | 200K 建议 `q8_0` |
| `NP` | `1` | 并发槽位；`NP=4` 聚合约 83 tok/s |
| `HOST` / `PORT` | `127.0.0.1:8080` | |
| `THREADS` | `nproc - 1` | |
| `MM` | 自动探测 `model/mmproj-BF16.gguf` | 设为空可禁用 |
| `EXTRA` | 空 | 例如 `EXTRA="--metrics -lv 3"` |

层切分示例：

```bash
SM=layer CTX=204800 UB=512 CTK=q8_0 CTV=q8_0 bash scripts/run-tp.sh
```

4 路并发（总上下文 16K，每槽 4K）：

```bash
NP=4 CTX=16384 bash scripts/run-tp.sh
```

后台日志：`model/bolt-serve.log`；PID：`model/bolt-serve.pid`。

---

## 6. `start-ui.sh`

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

## 7. `serve-ui.py`

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

## 8. 脚本的约定

1. **不写死个人路径**：全部基于 `$(dirname "$0")/..`；
2. **可覆盖**：所有关键参数都有环境变量；
3. **失败要响**：校验失败返回非 0，不静默继续；
4. **不碰模型权重**：除了下载和量化，不改动 GGUF；
5. **不联网做隐式操作**：只有 `download-model.sh` 联网，其余全本地。
