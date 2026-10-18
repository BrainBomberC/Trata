# 08 · 网页与 API

模型服务起来后，有两种用法：

1. **网页**：浏览器打开 `http://<机器IP>:8001`，就是 Trata 界面；
2. **API**：任何 OpenAI / Anthropic 兼容的客户端，指向 `http://<机器IP>:8001/v1`。

两者背后是同一个 `llama-server`。

---

## 1. 为什么中间要有一个 `serve-ui.py`

`llama-server` 本身已经有 OpenAI 兼容 API 和内置网页。
但本项目的前端 Trata 沿用了 Strata 的后端接口形状，需要：

- `/health`、`/metrics`（**JSON** 而非 Prometheus 文本）、`/settings`、`/v1/status`、`/api/requests`；
- 实时统计 decode tok/s、prefill tok/s、首 token 时间；
- 把 Anthropic 的 `/v1/messages` 也转发到同一个后端。

`scripts/serve-ui.py` 是一个**零依赖**（只用 Python 标准库）的小代理：
它实现上述端点，把其余 `/v1/*` 原样转发给 `127.0.0.1:8080`。

```
浏览器 / 客户端
      │  :8001
      ▼
serve-ui.py  ── 静态文件 ui/strata/（Trata 前端）
      │        ── Strata 形状的 /health /metrics /settings
      │        ── 代理 /v1/chat/completions、/v1/messages、/v1/models ...
      ▼  :8080
llama-server（层切分或 TP + 1M，可带 DFlash 草稿）
```

---

## 2. 启动网页

先确保 `llama-server` 已经在 8080 监听（见 07）。

```bash
# 前台
python3 -I scripts/serve-ui.py

# 指定监听地址和端口
python3 -I scripts/serve-ui.py --host 0.0.0.0 --port 8001 \
  --backend http://127.0.0.1:8080
```

成功输出：

```
[serve-ui] Strata UI on http://0.0.0.0:8001  -> backend http://127.0.0.1:8080
```

浏览器打开：

```
http://127.0.0.1:8001          # 本机
http://<局域网IP>:8001          # 同一局域网的其他电脑
```

### 2.1 用脚本

```bash
bash scripts/start-ui.sh                 # 后台启动，日志 model/trata-ui.log
bash scripts/start-ui.sh --foreground    # 前台
bash scripts/start-ui.sh --status
bash scripts/start-ui.sh --stop
```

### 2.2 systemd 常驻

```bash
cp systemd/bolt-strata-ui.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bolt-strata-ui.service
```

---

## 3. Trata 界面

| 标签页 | 内容 |
|---|---|
| **Chat** | 流式对话、停止生成、重新生成、Markdown/代码高亮、思考过程折叠、会话列表 |
| **Monitor** | 每秒刷新的三卡利用率/显存/温度/功耗、CPU、内存、磁盘读、decode/prefill 曲线、最近请求表 |
| **About** | 模型与引擎信息、上下文长度 |
| **Images** | 后端报告 `images: true` 时才显示；本项目需要加载 `mmproj` |

`/monitor` 是独立页面，只显示请求明细（id、状态、prompt/output tokens、首 token 延迟、输出内容）。
调试客户端时很有用：能直接看到请求体和模型返回。

### 3.1 界面上数字的来源

| 数字 | 来源 | 注意 |
|---|---|---|
| decode tok/s | llama-server `/metrics` 的 `predicted_tokens_seconds` | 引擎自算，最准 |
| prefill tok/s | `prompt_seconds_total` 的差值 | 只算真正新读的 token；缓存命中另计 |
| 首 token 时间 | 流里第一个 delta 的到达时间 | 包含排队 + prefill |
| GPU 显存/利用率 | `nvidia-smi` 每秒采样 | 三卡取平均；单卡看 Monitor 下方列表 |
| 请求 token 数 | 流末 `timings` | prompt_n / predicted_n |

> **"读提示"阶段没有百分比进度**：llama.cpp 在整段提示读完前不更新计数。
> 前端因此用不确定态动画，不伪造百分比——否则会显示一个一路衰减的假速度。

---

## 4. OpenAI 兼容 API

Base URL：`http://<host>:8001/v1`
API Key：随便填（服务本身不校验）。

### 4.1 curl

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bolt74b",
    "messages": [{"role": "user", "content": "用一句话解释张量并行"}],
    "max_tokens": 128,
    "stream": false
  }' | python3 -m json.tool
```

流式：

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bolt74b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

### 4.2 Python（openai SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")
r = client.chat.completions.create(
    model="bolt74b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(r.choices[0].message.content)
```

### 4.3 常用端点

| 端点 | 说明 |
|---|---|
| `GET /v1/models` | 模型列表（由 llama-server 提供） |
| `POST /v1/chat/completions` | 对话，支持 `stream` |
| `POST /v1/completions` | 补全 |
| `POST /tokenize`、`/detokenize` | 分词/反分词 |
| `GET /props` | 引擎属性（上下文长度、模板等） |
| `GET /slots` | 槽位状态 |

### 4.4 思考模式

模型是推理模型。要关闭思考：

```json
{
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Trata 界面里的 `reasoning_effort` 会被 `serve-ui.py` 翻译成这个字段。

---

## 5. Anthropic Messages API

`serve-ui.py` 实现了 `POST /v1/messages`，并把 `anthropic-version` / `anthropic-beta` 头转发给后端。
因此 Claude Code 一类客户端可以直接接：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8001
export ANTHROPIC_AUTH_TOKEN=none
# 某些客户端还需要：
export ANTHROPIC_MODEL=bolt74b
```

直接测试：

```bash
curl -s http://127.0.0.1:8001/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "bolt74b",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "你好"}]
  }' | python3 -m json.tool
```

> 如果返回 404 / `not supported`，说明当前 llama.cpp 构建没有开 Anthropic 兼容层。
> 本项目钉的两个基线（生产 `5aeac37`、回退 `69f201a`）都已包含 `/v1/messages`；
> 换 commit 后需要重新确认。

---

## 6. 原生 llama-server 接口

绕过 `serve-ui.py` 直连 8080，可以拿到：

- `GET http://127.0.0.1:8080/metrics`：**Prometheus 文本**（`llamacpp:predicted_tokens_seconds` 等）；
- `GET http://127.0.0.1:8080/health`：引擎健康；
- `POST /v1/chat/completions`：同样的 OpenAI 协议，但没有 Trata 的统计。

`serve-ui.py` 的 `/metrics` 是**另一个东西**：返回 JSON，供网页画图。
两者路径相同、内容不同，注意区分端口。

常用指标：

| 指标 | 含义 |
|---|---|
| `llamacpp:predicted_tokens_seconds` | 当前 decode 速度 |
| `llamacpp:prompt_tokens_seconds` | 当前 prefill 速度 |
| `llamacpp:prompt_tokens_total` | 累计读入 token |
| `llamacpp:tokens_predicted_total` | 累计生成 token |
| `llamacpp:kv_cache_usage_ratio` | KV 使用率 |

---

## 7. 图片输入（可选）

需要启动时加了 `-mm model/mmproj-BF16.gguf`。OpenAI 格式：

```json
{
  "model": "bolt74b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "描述这张图片"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}
    ]
  }]
}
```

注意：

- `--no-mmproj-offload` 把视觉编码器放 CPU，首张图会慢几秒；
- 视觉不是本项目性能重点，长上下文 + 图片更容易 OOM；
- 不用图片就别加载 `mmproj`，省 1.6 GiB 内存/显存。

---

## 8. 安全：默认不要暴露到公网

`serve-ui.py` 默认监听 `0.0.0.0:8001`，**没有鉴权**。这意味着同一局域网的任何人都能：

- 使用你的模型（占满显存/算力）；
- 读取 `/monitor` 里的历史请求和输出；
- 通过 `/settings` 改默认参数。

建议：

```bash
# 只给本机用
python3 -I scripts/serve-ui.py --host 127.0.0.1

# 要给局域网用，至少加防火墙限制来源
sudo ufw allow from 192.168.1.0/24 to any port 8001
```

**不要**把 8001 直接映射到公网。需要远程访问时用 SSH 隧道：

```bash
ssh -L 8001:127.0.0.1:8001 user@server
```

---

## 9. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面打开是空白 | `ui/strata/` 路径不对 | 从仓库根目录启动，别 `cd scripts` |
| 右上角一直 "loading" | 8080 后端没起来 | 先看 `curl 127.0.0.1:8080/health` |
| 有回复但 tok/s 为 `–` | 后端没加 `--metrics` | 启动参数补上 |
| `/monitor` 没有请求 | 请求走了 8080 直连 | 改为走 8001 |
| 局域网访问不了 | 监听在 127.0.0.1 / 防火墙 | `--host 0.0.0.0` + 放行端口 |
| 图片上传报错 | 没加载 mmproj | 加 `-mm`；回退引擎需 0004 补丁，生产引擎已内置 |
| `Address already in use` | 端口被占 | `ss -ltnp \| grep 8001`，换端口 |

---

下一篇：看完整实测数据 → [09-实测数据](09-实测数据.md)。
