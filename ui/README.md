# ui/ —— Trata 网页前端

`ui/strata/` 是模型服务的浏览器界面。外观与交互来自 **Strata**（MIT），
本项目改名为 **Trata** 并换了配色，后端由 `scripts/serve-ui.py` 提供。

```
ui/strata/
├── index.html          # 主页面
└── web/
    ├── app.js          # 聊天、SSE、监控、会话管理
    ├── app.css
    ├── components.css
    ├── tokens.css      # 设计令牌：浅色/深色两套
    ├── monitor.html    # /monitor 请求明细页
    ├── monitor.js
    ├── sprite.svg      # 图标
    ├── favicon.svg
    └── fonts/          # 内嵌 Outfit 字体（离线可用）
```

---

## 1. 页面

| 标签页 | 内容 |
|---|---|
| **Chat** | 流式对话、停止/重新生成、Markdown 与代码、思考过程折叠、会话列表 |
| **Monitor** | 三卡 GPU 利用率/显存/温度/功耗、CPU、内存、磁盘读、decode/prefill 曲线、请求表 |
| **About** | 模型名、上下文长度、KV 类型、硬件信息、API 地址 |
| **Images** | 仅当后端报告视觉能力时显示 |

`/monitor` 是独立页面：逐条请求的状态、prompt/output tokens、复用数、首 token 时间、输出内容。
调试 API 客户端时最有用。

---

## 2. 相对 Strata 原版的改动

1. **改名 Trata**：标题、页头品牌、Monitor 页；页脚保留 "UI based on Strata" 的出处说明。
2. **灰紫赛博主题**（`web/tokens.css`）：
   暗色底 `#0f0c17`、紫罗兰强调 `#a78bfa`、青色辅色 `#22d3ee`，卡片紫色辉光。
3. **进度条动效**（`web/components.css` + `web/app.js`）：
   生成中流动高光；"读提示"阶段用不确定态动画，因为 llama.cpp 不提供 prefill 进度。

前端逻辑（`app.js`）基本保持 Strata 原样，改动集中在品牌、样式和进度显示。

---

## 3. 后端端点

由 `scripts/serve-ui.py` 实现：

| 端点 | 用途 |
|---|---|
| `POST /v1/chat/completions` | 流式聊天（OpenAI 兼容），代理到 llama-server |
| `POST /v1/messages` | Anthropic Messages 协议，原样转发 |
| `GET /health` | `{model, images, image_generation, max_context}` |
| `GET /metrics` | **JSON**（不是 Prometheus）：live / hardware / engine / history / requests / totals |
| `GET/POST /settings` | 服务端保存的默认采样参数 |
| `GET /mcp` | 空列表，前端自动隐藏卡片 |
| `GET /config` | 404，前端隐藏运行配置卡片 |
| `GET /v1/status`、`GET /api/requests` | `/monitor` 用 |

### 数字从哪来

- **decode tok/s**：llama-server `/metrics` 的 `llamacpp:predicted_tokens_seconds`；
- **prefill tok/s**：`prompt_seconds_total` 的差值（只算新读的 token）；
- **首 token 时间**：流里第一个 delta 的到达时间；
- **GPU/CPU/RAM/磁盘**：每秒采样 `nvidia-smi` 与 `/proc`。

---

## 4. 启动

```bash
# 先启动 llama-server（8080）
bash scripts/run-tp.sh --daemon
# 再启动网页/API（8001）
bash scripts/start-ui.sh
```

浏览器打开 `http://<机器IP>:8001/`。

---

## 5. 修改样式

颜色/间距/字号都在 `web/tokens.css`：

```css
:root {
  --bg: #0f0c17;
  --accent: #a78bfa;
  --accent-2: #22d3ee;
  ...
}
```

改完刷新即可（没有构建步骤，纯静态文件）。

---

## 6. 许可

Strata 原版为 MIT。本目录保留其版权声明的要求见根目录
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)；
字体为 SIL Open Font License（`web/fonts/OFL.txt`）。
