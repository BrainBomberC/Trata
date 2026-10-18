#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""serve-ui.py —— 把 llama-server 包装成 Strata 的 API 形状，并原样托管 Strata 网页前端。

为什么这样写：
  Strata 的网页（ui/strata/）是一个完整应用，依赖它自己后端的一组端点：
    GET  /health            {model, images, image_generation, max_context}
    GET  /metrics           JSON（不是 Prometheus 文本）：live / hardware / engine / history / requests
    GET/POST /settings      {shared, defaults}
    GET  /config            没有运行配置时返回 404 → 前端自动隐藏那张卡片
    GET  /mcp               {servers: [], tools: 0}
    GET  /v1/status         {model, loaded, auto_load, activity:{in_flight}}
    GET  /api/requests      {requests: [...], loaded, auto_load}
    POST /v1/chat/completions   流式转发到 llama-server，同时统计实时 tps/prefill
  本文件把这些端点一一实现，因此网页的**外观与交互**与 Strata 完全一致，只是数据来自这台机器上的 llama.cpp。

对外监听 0.0.0.0:8001（其他电脑可直接连），后端是 127.0.0.1:8080 的 llama-server。

用法:
    python3 -I scripts/serve-ui.py [--host 0.0.0.0] [--port 8001] [--backend http://127.0.0.1:8080]
"""

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
UI_DIR = os.path.join(ROOT, "ui", "strata")

BACKEND = "http://127.0.0.1:8080"
MODEL_NAME = "bolt74b"

# ----------------------------------------------------------------------------- 共享状态
LOCK = threading.Lock()


def _hist():
    return deque(maxlen=90)


STATE = {
    "live": {
        "state": "idle", "phase": "", "tok_s": None, "prefill_tok_s_mean": None,
        "prompt_total": 0, "prompt_read": 0, "prompt_tokens": 0,
        "generated": 0, "max_tokens": 0, "queued": 0,
    },
    "hardware": {},
    "hardware_static": {},
    "engine": {},
    "history": {k: _hist() for k in (
        "tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used", "gpu_temp",
        "gpu_power", "gpu_pcie_rx_mb", "cpu", "disk_read_mb")},
    "requests": deque(maxlen=200),
    "totals": {"since": int(time.time()), "requests": 0, "prompt_tokens": 0, "reused": 0,
               "prompt_ms": 0.0, "output_tokens": 0, "decode_ms": 0.0},
    "settings": {"shared": False, "defaults": None},
}
REQ_DETAIL = {}       # id -> 详情（输入/输出/错误）
REQ_SEQ = [0]

SETTINGS_FILE = os.path.join(ROOT, "model", "ui-settings.json")


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            STATE["settings"]["shared"] = bool(d.get("shared"))
            STATE["settings"]["defaults"] = d.get("defaults")
    except Exception:
        pass


def save_settings():
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(STATE["settings"], f)
    except Exception:
        pass


# ----------------------------------------------------------------------------- 后端探测
def backend_get(path, timeout=3.0):
    try:
        with urllib.request.urlopen(BACKEND + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def read_prom():
    """读 llama-server 的 Prometheus 指标——引擎自己算好的实时 prefill / decode 速率最准。"""
    try:
        with urllib.request.urlopen(BACKEND + "/metrics", timeout=2.0) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    vals = {}
    for line in raw.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        try:
            k, v = line.rsplit(None, 1)
            vals[k[9:]] = float(v)
        except Exception:
            pass
    return vals


def backend_reachable():
    try:
        with urllib.request.urlopen(BACKEND + "/health", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def detect_engine():
    """从 /props 取模型名、上下文长度等，填充 engine 段。"""
    props = backend_get("/props", 5.0) or {}
    model_path = props.get("model_path") or ""
    alias = props.get("model_alias") or ""
    if not model_path and not alias:
        return STATE["engine"]          # 后端还没起来，保留旧的（poller 会重试）
    name = os.path.basename(model_path or alias)
    name = re.sub(r"\.gguf$", "", name)
    n_ctx = props.get("n_ctx") or 0
    dgs = props.get("default_generation_settings") or {}
    n_ctx = n_ctx or dgs.get("n_ctx") or 0
    engine = {
        "model": name or MODEL_NAME,
        "version": props.get("build_info") or "",
        "max_context": int(n_ctx),
        "kv": "q8_0",
        "kv_resident": 0,
        "expert_slots": 0,
        "expert_cache_mib": 0,
        "cvec": 0,
        "spec": 0,
        # 视觉输入（mmproj 已加载时后端会报告 modalities.vision）
        "images": bool((props.get("modalities") or {}).get("vision", False)),
        "backend": "llama.cpp",
        # 客户端没给 max_tokens 时，用引擎的默认上限当作进度条分母（-1 = 不限 → 0）
        "n_predict_default": (lambda n: n if n > 0 else 0)(
            int((dgs.get("params") or {}).get("n_predict") or -1)),
    }
    with LOCK:
        STATE["engine"] = engine
    return engine


# ----------------------------------------------------------------------------- 硬件采样
GPU_FIELDS = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,"
              "power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current")

_hw_prev = {"t": None, "disk": None}
_static_done = False


def read_cpu_pct():
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()[1:]
        vals = [int(x) for x in parts]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
    except Exception:
        return None
    key = "cpu"
    prev = _hw_prev.get(key)
    _hw_prev[key] = (idle, total)
    if not prev:
        return None
    di, dt = idle - prev[0], total - prev[1]
    if dt <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))


def read_mem():
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                d[k] = int(v.strip().split()[0]) * 1024
        total = d.get("MemTotal", 0)
        avail = d.get("MemAvailable", d.get("MemFree", 0))
        return total - avail, total
    except Exception:
        return None, None


def read_disk_mb():
    """整机磁盘读速率（MB/s），进程级更细的统计需要 psutil；这里用 /proc/diskstats 全盘求和。"""
    try:
        sectors = 0
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) < 6:
                    continue
                name = p[2]
                if re.match(r"^(sd[a-z]+|nvme\d+n\d+|vd[a-z]+)$", name):
                    sectors += int(p[5])          # 读扇区数
        now = time.time()
        prev = _hw_prev.get("disk")
        _hw_prev["disk"] = (now, sectors)
        if not prev:
            return None
        dt = now - prev[0]
        if dt <= 0:
            return None
        return max(0.0, (sectors - prev[1]) * 512 / dt / 1e6)
    except Exception:
        return None


def sample_hardware():
    hw = {}
    gpus = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4).stdout
        for line in out.strip().splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) < 11:
                continue
            def num(x):
                try:
                    return float(x)
                except Exception:
                    return None
            g = {"index": int(num(v[0]) or 0), "name": v[1], "util": num(v[2]),
                 "mem_used": (num(v[3]) or 0) * 1048576, "mem_total": (num(v[4]) or 0) * 1048576,
                 "temp": num(v[5]), "power": num(v[6]), "power_limit": num(v[7]),
                 "pcie_gen": num(v[8]), "pcie_gen_max": num(v[9]), "pcie_width": num(v[10])}
            gpus.append(g)
    except Exception:
        pass
    if gpus:
        def agg(key, how="sum"):
            vals = [g[key] for g in gpus if g.get(key) is not None]
            if not vals:
                return None
            if how == "sum":
                return sum(vals)
            if how == "max":
                return max(vals)
            return sum(vals) / len(vals)
        hw["gpu_util"] = agg("util", "mean")   # Strata 显示的是三卡平均；看单卡看 gpus 列表
        hw["gpu_mem_used"] = agg("mem_used")
        hw["gpu_mem_total"] = agg("mem_total")
        hw["gpu_temp"] = agg("temp", "max")
        hw["gpu_power"] = agg("power")
        hw["gpu_power_limit"] = agg("power_limit")
        hw["gpu_pcie_gen"] = gpus[0].get("pcie_gen")
        hw["gpu_pcie_gen_max"] = gpus[0].get("pcie_gen_max")
        hw["gpu_pcie_width"] = gpus[0].get("pcie_width")
        hw["gpus"] = [{"index": g["index"], "util": g["util"], "mem_used": g["mem_used"],
                       "mem_total": g["mem_total"], "temp": g["temp"], "power": g["power"]} for g in gpus]
    # PCIe 速率：用 nvidia-smi 的 pcie 计数不可得，留空（前端会显示 "–"）
    hw["cpu"] = read_cpu_pct()
    ram_used, ram_total = read_mem()
    hw["ram_used"], hw["ram_total"] = ram_used, ram_total
    hw["disk_read_mb"] = read_disk_mb()
    with LOCK:
        STATE["hardware"] = hw
        h = STATE["history"]
        for k in h:
            if k in hw:
                h[k].append(hw[k])
    return hw


def hardware_static():
    global _static_done
    if _static_done:
        return STATE["hardware_static"]
    st = {"threads": os.cpu_count() or 0, "cores": os.cpu_count() or 0, "psutil": True}
    try:
        st["cpu_name"] = re.sub(r"\s+", " ", open("/proc/cpuinfo").readline().split(":", 1)[1]).strip()
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
        if out:
            st["gpu_name"] = out[0] + (f" × {len(out)}" if len(out) > 1 else "")
    except Exception:
        st["gpu_note"] = "nvidia-smi 不可用"
    with LOCK:
        STATE["hardware_static"] = st
    _static_done = True
    return st


# ----------------------------------------------------------------------------- 请求记账
def new_request(path, stream, body_text):
    with LOCK:
        REQ_SEQ[0] += 1
        rid = f"r{REQ_SEQ[0]:05d}"
        rec = {"id": rid, "path": path, "model": STATE["engine"].get("model", MODEL_NAME),
               "state": "active", "started_at": time.time(), "wallclock_s": None,
               "stream": bool(stream), "http_status": None,
               "usage": {}, "timings": {}}
        REQ_DETAIL[rid] = {"input": body_text[:262144], "output": "", "reasoning": "",
                           "error": None, "first_token_s": None}
    return rid


def finish_request(rid, rec_fields, detail_fields=None):
    with LOCK:
        for r in STATE["requests"]:
            if r.get("id") == rid:
                r.update(rec_fields)
                break
    if detail_fields:
        with LOCK:
            REQ_DETAIL.setdefault(rid, {}).update(detail_fields)


def account_request(rec):
    """把一条完成的请求并入 totals 和 requests 列表。"""
    t = STATE["totals"]
    t["requests"] += 1
    t["prompt_tokens"] += rec.get("prompt_tokens") or 0
    t["reused"] += rec.get("reused") or 0
    t["prompt_ms"] += rec.get("prompt_ms") or 0
    t["output_tokens"] += rec.get("output_tokens") or 0
    t["decode_ms"] += rec.get("decode_ms") or 0


# ----------------------------------------------------------------------------- HTTP 处理
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BoltStrataUI/1.0"

    def log_message(self, fmt, *args):
        pass

    # ---- 小工具
    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.cors()
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_json({"error": {"message": "not found"}}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.cors()
        self.end_headers()
        self.wfile.write(data)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---- 静态
    STATIC = {
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }

    def serve_static(self, rel):
        rel = rel.lstrip("/")
        full = os.path.normpath(os.path.join(UI_DIR, rel))
        if not full.startswith(os.path.normpath(UI_DIR)) or not os.path.isfile(full):
            self.send_json({"error": {"message": "not found"}}, 404)
            return
        ext = os.path.splitext(full)[1].lower()
        self.send_file(full, self.STATIC.get(ext, "application/octet-stream"))

    # ---- 路由
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        q = urllib.parse.parse_qs(u.query)

        if p in ("/", "/index.html"):
            return self.serve_static("index.html")
        if p in ("/monitor", "/monitor.html"):
            return self.serve_static("web/monitor.html")
        if p.startswith("/web/"):
            return self.serve_static(p[1:])

        if p in ("/health", "/healthz"):
            eng = STATE["engine"]
            return self.send_json({
                "status": "ok" if backend_reachable() else "loading",
                "model": eng.get("model", MODEL_NAME),
                "images": bool(eng.get("images", False)), "image_generation": False,
                "max_context": eng.get("max_context", 0),
            })
        if p == "/metrics":
            return self.send_json(build_metrics(all_requests=(q.get("requests", [""])[0] == "all")))
        if p == "/settings":
            return self.send_json({"shared": STATE["settings"]["shared"]})
        if p == "/config":
            # 没有 Strata 的 run config：404 让前端隐藏该卡片
            return self.send_json({"error": {"message": "no run config"}}, 404)
        if p == "/mcp":
            return self.send_json({"servers": [], "tools": 0})
        if p == "/v1/status":
            return self.send_json({"model": STATE["engine"].get("model", MODEL_NAME),
                                   "loaded": backend_reachable(), "auto_load": True,
                                   "activity": {"in_flight": 1 if STATE["live"]["state"] != "idle" else 0}})
        if p == "/api/requests":
            if "id" in q:
                rid = q["id"][0]
                return self.send_json(request_detail(rid))
            with LOCK:
                recs = list(STATE["requests"])
            return self.send_json({"requests": recs[-50:], "loaded": backend_reachable(), "auto_load": True})
        if p == "/api/stats":
            return self.send_json({"hardware": STATE["hardware"], "engine": STATE["engine"]})

        # 其余 /v1/*、/props、/slots 等交给 llama-server（含 /v1/models）
        if p.startswith("/v1/") or p in ("/props", "/slots", "/tokenize", "/detokenize"):
            return self.proxy(p, q, method="GET")
        return self.send_json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p in ("/v1/chat/completions", "/v1/completions"):
            return self.proxy_chat(p)
        if p == "/settings":
            try:
                d = json.loads(self.read_body() or b"{}")
            except Exception:
                d = {}
            STATE["settings"]["shared"] = bool(d.get("defaults"))
            STATE["settings"]["defaults"] = d.get("defaults")
            save_settings()
            return self.send_json({"shared": STATE["settings"]["shared"]})
        if p in ("/load", "/unload"):
            return self.send_json({"ok": True, "loaded": backend_reachable()})
        if p == "/v1/images/generations":
            return self.send_json({"error": {"message": "本机未启用图像生成"}}, 501)
        # Claude Code 走 Anthropic 协议：必须原样转发 body，并做实时统计
        if p == "/v1/messages":
            return self.proxy_anthropic(p)
        if p.startswith("/v1/") or p in ("/tokenize", "/detokenize", "/props"):
            return self.proxy(p, {}, method="POST", body=self.read_body())
        return self.send_json({"error": {"message": "not found"}}, 404)

    # ---- 代理（非流式 / GET）
    def proxy(self, path, query, method="GET", body=None):
        url = BACKEND + path + (("?" + urllib.parse.urlencode(query, doseq=True)) if query else "")
        data = body if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", self.headers.get("Content-Type") or "application/json")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.cors()
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.cors()
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_json({"error": {"message": f"backend unreachable: {e}"}}, 502)

    # ---- Anthropic Messages 代理（Claude Code 用；流式 + 实时统计）
    def proxy_anthropic(self, path):
        raw = self.read_body()
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self.send_json({"error": {"type": "invalid_request_error", "message": "invalid JSON"}}, 400)

        stream = bool(body.get("stream"))
        rid = new_request(path, stream, raw.decode("utf-8", "replace"))
        started = time.time()
        prom_before = read_prom()
        with LOCK:
            STATE["live"].update({"state": "reading", "phase": "", "tok_s": None,
                                  "prefill_tok_s_mean": None, "prompt_total": 0, "prompt_read": 0,
                                  "generated": 0,
                                  "max_tokens": int(body.get("max_tokens") or
                                                    STATE["engine"].get("n_predict_default") or 0),
                                  "_started": started})
        # 后台估算提示长度（只影响显示，失败无所谓）
        threading.Thread(target=_tokenize_hint, args=(body,), daemon=True).start()

        req = urllib.request.Request(BACKEND + path, data=raw, method="POST")
        req.add_header("Content-Type", "application/json")
        for h in ("anthropic-version", "anthropic-beta"):
            hv = self.headers.get(h)
            if hv:
                req.add_header(h, hv)

        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            payload = e.read()
            finish_request(rid, {"state": "error", "http_status": e.code, "wallclock_s": time.time() - started})
            with LOCK:
                STATE["live"]["state"] = "idle"
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.cors()
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as e:
            finish_request(rid, {"state": "error", "wallclock_s": time.time() - started})
            with LOCK:
                STATE["live"]["state"] = "idle"
            return self.send_json({"error": {"type": "api_error", "message": f"backend unreachable: {e}"}}, 502)

        if not stream:
            payload = resp.read()
            try:
                j = json.loads(payload)
            except Exception:
                j = {}
            u = j.get("usage") or {}
            reused = int(u.get("cache_read_input_tokens") or 0)
            ptoks = int(u.get("input_tokens") or 0) + reused
            out = int(u.get("output_tokens") or 0)
            wall = time.time() - started
            prompt_ms = 0.0
            prom_after = read_prom()
            if prom_before and prom_after:
                dnew = max(0, prom_after.get("prompt_tokens_total", 0) - prom_before.get("prompt_tokens_total", 0))
                dsecp = prom_after.get("prompt_seconds_total", 0) - prom_before.get("prompt_seconds_total", 0)
                if dnew > 0 and dsecp > 0:
                    prompt_ms = dsecp * 1000.0
                    ptoks = max(ptoks, int(dnew) + reused)
            rec = {"id": rid, "path": path, "model": STATE["engine"].get("model", MODEL_NAME),
                   "started_at": started, "stream": False, "state": "completed", "http_status": 200,
                   "wallclock_s": wall, "prompt_tokens": ptoks,
                   "reused": reused, "output_tokens": out,
                   "prompt_ms": prompt_ms, "decode_ms": 0.0, "decode_tok_s": None, "time": int(started),
                   "finish": "stop", "hit_rate": None, "projection": 0, "duration_s": wall,
                   "pcie_share": None, "usage": {"prompt_tokens": ptoks, "completion_tokens": out},
                   "timings": {}}
            finish_request(rid, rec)
            with LOCK:
                STATE["requests"].append(rec)
                account_request(rec)
                STATE["live"].update({"state": "idle", "generated": 0, "tok_s": None, "prefill_tok_s_mean": None})
            STATE["live"].pop("_started", None)
            STATE["live"].pop("_first_at", None)
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.cors()
        self.end_headers()

        first_at = None
        gen = 0
        ptoks = 0
        reused = 0
        out_tokens = 0
        text_out, text_reason = [], []
        buf = b""
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except Exception:
                    pass
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", "replace").strip()
                    if not s.startswith("data:"):
                        continue
                    data = s[5:].strip()
                    if not data:
                        continue
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    ev = j.get("type")
                    if ev == "message_start":
                        u = ((j.get("message") or {}).get("usage") or {})
                        ptoks = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                        reused = int(u.get("cache_read_input_tokens") or 0)
                    elif ev == "content_block_delta":
                        d = j.get("delta") or {}
                        piece = d.get("text")
                        rpiece = d.get("thinking")
                        if piece or rpiece:
                            if first_at is None:
                                first_at = time.time()
                                # 用引擎计数器算这次请求真实的 prefill 速率
                                prom_now = read_prom()
                                prefill_live = None
                                if prom_before and prom_now:
                                    dnew = max(0, prom_now.get("prompt_tokens_total", 0) -
                                                  prom_before.get("prompt_tokens_total", 0))
                                    dsec = prom_now.get("prompt_seconds_total", 0) - prom_before.get("prompt_seconds_total", 0)
                                    if dnew > 0 and dsec > 0:
                                        prefill_live = dnew / dsec
                                    if dnew:
                                        ptoks = max(ptoks, int(dnew) + reused)
                                with LOCK:
                                    STATE["live"]["state"] = "generating"
                                    STATE["live"]["_first_at"] = first_at
                                    STATE["live"]["prompt_total"] = ptoks
                                    STATE["live"]["prompt_read"] = ptoks
                                    if prefill_live:
                                        STATE["live"]["prefill_tok_s_mean"] = prefill_live
                            gen += 1
                            if piece:
                                text_out.append(piece)
                            if rpiece:
                                text_reason.append(rpiece)
                            with LOCK:
                                now = time.time()
                                STATE["live"]["generated"] = gen
                                if first_at and now > first_at:
                                    v = gen / (now - first_at)
                                    if 0 < v < 1000:      # 首包可能一次带很多 delta，夹掉尖峰
                                        STATE["live"]["tok_s"] = v
                    elif ev == "message_delta":
                        u = j.get("usage") or {}
                        if u.get("output_tokens"):
                            out_tokens = int(u["output_tokens"])
        except Exception:
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass

        wall = time.time() - started
        decode_s = (time.time() - first_at) if first_at else 0.0
        out_tokens = out_tokens or gen
        # 用引擎自己的计数器算本请求的真实 decode 速率（比客户端侧计时准）
        prom_after = read_prom()
        prompt_ms = 0.0
        if prom_before and prom_after:
            dtok = prom_after.get("tokens_predicted_total", 0) - prom_before.get("tokens_predicted_total", 0)
            dsec = prom_after.get("tokens_predicted_seconds_total", 0) - prom_before.get("tokens_predicted_seconds_total", 0)
            if dtok > 0 and dsec > 0:
                decode_s = dsec
                out_tokens = int(dtok)
            dnew = max(0, prom_after.get("prompt_tokens_total", 0) - prom_before.get("prompt_tokens_total", 0))
            dsecp = prom_after.get("prompt_seconds_total", 0) - prom_before.get("prompt_seconds_total", 0)
            if dnew > 0 and dsecp > 0:
                prompt_ms = dsecp * 1000.0
                ptoks = max(ptoks, int(dnew) + reused)
        rec = {"id": rid, "path": path, "model": STATE["engine"].get("model", MODEL_NAME),
               "started_at": started, "stream": True, "state": "completed", "http_status": 200,
               "wallclock_s": wall, "prompt_tokens": ptoks, "reused": reused, "output_tokens": out_tokens,
               "prompt_ms": prompt_ms, "decode_ms": decode_s * 1000.0,
               "decode_tok_s": (out_tokens / decode_s) if decode_s > 0 else None,
               "time": int(started), "finish": "stop", "hit_rate": None, "projection": 0,
               "duration_s": wall, "pcie_share": None,
               "usage": {"prompt_tokens": ptoks, "completion_tokens": out_tokens,
                         "total_tokens": ptoks + out_tokens},
               "timings": {"predicted_n": out_tokens, "predicted_ms": decode_s * 1000.0,
                           "predicted_per_second": (out_tokens / decode_s) if decode_s > 0 else 0.0}}
        finish_request(rid, rec, {"output": "".join(text_out), "reasoning": "".join(text_reason),
                                  "first_token_s": (first_at - started) if first_at else None})
        with LOCK:
            STATE["requests"].append(rec)
            account_request(rec)
            STATE["live"].update({"state": "idle", "phase": "", "tok_s": None, "generated": 0, "prefill_tok_s_mean": None})
            STATE["live"].pop("_started", None)
            STATE["live"].pop("_first_at", None)

    # ---- 聊天代理（流式 + 实时统计）
    def proxy_chat(self, path):
        raw = self.read_body()
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self.send_json({"error": {"message": "invalid JSON"}}, 400)

        # 把 Strata 专有字段翻译成 llama.cpp 能理解的形式
        body.pop("strata_mcp", None)
        body.pop("experimental_speed_projection", None)
        eff = body.pop("reasoning_effort", None)
        if eff is not None:
            ctk = body.get("chat_template_kwargs") or {}
            if isinstance(ctk, dict):
                ctk.setdefault("enable_thinking", str(eff).lower() not in ("none", "off", "0"))
                body["chat_template_kwargs"] = ctk
        stream = bool(body.get("stream"))
        if stream:
            so = body.get("stream_options") or {}
            if isinstance(so, dict):
                so["include_usage"] = True
                body["stream_options"] = so

        rid = new_request(path, stream, raw.decode("utf-8", "replace"))
        started = time.time()
        with LOCK:
            STATE["live"].update({"state": "reading", "phase": "", "tok_s": None,
                                  "prefill_tok_s_mean": None, "prompt_total": 0, "prompt_read": 0,
                                  "generated": 0,
                                  "max_tokens": int(body.get("max_tokens") or
                                                    STATE["engine"].get("n_predict_default") or 0),
                                  "_started": started})
        prom_before = read_prom()          # 用于算本请求的 prompt token 数（引擎计数器差值）

        url = BACKEND + path
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            payload = e.read()
            finish_request(rid, {"state": "error", "http_status": e.code, "wallclock_s": time.time() - started})
            with LOCK:
                STATE["live"]["state"] = "idle"
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.cors()
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as e:
            finish_request(rid, {"state": "error", "wallclock_s": time.time() - started})
            with LOCK:
                STATE["live"]["state"] = "idle"
            return self.send_json({"error": {"message": f"backend unreachable: {e}"}}, 502)

        if not stream:
            payload = resp.read()
            self._finish_nonstream(rid, started, payload)
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.cors()
        self.end_headers()

        first_at = None
        gen = 0
        prompt_tokens = 0
        timings = {}
        text_out, text_reason = [], []
        buf = b""
        client_gone = False
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except Exception:
                    client_gone = True
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    s = line.decode("utf-8", "replace").strip()
                    if not s.startswith("data:"):
                        continue
                    data = s[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(j.get("timings"), dict):
                        timings = j["timings"]
                    u = j.get("usage") or {}
                    if u.get("prompt_tokens"):
                        prompt_tokens = u["prompt_tokens"]
                    d = ((j.get("choices") or [{}])[0] or {}).get("delta") or {}
                    piece = d.get("content")
                    rpiece = d.get("reasoning_content")
                    if piece or rpiece:
                        if first_at is None:
                            first_at = time.time()
                            prom_now = read_prom()
                            ptoks_live, prefill_live = 0, None
                            if prom_before and prom_now:
                                d_new = max(0, prom_now.get("prompt_tokens_total", 0) -
                                               prom_before.get("prompt_tokens_total", 0))
                                d_cache = max(0, prom_now.get("prompt_tokens_cached_total", 0) -
                                                 prom_before.get("prompt_tokens_cached_total", 0))
                                ptoks_live = int(d_new + d_cache)      # 提示总长（命中缓存的也算）
                                dsec = prom_now.get("prompt_seconds_total", 0) - prom_before.get("prompt_seconds_total", 0)
                                if d_new > 0 and dsec > 0:
                                    prefill_live = d_new / dsec        # 只有新读的部分算 prefill 速率
                            with LOCK:
                                STATE["live"]["state"] = "generating"
                                STATE["live"]["_first_at"] = first_at
                                STATE["live"]["prompt_total"] = ptoks_live or STATE["live"].get("prompt_total") or 0
                                STATE["live"]["prompt_read"] = STATE["live"]["prompt_total"]
                                if prefill_live:
                                    STATE["live"]["prefill_tok_s_mean"] = prefill_live
                                if ptoks_live:
                                    prompt_tokens = ptoks_live
                        gen += 1
                        if piece:
                            text_out.append(piece)
                        if rpiece:
                            text_reason.append(rpiece)
                        with LOCK:
                            now = time.time()
                            STATE["live"]["generated"] = gen
                            if first_at and now > first_at:
                                STATE["live"]["tok_s"] = gen / (now - first_at)
                if client_gone:
                    continue
        except Exception:
            pass
        finally:
            try:
                resp.close()
            except Exception:
                pass

        wall = time.time() - started
        decode_s = (time.time() - first_at) if first_at else 0.0
        prompt_ms = timings.get("prompt_ms") or (0.0 if first_at is None else (first_at - started) * 1000.0)
        decode_ms = timings.get("predicted_ms") or decode_s * 1000.0
        out_tokens = (timings.get("predicted_n") or gen) or 0
        reused = timings.get("cache_n") or 0
        ptoks = (timings.get("prompt_n") or 0) + reused
        if not ptoks:
            ptoks = prompt_tokens or 0
        rec = {"id": rid, "path": path, "model": STATE["engine"].get("model", MODEL_NAME),
               "started_at": started, "stream": True,
               "state": "completed", "http_status": 200, "wallclock_s": wall,
               "prompt_tokens": ptoks, "reused": reused, "output_tokens": out_tokens,
               "prompt_ms": prompt_ms, "decode_ms": decode_ms,
               "decode_tok_s": (out_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else None,
               "time": int(started), "finish": "stop", "hit_rate": None, "projection": 0,
               "duration_s": wall, "pcie_share": None,
               "usage": {"prompt_tokens": ptoks, "completion_tokens": out_tokens, "total_tokens": ptoks + out_tokens},
               "timings": {"prompt_n": ptoks, "predicted_n": out_tokens, "prompt_ms": prompt_ms,
                           "predicted_ms": decode_ms,
                           "predicted_per_second": (out_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0}}
        finish_request(rid, rec, {"output": "".join(text_out), "reasoning": "".join(text_reason),
                                  "first_token_s": (first_at - started) if first_at else None})
        with LOCK:
            STATE["requests"].append(rec)
            account_request(rec)
            STATE["live"].update({"state": "idle", "phase": "", "tok_s": None, "generated": 0, "prefill_tok_s_mean": None})
            STATE["live"].pop("_started", None)
            STATE["live"].pop("_first_at", None)

    def _finish_nonstream(self, rid, started, payload):
        try:
            j = json.loads(payload)
        except Exception:
            j = {}
        t = j.get("timings") or {}
        ptoks = (j.get("usage") or {}).get("prompt_tokens") or t.get("prompt_n") or 0
        out = (j.get("usage") or {}).get("completion_tokens") or t.get("predicted_n") or 0
        wall = time.time() - started
        rec = {"id": rid, "path": "/v1/chat/completions", "model": STATE["engine"].get("model", MODEL_NAME),
               "started_at": started, "stream": False,
               "state": "completed", "http_status": 200, "wallclock_s": wall,
               "prompt_tokens": ptoks, "reused": t.get("cache_n") or 0, "output_tokens": out,
               "prompt_ms": t.get("prompt_ms") or 0.0, "decode_ms": t.get("predicted_ms") or 0.0,
               "decode_tok_s": t.get("predicted_per_second"), "time": int(started), "finish": "stop",
               "hit_rate": None, "projection": 0, "duration_s": wall, "pcie_share": None,
               "usage": j.get("usage") or {}, "timings": t}
        finish_request(rid, rec)
        with LOCK:
            STATE["requests"].append(rec)
            account_request(rec)
            STATE["live"].update({"state": "idle", "generated": 0, "tok_s": None, "prefill_tok_s_mean": None})
            STATE["live"].pop("_started", None)
            STATE["live"].pop("_first_at", None)


def _text_of(content):
    """Anthropic/OpenAI 两种 content 形态都取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    out.append(str(p.get("text") or ""))
                elif isinstance(p.get("content"), str):
                    out.append(p["content"])
            elif isinstance(p, str):
                out.append(p)
        return "\n".join(out)
    return ""


def _tokenize_hint(body):
    """把消息拼起来让后端 tokenize 一次，用于「Reading prompt」的 token 数显示。"""
    try:
        msgs = body.get("messages") or []
        text = "\n".join(_text_of(m.get("content")) for m in msgs if isinstance(m, dict))
        if not text or len(text) > 400000:
            return
        req = urllib.request.Request(BACKEND + "/tokenize",
                                     data=json.dumps({"content": text}).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as r:
            n = len(json.loads(r.read().decode("utf-8")).get("tokens") or [])
        with LOCK:
            if STATE["live"].get("state") in ("reading", "generating") and n > 0:
                STATE["live"]["prompt_total"] = n
                STATE["live"]["prompt_tokens"] = n
    except Exception:
        pass


def request_detail(rid):
    with LOCK:
        rec = next((r for r in STATE["requests"] if r["id"] == rid), None)
        det = dict(REQ_DETAIL.get(rid) or {})
    if rec is None:
        return {"error": {"message": "no such request"}}
    out = dict(rec)
    out.update({"load_s": 0.0, "queue_s": 0.0,
                "first_token_s": det.get("first_token_s"),
                "input": det.get("input", ""), "output": det.get("output", ""),
                "reasoning": det.get("reasoning", ""), "response_format": "text",
                "error": det.get("error")})
    return out


def build_metrics(all_requests=False):
    with LOCK:
        live = dict(STATE["live"])
        hw = dict(STATE["hardware"])
        st = dict(STATE["hardware_static"])
        eng = dict(STATE["engine"])
        hist = {k: list(v) for k, v in STATE["history"].items()}
        reqs = list(STATE["requests"])
        totals = dict(STATE["totals"])
        kept = len(reqs)
    # 注意：llama.cpp 在整段提示读完前不更新任何计数，所以「读提示」阶段没有真实的 prefill 速率可算。
    # 这里不做任何推算（否则会得到「总量÷已用时间」这种一路衰减的假数字）；
    # 真实值由首个 token 到达时用引擎计数器差值算出，并在生成阶段显示。
    live.pop("_started", None)
    live.pop("_first_at", None)

    reqs = [r for r in reqs if r.get("state") == "completed" or r.get("state") == "error"]
    if not all_requests:
        reqs = reqs[-12:]
    return {"live": live, "hardware": hw, "hardware_static": st, "engine": eng,
            "history": hist, "requests": list(reversed(reqs)), "totals": totals,
            "requests_kept": kept, "conversation_cache": {"enabled": False}}


# ----------------------------------------------------------------------------- 后台采样
def poller():
    while True:
        try:
            sample_hardware()
            # 引擎侧实时速率（最准）：generating → predicted_tokens_seconds；非 idle → prompt_tokens_seconds
            with LOCK:
                busy = STATE["live"].get("state") != "idle"
            if busy:
                prom = read_prom()
                if prom:
                    with LOCK:
                        if STATE["live"].get("state") == "generating":
                            v = prom.get("predicted_tokens_seconds")
                            # 首 token 时引擎会给出一个除零式的尖峰，夹掉
                            if v is not None and 0 < v < 1000 and STATE["live"].get("generated", 0) >= 3:
                                STATE["live"]["tok_s"] = v
            with LOCK:
                h = STATE["history"]
                liv = STATE["live"]
                h["tok_s"].append(liv.get("tok_s") if liv.get("state") == "generating" else None)
                h["prefill_tok_s_mean"].append(liv.get("prefill_tok_s_mean") if busy else None)
            # 后端可能晚于 UI 启动：直到拿到真实的上下文长度为止，每秒重试一次
            if not STATE["engine"] or not STATE["engine"].get("max_context"):
                detect_engine()
        except Exception:
            pass
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--backend", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    global BACKEND, MODEL_NAME
    BACKEND = args.backend.rstrip("/")
    load_settings()
    hardware_static()
    sample_hardware()
    detect_engine()
    threading.Thread(target=poller, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"[serve-ui] Strata UI on http://{args.host}:{args.port}  -> backend {BACKEND}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
