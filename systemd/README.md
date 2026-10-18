# systemd/ —— 用户级常驻服务

两个单元：

| 单元 | 作用 | 端口 |
|---|---|---|
| `bolt-serve.service` | llama-server（3 卡 TP + 1M） | 127.0.0.1:8080 |
| `bolt-strata-ui.service` | Trata 网页 + API 代理 | 0.0.0.0:8001 |

---

## 1. 安装

```bash
mkdir -p ~/.config/systemd/user
cp systemd/bolt-serve.service systemd/bolt-strata-ui.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bolt-serve.service bolt-strata-ui.service
```

开机自启（不需要登录）：

```bash
sudo loginctl enable-linger "$USER"
```

---

## 2. 使用前必须改的地方

单元文件里的路径假设仓库在 `~/Nemotron3Super`，NCCL 在
`~/nccl-venv/lib/python3.12/site-packages/nvidia/nccl`。

```bash
# 看实际路径
pwd
find ~/nccl-venv -name libnccl.so.2
```

如果不是这两个路径，编辑 `~/.config/systemd/user/bolt-serve.service`：

- `WorkingDirectory=`
- `LD_LIBRARY_PATH=`
- `ExecStart=` 里的二进制、模型路径
- `-t 17` 改成你的 `nproc - 1`

改完：

```bash
systemctl --user daemon-reload
systemctl --user restart bolt-serve.service
```

---

## 3. 常用命令

```bash
systemctl --user status bolt-serve.service
systemctl --user restart bolt-serve.service
journalctl --user -u bolt-serve.service -f
journalctl --user -u bolt-serve.service -n 100 --no-pager
```

两个一起：

```bash
systemctl --user start   bolt-serve.service bolt-strata-ui.service
systemctl --user stop    bolt-serve.service bolt-strata-ui.service
```

---

## 4. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `status=203/EXEC` | 路径错/无执行权限 | 用绝对路径，`ls -l` 检查 |
| 启动即退出 | 找不到 `libnccl.so.2` | 检查 `LD_LIBRARY_PATH` |
| 反复重启 | OOM | `MemorySwapMax=0` 下 OOM 会失败；先降 `-ub`/`-c` |
| 开机不启动 | 没 enable-linger | `sudo loginctl enable-linger "$USER"` |
| 改了文件不生效 | 没 daemon-reload | `systemctl --user daemon-reload` |

> systemd 不展开 `~`，只展开 `%h`（用户 home）。单元里用的是 `%h`。

---

## 5. 不想用 systemd

直接用脚本：

```bash
bash scripts/run-tp.sh --daemon
bash scripts/start-ui.sh
```

脚本和 systemd 不要同时用，否则会抢 8080/8001 端口。
