# environment/ —— 运行与编译环境

这个目录记录"跑起来依赖哪些外部组件"，以及实测过的环境坑。

| 文件 | 内容 |
|---|---|
| `ENVIRONMENT.md` | 运行时依赖（NCCL、二进制、驱动/CUDA）、编译依赖、网络与磁盘实测 |

---

## 三个最容易踩的点

1. **NCCL 通常不在系统里**
   最省事的方式是从 Python wheel 取：

   ```bash
   python3 -m venv ~/nccl-venv
   ~/nccl-venv/bin/pip install nvidia-nccl-cu12
   find ~/nccl-venv -name libnccl.so.2
   ```

   启动前把它的 `lib/` 加进 `LD_LIBRARY_PATH`，否则
   `error while loading shared libraries: libnccl.so.2`。

   张量并行必需；层切分不需要，但二进制仍然链接了它。

2. **系统可能没有 cmake/ninja**
   `apt install cmake ninja-build` 即可；或者用任意带它们的 venv，把 `bin/` 加进 `PATH`。
   只有编译需要，运行不需要。

3. **不要为了这个项目升级驱动/内核**
   推理只依赖用户态二进制与 NCCL。驱动升级带来的重启风险，远大于收益。
   如果 CUDA toolkit 与驱动版本不完全一致，通常向后兼容。

其余细节见 `ENVIRONMENT.md`。
