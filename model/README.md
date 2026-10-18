# model/ —— 模型文件（不进入 git）

**本仓库不分发模型权重。** 这个目录只放一个说明；实际文件由你下载/量化生成，
并被 `.gitignore` 排除。

---

## 1. 需要的文件

| 文件 | 大小 | 必需 | 来源 |
|---|---:|---|---|
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf` | 22,018,326,464 B | ✅ | [ModelScope](https://modelscope.cn/models/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF) / [HF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF) |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf` | 21,025,816,576 B | TP+1M 时 | 由 `scripts/quantize-tpfit.sh` 生成 |
| `mmproj-BF16.gguf` | 1,733,692,672 B | 可选（图片） | 同一仓库 |

推荐目录结构：

```
model/
├── bolt74b/
│   ├── NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS.gguf
│   ├── NVIDIA-Nemotron-3.5-Bolt-74B-A9B-IQ2_XXS-tpfit.gguf
│   └── .tpfit-types.txt          # 量化时自动生成
├── mmproj-BF16.gguf
├── ui-settings.json              # Trata 保存的默认参数
└── *.log                         # 运行日志
```

---

## 2. 下载

```bash
bash scripts/download-model.sh
bash scripts/verify-model.sh --all
```

SHA256：

```
IQ2_XXS  30c4eac4cdd63a7c73fcf29254a9fd3f40ab7ef09b555c1dd6653140cc94108f
mmproj   acad03175d80ce7caea6854795e9ecc978e86d4b4caae0125efba5f970216f4f
```

> SHA 以托管方当前页面为准。若下载后大小对但 SHA 不符，删除重下，不要继续使用。

托管版 mmproj 有 519 个张量，注意力权重是分开的 `v.blk.N.attn_q/k/v`；
本项目自转的版本只有 393 个张量，用 fused `v.blk.N.attn_qkv`。两种布局
都已在打完 0004 补丁的构建上实测：加载成功，512×512 图像编码成功。
视觉编码固定走 CPU（`--no-mmproj-offload`），与测试路径一致。

---

## 3. 为什么选 IQ2_XXS

| 量化 | 大小 | 本机 46.9 GiB 内存 | 3×12 GB 显存 | 结论 |
|---|---:|---|---|---|
| IQ2_XXS | 20.5 GiB | ✅ 余量 26 GiB | ✅ | **采用** |
| UD-IQ1_M | 18.9 GiB | ✅ | ✅ | 更小但质量更差 |
| UD-IQ3_XXS | 29.4 GiB | ✅ | ⚠️ TP+1M 装不下 | 质量更好，显存不够 |
| IQ4_XS | 43.4 GiB | ❌ 余量 3.5 GiB | ❌ | 实测掉盘，11–14 tok/s |

判据（第一性原理）：

```
模型文件 ≤ 物理内存 − 6~8 GiB
每卡权重 + KV + 缓冲 ≤ 单卡显存
```

详细推导见 [docs/02-选型计算](../docs/02-选型计算.md)。

---

## 4. tpfit 是什么

不是剪枝，是**混合精度重量化**：

| 张量 | 数量 | 类型 |
|---|---:|---|
| `ssm_in` / `ssm_out` | 80 | q3_K |
| 专家（`*exps*`） | 160 | iq2_xxs（保持原样） |
| 其余稠密张量 | — | q6_K |

文件从 20.51 GiB 降到 19.58 GiB，每卡少约 0.31 GiB——
这是 1M 上下文 + 3 卡 TP 能装下的关键。

完整配方与命令见 [docs/06-裁剪与量化](../docs/06-裁剪与量化.md)。

---

## 5. 磁盘与内存

| 阶段 | 需要 |
|---|---|
| 下载 | 约 44 GB 空闲（分块 + 最终文件） |
| 量化 | 约 45 GB（输入 + 输出） |
| 运行 | 模型文件必须能被页缓存容纳；内存至少 文件 + 7 GiB |

模型放在 **NVMe/SATA** 上。USB2（43.8 MB/s）加载 22 GB 要 8 分钟，
运行中掉盘会直接掉到个位数 tok/s。

---

## 6. 不要提交到 git

根目录 `.gitignore` 已排除：

```
model/*
!model/README.md
*.gguf
```

如果你改过 `.gitignore`，提交前确认：

```bash
git status --short
du -sh .git
```

仓库里不应该出现任何 `.gguf`。
