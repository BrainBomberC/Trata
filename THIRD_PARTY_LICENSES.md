# 第三方许可与归属

本仓库自己的代码与文档使用 [MIT](LICENSE)。但它包含或派生自以下第三方作品，
各自保留原许可。**分发或再发布时请一并保留本文件。**

---

## 1. llama.cpp（补丁的上游）

- 项目：https://github.com/ggml-org/llama.cpp
- 本仓库钉定的上游 commit：
  - 生产组（补丁 0005）：`5aeac37a35b44fde4bdffaec9159f2fcb253b622`
  - 回退组（补丁 0001–0004）：`69f201a2051ea9b9e9b50c3cc56afd9e2ae64414`
- 许可：MIT
- 版权：`Copyright (c) 2023-2026 The ggml authors`

`patches/*.patch` 是对 llama.cpp 源码的修改，属于 MIT 许可下的衍生作品。
补丁本身也以 MIT 发布。补丁**不是** llama.cpp 官方代码，上游不为其提供支持。

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Strata（网页前端）

- 项目：https://github.com/Niko1221/Strata
- 许可：MIT
- 版权：`Copyright (c) 2026 Niko1221 and the Strata contributors`

`ui/strata/` 中的 HTML/CSS/JS 源自 Strata，本项目在其基础上改造为 **Trata**：
重做品牌、配色与进度显示逻辑，并适配本项目后端。页面页脚保留了
“UI based on Strata” 的出处链接。

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Outfit 字体

- 项目：https://github.com/Outfitio/Outfit-Fonts
- 版权：`Copyright 2021 The Outfit Project Authors`
- 许可：SIL Open Font License, Version 1.1

文件：`ui/strata/web/fonts/outfit-latin-wght.woff2`、
`ui/strata/web/fonts/outfit-latin-ext-wght.woff2`

完整许可文本随字体一同分发：`ui/strata/web/fonts/OFL.txt`。
OFL 允许嵌入与再分发，但**不允许单独售卖字体**；若修改字体文件，
不得使用 “Outfit” 作为保留字体名。

---

## 4. NVIDIA Nemotron-3.5-Bolt-74B-A9B（模型权重）

- 模型页：https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B
- 量化文件托管：https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF
- 镜像：https://modelscope.cn/models/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF

**本仓库不包含、不分发任何模型权重。** 主模型、DFlash 草稿模型
（`dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf`）、视觉投影器，
以及由 `scripts/quantize-tpfit.sh` 生成的再量化文件，均受 NVIDIA 模型许可证
与量化发布者条款约束。下载、使用、商用前请阅读模型页面的原始许可证。
本仓库的 MIT 许可**不覆盖**模型权重。

---

## 5. 本仓库原创部分

以下文件为原创，使用根目录 MIT 许可：

- `docs/`、`blog/`、各目录的 `README.md`
- `scripts/` 下的全部脚本
- `patches/` 下的补丁
- `systemd/` 下的服务单元
- `benchmarks/logs/` 下的实测日志

`ui/strata/` 是第 2 节的衍生作品，不适用本仓库的版权声明。
