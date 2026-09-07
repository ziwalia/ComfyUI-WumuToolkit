# ComfyUI-WumuToolkit

wumu-movie 影视流水线专用 ComfyUI 节点集：**一个节点产出角色全套装帧参考图**，产物自动按引擎契约落盘命名。

```
基础定妆照 → 造型定妆照 → 证件照/正面/侧面/背面 → 1920×1080 四联图
```

## 节点

### Wumu 定妆照工坊（FLUX.2 Dev）

- 文生图生成穿内衣白底全身**基础定妆照**（锁定脸/发型/体型/肤色）
- 通过 FLUX.2 **ReferenceLatent 参考链**给同一张脸换装，生成最多 5 套**造型定妆照**（脸不变衣服换）
- 原生兼容 Kontext 提示词（`Edit the reference photo: the SAME person...`）
- 人物描述框两种填法皆可：只写身份特征（自动套影棚模板）或粘贴完整引擎提示词（自动剥离模板提取身份核心）
- 3 种模式：全流程 / 仅基础照 / 仅造型照（需接 `base_image` 输入口）
- 可选 Turbo LoRA 加速：20 步 → 8 步（快 2.5 倍，画质略降）
- 风格下拉由 `styles.json` 驱动（9 种风格），节点内实时显示所需 LoRA 是否已下载

### Wumu 四联图工坊（Klein）

- 输入造型定妆照（联动工坊输出或 LoadImage 手动喂图），每个造型自动生成四条参考链：
  **证件照 1024×1024 → 全身正视 1024×1920 → 侧视（参考正视输出+原图）→ 后视（仅参考正视输出）**
- 合成 1920×1080 白底四联图，四链锁定纯白摄影棚背景
- 自动跳过未连线/黑图槽位；种子策略可重复（每槽 +100，四链再 +1/+10/+20/+30）
- 每造型落盘 5 个文件：`角色_造型_证件照/正面图/侧面图/背面图/四联图.png`

## 模型准备

| 文件（下拉按关键词自动过滤） | 放置目录 | 用途 |
|---|---|---|
| `flux2_dev_fp8mixed.safetensors` | `models/diffusion_models/` | FLUX.2 Dev 底模（定妆照） |
| `mistral_3_small_flux2_bf16.safetensors` | `models/text_encoders/` | FLUX.2 文本编码器 |
| `flux-2-klein-9b-kv-fp8.safetensors` | `models/diffusion_models/` | Klein 9B 快速底模（四联图） |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` | Qwen3 编码器（中文提示词） |
| `full_encoder_small_decoder.safetensors` | `models/vae/` | FLUX.2 VAE（128ch latent） |

可选：

- `Flux_2-Turbo-LoRA_comfyui.safetensors` → `models/loras/`（8 步加速）
- 风格 LoRA 若缺文件，节点会直接报出下载地址（见 `styles.json` 的 `hf_url` 字段）

16G 显存即可运行；两套模型由 ComfyUI 自动换入换出。

## 语言 / Language

两个工坊节点均有 `language` 下拉（默认 **中文**）：

- **中文**：产物落盘 `资产\角色\`，文件名 `角色_造型_证件照/正面图/侧面图/背面图/四联图.png` —— wumu-movie 引擎 / 导演台 `角色名@造型名` 引用契约所需，**跑 wumu-movie 流水线请保持此档**
- **English**：产物落盘 `assets\characters\`，文件名 `character_outfit_id/front/side/back/quadtych.png`，四视角生成提示词与 info/日志同步切英文 —— 供国际用户接入自有流水线

参数提示（tooltip）语言为全局设置：在节点上切换 `language` 后自动保存，**刷新页面**后所有 Wumu 节点的参数提示切换语言（模式选项、风格名、输出端口名等数据项保持原文）。

## 安装

**方式一：ComfyUI Manager（推荐）**

Custom Nodes Manager → **Install via Git URL** → 填入本仓库地址，重启 ComfyUI。

**方式二：手动 clone**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ziwalia/ComfyUI-WumuToolkit
```

**更新**

- Manager：Custom Nodes Manager → 找到 WumuToolkit → Update
- 或手动：插件目录里 `git pull`，重启 ComfyUI

## 使用

仓库内附带完整教程和联动工作流：

- [`人物资产全流程-使用教程.md`](人物资产全流程-使用教程.md) —— 逐步操作、参数速查、常见问题
- [`人物资产全流程-定妆照+四联图.json`](人物资产全流程-定妆照+四联图.json) —— 两节点已接线的联动工作流（拖入 ComfyUI 画布即可用，含参数手册便签）

产物默认保存到 `save_dir\资产\角色\`（默认 `F:\AI\MINIMAXH3`，按需修改 save_dir 参数），文件名与 wumu-movie 引擎 / 导演台资产库的 `角色名@造型名` 引用契约对齐。

## License

[MIT](LICENSE)
