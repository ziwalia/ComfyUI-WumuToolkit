# ComfyUI-WumuToolkit —— 人物资产全流程工具包（FLUX.2）

wumu-movie 影视流水线专用 ComfyUI 节点集：**一次 Queue，产出一个角色的全套标准资产**，自动按导演台资产库契约落盘命名。

```
基础定妆照 → N 套造型定妆照 → 每套造型的 证件照/正面/侧面/背面 + 1920×1080 四联图
```

| 模块 | 节点 | 引擎 | 产出 |
|---|---|---|---|
| 定妆照工坊 | `WumuCharacterAtelier` | FLUX.2 Dev | 素体白棚基础定妆照 + 最多 5 套造型定妆照 |
| 四联图工坊 | `WumuTriptychAtelier` | FLUX.2 Klein | 每套造型 4 视角 + 四联合成图（4 步快出） |
| 联动工作流 | `人物资产全流程-定妆照+四联图.json` | 双工坊串联 | 一次队列全自动，每造型 6 个文件 |

**核心机制**：造型换装使用 FLUX.2 原生 **ReferenceLatent 参考链**（Kontext 句式提示词），换衣服不换脸 —— 同一角色跨造型保持一致性。16G 显存即可运行，两套模型由 ComfyUI 自动换入换出。

---

## 📦 安装

**方式一：ComfyUI Manager（推荐）**
Custom Nodes Manager → Install via Git URL → 填入本仓库地址 → 重启 ComfyUI。

**方式二：手动 clone**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ziwalia/ComfyUI-WumuToolkit
# 国内加速：
git clone https://gitclone.com/github.com/ziwalia/ComfyUI-WumuToolkit
```

**更新**：Manager 里 Update，或插件目录 `git pull`，重启 ComfyUI。

安装后节点位于 `wumu` 分类；把仓库根目录的 `人物资产全流程-定妆照+四联图.json` 拖进画布即可开始（工作流内嵌参数手册便签）。

---

## 🧠 模型准备

| 文件 | 放置目录 | 用途 |
|---|---|---|
| `flux2_dev_fp8mixed.safetensors` | `models/diffusion_models/` | FLUX.2 Dev 底模（定妆照） |
| `flux-2-klein-9b-kv-fp8.safetensors` | `models/diffusion_models/` | Klein 9B 快速底模（四联图） |
| `mistral_3_small_flux2_bf16.safetensors` | `models/text_encoders/` | FLUX.2 文本编码器 |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` | Qwen3 编码器 |
| `full_encoder_small_decoder.safetensors` | `models/vae/` | FLUX.2 VAE（128ch latent） |

可选：
- `Flux_2-Turbo-LoRA_comfyui.safetensors` → `models/loras/`（8 步加速 2.5 倍）
- 风格 LoRA 缺文件时节点会直接报出下载地址（见 `styles.json` 的 `hf_url` 字段）

---

## 🚀 快速开始：一次队列出全套资产

1. 打开 `人物资产全流程-定妆照+四联图.json`
2. **左节点（定妆照工坊）**改三处：
   - `角色名`：中文，用于文件命名
   - `人物英文描述`：只写身份特征，**不写服装**（写法见下方样板）
   - 造型槽：要几套衣服就启用几个槽，填造型名 + 服装提示词
3. **右节点（四联图工坊）**：`角色名` 与造型名和左边**保持一致**（仅用于命名）
4. 点 Queue。FLUX.2 Dev 每造型约 1-2 分钟，Klein 每造型四链约半分钟
5. 产物自动保存（默认 `F:\AI\MINIMAXH3\资产\角色\`，改 `save_dir` 参数即可）：

```
资产/角色/
├─ 角色名_基础定妆照.png
└─ 每套造型 × 6：
   角色名_造型名_定妆照 / _证件照 / _正面图 / _侧面图 / _背面图 / _四联图
```

文件名与 wumu-movie 引擎 / 导演台资产库的 `角色名@造型名` 引用契约对齐。

---

## 🏭 节点 A：定妆照工坊（WumuCharacterAtelier）

**三种模式**：

| 模式 | 行为 |
|---|---|
| `①+② 全流程` | 文生图出基础照（素体白棚）→ 参考图生图换装出各造型 |
| `仅① 基础定妆照` | 只出基础照 |
| `仅② 造型定妆照` | 用 `base_image` 输入口喂已有基础照直接换装 |

### 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `desc_en` | — | 人物英文描述，只写身份特征：**国籍人种放最前**（防漂移）→ 年龄体型 → 脸型发型 → 肤质 → 特殊标记。也可直接粘贴引擎生成的完整提示词（节点自动剥离模板提取身份核心） |
| `style_name` / `style_strength` | 写实电影 / 0.8 | 风格下拉由 `styles.json` 驱动（9 种），写实类 0.6-0.8，风格化 0.85-1.0 |
| `steps` / `guidance` | 20 / 4.0 | FLUX.2 标准；长提示词 guidance 可降 2-3 |
| `width × height` | 832×1216 | 竖构图全身照标准，**不要改** |
| `seed` | 随机 | 基础用 seed，造型 N 自动用 seed+N×10 |
| `use_turbo` | 关 | 自动挂 Turbo LoRA（步数记得改 8），快 2.5 倍画质略降。测试期开，正式出片关 |
| 造型槽 ×5 | 槽 1 启用 | 启用勾选 + 中文名 + 服装提示词（Kontext 句式，见下方样板） |

### 🆕 v1.1 功能一：基础照图生图

勾选 **`base_use_reference`** + 连接 **`base_reference_image`** 输入口（喂已有定妆照/素体照/真人参考照，自动缩放到出图尺寸），① 基础照即改为图生图。用 **`base_denoise`** 控制贴合度（ComfyUI 官方 KSampler 同款部分加噪公式）：

| denoise | 效果 | 适用 |
|---|---|---|
| 0.9~1.0 | 几乎重画，只参考构图姿势 | 换风格重出 |
| 0.7~0.8 | 保人物、改质感细节（**推荐**） | 修皮肤/光影/清晰度 |
| 0.4~0.6 | 轻度修整 | 微调 |

### 🆕 v1.1 功能二：外部基础照重抽造型 ⭐

**场景：全流程跑完，基础照满意、造型不满意 —— 不用重跑 ①。**

1. 勾选 **`slot_use_external_base`**
2. 加 `LoadImage` 加载已保存的 `角色名_基础定妆照.png`，连到 `base_image` 输入口
3. 直接 Queue —— **跳过 ①（省 1-2 分钟）只重抽造型**；基础照输出口透传原图，下游四联图照常工作
4. 不满意换种子再 Queue；满意后**取消勾选**恢复正常全流程

---

## 🏭 节点 B：四联图工坊（WumuTriptychAtelier）

吃 5 路 `slotN_image`（联动：直连定妆照工坊造型 N 输出；手动：接 LoadImage，未连线/黑图槽自动跳过）。每造型生成四条参考链：**证件照 → 全身正视 → 侧视（参考正视+原图）→ 后视（仅参考正视）**，合成 1920×1080 白底四联图。

| 参数 | 默认 | 说明 |
|---|---|---|
| `steps` | 4 | Klein 蒸馏标准值，不用动 |
| `seed` | 随机 | 每槽 +100，四链再 +1/+10/+20/+30，可重复 |
| `canvas_w × canvas_h` | 1920×1080 | 资产库横版标准，不要改 |
| `证件照高度 / 全身照高度` | 960 / 820 | 四图自动等比适配，调这两个就行 |
| `另存单视角` | 开 | 证件照/正/侧/背各存一张（引擎引用必需），建议保持开 |

---

## 📝 提示词样板

### 基础定妆照（desc_en，只写"人"）
```
Chinese Han ethnicity woman, 22,             ← 国籍人种放最前防漂移
petite 155cm slight build,                   ← 年龄体型
round face with soft cheeks, black ponytail, ← 脸型发型
flight-attendant cap never quite straight,   ← 标志物（角色记忆点）
true skin texture with visible pores, matte finish,
standing facing the camera, full body visible head to toe,
arms relaxed at sides with hands visible
```
素体/白棚/布光/负面清单节点自动补全，不用写。

### 造型定妆照（slotN_desc，Kontext 句式）
```
Edit the reference photo: the SAME person with identical face, hairstyle,
body shape and skin tone, now wearing [服装总类与设计系统，1-2 句].

[材质与颜色] pearl-white segmented armor panels with pale blue-grey fabric between segments,
[细节层次] glowing cyan piping tracing the collar, cuffs and armor seams,
[标志配饰] a silver-white communication ear pod on her right ear,
[适配体型] tailored for her petite 155cm frame, not bulky
```
**要诀**：① 固定开头锁人句必须保留；② 服装按「总类→材质→颜色→细节→配饰→合身度」分层写，每层一小句；③ 写一个标志配饰做角色记忆点；④ 站姿/背景/布光可省（自动补），负面清单建议保留。

---

## 🎨 风格扩展

编辑 `styles.json` 即可新增风格（节点下拉自动出现）：

```json
{
  "name": "动漫剧场版",
  "lora": "your_anime_lora.safetensors",
  "strength": 0.9,
  "prompt_en": "anime movie style, cel shading, vibrant color",
  "hf_url": "https://huggingface.co/..."
}
```
`lora` 留空则只追加提示词不挂 LoRA。

---

## ❓ 常见问题

- **报错缺少 LoRA 文件**：按报错里的下载地址下载，放进 `models/loras/`
- **造型照脸变了**：确认服装提示词以 `Edit the reference photo: the SAME person...` 开头
- **四联图侧/背面脸跑偏**：只换四联工坊的种子重抽，别动基础照
- **想加速测试**：勾 `use_turbo` 步数改 8；正式出片关掉回 20 步
- **图片糊/构图怪**：别改 832×1216，竖构图全身照最优分辨率
- **造型不满意重抽最快路径**：用「外部基础照重抽造型」（见上）

## 更多文档

- [`人物资产全流程-使用教程.md`](人物资产全流程-使用教程.md) —— 逐步操作、参数速查、常见问题
- 工作流内嵌 📖 手册便签 —— 打开工作流即可见（v1.1 起含提示词样板与图生图开关说明）

## 📋 版本历史

- **v1.1.0**（定妆照工坊内部版本 v4）：新增 ① 基础照图生图（`base_use_reference` + `base_reference_image` + `base_denoise`）；② 外部基础照重抽造型（`slot_use_external_base`，跳过 ① 只重跑 ②，基础照输出口透传）；③ 工作流内嵌手册扩充提示词样板
- **v1.0.0**：定妆照工坊（FLUX.2 Dev）+ 四联图工坊（Klein）+ 联动工作流首发

## License

[MIT](LICENSE)
