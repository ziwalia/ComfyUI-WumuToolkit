# -*- coding: utf-8 -*-
"""节点A族：Wumu 定妆照工坊（FLUX.2 Dev）v5 —— ①②分离
FLUX.2 架构：128ch latent / 单Mistral编码器 / Flux2Scheduler / ReferenceLatent 图生图

v5（2026-09-07）①②拆分为两个独立节点，连线组合：
  WumuBaseAtelier   ① 基础定妆照工坊：文生图/图生图出基础定妆照 + 身份核心字符串
  WumuOutfitAtelier ② 服装造型定妆照工坊：必连 base_image，ReferenceLatent 换装出 5 造型
  （①的「身份核心」输出 → ②的 identity 输入，免重复填人物描述）
  WumuCharacterAtelier 旧版合并节点：仅为已存工作流兼容保留
v4：基础照图生图（KSampler 部分加噪）/ 外部基础照重抽 / language 双语
"""
import os, json, glob
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.samplers
from comfy.sd import CLIPType
from comfy_extras.nodes_flux import get_schedule
import folder_paths

from .wumu_lang import L, asset_dir

# ============================ 风格引擎 ============================
STYLE_FILE = os.path.join(os.path.dirname(__file__), "styles.json")

def load_styles():
    try:
        with open(STYLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Wumu工坊] ⚠ styles.json 读取失败: {e}")
        return [{"name": "真实实拍", "lora": None, "strength": 0, "prompt_en": "", "prompt_zh": ""}]

def get_style_names():
    return [s["name"] for s in load_styles()]

def find_style(name):
    for s in load_styles():
        if s["name"] == name:
            return s
    return None

def scan_loras():
    loras = set()
    for d in folder_paths.get_folder_paths("loras"):
        for f in glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True):
            loras.add(os.path.basename(f))
    return loras

# ============================ 工具 ============================
def tensor_to_pil(t):
    arr = (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def save_img(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_pil(tensor).save(path)
    print(f"[Wumu工坊] 💾 {path}")

def empty_flux2_latent(w, h, batch=1):
    return {"samples": torch.zeros([batch, 128, h // 16, w // 16],
                                   device=comfy.model_management.intermediate_device())}

def flux2_sigmas(steps, w, h):
    seq_len = round((w * h) / (16 * 16))
    return get_schedule(steps, seq_len)

# ============================ 公共基类（模型/风格/Turbo/采样/身份提取） ============================
class _WumuFlux2Mixin:
    def _load_models(self, unet_name, clip_name, vae_name):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise RuntimeError(f"[Wumu工坊] ❌ 找不到底模 {unet_name}")
        model = comfy.sd.load_diffusion_model(unet_path)

        clip_path = folder_paths.get_full_path("text_encoders", clip_name)
        if clip_path is None:
            raise RuntimeError(f"[Wumu工坊] ❌ 找不到编码器 {clip_name}")
        clip = comfy.sd.load_clip([clip_path], clip_type=CLIPType.FLUX2)

        vae_path = folder_paths.get_full_path("vae", vae_name)
        if vae_path is None:
            raise RuntimeError(f"[Wumu工坊] ❌ 找不到VAE {vae_name}")
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        return model, clip, vae

    def _apply_style(self, model, clip, style_name, strength):
        style = find_style(style_name)
        if style is None:
            return model, clip, ""
        prompt_en = style.get("prompt_en", "")
        lora_file = style.get("lora")
        if not lora_file:
            return model, clip, prompt_en
        available = scan_loras()
        if lora_file not in available:
            url = style.get("hf_url", "")
            raise RuntimeError(
                f"[Wumu工坊] ⚠ 风格【{style_name}】缺少LoRA文件 {lora_file}\n"
                f"  下载地址：{url}\n  保存到：models/loras/"
            )
        lora_path = None
        for d in folder_paths.get_folder_paths("loras"):
            p = os.path.join(d, lora_file)
            if os.path.exists(p):
                lora_path = p
                break
        if lora_path is None:
            raise RuntimeError(f"[Wumu工坊] ❌ LoRA路径查找失败 {lora_file}")
        lora_sd = comfy.utils.load_torch_file(lora_path)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, strength, strength)
        print(f"[Wumu工坊] 🎨 风格 {style_name} LoRA 已挂载（强度 {strength}）")
        return model, clip, prompt_en

    def _apply_turbo(self, model, clip):
        turbo_file = "Flux_2-Turbo-LoRA_comfyui.safetensors"
        available = scan_loras()
        if turbo_file not in available:
            print(f"[Wumu工坊] ⚠ Turbo LoRA 不存在（{turbo_file}），跳过加速")
            return model, clip
        turbo_path = None
        for d in folder_paths.get_folder_paths("loras"):
            p = os.path.join(d, turbo_file)
            if os.path.exists(p):
                turbo_path = p
                break
        if turbo_path:
            lora_sd = comfy.utils.load_torch_file(turbo_path)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, 1.0, 1.0)
            print(f"[Wumu工坊] ⚡ Turbo LoRA 已挂载（步数可降至 8）")
        return model, clip

    def _encode_flux2(self, clip, text, guidance, ref_latent=None):
        tokens = clip.tokenize(text)
        add_dict = {"guidance": guidance}
        cond = clip.encode_from_tokens_scheduled(tokens, add_dict=add_dict)
        if ref_latent is not None:
            c = []
            for t in cond:
                d = dict(t[1])
                if "reference_latents" in d:
                    d["reference_latents"] = d["reference_latents"] + [ref_latent]
                else:
                    d["reference_latents"] = [ref_latent]
                c.append((t[0], d))
            cond = c
        return cond

    def _sample_flux2(self, model, clip, prompt, guidance, w, h, steps, seed, vae,
                      ref_latent=None, style_en="", init_latent=None, denoise=1.0):
        full_prompt = f"{prompt} {style_en}" if style_en else prompt
        cond = self._encode_flux2(clip, full_prompt, guidance, ref_latent)
        latent_samples = init_latent if init_latent is not None else empty_flux2_latent(w, h)["samples"]
        if init_latent is not None and denoise < 0.9999:
            # 图生图：官方 KSampler 同款公式——用更大步数的调度表取尾段（起点 sigma 低 = 部分加噪）
            new_steps = max(1, int(steps / denoise))
            sigmas = flux2_sigmas(new_steps, w, h)[-(steps + 1):]
        else:
            sigmas = flux2_sigmas(steps, w, h)
        sampler = comfy.samplers.sampler_object("euler")
        noise = comfy.sample.prepare_noise(latent_samples, seed)
        samples = comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, cond, cond, latent_samples, seed=seed
        )
        return vae.decode(samples)

    # ---------- 提示词组装：兼容两种 desc_en 输入 ----------
    # A) 只写身份特征："Chinese Han ethnicity man, 28 years old, lean build..."
    # B) 直接粘贴引擎生成的完整模板（含 Studio 前缀/内衣/背景/no text 尾巴）
    _PREFIXES = ("studio full-body reference portrait of", "studio full body reference portrait of")
    _TAIL_MARK = "wearing plain simple underwear"

    @classmethod
    def _identity_core(cls, desc_en):
        """从 desc_en 提取纯身份核心：剥掉模板前缀与'穿内衣'之后的全部模板尾部"""
        core = desc_en.strip()
        low = core.lower()
        for p in cls._PREFIXES:
            if low.startswith(p):
                core = core[len(p):].lstrip(" ,")
                break
        idx = core.lower().find(cls._TAIL_MARK)
        if idx > 0:
            core = core[:idx].strip().rstrip(",")
        return core.strip()

    @staticmethod
    def _vae_encode_samples(vae, img_tensor):
        """VAE 编码为裸 latent tensor；兼容新旧 ComfyUI（新版 encode 直接返回 tensor，旧版返回 {'samples': ...}）"""
        lat = vae.encode(img_tensor)
        return lat["samples"] if isinstance(lat, dict) else lat

    def _zh(self, language):
        return language == "中文"

    def _base_prompt_from(self, desc_en):
        # desc_en 已含完整模板（粘贴引擎输出）→ 原样使用；只写身份特征 → 套标准模板
        if "underwear" in desc_en.lower():
            return desc_en.strip()
        return (
            f"Studio full-body reference portrait of {desc_en.strip()}, "
            f"wearing plain simple underwear only, no clothing, no accessories, no jewelry, "
            f"standing facing the camera, full body visible head to toe, "
            f"arms relaxed at sides with hands visible, "
            f"seamless pure white studio background, nothing else in frame, "
            f"soft even frontal studio lighting, photorealistic, sharp focus. "
            f"no text, no watermark, no logo."
        )


# ---------- 公共控件块 ----------
def _model_inputs():
    style_names = get_style_names()
    unets = [f for f in folder_paths.get_filename_list("diffusion_models") if "flux2" in f.lower() or "flux_2" in f.lower()]
    if not unets:
        unets = folder_paths.get_filename_list("diffusion_models")
    clips = [f for f in folder_paths.get_filename_list("text_encoders") if "mistral" in f.lower()]
    if not clips:
        clips = folder_paths.get_filename_list("text_encoders")
    vaes = [f for f in folder_paths.get_filename_list("vae") if "full_encoder" in f.lower()]
    if not vaes:
        vaes = folder_paths.get_filename_list("vae")
    return style_names, unets, clips, vaes


# ============================ 节点①：基础照工坊 ============================
class WumuBaseAtelier(_WumuFlux2Mixin):
    @classmethod
    def INPUT_TYPES(cls):
        style_names, unets, clips, vaes = _model_inputs()
        return {
            "required": {
                "char_name": ("STRING", {"default": "角色名",
                    "tooltip": L("角色中文名，用于文件命名", "Character name, used for file naming")}),
                "desc_en": ("STRING", {"default": "Chinese Han ethnicity man, 28 years old, lean build, short black hair slightly messy, tired eyes, a small mole at the end of his left eyebrow", "multiline": True,
                    "tooltip": L("人物英文描述（只写身份特征！）\n✅ 国籍人种/年龄体型/脸型发型/肤色/特殊标记\n❌ 任何服装（服装写在②服装造型定妆照工坊的造型槽）\n⚠ 国籍写最前面防人种漂移",
                                 "Identity-only description (English)\n✅ ethnicity / age & build / face & hair / skin / distinctive marks\n❌ any clothing (clothing goes in ② outfit slots)\n⚠ put nationality first to avoid ethnicity drift")}),
                "style_name": (style_names, {"default": style_names[0] if style_names else "真实实拍",
                    "tooltip": L("风格下拉一选即生效\nFLUX.2 LoRA 生态：写实电影(boreal)✅ / Turbo加速✅\n其他风格暂用提示词版",
                                 "Style dropdown, applies instantly\nFLUX.2 LoRA ecosystem: cinematic (boreal)✅ / Turbo✅\nOther styles are prompt-based for now")}),
                "style_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": L("LoRA 强度\n写实类 0.6-0.8\n风格化类 0.85-1.0",
                                 "LoRA strength\nrealistic styles 0.6-0.8\nstylized 0.85-1.0")}),
                "unet_name": (unets, {"default": unets[0] if unets else "flux2_dev_fp8mixed.safetensors",
                    "tooltip": L("FLUX.2 Dev 底模（自动过滤 flux2）", "FLUX.2 Dev base model (auto-filtered by 'flux2')")}),
                "clip_name": (clips, {"default": clips[0] if clips else "mistral_3_small_flux2_bf16.safetensors",
                    "tooltip": L("FLUX.2 文本编码器（Mistral，单个非双编码器）", "FLUX.2 text encoder (Mistral, single encoder)")}),
                "vae_name": (vaes, {"default": vaes[0] if vaes else "full_encoder_small_decoder.safetensors",
                    "tooltip": L("FLUX.2 VAE（128 通道 latent）", "FLUX.2 VAE (128-channel latent)")}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                    "tooltip": L("采样步数\n20 = FLUX.2 标准值\n8 = Turbo 模式（勾 use_turbo 后步数手动改 8）",
                                 "Sampling steps\n20 = FLUX.2 standard\n8 = Turbo mode (set manually after enabling use_turbo)")}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": L("FluxGuidance 值\n4.0 = FLUX.2 dev 甜点值",
                                 "FluxGuidance value\n4.0 = FLUX.2 dev sweet spot")}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片宽度", "Image width")}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片高度（竖构图全身照标准 1216）", "Image height (1216 = standard vertical full-body)")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": L("随机种子\n固定=可重复",
                                 "Seed\nfixed = reproducible")}),
                "use_turbo": ("BOOLEAN", {"default": False,
                    "tooltip": L("勾选=自动挂载 Flux_2-Turbo-LoRA\n步数从 20 降到 8（速度提升 2.5 倍）\n画质略降，测试阶段推荐开启",
                                 "Check = auto-attach Flux_2-Turbo-LoRA\nsteps 20→8 (2.5× faster)\nslight quality drop — recommended while testing")}),
                "base_use_reference": ("BOOLEAN", {"default": False,
                    "tooltip": L("① 基础照改用图生图\n勾选后连接 base_reference_image 输入口\n用已有照片做底，提示词继续驱动身份与质感\n适合：基于已有定妆照/真人参考照重出基础照",
                                 "① Base portrait via img2img\nCheck, then connect base_reference_image\nUse an existing photo as the canvas; the prompt still drives identity & texture\nGood for: re-rolling the base from an existing portrait / real-person reference")}),
                "base_denoise": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": L("① 图生图重绘幅度（勾 base_use_reference 后生效）\n1.0=几乎重画，只参考构图\n0.7~0.8=保人物改质感细节（推荐）\n0.4~0.6=轻度修整\n0.3 以下基本不动",
                                 "① img2img denoise (active when base_use_reference is on)\n1.0 = almost a full redraw, composition only\n0.7-0.8 = keep the person, redo texture & details (recommended)\n0.4-0.6 = light retouch\nbelow 0.3 = barely changes")}),
                "auto_save": ("BOOLEAN", {"default": True}),
                "save_dir": ("STRING", {"default": r"F:\AI\MINIMAXH3"}),
                "language": (["中文", "English"], {"default": "中文",
                    "tooltip": L("输出语言（本节点）：落盘目录/文件后缀/info/日志\n中文→资产\\角色\\角色_基础定妆照.png；English→assets\\characters\\character_base.png\n参数提示语言为全局设置：切换后刷新页面生效",
                                 "Output language (this node): save folders / file suffixes / info / logs\n中文 → 资产\\角色\\角色_基础定妆照.png; English → assets\\characters\\character_base.png\nTooltip language is global: switch here, then refresh the page")}),
            },
            "optional": {
                "base_reference_image": ("IMAGE",
                    {"tooltip": L("① 图生图参考图输入口（勾 base_use_reference 后必须连接）\n建议喂：已有基础定妆照 / 素体照 / 真人参考照\n会自动缩放到出图尺寸",
                                  "① img2img reference input (required when base_use_reference is on)\nFeed: an existing base portrait / body reference / real-person photo\nAuto-resized to the output resolution")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("基础定妆照", "身份核心", "info")
    FUNCTION = "run"
    CATEGORY = "wumu"
    DESCRIPTION = "Wumu 基础定妆照工坊①（FLUX.2 Dev）：文生图/图生图出白底内衣基础定妆照 + 身份核心字符串（连给②造型工坊免重填描述）/ Base portrait atelier: white-background underwear reference + identity string"

    def run(self, char_name, desc_en, style_name, style_strength,
            unet_name, clip_name, vae_name,
            steps, guidance, width, height, seed, use_turbo,
            base_use_reference, base_denoise, auto_save, save_dir, language="中文",
            base_reference_image=None, **kwargs):
        zh = self._zh(language)
        def msg(zh_s, en_s):
            return zh_s if zh else en_s

        print(msg(f"[Wumu基础定妆照工坊] 🚀 角色={char_name} 风格={style_name} Turbo={'开' if use_turbo else '关'} "
                  f"图生图={'开' if base_use_reference else '关'}",
                  f"[Wumu Base] 🚀 character={char_name} style={style_name} turbo={'on' if use_turbo else 'off'} "
                  f"img2img={'on' if base_use_reference else 'off'}"))

        model, clip, vae = self._load_models(unet_name, clip_name, vae_name)
        model, clip, style_en = self._apply_style(model, clip, style_name, style_strength)
        if use_turbo:
            model, clip = self._apply_turbo(model, clip)

        base_prompt = self._base_prompt_from(desc_en)
        init_latent = None
        denoise = 1.0
        if base_use_reference:
            if base_reference_image is None:
                raise RuntimeError(msg("[Wumu基础定妆照工坊] ❌ 已勾选图生图（base_use_reference），请连接 base_reference_image 输入口",
                                       "[Wumu Base] ❌ base_use_reference is on — connect the base_reference_image input"))
            # 参考图先统一到出图尺寸，保证 latent 形状与调度表一致
            ref_pil = tensor_to_pil(base_reference_image).resize((width, height), Image.LANCZOS)
            init_latent = self._vae_encode_samples(vae, pil_to_tensor(ref_pil))
            denoise = base_denoise
            print(msg(f"[Wumu基础定妆照工坊] 图生图基础定妆照 (denoise={base_denoise}, {width}×{height}, {steps}步)",
                      f"[Wumu Base] img2img (denoise={base_denoise}, {width}×{height}, {steps} steps)"))
        else:
            print(msg(f"[Wumu基础定妆照工坊] 文生图基础定妆照 ({width}×{height}, {steps}步)",
                      f"[Wumu Base] txt2img ({width}×{height}, {steps} steps)"))

        base_img = self._sample_flux2(model, clip, base_prompt, guidance, width, height,
                                      steps, seed, vae, ref_latent=None, style_en=style_en,
                                      init_latent=init_latent, denoise=denoise)
        if auto_save:
            p = os.path.join(asset_dir(save_dir, language), f"{char_name}_{msg('基础定妆照', 'base')}.png")
            save_img(base_img, p)

        identity = self._identity_core(desc_en)
        info = msg(f"角色={char_name} | 风格={style_name} | Turbo={'✓' if use_turbo else '✗'} | "
                   f"基础照={'图生图(denoise=' + str(base_denoise) + ')' if base_use_reference else '文生图'}✓ | 身份核心已提取",
                   f"character={char_name} | style={style_name} | turbo={'Y' if use_turbo else 'N'} | "
                   f"base={'img2img(denoise=' + str(base_denoise) + ')' if base_use_reference else 'txt2img'} done | identity extracted")
        return (base_img, identity, info)


# ============================ 节点②：造型照工坊 ============================
class WumuOutfitAtelier(_WumuFlux2Mixin):
    @classmethod
    def INPUT_TYPES(cls):
        style_names, unets, clips, vaes = _model_inputs()
        outfit_default = "Edit the reference photo: the SAME person with identical face, hairstyle, body shape and skin tone, now wearing [此处填写服装描述]. Standing facing the camera, full body visible head to toe, arms relaxed at sides with hands visible, seamless pure white studio background, nothing else in frame, soft even frontal studio lighting, photorealistic, sharp focus. no text, no watermark, no logo."
        return {
            "required": {
                "base_image": ("IMAGE",
                    {"tooltip": L("基础定妆照输入（必连）：接①基础定妆照工坊的「基础定妆照」输出 或 LoadImage",
                                  "Base portrait input (required): from ① Base Atelier output or a LoadImage node")}),
                "char_name": ("STRING", {"default": "角色名",
                    "tooltip": L("角色中文名（与①保持一致，文件命名用）",
                                 "Character name (keep consistent with ①, used for file naming)")}),
                "desc_en": ("STRING", {"default": "", "multiline": True,
                    "tooltip": L("人物英文描述（备用的身份锚）\n推荐：连①的「身份核心」输出，本框留空即可\n未连 identity 时：本框填写身份描述（自动剥离模板）",
                                 "Identity description (fallback anchor)\nRecommended: wire ①'s 'identity' output and leave this empty\nIf identity is not wired: fill identity here (template auto-stripped)")}),
                "style_name": (style_names, {"default": style_names[0] if style_names else "真实实拍",
                    "tooltip": L("风格（与①保持一致，保证基础照与造型照同风格）",
                                 "Style (keep consistent with ① so base & outfits match)")}),
                "style_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": L("LoRA 强度\n写实类 0.6-0.8\n风格化类 0.85-1.0",
                                 "LoRA strength\nrealistic styles 0.6-0.8\nstylized 0.85-1.0")}),
                "unet_name": (unets, {"default": unets[0] if unets else "flux2_dev_fp8mixed.safetensors",
                    "tooltip": L("FLUX.2 Dev 底模（自动过滤 flux2）", "FLUX.2 Dev base model (auto-filtered by 'flux2')")}),
                "clip_name": (clips, {"default": clips[0] if clips else "mistral_3_small_flux2_bf16.safetensors",
                    "tooltip": L("FLUX.2 文本编码器（Mistral，单个非双编码器）", "FLUX.2 text encoder (Mistral, single encoder)")}),
                "vae_name": (vaes, {"default": vaes[0] if vaes else "full_encoder_small_decoder.safetensors",
                    "tooltip": L("FLUX.2 VAE（128 通道 latent）", "FLUX.2 VAE (128-channel latent)")}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                    "tooltip": L("采样步数\n20 = FLUX.2 标准值\n8 = Turbo 模式（勾 use_turbo 后步数手动改 8）",
                                 "Sampling steps\n20 = FLUX.2 standard\n8 = Turbo mode (set manually after enabling use_turbo)")}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": L("FluxGuidance 值\n4.0 = FLUX.2 dev 甜点值",
                                 "FluxGuidance value\n4.0 = FLUX.2 dev sweet spot")}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片宽度（与①保持一致）", "Image width (keep consistent with ①)")}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片高度（与①保持一致）", "Image height (keep consistent with ①)")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": L("随机种子（独立于①）\n固定=可重复\n造型1用 seed+10，造型2用 seed+20…",
                                 "Seed (independent of ①)\nfixed = reproducible\noutfit1 uses seed+10, outfit2 uses seed+20…")}),
                "use_turbo": ("BOOLEAN", {"default": False,
                    "tooltip": L("勾选=自动挂载 Flux_2-Turbo-LoRA\n步数从 20 降到 8（速度提升 2.5 倍）\n画质略降，测试阶段推荐开启",
                                 "Check = auto-attach Flux_2-Turbo-LoRA\nsteps 20→8 (2.5× faster)\nslight quality drop — recommended while testing")}),
                "slot1_enable": ("BOOLEAN", {"default": True, "tooltip": L("勾选=启用", "Check = enable")}),
                "slot1_name": ("STRING", {"default": "常服", "tooltip": L("造型中文名(用于文件命名)", "Outfit name (used in file naming)")}),
                "slot1_desc": ("STRING", {"default": outfit_default, "multiline": True,
                    "tooltip": L("FLUX.2 原生支持 Kontext 提示词！\n'Edit the reference photo: the SAME person...now wearing [服装]'\nFLUX.2 能理解参考图锚定，脸不会变",
                                 "FLUX.2 natively supports Kontext prompts!\n'Edit the reference photo: the SAME person...now wearing [clothing]'\nFLUX.2 anchors on the reference image — the face stays consistent")}),
                "slot2_enable": ("BOOLEAN", {"default": False}),
                "slot2_name": ("STRING", {"default": "造型2"}),
                "slot2_desc": ("STRING", {"default": "", "multiline": True}),
                "slot3_enable": ("BOOLEAN", {"default": False}),
                "slot3_name": ("STRING", {"default": "造型3"}),
                "slot3_desc": ("STRING", {"default": "", "multiline": True}),
                "slot4_enable": ("BOOLEAN", {"default": False}),
                "slot4_name": ("STRING", {"default": "造型4"}),
                "slot4_desc": ("STRING", {"default": "", "multiline": True}),
                "slot5_enable": ("BOOLEAN", {"default": False}),
                "slot5_name": ("STRING", {"default": "造型5"}),
                "slot5_desc": ("STRING", {"default": "", "multiline": True}),
                "auto_save": ("BOOLEAN", {"default": True}),
                "save_dir": ("STRING", {"default": r"F:\AI\MINIMAXH3"}),
                "language": (["中文", "English"], {"default": "中文",
                    "tooltip": L("输出语言（本节点）：落盘目录/文件后缀/info/日志\n中文→资产\\角色\\角色_造型_定妆照.png；English→assets\\characters\\character_outfit_outfit.png\n参数提示语言为全局设置：切换后刷新页面生效",
                                 "Output language (this node): save folders / file suffixes / info / logs\n中文 → 资产\\角色\\角色_造型_定妆照.png; English → assets\\characters\\character_outfit_outfit.png\nTooltip language is global: switch here, then refresh the page")}),
            },
            "optional": {
                "identity": ("STRING",
                    {"tooltip": L("身份核心输入（推荐）：接①基础定妆照工坊的「身份核心」输出\n连了就不用在 desc_en 里重复填人物描述",
                                  "Identity input (recommended): wire from ① Base Atelier's 'identity' output\nWhen wired, no need to retype the identity in desc_en")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("造型1定妆照", "造型2定妆照", "造型3定妆照", "造型4定妆照", "造型5定妆照", "info")
    FUNCTION = "run"
    CATEGORY = "wumu"
    DESCRIPTION = "Wumu 服装造型定妆照工坊②（FLUX.2 Dev）：基础照 + ReferenceLatent 参考链换装（Kontext 提示词，脸不变衣服换）/ Outfit atelier: outfit swap from the base portrait with consistent face"

    def run(self, base_image, char_name, desc_en, style_name, style_strength,
            unet_name, clip_name, vae_name,
            steps, guidance, width, height, seed, use_turbo,
            slot1_enable, slot1_name, slot1_desc,
            slot2_enable, slot2_name, slot2_desc,
            slot3_enable, slot3_name, slot3_desc,
            slot4_enable, slot4_name, slot4_desc,
            slot5_enable, slot5_name, slot5_desc,
            auto_save, save_dir, language="中文", identity=None, **kwargs):
        if base_image is None:
            raise RuntimeError("[Wumu工坊] ❌ ②造型照工坊需要连接基础照输入（base_image）/ base_image input is required")

        zh = self._zh(language)
        def msg(zh_s, en_s):
            return zh_s if zh else en_s

        slots = [
            (slot1_enable, slot1_name, slot1_desc),
            (slot2_enable, slot2_name, slot2_desc),
            (slot3_enable, slot3_name, slot3_desc),
            (slot4_enable, slot4_name, slot4_desc),
            (slot5_enable, slot5_name, slot5_desc),
        ]
        active = [s for s in slots if s[0] and s[2].strip()]
        print(msg(f"[Wumu服装造型定妆照工坊] 🚀 角色={char_name} 风格={style_name} Turbo={'开' if use_turbo else '关'} "
                  f"换装 ×{len(active)}（ReferenceLatent，脸不变）",
                  f"[Wumu Outfit] 🚀 character={char_name} style={style_name} turbo={'on' if use_turbo else 'off'} "
                  f"swaps ×{len(active)} (ReferenceLatent, face locked)"))

        model, clip, vae = self._load_models(unet_name, clip_name, vae_name)
        model, clip, style_en = self._apply_style(model, clip, style_name, style_strength)
        if use_turbo:
            model, clip = self._apply_turbo(model, clip)

        out_dir = asset_dir(save_dir, language)
        suffix_outfit = msg("定妆照", "outfit")

        ref_latent = vae.encode(base_image)

        # 身份锚：优先用①传来的身份核心；未连则从本节点 desc_en 提取
        identity_core = (identity or "").strip() or self._identity_core(desc_en)

        results = []
        for i, (en, name, desc) in enumerate(slots):
            if not en or not desc.strip():
                results.append(None)
                continue
            slot_seed = seed + (i + 1) * 10
            # FLUX.2 Kontext 提示词直接使用 + 纯身份核心双保险
            first_word = identity_core.split(",")[0].strip().lower() if identity_core else ""
            if identity_core and first_word not in desc.lower():
                full_prompt = f"{identity_core}, {desc}"
            else:
                full_prompt = desc

            img = self._sample_flux2(model, clip, full_prompt, guidance, width, height,
                                    steps, slot_seed, vae, ref_latent=ref_latent, style_en=style_en)
            if auto_save:
                p = os.path.join(out_dir, f"{char_name}_{name}_{suffix_outfit}.png")
                save_img(img, p)
            results.append(img)

        blank = torch.zeros(1, 4, 4, 3)
        out_slots = [(r if r is not None else blank) for r in results]
        while len(out_slots) < 5:
            out_slots.append(blank)

        info = msg(f"角色={char_name} | 风格={style_name} | Turbo={'✓' if use_turbo else '✗'} | "
                   f"基础照=外部输入 | 身份锚={'①传入' if (identity or '').strip() else 'desc_en提取'} | 造型={len(active)}张",
                   f"character={char_name} | style={style_name} | turbo={'Y' if use_turbo else 'N'} | "
                   f"base=external | identity={'from ①' if (identity or '').strip() else 'from desc_en'} | outfits={len(active)}")
        return (out_slots[0], out_slots[1], out_slots[2], out_slots[3], out_slots[4], info)


# ============================ 旧版合并节点（仅为已存工作流兼容保留） ============================
class WumuCharacterAtelier(_WumuFlux2Mixin):
    @classmethod
    def INPUT_TYPES(cls):
        style_names, unets, clips, vaes = _model_inputs()

        outfit_default = "Edit the reference photo: the SAME person with identical face, hairstyle, body shape and skin tone, now wearing [此处填写服装描述]. Standing facing the camera, full body visible head to toe, arms relaxed at sides with hands visible, seamless pure white studio background, nothing else in frame, soft even frontal studio lighting, photorealistic, sharp focus. no text, no watermark, no logo."

        return {
            "required": {
                "mode": (["①+② 全流程", "仅① 基础定妆照", "仅② 造型定妆照(需接基础图)"],
                         {"default": "①+② 全流程",
                          "tooltip": L("旧版合并节点——新流程请用「①基础定妆照工坊 → ②服装造型定妆照工坊」连线组合\n①+②=文生图出基础→参考图生图出各造型\n仅①=只要基础定妆照\n仅②=用已有基础图直接换装",
                                       "Legacy combo node — prefer chaining ① Base Atelier → ② Outfit Atelier\n①+② = txt2img base → reference-based outfit swap\n① only / ② only as labeled")}),
                "char_name": ("STRING", {"default": "角色名",
                    "tooltip": L("角色中文名，用于文件命名", "Character name, used for file naming")}),
                "desc_en": ("STRING", {"default": "Chinese Han ethnicity man, 28 years old, lean build, short black hair slightly messy, tired eyes, a small mole at the end of his left eyebrow", "multiline": True,
                    "tooltip": L("人物英文描述（只写身份特征！）\n✅ 国籍人种/年龄体型/脸型发型/肤色/特殊标记\n❌ 任何服装（服装写在造型槽）\n⚠ 国籍写最前面防人种漂移",
                                 "Identity-only description (English)\n✅ ethnicity / age & build / face & hair / skin / distinctive marks\n❌ any clothing (clothing goes in outfit slots)\n⚠ put nationality first to avoid ethnicity drift")}),
                "style_name": (style_names, {"default": style_names[0] if style_names else "真实实拍",
                    "tooltip": L("风格下拉一选即生效\nFLUX.2 LoRA 生态：写实电影(boreal)✅ / Turbo加速✅\n其他风格暂用提示词版",
                                 "Style dropdown, applies instantly\nFLUX.2 LoRA ecosystem: cinematic (boreal)✅ / Turbo✅\nOther styles are prompt-based for now")}),
                "style_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": L("LoRA 强度\n写实类 0.6-0.8\n风格化类 0.85-1.0",
                                 "LoRA strength\nrealistic styles 0.6-0.8\nstylized 0.85-1.0")}),
                "unet_name": (unets, {"default": unets[0] if unets else "flux2_dev_fp8mixed.safetensors",
                    "tooltip": L("FLUX.2 Dev 底模（自动过滤 flux2）", "FLUX.2 Dev base model (auto-filtered by 'flux2')")}),
                "clip_name": (clips, {"default": clips[0] if clips else "mistral_3_small_flux2_bf16.safetensors",
                    "tooltip": L("FLUX.2 文本编码器（Mistral，单个非双编码器）", "FLUX.2 text encoder (Mistral, single encoder)")}),
                "vae_name": (vaes, {"default": vaes[0] if vaes else "full_encoder_small_decoder.safetensors",
                    "tooltip": L("FLUX.2 VAE（128 通道 latent）", "FLUX.2 VAE (128-channel latent)")}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                    "tooltip": L("采样步数\n20 = FLUX.2 标准值\n8 = Turbo 模式（勾 use_turbo 后步数手动改 8）",
                                 "Sampling steps\n20 = FLUX.2 standard\n8 = Turbo mode (set manually after enabling use_turbo)")}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": L("FluxGuidance 值\n4.0 = FLUX.2 dev 甜点值",
                                 "FluxGuidance value\n4.0 = FLUX.2 dev sweet spot")}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片宽度", "Image width")}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 32,
                    "tooltip": L("图片高度（竖构图全身照标准 1216）", "Image height (1216 = standard vertical full-body)")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": L("随机种子\n固定=可重复\n基础用 seed，造型1用 seed+10…",
                                 "Seed\nfixed = reproducible\nbase uses seed, outfit1 uses seed+10…")}),
                "use_turbo": ("BOOLEAN", {"default": False,
                    "tooltip": L("勾选=自动挂载 Flux_2-Turbo-LoRA\n步数从 20 降到 8（速度提升 2.5 倍）\n画质略降，测试阶段推荐开启",
                                 "Check = auto-attach Flux_2-Turbo-LoRA\nsteps 20→8 (2.5× faster)\nslight quality drop — recommended while testing")}),
                "base_use_reference": ("BOOLEAN", {"default": False,
                    "tooltip": L("① 基础照改用图生图\n勾选后连接 base_reference_image 输入口\n用已有照片做底，提示词继续驱动身份与质感\n适合：基于已有定妆照/真人参考照重出基础照",
                                 "① Base portrait via img2img\nCheck, then connect base_reference_image\nUse an existing photo as the canvas; the prompt still drives identity & texture\nGood for: re-rolling the base from an existing portrait / real-person reference")}),
                "base_denoise": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": L("① 图生图重绘幅度（勾 base_use_reference 后生效）\n1.0=几乎重画，只参考构图\n0.7~0.8=保人物改质感细节（推荐）\n0.4~0.6=轻度修整\n0.3 以下基本不动",
                                 "① img2img denoise (active when base_use_reference is on)\n1.0 = almost a full redraw, composition only\n0.7-0.8 = keep the person, redo texture & details (recommended)\n0.4-0.6 = light retouch\nbelow 0.3 = barely changes")}),
                "slot_use_external_base": ("BOOLEAN", {"default": False,
                    "tooltip": L("② 改用外部基础照（重抽造型专用）\n勾选后连接 base_image 输入口 → 跳过①，直接用外部基础照换装\n（新流程直接用②服装造型定妆照工坊即可，无需此开关）",
                                 "② Use an external base portrait (outfit re-roll mode)\nCheck, then connect base_image → skip ① and swap outfits directly\n(the new ② Outfit Atelier does this natively — no switch needed)")}),
                "slot1_enable": ("BOOLEAN", {"default": True, "tooltip": L("勾选=启用", "Check = enable")}),
                "slot1_name": ("STRING", {"default": "常服", "tooltip": L("造型中文名(用于文件命名)", "Outfit name (used in file naming)")}),
                "slot1_desc": ("STRING", {"default": outfit_default, "multiline": True,
                    "tooltip": L("FLUX.2 原生支持 Kontext 提示词！\n'Edit the reference photo: the SAME person...now wearing [服装]'\nFLUX.2 能理解参考图锚定，脸不会变",
                                 "FLUX.2 natively supports Kontext prompts!\n'Edit the reference photo: the SAME person...now wearing [clothing]'\nFLUX.2 anchors on the reference image — the face stays consistent")}),
                "slot2_enable": ("BOOLEAN", {"default": False}),
                "slot2_name": ("STRING", {"default": "造型2"}),
                "slot2_desc": ("STRING", {"default": "", "multiline": True}),
                "slot3_enable": ("BOOLEAN", {"default": False}),
                "slot3_name": ("STRING", {"default": "造型3"}),
                "slot3_desc": ("STRING", {"default": "", "multiline": True}),
                "slot4_enable": ("BOOLEAN", {"default": False}),
                "slot4_name": ("STRING", {"default": "造型4"}),
                "slot4_desc": ("STRING", {"default": "", "multiline": True}),
                "slot5_enable": ("BOOLEAN", {"default": False}),
                "slot5_name": ("STRING", {"default": "造型5"}),
                "slot5_desc": ("STRING", {"default": "", "multiline": True}),
                "auto_save": ("BOOLEAN", {"default": True}),
                "save_dir": ("STRING", {"default": r"F:\AI\MINIMAXH3"}),
                "language": (["中文", "English"], {"default": "中文",
                    "tooltip": L("输出语言（本节点）：落盘目录/文件后缀/info/日志\n中文→资产\\角色\\角色_造型_定妆照.png；English→assets\\characters\\character_outfit_outfit.png\n参数提示语言为全局设置：切换后刷新页面生效",
                                 "Output language (this node): save folders / file suffixes / info / logs\n中文 → 资产\\角色\\角色_造型_定妆照.png; English → assets\\characters\\character_outfit_outfit.png\nTooltip language is global: switch here, then refresh the page")}),
            },
            "optional": {
                "base_image": ("IMAGE",
                    {"tooltip": L("基础定妆照输入口\n· 模式【仅②】时必须连接\n· 勾选 slot_use_external_base 后=外部基础照（跳过①只重抽造型）",
                                  "Base portrait input\n· required in mode ② only\n· with slot_use_external_base on = external base (skip ①, re-roll outfits only)")}),
                "base_reference_image": ("IMAGE",
                    {"tooltip": L("① 图生图参考图输入口（勾 base_use_reference 后必须连接）\n建议喂：已有基础定妆照 / 素体照 / 真人参考照\n会自动缩放到出图尺寸",
                                  "① img2img reference input (required when base_use_reference is on)\nFeed: an existing base portrait / body reference / real-person photo\nAuto-resized to the output resolution")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("基础定妆照", "造型1定妆照", "造型2定妆照", "造型3定妆照", "造型4定妆照", "造型5定妆照", "info")
    FUNCTION = "run"
    CATEGORY = "wumu"
    DESCRIPTION = "Wumu 定妆照工坊（FLUX.2 Dev·旧版合并节点）：新流程请用 ①基础定妆照工坊 → ②服装造型定妆照工坊 / Legacy combo node — prefer the split ① → ② nodes"

    def run(self, mode, char_name, desc_en, style_name, style_strength,
            unet_name, clip_name, vae_name,
            steps, guidance, width, height, seed, use_turbo,
            base_use_reference, base_denoise, slot_use_external_base,
            slot1_enable, slot1_name, slot1_desc,
            slot2_enable, slot2_name, slot2_desc,
            slot3_enable, slot3_name, slot3_desc,
            slot4_enable, slot4_name, slot4_desc,
            slot5_enable, slot5_name, slot5_desc,
            auto_save, save_dir, language="中文",
            base_image=None, base_reference_image=None, **kwargs):
        zh = self._zh(language)
        def msg(zh_s, en_s):
            return zh_s if zh else en_s

        print("[Wumu工坊] 🚀 " + msg(
            f"模式={mode} 角色={char_name} 风格={style_name} Turbo={'开' if use_turbo else '关'} "
            f"①图生图={'开' if base_use_reference else '关'} ②外部基础照={'开' if slot_use_external_base else '关'}",
            f"mode={mode} character={char_name} style={style_name} turbo={'on' if use_turbo else 'off'} "
            f"base-img2img={'on' if base_use_reference else 'off'} external-base={'on' if slot_use_external_base else 'off'}"))

        model, clip, vae = self._load_models(unet_name, clip_name, vae_name)
        model, clip, style_en = self._apply_style(model, clip, style_name, style_strength)
        if use_turbo:
            model, clip = self._apply_turbo(model, clip)

        out_dir = asset_dir(save_dir, language)
        suffix_base = msg("基础定妆照", "base")
        suffix_outfit = msg("定妆照", "outfit")

        # 基础定妆照
        base_img = None
        if mode == "仅② 造型定妆照(需接基础图)":
            if base_image is None:
                raise RuntimeError("[Wumu工坊] ❌ 模式②需要连接基础图输入（base_image）/ mode ② requires base_image input")
            base_img = base_image
        elif slot_use_external_base and base_image is not None:
            # 重抽造型模式：跳过①，直接用外部基础照换装（基础照输出口透传，下游四联不受影响）
            print(msg("[Wumu工坊] 🔁 ② 外部基础照模式：跳过①，直接重抽造型",
                      "[Wumu工坊] 🔁 ② external base mode: skip ①, re-roll outfits directly"))
            base_img = base_image
        else:
            base_prompt = self._base_prompt_from(desc_en)
            init_latent = None
            denoise = 1.0
            if base_use_reference:
                if base_reference_image is None:
                    raise RuntimeError(msg("[Wumu工坊] ❌ 已勾选①图生图（base_use_reference），请连接 base_reference_image 输入口",
                                           "[Wumu工坊] ❌ base_use_reference is on — connect the base_reference_image input"))
                # 参考图先统一到出图尺寸，保证 latent 形状与调度表一致
                ref_pil = tensor_to_pil(base_reference_image).resize((width, height), Image.LANCZOS)
                init_latent = self._vae_encode_samples(vae, pil_to_tensor(ref_pil))
                denoise = base_denoise
                print(msg(f"[Wumu工坊] ① 图生图基础定妆照 (denoise={base_denoise}, {width}×{height}, {steps}步)",
                          f"[Wumu工坊] ① base portrait img2img (denoise={base_denoise}, {width}×{height}, {steps} steps)"))
            else:
                print(msg(f"[Wumu工坊] ① 文生图基础定妆照 ({width}×{height}, {steps}步)",
                          f"[Wumu工坊] ① txt2img base portrait ({width}×{height}, {steps} steps)"))
            base_img = self._sample_flux2(model, clip, base_prompt, guidance, width, height,
                                          steps, seed, vae, ref_latent=None, style_en=style_en,
                                          init_latent=init_latent, denoise=denoise)
            if auto_save:
                p = os.path.join(out_dir, f"{char_name}_{suffix_base}.png")
                save_img(base_img, p)

        # 造型定妆照（FLUX.2 ReferenceLatent 参考链）
        slots = [
            (slot1_enable, slot1_name, slot1_desc),
            (slot2_enable, slot2_name, slot2_desc),
            (slot3_enable, slot3_name, slot3_desc),
            (slot4_enable, slot4_name, slot4_desc),
            (slot5_enable, slot5_name, slot5_desc),
        ]
        if mode == "仅① 基础定妆照":
            slots = []
        else:
            active = [s for s in slots if s[0] and s[2].strip()]
            print(msg(f"[Wumu工坊] ② 参考图生图 ×{len(active)}（ReferenceLatent，脸不变）",
                      f"[Wumu工坊] ② reference img2img ×{len(active)} (ReferenceLatent, face locked)"))

        ref_latent = None
        if base_img is not None:
            ref_latent = vae.encode(base_img)

        identity = self._identity_core(desc_en)
        results = []
        for i, (en, name, desc) in enumerate(slots):
            if not en or not desc.strip():
                results.append(None)
                continue
            slot_seed = seed + (i + 1) * 10
            first_word = identity.split(",")[0].strip().lower() if identity else ""
            if identity and first_word not in desc.lower():
                full_prompt = f"{identity}, {desc}"
            else:
                full_prompt = desc

            img = self._sample_flux2(model, clip, full_prompt, guidance, width, height,
                                    steps, slot_seed, vae, ref_latent=ref_latent, style_en=style_en)
            if auto_save:
                p = os.path.join(out_dir, f"{char_name}_{name}_{suffix_outfit}.png")
                save_img(img, p)
            results.append(img)

        blank = torch.zeros(1, 4, 4, 3)
        out_slots = [(r if r is not None else blank) for r in results]
        while len(out_slots) < 5:
            out_slots.append(blank)

        n_outfits = sum(1 for s in slots if s[0] and s[2].strip())
        info = msg(
            f"角色={char_name} | 风格={style_name} | Turbo={'✓' if use_turbo else '✗'} | "
            f"①{'图生图(denoise=' + str(base_denoise) + ')' if base_use_reference else '文生图'}"
            f"{'→外部基础照' if (slot_use_external_base and base_image is not None) else ''} | "
            f"造型={n_outfits}张",
            f"character={char_name} | style={style_name} | turbo={'Y' if use_turbo else 'N'} | "
            f"base={'img2img(denoise=' + str(base_denoise) + ')' if base_use_reference else 'txt2img'}"
            f"{'->external' if (slot_use_external_base and base_image is not None) else ''} | "
            f"outfits={n_outfits}")
        return (base_img, out_slots[0], out_slots[1], out_slots[2], out_slots[3], out_slots[4], info)

NODE_CLASS_MAPPINGS_ATelier = {
    "WumuBaseAtelier": WumuBaseAtelier,
    "WumuOutfitAtelier": WumuOutfitAtelier,
    "WumuCharacterAtelier": WumuCharacterAtelier,
}
NODE_DISPLAY_NAME_MAPPINGS_ATelier = {
    "WumuBaseAtelier": "Wumu 基础定妆照工坊①（FLUX.2 Dev）",
    "WumuOutfitAtelier": "Wumu 服装造型定妆照工坊②（FLUX.2 Dev）",
    "WumuCharacterAtelier": "Wumu 定妆照工坊（FLUX.2 Dev·旧版合并）",
}
