# -*- coding: utf-8 -*-
"""节点A：Wumu 定妆照工坊（FLUX.2 Dev）v3
FLUX.2 架构：128ch latent / 单Mistral编码器 / Flux2Scheduler / ReferenceLatent 图生图
支持：Kontext 提示词原生兼容 / Turbo LoRA 8步加速 / boreal 写实风格
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

# ============================ 节点 ============================
class WumuCharacterAtelier:
    @classmethod
    def INPUT_TYPES(cls):
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

        outfit_default = "Edit the reference photo: the SAME person with identical face, hairstyle, body shape and skin tone, now wearing [此处填写服装描述]. Standing facing the camera, full body visible head to toe, arms relaxed at sides with hands visible, seamless pure white studio background, nothing else in frame, soft even frontal studio lighting, photorealistic, sharp focus. no text, no watermark, no logo."

        return {
            "required": {
                "mode": (["①+② 全流程", "仅① 基础定妆照", "仅② 造型定妆照(需接基础图)"],
                         {"default": "①+② 全流程",
                          "tooltip": "①+②=文生图出基础(穿内衣)→参考图生图出各造型\n仅①=只要基础定妆照\n仅②=用已有基础图直接换装"}),
                "char_name": ("STRING", {"default": "角色名",
                    "tooltip": "角色中文名，用于文件命名"}),
                "desc_en": ("STRING", {"default": "Chinese Han ethnicity man, 28 years old, lean build, short black hair slightly messy, tired eyes, a small mole at the end of his left eyebrow", "multiline": True,
                    "tooltip": "人物英文描述（只写身份特征！）\n✅ 国籍人种/年龄体型/脸型发型/肤色/特殊标记\n❌ 任何服装（服装写在造型槽）\n⚠ 国籍写最前面防人种漂移"}),
                "style_name": (style_names, {"default": style_names[0] if style_names else "真实实拍",
                    "tooltip": "风格下拉一选即生效\nFLUX.2 LoRA 生态：写实电影(boreal)✅ / Turbo加速✅\n其他风格暂用提示词版"}),
                "style_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA 强度\n写实类 0.6-0.8\n风格化类 0.85-1.0"}),
                "unet_name": (unets, {"default": unets[0] if unets else "flux2_dev_fp8mixed.safetensors",
                    "tooltip": "FLUX.2 Dev 底模（自动过滤 flux2）"}),
                "clip_name": (clips, {"default": clips[0] if clips else "mistral_3_small_flux2_bf16.safetensors",
                    "tooltip": "FLUX.2 文本编码器（Mistral，单个非双编码器）"}),
                "vae_name": (vaes, {"default": vaes[0] if vaes else "full_encoder_small_decoder.safetensors",
                    "tooltip": "FLUX.2 VAE（128 通道 latent）"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100,
                    "tooltip": "采样步数\n20 = FLUX.2 标准值\n8 = Turbo 模式（勾 use_turbo 后步数手动改 8）"}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "FluxGuidance 值\n4.0 = FLUX.2 dev 甜点值"}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 32,
                    "tooltip": "图片宽度"}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 32,
                    "tooltip": "图片高度（竖构图全身照标准 1216）"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": "随机种子\n固定=可重复\n基础用 seed，造型1用 seed+10…"}),
                "use_turbo": ("BOOLEAN", {"default": False,
                    "tooltip": "勾选=自动挂载 Flux_2-Turbo-LoRA\n步数从 20 降到 8（速度提升 2.5 倍）\n画质略降，测试阶段推荐开启"}),
                "slot1_enable": ("BOOLEAN", {"default": True, "tooltip": "勾选=启用"}),
                "slot1_name": ("STRING", {"default": "常服", "tooltip": "造型中文名(用于文件命名)"}),
                "slot1_desc": ("STRING", {"default": outfit_default, "multiline": True,
                    "tooltip": "FLUX.2 原生支持 Kontext 提示词！\n'Edit the reference photo: the SAME person...now wearing [服装]'\nFLUX.2 能理解参考图锚定，脸不会变"}),
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
            },
            "optional": {
                "base_image": ("IMAGE",
                    {"tooltip": "基础定妆照输入口（仅模式②时必须连接）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("基础定妆照", "造型1定妆照", "造型2定妆照", "造型3定妆照", "造型4定妆照", "造型5定妆照", "info")
    FUNCTION = "run"
    CATEGORY = "wumu"
    DESCRIPTION = "Wumu 定妆照工坊（FLUX.2 Dev）：ReferenceLatent 参考链换装，Kontext 提示词原生支持，脸不变衣服换"

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
                      ref_latent=None, style_en=""):
        full_prompt = f"{prompt} {style_en}" if style_en else prompt
        cond = self._encode_flux2(clip, full_prompt, guidance, ref_latent)
        latent = empty_flux2_latent(w, h)
        sigmas = flux2_sigmas(steps, w, h)
        sampler = comfy.samplers.sampler_object("euler")
        noise = comfy.sample.prepare_noise(latent["samples"], seed)
        samples = comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, cond, cond, latent["samples"], seed=seed
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

    def run(self, mode, char_name, desc_en, style_name, style_strength,
            unet_name, clip_name, vae_name,
            steps, guidance, width, height, seed, use_turbo,
            slot1_enable, slot1_name, slot1_desc,
            slot2_enable, slot2_name, slot2_desc,
            slot3_enable, slot3_name, slot3_desc,
            slot4_enable, slot4_name, slot4_desc,
            slot5_enable, slot5_name, slot5_desc,
            auto_save, save_dir, base_image=None, **kwargs):
        print(f"[Wumu工坊] 🚀 模式={mode} 角色={char_name} 风格={style_name} Turbo={'开' if use_turbo else '关'}")

        model, clip, vae = self._load_models(unet_name, clip_name, vae_name)
        model, clip, style_en = self._apply_style(model, clip, style_name, style_strength)
        if use_turbo:
            model, clip = self._apply_turbo(model, clip)

        # 基础定妆照
        base_img = None
        if mode == "仅② 造型定妆照(需接基础图)":
            if base_image is None:
                raise RuntimeError("[Wumu工坊] ❌ 模式②需要连接基础图输入（base_image）")
            base_img = base_image
        else:
            # desc_en 已含完整模板（粘贴引擎输出）→ 原样使用；只写身份特征 → 套标准模板
            if "underwear" in desc_en.lower():
                base_prompt = desc_en.strip()
            else:
                base_prompt = (
                    f"Studio full-body reference portrait of {desc_en.strip()}, "
                    f"wearing plain simple underwear only, no clothing, no accessories, no jewelry, "
                    f"standing facing the camera, full body visible head to toe, "
                    f"arms relaxed at sides with hands visible, "
                    f"seamless pure white studio background, nothing else in frame, "
                    f"soft even frontal studio lighting, photorealistic, sharp focus. "
                    f"no text, no watermark, no logo."
                )
            print(f"[Wumu工坊] ① 文生图基础定妆照 ({width}×{height}, {steps}步)")
            base_img = self._sample_flux2(model, clip, base_prompt, guidance, width, height,
                                          steps, seed, vae, ref_latent=None, style_en=style_en)
            if auto_save:
                p = os.path.join(save_dir, "资产", "角色", f"{char_name}_基础定妆照.png")
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
            print(f"[Wumu工坊] ② 参考图生图 ×{len(active)}（ReferenceLatent，脸不变）")

        ref_latent = None
        if base_img is not None:
            ref_latent = vae.encode(base_img)

        results = []
        for i, (en, name, desc) in enumerate(slots):
            if not en or not desc.strip():
                results.append(None)
                continue
            slot_seed = seed + (i + 1) * 10
            # FLUX.2 Kontext 提示词直接使用 + 纯身份核心双保险（剥掉模板前缀/内衣尾部）
            identity = self._identity_core(desc_en)
            first_word = identity.split(",")[0].strip().lower()
            if identity and first_word not in desc.lower():
                full_prompt = f"{identity}, {desc}"
            else:
                full_prompt = desc

            img = self._sample_flux2(model, clip, full_prompt, guidance, width, height,
                                    steps, slot_seed, vae, ref_latent=ref_latent, style_en=style_en)
            if auto_save:
                p = os.path.join(save_dir, "资产", "角色", f"{char_name}_{name}_定妆照.png")
                save_img(img, p)
            results.append(img)

        blank = torch.zeros(1, 4, 4, 3)
        out_slots = [(r if r is not None else blank) for r in results]
        while len(out_slots) < 5:
            out_slots.append(blank)

        info = f"角色={char_name} | 风格={style_name} | Turbo={'✓' if use_turbo else '✗'} | 基础✓ | 造型={sum(1 for s in slots if s[0] and s[2].strip())}张"
        return (base_img, out_slots[0], out_slots[1], out_slots[2], out_slots[3], out_slots[4], info)

NODE_CLASS_MAPPINGS_ATelier = {"WumuCharacterAtelier": WumuCharacterAtelier}
NODE_DISPLAY_NAME_MAPPINGS_ATelier = {"WumuCharacterAtelier": "Wumu 定妆照工坊（FLUX.2 Dev）"}
