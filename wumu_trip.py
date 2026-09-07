# -*- coding: utf-8 -*-
"""节点B：Wumu 四联图工坊（Klein 底模）v4
输入：造型定妆照（定妆照工坊输出 或 LoadImage 手动喂图）
每槽四链（完整复刻 人物一键三视图-Klein 工作流的串行参考结构）：
  ①证件照 1024×1024（参考原图）→ 合成图大图位
  ②全身正视 1024×1920（参考原图）→ 后续链的身份锚 + 合成第2格
  ③侧视 1024×1920（参考正视输出latent + 原图latent）
  ④后视 1024×1920（参考正视输出latent）
输出：1920×1080 合成四联图（证件照大 + 全身正视 + 侧视 + 后视），四链均锁定纯白摄影棚背景
落盘命名对齐 wumu-movie 引擎契约：角色_造型_证件照/正面图/侧面图/背面图/四联图.png
  （English 模式：character_outfit_id/front/side/back/quadtych.png）
v4（2026-09-07）：
  - 新增 language 下拉（默认中文）：落盘目录/文件后缀/四视角生成提示词/info 双语
v3（2026-08-30）：
  - 三联图 → 四联图：正面图加入合成（证件照与侧视之间）
  - 合成超宽缩放公式修正：缝隙/边距不参与缩放，四图精确铺满画布（v2 公式四图会横向溢出约18px）
  - 正/侧/后提示词补"纯白色摄影棚背景"（v2 复刻原工作流时没写背景，导致底色发灰）
v2 修复（v1 三处必崩 bug）：
  - latent 16ch → 128ch（Klein 是 FLUX.2 家族）
  - encode_from_tokens → encode_from_tokens_scheduled（conditioning 格式）
  - prepare_noise / sample_custom 必须传裸 tensor（latent["samples"] 同族坑）
"""
import os
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.samplers
from comfy.sd import CLIPType
import folder_paths
from comfy_extras.nodes_flux import get_schedule

from .wumu_lang import L, asset_dir

# ============================ 工具 ============================
def tensor_to_pil(t):
    arr = (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def scale_longest(img, length):
    w, h = img.size
    if max(w, h) <= length:
        return img
    if w >= h:
        nw, nh = length, round(h * length / w)
    else:
        nw, nh = round(w * length / h), length
    return img.resize((nw, nh), Image.LANCZOS)

def compose_quadtych(id_photo, front, side, back, cw=1920, ch=1080,
                     fh=960, vh=820, gap=28, margin=36, bg="#FFFFFF"):
    """证件照（大）+ 全身正视 + 侧视 + 后视 → 横向排布白底四联图
    超宽时按高度等比缩小（缝隙/边距固定不缩），保证总宽精确不超过画布"""
    def scaled(img, h):
        w = max(1, round(img.width * h / img.height))
        return img.resize((w, h), Image.LANCZOS)

    def panels_width(f, v):
        rs = [im.width / im.height for im in (id_photo, front, side, back)]
        return round(f * rs[0] + v * rs[1] + v * rs[2] + v * rs[3])

    # k = 可用宽度 / 面板总宽（缝隙×3 + 边距×2 从画布宽里扣除，不参与缩放）
    avail = cw - gap * 3 - margin * 2
    w0 = panels_width(fh, vh)
    if w0 > avail:
        k = avail / w0
        fh, vh = max(64, round(fh * k)), max(64, round(vh * k))

    panels = [scaled(id_photo, fh), scaled(front, vh), scaled(side, vh), scaled(back, vh)]
    total = sum(p.width for p in panels) + gap * 3
    x0 = max(margin, (cw - total) // 2)
    canvas = Image.new("RGB", (cw, ch), bg)
    x = x0
    for img in panels:
        y = (ch - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas

def save_img(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_pil(tensor).save(path)
    print(f"[Wumu四联工坊] 💾 {path}")

# ============================ 双语提示词 / 文件后缀 ============================
BASE_PHRASE = {"中文": "生成图中角色", "English": "the character in the reference image"}

VIEW_PROMPTS = {
    "中文": {
        "id": "{base}的证件照式正面半身照（头肩特写、五官清晰、表情自然、纯白背景），保持人物一致性。",
        "front": "{base}全身的正视图，纯白色摄影棚背景，保持人物一致性。",
        "side": "{base}的侧视图，纯白色摄影棚背景。",
        "back": "{base}的后视图，纯白色摄影棚背景，保持人物一致性。",
    },
    "English": {
        "id": "An ID-photo-style front-facing half-body portrait of {base} (head-and-shoulders close-up, clear facial features, natural expression, pure white background), keeping the character consistent.",
        "front": "A full-body front view of {base}, pure white photography-studio background, keeping the character consistent.",
        "side": "A side view of {base}, pure white photography-studio background.",
        "back": "A back view of {base}, pure white photography-studio background, keeping the character consistent.",
    },
}

# 落盘后缀：中文 = wumu-movie 引擎契约；English = 通用英文命名
SUFFIX_VIEWS = {
    "中文": {"id": "证件照", "front": "正面图", "side": "侧面图", "back": "背面图", "quad": "四联图"},
    "English": {"id": "id", "front": "front", "side": "side", "back": "back", "quad": "quadtych"},
}

# ============================ 节点 ============================
class WumuTriptychAtelier:
    @classmethod
    def INPUT_TYPES(cls):
        unets = [f for f in folder_paths.get_filename_list("diffusion_models") if "klein" in f.lower()]
        if not unets:
            unets = folder_paths.get_filename_list("diffusion_models")
        clips = [f for f in folder_paths.get_filename_list("text_encoders") if "qwen" in f.lower()]
        if not clips:
            clips = folder_paths.get_filename_list("text_encoders")
        vaes = [f for f in folder_paths.get_filename_list("vae") if "full_encoder" in f.lower()]
        if not vaes:
            vaes = folder_paths.get_filename_list("vae")

        return {
            "required": {
                "char_name": ("STRING", {"default": "角色名",
                    "tooltip": L("角色中文名（文件命名用，与定妆照工坊保持一致）",
                                 "Character name (file naming, keep consistent with the Character Atelier)")}),
                "unet_name": (unets, {"default": unets[0] if unets else "flux-2-klein-9b-kv-fp8.safetensors",
                    "tooltip": L("Klein 底模（9B，快速，自动过滤 klein）", "Klein base model (9B, fast, auto-filtered by 'klein')")}),
                "clip_name": (clips, {"default": clips[0] if clips else "qwen_3_8b_fp8mixed.safetensors",
                    "tooltip": L("Qwen3 编码器（中文提示词原生）", "Qwen3 encoder (native Chinese & English prompts)")}),
                "vae_name": (vaes, {"default": vaes[0] if vaes else "full_encoder_small_decoder.safetensors",
                    "tooltip": L("FLUX.2 VAE（128 通道）", "FLUX.2 VAE (128-channel latent)")}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50,
                    "tooltip": L("采样步数，4 = Klein 标准值", "Sampling steps, 4 = Klein standard")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1,
                    "tooltip": L("种子：每槽 +100，每条链再 +1/+10/+20/+30",
                                 "Seed: each slot +100, each chain +1/+10/+20/+30")}),
                "canvas_w": ("INT", {"default": 1920, "min": 512, "max": 4096,
                    "tooltip": L("四联图画布宽", "Quadtych canvas width")}),
                "canvas_h": ("INT", {"default": 1080, "min": 512, "max": 4096,
                    "tooltip": L("四联图画布高", "Quadtych canvas height")}),
                "front_h": ("INT", {"default": 960, "min": 256, "max": 4096,
                    "tooltip": L("证件照目标高度（大图）\n四图自动适配后实际约 745px",
                                 "ID photo target height (large panel)\nactual ≈745px after auto-fit")}),
                "side_h": ("INT", {"default": 820, "min": 256, "max": 4096,
                    "tooltip": L("正面/侧面/背面三张全身照目标高度\n四图自动适配后实际约 637px",
                                 "Target height for the three full-body views\nactual ≈637px after auto-fit")}),
                "slot1_name": ("STRING", {"default": "睡衣",
                    "tooltip": L("造型1中文名（须与定妆照工坊同名）", "Outfit1 name (must match the Character Atelier)")}),
                "slot2_name": ("STRING", {"default": "造型2"}),
                "slot3_name": ("STRING", {"default": "造型3"}),
                "slot4_name": ("STRING", {"default": "造型4"}),
                "slot5_name": ("STRING", {"default": "造型5"}),
                "auto_save": ("BOOLEAN", {"default": True,
                    "tooltip": L("自动保存到 资产\\角色\\", "Auto-save to the assets folder")}),
                "save_views": ("BOOLEAN", {"default": True,
                    "tooltip": L("另存单视角图（证件照/正面图/侧面图/背面图）\n导演台引用和引擎 refs 需要这些文件，建议开",
                                 "Also save single views (id/front/side/back)\nThe director console & engine refs need these files — keep enabled")}),
                "save_dir": ("STRING", {"default": r"F:\AI\MINIMAXH3"}),
                "language": (["中文", "English"], {"default": "中文",
                    "tooltip": L("输出语言（本节点）：落盘目录/文件后缀/四视角提示词/info\n中文→资产\\角色\\角色_造型_正面图.png（wumu-movie 引擎契约）；English→assets\\characters\\character_outfit_front.png\n参数提示语言为全局设置：切换后刷新页面生效",
                                 "Output language (this node): save folders / file suffixes / four-view prompts / info\n中文 → 资产\\角色\\角色_造型_正面图.png (wumu-movie engine contract); English → assets\\characters\\character_outfit_front.png\nTooltip language is global: switch here, then refresh the page")}),
            },
            "optional": {
                "slot1_image": ("IMAGE", {"tooltip": L("造型1定妆照：接定妆照工坊输出 或 LoadImage",
                                                       "Outfit1 portrait: from the Character Atelier or a LoadImage node")}),
                "slot2_image": ("IMAGE", {"tooltip": L("造型2定妆照", "Outfit2 portrait")}),
                "slot3_image": ("IMAGE", {"tooltip": L("造型3定妆照", "Outfit3 portrait")}),
                "slot4_image": ("IMAGE", {"tooltip": L("造型4定妆照", "Outfit4 portrait")}),
                "slot5_image": ("IMAGE", {"tooltip": L("造型5定妆照", "Outfit5 portrait")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("四联图1", "四联图2", "四联图3", "四联图4", "四联图5", "info")
    FUNCTION = "run"
    CATEGORY = "wumu"
    DESCRIPTION = "Wumu 四联图工坊（Klein）：造型定妆照 → 证件照+正/侧/后四链（纯白背景）→ 1920×1080 合成四联图 / Quadtych atelier: outfit portrait → id/front/side/back chains → composite"

    # ---------- 模型 ----------
    def _load_models(self, unet_name, clip_name, vae_name):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        clip_path = folder_paths.get_full_path("text_encoders", clip_name)
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not all([unet_path, clip_path, vae_path]):
            raise RuntimeError(f"[Wumu四联工坊] ❌ 模型路径查找失败: {unet_name}/{clip_name}/{vae_name}")
        model = comfy.sd.load_diffusion_model(unet_path)
        clip = comfy.sd.load_clip([clip_path], clip_type=CLIPType.FLUX2)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        return model, clip, vae

    # ---------- 编码/采样（FLUX.2 家族正确姿势） ----------
    def _encode(self, clip, text, refs=None):
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if refs:
            c = []
            for t in cond:
                d = dict(t[1])
                d["reference_latents"] = list(d.get("reference_latents", [])) + list(refs)
                c.append((t[0], d))
            cond = c
        return cond

    def _sample(self, model, cond, w, h, steps, seed):
        latent = torch.zeros([1, 128, h // 16, w // 16],
                             device=comfy.model_management.intermediate_device())
        sigmas = get_schedule(steps, round((w * h) / (16 * 16)))
        sampler = comfy.samplers.sampler_object("euler")
        noise = comfy.sample.prepare_noise(latent, seed)
        return comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, cond, cond, latent, seed=seed
        )

    # ---------- 每槽四链生成 ----------
    def _generate_slot(self, model, clip, vae, img, seed, steps,
                       canvas_w, canvas_h, front_h, side_h, language):
        # 原图长边缩到1920 → 参考latent（与 Klein 工作流一致）
        pil = scale_longest(tensor_to_pil(img).convert("RGB"), 1920)
        ref_latent = vae.encode(pil_to_tensor(pil))

        vp = VIEW_PROMPTS.get(language, VIEW_PROMPTS["中文"])
        base = BASE_PHRASE.get(language, BASE_PHRASE["中文"])
        p_id = vp["id"].format(base=base)
        p_front = vp["front"].format(base=base)
        p_side = vp["side"].format(base=base)
        p_back = vp["back"].format(base=base)

        zh = language == "中文"
        t_id = "证件照 / ID photo" if zh else "ID photo"
        t_front = "全身正视图 / full-body front" if zh else "full-body front view"
        t_side = "侧视图（参考正视输出+原图）/ side (refs: front output + source)" if zh else "side view (refs: front output + source)"
        t_back = "后视图（参考正视输出）/ back (ref: front output)" if zh else "back view (ref: front output)"

        print(f"[Wumu四联工坊] 📸 {t_id} (1024×1024)")
        id_img = tensor_to_pil(vae.decode(
            self._sample(model, self._encode(clip, p_id, [ref_latent]), 1024, 1024, steps, seed + 1)))

        print(f"[Wumu四联工坊] 🧍 {t_front} (1024×1920)")
        samples_front = self._sample(model, self._encode(clip, p_front, [ref_latent]), 1024, 1920, steps, seed + 10)
        front_img = tensor_to_pil(vae.decode(samples_front))

        print(f"[Wumu四联工坊] 🔄 {t_side}")
        side_img = tensor_to_pil(vae.decode(
            self._sample(model, self._encode(clip, p_side, [samples_front, ref_latent]), 1024, 1920, steps, seed + 20)))

        print(f"[Wumu四联工坊] 🔙 {t_back}")
        back_img = tensor_to_pil(vae.decode(
            self._sample(model, self._encode(clip, p_back, [samples_front]), 1024, 1920, steps, seed + 30)))

        tri = compose_quadtych(id_img, front_img, side_img, back_img,
                               canvas_w, canvas_h, front_h, side_h)
        return id_img, front_img, side_img, back_img, tri

    # ---------- 主入口 ----------
    def run(self, char_name, unet_name, clip_name, vae_name, steps, seed,
            canvas_w, canvas_h, front_h, side_h,
            slot1_name, slot2_name, slot3_name, slot4_name, slot5_name,
            auto_save, save_views, save_dir, language="中文",
            slot1_image=None, slot2_image=None, slot3_image=None, slot4_image=None, slot5_image=None, **kwargs):
        zh = language == "中文"
        def msg(zh_s, en_s):
            return zh_s if zh else en_s

        slots = [
            (slot1_image, slot1_name), (slot2_image, slot2_name),
            (slot3_image, slot3_name), (slot4_image, slot4_name),
            (slot5_image, slot5_name),
        ]
        # 跳过空槽 + 跳过定妆照工坊输出的占位黑图（<64px）
        def valid(im):
            return im is not None and im.shape[1] >= 64 and im.shape[2] >= 64
        active = [(im, nm) for im, nm in slots if valid(im)]
        if not active:
            raise RuntimeError(msg(
                "[Wumu四联工坊] ❌ 没有可用输入。\n"
                "联动模式：把定妆照工坊的『造型N定妆照』连到 slotN_image（只连启用了的槽）。\n"
                "手动模式：加一个 LoadImage 节点连到任意 slot。",
                "[Wumu Quadtych] ❌ No usable input.\n"
                "Linked mode: connect the Character Atelier 'outfit N' outputs to slotN_image (enabled slots only).\n"
                "Manual mode: add a LoadImage node and connect it to any slot."))

        print(msg(f"[Wumu四联工坊] 🚀 角色={char_name} 槽数={len(active)} 模型={unet_name} 步数={steps}",
                  f"[Wumu Quadtych] 🚀 character={char_name} slots={len(active)} model={unet_name} steps={steps}"))
        model, clip, vae = self._load_models(unet_name, clip_name, vae_name)

        sf = SUFFIX_VIEWS.get(language, SUFFIX_VIEWS["中文"])
        results = []
        for i, (im, nm) in enumerate(slots):
            if not valid(im):
                results.append(None)
                continue
            print(msg(f"[Wumu四联工坊] --- 槽{i + 1}: {nm} ---", f"[Wumu Quadtych] --- slot {i + 1}: {nm} ---"))
            slot_seed = seed + i * 100
            id_img, front_img, side_img, back_img, tri = self._generate_slot(
                model, clip, vae, im, slot_seed, steps, canvas_w, canvas_h, front_h, side_h, language)
            if auto_save:
                d = asset_dir(save_dir, language)
                os.makedirs(d, exist_ok=True)
                if save_views:
                    id_img.save(os.path.join(d, f"{char_name}_{nm}_{sf['id']}.png"))
                    front_img.save(os.path.join(d, f"{char_name}_{nm}_{sf['front']}.png"))
                    side_img.save(os.path.join(d, f"{char_name}_{nm}_{sf['side']}.png"))
                    back_img.save(os.path.join(d, f"{char_name}_{nm}_{sf['back']}.png"))
                tri.save(os.path.join(d, f"{char_name}_{nm}_{sf['quad']}.png"))
                print(msg(f"[Wumu四联工坊] 💾 已保存 角色={char_name} 造型={nm}（四联图{'+四视角' if save_views else ''}）",
                          f"[Wumu Quadtych] 💾 saved character={char_name} outfit={nm} (quadtych{'+views' if save_views else ''})"))
            results.append(pil_to_tensor(tri))

        blank = torch.zeros(1, 4, 4, 3)
        out = [(r if r is not None else blank) for r in results]
        while len(out) < 5:
            out.append(blank)
        info = msg(f"角色={char_name} | 四联图×{len(active)} | Klein={unet_name} | {steps}步",
                   f"character={char_name} | quadtych×{len(active)} | Klein={unet_name} | {steps} steps")
        return (out[0], out[1], out[2], out[3], out[4], info)

NODE_CLASS_MAPPINGS_TRIP = {"WumuTriptychAtelier": WumuTriptychAtelier}
NODE_DISPLAY_NAME_MAPPINGS_TRIP = {"WumuTriptychAtelier": "Wumu 四联图工坊（Klein）"}
