# -*- coding: utf-8 -*-
"""WumuToolkit：wumu-movie 专用节点集
主路径节点：
  WumuCharacterAtelier —— 定妆照工坊（FLUX.2 Dev）：文生图基础照 + ReferenceLatent 换装造型照
  WumuTriptychAtelier —— 四联图工坊（Klein）：造型照 → 证件照+正/侧/背四链 → 1920×1080 合成四联图
"""
import json
import os
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============================ 工坊节点注册 ============================
try:
    from .wumu_atelier import (WumuCharacterAtelier, STYLE_FILE,
                               NODE_CLASS_MAPPINGS_ATelier, NODE_DISPLAY_NAME_MAPPINGS_ATelier)
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_ATelier)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_ATelier)
except Exception as e:
    print(f"[WumuToolkit] ⚠ 定妆照工坊加载失败: {e}")

try:
    from .wumu_trip import WumuTriptychAtelier, NODE_CLASS_MAPPINGS_TRIP, NODE_DISPLAY_NAME_MAPPINGS_TRIP
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_TRIP)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_TRIP)
except Exception as e:
    print(f"[WumuToolkit] ⚠ 四联图工坊加载失败: {e}")

# ============================ 风格状态 API ============================
# 返回指定风格的 LoRA 文件名、是否已下载、下载地址
WEB_DIRECTORY = "./web"

try:
    from . import wumu_lang
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/wumu/lang")
    async def wumu_get_lang(request):
        return web.json_response({"lang": wumu_lang.get_lang(), "supported": wumu_lang.SUPPORTED})

    @PromptServer.instance.routes.post("/wumu/lang")
    async def wumu_set_lang(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        return web.json_response({"lang": wumu_lang.set_lang(data.get("lang"))})
except Exception as e:
    print(f"[WumuToolkit] ⚠ 语言API注册失败: {e}")

try:
    STYLE_FILE
except NameError:
    STYLE_FILE = os.path.join(os.path.dirname(__file__), "styles.json")

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/wumu/styles")
    async def wumu_style_info(request):
        name = request.rel_url.query.get("name", "")
        try:
            with open(STYLE_FILE, "r", encoding="utf-8") as f:
                styles = json.load(f)
        except Exception:
            return web.json_response({"error": "styles.json 读取失败"})

        style = next((s for s in styles if s.get("name") == name), None)
        if not style:
            return web.json_response({"lora": None, "available": False, "hf_url": "", "strength": 0})

        lora = style.get("lora")
        if not lora:
            return web.json_response({"lora": None, "available": True, "hf_url": "", "strength": 0})

        # 检查文件是否存在
        available = False
        for d in folder_paths.get_folder_paths("loras"):
            if os.path.exists(os.path.join(d, lora)):
                available = True
                break

        return web.json_response({
            "lora": lora,
            "available": available,
            "hf_url": style.get("hf_url", ""),
            "strength": style.get("strength", 0.8),
            "note": style.get("note", ""),
        })
except Exception as e:
    print(f"[WumuToolkit] ⚠ 风格状态API注册失败: {e}")
