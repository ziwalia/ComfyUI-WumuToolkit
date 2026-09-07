# -*- coding: utf-8 -*-
"""语言设置（节点级输出语言 + 全局提示语言）
- 节点上的 language 下拉：控制该节点运行期输出（落盘目录/文件后缀/四视角提示词/info/日志）
- 全局提示语言：前端切换下拉时 POST /wumu/lang 写入 lang.json，
  INPUT_TYPES 通过 L() 读取 → 刷新页面后参数提示（tooltip）切换语言。默认中文。
"""
import os
import json

LANG_FILE = os.path.join(os.path.dirname(__file__), "lang.json")
SUPPORTED = ["中文", "English"]

# 资产落盘子目录：资产\角色 ↔ assets\characters
ASSET_DIRS = {"中文": ("资产", "角色"), "English": ("assets", "characters")}


def get_lang():
    try:
        with open(LANG_FILE, "r", encoding="utf-8") as f:
            v = json.load(f).get("lang")
        if v in SUPPORTED:
            return v
    except Exception:
        pass
    return "中文"


def set_lang(v):
    if v not in SUPPORTED:
        v = "中文"
    try:
        with open(LANG_FILE, "w", encoding="utf-8") as f:
            json.dump({"lang": v}, f)
    except Exception:
        pass
    return v


def L(zh, en):
    return zh if get_lang() == "中文" else en


def asset_dir(save_dir, lang="中文"):
    return os.path.join(save_dir, *ASSET_DIRS.get(lang, ASSET_DIRS["中文"]))
