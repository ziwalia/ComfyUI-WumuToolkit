import { app } from "/scripts/app.js";

const NODE_NAMES = ["WumuBaseAtelier", "WumuOutfitAtelier", "WumuCharacterAtelier", "WumuTriptychAtelier"];

let uiLang = "中文";
let pluginVersion = "";

async function fetchLang() {
    try {
        const r = await fetch("/wumu/lang");
        const d = await r.json();
        uiLang = d.lang || "中文";
        pluginVersion = d.version || "";
    } catch (e) { /* 默认中文 */ }
}

// 控件行标题双语：中文模式「中文名 + 原始键名」，English 模式简洁英文名
const LABELS = {
    mode: ["模式", "Mode"],
    char_name: ["角色", "Character"],
    desc_en: ["人物英文描述", "Identity description"],
    style_name: ["风格", "Style"],
    style_strength: ["风格强度", "Style strength"],
    unet_name: ["底模", "Base model"],
    clip_name: ["文本编码器", "Text encoder"],
    vae_name: ["VAE", "VAE"],
    steps: ["步数", "Steps"],
    guidance: ["引导值", "Guidance"],
    width: ["宽", "Width"],
    height: ["高", "Height"],
    seed: ["种子", "Seed"],
    use_turbo: ["Turbo加速", "Turbo"],
    base_use_reference: ["①图生图", "① img2img"],
    base_denoise: ["①重绘幅度", "① Denoise"],
    slot_use_external_base: ["②外部基础照", "② External base"],
    slot1_enable: ["造型1启用", "Outfit 1 on"],
    slot2_enable: ["造型2启用", "Outfit 2 on"],
    slot3_enable: ["造型3启用", "Outfit 3 on"],
    slot4_enable: ["造型4启用", "Outfit 4 on"],
    slot5_enable: ["造型5启用", "Outfit 5 on"],
    slot1_name: ["造型1名称", "Outfit 1 name"],
    slot2_name: ["造型2名称", "Outfit 2 name"],
    slot3_name: ["造型3名称", "Outfit 3 name"],
    slot4_name: ["造型4名称", "Outfit 4 name"],
    slot5_name: ["造型5名称", "Outfit 5 name"],
    slot1_desc: ["造型1提示词", "Outfit 1 prompt"],
    slot2_desc: ["造型2提示词", "Outfit 2 prompt"],
    slot3_desc: ["造型3提示词", "Outfit 3 prompt"],
    slot4_desc: ["造型4提示词", "Outfit 4 prompt"],
    slot5_desc: ["造型5提示词", "Outfit 5 prompt"],
    canvas_w: ["画布宽", "Canvas W"],
    canvas_h: ["画布高", "Canvas H"],
    front_h: ["证件照高度", "ID photo height"],
    side_h: ["全身照高度", "Body height"],
    auto_save: ["自动保存", "Auto-save"],
    save_views: ["另存单视角", "Save views"],
    save_dir: ["保存目录", "Save dir"],
    language: ["语言", "Language"],
};

function applyWidgetLabels(node) {
    for (const w of node.widgets || []) {
        const pair = LABELS[w.name];
        if (!pair) continue;
        w.label = uiLang === "English" ? pair[1] : `${pair[0]} ${w.name}`;
    }
    node.setDirtyCanvas(true, true);
}

function applyVersionTitle(node) {
    if (pluginVersion && node.title && !node.title.includes(`v${pluginVersion}`)) {
        node.title = `${node.title} v${pluginVersion}`;
    }
}

function toast(msg) {
    const el = document.createElement("div");
    el.textContent = msg;
    Object.assign(el.style, {
        position: "fixed", top: "16px", left: "50%", transform: "translateX(-50%)",
        background: "#335533", color: "#cfe8cf", padding: "8px 18px", borderRadius: "6px",
        zIndex: 99999, fontSize: "13px", boxShadow: "0 2px 8px rgba(0,0,0,.4)"
    });
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// Wumu 语言切换 + 行标题双语 + 标题栏版本号 + 风格 LoRA 状态显示
app.registerExtension({
    name: "Wumu.StyleInfo",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (!NODE_NAMES.includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            // 初始：拉取全局语言与版本 → 行标题双语 + 标题栏版本号
            fetchLang().then(() => {
                applyWidgetLabels(node);
                applyVersionTitle(node);
            });

            // language 下拉：保存全局设置；本节点行标题即时切换，参数提示刷新页面后生效
            const langWidget = node.widgets?.find(w => w.name === "language");
            if (langWidget) {
                const origLangCb = langWidget.callback;
                langWidget.callback = function () {
                    origLangCb?.apply(this, arguments);
                    uiLang = langWidget.value;
                    applyWidgetLabels(node);
                    fetch("/wumu/lang", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ lang: langWidget.value })
                    }).then(r => r.json())
                      .then(d => toast(d.lang === "English"
                          ? "Labels switched; refresh the page to switch tooltips"
                          : "本节点标签已切换；刷新页面后全部节点与参数提示同步切换"))
                      .catch(() => {});
                };
            }

            // LoRA 状态行（仅定妆照工坊）
            if (nodeData.name === "WumuCharacterAtelier") {
                const styleWidget = node.widgets?.find(w => w.name === "style_name");
                if (styleWidget) {
                    const infoWidget = node.addWidget("text", "LoRA状态", "检查中…", () => {}, { serialize: false });
                    infoWidget.disabled = true;
                    infoWidget.serialize = false;

                    const updateInfo = async () => {
                        const styleName = styleWidget.value;
                        try {
                            await fetchLang();
                            const resp = await fetch(`/wumu/styles?name=${encodeURIComponent(styleName)}`);
                            const data = await resp.json();
                            if (!data.lora) {
                                infoWidget.value = uiLang === "English" ? "✅ Prompt-only style (no LoRA needed)" : "✅ 纯提示词风格（无需LoRA）";
                                infoWidget.color = "#7ee0a3";
                            } else if (data.available) {
                                infoWidget.value = uiLang === "English"
                                    ? `✅ ${data.lora} (ready, strength ${data.strength})`
                                    : `✅ ${data.lora}（已就绪，强度${data.strength}）`;
                                infoWidget.color = "#7ee0a3";
                            } else {
                                infoWidget.value = uiLang === "English"
                                    ? `⚠️ Missing ${data.lora} → ${data.hf_url || "see styles.json"}`
                                    : `⚠️ 缺 ${data.lora} → ${data.hf_url || "见styles.json"}`;
                                infoWidget.color = "#ff7b7b";
                            }
                        } catch (e) {
                            infoWidget.value = uiLang === "English" ? "Style status load failed" : "风格状态加载失败";
                            infoWidget.color = "#ff7b7b";
                        }
                        node.setDirtyCanvas(true, true);
                    };

                    const origCallback = styleWidget.callback;
                    styleWidget.callback = function () {
                        origCallback?.apply(this, arguments);
                        updateInfo();
                    };

                    setTimeout(updateInfo, 500);
                }
            }

            return result;
        };
    },
});
