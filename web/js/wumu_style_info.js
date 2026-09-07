import { app } from "/scripts/app.js";

const NODE_NAMES = ["WumuCharacterAtelier", "WumuTriptychAtelier"];

let uiLang = "中文";
async function fetchLang() {
    try {
        const r = await fetch("/wumu/lang");
        const d = await r.json();
        uiLang = d.lang || "中文";
    } catch (e) { /* 默认中文 */ }
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

// Wumu 语言切换 + 风格 LoRA 状态显示
app.registerExtension({
    name: "Wumu.StyleInfo",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (!NODE_NAMES.includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // language 下拉：切换时保存全局设置（参数提示语言），刷新页面后生效
            const langWidget = this.widgets?.find(w => w.name === "language");
            if (langWidget) {
                const origLangCb = langWidget.callback;
                langWidget.callback = function () {
                    origLangCb?.apply(this, arguments);
                    uiLang = langWidget.value;
                    fetch("/wumu/lang", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ lang: langWidget.value })
                    }).then(r => r.json())
                      .then(d => toast(d.lang === "English"
                          ? "Tooltips will switch to English after page refresh"
                          : "参数提示将在刷新页面后切换为中文"))
                      .catch(() => {});
                };
            }

            // LoRA 状态行（仅定妆照工坊）
            if (nodeData.name === "WumuCharacterAtelier") {
                const styleWidget = this.widgets?.find(w => w.name === "style_name");
                if (styleWidget) {
                    const infoWidget = this.addWidget("text", "LoRA状态", "检查中…", () => {}, { serialize: false });
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
                        this.setDirtyCanvas(true, true);
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
