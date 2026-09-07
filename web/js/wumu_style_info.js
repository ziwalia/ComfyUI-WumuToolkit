import { app } from "/scripts/app.js";

// Wumu 风格状态显示：选风格后实时显示 LoRA 文件名、是否已下载、下载地址
app.registerExtension({
    name: "Wumu.StyleInfo",

    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData?.name !== "WumuCharacterAtelier") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const styleWidget = this.widgets?.find(w => w.name === "style_name");
            if (!styleWidget) return result;

            // 创建只读信息行
            const infoWidget = this.addWidget("text", "LoRA状态", "检查中…", () => {}, { serialize: false });
            infoWidget.disabled = true;
            infoWidget.serialize = false;

            const updateInfo = async () => {
                const styleName = styleWidget.value;
                try {
                    const resp = await fetch(`/wumu/styles?name=${encodeURIComponent(styleName)}`);
                    const data = await resp.json();
                    if (!data.lora) {
                        infoWidget.value = "✅ 纯提示词风格（无需LoRA）";
                        infoWidget.color = "#7ee0a3";
                    } else if (data.available) {
                        infoWidget.value = `✅ ${data.lora}（已就绪，强度${data.strength}）`;
                        infoWidget.color = "#7ee0a3";
                    } else {
                        infoWidget.value = `⚠️ 缺 ${data.lora} → ${data.hf_url || "见styles.json"}`;
                        infoWidget.color = "#ff7b7b";
                    }
                } catch (e) {
                    infoWidget.value = "风格状态加载失败";
                    infoWidget.color = "#ff7b7b";
                }
                this.setDirtyCanvas(true, true);
            };

            // 挂在 widget callback 上（选风格即触发）
            const origCallback = styleWidget.callback;
            styleWidget.callback = function () {
                origCallback?.apply(this, arguments);
                updateInfo();
            };

            // 初始加载
            setTimeout(updateInfo, 500);

            return result;
        };
    },
});
