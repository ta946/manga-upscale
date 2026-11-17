function setupContextMenu() {
    chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({
            id: "force-upscale-img",
            title: "Force Upscale Image",
            documentUrlPatterns: ["*://nelomanga.net/*", "*://*.nelomanga.net/*"],
            contexts: ["image"]
        }, () => {
            if (chrome.runtime.lastError) {
                console.log("Context menu creation note:", chrome.runtime.lastError.message);
            } else {
                console.log("Context menu 'Force Upscale Image' created successfully.");
            }
        });
    });
}

chrome.runtime.onInstalled.addListener(() => {
    console.log("Manga Upscaler Extension Installed.");
    chrome.storage.sync.get(['ramMode', 'engine', 'enabled'], (settings) => {
        const defaults = {};
        if (settings.ramMode === undefined) defaults.ramMode = 'low';
        if (settings.engine === undefined || settings.engine === 'off') defaults.engine = 'canvas';
        if (settings.enabled === undefined) defaults.enabled = settings.engine !== 'off';
        if (Object.keys(defaults).length) chrome.storage.sync.set(defaults);
    });

    setupContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
    setupContextMenu();
});


// Listen for when the user clicks the context menu item
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "force-upscale-img" && tab?.id) {
        // Send a message to the content script on the active tab with the targeted image URL
        chrome.tabs.sendMessage(tab.id, {
            action: "forceUpscale",
            srcUrl: info.srcUrl
        }, { frameId: info.frameId }).catch(err => {
            console.error("Could not send force-upscale message to tab:", err);
        });
    }
});
// Fetch remote image bytes from the extension service worker. Host permissions
// allow this request even when the image host does not grant page CORS access.
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request?.action !== "fetchImageBytes" || typeof request.url !== "string") return;
    (async () => {
        try {
            const response = await fetch(request.url, { credentials: "include", cache: "force-cache" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const blob = await response.blob();
            sendResponse({ ok: true, bytes: await blob.arrayBuffer(), type: blob.type });
        } catch (error) {
            sendResponse({ ok: false, error: error.message });
        }
    })();
    return true;
});
