/*
 * Extension-origin bridge page, embedded as a hidden iframe by the content script
 * (see content/onnx-client.js).
 *
 * A content script cannot construct a chrome-extension:// Worker, and
 * chrome.runtime messaging is JSON-only (no transferables). This page is
 * extension-origin, so it can create the worker, and it hands the worker one end of a
 * MessageChannel supplied by the content script. After that single handshake the
 * content script and the worker talk directly and ImageBitmaps are transferred.
 *
 * The worker lives and dies with this iframe: removing the iframe frees the GPU.
 */
(() => {
    let parentOrigin = null;
    let worker = null;

    const tell = (message) => {
        if (parentOrigin) window.parent.postMessage({ type: 'manga-upscale:worker-error', message }, parentOrigin);
    };

    window.addEventListener('message', (event) => {
        if (event.source !== window.parent) return;
        const message = event.data;
        const port = event.ports && event.ports[0];
        if (!message || message.type !== 'manga-upscale:connect' || !port) return;
        if (worker) return;                   // first connection wins; the client recreates the iframe

        parentOrigin = event.origin;
        try {
            worker = new Worker(chrome.runtime.getURL('content/onnx-worker.js'));
            // A dead worker leaves the port silent: tell the content script so it can fail fast.
            worker.onerror = (e) => tell(e.message || 'ONNX worker crashed.');
            worker.postMessage({ type: 'connect' }, [port]);
        } catch (error) {
            tell(`Could not start the ONNX worker: ${error.message}`);
        }
    });
})();
