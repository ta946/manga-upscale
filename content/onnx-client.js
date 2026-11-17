/*
 * Content-script side of the ONNX transport.
 *
 *   content script <== MessagePort (transferables) ==> worker (via the bridge iframe)
 *
 * See bridge/bridge.js for why the iframe exists and content/onnx-worker.js for
 * the wire protocol. This is a thin promise wrapper around it.
 */
class OnnxBridge {
    constructor() {
        this.iframe = null;
        this.port = null;
        this.pending = new Map();
        this.nextId = 0;
        this.waiters = {};
        this.disposed = false;
        this.lost = false;
        this.gpuIO = false;
        this.limits = null;
        this.onFailure = null;           // called once if the worker dies under us
        this._onWindowMessage = null;
    }

    _waitFor(type, timeoutMs, timeoutMessage) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                delete this.waiters[type];
                reject(new Error(typeof timeoutMessage === 'function' ? timeoutMessage() : timeoutMessage));
            }, timeoutMs);
            this.waiters[type] = {
                resolve: (value) => { clearTimeout(timer); delete this.waiters[type]; resolve(value); },
                reject: (error) => { clearTimeout(timer); delete this.waiters[type]; reject(error); }
            };
        });
    }

    async start({ modelPath, scale, gpuIO = true, cpuOutput = true, warmup = [256, 256] }) {
        const extensionOrigin = new URL(chrome.runtime.getURL('/')).origin;

        const iframe = document.createElement('iframe');
        iframe.src = chrome.runtime.getURL(`bridge/bridge.html`);
        iframe.setAttribute('aria-hidden', 'true');
        iframe.tabIndex = -1;
        iframe.style.cssText =
            'position:fixed!important;left:-10px!important;top:-10px!important;' +
            'width:1px!important;height:1px!important;border:0!important;' +
            'opacity:0!important;pointer-events:none!important;';
        this.iframe = iframe;

        this._onWindowMessage = (event) => {
            if (event.source !== iframe.contentWindow || event.origin !== extensionOrigin) return;
            if (event.data?.type === 'manga-upscale:worker-error') {
                const error = new Error(event.data.message || 'ONNX worker crashed.');
                error.gpuFatal = true;
                this._fail(error);
                this.onFailure?.(error);
            }
        };
        window.addEventListener('message', this._onWindowMessage);

        const loaded = new Promise((resolve, reject) => {
            iframe.addEventListener('load', resolve, { once: true });
            iframe.addEventListener('error', () => reject(new Error('Bridge iframe failed to load.')), { once: true });
        });
        // documentElement (not body) so the page-wide MutationObserver does not fire for our node.
        document.documentElement.appendChild(iframe);
        await loaded;
        if (this.disposed) throw new Error('ONNX bridge was disposed during startup.');

        const channel = new MessageChannel();
        this.port = channel.port1;
        this.port.onmessage = (event) => this._onPortMessage(event.data || {});

        const connected = this._waitFor(
            'connected', 8000,
            'ONNX bridge did not respond. The page may block extension iframes (CSP frame-src).'
        );
        iframe.contentWindow.postMessage({ type: 'manga-upscale:connect' }, extensionOrigin, [channel.port2]);
        await connected;

        const ready = this._waitFor('ready', 120000, 'Timed out initializing the ONNX worker.');
        this.port.postMessage({ type: 'init', modelPath, scale, gpuIO, cpuOutput, warmup });
        const info = await ready;
        this.gpuIO = !!info.gpuIO;
        this.limits = info.limits;
        return info;
    }

    _onPortMessage(message) {
        switch (message.type) {
            case 'connected': this.waiters.connected?.resolve(); break;
            case 'ready': this.waiters.ready?.resolve(message); break;
            case 'initError': this.waiters.ready?.reject(new Error(message.error || 'ONNX worker initialization failed.')); break;
            case 'deviceLost':
                this.lost = true;
                MU.error('WebGPU device lost:', message.message);
                break;
            case 'result':
            case 'error': {
                const entry = this.pending.get(message.id);
                if (!entry) { message.bitmap?.close?.(); return; }     // cancelled / timed out meanwhile
                this.pending.delete(message.id);
                clearTimeout(entry.timer);
                if (message.type === 'error') {
                    const error = new Error(message.error || 'ONNX inference failed.');
                    error.gpuFatal = !!message.gpuFatal;
                    error.code = message.code || null;
                    entry.reject(error);
                } else {
                    entry.resolve(message);
                }
                break;
            }
        }
    }

    /**
     * Runs inference. `bitmap` is TRANSFERRED (detached). Resolves with
     * { bitmap, width, height, timings }; the caller owns the returned bitmap.
     * `plan` = { scale, targetW, targetH, tile: null | { size, overlap } }.
     */
    run(bitmap, plan, { signal, timeoutMs = 60000 } = {}) {
        if (this.disposed || !this.port) return Promise.reject(new Error('ONNX bridge is not running.'));
        const id = ++this.nextId;
        return new Promise((resolve, reject) => {
            const abort = () => {
                if (!this.pending.delete(id)) return;
                clearTimeout(timer);
                try { this.port?.postMessage({ type: 'cancel', id }); } catch (_) {}
                reject(new DOMException('Cancelled', 'AbortError'));
            };
            const timer = setTimeout(() => {
                this.pending.delete(id);
                signal?.removeEventListener('abort', abort);
                const error = new Error('ONNX worker timed out.');
                error.gpuFatal = true;
                reject(error);
            }, timeoutMs);
            this.pending.set(id, {
                resolve: (v) => { signal?.removeEventListener('abort', abort); resolve(v); },
                reject: (e) => { signal?.removeEventListener('abort', abort); reject(e); },
                timer
            });
            signal?.addEventListener('abort', abort, { once: true });
            try {
                this.port.postMessage({ type: 'run', id, bitmap, plan }, [bitmap]);
            } catch (error) {
                clearTimeout(timer);
                this.pending.delete(id);
                reject(error);
            }
        });
    }

    _fail(error) {
        this.waiters.connected?.reject(error);
        this.waiters.ready?.reject(error);
        for (const entry of this.pending.values()) { clearTimeout(entry.timer); entry.reject(error); }
        this.pending.clear();
    }

    dispose() {
        if (this.disposed) return;
        this.disposed = true;
        this.onFailure = null;
        this._fail(new Error('ONNX bridge was disposed.'));
        try { this.port?.postMessage({ type: 'dispose' }); } catch (_) {}
        try { this.port?.close(); } catch (_) {}
        if (this._onWindowMessage) window.removeEventListener('message', this._onWindowMessage);
        this.iframe?.remove();   // unloading the bridge document kills its worker / releases its lock
        this.port = null;
        this.iframe = null;
    }
}

window.OnnxBridge = OnnxBridge;
