
/**
 * Engine Manager (Classic Script Global)
 *
 * Owns the active upscaling engine. For ONNX it also owns the worker lifecycle:
 *
 *   idle --ensureONNX--> starting --> ready --releaseONNX--> idle
 *                                       |
 *                            failure -> recovering (backoff, <= 3 per minute) -> ready
 *                                       |
 *                                  give up -> failed (canvas fallback)
 */

// GPU memory guard. Feature maps cost ~1.3 KB per input pixel (64 channels, several live at
// once) and ORT pools freed buffers by exact size, so very large passes exhaust GPU memory and
// take the whole GPU process down (black screen, blank canvases). 0.75 MP is the largest single
// pass known to be safe; tiles get extra headroom. Raise these only if your GPU is known to cope.
const ONNX_SAFE_PIXELS = 750000;                    // ceiling for ANY single model pass, whatever the setting
const ONNX_TILE_PIXELS = 500000;                    // per-tile budget in tile mode
const ONNX_SHAPE_UNIT = 64;                         // the worker pads model inputs to this multiple
const ONNX_TILE_OVERLAP = 32;                       // input px of context around each tile
const ONNX_MAX_RECOVERIES = 3;
const ONNX_RECOVERY_WINDOW_MS = 60000;
const ONNX_RECOVERY_BACKOFF_MS = [0, 1000, 3000];
const ONNX_PERMANENT_ERROR = /WebGPU is unavailable|Refusing to load|no available backend|Bridge iframe failed/i;
const ONNX_RECOVERABLE_ERROR = /timed out|worker crashed|device.*lost|WebGPU.*lost|GPUDeviceLost|external Instance reference|Failed to get GPU adapter|did not respond/i;

class EngineManager {
    constructor({ settings }) {
        this.settings = settings;               // live object, mutated in place by content.js
        this.currentEngine = null;
        this.engineType = 'canvas';
        this.canvas = document.createElement('canvas');
        this.initializationPromise = Promise.resolve();
        this.gpuAdapter = null;
        this.gpuDevice = null;
        this.modelCatalog = null;
        this.canvasScaleFactor = 1;
        this.onnxFallbackToCanvas = false;
        this.forceDownscale = false;            // set when tiling turns out to be impossible

        this.onnx = {
            state: 'idle',                      // idle | starting | ready | recovering | failed
            modelPath: null,
            scale: 2,
            bridge: null,
            startPromise: null,
            gpuIO: null,
            limits: null,
            error: null,
            recoveries: [],                     // timestamps, pruned to the last minute
            failCounts: new Map()               // image key -> recoverable failures
        };
    }

    /* -------------------------------------------------------------- */
    /* Engine selection                                                 */
    /* -------------------------------------------------------------- */

    switchEngine(type, options = {}) {
        const changed = type !== this.engineType || options.modelPath !== this.onnx.modelPath;
        this.engineType = type;
        this.canvasScaleFactor = type === 'onnx' ? 1 : 2;
        this.onnxFallbackToCanvas = false;
        this.forceDownscale = false;
        if (changed) MU.info(`engine → ${type}${type === 'onnx' ? ` (${MU.name(options.modelPath)})` : ''}`);

        const previous = this.initializationPromise.catch(() => {});
        this.initializationPromise = (async () => {
            await previous;   // never run two init/destroy sequences concurrently
            try {
                await this.destroy();

                switch (type) {
                    case 'anime4k-webgl':
                        this.currentEngine = await this._initAnime4KWebGL();
                        break;
                    case 'anime4k-webgpu':
                        this.currentEngine = await this._initAnime4KWebGPU();
                        break;
                    case 'onnx':
                        // The worker itself starts on demand (ensureONNX), so an idle or
                        // lazy tab costs nothing until an image actually needs it.
                        this.onnx.modelPath = options.modelPath || this.onnx.modelPath;
                        this.onnx.state = 'idle';
                        this.onnx.error = null;
                        this.onnx.recoveries = [];
                        this.onnx.failCounts.clear();
                        break;
                    default:
                        this.currentEngine = null;
                }
            } catch (error) {
                MU.error(`engine ${type} failed to start, using canvas:`, error.message || error);
                this.currentEngine = null;
                this.engineType = 'canvas';
                this.canvasScaleFactor = type === 'onnx' ? 1 : 2;
                if (type === 'onnx') this.onnxFallbackToCanvas = true;
            }
        })();
        return this.initializationPromise;
    }

    async resolveModelScaleFactor(modelPath) {
        if (!this.modelCatalog) {
            const response = await fetch(chrome.runtime.getURL('models.json'));
            if (!response.ok) throw new Error(`HTTP ${response.status} loading models.json`);
            this.modelCatalog = await response.json();
        }
        const selectedModel = this.modelCatalog.models?.find(model => model.path === modelPath);
        const scale = Number(selectedModel?.scaleFactor) || 2;
        if (![1, 2, 4].includes(scale)) {
            throw new Error(`models.json has no valid scaleFactor for model: ${modelPath}`);
        }
        return scale;
    }

    getStatus() {
        const o = this.onnx;
        const usingOnnx = this.engineType === 'onnx' || this.onnxFallbackToCanvas;
        return {
            engine: this.engineType,
            fallback: this.onnxFallbackToCanvas,
            onnx: usingOnnx ? {
                state: o.state,
                gpuIO: o.gpuIO,
                error: o.error,
                recoveries: o.recoveries.length
            } : null
        };
    }

    /* -------------------------------------------------------------- */
    /* Execution                                                        */
    /* -------------------------------------------------------------- */

    /**
     * Upscales `imageBitmap` (never consumed or closed). Resolves with
     * { canvas, info } where info = { engine, outW, outH, timings? }.
     */
    async execute(imageBitmap, options = {}) {
        if (this.engineType === 'onnx' && options.modelPath && this.onnx.modelPath !== options.modelPath) {
            await this.switchEngine('onnx', options);
        }
        await this.initializationPromise;

        const width = imageBitmap.width;
        const height = imageBitmap.height;
        const widthCap = this.settings.maxOutputWidth > 0 ? this.settings.maxOutputWidth : Infinity;

        if (this.engineType === 'onnx') {
            return this._executeONNX(imageBitmap, options);
        }

        const targetScale = this.engineType === 'canvas' ? this.canvasScaleFactor : 2;
        const outputScale = Math.min(targetScale, widthCap / width);
        const targetWidth = Math.max(1, Math.round(width * outputScale));
        const targetHeight = Math.max(1, Math.round(height * outputScale));

        let canvas;
        switch (this.engineType) {
            case 'anime4k-webgl':
                canvas = await this._executeAnime4KWebGL(imageBitmap, targetWidth, targetHeight);
                break;
            case 'anime4k-webgpu':
                canvas = await this._executeAnime4KWebGPU(imageBitmap, targetWidth, targetHeight);
                break;
            default:
                canvas = await this._executeStandardCanvas(imageBitmap, targetWidth, targetHeight);
        }
        return { canvas, info: { engine: this.engineType, outW: targetWidth, outH: targetHeight } };
    }

    async _initAnime4KWebGL() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 512;
        this.canvas.height = 512;

        const ns = window.Anime4KJS;

        if (!ns || !ns.ImageUpscaler) {
            throw new Error(
                'Could not find window.Anime4KJS or ImageUpscaler.'
            );
        }

        const ImageUpscaler = ns.ImageUpscaler;
        const profile = ns.ANIME4KJS_SIMPLE_M_2X;

        const engine = new ImageUpscaler(profile);

        return {
            engine,

            render: (textureSource, targetCanvas) => {
                engine.attachSource(
                    textureSource,
                    targetCanvas || this.canvas
                );
                engine.upscale();
            }
        };
    }

    async _initAnime4KWebGPU() {
        if (!navigator.gpu) {
            throw new Error(
                'WebGPU is not supported in this browser.'
            );
        }

        // anime4k-webgpu is injected as a classic content script
        // before engine-manager.js.
        const anime4k = window['anime4k-webgpu'];

        if (!anime4k || typeof anime4k.ModeAA !== 'function') {
            throw new Error(
                'Could not find anime4k-webgpu.ModeAA. ' +
                'Make sure its bundle is loaded before engine-manager.js.'
            );
        }

        const adapter = await navigator.gpu.requestAdapter();

        if (!adapter) {
            throw new Error(
                'Failed to find an appropriate WebGPU adapter.'
            );
        }

        const device = await adapter.requestDevice();

        device.lost.then((info) => {
            if (this.currentEngine?.device === device) {
                MU.error('anime4k WebGPU device lost:', info.message || info.reason);

                this.currentEngine = null;
            }
        });

        this.canvas = document.createElement('canvas');

        const context = this.canvas.getContext('webgpu');

        if (!context) {
            device.destroy();

            throw new Error(
                'Failed to create a WebGPU canvas context.'
            );
        }

        const format = navigator.gpu.getPreferredCanvasFormat();

        context.configure({
            device,
            format,
            alphaMode: 'premultiplied'
        });

        // Output shader: draws the final Anime4K texture
        // to the WebGPU canvas.
        const renderShader = device.createShaderModule({
            label: 'Anime4K WebGPU image output shader',

            code: `
                struct VertexOutput {
                    @builtin(position) position: vec4f,
                    @location(0) uv: vec2f,
                };

                @vertex
                fn vertexMain(
                    @builtin(vertex_index) index: u32
                ) -> VertexOutput {

                    var positions = array<vec2f, 6>(
                        vec2f(-1.0, -1.0),
                        vec2f( 1.0, -1.0),
                        vec2f(-1.0,  1.0),

                        vec2f(-1.0,  1.0),
                        vec2f( 1.0, -1.0),
                        vec2f( 1.0,  1.0)
                    );

                    var uvs = array<vec2f, 6>(
                        vec2f(0.0, 1.0),
                        vec2f(1.0, 1.0),
                        vec2f(0.0, 0.0),

                        vec2f(0.0, 0.0),
                        vec2f(1.0, 1.0),
                        vec2f(1.0, 0.0)
                    );

                    var output: VertexOutput;

                    output.position = vec4f(
                        positions[index],
                        0.0,
                        1.0
                    );

                    output.uv = uvs[index];

                    return output;
                }

                @group(0) @binding(0)
                var imageSampler: sampler;

                @group(0) @binding(1)
                var imageTexture: texture_2d<f32>;

                @fragment
                fn fragmentMain(
                    input: VertexOutput
                ) -> @location(0) vec4f {

                    return textureSample(
                        imageTexture,
                        imageSampler,
                        input.uv
                    );
                }
            `
        });

        const outputPipeline = device.createRenderPipeline({
            label: 'Anime4K WebGPU output pipeline',

            layout: 'auto',

            vertex: {
                module: renderShader,
                entryPoint: 'vertexMain'
            },

            fragment: {
                module: renderShader,
                entryPoint: 'fragmentMain',

                targets: [{
                    format
                }]
            },

            primitive: {
                topology: 'triangle-list'
            }
        });

        const sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear'
        });

        return {
            device,
            context,
            format,
            Anime4KPipeline: anime4k.ModeAA, // 1
            // Anime4KPipeline: anime4k.ModeA, // 2
            // Anime4KPipeline: anime4k.ModeBB, // 3
            // Anime4KPipeline: anime4k.ModeCA, // 4
            // Anime4KPipeline: anime4k.ModeB, // 5
            // Anime4KPipeline: anime4k.ModeC, // 6
            outputPipeline,
            sampler
        };
    }

    /* -------------------------------------------------------------- */
    /* ONNX lifecycle                                                   */
    /* -------------------------------------------------------------- */

    /** Resolves with a ready bridge, starting (or joining a start of) the worker if needed. */
    ensureONNX() {
        const o = this.onnx;
        if (o.state === 'ready' && o.bridge) return Promise.resolve(o.bridge);
        if (o.startPromise) return o.startPromise;
        return this._startONNX('start', 0);
    }

    /** Best-effort warm start for "auto" mode; failures surface on the first real job. */
    prewarmONNX() {
        if (this.engineType === 'onnx' && !this.onnxFallbackToCanvas) this.ensureONNX().catch(() => {});
    }

    _startONNX(kind, delayMs) {
        const o = this.onnx;
        o.state = kind === 'recover' ? 'recovering' : 'starting';
        const startedAt = performance.now();

        o.startPromise = (async () => {
            if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
            try {
                const scale = await this.resolveModelScaleFactor(o.modelPath);
                const base = { modelPath: o.modelPath, scale, gpuIO: this.settings.onnxGpuIO, cpuOutput: this.settings.onnxCpuOutput };
                MU.info(`onnx ${kind === 'recover' ? 'restarting' : 'starting'} · ${MU.name(o.modelPath)}`);

                const bridge = new OnnxBridge();
                let info;
                try {
                    info = await bridge.start(base);
                } catch (error) {
                    bridge.dispose();
                    throw error;
                }

                bridge.onFailure = (error) => this._onBridgeFailure(bridge, error);
                o.bridge = bridge;
                o.scale = scale;
                o.gpuIO = info.gpuIO;
                o.limits = info.limits;
                this.forceDownscale = false;           // a restarted worker may well support tiling again
                o.state = 'ready';
                o.error = null;
                MU.info(`onnx ready in ${MU.ms(performance.now() - startedAt)}ms · ${info.gpuIO ? 'GPU I/O' : 'CPU I/O'}`);
                return bridge;
            } catch (error) {
                o.state = 'failed';
                o.error = error.message || String(error);
                o.bridge = null;
                throw error;
            } finally {
                o.startPromise = null;
            }
        })();
        return o.startPromise;
    }

    /** Frees the GPU/worker while idle. The next image wakes it via ensureONNX(). */
    releaseONNX(reason = 'idle') {
        const o = this.onnx;
        if (!o.bridge || o.startPromise) return false;
        o.bridge.dispose();
        o.bridge = null;
        o.state = 'idle';
        MU.info(`onnx ${reason} → worker released`);
        return true;
    }

    _onBridgeFailure(bridge, error) {
        const o = this.onnx;
        if (o.bridge !== bridge) return;
        // The next job notices and recovers (rate-limited); nothing to do eagerly.
        MU.warn('onnx worker died:', error.message);
        bridge.dispose();
        o.bridge = null;
        o.state = 'idle';
        o.error = error.message;
    }

    /** Restart after a failure. Single-flight, rate limited, with backoff. */
    _recover(failedBridge, error) {
        const o = this.onnx;
        if (failedBridge && o.bridge !== failedBridge) return this.ensureONNX();   // already replaced
        if (!failedBridge && o.startPromise) return o.startPromise;

        const t = Date.now();
        o.recoveries = o.recoveries.filter((x) => t - x < ONNX_RECOVERY_WINDOW_MS);
        if (o.recoveries.length >= ONNX_MAX_RECOVERIES) {
            return Promise.reject(Object.assign(
                new Error(`${error.message} (gave up after ${ONNX_MAX_RECOVERIES} restarts in a minute)`),
                { gaveUp: true }
            ));
        }
        o.recoveries.push(t);
        const attempt = o.recoveries.length;
        MU.warn(`onnx recovering ${attempt}/${ONNX_MAX_RECOVERIES}: ${error.message}`);
        if (failedBridge) { failedBridge.dispose(); o.bridge = null; }
        return this._startONNX('recover', ONNX_RECOVERY_BACKOFF_MS[attempt - 1] || 3000);
    }

    _giveUp(error) {
        const o = this.onnx;
        this.engineType = 'canvas';
        this.canvasScaleFactor = 1;
        this.onnxFallbackToCanvas = true;
        o.state = 'failed';
        o.error = error.message || String(error);
        o.bridge?.dispose();
        o.bridge = null;
        MU.error(`onnx failed → canvas fallback: ${o.error}`);
        return Object.assign(error, { gaveUp: true });
    }

    /** Re-enable ONNX after a fallback (popup "Retry" or a settings change). */
    retryONNX() {
        if (!this.onnxFallbackToCanvas && this.engineType === 'onnx') return this.ensureONNX().catch(() => {});
        return this.switchEngine('onnx', { modelPath: this.onnx.modelPath }).then(() => this.prewarmONNX());
    }

    /**
     * Restart until a worker is up, the error is permanent, or the restart budget is
     * spent. Each attempt counts against the budget, so this always terminates.
     */
    async _recoverLoop(failedBridge, error) {
        let bridge = failedBridge;
        let err = error;
        for (;;) {
            if (ONNX_PERMANENT_ERROR.test(err.message)) throw this._giveUp(err);
            try {
                return await this._recover(bridge, err);
            } catch (next) {
                if (next.gaveUp) throw this._giveUp(next);
                bridge = null;          // the failed bridge is already disposed; this was a start failure
                err = next;
            }
        }
    }

    /**
     * Runs `fn(bridge)` with recovery: start failures and worker deaths are retried
     * on a fresh worker (rate limited). An image that kills the worker twice is
     * reported as poison instead of burning the restart budget.
     */
    async _withRecovery(fn, key) {
        const o = this.onnx;
        let lastError;
        for (let pass = 0; pass < 3; pass++) {
            let bridge;
            try {
                bridge = await this.ensureONNX();
            } catch (error) {
                lastError = error;
                bridge = await this._recoverLoop(null, error);
            }

            try {
                return await fn(bridge);
            } catch (error) {
                if (error.name === 'AbortError') throw error;
                // A sibling job's crash already replaced this worker: our run was merely collateral.
                // Retry on the new one without counting it as a failure of ours.
                if (/was disposed/.test(error.message) && o.bridge !== bridge) { lastError = error; continue; }
                if (!(error.gpuFatal || ONNX_RECOVERABLE_ERROR.test(error.message))) throw error;
                lastError = error;

                const count = key ? (o.failCounts.get(key) || 0) + 1 : 1;
                if (key) o.failCounts.set(key, count);
                await this._recoverLoop(bridge, error);
                if (count >= 2) {
                    throw Object.assign(new Error(`image repeatedly crashed the worker: ${error.message}`), { poison: true });
                }
            }
        }
        throw lastError;
    }

    /* -------------------------------------------------------------- */
    /* ONNX execution                                                   */
    /* -------------------------------------------------------------- */

    /**
     * Decide input size, target size and tiling for one image. Pure; easy to unit test.
     *
     * The TARGET size depends only on the width cap, never on the pixel budget, so every page
     * of a chapter ends up the same width. When the budget is exceeded the model input is
     * shrunk (downscale) or split (tile), and the result is resampled to the same target.
     */
    planONNX(width, height, scale, limits) {
        const cfg = this.settings;
        const widthCap = cfg.maxOutputWidth > 0 ? cfg.maxOutputWidth : Infinity;
        const passCap = Math.min(ONNX_SAFE_PIXELS, limits?.pixelCap ?? Infinity);
        const maxTexture = limits?.maxTexture ?? 8192;
        const unit = ONNX_SHAPE_UNIT;
        const padded = (w, h) => Math.ceil(w / unit) * unit * Math.ceil(h / unit) * unit;

        const outScale = Math.min(1, widthCap / (width * scale));
        let targetW = width * scale * outScale;
        let targetH = height * scale * outScale;
        let inW = targetW / scale;
        let inH = targetH / scale;

        let tile = null;
        let mode = 'single';
        if (padded(inW, inH) > passCap) {
            if (cfg.oversizeMode === 'tile' && this.onnx.gpuIO && !this.forceDownscale) {
                const budget = Math.min(passCap, ONNX_TILE_PIXELS);
                tile = { size: Math.max(256, Math.floor(Math.sqrt(budget) / unit) * unit), overlap: ONNX_TILE_OVERLAP };
                mode = 'tile';
            } else {
                // Shrink only what the MODEL sees; the target (and so the page width) is untouched.
                let k = Math.sqrt(passCap / (inW * inH));
                for (let i = 0; i < 12 && padded(inW * k, inH * k) > passCap; i++) k *= 0.97;
                inW *= k; inH *= k;
                mode = 'downscale';
            }
        }
        // Source and target must each fit in a GPU texture.
        const fit = Math.min(1, maxTexture / Math.max(inW, inH, targetW, targetH));
        if (fit < 1) { targetW *= fit; targetH *= fit; inW *= fit; inH *= fit; mode += '+fit'; }

        return {
            inW: Math.max(2, Math.floor(inW / 2) * 2),        // even input, as the model set expects
            inH: Math.max(2, Math.floor(inH / 2) * 2),
            targetW: Math.max(1, Math.round(targetW)),
            targetH: Math.max(1, Math.round(targetH)),
            tile,
            mode,
            passCap
        };
    }

    async _executeONNX(imageBitmap, { signal, key } = {}) {
        if (this.onnxFallbackToCanvas) return this._executeCanvasFallback(imageBitmap);

        try {
            return await this._withRecovery((bridge) => this._runONNX(bridge, imageBitmap, signal), key);
        } catch (error) {
            if (error.gaveUp) return this._executeCanvasFallback(imageBitmap);
            throw error;
        }
    }

    async _runONNX(bridge, imageBitmap, signal) {
        const o = this.onnx;
        const scale = o.scale;
        const w = imageBitmap.width;
        const h = imageBitmap.height;

        for (let attempt = 0; attempt < 2; attempt++) {
            const plan = this.planONNX(w, h, scale, bridge.limits);
            const resize = plan.inW !== w || plan.inH !== h;
            const prepStart = performance.now();

            // One browser-side op: a resize, or a plain copy. The copy is needed because
            // the transfer detaches the bitmap and the caller still owns `imageBitmap`.
            let input = await createImageBitmap(
                imageBitmap,
                resize ? { resizeWidth: plan.inW, resizeHeight: plan.inH, resizeQuality: 'high' } : undefined
            );
            const prepMs = performance.now() - prepStart;
            if (signal?.aborted) { input.close?.(); throw new DOMException('Cancelled', 'AbortError'); }

            let result;
            const sendStart = performance.now();
            try {
                result = await bridge.run(
                    input,
                    { scale, targetW: plan.targetW, targetH: plan.targetH, tile: plan.tile },
                    { signal }
                );
            } catch (error) {
                if (error.code === 'NEEDS_GPU_IO' && attempt === 0) {
                    MU.warn('tiling unavailable (no GPU I/O); downscaling oversize images instead');
                    this.forceDownscale = true;
                    continue;
                }
                if (error.code === 'SCALE_MISMATCH') {
                    error.message += ' Check scaleFactor for this model in models.json.';
                }
                throw error;
            } finally {
                input.close?.();    // no-op once transferred
            }
            const roundTripMs = performance.now() - sendStart;

            let bitmap = result.bitmap;
            if (!result.timings.gpuIO && (result.width !== plan.inW * scale || result.height !== plan.inH * scale)) {
                bitmap.close?.();
                throw new Error(`Model output ${result.width}x${result.height} does not match ${scale}x of ${plan.inW}x${plan.inH}.`);
            }
            // CPU path returns model-sized output; the GPU path already resampled to the target.
            if (result.width !== plan.targetW || result.height !== plan.targetH) {
                const resized = await createImageBitmap(bitmap, {
                    resizeWidth: plan.targetW, resizeHeight: plan.targetH, resizeQuality: 'high'
                });
                bitmap.close?.();
                bitmap = resized;
            }

            return {
                canvas: this._bitmapToCanvas(bitmap),
                info: {
                    engine: 'onnx',
                    outW: plan.targetW,
                    outH: plan.targetH,
                    mode: plan.mode,
                    timings: { ...result.timings, prepMs: result.timings.prepMs, clientPrepMs: prepMs, roundTripMs }
                }
            };
        }
        throw new Error('ONNX run failed after downscale fallback.');
    }

    /** Wrap an ImageBitmap in a canvas without copying pixels. */
    _bitmapToCanvas(bitmap) {
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const context = canvas.getContext('bitmaprenderer');
        if (!context) {
            canvas.getContext('2d').drawImage(bitmap, 0, 0);
            bitmap.close?.();
            return canvas;
        }
        context.transferFromImageBitmap(bitmap);   // consumes the bitmap
        return canvas;
    }

    async _executeCanvasFallback(imageBitmap) {
        const widthCap = this.settings.maxOutputWidth > 0 ? this.settings.maxOutputWidth : Infinity;
        const outW = Math.min(imageBitmap.width, widthCap);
        const outH = Math.max(1, Math.round(imageBitmap.height * outW / imageBitmap.width));
        const canvas = await this._executeStandardCanvas(imageBitmap, outW, outH);
        return { canvas, info: { engine: 'canvas (fallback)', outW, outH } };
    }

    async _executeAnime4KWebGL(
        imageBitmap,
        targetWidth,
        targetHeight
    ) {
        this.canvas.width = targetWidth;
        this.canvas.height = targetHeight;

        if (
            this.currentEngine &&
            typeof this.currentEngine.render === 'function'
        ) {
            this.currentEngine.render(
                imageBitmap,
                this.canvas
            );
        }

        // return await this._canvasToBlobUrl();
        return await this._transferCanvas(targetWidth, targetHeight);
    }

    async _executeAnime4KWebGPU(
        imageBitmap,
        targetWidth,
        targetHeight
    ) {
        if (!this.currentEngine) {
            throw new Error('Anime4K WebGPU engine is not initialized.');
        }

        const {
            device,
            context,
            format,
            Anime4KPipeline,
            outputPipeline,
            sampler
        } = this.currentEngine;

        const width = imageBitmap.width;
        const height = imageBitmap.height;

        this.canvas.width = targetWidth;
        this.canvas.height = targetHeight;


        // Anime4K pipelines use rgba16float textures.
        const sourceTexture = device.createTexture({
            label: 'Anime4K source image',
            size: {
                width,
                height,
                depthOrArrayLayers: 1
            },
            format: 'rgba16float',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT
        });

        let anime4kPipeline = null;
        const texturesToDestroy = new Set([
            sourceTexture
        ]);
        // Collect intermediate textures allocated.
        const collectPipelineTextures = (
            pipeline,
            visited = new Set()
        ) => {
            if (!pipeline || visited.has(pipeline)) {
                return;
            }
            visited.add(pipeline);
            if (pipeline.outputTexture) {
                texturesToDestroy.add(pipeline.outputTexture);
            }
            if (Array.isArray(pipeline.pipelines)) {
                for (const child of pipeline.pipelines) {
                    collectPipelineTextures(child, visited);
                }
            }
        };


        try {
            // Upload source bitmap to GPU.
            device.queue.copyExternalImageToTexture(
                {source: imageBitmap, colorSpace: 'srgb'},
                {texture: sourceTexture},
                {width, height, depthOrArrayLayers: 1}
            );

            // Build Anime4K preset.
            anime4kPipeline = new Anime4KPipeline({
                device,
                inputTexture: sourceTexture,
                nativeDimensions: {
                    width,
                    height
                },
                targetDimensions: {
                    width: targetWidth,
                    height: targetHeight
                }
            });

            collectPipelineTextures(anime4kPipeline);

            const outputTexture =
                anime4kPipeline.getOutputTexture();

            if (!outputTexture) {
                throw new Error(
                    'Anime4K did not produce an output texture.'
                );
            }
            // Bind final Anime4K output texture.
            const bindGroup = device.createBindGroup({
                layout: outputPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: sampler},
                    {binding: 1, resource: outputTexture.createView()}
                ]
            });
            const encoder = device.createCommandEncoder({
                label: 'Anime4K still-image upscale commands'
            });


            // Execute Anime4K compute passes.
            anime4kPipeline.pass(encoder);

            // Render final output to canvas.
            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: context
                        .getCurrentTexture()
                        .createView(),
                    clearValue: {r: 0, g: 0, b: 0, a: 1},
                    loadOp: 'clear',
                    storeOp: 'store'
                }]
            });

            renderPass.setPipeline(outputPipeline);
            renderPass.setBindGroup(0, bindGroup);
            renderPass.draw(6);
            renderPass.end();

            device.queue.submit([
                encoder.finish()
            ]);

            await device.queue.onSubmittedWorkDone();

            // return await this._canvasToBlobUrl();
            return await this._transferCanvas(targetWidth, targetHeight);

        } finally {
            // Wait for GPU work before releasing textures.
            try {
                await device.queue.onSubmittedWorkDone();
            } catch (_) {
                // Device may have been lost.
            }

            for (const texture of texturesToDestroy) {
                try {
                    texture.destroy();
                } catch (_) {
                    // Ignore already released textures.
                }
            }
        }
    }

    async _executeStandardCanvas(
        imageBitmap,
        targetWidth,
        targetHeight
    ) {
        this.canvas.width = targetWidth;
        this.canvas.height = targetHeight;

        const ctx = this.canvas.getContext('2d');

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        ctx.drawImage(imageBitmap, 0, 0, targetWidth, targetHeight);

        // return await this._canvasToBlobUrl();
        return await this._transferCanvas(targetWidth, targetHeight);
    }

    async _transferCanvas(width, height, canvas = this.canvas) {
        const outputCanvas = document.createElement('canvas');
        outputCanvas.width = width;
        outputCanvas.height = height;

        const ctx = outputCanvas.getContext('2d');
        ctx.drawImage(
            canvas,
            0,
            0,
            width,
            height
        );

        return outputCanvas;
    }

    async destroy() {
        this.gpuAdapter = null;
        this.gpuDevice = null;

        // Drop the ONNX worker (the bridge iframe and the worker inside it).
        const o = this.onnx;
        if (o.bridge) { o.bridge.dispose(); o.bridge = null; }
        o.state = 'idle';

        if (this.currentEngine) {
            if (typeof this.currentEngine.terminate === 'function') this.currentEngine.terminate();
            if (this.currentEngine.device) {
                try { this.currentEngine.context?.unconfigure(); } catch (_) { /* already unconfigured */ }
                this.currentEngine.device.destroy();
            }
        }
        this.currentEngine = null;
    }
}

window.EngineManager = EngineManager;
