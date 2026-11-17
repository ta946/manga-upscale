/*
 * ONNX Runtime WebGPU worker (extension origin).
 *
 * GPU-resident pipeline. Pixels never touch the CPU between decode and display:
 *
 *   ImageBitmap -> rgba8 texture                       (copyExternalImageToTexture, once per image)
 *   per tile:  compute shader -> planar f32 NCHW buffer -> ORT -> f32 NCHW GPUBuffer
 *              render pass (bilinear resample, scissored to the tile's core) -> accum texture
 *   finish:    accum -> CPU readback -> ImageBitmap          (default: survives a GPU reset)
 *              or accum -> OffscreenCanvas(webgpu) -> transferToImageBitmap  (cpuOutput:false, fastest)
 *
 * One code path serves plain (1 tile = whole image) and tiled runs. The final
 * resample to the exact target size happens in the composite shader, so there is
 * no separate resize pass. If GPU I/O fails the worker permanently drops to a
 * single-pass CPU path (tiling then reports NEEDS_GPU_IO so the caller can
 * downscale instead).
 *
 * All runs are serialised, since an ORT session cannot run concurrently.
 *
 * Messages (per port):
 *   <- init    { modelPath, scale, gpuIO, cpuOutput, warmup:[w,h] }
 *   -> ready   { gpuIO, limits:{maxTexture,pixelCap}, inputNames, outputNames, reused }
 *   -> initError { error }
 *   <- run     { id, bitmap, plan:{ scale, targetW, targetH, tile:null|{size,overlap} } }
 *   -> result  { id, width, height, bitmap, timings:{prepMs,runMs,postMs,workerMs,tiles,gpuIO} }
 *   -> error   { id, error, code, gpuFatal }
 *   <- cancel  { id }       <- dispose {}
 *   -> deviceLost { message }
 */

let ortApi = null;   // NOT named `ort`: ort.webgpu.min.js declares a global `ort`, which would collide
let session = null;
let sessionKey = null;
let sessionPromise = null;
let device = null;
let deviceLost = false;
let useGpuIO = false;
let cpuOutput = true;      // return CPU-backed bitmaps: a GPU process reset blanks GPU-backed ones
let gpuIO = null;
let limits = { maxTexture: 8192, pixelCap: 174762 };
let runChain = Promise.resolve();   // ORT sessions cannot run concurrently: serialise.

const connections = new Set();
const cancelled = new Set();
let connectionCounter = 0;

const MODEL_PATH_RE = /^models\/[\w.\-]+\.onnx$/;
const ACTIVATION_BYTES_PER_PIXEL = 768;   // conservative: 192 channels x f32 (RRDB growth concat)

const coded = (message, code) => Object.assign(new Error(message), { code });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Errors thrown while a device is dying often do not mention it; give device.lost a moment to settle. */
async function deviceGone() {
    if (deviceLost) return true;
    if (!device?.lost) return false;
    return Promise.race([device.lost.then(() => true), sleep(150).then(() => false)]);
}
const now = () => performance.now();

function isFatalGpuError(error) {
    return /external Instance reference|device.*lost|WebGPU.*lost|GPUDeviceLost/i
        .test(String(error?.message || error));
}

/* ------------------------------------------------------------------ */
/* Connections                                                          */
/* ------------------------------------------------------------------ */

self.onmessage = (event) => {
    if (event.data?.type !== 'connect' || !event.ports?.[0]) return;
    const conn = { port: event.ports[0], n: ++connectionCounter };
    connections.add(conn);
    conn.port.onmessage = (e) => onPortMessage(conn, e.data || {});
    conn.port.postMessage({ type: 'connected' });
};

function onPortMessage(conn, message) {
    switch (message.type) {
        case 'init':
            ensureSession(message)
                .then((info) => conn.port.postMessage({ type: 'ready', ...info }))
                .catch((error) => conn.port.postMessage({ type: 'initError', error: String(error?.stack || error) }));
            break;
        case 'run':
            runChain = runChain.then(() => handleRun(conn, message));
            break;
        case 'cancel':
            cancelled.add(`${conn.n}:${message.id}`);
            break;
        case 'dispose':
            // Lifetime is owned by whoever created the worker (the bridge iframe), never by
            // the worker itself.
            connections.delete(conn);
            try { conn.port.close(); } catch (_) {}
            break;
    }
}

function broadcast(message) {
    for (const conn of connections) conn.port.postMessage(message);
}

function releaseSession() {
    try { gpuIO?.destroy(); } catch (_) {}
    try { session?.release?.(); } catch (_) {}
    session = null; gpuIO = null; sessionKey = null; useGpuIO = false;
}

/* ------------------------------------------------------------------ */
/* Session lifecycle (idempotent per model, so extra tabs reuse it)     */
/* ------------------------------------------------------------------ */

function ensureSession(options) {
    const key = `${options.modelPath}|${options.gpuIO === false ? 0 : 1}|${options.scale}`;
    if (sessionPromise && sessionKey === key) {
        return sessionPromise.then((info) => ({ ...info, reused: true }));
    }
    const previous = sessionPromise;
    sessionKey = key;
    sessionPromise = (async () => {
        if (previous) {
            await previous.catch(() => {});
            await runChain;           // let in-flight runs finish before swapping models
            releaseSession();
            sessionKey = key;
        }
        return createSession(options);
    })();
    sessionPromise.catch(() => { if (sessionKey === key) sessionKey = null; });
    return sessionPromise.then((info) => ({ ...info, reused: false }));
}

async function createSession({ modelPath, scale, gpuIO: wantGpuIO = true, cpuOutput: wantCpuOutput = true, warmup }) {
    cpuOutput = wantCpuOutput !== false;
    if (!MODEL_PATH_RE.test(String(modelPath))) {
        throw new Error(`Refusing to load unexpected model path: ${modelPath}`);
    }
    if (!self.navigator?.gpu) throw new Error('WebGPU is unavailable inside the extension worker.');

    const extensionRoot = new URL('../', self.location.href).href;
    const distRoot = extensionRoot + 'node_modules/onnxruntime-web/dist/';

    if (!ortApi) {
        importScripts(distRoot + 'ort.webgpu.min.js');
        ortApi = self.ort;
        if (!ortApi?.InferenceSession) throw new Error('ONNX Runtime bundle did not expose the ort API.');
        if (!/Windows/i.test(self.navigator.userAgent || '')) ortApi.env.webgpu.powerPreference = 'high-performance';
        ortApi.env.wasm.wasmPaths = distRoot;
        ortApi.env.wasm.numThreads = 1;
        ortApi.env.wasm.proxy = false;
    }

    const sessionOptions = {
        executionProviders: [{ name: 'webgpu', preferredLayout: 'NHWC' }],
        executionMode: 'sequential',
        graphOptimizationLevel: 'all',
        enableGraphCapture: false,
        enableCpuMemArena: false,
        enableMemPattern: false,
        logSeverityLevel: 3
    };
    if (wantGpuIO) sessionOptions.preferredOutputLocation = 'gpu-buffer';
    session = await ortApi.InferenceSession.create(extensionRoot + modelPath, sessionOptions);

    // `env.webgpu.device` is a Promise in recent ORT builds, a plain object in older ones.
    try { device = await ortApi.env.webgpu.device; } catch (_) { device = null; }
    if (device?.lost && !device.__muWatched) {
        device.__muWatched = true;
        device.lost.then((info) => {
            deviceLost = true;
            broadcast({ type: 'deviceLost', message: info?.message || info?.reason || 'WebGPU device lost' });
        });
    }
    deviceLost = false;

    const lim = device?.limits;
    limits = {
        maxTexture: lim?.maxTextureDimension2D ?? 8192,
        pixelCap: Math.floor(
            Math.min(lim?.maxStorageBufferBindingSize ?? 134217728, lim?.maxBufferSize ?? 268435456) /
            ACTIVATION_BYTES_PER_PIXEL
        )
    };

    useGpuIO = false;
    if (wantGpuIO && device && typeof device.createBuffer === 'function' &&
        typeof ortApi.Tensor?.fromGpuBuffer === 'function') {
        try {
            gpuIO = await createGpuIO(device);
            useGpuIO = true;
        } catch (error) {
            console.warn('[MU worker] GPU I/O unavailable, using CPU path:', error);
            gpuIO = null;
        }
    }

    // Warm up: compiles ORT's and our pipelines so the first real page does not.
    // Faults in the GPU path surface here and degrade cleanly before any page runs.
    try {
        const [ww, wh] = Array.isArray(warmup) ? warmup : [256, 256];
        const bitmap = await createImageBitmap(new ImageData(ww, wh));
        const out = await processBitmap(bitmap, { scale, targetW: ww * scale, targetH: wh * scale, tile: null }, null);
        out.bitmap.close?.();
        bitmap.close?.();
    } catch (error) {
        console.warn('[MU worker] Warmup failed (non-fatal):', error);
    }

    return { gpuIO: useGpuIO, limits, inputNames: session.inputNames, outputNames: session.outputNames };
}

/* ------------------------------------------------------------------ */
/* WGSL                                                                 */
/* ------------------------------------------------------------------ */

const PACK_WGSL = /* wgsl */ `
struct P { ox : i32, oy : i32, tw : u32, th : u32 };
@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> p : P;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= p.tw || gid.y >= p.th) { return; }
    let d = vec2<i32>(textureDimensions(srcTex));
    // Clamp-to-edge: tiles that hang over the image border see replicated edge pixels.
    let c = clamp(vec2<i32>(p.ox + i32(gid.x), p.oy + i32(gid.y)), vec2<i32>(0, 0), d - vec2<i32>(1, 1));
    let v = textureLoad(srcTex, c, 0);
    let plane = p.tw * p.th;
    let i = gid.y * p.tw + gid.x;
    dst[i] = v.r;
    dst[plane + i] = v.g;
    dst[2u * plane + i] = v.b;
}
`;

const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}
`;

const COMPOSITE_WGSL = FULLSCREEN_VS + /* wgsl */ `
struct U {
    srcDims    : vec2<u32>,   // tile output buffer size
    pad        : vec2<u32>,
    mapScale   : vec2<f32>,   // full model-output px per target px
    tileOrigin : vec2<f32>,   // where this tile buffer's (0,0) sits in full model-output space
};
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<uniform> u : U;

fn fetch(x : i32, y : i32) -> vec3<f32> {
    let cx = clamp(x, 0, i32(u.srcDims.x) - 1);
    let cy = clamp(y, 0, i32(u.srcDims.y) - 1);
    let plane = u.srcDims.x * u.srcDims.y;
    let i = u32(cy) * u.srcDims.x + u32(cx);
    return vec3<f32>(src[i], src[plane + i], src[2u * plane + i]);
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    // Texel-centre convention: integer coordinates are texel centres. With
    // mapScale == 1 this lands exactly on texels (plain copy).
    let g = pos.xy * u.mapScale - vec2<f32>(0.5, 0.5) - u.tileOrigin;
    let b = floor(g);
    let f = g - b;
    let x0 = i32(b.x);
    let y0 = i32(b.y);
    let top = mix(fetch(x0, y0), fetch(x0 + 1, y0), f.x);
    let bot = mix(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), f.x);
    // The unorm render target clamps to [0,1].
    return vec4<f32>(mix(top, bot, f.y), 1.0);
}
`;

const BLIT_WGSL = FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var t : texture_2d<f32>;
@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(t, vec2<i32>(pos.xy), 0);
}
`;

/* ------------------------------------------------------------------ */
/* GPU I/O                                                              */
/* ------------------------------------------------------------------ */

async function createGpuIO(dev) {
    const canvasFormat = self.navigator.gpu.getPreferredCanvasFormat();
    const ACCUM_FORMAT = 'rgba8unorm';

    const [packPipeline, compositePipeline, blitPipeline] = await Promise.all([
        dev.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: dev.createShaderModule({ code: PACK_WGSL }), entryPoint: 'main' }
        }),
        (() => {
            const module = dev.createShaderModule({ code: COMPOSITE_WGSL });
            return dev.createRenderPipelineAsync({
                layout: 'auto',
                vertex: { module, entryPoint: 'vs' },
                fragment: { module, entryPoint: 'fs', targets: [{ format: ACCUM_FORMAT }] },
                primitive: { topology: 'triangle-list' }
            });
        })(),
        (() => {
            const module = dev.createShaderModule({ code: BLIT_WGSL });
            return dev.createRenderPipelineAsync({
                layout: 'auto',
                vertex: { module, entryPoint: 'vs' },
                fragment: { module, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
                primitive: { topology: 'triangle-list' }
            });
        })()
    ]);

    const packUniform = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compositeUniform = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const compositeData = new DataView(new ArrayBuffer(32));

    // Per-size caches; consecutive pages are very often identical in size.
    const st = {
        srcTex: null, srcKey: '',
        inputBuffer: null, inputKey: '',
        accum: null, accumKey: '',
        readback: null, readbackKey: '',
        canvas: null, context: null, canvasKey: ''
    };

    // WebGPU validation failures do not throw. Scope them so a bad binding becomes
    // an exception (and a CPU fallback) instead of a silently wrong image.
    async function guarded(fn) {
        dev.pushErrorScope('validation');
        let value;
        let thrown = null;
        try { value = fn(); } catch (error) { thrown = error; }
        const validationError = await dev.popErrorScope();
        if (thrown || validationError) {
            value?.close?.();
            throw thrown || new Error(`WebGPU validation error: ${validationError.message}`);
        }
        return value;
    }

    return {
        /** Upload the source once and size the accumulation texture. */
        beginImage(bitmap, targetW, targetH) {
            if (Math.max(targetW, targetH, bitmap.width, bitmap.height) > limits.maxTexture) {
                throw coded(`Image exceeds the max texture size (${limits.maxTexture}px).`, 'TOO_LARGE');
            }
            return guarded(() => {
                const sKey = `${bitmap.width}x${bitmap.height}`;
                if (st.srcKey !== sKey) {
                    st.srcTex?.destroy();
                    st.srcTex = dev.createTexture({
                        size: [bitmap.width, bitmap.height],
                        format: 'rgba8unorm',
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
                               GPUTextureUsage.RENDER_ATTACHMENT   // required by copyExternalImageToTexture
                    });
                    st.srcKey = sKey;
                }
                dev.queue.copyExternalImageToTexture({ source: bitmap }, { texture: st.srcTex }, [bitmap.width, bitmap.height]);

                const aKey = `${targetW}x${targetH}`;
                if (st.accumKey !== aKey) {
                    st.accum?.destroy();
                    st.accum = dev.createTexture({
                        size: [targetW, targetH],
                        format: ACCUM_FORMAT,
                        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING |
                               GPUTextureUsage.COPY_SRC
                    });
                    st.accumKey = aKey;
                }
            });
        },

        /** Source rect -> float32 [1,3,th,tw] ORT tensor living on the GPU. */
        async packTile(ox, oy, tw, th) {
            await guarded(() => {
                const key = `${tw}x${th}`;
                if (st.inputKey !== key) {
                    st.inputBuffer?.destroy();
                    st.inputBuffer = dev.createBuffer({
                        size: tw * th * 3 * 4,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                    });
                    st.inputKey = key;
                }
                dev.queue.writeBuffer(packUniform, 0, new Int32Array([ox, oy, tw, th]));
                const bindGroup = dev.createBindGroup({
                    layout: packPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: st.srcTex.createView() },
                        { binding: 1, resource: { buffer: st.inputBuffer } },
                        { binding: 2, resource: { buffer: packUniform } }
                    ]
                });
                const encoder = dev.createCommandEncoder();
                const pass = encoder.beginComputePass();
                pass.setPipeline(packPipeline);
                pass.setBindGroup(0, bindGroup);
                pass.dispatchWorkgroups(Math.ceil(tw / 16), Math.ceil(th / 16));
                pass.end();
                dev.queue.submit([encoder.finish()]);
            });
            return ortApi.Tensor.fromGpuBuffer(st.inputBuffer, { dataType: 'float32', dims: [1, 3, th, tw] });
        },

        /** Resample one tile's output into its core rectangle of the accumulation texture. */
        composite(gpuBuffer, t) {
            return guarded(() => {
                compositeData.setUint32(0, t.outW, true);
                compositeData.setUint32(4, t.outH, true);
                compositeData.setFloat32(16, t.mapX, true);
                compositeData.setFloat32(20, t.mapY, true);
                compositeData.setFloat32(24, t.originX, true);
                compositeData.setFloat32(28, t.originY, true);
                dev.queue.writeBuffer(compositeUniform, 0, compositeData.buffer);

                const bindGroup = dev.createBindGroup({
                    layout: compositePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: gpuBuffer } },
                        { binding: 1, resource: { buffer: compositeUniform } }
                    ]
                });
                const encoder = dev.createCommandEncoder();
                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: st.accum.createView(),
                        loadOp: t.first ? 'clear' : 'load',
                        storeOp: 'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 1 }
                    }]
                });
                pass.setPipeline(compositePipeline);
                pass.setBindGroup(0, bindGroup);
                pass.setScissorRect(t.dst.x, t.dst.y, t.dst.w, t.dst.h);
                pass.draw(3);
                pass.end();
                dev.queue.submit([encoder.finish()]);
            });
        },

        /**
         * accum -> ImageBitmap.
         * Default: read the pixels back and build a CPU-backed bitmap. GPU-backed bitmaps die
         * with the GPU process (the page goes blank after a reset); CPU memory does not.
         */
        async finish(targetW, targetH) {
            if (!cpuOutput) return this.finishGpu(targetW, targetH);

            const bytesPerRow = Math.ceil((targetW * 4) / 256) * 256;     // WebGPU row alignment
            const key = `${bytesPerRow}x${targetH}`;
            if (st.readbackKey !== key) {
                st.readback?.destroy();
                st.readback = dev.createBuffer({
                    size: bytesPerRow * targetH,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
                });
                st.readbackKey = key;
            }
            await guarded(() => {
                const encoder = dev.createCommandEncoder();
                encoder.copyTextureToBuffer(
                    { texture: st.accum },
                    { buffer: st.readback, bytesPerRow, rowsPerImage: targetH },
                    [targetW, targetH]
                );
                dev.queue.submit([encoder.finish()]);
            });
            await st.readback.mapAsync(GPUMapMode.READ);
            const mapped = new Uint8Array(st.readback.getMappedRange());
            const rgba = new Uint8ClampedArray(targetW * targetH * 4);
            const rowBytes = targetW * 4;
            if (bytesPerRow === rowBytes) rgba.set(mapped.subarray(0, rgba.length));
            else for (let y = 0; y < targetH; y++) rgba.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + rowBytes), y * rowBytes);
            st.readback.unmap();
            return createImageBitmap(new ImageData(rgba, targetW, targetH));
        },

        /** accum -> canvas -> GPU-backed ImageBitmap (same task: the canvas texture expires after it). */
        finishGpu(targetW, targetH) {
            return guarded(() => {
                const key = `${targetW}x${targetH}`;
                if (st.canvasKey !== key) {
                    st.canvas = new OffscreenCanvas(targetW, targetH);
                    st.context = st.canvas.getContext('webgpu');
                    st.context.configure({ device: dev, format: canvasFormat, alphaMode: 'opaque' });
                    st.canvasKey = key;
                }
                const bindGroup = dev.createBindGroup({
                    layout: blitPipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: st.accum.createView() }]
                });
                const encoder = dev.createCommandEncoder();
                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: st.context.getCurrentTexture().createView(),
                        loadOp: 'clear', storeOp: 'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 1 }
                    }]
                });
                pass.setPipeline(blitPipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(3);
                pass.end();
                dev.queue.submit([encoder.finish()]);
                return st.canvas.transferToImageBitmap();
            });
        },

        destroy() {
            st.srcTex?.destroy(); st.inputBuffer?.destroy(); st.accum?.destroy(); st.readback?.destroy();
            packUniform.destroy(); compositeUniform.destroy();
            st.srcTex = st.inputBuffer = st.accum = st.canvas = st.context = null;
        }
    };
}

function disableGpuIO(stage, error) {
    console.warn(`[MU worker] GPU I/O failed at "${stage}"; using the CPU path.`, error);
    if (isFatalGpuError(error)) throw error;   // device is gone: nothing to fall back to
    useGpuIO = false;
}

const SHAPE_UNIT = 64;
const roundUp = (n) => Math.ceil(n / SHAPE_UNIT) * SHAPE_UNIT;

/**
 * Split [0,n) into tiles. Every model input length is a multiple of 64 (the tail is
 * filled by clamp-to-edge sampling and never composited), so across a chapter the
 * model sees a handful of repeating shapes. That keeps ORT's size-keyed GPU buffer
 * pool and its shader cache bounded, which is what keeps long sessions from running
 * the GPU out of memory. Tiles are equalised (no sliver of a last tile).
 */
function planAxis(n, T, overlap) {
    if (!T || n <= T) return [{ in0: 0, len: roundUp(n), c0: 0, c1: n }];
    const coreMax = T - 2 * overlap;
    const count = Math.ceil(n / coreMax);
    const core = Math.ceil(n / count);
    const len = roundUp(core + 2 * overlap);       // <= T because T is a multiple of 64
    const tiles = [];
    for (let i = 0; i < count; i++) {
        const c0 = i * core;
        if (c0 >= n) break;
        tiles.push({ in0: c0 - overlap, len, c0, c1: Math.min(n, c0 + core) });
    }
    return tiles;
}

/* ------------------------------------------------------------------ */
/* CPU fallback (single pass)                                           */
/* ------------------------------------------------------------------ */

function buildCpuInputTensor(bitmap) {
    const w = bitmap.width, h = bitmap.height;
    const ctx = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const plane = w * h;
    const chw = new Float32Array(plane * 3);
    const k = 1 / 255;
    for (let i = 0, s = 0; i < plane; i++, s += 4) chw[i] = rgba[s] * k;
    for (let i = 0, s = 1; i < plane; i++, s += 4) chw[plane + i] = rgba[s] * k;
    for (let i = 0, s = 2; i < plane; i++, s += 4) chw[2 * plane + i] = rgba[s] * k;
    return new ortApi.Tensor('float32', chw, [1, 3, h, w]);
}

async function cpuOutputBitmap(tensor, w, h) {
    if (tensor.type !== 'float32') throw new Error(`Unsupported model output type "${tensor.type}".`);
    const data = await tensor.getData();       // works for cpu and gpu-buffer tensors
    const plane = w * h;
    const rgba = new Uint8ClampedArray(plane * 4);   // clamps + rounds for us
    for (let i = 0, d = 0; i < plane; i++, d += 4) rgba[d] = data[i] * 255;
    for (let i = 0, d = 1; i < plane; i++, d += 4) rgba[d] = data[plane + i] * 255;
    for (let i = 0, d = 2; i < plane; i++, d += 4) rgba[d] = data[2 * plane + i] * 255;
    for (let d = 3; d < rgba.length; d += 4) rgba[d] = 255;
    return createImageBitmap(new ImageData(rgba, w, h));
}

async function processCpu(bitmap) {
    const timings = { prepMs: 0, runMs: 0, postMs: 0, tiles: 1, gpuIO: false };
    let input = null, results = null;
    try {
        let t = now();
        input = buildCpuInputTensor(bitmap);
        timings.prepMs = now() - t;
        t = now();
        results = await session.run({ [session.inputNames[0]]: input });
        timings.runMs = now() - t;
        t = now();
        const out = results[session.outputNames[0]];
        const [, c, h, w] = out.dims;
        if (c !== 3) throw new Error(`Unexpected model output shape: [${out.dims.join(', ')}]`);
        const outBitmap = await cpuOutputBitmap(out, w, h);
        timings.postMs = now() - t;
        return { bitmap: outBitmap, width: w, height: h, timings };
    } finally {
        input?.dispose?.();
        if (results) for (const tensor of Object.values(results)) tensor?.dispose?.();
    }
}

/* ------------------------------------------------------------------ */
/* GPU path (plain and tiled)                                           */
/* ------------------------------------------------------------------ */

async function processGpu(bitmap, plan, key) {
    const s = plan.scale;
    const w = bitmap.width, h = bitmap.height;
    const { targetW, targetH } = plan;
    const T = plan.tile ? plan.tile.size : null;
    const overlap = plan.tile ? plan.tile.overlap : 0;
    const xs = planAxis(w, T ?? w, overlap);
    const ys = planAxis(h, T ?? h, overlap);
    const timings = { prepMs: 0, runMs: 0, postMs: 0, tiles: xs.length * ys.length, gpuIO: true };

    let t = now();
    await gpuIO.beginImage(bitmap, targetW, targetH);
    timings.prepMs += now() - t;

    const mapX = (w * s) / targetW;
    const mapY = (h * s) / targetH;
    let first = true;

    for (const ty of ys) {
        for (const tx of xs) {
            if (key && cancelled.has(key)) throw coded('Cancelled', 'CANCELLED');
            let input = null, results = null;
            try {
                t = now();
                input = await gpuIO.packTile(tx.in0, ty.in0, tx.len, ty.len);
                timings.prepMs += now() - t;

                t = now();
                results = await session.run({ [session.inputNames[0]]: input });
                if (deviceLost) throw new Error('WebGPU device was lost during ONNX inference.');
                timings.runMs += now() - t;

                t = now();
                const out = results[session.outputNames[0]];
                const dims = out.dims;
                if (out.location !== 'gpu-buffer' || out.type !== 'float32') {
                    throw coded(`Output not usable on GPU (${out.location}/${out.type}).`, 'NO_GPU_OUTPUT');
                }
                if (dims.length !== 4 || dims[1] !== 3 || dims[2] !== ty.len * s || dims[3] !== tx.len * s) {
                    throw coded(
                        `Model output [${dims.join(', ')}] does not match ${s}x scale of ${tx.len}x${ty.len}.`,
                        'SCALE_MISMATCH'
                    );
                }
                // Core rectangle in target pixels. The same rounding function is used on
                // both sides of every seam, so tiles never overlap or leave gaps.
                const x0 = Math.round(tx.c0 * targetW / w), x1 = Math.round(tx.c1 * targetW / w);
                const y0 = Math.round(ty.c0 * targetH / h), y1 = Math.round(ty.c1 * targetH / h);
                if (x1 > x0 && y1 > y0) {
                    await gpuIO.composite(out.gpuBuffer, {
                        outW: tx.len * s, outH: ty.len * s,
                        mapX, mapY, originX: tx.in0 * s, originY: ty.in0 * s,
                        dst: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
                        first
                    });
                    first = false;
                }
                timings.postMs += now() - t;
            } finally {
                input?.dispose?.();
                if (results) for (const tensor of Object.values(results)) tensor?.dispose?.();
            }
        }
    }

    if (cpuOutput) {
        // run() resolves when work is queued, not finished. Wait here so "run" is the real compute
        // time and "post" is only composite + readback (the readback would block on it anyway).
        t = now();
        await device.queue.onSubmittedWorkDone();
        timings.runMs += now() - t;
    }
    t = now();
    const outBitmap = await gpuIO.finish(targetW, targetH);
    timings.postMs += now() - t;
    return { bitmap: outBitmap, width: targetW, height: targetH, timings };
}

async function processBitmap(bitmap, plan, key) {
    if (!session || deviceLost) {
        throw new Error('ONNX WebGPU session is unavailable because its device has been lost.');
    }
    if (useGpuIO) {
        try {
            return await processGpu(bitmap, plan, key);
        } catch (error) {
            if (error.code === 'CANCELLED' || error.code === 'TOO_LARGE' || error.code === 'SCALE_MISMATCH') throw error;
            if (isFatalGpuError(error) || await deviceGone()) throw Object.assign(error, { fatal: true });
            disableGpuIO(error.code || 'run', error);
        }
    }
    if (plan.tile) throw coded('Tiling needs WebGPU I/O, which is unavailable.', 'NEEDS_GPU_IO');
    return processCpu(bitmap);
}

async function handleRun(conn, { id, bitmap, plan }) {
    const key = `${conn.n}:${id}`;
    try {
        await sessionPromise;                       // rejects if init failed
        if (cancelled.has(key)) throw coded('Cancelled', 'CANCELLED');
        if (!bitmap?.width || !bitmap?.height) throw new Error(`Invalid bitmap ${bitmap?.width}x${bitmap?.height}`);
        const started = now();
        const result = await processBitmap(bitmap, plan, key);
        if (cancelled.has(key)) { result.bitmap.close?.(); throw coded('Cancelled', 'CANCELLED'); }
        result.timings.workerMs = now() - started;
        conn.port.postMessage({ type: 'result', id, ...result }, [result.bitmap]);
    } catch (error) {
        conn.port.postMessage({
            type: 'error', id,
            error: String(error?.message || error),
            code: error?.code || null,
            gpuFatal: deviceLost || !!error?.fatal || isFatalGpuError(error)
        });
    } finally {
        bitmap?.close?.();
        cancelled.delete(key);
    }
}
