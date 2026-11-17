// Runs content/onnx-worker.js in a VM with fake ORT / WebGPU / canvas and checks protocol + tiling geometry.
const assert = require('assert'), vm = require('vm'), fs = require('fs'), path = require('path');
const src = fs.readFileSync(path.join(__dirname, '../content/onnx-worker.js'), 'utf8');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function ports() {
  const a = { onmessage: null, sent: [] }, b = { onmessage: null };
  a.postMessage = (m) => { a.sent.push(m); setTimeout(() => b.onmessage && b.onmessage({ data: m }), 0); };   // a = client side
  b.postMessage = (m) => { setTimeout(() => a.onmessage && a.onmessage({ data: m }), 0); };
  a.close = b.close = () => {};
  return [a, b];
}

function makeWorld({ gpu = true, failAt = null, scale = 2 } = {}) {
  const ctrl = { dying: false }; let loseDevice;
  const rec = { passes: [], uniforms: [], runs: [], sessions: 0, released: 0, copies: 0 };
  const buf = (d) => ({ size: d.size, destroy() {}, async mapAsync() {}, getMappedRange() { return new ArrayBuffer(d.size); }, unmap() {} });
  const device = {
    limits: { maxTextureDimension2D: 8192, maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 },
    lost: new Promise((res) => { loseDevice = res; }),
    createBuffer: buf, createShaderModule: () => ({}),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createRenderPipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createTexture: (d) => ({ ...d, destroy() {}, createView: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      copyTextureToBuffer() { rec.readbacks = (rec.readbacks || 0) + 1; },
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      beginRenderPass: (d) => { const p = { load: d.colorAttachments[0].loadOp }; rec.passes.push(p);
        return { setPipeline() {}, setBindGroup() {}, setScissorRect(x, y, w, h) { p.scissor = { x, y, w, h }; }, draw() {}, end() {} }; },
      finish: () => ({}),
    }),
    queue: {
      copyExternalImageToTexture() { if (ctrl.dying) { loseDevice({ message: 'gone' }); throw new Error('Invalid Texture (something odd)'); } if (failAt === 'copy') throw new Error('copy failed'); rec.copies++; },
      writeBuffer(b, off, data) {
        if (b.size === 32) { const dv = new DataView(data.buffer || data); rec.uniforms.push({ outW: dv.getUint32(0, true), outH: dv.getUint32(4, true), mapX: dv.getFloat32(16, true), mapY: dv.getFloat32(20, true), ox: dv.getFloat32(24, true), oy: dv.getFloat32(28, true) }); }
      },
      submit() {}, async onSubmittedWorkDone() { rec.waits = (rec.waits || 0) + 1; },
    },
    pushErrorScope() {}, popErrorScope: async () => null,
  };
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; this.location = 'cpu'; }
    static fromGpuBuffer(b, o) { const t = new Tensor(o.dataType, null, o.dims); t.location = 'gpu-buffer'; t.gpuBuffer = {}; return t; }
    dispose() {} async getData() { return this.data; }
  }
  let active = 0, maxActive = 0;
  const ort = {
    env: { webgpu: { device: Promise.resolve(gpu ? device : null) }, wasm: {} }, Tensor,
    InferenceSession: { create: async (url) => { rec.sessions++; rec.lastModel = url; return {
      inputNames: ['in'], outputNames: ['out'], release() { rec.released++; },
      run: async (feeds) => {
        active++; maxActive = Math.max(maxActive, active); await sleep(2);
        const [, , h, w] = feeds.in.dims; rec.runs.push([w, h]);
        const H = h * scale, W = w * scale;
        const t = new Tensor('float32', new Float32Array(H * W * 3).fill(0.5), [1, 3, H, W]);
        if (gpu) { t.location = 'gpu-buffer'; t.gpuBuffer = {}; }
        active--; return { out: t };
      } }; } },
  };
  const ctx = {
    console: { log() {}, warn() {}, error: console.error }, setTimeout, performance, URL, Promise, Float32Array, Uint8ClampedArray, Uint32Array, Int32Array, DataView, ArrayBuffer, Array, Math, Object, String, Error, Number, Set,
    GPUBufferUsage: {}, GPUTextureUsage: {}, GPUMapMode: { READ: 1 },
    importScripts: () => { ctx.self.ort = ort; },
    ImageData: class { constructor(a, b, c) { if (typeof a === 'number') { this.width = a; this.height = b; this.data = new Uint8ClampedArray(a * b * 4); } else { this.data = a; this.width = b; this.height = c; } } },
    createImageBitmap: async (x) => ({ width: x.width, height: x.height, cpu: x instanceof ctx.ImageData, close() { this.closed = true; } }),
    OffscreenCanvas: class { constructor(w, h) { this.width = w; this.height = h; }
      getContext(k) { return k === '2d' ? { drawImage() {}, getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4).fill(255) }) } : { configure() {}, getCurrentTexture: () => ({ createView: () => ({}) }) }; }
      transferToImageBitmap() { if (failAt === 'transfer') throw new Error('transfer failed'); return { width: this.width, height: this.height, gpu: true, close() {} }; } },
  };
  ctx.self = { location: { href: 'chrome-extension://x/content/onnx-worker.js' }, navigator: { gpu: { getPreferredCanvasFormat: () => 'bgra8unorm' } }, close() {}, postMessage() {} };
  vm.createContext(ctx); vm.runInContext(src, ctx);

  function connect() {
    const [client, side] = ports(); const msgs = [];
    client.onmessage = (e) => msgs.push(e.data);
    ctx.self.onmessage({ data: { type: 'connect' }, ports: [side] });
    const api = {
      msgs, client,
      send: (m) => client.postMessage(m),
      wait: async (pred, n = 600) => { for (let i = 0; i < n && !pred(); i++) await sleep(3); assert(pred(), 'timed out waiting'); },
      init: async (o = {}) => { const from = msgs.length; api.send({ type: 'init', modelPath: 'models/m.onnx', scale, gpuIO: true, warmup: [16, 16], ...o }); const isNew = () => msgs.slice(from).find(m => m.type === 'ready' || m.type === 'initError'); await api.wait(isNew); return isNew(); },
      run: async (id, w, h, plan) => { api.send({ type: 'run', id, bitmap: { width: w, height: h, close() { this.closed = true; } }, plan: { scale, ...plan } }); await api.wait(() => msgs.some(m => m.id === id && (m.type === 'result' || m.type === 'error'))); return msgs.find(m => m.id === id && (m.type === 'result' || m.type === 'error')); },
    };
    return api;
  }
  return { rec, ctrl, connect, maxActive: () => maxActive, ctx };
}

(async () => {
  // 1. plain GPU run: one tile, resampled to the exact target (even a non-2x target)
  {
    const w = makeWorld(); const c = w.connect(); const ready = await c.init();
    assert.strictEqual(ready.type, 'ready'); assert.strictEqual(ready.gpuIO, true);
    assert.strictEqual(ready.limits.maxTexture, 8192); assert(ready.limits.pixelCap > 100000);
    w.rec.passes.length = 0; w.rec.uniforms.length = 0; w.rec.readbacks = 0;
    const r = await c.run(1, 750, 398, { targetW: 1500, targetH: 797, tile: null });
    assert.strictEqual(r.type, 'result'); assert.deepStrictEqual([r.width, r.height], [1500, 797]);
    assert(r.bitmap.cpu && !r.bitmap.gpu, 'default output is CPU-backed, so a GPU reset cannot blank it');
    assert(r.timings.gpuIO && r.timings.tiles === 1);
    const u = w.rec.uniforms[0];
    assert(Math.abs(u.mapY - (398 * 2) / 797) < 1e-6 && u.mapX === 1, 'shader resamples 796 -> 797 rows instead of a separate resize pass');
    assert.strictEqual(w.rec.passes.length, 1, 'one composite pass; output is read back, no blit'); assert.strictEqual(w.rec.readbacks, 1);
    assert(w.rec.waits >= 1, 'waits for the GPU before reading back, so run/post timings are honest');
    const w2 = makeWorld(); const c2 = w2.connect(); await c2.init({ cpuOutput: false }); w2.rec.passes.length = 0;
    const r2 = await c2.run(1, 64, 64, { targetW: 128, targetH: 128, tile: null });
    assert(r2.bitmap.gpu && !r2.bitmap.cpu && w2.rec.passes.length === 2, 'cpuOutput:false keeps the zero-copy GPU path (composite + blit)');
    console.log('ok  plain run, shader resample, CPU-backed output by default');
  }
  // 2. tiling geometry: cores tile the target exactly; sampling stays inside each tile buffer
  for (const [W, H, T, ov, tw, th] of [[1125, 1500, 512, 32, 2250, 3000], [1000, 900, 400, 32, 1500, 1350], [800, 2200, 512, 32, 1600, 4400]]) {
    const w = makeWorld(); const c = w.connect(); await c.init();
    w.rec.passes.length = 0; w.rec.uniforms.length = 0; w.rec.runs.length = 0;
    const r = await c.run(2, W, H, { targetW: tw, targetH: th, tile: { size: T, overlap: ov } });
    assert.strictEqual(r.type, 'result', r.error);
    const tiles = w.rec.passes.filter(p => p.scissor);    // composite passes only
    assert.strictEqual(tiles.length, w.rec.uniforms.length);
    assert.strictEqual(tiles[0].load, 'clear'); assert(tiles.slice(1).every(p => p.load === 'load'));
    assert(tiles.every(p => p.scissor.w > 0 && p.scissor.h > 0));
    const cover = new Uint8Array(tw * th);
    tiles.forEach((p, k) => {
      const { x, y, w: cw, h: ch } = p.scissor, u = w.rec.uniforms[k];
      for (let yy = y; yy < y + ch; yy++) for (let xx = x; xx < x + cw; xx++) cover[yy * tw + xx]++;
      // sampled texel footprint [floor(g), floor(g)+1] must lie inside the tile buffer (no edge clamping artefacts) unless at the image border
      const gx0 = (x + 0.5) * u.mapX - 0.5 - u.ox, gx1 = (x + cw - 0.5) * u.mapX - 0.5 - u.ox;
      const gy0 = (y + 0.5) * u.mapY - 0.5 - u.oy, gy1 = (y + ch - 0.5) * u.mapY - 0.5 - u.oy;
      if (x > 0) assert(gx0 >= 0, `left seam samples outside tile (${gx0})`);
      if (x + cw < tw) assert(Math.floor(gx1) + 1 <= u.outW - 1, `right seam samples outside tile (${gx1} vs ${u.outW})`);
      if (y > 0) assert(gy0 >= 0, `top seam samples outside tile (${gy0})`);
      if (y + ch < th) assert(Math.floor(gy1) + 1 <= u.outH - 1, `bottom seam samples outside tile (${gy1} vs ${u.outH})`);
    });
    assert(cover.every(v => v === 1), 'every target pixel written exactly once (no gaps, no overlap)');
    const shapes = new Set(w.rec.runs.map(r => r.join('x')));
    assert(shapes.size <= 4, 'tiles share few distinct input shapes: ' + [...shapes]);
    assert(w.rec.runs.every(([a, b]) => a % 64 === 0 && b % 64 === 0), 'every model input is a multiple of 64: stable shapes keep ORT\'s buffer pool bounded: ' + [...shapes]);
    assert.strictEqual(r.timings.tiles, tiles.length);
    console.log(`ok  tiling ${W}x${H} -> ${tw}x${th}: ${tiles.length} tiles, exact cover, ${shapes.size} shape(s)`);
  }
  // 3. cancel between tiles
  {
    const w = makeWorld(); const c = w.connect(); await c.init(); w.rec.runs.length = 0;
    c.send({ type: 'run', id: 3, bitmap: { width: 1200, height: 1200, close() {} }, plan: { scale: 2, targetW: 2400, targetH: 2400, tile: { size: 320, overlap: 32 } } });
    await sleep(8); c.send({ type: 'cancel', id: 3 });
    await c.wait(() => c.msgs.some(m => m.id === 3));
    const m = c.msgs.find(m => m.id === 3);
    assert.strictEqual(m.type, 'error'); assert.strictEqual(m.code, 'CANCELLED');
    assert(w.rec.runs.length < 16, `stopped early (${w.rec.runs.length} of 16 tiles ran)`);
    console.log('ok  cancel stops a tiled run early');
  }
  // 4. GPU failure at init is caught by warmup; tiled request then asks the caller to downscale
  {
    const w = makeWorld({ failAt: 'copy' }); const c = w.connect(); const ready = await c.init();
    assert.strictEqual(ready.gpuIO, false, 'warmup detects the broken GPU path before any page runs');
    const r = await c.run(4, 64, 64, { targetW: 128, targetH: 128, tile: null });
    assert.strictEqual(r.type, 'result'); assert.strictEqual(r.timings.gpuIO, false); assert.deepStrictEqual([r.width, r.height], [128, 128]);
    const t = await c.run(5, 64, 64, { targetW: 128, targetH: 128, tile: { size: 128, overlap: 32 } });
    assert.strictEqual(t.code, 'NEEDS_GPU_IO');
    console.log('ok  GPU failure -> CPU path; tiling reports NEEDS_GPU_IO');
  }
  // 5. no GPU device at all
  { const w = makeWorld({ gpu: false }); const c = w.connect(); const ready = await c.init(); assert.strictEqual(ready.gpuIO, false);
    const r = await c.run(6, 32, 32, { targetW: 64, targetH: 64, tile: null }); assert.strictEqual(r.type, 'result'); console.log('ok  no GPU device -> CPU path'); }
  // 6. several connections to one worker reuse the session; runs are serialised; a model change swaps the session
  {
    const w = makeWorld(); const a = w.connect(), b = w.connect();
    const ra = await a.init(); const rb = await b.init();
    assert.strictEqual(w.rec.sessions, 1, 'a second connection reuses the session'); assert.strictEqual(rb.reused, true);
    await Promise.all([a.run(10, 64, 64, { targetW: 128, targetH: 128, tile: null }), b.run(10, 64, 64, { targetW: 128, targetH: 128, tile: null }), a.run(11, 64, 64, { targetW: 128, targetH: 128, tile: null })]);
    assert(a.msgs.filter(m => m.type === 'result').length === 2 && b.msgs.filter(m => m.type === 'result').length === 1, 'results go back to the right tab, even with equal ids');
    assert.strictEqual(w.maxActive(), 1, 'ORT never runs concurrently');
    await b.init({ modelPath: 'models/other.onnx' });
    assert.strictEqual(w.rec.sessions, 2); assert.strictEqual(w.rec.released, 1); assert(/other\.onnx$/.test(w.rec.lastModel));
    console.log('ok  session reuse, serialised runs, correct routing, model swap');
  }
  // 6b. errors raised while the device dies look harmless; they must be treated as device loss
  {
    const w = makeWorld(); const c = w.connect(); const ready = await c.init(); assert.strictEqual(ready.gpuIO, true);
    w.ctrl.dying = true;
    const r = await c.run(20, 64, 64, { targetW: 128, targetH: 128, tile: { size: 256, overlap: 32 } });
    assert.strictEqual(r.type, 'error'); assert.strictEqual(r.gpuFatal, true, 'reported as a lost device so the caller restarts the worker');
    assert.notStrictEqual(r.code, 'NEEDS_GPU_IO', 'GPU I/O was NOT silently disabled (that is what turned tiling off after the crash)');
    console.log('ok  dying device -> gpuFatal, GPU I/O kept');
  }
  // 7. bad model path
  { const w = makeWorld(); const c = w.connect(); const r = await c.init({ modelPath: '../../evil.onnx' }); assert.strictEqual(r.type, 'initError'); assert(/Refusing/.test(r.error)); console.log('ok  model path validation'); }
  console.log('worker tests passed');
})().catch(e => { console.error('FAIL', e); process.exit(1); });
