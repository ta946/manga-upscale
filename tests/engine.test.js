const assert = require('assert');
const load = require('./load');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function world(bridgeFactory, settingsOver = {}) {
  const logs = [];
  const fakeCanvas = () => ({ width: 0, height: 0, getContext: () => ({ transferFromImageBitmap(b) { this.got = b; }, drawImage() {} }), remove() {} });
  const sb = load(['content/log.js'], {
    chrome: { runtime: { getURL: (p) => 'chrome-extension://x/' + p } },
    document: { createElement: fakeCanvas },
    fetch: async () => ({ ok: true, json: async () => ({ models: [{ path: 'models/m.onnx' }] }) }),
    createImageBitmap: async (src, o) => ({ width: o?.resizeWidth ?? src.width, height: o?.resizeHeight ?? src.height, close() {} }),
  });
  sb.console = { log: (...a) => logs.push(['info', a.join(' ')]), warn: (...a) => logs.push(['warn', a.join(' ')]), error: (...a) => logs.push(['error', a.join(' ')]), debug() {} };
  sb.MU.info = (...a) => logs.push(['info', a.join(' ')]); sb.MU.warn = (...a) => logs.push(['warn', a.join(' ')]); sb.MU.error = (...a) => logs.push(['error', a.join(' ')]);
  sb.OnnxBridge = bridgeFactory(sb);
  const fs = require('fs'), vm = require('vm');
  // setTimeout inside the VM is real; shrink recovery backoff by patching constants for speed
  let code = fs.readFileSync(require('path').join(__dirname, '../content/engine-manager.js'), 'utf8')
    .replace('[0, 1000, 3000]', '[0, 5, 5]');
  vm.runInContext(code, sb);
  sb.settings = { maxOutputWidth: 1500, oversizeMode: 'downscale', onnxGpuIO: true, ...settingsOver };
  const em = new sb.EngineManager({ settings: sb.settings });
  return { em, sb, logs };
}

// Fake bridge whose behaviour is scripted per test.
function bridgeClass(script) {
  return () => class FakeBridge {
    constructor() { script.instances.push(this); this.disposed = false; this.limits = script.limits; }
    async start(opts) { script.starts.push(opts.scope); if (script.startFail?.(opts, script.starts.length)) throw new Error(script.startFail(opts, script.starts.length)); this.gpuIO = true; return { gpuIO: true, limits: script.limits }; }
    async run(bitmap, plan, o) { script.plans.push(plan); script.runs++; const r = await script.onRun?.(this, script.runs, plan); if (r instanceof Error) throw r; await sleep(1);
      return { bitmap: { width: plan.targetW, height: plan.targetH, close() {} }, width: plan.targetW, height: plan.targetH, timings: { gpuIO: true, workerMs: 1, prepMs: 0, runMs: 1, postMs: 0, tiles: plan.tile ? 9 : 1 } }; }
    dispose() { this.disposed = true; }
  };
}
const newScript = (o = {}) => ({ instances: [], starts: [], plans: [], runs: 0, limits: { maxTexture: 8192, pixelCap: 5e6 }, ...o });
const img = (w, h) => ({ width: w, height: h, close() {} });
const crash = (m = 'ONNX worker timed out.') => Object.assign(new Error(m), { gpuFatal: true });

(async () => {
  // ---- planning --------------------------------------------------------------
  {
    const { em } = world(bridgeClass(newScript()));
    em.onnx.gpuIO = true;
    const L = { pixelCap: 5e6, maxTexture: 8192 };
    const pad = (w, h) => Math.ceil(w / 64) * 64 * Math.ceil(h / 64) * 64;
    let p = em.planONNX(1200, 692, 2, L);
    assert.deepStrictEqual([p.inW, p.inH, p.targetW, p.targetH, p.mode], [750, 432, 1500, 865, 'single'], 'identical to the old behaviour for your 1200x692 page: the 1500px OUTPUT cap is what shrinks it');
    // no width cap: the 0.75 MP pass budget is what bites. downscale keeps the TARGET: only what the model sees shrinks
    em.settings.maxOutputWidth = 0; em.settings.oversizeMode = 'downscale';
    p = em.planONNX(1200, 692, 2, L);
    assert.strictEqual(p.mode, 'downscale'); assert.deepStrictEqual([p.targetW, p.targetH], [2400, 1384], 'target untouched by the pixel budget');
    assert(pad(p.inW, p.inH) <= 750000, `padded model input fits the 0.75 MP guard (${pad(p.inW, p.inH)})`);
    em.settings.oversizeMode = 'tile';
    p = em.planONNX(1200, 692, 2, L);
    assert.strictEqual(p.mode, 'tile'); assert.deepStrictEqual([p.inW, p.inH, p.targetW, p.targetH], [1200, 692, 2400, 1384], 'tile keeps full detail');
    assert.strictEqual(p.tile.size, 704, 'tile edge is a multiple of 64 within the 0.5 MP tile budget'); assert.strictEqual(p.tile.overlap, 32);
    p = em.planONNX(900, 700, 2, L);
    assert.strictEqual(p.mode, 'single', 'images within the guard run in one pass');
    // the device guard can only LOWER the budget
    p = em.planONNX(3000, 3000, 2, { pixelCap: 2e5, maxTexture: 8192 });
    assert.strictEqual(p.mode, 'tile'); assert(p.tile.size * p.tile.size <= 2e5);
    p = em.planONNX(800, 20000, 2, L);
    assert(/fit/.test(p.mode) && Math.max(p.inH, p.targetH) <= 8192, 'webtoon strips are shrunk to fit a GPU texture');
    em.onnx.gpuIO = false;
    p = em.planONNX(1200, 692, 2, L);
    assert.strictEqual(p.mode, 'downscale', 'tile mode needs GPU I/O; otherwise downscale');
    // consistency: same width for every page of a chapter, whatever the height
    em.onnx.gpuIO = true; em.settings.oversizeMode = 'downscale';
    for (const cap of [1500, 1700]) {
      em.settings.maxOutputWidth = cap;
      const plans = [1500, 1700, 2200, 3000].map(h => em.planONNX(1125, h, 2, L));
      assert.strictEqual(new Set(plans.map(x => x.targetW)).size, 1, `all pages are ${cap}px wide`); assert.strictEqual(plans[0].targetW, cap);
      assert(plans.some(x => x.mode === 'downscale'), 'some tall pages exceed the budget, yet keep the width');
    }
    console.log('ok  planning (fixed memory guard, width-stable downscale, tile sizes)');
  }
  // ---- lifecycle: lazy start, idle release, wake -------------------------------
  {
    const s = newScript(); const { em } = world(bridgeClass(s));
    await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    assert.strictEqual(em.onnx.state, 'idle'); assert.strictEqual(s.instances.length, 0, 'nothing starts until needed');
    const r = await em.execute(img(1000, 600), { modelPath: 'models/m.onnx' });
    assert.strictEqual(em.onnx.state, 'ready'); assert.strictEqual(r.info.engine, 'onnx');
    assert(em.releaseONNX('idle')); assert(s.instances[0].disposed); assert.strictEqual(em.onnx.state, 'idle');
    await em.execute(img(1000, 600), { modelPath: 'models/m.onnx' });
    assert.strictEqual(s.instances.length, 2, 'next image wakes a fresh worker'); assert.strictEqual(em.onnx.state, 'ready');
    // concurrent first jobs share one start
    const s2 = newScript(); const w2 = world(bridgeClass(s2)); await w2.em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    await Promise.all([1, 2, 3].map(() => w2.em.execute(img(900, 900), { modelPath: 'models/m.onnx' })));
    assert.strictEqual(s2.instances.length, 1, 'single-flight start');
    console.log('ok  lazy start, release, wake, single-flight');
  }
  // ---- recovery -----------------------------------------------------------------
  {
    // one crash with two jobs in flight -> exactly one restart, both jobs succeed
    const s = newScript({ onRun: (b, n) => (n <= 2 && b === s.instances[0] ? crash() : undefined) });
    const { em, logs } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    const rs = await Promise.all([em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'a' }), em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'b' })]);
    assert(rs.every(r => r.info.engine === 'onnx'), 'both jobs recovered on the new worker');
    assert.strictEqual(s.instances.length, 2, 'one restart for two simultaneous failures'); assert(s.instances[0].disposed);
    assert(logs.some(l => /recovering 1\/3/.test(l[1])));
    console.log('ok  single recovery for concurrent failures');
  }
  {
    // keeps crashing -> after 3 restarts in a window, falls back to canvas and reports why
    const s = newScript({ onRun: () => crash('device lost') });
    const { em, logs } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    const outcomes = [];
    for (let i = 0; i < 8 && !em.onnxFallbackToCanvas; i++) {
      try { const r = await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'img' + i }); outcomes.push(r.info.engine); }
      catch (e) { outcomes.push('err:' + (e.poison ? 'poison' : e.message)); }
    }
    assert(em.onnxFallbackToCanvas, 'gave up'); assert.strictEqual(em.engineType, 'canvas');
    assert(em.onnx.recoveries.length <= 3 && s.instances.length <= 4, `bounded restarts (${s.instances.length} workers)`);
    assert(/gave up/.test(em.onnx.error), em.onnx.error); assert(em.getStatus().fallback);
    assert(logs.some(l => l[0] === 'error' && /canvas fallback/.test(l[1])));
    const r = await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'later' });
    assert(/^canvas/.test(r.info.engine), 'later images still get displayed via canvas');
    // retry re-enables ONNX
    s.onRun = null; await em.retryONNX(); const r2 = await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'again' });
    assert.strictEqual(r2.info.engine, 'onnx'); assert(!em.onnxFallbackToCanvas);
    console.log('ok  bounded recovery -> fallback with reason -> retry');
  }
  {
    // an image that crashes the worker twice is flagged poison and does not use up the restart budget for good
    const s = newScript({ onRun: (b, n, plan) => (plan.targetW === 777 ? crash() : undefined) });
    const { em } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    em.settings.maxOutputWidth = 0;
    let err; try { await em.execute(img(777 / 2, 400), { modelPath: 'models/m.onnx', key: 'bad.webp' }); } catch (e) { err = e; }
    assert(err?.poison, 'poison image rejected: ' + err?.message);
    assert(!em.onnxFallbackToCanvas, 'one bad image does not disable ONNX');
    const ok = await em.execute(img(500, 500), { modelPath: 'models/m.onnx', key: 'good.webp' });
    assert.strictEqual(ok.info.engine, 'onnx');
    console.log('ok  poison image isolated');
  }
  // ---- start failures --------------------------------------------------------------
  {
    const s = newScript({ startFail: () => 'WebGPU is unavailable inside the extension worker.' });
    const { em } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    const r = await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'x' });
    assert.strictEqual(s.starts.length, 1, 'permanent errors are not retried'); assert.strictEqual(r.info.engine, 'canvas (fallback)');
    const s2 = newScript({ startFail: (o, n) => (n < 3 ? 'ONNX bridge did not respond.' : null) });
    const w2 = world(bridgeClass(s2)); await w2.em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    const r2 = await w2.em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'y' });
    assert.strictEqual(r2.info.engine, 'onnx', 'transient start failures are retried'); assert.strictEqual(s2.starts.length, 3);
    console.log('ok  permanent vs transient start failures');
  }
  // ---- concurrent jobs survive a sibling's crash instead of failing with "bridge was disposed" ----------
  {
    const s = newScript({ onRun: (b, n) => {
      if (b !== s.instances[0]) return undefined;
      if (n === 1) return crash('device lost');
      return (async () => { while (!b.disposed) await sleep(1); return new Error('ONNX bridge was disposed.'); })();
    } });
    const { em } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    const rs = await Promise.all([em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'a' }), em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'b' })]);
    assert(rs.every(r => r.info.engine === 'onnx'), 'the collateral job was retried on the new worker, not failed');
    assert.strictEqual(s.instances.length, 2); assert.strictEqual(em.onnx.recoveries.length, 1, 'still just one restart');
    assert(!em.onnx.failCounts.has('b'), 'the collateral job is not blamed');
    console.log('ok  in-flight sibling retried after a restart');
  }
  // ---- forceDownscale describes the OLD worker; a restart clears it ---------------------------------------
  {
    const s = newScript(); const { em } = world(bridgeClass(s)); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'a' });
    em.forceDownscale = true; em.releaseONNX('idle');
    await em.execute(img(900, 900), { modelPath: 'models/m.onnx', key: 'b' });
    assert.strictEqual(em.forceDownscale, false, 'tiling is available again after the worker restarts');
    console.log('ok  forceDownscale cleared on restart');
  }
  // ---- NEEDS_GPU_IO retry downscales ------------------------------------------------------
  {
    const s = newScript({ onRun: (b, n) => (n === 1 ? Object.assign(new Error('Tiling needs WebGPU I/O'), { code: 'NEEDS_GPU_IO' }) : undefined) });
    const { em } = world(bridgeClass(s), { oversizeMode: 'tile', maxOutputWidth: 0 }); await em.switchEngine('onnx', { modelPath: 'models/m.onnx' });
    em.onnx.gpuIO = true;
    const r = await em.execute(img(1200, 900), { modelPath: 'models/m.onnx', key: 'big' });
    assert(s.plans[0].tile && !s.plans[1].tile, 'second attempt is a plain downscale'); assert.strictEqual(r.info.engine, 'onnx');
    console.log('ok  tiling impossible -> automatic downscale');
  }
  console.log('engine tests passed');
})().catch(e => { console.error('FAIL', e); process.exit(1); });
