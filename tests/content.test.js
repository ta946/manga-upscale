const assert = require('assert'), vm = require('vm'), fs = require('fs'), path = require('path');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const read = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');

// ---- minimal fake DOM ----------------------------------------------------------
class El {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.dataset = {}; this.children = []; this.parentNode = null; this.parentElement = null; this._style = {};
    this.style = { setProperty: (k, v) => { this._style[k] = v; }, removeProperty: (k) => { delete this._style[k]; }, set cssText(v) { this._css = v; }, get cssText() { return this._css; }, set verticalAlign(v) {} };
    this.width = 0; this.height = 0; }
  get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n.tagName === 'DOC'; }
  appendChild(c) { if (c.parentNode) c.parentNode.children.splice(c.parentNode.children.indexOf(c), 1); c.parentNode = this; c.parentElement = this; this.children.push(c); return c; }
  insertBefore(c, ref) { if (c.parentNode) c.parentNode.children.splice(c.parentNode.children.indexOf(c), 1); c.parentNode = this; c.parentElement = this; this.children.splice(this.children.indexOf(ref), 0, c); return c; }
  remove() { if (this.parentNode) { this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = this.parentElement = null; } }
  querySelectorAll(sel) { const out = []; const walk = (n) => n.children.forEach(c => { if (c.tagName === sel.toUpperCase()) out.push(c); walk(c); }); walk(this); return out; }
  getContext() { return {}; }
}
class FakeCanvas extends El { constructor() { super('canvas'); } }

function setup({ ram = 'low', count = 10, widths = [] } = {}) {
  const doc = new El('doc'); const reader = new El('div'); doc.appendChild(reader);
  const imgs = Array.from({ length: count }, (_, i) => { const im = new El('img'); im.src = `https://cdn/x/${i}.webp`; im.currentSrc = im.src; im.complete = false; im.naturalWidth = 0; im.getBoundingClientRect = () => ({ bottom: (i + 1) * 1000 - scrollY, top: i * 1000 - scrollY }); reader.appendChild(im); return im; });
  let scrollY = 0;
  const listeners = {}; let onMessage; let onStorageChanged; const stored = { ramMode: ram, engine: 'onnx', enabled: true, onnxModelPath: 'models/m.onnx' };
  const em = { calls: [], engineType: 'onnx', onnx: { state: 'idle', gpuIO: true }, settings: null, executed: [],
    switchEngine(t) { this.calls.push('switch:' + t); this.engineType = t; }, prewarmONNX() { this.calls.push('prewarm'); this.onnx.state = 'ready'; },
    releaseONNX(r) { this.calls.push('release'); this.onnx.state = 'idle'; return true; }, retryONNX() { this.calls.push('retry'); },
    getStatus() { return { engine: this.engineType, fallback: false, onnx: { state: this.onnx.state } }; },
    async execute(bitmap, o) { this.executed.push(o.key); await sleep(3); this.onnx.state = 'ready'; if (o.signal?.aborted) throw new DOMException('x', 'AbortError');
      const c = new FakeCanvas(); c.width = bitmap.width * 2; c.height = bitmap.height * 2; return { canvas: c, info: { engine: 'onnx', outW: c.width, outH: c.height, timings: { workerMs: 2, prepMs: 1, runMs: 1, postMs: 0, tiles: 1, gpuIO: true } } }; } };
  const logs = [];
  const sb = {
    console: { log() {}, warn() {}, error: console.error, debug() {} }, setTimeout, clearTimeout, performance, AbortController, DOMException, Promise, Map, Set, WeakMap, Array, Math, Object, Error, Number, URL, Date, JSON,
    HTMLCanvasElement: FakeCanvas, location: { href: 'https://site/ch/1', origin: 'https://site' },
    requestAnimationFrame: (cb) => setTimeout(cb, 0), MutationObserver: class { observe() {} },
    Image: class { set src(v) { this._s = v; setTimeout(() => this.onload && this.onload(), 1); } },
    createImageBitmap: async (x) => ({ width: widths[+/(\d+)\.webp/.exec(x._s || x.src)?.[1]] || 1000, height: 1400, close() {} }),
    getComputedStyle: () => ({ verticalAlign: 'baseline' }),
    document: { querySelector: () => reader, querySelectorAll: (s) => reader.querySelectorAll(s), body: new El('body'), addEventListener: (t, f) => { listeners[t] = f; }, createElement: (t) => t === 'canvas' ? new FakeCanvas() : new El(t) },
    chrome: { storage: { sync: { get: (k, cb) => cb({ ...stored }), set() { throw new Error('window-only settings must NOT touch storage'); } }, onChanged: { addListener: (f) => { onStorageChanged = f; } } }, runtime: { onMessage: { addListener: (f) => { onMessage = f; } } } },
  };
  sb.window = sb; sb.addEventListener = (t, f) => { listeners[t] = f; };
  vm.createContext(sb);
  for (const f of ['content/log.js', 'content/settings.js', 'content/viewport.js', 'content/queue.js']) vm.runInContext(read(f), sb, { filename: f });
  sb.MU.info = (...a) => logs.push(a.join(' ')); sb.MU.warn = (...a) => logs.push('WARN ' + a.join(' '));
  sb.EngineManager = class { constructor(o) { Object.assign(em, { settings: o.settings }); return em; } };
  sb.OnnxBridge = class {};
  vm.runInContext(read('content/content.js').replace('const TEARDOWN_GRACE_MS = 2000;', 'const TEARDOWN_GRACE_MS = 40;'), sb, { filename: 'content.js' });
  return { imgs, em, logs, sb, onMessage: (m) => { let r; onMessage(m, {}, (x) => { r = x; }); return r; },
    scrollTo: async (y) => { scrollY = y; listeners.scroll(); await sleep(15); },
    settle: async (ms = 60) => sleep(ms),
    changeStorage: async (patch) => { Object.assign(stored, patch); onStorageChanged(Object.fromEntries(Object.keys(patch).map(k => [k, { newValue: patch[k] }])), 'sync'); await sleep(30); },
    reader: imgs[0].parentNode.tagName === 'DIV' ? imgs[0].parentNode : null,
    hasCanvas: (i) => imgs[i].parentNode.children.some(c => c instanceof FakeCanvas),
    visible: (i) => imgs[i]._style.visibility !== 'hidden' };
}

(async () => {
  // 1. low RAM: forward window processed, behind released on scroll, re-processed on scroll up
  {
    const t = setup({ ram: 'low' }); await t.settle();
    assert.deepStrictEqual([...new Set(t.em.executed)].sort(), ['https://cdn/x/0.webp', 'https://cdn/x/1.webp', 'https://cdn/x/2.webp'], 'initial window: current + 2 ahead');
    assert(t.em.calls.includes('prewarm'), 'auto-start warms the worker when a reader page has work');
    assert(t.hasCanvas(0) && !t.visible(0));
    await t.scrollTo(5500); await t.settle();                       // viewport on image 5 -> window [4..7]
    assert(!t.hasCanvas(0) && t.visible(0), 'image far behind: canvas freed, original visible again');
    assert(!t.hasCanvas(2) && t.visible(2));
    assert(t.hasCanvas(4) && t.hasCanvas(5) && t.hasCanvas(6) && t.hasCanvas(7), 'new window (one behind + three ahead) processed');
    const before = t.em.executed.length;
    await t.scrollTo(1500); await t.settle();                       // scroll back up: viewport on image 1 -> window [0..3]
    assert(t.hasCanvas(1) && !t.visible(1), 'released image re-enters the window and is upscaled again');
    assert(t.hasCanvas(0) && t.hasCanvas(2) && t.hasCanvas(3), 'the whole window is restored, including images released earlier');
    assert(t.em.executed.length > before, 'scroll up triggered new work');
    assert.strictEqual(t.imgs[1].parentNode.children.filter(c => c instanceof FakeCanvas).length, 1, 'never two canvases on one image');
    assert(!t.em.calls.includes('release'), 'released images keep the worker alive (they may be needed again)');
    console.log('ok  low RAM: window, release, scroll-up reprocessing, worker kept');
  }
  // 2. aggressive: everything processed; worker released once ALL are done; no re-wake loop
  {
    const t = setup({ ram: 'aggressive', count: 4 }); await t.settle(150);
    assert([0, 1, 2, 3].every(i => t.hasCanvas(i)), 'all images upscaled');
    assert.strictEqual(t.em.calls.filter(c => c === 'release').length, 1, 'worker released exactly once after the grace period');
    const prewarms = t.em.calls.filter(c => c === 'prewarm').length;
    await t.scrollTo(500); await t.settle(100);
    assert.strictEqual(t.em.calls.filter(c => c === 'prewarm').length, prewarms, 'nothing pending -> no re-wake');
    console.log('ok  aggressive: single idle release, no thrash');
  }
  // 3. skipped (too wide) counts as finished; failed counts as finished; stats median of last 5
  {
    const t = setup({ ram: 'aggressive', count: 6, widths: [1000, 1800, 1000, 1000, 1000, 1000] }); await t.settle(200);
    assert(!t.hasCanvas(1), 'wide image (1800 > 1500) is left alone');
    assert(t.em.calls.includes('release'), 'skipped images do not block teardown');
    const st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.images.skipped, 1); assert.strictEqual(st.images.done, 5);
    assert.strictEqual(st.stats.n, 5); assert(st.stats.medianMs > 0);
    assert(t.logs.some(l => /^– 1\.webp skipped/.test(l)), 'one concise skip line');
    const done = t.logs.filter(l => /^✓ /.test(l)); assert.strictEqual(done.length, 5);
    assert(/✓ 0\.webp 1000×1400→2000×2800 \d+(\.\d)?ms · decode .* · onnx .*gpu .*= prep .* \+ run .* \+ post .* · show .* · waited/.test(done[0]), done[0]);
    // force upscale bypasses the skip rule
    t.onMessage({ action: 'forceUpscale', srcUrl: 'https://cdn/x/1.webp' }); await t.settle(60);
    assert(t.hasCanvas(1), 'forced upscale processes a skipped image');
    console.log('ok  skip/finished semantics, status, log line shape, force');
  }
  // 4. cancel: low RAM fast scroll drops stale queued jobs and leaves no stuck state
  {
    const t = setup({ ram: 'low', count: 30 }); await t.scrollTo(20000); await t.settle(100);
    const ex = new Set(t.em.executed);
    assert([...ex].some(u => /\/20\.webp/.test(u)), 'new position processed');
    assert(![...ex].some(u => /\/(5|6|7)\.webp/.test(u)), 'images skipped over by a fast scroll are never processed');
    console.log('ok  fast scroll cancels stale work');
  }
  // 5. reset page: original DOM restored, state cleared, processing starts again
  {
    const t = setup({ ram: 'aggressive', count: 4 }); await t.settle(150);
    assert([0, 1, 2, 3].every(i => t.hasCanvas(i)) && t.imgs[0].parentNode !== t.reader, 'precondition: upscaled and wrapped');
    const switches = t.em.calls.filter(c => c.startsWith('switch')).length;
    t.onMessage({ action: 'resetPage' });
    assert([0, 1, 2, 3].every(i => t.imgs[i].parentNode === t.reader), 'every <img> is back directly under the reader, wrappers gone');
    assert([0, 1, 2, 3].every(i => t.visible(i)), 'originals visible again');
    assert.deepStrictEqual(t.reader.children.map(c => c.tagName), ['IMG', 'IMG', 'IMG', 'IMG'], 'original order and no leftovers');
    assert.strictEqual(t.onMessage({ action: 'getStatus' }).stats.n, 0, 'stats cleared');
    assert(t.em.calls.filter(c => c.startsWith('switch')).length > switches, 'engine restarted from a clean state');
    await t.settle(150);
    assert([0, 1, 2, 3].every(i => t.hasCanvas(i)), 'processing begins again after the reset');
    assert(t.logs.some(l => /page reset: 4 image\(s\) restored/.test(l)));
    console.log('ok  reset page restores the DOM and restarts processing');
  }
  // 6. window-only settings: in memory, never stored, survive storage changes, revert cleanly
  {
    const t = setup({ ram: 'low' }); await t.settle();
    let st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.settings.ramMode, 'low'); assert.deepStrictEqual(st.localKeys, []);
    t.onMessage({ action: 'setLocalSettings', overrides: { ramMode: 'aggressive', engine: 'canvas' } });   // would throw if it persisted
    await t.settle(40);
    st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.settings.ramMode, 'aggressive'); assert.strictEqual(st.settings.engine, 'canvas');
    assert.deepStrictEqual(st.localKeys.sort(), ['engine', 'ramMode']);
    assert(t.em.calls.includes('switch:canvas'), 'engine override took effect in this window');
    await t.changeStorage({ ramMode: 'low', maxOutputWidth: 900 });                    // a global save elsewhere
    st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.settings.ramMode, 'aggressive', 'overridden key keeps its window-only value');
    assert.strictEqual(st.settings.maxOutputWidth, 900, 'non-overridden keys still follow storage');
    t.onMessage({ action: 'clearLocalSettings' }); await t.settle(40);
    st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.settings.ramMode, 'low'); assert.strictEqual(st.settings.engine, 'onnx'); assert.deepStrictEqual(st.localKeys, []);
    t.onMessage({ action: 'setLocalSettings', overrides: { bogusKey: 1, ramMode: 'nonsense' } }); await t.settle(20);
    st = t.onMessage({ action: 'getStatus' });
    assert.strictEqual(st.settings.ramMode, 'low', 'invalid values are sanitised, unknown keys ignored');
    console.log('ok  window-only settings: in memory, layered over storage, revertible');
  }
  console.log('content tests passed');
})().catch(e => { console.error('FAIL', e); process.exit(1); });
