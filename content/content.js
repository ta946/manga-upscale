/*
 * Content script: finds reader images, decides what to upscale and when, and
 * swaps the result in. Heavy lifting lives elsewhere:
 *   viewport.js        pure windowing/priority rules
 *   queue.js           priority queue with cancellation
 *   engine-manager.js  engines + ONNX worker lifecycle and recovery
 */

const { findCurrentIndex, planWindow, median } = window.MUViewport;

const TEARDOWN_GRACE_MS = 2000;     // idle time before an ONNX worker is released
const STATS_WINDOW = 5;             // images in the "median total time" readout
const TERMINAL = new Set(['done', 'skipped', 'failed']);

const settings = { ...MangaSettings.DEFAULTS };      // EFFECTIVE settings; live, mutated in place
let storedSettings = { ...MangaSettings.DEFAULTS };  // what chrome.storage says
let localOverrides = {};                             // "this window only": in memory, never persisted
const upscaleQueue = new PriorityJobQueue(1);
const engineManager = new EngineManager({ settings });

/** Per-image state: { status, url, canvas, wrapper }. Absent entry = new image. */
let imgState = new WeakMap();           // replaced wholesale by resetPage()
const recentTotals = [];            // last STATS_WINDOW per-image totals (ms)
let engineEpoch = 0;                // bumped when a replan is required regardless of scroll
let teardownTimer = null;

/* ------------------------------------------------------------------ */
/* Settings and engine                                                  */
/* ------------------------------------------------------------------ */

function applySettings(next, changed = null) {
    Object.assign(settings, next);
    MU.setVerbose(settings.verboseLogs);

    const engineKeys = ['enabled', 'engine', 'onnxModelPath', 'onnxGpuIO', 'onnxCpuOutput'];
    const restart = !changed || changed.some((key) => engineKeys.includes(key));

    if (restart) {
        upscaleQueue.cancelWhere(() => true);
        for (const img of getReaderImages()) {
            const st = imgState.get(img);
            if (st && !TERMINAL.has(st.status) && st.status !== 'released') {
                if (st.canvas) st.status = 'done'; else imgState.delete(img);
            }
        }
        startEngine();
    }
    engineEpoch++;
    scheduleViewportCheck(true);
}

function startEngine() {
    upscaleQueue.concurrency = settings.engine === 'onnx' ? 2 : 1;   // ONNX serialises on the worker itself
    if (!settings.enabled) {
        engineManager.switchEngine('off');
        MU.info('disabled');
    } else {
        engineManager.switchEngine(settings.engine, { modelPath: settings.onnxModelPath });
    }
}

const effectiveSettings = () => MangaSettings.sanitize({ ...storedSettings, ...localOverrides });

MangaSettings.load().then((loaded) => { storedSettings = loaded; applySettings(effectiveSettings()); });
MangaSettings.onChange((next, changed) => {
    storedSettings = next;
    // Keys this window overrides keep their local value until the override is cleared.
    const relevant = changed.filter((key) => !(key in localOverrides));
    if (relevant.length) applySettings(effectiveSettings(), relevant);
});

/** "This window only": merge overrides over the stored settings, in memory. */
function setLocalOverrides(overrides) {
    const previous = effectiveSettings();
    localOverrides = {};
    for (const key of MangaSettings.KEYS) if (key in overrides) localOverrides[key] = overrides[key];
    const next = effectiveSettings();
    const changed = MangaSettings.KEYS.filter((key) => previous[key] !== next[key]);
    const keys = Object.keys(localOverrides);
    MU.info(keys.length ? `window-only settings: ${keys.join(', ')}` : 'window-only settings cleared');
    applySettings(next, changed);
}

/** Undo every visible change and start from a clean slate with the current settings. */
function resetPage() {
    upscaleQueue.cancelWhere(() => true);
    clearTimeout(teardownTimer);
    teardownTimer = null;

    let restored = 0;
    for (const img of getReaderImages()) {
        discardCanvas(imgState.get(img));
        img.style.removeProperty('visibility');
        const wrapper = img.parentElement;
        if (wrapper?.dataset.upscaleWrapper) {      // unwrap: put the <img> back exactly where it was
            wrapper.parentNode.insertBefore(img, wrapper);
            wrapper.remove();
            restored++;
        }
    }
    const reader = document.querySelector('div.container-chapter-reader');
    if (reader?.dataset.muStyled) {
        Object.assign(reader.style, reader._muPrevStyle || {});
        delete reader.dataset.muStyled;
    }

    imgState = new WeakMap();
    recentTotals.length = 0;
    lastSignature = '';
    engineEpoch++;
    startEngine();                                  // fresh engine/worker, clears any fallback or failure state
    scheduleViewportCheck(true);
    MU.info(`page reset: ${restored} image(s) restored, processing restarts with ${settings.engine}`);
}

/* ------------------------------------------------------------------ */
/* Image helpers                                                        */
/* ------------------------------------------------------------------ */

let lastRightClickedElement = null;
document.addEventListener('mousedown', (event) => {
    if (event.button === 2) lastRightClickedElement = event.target;
}, true);

const getImageUrl = (img) => img.currentSrc || img.src;

function getReaderImages() {
    const reader = document.querySelector('div.container-chapter-reader');
    return reader ? Array.from(reader.querySelectorAll('img')) : [];
}

const statusOf = (img) => imgState.get(img)?.status;
function setState(img, status, extra) {
    const st = imgState.get(img) || {};
    Object.assign(st, extra, { status });
    imgState.set(img, st);
    return st;
}

/** Decode through an <img crossorigin>: the image host grants CORS to that but not to fetch(). */
function decodeViaImageElement(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => createImageBitmap(img).then(resolve, reject);
        img.onerror = () => reject(new Error('image failed to load'));
        img.src = url;
    });
}

async function getImageBitmap(img) {
    const url = getImageUrl(img);
    // Same-origin DOM pixels can be used directly. Cross-origin bitmaps may stay
    // origin-tainted and fail later during GPU upload, so those are re-decoded.
    let sameOrigin = false;
    try { sameOrigin = new URL(url, location.href).origin === location.origin; } catch (_) { /* ignore */ }
    if (sameOrigin && img.complete && img.naturalWidth > 0) {
        try { return await createImageBitmap(img); } catch (_) { /* fall through */ }
    }
    return decodeViaImageElement(url);
}

function throwIfAborted(signal) {
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
}

/* ------------------------------------------------------------------ */
/* Display                                                              */
/* ------------------------------------------------------------------ */

function discardCanvas(st) {
    if (!st?.canvas) return;
    st.canvas.width = 0;            // frees the backing store immediately
    st.canvas.height = 0;
    st.canvas.remove();
    st.canvas = null;
}

function displayUpscaledCanvas(targetImg, canvas) {
    if (!(canvas instanceof HTMLCanvasElement)) throw new Error('engine did not return an HTMLCanvasElement');
    if (canvas.width === 0 || canvas.height === 0) {
        throw new Error(`output canvas has invalid dimensions ${canvas.width}x${canvas.height}`);
    }
    if (!targetImg.isConnected) { canvas.width = canvas.height = 0; return false; }

    const st = imgState.get(targetImg) || {};
    discardCanvas(st);                                   // re-processing replaces the old canvas

    canvas.style.cssText =
        'position:absolute!important;inset:0!important;width:100%!important;height:100%!important;z-index:2!important;';
    canvas.dataset.processed = 'true';

    // Reuse the wrapper from an earlier pass (or a released image) so layout does not jump.
    let wrapper = st.wrapper?.isConnected ? st.wrapper
        : (targetImg.parentElement?.dataset.upscaleWrapper ? targetImg.parentElement : null);
    if (!wrapper) {
        wrapper = document.createElement('span');
        wrapper.dataset.upscaleWrapper = 'true';
        wrapper.style.cssText = 'position:relative;display:inline-block;margin:-2px 0 !important;';
        wrapper.style.verticalAlign = getComputedStyle(targetImg).verticalAlign;
        targetImg.parentNode.insertBefore(wrapper, targetImg);
        wrapper.appendChild(targetImg);
    }
    wrapper.style.width = `${canvas.width}px`;
    wrapper.style.height = `${canvas.height}px`;
    wrapper.appendChild(canvas);

    targetImg.style.setProperty('visibility', 'hidden', 'important');   // hidden, but kept in the DOM
    setState(targetImg, 'done', { canvas, wrapper });
    return true;
}

/** Low-RAM: drop the canvas of an image far behind the viewport and show the original again. */
function releaseImage(img) {
    const st = imgState.get(img);
    if (!st) return;
    discardCanvas(st);
    img.style.removeProperty('visibility');   // wrapper (and its size) stays: no layout jump above the viewport
    setState(img, 'released');
}

/* ------------------------------------------------------------------ */
/* Queueing                                                             */
/* ------------------------------------------------------------------ */

function enqueueImage(img, priority, force = false) {
    if (!settings.enabled) return;
    const url = getImageUrl(img);
    if (!url) return;
    clearTimeout(teardownTimer);
    const enqueuedAt = performance.now();
    setState(img, 'queued', { url });
    upscaleQueue
        .add(img, (signal) => runUpscale(img, { signal, force, waitMs: performance.now() - enqueuedAt }), priority)
        .catch((error) => { if (error?.name !== 'AbortError') MU.error('unexpected job error:', error); });
}

function cancelImage(img) {
    upscaleQueue.cancel(img);
    const st = imgState.get(img);
    // Keep an existing canvas's bookkeeping; otherwise the image is simply "new" again.
    if (st?.canvas) st.status = 'done'; else imgState.delete(img);
}

async function runUpscale(img, { signal, force, waitMs }) {
    const url = getImageUrl(img);
    const name = MU.name(url);
    const t0 = performance.now();
    setState(img, 'running');
    let bitmap = null;
    let canvas = null;

    try {
        const decodeStart = performance.now();
        bitmap = await getImageBitmap(img);
        if (!bitmap.width || !bitmap.height) throw new Error('decoded image is empty');
        const decodeMs = performance.now() - decodeStart;
        throwIfAborted(signal);

        // Already-large images are left alone unless forced. Same value as the output cap
        // for now; kept as its own name so the two can be split later.
        const skipThresh = settings.maxOutputWidth;
        if (!force && skipThresh > 0 && bitmap.width > skipThresh) {
            setState(img, 'skipped');
            MU.info(`– ${name} skipped (${bitmap.width}px wide > ${skipThresh}px)`);
            return;
        }

        const engineStart = performance.now();
        const result = await engineManager.execute(bitmap, { modelPath: settings.onnxModelPath, signal, key: url });
        canvas = result.canvas;
        const engineMs = performance.now() - engineStart;
        throwIfAborted(signal);

        const showStart = performance.now();
        if (!displayUpscaledCanvas(img, canvas)) { setState(img, 'skipped'); return; }
        canvas = null;
        const showMs = performance.now() - showStart;

        const totalMs = performance.now() - t0;
        recentTotals.push(totalMs);
        if (recentTotals.length > STATS_WINDOW) recentTotals.shift();
        logImageDone(name, bitmap, result, { totalMs, decodeMs, engineMs, showMs, waitMs });
    } catch (error) {
        if (error?.name === 'AbortError') throw error;
        setState(img, 'failed');
        MU.warn(`✗ ${name} failed${error.poison ? ' (skipped, worker crashed twice on it)' : ''}: ${error.message || error}`);
    } finally {
        bitmap?.close?.();
        if (canvas) canvas.width = canvas.height = 0;     // result was never displayed
        if (signal?.aborted && statusOf(img) === 'running') imgState.delete(img);
    }
}

/** One line per image: enough to see where time went, nothing more. */
function logImageDone(name, bitmap, result, t) {
    const { info } = result;
    const tm = info.timings;
    let engine = `${info.engine}`;
    if (tm) {
        engine += ` ${MU.ms(t.engineMs)} (gpu ${MU.ms(tm.workerMs)} = prep ${MU.ms(tm.prepMs)} + run ${MU.ms(tm.runMs)} + post ${MU.ms(tm.postMs)}` +
                  `${tm.tiles > 1 ? `, ${tm.tiles} tiles` : ''}${tm.gpuIO ? '' : ', cpu I/O'})`;
    } else {
        engine += ` ${MU.ms(t.engineMs)}`;
    }
    MU.info(
        `✓ ${name} ${bitmap.width}×${bitmap.height}→${info.outW}×${info.outH} ${MU.ms(t.totalMs)}ms · ` +
        `decode ${MU.ms(t.decodeMs)} · ${engine} · show ${MU.ms(t.showMs)} · waited ${MU.ms(t.waitMs)}` +
        `${info.mode && info.mode !== 'single' ? ` · ${info.mode}` : ''}`
    );
}

/* ------------------------------------------------------------------ */
/* Viewport handling                                                    */
/* ------------------------------------------------------------------ */

let lastSignature = '';
let ticking = false;
let forceNext = false;

function scheduleViewportCheck(force = false) {
    forceNext = forceNext || force;
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
        ticking = false;
        const f = forceNext;
        forceNext = false;
        handleViewports(f);
    });
}

function pendingCount(images) {
    let n = 0;
    for (const img of images) {
        if (getImageUrl(img) && !TERMINAL.has(statusOf(img))) n++;
    }
    return n;
}

function handleViewports(force = false) {
    if (!settings.enabled) return;
    const reader = document.querySelector('div.container-chapter-reader');
    if (!reader) return;
    if (!reader.dataset.muStyled) {
        reader._muPrevStyle = { padding: reader.style.padding, textAlign: reader.style.textAlign };
        reader.style.padding = 0;
        reader.style.textAlign = 'center';
        reader.dataset.muStyled = '1';
    }

    const images = Array.from(reader.querySelectorAll('img'));
    if (!images.length) return;

    const cur = findCurrentIndex(images.length, (i) => images[i].getBoundingClientRect().bottom);
    // Nothing relevant changed since the last pass: skip all the work.
    const signature = `${cur}|${images.length}|${settings.ramMode}|${engineEpoch}`;
    if (!force && signature === lastSignature) return;
    lastSignature = signature;

    // An <img> whose source was swapped after we processed it (lazy loading) is new again.
    for (const img of images) {
        const st = imgState.get(img);
        if (st && TERMINAL.has(st.status) && st.url && st.url !== getImageUrl(img) && !st.canvas) imgState.delete(img);
    }

    const ops = planWindow(images.length, cur, settings.ramMode, (i) => {
        // Images that have no source yet are not eligible.
        return getImageUrl(images[i]) ? statusOf(images[i]) : 'skipped';
    });
    for (const i of ops.cancel) cancelImage(images[i]);
    for (const i of ops.release) releaseImage(images[i]);
    for (const { i, priority } of ops.reprioritize) upscaleQueue.setPriority(images[i], priority);
    for (const { i, priority } of ops.enqueue) enqueueImage(images[i], priority);

    // Auto-start: warm the worker as soon as a reader page has work, not on the first job.
    if (settings.workerStart === 'auto' && engineManager.engineType === 'onnx' &&
        engineManager.onnx.state === 'idle' && pendingCount(images) > 0) {
        engineManager.prewarmONNX();
    }
    evaluateTeardown();
}

window.addEventListener('scroll', () => scheduleViewportCheck(false), { passive: true });
window.addEventListener('resize', () => scheduleViewportCheck(false), { passive: true });
document.addEventListener('load', (event) => {
    if (event.target?.tagName === 'IMG') scheduleViewportCheck(true);   // a source may have appeared
}, true);
new MutationObserver(() => scheduleViewportCheck(false)).observe(document.body, { childList: true, subtree: true });

/* ------------------------------------------------------------------ */
/* Idle teardown                                                        */
/* ------------------------------------------------------------------ */

/**
 * Release the ONNX worker once EVERY eligible image has reached a terminal state
 * (done / skipped / failed). Images that low-RAM mode released count as unfinished,
 * since scrolling back would need the worker again, so they keep it alive.
 */
function evaluateTeardown() {
    clearTimeout(teardownTimer);
    teardownTimer = null;
    if (engineManager.engineType !== 'onnx' || engineManager.onnx.state !== 'ready') return;
    if (!upscaleQueue.idle || pendingCount(getReaderImages()) > 0) return;
    teardownTimer = setTimeout(() => {
        teardownTimer = null;
        if (upscaleQueue.idle && pendingCount(getReaderImages()) === 0) engineManager.releaseONNX('idle');
    }, TEARDOWN_GRACE_MS);
}
upscaleQueue.onDrain = evaluateTeardown;

/* ------------------------------------------------------------------ */
/* Messages: context menu, popup status                                 */
/* ------------------------------------------------------------------ */

function buildStatus() {
    const counts = { done: 0, skipped: 0, failed: 0, working: 0, waiting: 0, released: 0 };
    for (const img of getReaderImages()) {
        const s = statusOf(img);
        if (s === 'done' || s === 'skipped' || s === 'failed' || s === 'released') counts[s]++;
        else if (s === 'queued' || s === 'running') counts.working++;
        else if (getImageUrl(img)) counts.waiting++;
    }
    return {
        enabled: settings.enabled,
        settings: { ...settings },                       // effective values, so the popup shows what THIS window uses
        localKeys: Object.keys(localOverrides),
        ...engineManager.getStatus(),
        stats: { medianMs: median(recentTotals), n: recentTotals.length, window: STATS_WINDOW },
        images: counts
    };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getStatus') { sendResponse(buildStatus()); return; }
    if (request.action === 'setLocalSettings') { setLocalOverrides(request.overrides || {}); sendResponse({ ok: true }); return; }
    if (request.action === 'clearLocalSettings') { if (Object.keys(localOverrides).length) setLocalOverrides({}); sendResponse({ ok: true }); return; }
    if (request.action === 'resetPage') { resetPage(); sendResponse({ ok: true }); return; }
    if (request.action === 'retryEngine') {
        engineManager.retryONNX();
        engineEpoch++;
        scheduleViewportCheck(true);
        sendResponse({ ok: true });
        return;
    }
    if (!settings.enabled || request.action !== 'forceUpscale') return;

    let targetImg = Array.from(document.querySelectorAll('img')).find((img) =>
        img.src === request.srcUrl || img.currentSrc === request.srcUrl ||
        (request.srcUrl && img.src.includes(request.srcUrl)));
    if (!targetImg && lastRightClickedElement?.tagName === 'IMG') targetImg = lastRightClickedElement;

    if (!targetImg) {
        MU.warn('force upscale: no <img> under the cursor');
        return;
    }
    MU.info(`force upscale ${MU.name(getImageUrl(targetImg))}`);
    cancelImage(targetImg);                 // drop any queued copy, then run at top priority
    enqueueImage(targetImg, -1, true);
});
