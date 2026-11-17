/*
 * Single source of truth for user settings. Loaded by the content script AND the
 * popup, so defaults and validation live in exactly one place.
 *
 *   maxOutputWidth  cap on the final output width in px; 0 = no cap
 *   oversizeMode    what to do with an image above the GPU-memory budget (fixed in engine-manager.js):
 *                   'downscale' shrink it first, 'tile' run it in overlapping tiles
 *   workerStart     'auto' start on reader pages, 'lazy' start at the first image
 *
 * "0 = off" only disables the *user* limit. A hard device-limit guard in the
 * worker still prevents allocations the GPU cannot satisfy.
 */
(() => {
    const DEFAULTS = Object.freeze({
        enabled: true,
        engine: 'canvas',
        ramMode: 'low',
        onnxModelPath: 'models/realesr-animevideov3.onnx',
        maxOutputWidth: 1500,
        oversizeMode: 'downscale',
        workerStart: 'auto',
        onnxGpuIO: true,
        onnxCpuOutput: true,
        verboseLogs: false
    });

    const oneOf = (value, allowed, fallback) => (allowed.includes(value) ? value : fallback);
    const number = (value, fallback, min, max) => {
        const n = Number(value);
        if (!Number.isFinite(n) || n < 0) return fallback;
        if (n === 0) return 0;                       // 0 = disabled
        return Math.min(max, Math.max(min, n));
    };

    function sanitize(raw = {}) {
        const out = { ...DEFAULTS };
        // Migrate the former engine='off' option to the independent toggle.
        out.enabled = raw.enabled !== undefined ? raw.enabled !== false : raw.engine !== 'off';
        out.engine = raw.engine && raw.engine !== 'off' ? raw.engine : DEFAULTS.engine;
        out.ramMode = oneOf(raw.ramMode, ['low', 'aggressive'], DEFAULTS.ramMode);
        out.onnxModelPath = raw.onnxModelPath || DEFAULTS.onnxModelPath;
        out.maxOutputWidth = Math.round(number(raw.maxOutputWidth, DEFAULTS.maxOutputWidth, 64, 16384));
        out.oversizeMode = oneOf(raw.oversizeMode, ['downscale', 'tile'], DEFAULTS.oversizeMode);
        out.workerStart = oneOf(raw.workerStart, ['auto', 'lazy'], DEFAULTS.workerStart);
        out.onnxGpuIO = raw.onnxGpuIO !== false;
        out.onnxCpuOutput = raw.onnxCpuOutput !== false;
        out.verboseLogs = raw.verboseLogs === true;
        return out;
    }

    const KEYS = Object.keys(DEFAULTS);

    window.MangaSettings = {
        DEFAULTS,
        KEYS,
        sanitize,

        /** Resolves with a sanitised settings object. */
        load() {
            return new Promise((resolve) => {
                chrome.storage.sync.get(KEYS, (data) => resolve(sanitize(data)));
            });
        },

        /** Calls cb(newSettings, changedKeys) after any relevant storage change. */
        onChange(cb) {
            chrome.storage.onChanged.addListener((changes, area) => {
                if (area !== 'sync') return;
                const changed = Object.keys(changes).filter((key) => KEYS.includes(key));
                if (!changed.length) return;
                this.load().then((settings) => cb(settings, changed));
            });
        }
    };
})();
