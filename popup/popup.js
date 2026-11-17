document.addEventListener('DOMContentLoaded', async () => {
    const $ = (id) => document.getElementById(id);
    const fields = {
        enabled: $('enabled'), ramMode: $('ramMode'), engine: $('engine'), onnxModelPath: $('onnxModel'),
        maxOutputWidth: $('maxOutputWidth'), oversizeMode: $('oversizeMode'),
        workerStart: $('workerStart'), verboseLogs: $('verboseLogs')
    };
    const FIELD_KEYS = Object.keys(fields);

    // Models come from models.json.
    try {
        const data = await (await fetch(chrome.runtime.getURL('models.json'))).json();
        for (const m of data.models) {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = m.name;
            fields.onnxModelPath.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load models.json:', e);
    }

    const updateVisibility = () => {
        $('onnxModelGroup').style.display = fields.engine.value === 'onnx' ? 'flex' : 'none';
    };
    fields.engine.addEventListener('change', updateVisibility);

    const fill = (values) => {
        for (const [key, el] of Object.entries(fields)) {
            if (el.type === 'checkbox') el.checked = !!values[key];
            else el.value = values[key] ?? '';
        }
        updateVisibility();
    };

    /* ---------------- talking to the active tab ---------------- */
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const tabId = tab?.id ?? null;
    const send = (message) => (tabId == null ? Promise.reject(new Error('no tab')) : chrome.tabs.sendMessage(tabId, message));

    // Fields show what THIS window actually uses (stored settings + any window-only overrides).
    let stored = await MangaSettings.load();
    let initial = stored;
    try { initial = (await send({ action: 'getStatus' })).settings || stored; } catch (_) { /* not a reader tab */ }
    fill(initial);

    /* ---------------- window-only mode ---------------- */
    let windowOnly = false;
    const saveLabel = () => { $('saveBtn').textContent = windowOnly ? 'Apply to this window' : 'Save Settings'; };
    $('scopeBtn').addEventListener('click', () => {
        windowOnly = !windowOnly;
        $('scopeBtn').setAttribute('aria-pressed', String(windowOnly));
        $('scopeNote').hidden = !windowOnly;
        saveLabel();
    });

    const flash = (text) => {
        $('saveBtn').textContent = text;
        setTimeout(saveLabel, 1500);
    };

    $('saveBtn').addEventListener('click', async () => {
        const raw = {};
        for (const [key, el] of Object.entries(fields)) {
            raw[key] = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
        }
        const clean = MangaSettings.sanitize(raw);          // clamps and validates (0 = disabled)
        fields.maxOutputWidth.value = clean.maxOutputWidth;

        if (windowOnly) {
            // In-memory only: send just the values that differ from the saved settings, so
            // everything else keeps following the saved settings. Nothing touches storage.
            const overrides = {};
            for (const key of FIELD_KEYS) if (clean[key] !== stored[key]) overrides[key] = clean[key];
            try {
                await send({ action: 'setLocalSettings', overrides });
                flash('Applied!');
            } catch (_) {
                flash('Open a reader page');
            }
            poll();
            return;
        }

        const toSave = {};
        for (const key of FIELD_KEYS) toSave[key] = clean[key];
        chrome.storage.sync.set(toSave, async () => {
            stored = { ...stored, ...clean };
            // A global save should be what this window follows: drop its window-only overrides.
            try { await send({ action: 'clearLocalSettings' }); } catch (_) { /* not a reader tab */ }
            flash('Saved!');
            poll();
        });
    });

    $('resetBtn').addEventListener('click', async () => {
        try {
            await send({ action: 'resetPage' });
            $('resetBtn').textContent = 'Done';
        } catch (_) {
            $('resetBtn').textContent = 'No page';
        }
        setTimeout(() => { $('resetBtn').textContent = 'Reset page'; }, 1500);
        poll();
    });

    $('revertBtn').addEventListener('click', async () => {
        try { await send({ action: 'clearLocalSettings' }); } catch (_) { /* ignore */ }
        fill(stored);
        poll();
    });

    $('retryBtn').addEventListener('click', async () => {
        try { await send({ action: 'retryEngine' }); } catch (_) { /* ignore */ }
        poll();
    });

    /* ---------------- live status ---------------- */
    function render(status) {
        const box = $('status');
        if (!status) { box.hidden = true; return; }
        box.hidden = false;

        const onnx = status.onnx;
        let text, color;
        if (!status.enabled) { text = 'Disabled'; color = 'grey'; }
        else if (status.fallback) { text = 'ONNX failed — using canvas'; color = 'red'; }
        else if (onnx) {
            const label = { idle: 'worker idle', starting: 'starting…', ready: 'ready', recovering: 'recovering…', failed: 'failed' }[onnx.state] || onnx.state;
            text = `ONNX: ${label}${onnx.gpuIO === false ? ' · CPU I/O' : ''}`;
            color = onnx.state === 'ready' ? 'green' : onnx.state === 'failed' ? 'red' : onnx.state === 'idle' ? 'grey' : 'amber';
        } else { text = `Engine: ${status.engine}`; color = 'green'; }
        $('statusText').textContent = text;
        $('statusDot').className = `dot ${color}`;

        const s = status.stats;
        $('statusStats').textContent = s && s.n
            ? `Median of last ${s.n}: ${Math.round(s.medianMs)} ms/image`
            : 'No images processed yet';
        const im = status.images || {};
        $('statusImages').textContent =
            `${im.done || 0} done · ${(im.working || 0) + (im.waiting || 0)} pending` +
            `${im.skipped ? ` · ${im.skipped} skipped` : ''}${im.failed ? ` · ${im.failed} failed` : ''}`;

        const local = status.localKeys || [];
        $('statusLocal').hidden = local.length === 0;
        $('statusLocalText').textContent = `Window-only: ${local.join(', ')}`;

        // Runtime problems are shown here instead of only in the console.
        const problems = [];
        if (onnx?.error && (status.fallback || onnx.state === 'failed')) problems.push(onnx.error);
        $('statusError').hidden = problems.length === 0;
        $('statusError').textContent = problems.join(' — ');
        $('retryBtn').hidden = !(status.fallback || onnx?.state === 'failed');
    }

    async function poll() {
        try { render(await send({ action: 'getStatus' })); }
        catch (_) { render(null); }           // not a reader tab / content script not present
    }
    poll();
    setInterval(poll, 1000);
});
