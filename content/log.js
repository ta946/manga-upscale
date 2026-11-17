/*
 * Tiny logger shared by every content script. One prefix, four levels, and a
 * verbose switch (settings.verboseLogs) for the chatty per-stage output.
 *
 *   MU.info  lifecycle events and one summary line per image
 *   MU.warn  recoverable problems and degradations
 *   MU.error failures
 *   MU.debug only when verboseLogs is on
 */
(() => {
    const PREFIX = '[MU]';
    let verbose = false;
    window.MU = {
        setVerbose(value) { verbose = !!value; },
        info: (...args) => console.log(PREFIX, ...args),
        warn: (...args) => console.warn(PREFIX, ...args),
        error: (...args) => console.error(PREFIX, ...args),
        debug: (...args) => { if (verbose) console.debug(PREFIX, ...args); },
        /** "https://host/a/b/12.webp?x=1" -> "12.webp" */
        name(url) {
            try { return new URL(url, location.href).pathname.split('/').pop() || url; }
            catch (_) { return String(url).slice(-24); }
        },
        ms: (value) => (value >= 100 ? Math.round(value) : Math.round(value * 10) / 10)
    };
})();
