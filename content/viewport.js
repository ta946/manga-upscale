/*
 * Pure viewport/window planning. No DOM access, so it is unit-testable and cheap:
 * it only looks at a status per image and the index of the first visible one.
 *
 * Statuses: undefined (new) | queued | running | done | skipped | failed | released
 *
 * Low-RAM mode keeps a small window around the viewport:
 *     [cur - BEHIND, cur + AHEAD - 1]
 *   - new/released images inside the window are enqueued (this is also what makes
 *     scrolling back up work: images that re-enter the window are processed again),
 *   - queued/running images that left the window are cancelled,
 *   - finished images further behind than the window are released (canvas freed).
 *
 * Aggressive mode processes everything from the viewport down:
 *   - new images at/after the viewport are enqueued by distance,
 *   - images ABOVE the viewport are never newly added,
 *   - images already queued that fell behind the viewport are not dropped, only
 *     demoted so they run after everything ahead of the viewport.
 */
(() => {
    const WINDOW_AHEAD = 3;      // current image + the next two
    const WINDOW_BEHIND = 1;     // keep one image behind the viewport
    const BEHIND_BASE = 1000;    // priority offset that sorts "behind" jobs after all "ahead" jobs

    /** First index whose bottom edge is at/below the viewport top. O(log n) layout reads. */
    function findCurrentIndex(count, bottomOf) {
        let lo = 0, hi = count - 1, answer = count - 1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (bottomOf(mid) >= 0) { answer = mid; hi = mid - 1; } else { lo = mid + 1; }
        }
        return answer;
    }

    function planWindow(count, cur, mode, statusAt) {
        const ops = { enqueue: [], reprioritize: [], cancel: [], release: [] };
        const aggressive = mode === 'aggressive';
        const from = aggressive ? cur : Math.max(0, cur - WINDOW_BEHIND);
        const to = aggressive ? count - 1 : Math.min(count - 1, cur + WINDOW_AHEAD - 1);

        for (let i = 0; i < count; i++) {
            const status = statusAt(i);
            if (i >= from && i <= to) {
                const priority = i >= cur ? i - cur : WINDOW_AHEAD + (cur - i);
                if (status === undefined || status === 'released') ops.enqueue.push({ i, priority });
                else if (status === 'queued') ops.reprioritize.push({ i, priority });
            } else if (aggressive) {
                if (i < cur && status === 'queued') ops.reprioritize.push({ i, priority: BEHIND_BASE + (cur - i) });
            } else if (status === 'queued' || status === 'running') {
                ops.cancel.push(i);
            } else if (i < from && status === 'done') {
                ops.release.push(i);
            }
        }
        return ops;
    }

    function median(values) {
        if (!values.length) return null;
        const sorted = [...values].sort((a, b) => a - b);
        const mid = sorted.length >> 1;
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    window.MUViewport = { WINDOW_AHEAD, WINDOW_BEHIND, BEHIND_BASE, findCurrentIndex, planWindow, median };
})();
