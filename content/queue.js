/*
 * Priority job queue with real cancellation and live re-prioritisation.
 *
 *   add(key, taskFn, priority)  enqueue (or update) the job for `key`
 *   setPriority(key, priority)  re-order a *queued* job; no-op once it runs
 *   cancel(key)                 drop a queued job / abort a running one
 *   cancelWhere(pred)           pred(key, job) -> bool
 *   onDrain                     called when nothing is queued or running
 *
 * Lower priority numbers run first; ties run in arrival order. `taskFn`
 * receives an AbortSignal that fires on cancel() so a running job can stop at
 * its next stage boundary. Cancelled jobs reject with an AbortError.
 *
 * The queue is keyed by an arbitrary object (the <img> element), so a key
 * can never be confused with a URL string.
 */
class PriorityJobQueue {
    constructor(concurrency = 1) {
        this._concurrency = concurrency;
        this.jobs = new Map();          // key -> job (queued or running)
        this.pending = [];              // queued jobs, unsorted; min-scan on pick (n is tiny)
        this.runningCount = 0;
        this.seq = 0;
        this.onDrain = null;
    }

    get concurrency() { return this._concurrency; }
    set concurrency(value) {
        this._concurrency = Math.max(1, value | 0);
        this._pump();
    }

    get size() { return this.pending.length; }
    get running() { return this.runningCount; }
    get idle() { return this.pending.length === 0 && this.runningCount === 0; }
    has(key) { return this.jobs.has(key); }
    stateOf(key) { return this.jobs.get(key)?.state; }

    add(key, taskFn, priority = 0) {
        const existing = this.jobs.get(key);
        // A running job that was aborted is still winding down: it must not swallow a
        // fresh request for the same key.
        if (existing && !existing.controller.signal.aborted) {
            if (existing.state === 'queued') existing.priority = Math.min(existing.priority, priority);
            return existing.promise;
        }
        const job = {
            key, taskFn, priority,
            seq: this.seq++,
            state: 'queued',
            enqueuedAt: performance.now(),
            controller: new AbortController()
        };
        job.promise = new Promise((resolve, reject) => { job.resolve = resolve; job.reject = reject; });
        // Callers handle rejections; this keeps cancelled-before-start jobs from
        // surfacing as "uncaught in promise".
        job.promise.catch(() => {});
        this.jobs.set(key, job);
        this.pending.push(job);
        this._pump();
        return job.promise;
    }

    setPriority(key, priority) {
        const job = this.jobs.get(key);
        if (job && job.state === 'queued') job.priority = priority;
    }

    cancel(key) {
        const job = this.jobs.get(key);
        if (!job) return false;
        job.controller.abort();
        if (job.state === 'queued') {
            this.pending.splice(this.pending.indexOf(job), 1);
            this.jobs.delete(key);
            job.reject(new DOMException('Job cancelled', 'AbortError'));
            this._checkDrain();
        }
        // A running job observes the signal and settles on its own.
        return true;
    }

    cancelWhere(predicate) {
        for (const [key, job] of [...this.jobs]) {
            if (predicate(key, job)) this.cancel(key);
        }
    }

    _takeNext() {
        let best = -1;
        for (let i = 0; i < this.pending.length; i++) {
            const a = this.pending[i];
            const b = this.pending[best];
            if (best < 0 || a.priority < b.priority || (a.priority === b.priority && a.seq < b.seq)) best = i;
        }
        return best < 0 ? null : this.pending.splice(best, 1)[0];
    }

    _pump() {
        while (this.runningCount < this._concurrency) {
            const job = this._takeNext();
            if (!job) break;
            this._run(job);
        }
    }

    async _run(job) {
        job.state = 'running';
        job.startedAt = performance.now();
        this.runningCount++;
        try {
            job.resolve(await job.taskFn(job.controller.signal, job));
        } catch (error) {
            job.reject(error);
        } finally {
            this.runningCount--;
            if (this.jobs.get(job.key) === job) this.jobs.delete(job.key);
            this._pump();
            this._checkDrain();
        }
    }

    _checkDrain() {
        if (this.idle && typeof this.onDrain === 'function') this.onDrain();
    }
}
window.PriorityJobQueue = PriorityJobQueue;
