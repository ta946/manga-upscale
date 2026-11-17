const assert = require('assert');
const { PriorityJobQueue: Q } = require('./load')('content/queue.js');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

(async () => {
  // order by priority, ties by arrival
  {
    const q = new Q(1), ran = [];
    const mk = (k) => ({ k });
    const a = mk('a'), b = mk('b'), c = mk('c'), d = mk('d');
    q.add(a, async () => { ran.push('a'); await sleep(5); }, 0);   // starts immediately
    q.add(b, async () => { ran.push('b'); }, 5);
    q.add(c, async () => { ran.push('c'); }, 1);
    q.add(d, async () => { ran.push('d'); }, 1);
    await sleep(30);
    assert.deepStrictEqual(ran, ['a', 'c', 'd', 'b']);
  }
  // live re-prioritisation of a queued job
  {
    const q = new Q(1), ran = [];
    const [a, b, c] = [{}, {}, {}];
    q.add(a, async () => { ran.push('a'); await sleep(5); }, 0);
    q.add(b, async () => { ran.push('b'); }, 1);
    q.add(c, async () => { ran.push('c'); }, 2);
    q.setPriority(c, -1);
    await sleep(30);
    assert.deepStrictEqual(ran, ['a', 'c', 'b']);
  }
  // cancel works on object keys (the old URL-vs-element bug), rejects with AbortError
  {
    const q = new Q(1); const a = {}, b = {};
    q.add(a, async () => { await sleep(10); });
    let err; const pb = q.add(b, async () => 'never').catch(e => { err = e; });
    assert(q.cancel(b));
    await pb;
    assert.strictEqual(err.name, 'AbortError');
    assert(!q.has(b));
  }
  // running job sees the abort signal; the same key can be re-added while it winds down
  {
    const q = new Q(1); const a = {}; const log = [];
    q.add(a, async (signal) => { await sleep(15); log.push(signal.aborted ? 'first:aborted' : 'first:ok'); });
    await sleep(2);
    q.cancel(a);
    const p2 = q.add(a, async () => { log.push('second'); });
    await p2; await sleep(20);
    assert.deepStrictEqual(log.sort(), ['first:aborted', 'second']);
    assert(q.idle);
  }
  // concurrency 2 and the drain hook
  {
    const q = new Q(2); let active = 0, max = 0, drained = 0;
    q.onDrain = () => drained++;
    const keys = [{}, {}, {}, {}];
    await Promise.all(keys.map(k => q.add(k, async () => { active++; max = Math.max(max, active); await sleep(5); active--; })));
    await sleep(5);
    assert.strictEqual(max, 2); assert(drained >= 1 && q.idle);
  }
  // errors propagate to the caller and do not wedge the queue
  {
    const q = new Q(1); const a = {}, b = {}; let got;
    await q.add(a, async () => { throw new Error('boom'); }).catch(e => { got = e.message; });
    assert.strictEqual(got, 'boom');
    assert.strictEqual(await q.add(b, async () => 42), 42);
  }
  console.log('queue tests passed');
})().catch(e => { console.error('FAIL', e); process.exit(1); });
