const assert = require('assert');
const { MUViewport: V } = require('./load')('content/viewport.js');

// Tiny simulator: statuses array + apply ops exactly like content.js does.
function sim(count, mode) {
  const status = new Array(count).fill(undefined);
  const prio = {};
  return {
    status, prio,
    step(cur) {
      const ops = V.planWindow(count, cur, mode, (i) => status[i]);
      ops.enqueue.forEach(({ i, priority }) => { status[i] = 'queued'; prio[i] = priority; });
      ops.reprioritize.forEach(({ i, priority }) => { prio[i] = priority; });
      ops.cancel.forEach((i) => { status[i] = undefined; });
      ops.release.forEach((i) => { status[i] = 'released'; });
      return ops;
    },
    finishAll() { status.forEach((s, i) => { if (s === 'queued') status[i] = 'done'; }); },
    order() { return Object.keys(prio).filter(i => status[i] === 'queued').sort((a, b) => prio[a] - prio[b]).map(Number); }
  };
}

// --- findCurrentIndex is a correct binary search --------------------------------
{
  const bottoms = [-900, -400, 100, 700, 1300, 1900];
  assert.strictEqual(V.findCurrentIndex(6, (i) => bottoms[i]), 2);
  assert.strictEqual(V.findCurrentIndex(6, (i) => -1 - i), 5, 'scrolled past everything -> last index, not 0');
  assert.strictEqual(V.findCurrentIndex(6, (i) => 10 + i), 0);
  let reads = 0; V.findCurrentIndex(100, (i) => { reads++; return i - 40; });
  assert(reads <= 8, `binary search should be O(log n), did ${reads} layout reads`);
}

// --- low RAM -------------------------------------------------------------------
{
  const s = sim(20, 'low');
  s.step(5);
  assert.deepStrictEqual(s.order(), [5, 6, 7, 4], 'current, next two, then the one behind (lowest priority)');
  s.finishAll();
  s.step(8);                                   // scrolled down 3
  assert.strictEqual(s.status[4], 'released', 'images behind the window are freed');
  assert.strictEqual(s.status[5], 'released');
  assert.strictEqual(s.status[6], 'released');
  assert.strictEqual(s.status[7], 'done', 'one image behind is kept');
  assert.deepStrictEqual(s.order(), [8, 9, 10], 'only the forward window is queued');
  s.finishAll();
  s.step(4);                                   // user scrolls back UP
  assert.deepStrictEqual(s.order(), [4, 5, 6, 3], 'released images that re-enter the window (4,5,6) and the never-seen one behind (3) are queued');
  assert.strictEqual(s.status[8], 'done', 'finished images ahead are not touched');
  // fast scroll: queued jobs that fall out of the window are cancelled
  const f = sim(30, 'low');
  f.step(0); f.step(20);
  assert.deepStrictEqual(f.order(), [20, 21, 22, 19], 'stale queued work from the old position is dropped');
}

// --- aggressive ------------------------------------------------------------------
{
  const s = sim(12, 'aggressive');
  s.step(3);
  assert.deepStrictEqual(s.order(), [3, 4, 5, 6, 7, 8, 9, 10, 11], 'everything from the viewport down');
  assert.strictEqual(s.status[0], undefined, 'images above the viewport are NOT added');
  s.status[3] = 'done'; s.status[4] = 'done';  // pretend two finished
  s.step(8);                                   // user jumps ahead; 5,6,7 still queued
  assert.deepStrictEqual(s.order(), [8, 9, 10, 11, 7, 6, 5], 'behind-viewport jobs are demoted, nearest first, not removed');
  assert(s.prio[5] >= V.BEHIND_BASE && s.prio[11] < V.BEHIND_BASE);
  s.finishAll();
  assert.strictEqual(s.status.filter(x => x === 'released').length, 0, 'aggressive never releases');
  s.step(1);                                   // scroll up to images that were never processed
  assert.deepStrictEqual(s.order(), [1, 2], 'scrolling up picks up unprocessed images now in/below the viewport');
  assert.strictEqual(s.status[0], undefined, 'still never adds images above the viewport');
}

// --- planning cost: one pass, no allocation proportional to scroll events -----------
{
  const t = process.hrtime.bigint();
  for (let k = 0; k < 2000; k++) V.planWindow(100, k % 100, 'low', () => 'done');
  const ms = Number(process.hrtime.bigint() - t) / 1e6;
  console.log(`planWindow: 2000 plans over 100 images in ${ms.toFixed(1)}ms`);
}
assert.strictEqual(V.median([5, 1, 3]), 3);
assert.strictEqual(V.median([4, 1, 3, 2]), 2.5);
assert.strictEqual(V.median([]), null);
console.log('viewport tests passed');
