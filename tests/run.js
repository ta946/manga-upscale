// node tests/run.js  -- runs every suite; exits non-zero on the first failure
const { spawnSync } = require('child_process');
for (const f of ['viewport', 'queue', 'worker', 'engine', 'content']) {
  const r = spawnSync(process.execPath, [`${__dirname}/${f}.test.js`], { stdio: 'inherit' });
  if (r.status !== 0) { console.error(`\n✗ ${f} suite failed`); process.exit(1); }
}
console.log('\nall suites passed');
