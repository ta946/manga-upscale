// Loads a classic content script into a sandbox with a fake `window`, like the browser does.
const vm = require('vm'), fs = require('fs'), path = require('path');
module.exports = function load(files, extra = {}) {
  const sandbox = { console, setTimeout, clearTimeout, performance, AbortController, DOMException, Promise, Map, Set, Array, Math, Object, Error, ...extra };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const f of [].concat(files)) vm.runInContext(fs.readFileSync(path.join(__dirname, '..', f), 'utf8'), sandbox, { filename: f });
  return sandbox;
};
