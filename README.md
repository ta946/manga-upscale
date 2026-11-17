Manga Upscale
===
Chrome extension to upscale manga images for **nelomanga.net**.
Can be modified to work on others

Heavily based on the work by [manga-scaler](https://github.com/Alocks/manga-scaler)

### Original
![Original](doc/Original.jpg)

### RealCUGAN
![RealCUGAN](doc/RealCUGAN.jpg)

You can add more ONNX models by downloading them into the `./models` folder and updating `models.json`
## ONNX pipeline

ONNX inference runs in a worker hosted by an extension-origin iframe (`bridge/`). The content script and the worker talk directly over a `MessagePort`, so `ImageBitmap`s are transferred, never encoded.

With WebGPU I/O (default) pixels never touch the CPU between decode and display: `ImageBitmap` → GPU texture → compute shader → ORT input buffer → ORT output buffer → render pass (bilinear resample to the exact target size) → `ImageBitmap`. Large images can be run in overlapping tiles composited on the GPU. If a GPU I/O step fails the worker falls back to a CPU path automatically.

### Settings (popup → Advanced)

| Setting | Meaning |
|---|---|
| Max output width | Cap on final width; `0` = no limit. Images wider than this are not auto-upscaled |
| Images too large for one GPU pass | *Downscale first* (fastest) or *Tile* (full detail) |
| Start worker | *When a reader page opens*, or *at the first image* |

A built-in GPU-memory guard caps a single model pass at 0.75 MP (tiles at 0.5 MP). It is not a setting: feature maps cost roughly 830 bytes per input pixel for the default model, ORT pools freed buffers by size, and larger passes can exhaust GPU memory, which resets the GPU process (black screen, blank canvases). The constants `ONNX_SAFE_PIXELS` / `ONNX_TILE_PIXELS` at the top of `content/engine-manager.js` can be raised if your GPU is known to cope.

The output width depends only on *Max output width*, never on the pixel budget: oversize images are tiled or have only their model input shrunk, then resampled to the same target width, so every page of a chapter has the same width.

Each tab runs its own worker, so several tabs processing at once each hold their own GPU memory. Model inputs are padded to multiples of 64 (clamp-to-edge) so the model sees a few repeating shapes; this bounds ORT's size-keyed GPU buffer pool and shader cache over long sessions.

Results are returned as CPU-backed bitmaps by default, so a GPU reset cannot blank them. `chrome.storage.sync.set({ onnxCpuOutput: false })` switches to the zero-copy GPU-backed path (a few ms faster per page, but vulnerable to resets).

The worker is released once every eligible image in the tab is finished (done, skipped or failed) and restarted on demand. In low-RAM mode, images that were released (canvas freed) count as unfinished, so the worker stays alive while they could still be scrolled back to.

### Recovery

If the worker dies or the device is lost, it is restarted (backoff 0 s / 1 s / 3 s, at most 3 restarts per minute) and the failed image is retried. An image that crashes the worker twice is skipped. Only when recovery is exhausted does the tab fall back to canvas; the popup shows the reason and a **Retry** button.

### Logs

One line per image, plus lifecycle events:

```
[MU] ✓ 3.webp 1125×598→1500×797 437ms · decode 34 · onnx 400 (gpu 203 = prep 1.4 + run 201 + post 0.6) · show 3 · waited 190
[MU] onnx ready in 840ms · GPU I/O
[MU] onnx idle → worker released
```

Enable *Verbose console logs* in the popup for per-stage detail. To force the CPU I/O path while debugging: `chrome.storage.sync.set({ onnxGpuIO: false })`.

### Model tools (`tools/`)

```
python tools/inspect_onnx.py models/*.onnx      # dtypes, ops, fp16-storage vs fp16-compute, PReLU pattern
python tools/convert_fp16.py model.onnx         # fp32 -> fp16 compute (float32 I/O kept), self-validating
python tools/fuse_prelu.py   model.onnx         # Less+Mul+Where -> PRelu, self-validating
```

### Tests

`node tests/run.js` (no dependencies). They run the worker, queue, viewport planner, engine manager and content script against fakes, so GPU behaviour itself still needs a real-browser check.
