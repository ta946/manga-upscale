#!/usr/bin/env python3
"""
Convert an fp32 ONNX model to fp16 compute (graph inputs/outputs stay float32,
so the extension needs no changes) and validate the result against the original.

    pip install onnx onnxconverter-common onnxruntime numpy pillow
    python convert_fp16.py model.onnx                       # -> model.fp16.onnx
    python convert_fp16.py model.onnx --image page.png      # validate on a real crop
    python convert_fp16.py model.onnx --block-ops Resize    # keep some ops in fp32
    python convert_fp16.py model.onnx --providers DmlExecutionProvider,CPUExecutionProvider

Run inspect_onnx.py first: a model whose weights are *already* fp16 storage with
Casts back to fp32 should be re-exported from fp32 before converting.

NOTE: this script is validated by comparing outputs, not by trust. If the report
says FAIL (NaN/Inf or low PSNR), do not ship the converted file; try --block-ops
or a smaller --max-finite.
"""
import argparse
import os
import sys

import numpy as np


def load_input(image_path, size):
    if image_path:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        s = min(size, w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    else:
        rng = np.random.default_rng(0)
        # smooth-ish noise: closer to image statistics than white noise
        base = rng.random((size // 8 + 1, size // 8 + 1, 3), dtype=np.float32)
        arr = np.kron(base, np.ones((8, 8, 1), dtype=np.float32))[:size, :size]
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def run(path, x, providers):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    use = [p for p in providers if p in avail] or ['CPUExecutionProvider']
    sess = ort.InferenceSession(path, providers=use)
    return sess.run(None, {sess.get_inputs()[0].name: x})[0], sess.get_providers()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('-o', '--output')
    ap.add_argument('--block-ops', default='', help='comma-separated op types to keep in fp32')
    ap.add_argument('--block-nodes', default='', help='comma-separated node names to keep in fp32')
    ap.add_argument('--max-finite', type=float, default=1e4, help='clamp for finite fp16 constants (default 1e4)')
    ap.add_argument('--min-positive', type=float, default=1e-7)
    ap.add_argument('--image', help='image used for validation (center crop); random if omitted')
    ap.add_argument('--size', type=int, default=256, help='validation crop size')
    ap.add_argument('--providers', default='CPUExecutionProvider')
    ap.add_argument('--force', action='store_true', help='convert even if weights already look fp16')
    args = ap.parse_args()

    import onnx
    from onnxconverter_common import float16

    out_path = args.output or os.path.splitext(args.model)[0] + '.fp16.onnx'
    model = onnx.load(args.model)

    fp16_inits = sum(1 for t in model.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16)
    fp32_inits = sum(1 for t in model.graph.initializer if t.data_type == onnx.TensorProto.FLOAT)
    if fp16_inits > fp32_inits and not args.force:
        sys.exit('This model already stores its weights as fp16 (see inspect_onnx.py). '
                 'Converting again will not give real fp16 compute; re-export from fp32 '
                 'or pass --force.')

    block_ops = [o for o in args.block_ops.split(',') if o] or None
    block_nodes = [n for n in args.block_nodes.split(',') if n] or None
    print(f'Converting {args.model} ...')
    m16 = float16.convert_float_to_float16(
        model,
        min_positive_val=args.min_positive,
        max_finite_val=args.max_finite,
        keep_io_types=True,                      # float32 in/out: no extension changes needed
        disable_shape_infer=False,
        op_block_list=block_ops,
        node_block_list=block_nodes,
    )
    onnx.checker.check_model(m16)
    onnx.save(m16, out_path)
    a, b = os.path.getsize(args.model), os.path.getsize(out_path)
    print(f'Saved {out_path}  ({a / 1e6:.1f} MB -> {b / 1e6:.1f} MB)')

    # ---- validation ------------------------------------------------------
    providers = [p for p in args.providers.split(',') if p]
    x = load_input(args.image, args.size)
    print(f'Validating on a {x.shape[2]}x{x.shape[3]} {"image crop" if args.image else "synthetic"} input ...')
    ref, ref_prov = run(args.model, x, ['CPUExecutionProvider'])
    try:
        got, got_prov = run(out_path, x, providers)
    except Exception as e:  # noqa: BLE001
        print(f'\nCould not run the fp16 model with {providers}: {type(e).__name__}: {str(e)[:200]}')
        print('CPU kernels for fp16 are limited. Re-run with a GPU provider, e.g.\n'
              '  --providers DmlExecutionProvider,CUDAExecutionProvider\n'
              'or A/B it directly in the extension (same file name, models/ folder).')
        return

    bad = ~np.isfinite(got)
    ref255 = np.clip(ref, 0, 1) * 255.0
    got255 = np.clip(np.nan_to_num(got), 0, 1) * 255.0
    diff = np.abs(ref255 - got255)
    mse = float(np.mean((ref255 - got255) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f'  providers      : {ref_prov[0]} (reference) vs {got_prov[0]} (fp16)')
    print(f'  NaN/Inf outputs: {int(bad.sum())}')
    print(f'  max |diff|     : {diff.max():.2f} / 255')
    print(f'  mean |diff|    : {diff.mean():.3f} / 255')
    print(f'  PSNR vs fp32   : {psnr:.1f} dB')
    ok = bad.sum() == 0 and psnr >= 45
    print('  RESULT         :', 'PASS (visually indistinguishable)' if ok else
          'FAIL: do not ship. Try --block-ops, a smaller --max-finite, or a real --image.')


if __name__ == '__main__':
    main()
