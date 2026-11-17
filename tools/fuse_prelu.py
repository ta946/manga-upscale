#!/usr/bin/env python3
"""
Fuse the unfused PReLU pattern  Where(Less(x, 0), Mul(x, slope), x)  into a single
PRelu(x, slope) node. Some exporters (e.g. the realesr-animevideov3 file) emit
this as THREE elementwise kernels per activation; on WebGPU those are
memory-bound passes over the whole feature map, so fusing them removes ~2/3 of
that traffic. The maths is identical.

    pip install onnx onnxruntime numpy
    python fuse_prelu.py model.onnx            # -> model.prelu.onnx

The result is validated against the original on random input (max abs diff must
be ~0) and the script refuses to save a file that does not match.
"""
import argparse
import collections
import os
import sys

import numpy as np


def const_value(name, init, producer):
    """Resolve a tensor name to a numpy array if it is an initializer, a Constant
    node, or a Cast of one of those. Otherwise None."""
    import onnx
    from onnx import numpy_helper
    if name in init:
        return numpy_helper.to_array(init[name])
    n = producer.get(name)
    if n is None:
        return None
    if n.op_type == 'Constant':
        for a in n.attribute:
            if a.name == 'value':
                return numpy_helper.to_array(a.t)
        return None
    if n.op_type == 'Cast':
        return const_value(n.input[0], init, producer)
    return None


def fuse(model):
    import onnx
    from onnx import helper
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    consumers = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)
    for o in g.output:
        consumers[o.name].append(None)          # graph outputs count as a consumer
    init = {t.name: t for t in g.initializer}

    remove, replace = set(), {}
    for w in g.node:
        if w.op_type != 'Where' or len(w.input) != 3:
            continue
        cond, neg, x = w.input
        less, mul = producer.get(cond), producer.get(neg)
        if not less or not mul or less.op_type != 'Less' or mul.op_type != 'Mul':
            continue
        if less.input[0] != x or x not in mul.input:
            continue
        zero = const_value(less.input[1], init, producer)
        if zero is None or not np.all(zero == 0):
            continue
        if len(consumers[cond]) != 1 or len(consumers[neg]) != 1:
            continue                              # intermediate used elsewhere: leave alone
        slope = mul.input[1] if mul.input[0] == x else mul.input[0]
        replace[id(w)] = helper.make_node('PRelu', [x, slope], list(w.output),
                                          name=(w.name or w.output[0]) + '_prelu')
        remove.update({id(less), id(mul)})

    new_nodes = []
    for n in g.node:
        if id(n) in remove:
            continue
        new_nodes.append(replace.get(id(n), n))
    del g.node[:]
    g.node.extend(new_nodes)
    return len(replace)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('-o', '--output')
    ap.add_argument('--size', type=int, default=64)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    out_path = args.output or os.path.splitext(args.model)[0] + '.prelu.onnx'

    model = onnx.load(args.model)
    n = fuse(model)
    if n == 0:
        sys.exit('No Less+Mul+Where PReLU pattern found; nothing to do.')
    onnx.checker.check_model(model)

    tmp = out_path + '.tmp'
    onnx.save(model, tmp)

    rng = np.random.default_rng(0)
    x = rng.random((1, 3, args.size, args.size), dtype=np.float32)
    a = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    name = a.get_inputs()[0].name
    ya, yb = a.run(None, {name: x})[0], b.run(None, {b.get_inputs()[0].name: x})[0]
    d = float(np.max(np.abs(ya - yb)))
    print(f'fused {n} activations; max |diff| vs original = {d:.2e}')
    if not np.isfinite(d) or d > 1e-4:
        os.remove(tmp)
        sys.exit('FAIL: fused model does not match the original; nothing saved.')
    os.replace(tmp, out_path)
    print(f'Saved {out_path}  ({os.path.getsize(args.model) / 1e6:.2f} MB -> {os.path.getsize(out_path) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
