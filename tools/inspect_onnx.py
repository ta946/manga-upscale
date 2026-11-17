#!/usr/bin/env python3
"""
Inspect an ONNX model: I/O, parameter dtypes, op histogram, and whether it is
"fp16 storage only" (weights stored fp16 but Cast to fp32 for compute) or has
real fp16 compute. Also detects the unfused PReLU pattern (Less + Mul + Where).

No dependency on the `onnx` package: it reads the protobuf wire format
directly, so it only needs Python 3.8+. (`onnxruntime` is optional and only used
to cross-check input/output types.)

    python inspect_onnx.py model.onnx [more.onnx ...]
"""
import collections
import os
import sys

DT = {1: 'float32', 2: 'uint8', 3: 'int8', 4: 'uint16', 5: 'int16', 6: 'int32',
      7: 'int64', 9: 'bool', 10: 'float16', 11: 'float64', 12: 'uint32',
      13: 'uint64', 16: 'bfloat16'}


def varint(b, i):
    r = sh = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << sh; sh += 7
        if not c & 0x80:
            return r, i


def fields(b):
    i = 0
    while i < len(b):
        key, i = varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 1:
            v = b[i:i + 8]; i += 8
        elif w == 5:
            v = b[i:i + 4]; i += 4
        elif w == 2:
            n, i = varint(b, i); v = b[i:i + n]; i += n
        else:
            raise ValueError(f'unsupported wire type {w}')
        yield f, w, v


def packed_varints(buf):
    out, j = [], 0
    while j < len(buf):
        v, j = varint(buf, j); out.append(v)
    return out


def parse_value_info(buf):
    name, elem, shape = '', None, []
    for f, _, v in fields(buf):
        if f == 1:
            name = v.decode()
        elif f == 2:                                   # TypeProto
            for tf, _, tv in fields(v):
                if tf == 1:                            # tensor_type
                    for xf, _, xv in fields(tv):
                        if xf == 1:
                            elem = xv
                        elif xf == 2:                  # shape
                            for df, _, dv in fields(xv):
                                if df == 1:            # dim
                                    d = '?'
                                    for kf, kw, kv in fields(dv):
                                        if kf == 1: d = kv
                                        elif kf == 2: d = kv.decode()
                                    shape.append(d)
    return name, DT.get(elem, elem), shape


def parse_model(path):
    data = open(path, 'rb').read()
    info = {'size': len(data), 'opsets': {}, 'producer': ''}
    graph = None
    for f, _, v in fields(data):
        if f == 2: info['producer'] = v.decode(errors='replace')
        elif f == 7: graph = v
        elif f == 8:
            dom, ver = '', 0
            for kf, _, kv in fields(v):
                if kf == 1: dom = kv.decode()
                elif kf == 2: ver = kv
            info['opsets'][dom or 'ai.onnx'] = ver
    if graph is None:
        raise SystemExit('no graph found (external-data or non-ONNX file?)')

    inits, nodes, inputs, outputs = {}, [], [], []
    for f, _, v in fields(graph):
        if f == 5:                                     # initializer
            dims, dt, name = [], None, ''
            for kf, kw, kv in fields(v):
                if kf == 1: dims.extend([kv] if kw == 0 else packed_varints(kv))
                elif kf == 2: dt = kv
                elif kf == 8: name = kv.decode()
            n = 1
            for d in dims: n *= d
            inits[name] = (DT.get(dt, dt), n)
        elif f == 1:                                   # node
            nd = {'in': [], 'out': [], 'op': '', 'attrs': {}}
            for kf, _, kv in fields(v):
                if kf == 1: nd['in'].append(kv.decode())
                elif kf == 2: nd['out'].append(kv.decode())
                elif kf == 4: nd['op'] = kv.decode()
                elif kf == 5:
                    an, ai = '', None
                    for af, _, av in fields(kv):
                        if af == 1: an = av.decode()
                        elif af == 3: ai = av
                    nd['attrs'][an] = ai
            nodes.append(nd)
        elif f == 11: inputs.append(parse_value_info(v))
        elif f == 12: outputs.append(parse_value_info(v))
    return info, inits, nodes, inputs, outputs


def report(path):
    info, inits, nodes, inputs, outputs = parse_model(path)
    print('=' * 72)
    print(f'{os.path.basename(path)}  ({info["size"] / 1e6:.2f} MB, opset {info["opsets"]}, producer "{info["producer"]}")')
    init_names = set(inits)
    real_inputs = [i for i in inputs if i[0] not in init_names]
    for n, t, s in real_inputs: print(f'  input : {n} {t} {s}')
    for n, t, s in outputs:     print(f'  output: {n} {t} {s}')

    by_dtype, params = collections.Counter(), collections.Counter()
    for dt, n in inits.values():
        by_dtype[dt] += 1; params[dt] += n
    print('  weights by dtype :', {k: f'{by_dtype[k]} tensors / {params[k]:,} values' for k in by_dtype})

    ops = collections.Counter(n['op'] for n in nodes)
    print('  nodes            :', len(nodes), dict(ops.most_common()))

    casts = [n for n in nodes if n['op'] == 'Cast']
    weight_casts = [n for n in casts if n['in'] and n['in'][0] in init_names]
    cast_to = collections.Counter(DT.get(n['attrs'].get('to'), n['attrs'].get('to')) for n in casts)
    producer_of = {o: n for n in nodes for o in n['out']}

    # The dtype a Conv really computes in is the dtype of its weight input *after*
    # any Cast, not the storage dtype of the initializer.
    conv_dtype = collections.Counter()
    for n in nodes:
        if n['op'] not in ('Conv', 'ConvTranspose', 'Gemm', 'MatMul') or len(n['in']) < 2:
            continue
        w = n['in'][1]
        src = producer_of.get(w)
        if src is not None and src['op'] == 'Cast':
            conv_dtype[DT.get(src['attrs'].get('to'), src['attrs'].get('to'))] += 1
        elif w in inits:
            conv_dtype[inits[w][0]] += 1
        else:
            conv_dtype['unknown'] += 1

    # Unfused PReLU: Where(Less(x,0), Mul(x,slope), x)
    producer = {o: n for n in nodes for o in n['out']}
    prelu = 0
    for n in nodes:
        if n['op'] != 'Where' or len(n['in']) != 3: continue
        c, a, b = (producer.get(t) for t in n['in'])
        if c and a and c['op'] == 'Less' and a['op'] == 'Mul' and c['in'][0] == n['in'][2] \
                and n['in'][2] in a['in']:
            prelu += 1

    print()
    stored_fp16 = params.get('float16', 0) > 0.5 * max(1, sum(params.values()))
    print(f'  Conv/MatMul compute dtype (after casts): {dict(conv_dtype)}')
    if conv_dtype and set(conv_dtype) == {'float16'}:
        print('  VERDICT: real fp16 compute. Nothing to gain from convert_fp16.py.')
    elif stored_fp16 and conv_dtype.get('float32'):
        print('  VERDICT: fp16 STORAGE only. Weights are stored fp16 (small file) but Cast to fp32')
        print('           at every use, so the maths still runs in fp32. Real fp16 compute needs the')
        print('           activations in fp16 as well: re-export from fp32 and run convert_fp16.py.')
    elif conv_dtype.get('float32') and not stored_fp16:
        print('  VERDICT: pure fp32. A candidate for convert_fp16.py.')
    else:
        print('  VERDICT: mixed/unknown; inspect the numbers above.')
    print(f'  Cast nodes: {len(casts)} (on weights: {len(weight_casts)}), cast targets: {dict(cast_to)}')
    if prelu:
        print(f'  PReLU pattern: {prelu} activations are built from Less+Mul+Where (3 elementwise')
        print('           kernels each). Fusing them into PRelu (tools/fuse_prelu.py) cuts memory traffic.')

    try:  # optional cross-check
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print('  ORT view:', [(i.name, i.type) for i in s.get_inputs()], '->', [(o.name, o.type) for o in s.get_outputs()])
    except Exception as e:  # noqa: BLE001
        print(f'  (onnxruntime cross-check skipped: {type(e).__name__})')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        report(p)
