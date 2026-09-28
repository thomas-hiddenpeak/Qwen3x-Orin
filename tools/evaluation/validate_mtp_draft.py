#!/usr/bin/env python3
"""Independent CPU FP64-projection check of native MTP draft captures.

Correctness only. Inputs are teacher-forced target hidden rows; no speed or
acceptance claim. Each projection/normalization/residual boundary publishes BF16.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def decode(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return decode(rounded)


def norm(x, weight):
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    inverse = (1 / np.sqrt(variance + 1e-6)).astype(np.float32)
    return bf16((x * inverse) * (weight + np.float32(1)))


class Checkpoint:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}
        self.payload_hashes = {}

    def bits(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            with (self.root / shard).open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(f.read(length)))
        origin, header = self.headers[shard]
        tensor = header[name]
        if tensor['dtype'] != 'BF16':
            raise ValueError(f'non-BF16 tensor: {name}')
        start, end = tensor['data_offsets']
        if end - start != int(np.prod(tensor['shape'])) * 2:
            raise ValueError(f'bad tensor extent: {name}')
        return np.memmap(self.root / shard, mode='r', dtype='<u2',
                         offset=origin + start, shape=tuple(tensor['shape']))

    def weight(self, name):
        bits = self.bits(name)
        self.payload_hashes[name] = hashlib.sha256(bits.tobytes()).hexdigest()
        # Independent double accumulation; GPU FP32 reduction is not copied.
        return decode(bits).astype(np.float64 if bits.ndim == 2 else np.float32)


def run(model, capture):
    checkpoint = Checkpoint(model)
    weights = {name: checkpoint.weight(name) for name in checkpoint.index if name.startswith('mtp.')}
    tokens = np.fromfile(capture / 'prompt.u32', dtype='<u4')
    target = decode(np.fromfile(capture / 'target-hidden.bf16', dtype='<u2')).reshape(-1, 5120)
    if len(tokens) != len(target) or len(tokens) < 2:
        raise ValueError('capture prompt/hidden shape mismatch')
    embedding = checkpoint.bits('model.language_model.embed_tokens.weight')
    embedded = decode(embedding[tokens[1:]])
    checkpoint.payload_hashes['selected_embedding_rows'] = hashlib.sha256(embedding[tokens[1:]].tobytes()).hexdigest()
    keys, values, hiddens = [], [], []
    layer = 'mtp.layers.0.'
    def project(x, name):
        return bf16(weights[name] @ x.astype(np.float64))
    for t in range(len(tokens) - 1):
        pair = np.concatenate([norm(embedded[t], weights['mtp.pre_fc_norm_embedding.weight']),
                               norm(target[t], weights['mtp.pre_fc_norm_hidden.weight'])])
        residual = project(pair, 'mtp.fc.weight')
        x = norm(residual, weights[layer + 'input_layernorm.weight'])
        qgate = project(x, layer + 'self_attn.q_proj.weight').reshape(24, 2, 256)
        q = norm(qgate[:, 0], weights[layer + 'self_attn.q_norm.weight'])
        k = norm(project(x, layer + 'self_attn.k_proj.weight').reshape(4, 256), weights[layer + 'self_attn.k_norm.weight'])
        v = project(x, layer + 'self_attn.v_proj.weight').reshape(4, 256)
        angle = t * 1e7 ** (-2 * np.arange(32, dtype=np.float64) / 64)
        cos, sin = bf16(np.cos(angle)), bf16(np.sin(angle))
        for vector in (q, k):
            low, high = vector[:, :32].copy(), vector[:, 32:64].copy()
            vector[:, :32] = bf16(low * cos - high * sin)
            vector[:, 32:64] = bf16(high * cos + low * sin)
        keys.append(k); values.append(v)
        ks, vs = np.stack(keys), np.stack(values)
        attention = np.empty((24, 256), dtype=np.float32)
        for h in range(24):
            scores = (ks[:, h // 6].astype(np.float64) @ q[h].astype(np.float64)) / 16
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            attention[h] = bf16(probs @ vs[:, h // 6].astype(np.float64))
        gated = bf16(attention / (1 + np.exp(-qgate[:, 1])))
        branch = project(gated.reshape(-1), layer + 'self_attn.o_proj.weight')
        residual = bf16(residual + branch)
        x = norm(residual, weights[layer + 'post_attention_layernorm.weight'])
        gate = project(x, layer + 'mlp.gate_proj.weight')
        up = project(x, layer + 'mlp.up_proj.weight')
        activated = bf16((gate / (1 + np.exp(-gate))) * up)
        residual = bf16(residual + project(activated, layer + 'mlp.down_proj.weight'))
        hiddens.append(norm(residual, weights['mtp.norm.weight']))
    report = {'scope': 'independent CPU FP64 projection and causal attention; BF16 boundaries',
              'threshold_max_row_relative_l2': 0.02, 'prompt_tokens': len(tokens),
              'payload_sha256': checkpoint.payload_hashes, 'comparisons': {}}
    for name, reference in [('hidden', np.stack(hiddens)), ('k', np.stack(keys)), ('v', np.stack(values))]:
        native = decode(np.fromfile(capture / f'draft-prefill-{name}.bf16', dtype='<u2')).reshape(reference.shape)
        if not np.isfinite(native).all() or not np.isfinite(reference).all():
            raise ValueError('nonfinite draft capture/reference')
        rows = reference.reshape(len(reference), -1).astype(np.float64)
        error = native.reshape(rows.shape).astype(np.float64) - rows
        relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(rows, axis=1), 1e-30)
        report['comparisons'][name] = {'max_row_relative_l2': float(relative.max()),
                                       'max_absolute_error': float(np.abs(error).max()),
                                       'elements': int(reference.size),
                                       'pass': bool(relative.max() <= 0.02)}
    report['pass'] = all(x['pass'] for x in report['comparisons'].values())
    report['capture_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in capture.iterdir() if p.suffix in ('.u32', '.bf16')}
    (capture / 'independent-draft-oracle.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['comparisons'], indent=2))
    if not report['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    run(args.model, args.capture)
