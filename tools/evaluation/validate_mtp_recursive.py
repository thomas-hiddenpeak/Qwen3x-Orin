#!/usr/bin/env python3
"""Independent original-weight FP64 recursive draft check, conditional on captured initial KV.

This tests three long-position recursive steps, not independent reconstruction
of the entire initial cache. Target/cache construction admission remains separate.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from validate_mtp_draft import Checkpoint,decode,bf16,norm

def run(model,capture):
    checkpoint=Checkpoint(model)
    weights={name:checkpoint.weight(name) for name in checkpoint.index if name.startswith('mtp.')}
    meta=json.loads((capture/'recursive.json').read_text());position=meta['initial_position']
    if meta.get('long_attention_steps',meta.get('private_split_steps'))!=3:raise ValueError('long private Attention route missing')
    if not 511 <= position <= 44092:raise ValueError('long position outside capacity')
    tokens=np.fromfile(capture/'recursive-tokens.u32',dtype='<u4')
    if len(tokens)!=3 or meta['steps']!=3:raise ValueError('three recursive steps required')
    initial=decode(np.fromfile(capture/'initial-hidden.bf16',dtype='<u2'))
    if initial.shape!=(5120,):raise ValueError('initial hidden shape')
    embedding=checkpoint.bits('model.language_model.embed_tokens.weight')
    embedded=decode(embedding[tokens])
    checkpoint.payload_hashes['selected_embedding_rows']=hashlib.sha256(embedding[tokens].tobytes()).hexdigest()
    keys = list(decode(np.fromfile(capture / 'initial-k.bf16',dtype='<u2')).reshape(-1,4,256))
    values = list(decode(np.fromfile(capture / 'initial-v.bf16',dtype='<u2')).reshape(-1,4,256))
    if len(keys)!=position or len(values)!=position:raise ValueError('initial cache shape')
    hiddens, new_keys, new_values = [], [], []
    layer = 'mtp.layers.0.'
    def project(x, name):
        return bf16(weights[name] @ x.astype(np.float64))
    for row in range(len(tokens)):
        t=position+row
        pair = np.concatenate([norm(embedded[row], weights['mtp.pre_fc_norm_embedding.weight']),
                               norm(hiddens[-1] if hiddens else initial, weights['mtp.pre_fc_norm_hidden.weight'])])
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
        keys.append(k); values.append(v); new_keys.append(k.copy()); new_values.append(v.copy())
        ks, vs = np.stack(keys), np.stack(values)
        probabilities = np.empty((24,len(keys)),dtype=np.float32)
        attention = np.empty((24, 256), dtype=np.float32)
        for h in range(24):
            scores = (ks[:, h // 6].astype(np.float64) @ q[h].astype(np.float64)) / 16
            probs = np.exp(scores - scores.max()); probs /= probs.sum()
            probabilities[h]=probs
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
    # Retain real prompt-derived FP64 reference operands for the isolated PV gate.
    ((vs.view(np.uint32)>>16).astype('<u2')).tofile(capture/'real-v.bf16')
    probabilities.astype('<f4').tofile(capture/'real-p.f32')
    report={'scope':'three recursively propagated original-weight FP64 steps, conditional on captured initial KV and fixed captured input token IDs','initial_position':position,'threshold_max_row_relative_l2':0.02,'comparisons':{},'payload_sha256':checkpoint.payload_hashes}
    for name,reference in [('hidden',np.stack(hiddens)),('k',np.stack(new_keys)),('v',np.stack(new_values))]:
        native=decode(np.fromfile(capture/f'recursive-{name}.bf16',dtype='<u2')).reshape(reference.shape)
        if not np.isfinite(native).all() or not np.isfinite(reference).all():raise ValueError('nonfinite capture/reference')
        rows=reference.reshape(3,-1).astype(np.float64);error=native.reshape(rows.shape).astype(np.float64)-rows
        relative=np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(rows,axis=1),1e-30)
        report['comparisons'][name]={'row_relative_l2':relative.tolist(),'max_row_relative_l2':float(relative.max()),'pass':bool(relative.max()<=0.02)}
    report['pass']=all(v['pass'] for v in report['comparisons'].values())
    report['capture_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in capture.iterdir() if p.suffix in ['.bf16','.u32','.f32']}
    (capture/'independent-recursive-oracle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['comparisons'],indent=2),flush=True)
    if not report['pass']:raise SystemExit(1)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',type=Path);parser.add_argument('capture',type=Path);args=parser.parse_args();run(args.model,args.capture)
