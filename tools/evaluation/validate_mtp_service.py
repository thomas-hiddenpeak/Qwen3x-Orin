#!/usr/bin/env python3
"""Whole-service MTP direction check; no release/promotion authority.

Runs one isolated process. Compare drafts 2/3 to a completed baseline directory.
All outputs must be inside the repository's ignored .q3x-work tree.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import qualify_service as q


class Client(q.Client):
    def request(self, body, endpoint='completions', label=None, delay=0):
        if not body.get('stream'):
            return super().request(body, endpoint, label, delay)
        payload = json.dumps(body, ensure_ascii=False).encode()
        started = time.monotonic(); events = []; pieces = []
        first = None; usage = None; finish = None; done = False
        req = urllib.request.Request(self.base + '/v1/' + endpoint, data=payload, headers=self.headers())
        with urllib.request.urlopen(req, timeout=1200) as response:
            for raw in response:
                if not raw.startswith(b'data:'): continue
                data = raw[5:].strip(); at = time.monotonic() - started
                if data == b'[DONE]': done = True; break
                event = json.loads(data); q.require('error' not in event, str(event))
                events.append({'at_s': at, 'event': event})
                if event.get('usage'): usage = event['usage']
                for choice in event.get('choices', []):
                    text = choice.get('text', choice.get('delta', {}).get('content', ''))
                    if text:
                        if first is None: first = at
                        pieces.append(text)
                    if choice.get('finish_reason'): finish = choice['finish_reason']
        q.require(done and finish in ('stop', 'length') and usage, 'incomplete SSE/usage')
        q.require(0 < usage['completion_tokens'] <= body['max_tokens'], 'invalid output count')
        q.require(finish != 'length' or usage['completion_tokens'] == body['max_tokens'], 'truncated output')
        q.require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'bad usage')
        if isinstance(body.get('prompt'), list):
            q.require(usage['prompt_tokens'] == len(body['prompt']), 'prompt count mismatch')
        record = {'label': label, 'request_id': events[0]['event']['id'],
                  'request_sha256': hashlib.sha256(payload).hexdigest(), 'endpoint': endpoint,
                  'body': body, 'text': ''.join(pieces), 'usage': usage, 'finish': finish,
                  'done': done, 'ttft_s': first, 'elapsed_s': time.monotonic() - started}
        q.save(self.out / f'{label}.json', dict(record, events=events))
        self.records.append(record)
        return record


def audit(out, records, draft):
    receipts = {}
    for line in (out / 'server.log').read_text().splitlines():
        if not line.startswith('{'): continue
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if value.get('schema') == 'mtp-multirow-api-witness-v1':
            q.require(value['request_id'] not in receipts, 'duplicate request receipt')
            receipts[value['request_id']] = value
    rows = []
    for result in records:
        w = receipts.get(result['request_id'])
        q.require(w and w['engine_result'] and not w['cancelled'], 'missing/incomplete MTP receipt')
        q.require(w['request_body_sha256'] == result['request_sha256'], 'receipt request identity')
        q.require(w['prompt_tokens'] == result['usage']['prompt_tokens'] and
                  w['completion_tokens'] == result['usage']['completion_tokens'], 'receipt usage')
        q.require(len(w['generated_token_ids']) == w['completion_tokens'], 'token coverage')
        if isinstance(result['body'].get('prompt'), list):
            import struct
            ids = result['body']['prompt']
            digest = hashlib.sha256(struct.pack('<' + 'I' * len(ids), *ids)).hexdigest()
            q.require(digest == w['prompt_token_ids_u32le_sha256'], 'prompt identity')
        m = w['mtp']; q.require(m['enabled'] and m['initialized'] and not m['failed'], 'MTP state')
        q.require(m['draft_length'] == draft and w['target_prefill_complete'] and
                  m['verifier'] == 'multirow-live-reduction-v30', 'route identity')
        q.require(m['startup_free_bytes'] >= 8*1024**3, 'composed memory reserve')
        hits = m['draft_ordered_attention_steps']
        q.require(0 <= hits <= m['proposed'], 'draft ordered work count')
        if w['prompt_tokens'] >= 512:
            q.require(hits == m['proposed'], 'missing ordered draft route')
        panels = (w['prompt_tokens'] + 7999)//8000
        q.require(w['prefill_layer_hits'] == 64 and w['prefill_fill_hits'] == 64*panels and
                  w['prefill_drain_hits'] == 64*panels and w['prefill_fp8_launches'] == 208*panels and
                  w['prefill_gdn_hits'] == 48 and w['prefill_attention_hits'] == 16, 'Prefill physical coverage')
        q.require(w['prefill_mlp_launches'] == 64*(1+(w['prompt_tokens']>=64)+(w['prompt_tokens']%64 != 0)), 'MLP coverage')
        q.require(m['draft_prefill_rows'] == w['prompt_tokens'] - 1, 'shifted prefix rows')
        q.require(m['committed_decode_tokens'] == w['completion_tokens'] - 1, 'committed token count')
        q.require(sum(m['accepted_by_position']) == m['accepted'] <= m['proposed'], 'acceptance count')
        q.require(m['verified_rows'] == m['proposed'] + m['rounds'], 'physical verifier count')
        q.require(m['verified_rows'] >= m['committed_decode_tokens'], 'verifier coverage')
        q.require(w['prefill_ms'] >= m['draft_initialization_ms'] >= 0, 'draft Prefill excluded')
        rows.append({'label': result['label'], 'prompt': w['prompt_tokens'], 'output': w['completion_tokens'],
                     'prefill_s': w['prefill_ms'] / 1000, 'prefill_tps': w['prompt_tokens'] * 1000 / w['prefill_ms'],
                     'ttft_s': result['ttft_s'], 'elapsed_s': result['elapsed_s'],
                     'decode_tps': m['committed_decode_tokens'] * 1000 / w['decode_ms'] if w['decode_ms'] else None,
                     'mtp': m})
    cancelled = [w for w in receipts.values() if w.get('cancelled')]
    q.require(any(not w['mtp']['initialized'] for w in cancelled), 'missing Prefill cancellation receipt')
    q.require(any(w['mtp']['initialized'] and w['mtp']['committed_decode_tokens'] > 0
                  for w in cancelled), 'missing Decode cancellation receipt')
    health = json.loads((out / 'health.json').read_text())['q3x_production']
    q.require(health['capacity']['max_sequence_length'] == 44095 and
              health['capacity']['maximum_output_tokens'] == 4096 and
              health['capacity']['request_arena_bytes'] == 9508218624 and
              health['retained_acceleration_sidecar_bytes'] == 11013898240 and
              health['decode']['graph_cache']['slots'] == 25 and
              not health['production_eligible'], 'MTP inventory/capacity identity')
    q.save(out / 'audited-metrics.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--prompt-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft-length', type=int, choices=[2, 3])
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--control-pid', type=int, action='append', default=[])
    args = parser.parse_args(); out = args.output.resolve()
    q.require(out.is_relative_to(q.ROOT / '.q3x-work'), 'output must be under .q3x-work')
    q.require(bool(args.baseline) == bool(args.draft_length), 'MTP run requires a baseline')
    out.mkdir(parents=True, exist_ok=False)
    q.preflight(out, args.control_pid)
    prompt = json.loads(args.prompt_request.read_text())['prompt']
    q.require(len(prompt) >= 40000, 'real prompt fixture too short')
    q.PROFILE = 'q3x.sm87.admission.mtp-live-reduction-api.v30' if args.draft_length else q.PROFILE
    key = os.urandom(24).hex(); keyfile = out / 'api-key'; keyfile.write_text(key); keyfile.chmod(0o600)
    cmd = [str(args.server.resolve()), str(args.model_dir.resolve()), '--port', '18872', '--api-key-file', str(keyfile)]
    if args.draft_length: cmd += ['--candidate-profile', 'whole-core-exact-decode']
    env = {k: v for k, v in os.environ.items() if not k.startswith(('Q3X_', 'VLLM_'))}
    if args.draft_length: env['Q3X_MTP_DRAFT_LENGTH'] = str(args.draft_length)
    for k, name in [('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cache/cuda'), ('XDG_CACHE_HOME', 'cache')]:
        p = out / name; p.mkdir(parents=True, exist_ok=True); env[k] = str(p)
    q.save(out / 'identity.json', {'command': cmd, 'binary_sha256': q.digest(args.server),
           'draft_length': args.draft_length, 'driver_sha256': q.digest(Path(__file__)),
           'prompt_fixture_sha256': q.digest(args.prompt_request),
           'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'scope': 'whole-service multi-row MTP direction; no release authority'})
    log = (out / 'server.log').open('w')
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    monitor = subprocess.Popen(['/usr/bin/tegrastats', '--interval', '1000'], stdout=subprocess.PIPE, text=True)
    temperatures = []; errors = []
    def telemetry():
        try:
            with (out / 'telemetry.log').open('w') as stream:
                for line in monitor.stdout:
                    line = re.sub(r'\b\S*fan\S*\s+\S+', '', line, flags=re.I)
                    stream.write(line); stream.flush()
                    values = [float(x) for x in re.findall(r'@([\d.]+)C', line)]; temperatures.extend(values)
                    if values and max(values) > 90: raise RuntimeError('thermal stop >90C')
        except Exception as error:
            errors.append(str(error))
            if server.poll() is None: os.killpg(server.pid, signal.SIGTERM)
    thread = threading.Thread(target=telemetry, daemon=True); thread.start()
    client = Client(18872, out, key); success = False
    try:
        deadline = time.monotonic() + 900
        while True:
            q.require(server.poll() is None, 'server exited before ready')
            try: health = client.health(); break
            except (OSError, urllib.error.URLError): pass
            q.require(time.monotonic() < deadline, 'startup deadline'); time.sleep(1)
        q.save(out / 'health.json', health); print('ready', flush=True)
        cases = [(65, 16, True, 'p65'), (8192, 256, True, 'p8192'),
                 (40000, 256, True, 'p40000'), (65, 16, False, 'p65-nonstream')]
        for p, o, stream, label in cases:
            result = client.request(q.body_for(prompt[:p], o, stream), label=label)
            # The receipt follows generation; await only its buffered log publication.
            deadline = time.monotonic() + 5
            while True:
                witnesses = []
                for line in (out / 'server.log').read_text().splitlines():
                    if not line.startswith('{'): continue
                    try: witness = json.loads(line)
                    except json.JSONDecodeError: continue
                    if witness.get('request_id', witness.get('request', {}).get('id')) == result['request_id']: witnesses.append(witness)
                if witnesses: break
                q.require(time.monotonic() < deadline, 'missing receipt'); time.sleep(.05)
            w = witnesses[-1]
            prefill = w['prefill_ms'] if args.draft_length else w['timing']['pure_prefill']['milliseconds']
            decode = w['decode_ms'] if args.draft_length else w['timing']['decode']['milliseconds']
            print(json.dumps({'stage': label, 'prompt': p, 'output': result['usage']['completion_tokens'],
                'prefill_s': prefill/1000, 'prefill_tps': p*1000/prefill,
                'decode_tps': (result['usage']['completion_tokens']-1)*1000/decode if decode else None,
                'ttft_s': result['ttft_s'], 'elapsed_s': result['elapsed_s'], 'mtp': w.get('mtp')}), flush=True)
            if args.baseline:
                base = next(r for r in json.loads((args.baseline / 'results.json').read_text()) if r['label']==label)
                for k in ('request_sha256', 'text', 'usage', 'finish', 'done'):
                    q.require(result[k] == base[k], 'baseline mismatch '+label+' '+k)
            q.save(out / 'results.json', client.records)
            if p >= 8192: time.sleep(30)
        client.request(q.body_for('The capital of France is', 16), label='raw-text')
        chat = {'model': q.MODEL_ID, 'messages': [{'role': 'user', 'content': '只回答数字1，不要解释。'}],
                'temperature': 0, 'max_tokens': 32, 'stream': True, 'stream_options': {'include_usage': True}}
        client.request(chat, endpoint='chat/completions', label='chat-stop')
        client.error(q.body_for(prompt[:65], 16), 401, {'Content-Type': 'application/json'})
        client.error(q.body_for(prompt[:65], 4097), 400)
        client.disconnect(q.body_for(prompt[:8192], 256), prefill=True)
        client.request(q.body_for(prompt[:65], 16), label='after-prefill-cancel')
        client.disconnect(q.body_for(prompt[:65], 256))
        client.request(q.body_for(prompt[:65], 16), label='after-decode-cancel')
        base = next(r for r in client.records if r['label'] == 'p65')
        for r in client.records:
            if r['label'] in ('p65-nonstream', 'after-prefill-cancel', 'after-decode-cancel'):
                for k in ('text', 'usage', 'finish'): q.require(r[k] == base[k], 'stream/recovery mismatch ' + k)
        if args.baseline:
            before = {r['label']: r for r in json.loads((args.baseline / 'results.json').read_text())}
            for result in client.records:
                for k in ('request_sha256', 'text', 'usage', 'finish', 'done'):
                    q.require(result[k] == before[result['label']][k], 'baseline mismatch ' + result['label'] + ' ' + k)
        success = True
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try: server.wait(timeout=60)
            except subprocess.TimeoutExpired: os.killpg(server.pid, signal.SIGKILL); server.wait(timeout=10)
        monitor.terminate(); monitor.wait(timeout=5); thread.join(timeout=5); log.close(); keyfile.unlink(missing_ok=True)
        q.save(out / 'results.json', client.records)
        q.save(out / 'cleanup.json', {'success': success, 'returncode': server.returncode,
                                    'errors': errors, 'max_temperature': max(temperatures) if temperatures else None})
    q.require(success and server.returncode == 0 and not errors, 'run/cleanup failure')
    if args.draft_length: audit(out, client.records, args.draft_length)
    print('integration checks passed', flush=True)


if __name__ == '__main__':
    main()
