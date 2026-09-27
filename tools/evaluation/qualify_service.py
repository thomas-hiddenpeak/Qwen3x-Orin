#!/usr/bin/env python3
"""Versioned installed-service validation. No synthetic performance fallback.

Explicit external inputs: pinned model, real token-ID prompt, optional long
prompt, capability cases and pinned reference answers. All generated evidence
is confined to .q3x-work. A smoke run never claims full qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'q3x.sm87.production.whole-core-service.v1'
MODEL_ID = 'qwen3.6-27b-nvfp4'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def resource_snapshot(pid):
    status = Path(f'/proc/{pid}/status').read_text()
    return {'time': time.monotonic(), 'rss_kib': int(re.search(r'VmRSS:\s+(\d+)', status)[1]),
            'threads': int(re.search(r'Threads:\s+(\d+)', status)[1]),
            'fds': len(list(Path(f'/proc/{pid}/fd').iterdir()))}


class Client:
    def __init__(self, port, out, key):
        self.port, self.out, self.key = port, out, key
        self.base = f'http://127.0.0.1:{port}'
        self.records = []

    def headers(self):
        return {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key}

    def health(self):
        with urllib.request.urlopen(self.base + '/healthz', timeout=3) as response:
            value = json.load(response)
        require(value['q3x_production']['profile'] == PROFILE, 'wrong service profile')
        return value

    def request(self, body, endpoint='completions', label=None, delay=0):
        payload = json.dumps(body, ensure_ascii=False).encode()
        start = time.monotonic(); events = []; pieces = []; usage = None
        finish = None; done = False; first = None
        req = urllib.request.Request(self.base + '/v1/' + endpoint, data=payload, headers=self.headers())
        with urllib.request.urlopen(req, timeout=1200) as response:
            if body.get('stream'):
                for raw in response:
                    if delay: time.sleep(delay)
                    line = raw.decode('utf-8').strip()
                    if not line.startswith('data:'): continue
                    data = line[5:].strip()
                    if data == '[DONE]': done = True; break
                    event = json.loads(data); events.append(event)
                    require('error' not in event, str(event))
                    if event.get('usage'): usage = event['usage']
                    for choice in event.get('choices', []):
                        text = choice.get('text', choice.get('delta', {}).get('content', ''))
                        if text:
                            if first is None: first = time.monotonic() - start
                            pieces.append(text)
                        if choice.get('finish_reason'): finish = choice['finish_reason']
                require(done and finish in ('stop', 'length'), 'incomplete SSE/finish')
                if body.get('stream_options', {}).get('include_usage'):
                    require(usage, 'missing SSE usage')
                else: require(usage is None, 'unexpected SSE usage')
            else:
                event = json.load(response); events.append(event)
                require('error' not in event, str(event))
                choice = event['choices'][0]
                pieces.append(choice.get('text', choice.get('message', {}).get('content', '')))
                usage, finish, done = event['usage'], choice['finish_reason'], True
        if usage:
            if isinstance(body.get('prompt'), list):
                require(usage['prompt_tokens'] == len(body['prompt']), 'input tokens were truncated or miscounted')
            require(0 < usage['completion_tokens'] <= body['max_tokens'], 'invalid output accounting')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'invalid total usage')
            require(finish != 'length' or usage['completion_tokens'] == body['max_tokens'], 'truncated length finish')
        record = {'label': label, 'request_id': events[0]['id'], 'request_sha256': hashlib.sha256(payload).hexdigest(),
                  'endpoint': endpoint, 'body': body, 'text': ''.join(pieces), 'usage': usage,
                  'finish': finish, 'done': done, 'ttft_s': first, 'elapsed_s': time.monotonic() - start}
        if label:
            save(self.out / f'{label}.json', dict(record, events=events))
        self.records.append(record)
        with (self.out / 'requests.jsonl').open('a') as f:
            f.write(json.dumps({k: v for k, v in record.items() if k != 'body'}, ensure_ascii=False) + '\n')
        return record

    def error(self, body, expected, headers=None):
        req = urllib.request.Request(self.base + '/v1/completions', data=json.dumps(body).encode(),
                                     headers=self.headers() if headers is None else headers)
        try:
            urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as error:
            data = error.read().decode(); require(error.code == expected, f'expected {expected}, got {error.code}: {data}')
            return
        raise RuntimeError('invalid request accepted')

    def disconnect(self, body, prefill=False):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=240)
        connection.request('POST', '/v1/completions', body=json.dumps(body), headers=self.headers())
        try:
            if prefill:
                time.sleep(.35)
            else:
                response = connection.getresponse(); seen = 0
                for raw in response:
                    if raw.startswith(b'data:') and raw[5:].strip() != b'[DONE]':
                        event = json.loads(raw[5:])
                        require('error' not in event, 'disconnect request error')
                        if any(c.get('text') for c in event.get('choices', [])): seen += 1
                        if seen == 3: break
                require(seen == 3, 'not enough tokens for Decode disconnect')
        finally:
            connection.close()


def body_for(prompt, output=16, stream=True):
    return {'model': MODEL_ID, 'prompt': prompt, 'max_tokens': output,
            'temperature': 0, 'stream': stream,
            **({'stream_options': {'include_usage': True}} if stream else {})}


def functional_panel(client, prompt, cycles):
    base = body_for(prompt[:65], 4, False)
    baseline = client.request(base, label='functional-baseline')
    for include in (False, True):
        streamed = client.request(dict(base, stream=True, stream_options={'include_usage': include}), label=f'usage-{include}')
        require(streamed['text'] == baseline['text'], 'SSE/nonstream output differs')
        if include: require(streamed['usage'] == baseline['usage'], 'SSE/nonstream usage differs')
    client.error(base, 401, {'Content-Type': 'application/json'})
    for updates in ({'prompt': []}, {'max_tokens': -1}, {'max_tokens': 4097},
                    {'prompt': [248320]}, {'temperature': .5}, {'unknown': True}):
        client.error(dict(base, **updates), 400)
    # Incomplete headers/bodies must not hold response threads or health.
    slow = [socket.create_connection(('127.0.0.1', client.port)) for _ in range(12)]
    try:
        for i, conn in enumerate(slow):
            if i % 2: conn.sendall(b'POST /v1/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\nx')
        start = time.monotonic(); client.health()
        require(time.monotonic() - start < 2, 'slow clients starved readiness')
        client.request(base, label='with-slow-clients')
    finally:
        for conn in slow: conn.close()
    languages = ['请输出汉字：中国。', '日本語で短く挨拶してください。', '한국어로 짧게 인사하세요.',
                 'Réponds en français : bonjour.', 'قل مرحبا بالعربية.', 'Reply with an emoji.']
    for index, text in enumerate(languages):
        for cap in (1, 2, 3, 16):
            common = {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': text}],
                      'max_tokens': cap, 'temperature': 0}
            one = client.request(dict(common, stream=False), 'chat/completions', f'language-{index}-{cap}-plain')
            two = client.request(dict(common, stream=True, stream_options={'include_usage': True}),
                                 'chat/completions', f'language-{index}-{cap}-sse')
            require(one['text'] == two['text'] and one['usage'] == two['usage'], 'multilingual stream parity')
    client.request(body_for(prompt[:513], 64), label='slow-reader', delay=.2)
    for i in range(cycles):
        client.disconnect(body_for(prompt[:8192], 256), prefill=True)
        after = client.request(base)
        require(after['text'] == baseline['text'] and after['usage'] == baseline['usage'], 'Prefill cancellation polluted next request')
        client.disconnect(body_for(prompt[:513], 256))
        after = client.request(base)
        require(after['text'] == baseline['text'] and after['usage'] == baseline['usage'], 'Decode cancellation polluted next request')
        client.health()
        print(f'cancellation cycle {i + 1}/{cycles}', flush=True)
    # Explicit active + one queued request, then reject third before execution.
    active = http.client.HTTPConnection('127.0.0.1', client.port, timeout=240)
    active.request('POST', '/v1/completions', body=json.dumps(body_for(prompt[:8192], 256)), headers=client.headers())
    time.sleep(.5); queued = []; errors = []
    def queue_request():
        try: queued.append(client.request(base))
        except Exception as error: errors.append(repr(error))
    worker = threading.Thread(target=queue_request); worker.start(); time.sleep(.5)
    try: client.error(base, 429)
    finally: active.close()
    worker.join(240)
    require(not worker.is_alive() and not errors and len(queued) == 1, 'queue recovery failed')
    require(queued[0]['text'] == baseline['text'], 'queued output changed')
    return {'multilingual_parity_pairs': 24, 'cancel_recovery_cycles': cycles,
            'slow_ingress_clients': 12, 'queue_overload_http': 429}


def preflight(out, allow):
    cmd = ['sudo', '-n', sys.executable, '-B', str(ROOT / 'tools/evaluation/orin_perf_preflight.py'),
           '--output', str(out / 'preflight.json'), '--samples', '5']
    for pid in [os.getpid(), *allow]: cmd += ['--allow-pid', str(pid)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out / 'preflight-console.log').write_text(result.stdout + result.stderr)
    # The privileged /proc audit writes owner-only evidence; return this exact
    # project-owned file to the invoking user before reading it.
    if (out / 'preflight.json').exists():
        subprocess.run(['sudo', '-n', 'chown', f'{os.getuid()}:{os.getgid()}',
                        str(out / 'preflight.json')], check=True)
    require(result.returncode == 0, 'preflight rejected; inspect retained evidence')
    report = json.loads((out / 'preflight.json').read_text())
    require(not report['gpu_device_fd_audit']['holders'], 'GPU already owned')
    before = Path('/proc/meminfo').read_text(); subprocess.run(['sync'], check=True)
    drop = subprocess.run(['sudo', '-n', 'tee', '/proc/sys/vm/drop_caches'], input='3\n', capture_output=True, text=True)
    save(out / 'cache-preparation.json', {'before': before, 'after': Path('/proc/meminfo').read_text(),
                                       'rc': drop.returncode, 'stderr': drop.stderr})
    require(drop.returncode == 0, 'cache preparation failed')


def run(args):
    out = args.output.resolve(); out.relative_to(ROOT / '.q3x-work'); out.mkdir(parents=True, exist_ok=False)
    prompt = json.loads(args.prompt_request.read_text())['prompt']
    require(isinstance(prompt, list) and len(prompt) >= 8192, 'real prompt fixture must provide >=8192 token IDs')
    if args.mode == 'qualify':
        require(len(prompt) >= 40000 and args.long_prompt_request and args.capability_cases and args.reference_results,
                'qualification requires P40/long prompt, capability and reference fixtures')
        require(args.soak_seconds >= 7200 and args.cancel_cycles >= 8, 'qualification needs 2h soak and 8 cancellation cycles')
    preflight(out, args.allow_pid)
    key = os.urandom(24).hex(); key_path = out / 'api-key'; key_path.write_text(key); key_path.chmod(0o600)
    command = [str(args.server.resolve()), str(args.model_dir.resolve()), '--port', str(args.port), '--api-key-file', str(key_path)]
    inputs = {name: {'path': str(value.resolve()), 'sha256': digest(value)} for name, value in
              vars(args).items() if isinstance(value, Path) and value.is_file()}
    save(out / 'identity.json', {'command': command, 'executable_sha256': digest(args.server), 'inputs': inputs,
                               'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                               'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
                               'driver_sha256': digest(Path(__file__)),
                               'mode': args.mode, 'soak_seconds': args.soak_seconds,
                               'soak_interval_seconds': args.soak_interval_seconds,
                               'capacity_idle_seconds': 60})
    env = {k: v for k, v in os.environ.items() if not k.startswith(('Q3X_', 'VLLM_'))}
    for name, relative in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                           ('XDG_CACHE_HOME', 'cache')):
        location = out / relative; location.mkdir(parents=True, exist_ok=True)
        env[name] = str(location)
    save(out / 'process-environment.json', {name: env.get(name) for name in
         ('TMPDIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'CUDA_MODULE_LOADING', 'LD_LIBRARY_PATH')})
    log = (out / 'server.log').open('w')
    srv = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    monitor = subprocess.Popen(['/usr/bin/tegrastats', '--interval', '1000'], stdout=subprocess.PIPE, text=True)
    monitor_errors = []; temperatures = []; stop_monitor = threading.Event()
    def telemetry():
        try:
            with (out / 'telemetry.log').open('w') as f, (out / 'clocks.jsonl').open('w') as clocks:
                for line in monitor.stdout:
                    if stop_monitor.is_set(): break
                    line = re.sub(r'\b\S*fan\S*\s+\S+', '', line, flags=re.I); f.write(line); f.flush()
                    sample = {'time': time.monotonic()}
                    for name in ('min_freq', 'max_freq', 'cur_freq'):
                        sample['gpu_' + name] = int(Path('/sys/class/devfreq/17000000.gpu', name).read_text())
                    clocks.write(json.dumps(sample) + '\n'); clocks.flush()
                    temps = [float(x) for x in re.findall(r'@([\d.]+)C', line)]
                    temperatures.extend(temps)
                    if temps and max(temps) > 90:
                        raise RuntimeError('thermal stop >90C')
        except Exception as error:
            monitor_errors.append(str(error))
            if srv.poll() is None: os.killpg(srv.pid, signal.SIGTERM)
    thread = threading.Thread(target=telemetry, daemon=True); thread.start()
    client = Client(args.port, out, key); summary = {'passed': False}; success = False
    try:
        ready = False
        for _ in range(1200):
            require(srv.poll() is None, 'server exited before ready')
            try:
                health = client.health(); ready = True; break
            except (OSError, urllib.error.URLError): time.sleep(1)
        require(ready, 'startup timeout'); save(out / 'health.json', health)
        print('server ready', flush=True)
        if args.mode not in ('restart', 'performance'):
            summary['functional'] = functional_panel(client, prompt, args.cancel_cycles)
        elif args.mode == 'restart':
            client.request(body_for(prompt[:65], 4, False), label='functional-baseline')
            client.request(body_for(prompt[:513], 64), label='restart-slow-reader', delay=.2)
        if args.mode == 'performance':
            require(len(prompt) >= 40000, 'performance panel requires real P40000 input')
            for p in (1, 65, 513, 1089, 8192, 40000):
                client.request(body_for(prompt[:p], 256 if p >= 8192 else 16), label=f'performance-{p}')
        if args.mode == 'qualify':
            for p, o in [(8192, 256), (40000, 256), (40000, 4096)]:
                print(f'capacity request P{p}/O{o}', flush=True)
                client.request(body_for(prompt[:p], o), label=f'capacity-{p}-{o}')
                print('capacity inter-request idle 60s', flush=True)
                time.sleep(60)
            long_prompt = json.loads(args.long_prompt_request.read_text())['prompt']
            require(len(long_prompt) >= 44095, 'long fixture too short')
            client.request(body_for(long_prompt[:44095], 1), label='maximum-prompt')
            client.error(body_for(long_prompt[:44095], 2), 400)
            cases = json.loads(args.capability_cases.read_text()); reference = json.loads(args.reference_results.read_text())
            require(len(cases) == len(reference) == 98, 'expected pinned 98-question panel')
            answers = []
            for i, (case, prior) in enumerate(zip(cases, reference)):
                result = client.request(case['request'], 'chat/completions', f'capability-{i}')
                matches = re.findall(r'答案：([A-D])', result['text'])
                require(result['finish'] == 'stop' and matches, 'invalid capability answer')
                require(result['request_sha256'] == prior['request_sha256'], 'capability request identity mismatch')
                require(matches[-1] == prior['answer'], 'capability answer differs from pinned reference')
                require(result['usage']['prompt_tokens'] == prior['usage']['prompt_tokens'], 'capability prompt mismatch')
                answers.append(matches[-1] == case['target'])
                time.sleep(args.soak_interval_seconds)
                if i % 10 == 0: print(f'capability {i + 1}/98', flush=True)
            summary['capability'] = {'valid': 98, 'correct': sum(answers), 'reference_answers_equal': 98}
        soak_body = body_for(prompt[:513], 16)
        baseline = client.request(soak_body, label='soak-baseline')
        # Warm up transport/request reuse before memory plateau observations.
        for _ in range(8): client.request(soak_body)
        snapshots = [resource_snapshot(srv.pid)]; deadline = time.monotonic() + args.soak_seconds
        iterations = 0; next_report = time.monotonic()
        while time.monotonic() < deadline:
            result = client.request(soak_body)
            require(result['text'] == baseline['text'] and result['usage'] == baseline['usage'], 'soak output drift')
            iterations += 1
            time.sleep(args.soak_interval_seconds)
            if iterations % 10 == 0:
                snapshots.append(resource_snapshot(srv.pid)); client.health()
                save(out / 'resources.json', snapshots)
            if time.monotonic() >= next_report:
                print(f'soak {iterations} requests; {max(0, int(deadline-time.monotonic()))}s remaining', flush=True)
                next_report = time.monotonic() + 30
        snapshots.append(resource_snapshot(srv.pid)); save(out / 'resources.json', snapshots)
        growth = snapshots[-1]['rss_kib'] - snapshots[0]['rss_kib']
        require(growth <= 256 * 1024, 'RSS growth exceeds 256MiB plateau budget')
        require(snapshots[-1]['fds'] <= snapshots[0]['fds'] + 4, 'FD leak')
        require(snapshots[-1]['threads'] <= snapshots[0]['threads'] + 1, 'thread leak')
        if args.mode == 'qualify': require(iterations >= 200, 'insufficient sustained request reuse')
        summary['soak'] = {'seconds': args.soak_seconds, 'requests': iterations, 'rss_growth_kib': growth,
                           'inter_request_idle_seconds': args.soak_interval_seconds}
        client.health(); success = True
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGINT)
            try: srv.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL); srv.wait(timeout=10)
        stop_monitor.set(); monitor.terminate(); monitor.wait(timeout=5); thread.join(timeout=5); log.close()
        key_path.unlink(missing_ok=True)
        save(out / 'cleanup.json', {'server_returncode': srv.returncode, 'clock_errors': monitor_errors,
                                   'max_temp_c': max(temperatures) if temperatures else None})
        save(out / 'results.json', [r for r in client.records if r['label'] and r['usage']])
        summary.update(passed=success and srv.returncode == 0 and not monitor_errors and bool(temperatures),
                       requests=len(client.records), mode=args.mode)
        save(out / 'summary.json', summary)
    require(summary['passed'], 'service validation or cleanup failed')
    from validate_whole_core_service import audit_run
    try:
        audit = audit_run(out)
        save(out / 'service-audit.json', audit)
        summary['complete_witnesses'] = audit['complete_witnesses']
        summary['capacity_metrics'] = [m for m in audit['metrics'] if m['prompt_tokens'] >= 8192]
    except Exception:
        summary['passed'] = False
        save(out / 'summary.json', summary)
        raise
    save(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--prompt-request', type=Path, required=True)
    parser.add_argument('--long-prompt-request', type=Path)
    parser.add_argument('--capability-cases', type=Path)
    parser.add_argument('--reference-results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18871)
    parser.add_argument('--mode', choices=('smoke', 'qualify', 'restart', 'performance'), default='smoke')
    parser.add_argument('--soak-seconds', type=int, default=0)
    parser.add_argument('--soak-interval-seconds', type=float, default=1.0)
    parser.add_argument('--cancel-cycles', type=int, default=1)
    parser.add_argument('--allow-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    require(args.soak_seconds >= 0 and args.cancel_cycles >= 1 and 0 <= args.soak_interval_seconds <= 10, 'invalid validation duration/count')
    run(args)


if __name__ == '__main__':
    main()
