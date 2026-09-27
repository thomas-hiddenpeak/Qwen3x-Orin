"""Freeze the bounded admission after every named run has closed."""
import hashlib
import json
import math
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
WORK = ROOT / '.q3x-work/decode-product-20260927'
DEST = ROOT / 'docs/metadata/qwen36-27b-decode-product-admission-2026-09-27.json'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

runs = []
for name in ('native-shared-regression', 'fused-shared-regression',
             'native-installed-regression', 'fused-output-long'):
    directory = WORK / name
    cleanup = read(directory / 'cleanup.json')
    assert cleanup['server_returncode'] == 0
    requests = []
    for item in read(directory / 'results.json'):
        events = read(directory / f"events-{item['prompt_tokens']}.json")
        times = [event['s'] for event in events
                 if any(choice.get('text') for choice in event['event'].get('choices', []))]
        intervals = sorted(1000 * (b-a) for a, b in zip(times, times[1:]))
        assert item['done'] and len(times) == item['usage']['completion_tokens']
        finish_reasons = [c['finish_reason'] for e in events for c in e['event'].get('choices', []) if c.get('finish_reason')]
        assert finish_reasons == ['length']
        assert item['usage']['prompt_tokens'] == item['prompt_tokens']
        requests.append({**item,
                         'decode_tokens_per_second': 1000/item['decode_ms_per_token'],
                         'itl_nearest_rank_p95_ms': intervals[math.ceil(.95*len(intervals))-1],
                         'maximum_itl_ms': max(intervals)})
    runs.append({'name': name, 'identity': read(directory/'identity.json'),
                 'health': read(directory/'health.json'), 'requests': requests,
                 'cache_drop_rc': read(directory/'cache-preparation.json')['rc'],
                 'exit_code': cleanup['server_returncode'],
                 'max_temperature_c': cleanup['max_temp_c']})

baseline, fused, installed, long_run = runs
assert len(baseline['requests']) == len(fused['requests']) == len(installed['requests']) == 3
for b, c, i in zip(baseline['requests'], fused['requests'], installed['requests']):
    assert b['request_sha256'] == c['request_sha256'] == i['request_sha256']
    assert b['text'] == i['text']
assert long_run['requests'][0]['usage']['completion_tokens'] == 256
historical_prefix_matches = {}
for n in (1089, 40000):
    old_events = read(ROOT/f'.q3x-work/decode-context-investigation-20260927/native/events-{n}.json')
    pieces = [choice['text'] for event in old_events
              for choice in event['event'].get('choices', []) if choice.get('text')]
    current = next(item for item in baseline['requests'] if item['prompt_tokens'] == n)
    historical_prefix_matches[str(n)] = ''.join(pieces[:16]) == current['text']
assert all(historical_prefix_matches.values())

source_paths = ['CMakeLists.txt', 'include/q3x/runtime/reference_runner.h',
                'src/runtime/reference_runner.cpp', 'src/runtime/reference_engine.cpp',
                'src/runtime/decode_fused_gqa_internal.h', 'src/kernels/sm87/decode_fused_gqa.cu',
                'src/kernels/reference/decode_ops.cu', 'src/kernels/reference/projection_dispatch.cpp',
                'src/server/evaluation_server.cpp', 'src/server/openai_protocol.cpp',
                'tests/fused_decode_test.cu', 'tests/fused_decode_model_test.cpp',
                'tests/openai_protocol_test.cpp', 'tests/fused_decode_protocol_test.cpp',
                'tests/reference_ordinary_generation_capture_test.cpp',
                'third_party/flashinfer/README.q3x.md',
                'third_party/flashinfer/include/flashinfer/attention/prefill.cuh']
graph = {}
for line in (WORK/'graph-shared-r1/process.log').read_text().splitlines():
    if line.startswith('decode_graph_production.') and '=' in line:
        key, value = line.split('=', 1)
        graph[key] = value
assert graph['decode_graph_production.status'] == 'pass'

record = {
    'schema': 'q3x.decode-product-admission.v1',
    'date': '2026-09-27',
    'source_base': '468b2938390f14b5af85fbb51944a79bd892e226',
    'decision_unit': 'bounded architecture admission; no release candidate',
    'architecture_candidate_id': 'AC-DECODE-FUSED-GQA-PRODUCT-20260927',
    'decisions': {
        'ordinary_scalar_arithmetic': 'retained',
        'old_split_dispatch': 'unlinked and removed from public launcher',
        'graph_shared_executable': 'bounded correctness/resource retention; installed API integration passed',
        'fused_v1': 'rejected by predeclared synthetic independent-FP64 admission',
        'fused_split_p_v2': 'non-installable accuracy-unqualified dependency; positive Legacy API direction only',
        'production_numerical_contract_changed': False,
        'whole_core_composition': 'not implemented; remains fixed P40000/O16',
        'release_qualification': False,
        'decode_10_tokens_per_second_target': 'open'},
    'implementation_source_sha256': {path: sha(ROOT/path) for path in source_paths},
    'admission_contract': read(WORK/'admission-contract.json'),
    'model_identity': read(WORK/'model-identity.json'),
    'default_final_build_bridge': read(WORK/'default-final-build-bridge.json'),
    'current_numerics': 'split-P BF16 hi/residual, FP32 QK/online/MMA, BF16 partial and final publication; unqualified',
    'numerical_summary': read(WORK/'numerical-summary.json'),
    'graph_production_test': graph,
    'final_witness_smoke': read(WORK/'final-witness-smoke.json'),
    'postflight_context': read(WORK/'postflight-context.json'),
    'ordinary_route_sealing': read(WORK/'ordinary-route-sealing.json'),
    'build_guards': read(WORK/'build-guards.json'),
    'env_rejection': read(WORK/'admission-env-rejection.json'),
    'api_runs': runs,
    'prechange_first_16_content_events_match': historical_prefix_matches,
    'authority_limits': [
        'P1089 numerical admission does not qualify long-context logits/state or public model capability.',
        'Full-vocabulary hashes detect differences, not error magnitude; no general teacher-forcing harness.',
        'Legacy API only. Do not add these Decode rates to whole-core Prefill measurements.',
        'B-C-B independent-process direction with an additional long-output candidate, not mirrored release qualification.',
        'CPU compilation overlapped ordinary initial P40 Prefill; no Prefill speedup claim.',
        'First ordinary API identity scope string said compiled fused admission; its ELF/health show scalar OFF build.',
        'Compute-sanitizer unavailable on this device, not passed.',
        'Earlier fused O16 ready log retained the configured Legacy profile name while health and decode route identified fusion; final O256 artifact corrects that log field.',
        'Numeric/Graph test binaries precede final comment/format-only changes and startup-log identity correction; algorithm unchanged, artifact hashes remain separate.',
        'Earlier v1 admission witness labels inherited fallback counts exact; final v2 removes that false qualification and has a separate host/API smoke.',
        'No completed fresh vLLM parity, cancellation, public capability suite or production fusion promotion.'],
    'invalid_or_unmatched_attempts': {
        'native-r1/native-r2': 'resource gate failure before readiness; no API timing',
        'fused-startup-diagnostic': 'missing production sidecars; unmatched short requests; P40 aborted; owned SIGKILL closure recorded',
        'fused-sealed-startup-diagnostic/fused-kernel-startup-diagnostic': 'complete-layout startup diagnostics; resource gate failure',
        'contract-test.log/contract-test-v2.log': 'initial launch/offset errors; corrected before accepted T1 run',
        'contract-test-r2.log': 'v1 numerical gate failure',
        'protocol-test-ordinary.log': 'testing fixture manually linked to OFF gateway; inapplicable inventory expectations',
        'protocol-test-testing.log/host-tests.log': 'stale O1 fixture; actual contract changed to O16 in 3c765164'},
    'raw_artifacts': {},
}
for path in sorted(WORK.rglob('*')):
    if not path.is_file() or any(part in ('guard-config', 'cmake-tmp', 'protocol-fixtures') for part in path.parts):
        continue
    if path.suffix in ('.json', '.jsonl', '.log', '.py', '.cpp', '.cu'):
        record['raw_artifacts'][str(path.relative_to(ROOT))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
DEST.write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
print(DEST)
