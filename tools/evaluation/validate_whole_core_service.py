#!/usr/bin/env python3
"""Audit retained whole-core service API runs; never promotes a route by itself."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def select_witness(result: dict, directory: Path, by_hash: dict, by_id: dict) -> dict:
    """Bind repeated request bodies to their actual response, never the last hash."""
    request_id = result.get('request_id')
    if not request_id and result.get('label'):
        response = directory / (result['label'] + '.json')
        require(response.resolve().parent == directory.resolve(), 'response label escapes run directory')
        if response.is_file():
            events = json.loads(response.read_text()).get('events', [])
            request_id = events[0].get('id') if events else None
    if request_id:
        require(request_id in by_id, 'response has no matching request witness')
        witness = by_id[request_id]
    else:
        matches = by_hash.get(result['request_sha256'], [])
        require(len(matches) == 1, 'ambiguous or missing request witness; response ID required')
        witness = matches[0]
    require(witness['request']['body_sha256'] == result['request_sha256'],
            'response ID/request body mismatch')
    return witness


def validate_prompt_identity(result: dict, witness: dict) -> None:
    prompt = result.get('body', {}).get('prompt')
    expected = len(prompt) if isinstance(prompt, list) else result.get('prompt_tokens')
    if expected is not None:
        require(witness['prompt']['tokens'] == expected, 'client prompt length mismatch')
    if isinstance(prompt, list):
        canonical = struct.pack(f'<{len(prompt)}I', *prompt)
        require(hashlib.sha256(canonical).hexdigest() ==
                witness['prompt']['token_ids_u32le_sha256'], 'client prompt token identity mismatch')


def audit_run(directory: Path) -> dict:
    identity = json.loads((directory / 'identity.json').read_text())
    health = json.loads((directory / 'health.json').read_text())['q3x_production']
    cleanup = json.loads((directory / 'cleanup.json').read_text())
    cache = json.loads((directory / 'cache-preparation.json').read_text())
    require(cache['rc'] == 0, 'cold-cache preparation failed')
    require(cleanup['server_returncode'] == 0, 'unclean server exit')
    require(not cleanup.get('clock_errors'), 'clock contract failure')
    require(cleanup['max_temp_c'] <= 90, 'thermal stop')
    clocks = [json.loads(line) for line in
              (directory / 'clocks.jsonl').read_text().splitlines()]
    require(bool(clocks) and all(sample['gpu_cur_freq'] == 1300500000 and
                                sample['gpu_min_freq'] == 1300500000 and
                                sample['gpu_max_freq'] == 1300500000
                                for sample in clocks), 'GPU clock mismatch')
    production = health['profile'] == 'q3x.sm87.production.whole-core-service.v1'
    require(production or health['profile'] ==
            'q3x.sm87.admission.whole-core-exact-decode.v9', 'wrong profile')
    if production:
        require(health['BUILD_TESTING'] is False and health['production_eligible'],
                'installed production identity mismatch')
        require(health['capacity']['max_sequence_length'] == 44095 and
                health['capacity']['maximum_output_tokens'] == 4096 and
                health['capacity']['request_arena_bytes'] == 9508218624,
                'wrong installed capacity')
        require(health['decode']['route'] ==
                'fixed-gqa-ordered-pipeline-s512-44095.v7' and
                health['decode']['graph_cache']['slots'] == 25,
                'wrong Decode inventory')
    witnesses = [json.loads(line) for line in (directory / 'server.log').read_text().splitlines()
                 if line.startswith('{"record":"target-prefill-witness')]
    require(bool(witnesses), 'missing request witnesses')
    by_hash = {}; by_id = {}
    for witness in witnesses:
        prompt, prefill = witness['prompt'], witness['prefill']
        rows = prompt['tokens']
        outputs = witness['completion']['tokens']
        require(1 <= rows and 1 <= outputs <= 4096 and
                rows + outputs - 1 <= 44095, 'invalid sequence accounting')
        panels = (rows + 7999) // 8000
        require(prompt['fully_consumed'] and prompt['consumed_tokens'] == rows,
                'prompt truncation')
        require(prefill['package_complete'] and
                prefill['prompt_wide_p40_whole_core_package']['complete'], 'incomplete package')
        expected = {'logical_panel_count': panels,
                    'submission_window_retirements': 64 * (2 * panels + 2),
                    'prompt_wide_p40_whole_core_layer_hits': 64,
                    'prompt_wide_p40_fill_panel_hits': 64 * panels,
                    'prompt_wide_p40_drain_panel_hits': 64 * panels,
                    'prompt_wide_p40_fp8_projection_physical_launches': 208 * panels,
                    'prompt_wide_p40_bf16_ab_hits': 48,
                    'prompt_wide_p40_gdn_hits': 48,
                    'native_flashinfer_exact_whole_prompt_hits': 16,
                    'persistent_p40_nvfp4_physical_launches':
                        64 * (1 + int(rows >= 64) + int(rows % 64 != 0))}
        require(all(prefill.get(key) == value for key, value in expected.items()),
                'actual-row physical receipt mismatch')
        hits = witness['route']['per_operator_route_hits']
        require(hits['complete'] and not any(hits['forbidden_route_hits'].values()),
                'forbidden or incomplete route')
        for counts in hits['operators'].values():
            require(not counts['completed_forbidden_hits'] and
                    not counts.get('completed_exact_fallback_hits',
                                   counts.get('completed_fallback_hits_unqualified')),
                    'unexpected fallback')
        plan = witness['route']['deployment_plan']
        if production:
            require(plan['id'] == 'q3x.sm87.production.whole-core-prefill.v1' and
                    plan['numerical_contract']['qualified'], 'wrong numerical plan')
            require(witness['decode_numerical_contract'] ==
                    'scalar-equivalent-ordered-fp32-v7', 'wrong Decode arithmetic')
        by_hash.setdefault(witness['request']['body_sha256'], []).append(witness)
        request_id = witness['request']['id']
        require(request_id not in by_id, 'duplicate request witness ID')
        by_id[request_id] = witness
    records = []
    results_path = directory / 'results.json'
    for result in json.loads(results_path.read_text()) if results_path.exists() else []:
        witness = select_witness(result, directory, by_hash, by_id)
        validate_prompt_identity(result, witness)
        p, o = result['usage']['prompt_tokens'], result['usage']['completion_tokens']
        require(result['done'] and witness['prompt']['tokens'] == p and
                witness['completion']['tokens'] == o, 'usage mismatch')
        prefill_ms = witness['timing']['pure_prefill']['milliseconds']
        decode_ms = witness['timing']['decode']['milliseconds']
        require(math.isfinite(prefill_ms) and prefill_ms > 0 and
                math.isfinite(decode_ms) and (decode_ms > 0 if o > 1 else decode_ms >= 0),
                'invalid phase timing')
        records.append({'request_id': witness['request']['id'], 'prompt_tokens': p, 'output_tokens': o,
                        'prefill_seconds': prefill_ms / 1000,
                        'prefill_tok_s': p * 1000 / prefill_ms,
                        'external_ttft_seconds': result['ttft_s'],
                        'decode_tok_s': (o - 1) * 1000 / decode_ms if o > 1 else None,
                        'request_sha256': result['request_sha256'],
                        'text_sha256': hashlib.sha256(result['text'].encode()).hexdigest()})
    return {'directory': str(directory), 'profile': health['profile'],
            'executable_sha256': identity['executable_sha256'],
            'complete_witnesses': len(witnesses), 'metrics': records,
            'scope': 'artifact/route/API audit; selection requires the frozen qualification protocol'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [audit_run(path) for path in args.runs]
    args.output.write_text(json.dumps({'runs': results}, indent=2) + '\n')
    print(f'PASS {len(results)} runs; {sum(r["complete_witnesses"] for r in results)} complete witnesses')


if __name__ == '__main__':
    main()
