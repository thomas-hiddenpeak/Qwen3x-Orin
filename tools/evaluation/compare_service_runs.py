#!/usr/bin/env python3
"""Independent-process B-C-C-B installed-service comparison, with explicit inputs.

Use only when a work package authorizes performance selection. The result
checks the same generated text and usage, and both engine phase regressions;
it never promotes a numerical route or replaces full product qualification.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from qualify_service import ROOT, require, run, save


class PerformanceRegression(RuntimeError):
    """Valid measurement rejects the predeclared performance gate (exit 3)."""


def compare_runs(directories, regression_percent=3.0):
    runs = []
    for directory in directories:
        report = json.loads((directory / 'service-audit.json').read_text())
        records = json.loads((directory / 'results.json').read_text())
        selected = {r['request_sha256']: r for r in records
                    if (r.get('label') or '').startswith('performance-')}
        by_id = {m['request_id']: m for m in report['metrics']}
        metrics = {key: by_id[record['request_id']] for key, record in selected.items()
                   if record.get('request_id') in by_id}
        require(len(metrics) == 6, 'missing performance panel')
        runs.append((selected, metrics))
    require(len(runs) == 4, 'B-C-C-B requires four independent runs')
    keys = set(runs[0][0])
    require(all(set(records) == keys for records, _ in runs), 'request bodies differ')
    rows = []
    for key in keys:
        base = runs[0][0][key]
        require(all(records[key]['text'] == base['text'] and
                    records[key]['usage'] == base['usage'] and
                    records[key]['finish'] == base['finish'] for records, _ in runs),
                'generated output or accounting differs')
        row = {'prompt_tokens': base['usage']['prompt_tokens'],
               'output_tokens': base['usage']['completion_tokens'], 'request_sha256': key}
        for phase in ('prefill_tok_s', 'decode_tok_s'):
            before = mean(runs[i][1][key][phase] for i in (0, 3))
            after = mean(runs[i][1][key][phase] for i in (1, 2))
            row[phase] = {'baseline': before, 'candidate': after, 'ratio': after / before}
            if after < before * (1 - regression_percent / 100):
                raise PerformanceRegression(f'{phase} regression at P{row["prompt_tokens"]}')
        ttft = [metrics[key]['external_ttft_seconds'] for _, metrics in runs]
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in ttft), 'missing or invalid external TTFT')
        before = mean(ttft[i] for i in (0, 3)); after = mean(ttft[i] for i in (1, 2))
        row['external_ttft_seconds'] = {'baseline': before, 'candidate': after, 'ratio': after / before}
        if after > before * (1 + regression_percent / 100):
            raise PerformanceRegression(f'external TTFT regression at P{row["prompt_tokens"]}')
        rows.append(row)
    return {'passed': True, 'order': 'B-C-C-B', 'regression_percent': regression_percent,
            'metrics': sorted(rows, key=lambda r: r['prompt_tokens'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-server', type=Path, required=True)
    parser.add_argument('--candidate-server', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--prompt-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18871)
    parser.add_argument('--allow-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    out = args.output.resolve(); out.relative_to(ROOT / '.q3x-work')
    out.mkdir(parents=True, exist_ok=False)
    directories = []
    for index, server in enumerate((args.baseline_server, args.candidate_server,
                                    args.candidate_server, args.baseline_server)):
        directory = out / f'run-{index + 1}'
        run(argparse.Namespace(server=server, model_dir=args.model_dir,
            prompt_request=args.prompt_request, long_prompt_request=None,
            capability_cases=None, reference_results=None, output=directory,
            port=args.port, mode='performance', soak_seconds=0, soak_interval_seconds=1., cancel_cycles=1,
            allow_pid=args.allow_pid))
        directories.append(directory)
    try:
        result = compare_runs(directories)
    except PerformanceRegression as error:
        result = {'passed': False, 'order': 'B-C-C-B', 'reason': str(error),
                  'scope': 'valid performance rejection; raw runs retained'}
        save(out / 'comparison.json', result)
        print(json.dumps(result), flush=True)
        raise SystemExit(3)
    save(out / 'comparison.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
