"""Host-only negative controls for the versioned mirrored comparison gate."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/evaluation'))
from compare_service_runs import PerformanceRegression, compare_runs


class ComparisonContract(unittest.TestCase):
    def setUp(self):
        work = ROOT / '.q3x-work'
        work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='qualification-test-', dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.runs = []
        for index in range(4):
            directory = Path(self.temporary.name) / str(index); directory.mkdir()
            records = []; metrics = []
            for p in (1, 65, 513, 1089, 8192, 40000):
                records.append({'label': f'performance-{p}', 'request_sha256': str(p), 'request_id': f'{index}/{p}',
                    'text': 'same output', 'finish': 'length',
                    'usage': {'prompt_tokens': p, 'completion_tokens': 16, 'total_tokens': p + 16}})
                metrics.append({'request_id': f'{index}/{p}', 'request_sha256': str(p), 'prefill_tok_s': 400., 'decode_tok_s': 8., 'external_ttft_seconds': p / 400 + .02})
            (directory / 'results.json').write_text(json.dumps(records))
            (directory / 'service-audit.json').write_text(json.dumps({'metrics': metrics}))
            self.runs.append(directory)

    def test_unchanged_panel_reports_both_phases(self):
        result = compare_runs(self.runs)
        self.assertEqual(len(result['metrics']), 6)
        self.assertEqual(result['metrics'][-1]['decode_tok_s']['ratio'], 1.)
        self.assertEqual(result['metrics'][-1]['prefill_tok_s']['ratio'], 1.)

    def test_later_same_body_warmup_does_not_replace_selected_timing(self):
        path = self.runs[1] / 'service-audit.json'; data = json.loads(path.read_text())
        data['metrics'].append({'request_id': 'later-warmup', 'request_sha256': '513',
                                'prefill_tok_s': 1., 'decode_tok_s': 1.})
        path.write_text(json.dumps(data))
        self.assertTrue(compare_runs(self.runs)['passed'])

    def test_each_phase_regression_is_rejected(self):
        for phase in ('prefill_tok_s', 'decode_tok_s'):
            path = self.runs[1] / 'service-audit.json'; original = path.read_text()
            value = json.loads(original); value['metrics'][-1][phase] *= .8
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PerformanceRegression, 'regression'):
                compare_runs(self.runs)
            path.write_text(original)

    def test_external_latency_regression_is_rejected_even_with_equal_engine_rates(self):
        path = self.runs[1] / 'service-audit.json'; data = json.loads(path.read_text())
        data['metrics'][-1]['external_ttft_seconds'] *= 1.2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(PerformanceRegression, 'external TTFT'):
            compare_runs(self.runs)

    def test_output_drift_is_not_a_performance_pass(self):
        path = self.runs[2] / 'results.json'; data = json.loads(path.read_text())
        data[0]['text'] = 'different'; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'output or accounting'):
            compare_runs(self.runs)

    def test_missing_bucket_fails_closed(self):
        path = self.runs[3] / 'service-audit.json'; data = json.loads(path.read_text())
        data['metrics'].pop(); path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'missing performance'):
            compare_runs(self.runs)


class WitnessIdentityContract(unittest.TestCase):
    def setUp(self):
        from validate_whole_core_service import select_witness
        self.select = select_witness
        work = ROOT / '.q3x-work'; work.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='witness-test-', dir=work)
        self.addCleanup(self.temporary.cleanup); self.directory = Path(self.temporary.name)
        self.first = {'request': {'id': 'first', 'body_sha256': 'same'}, 'milliseconds': 10}
        self.last = {'request': {'id': 'last', 'body_sha256': 'same'}, 'milliseconds': 20}
        self.hashes = {'same': [self.first, self.last]}
        self.ids = {'first': self.first, 'last': self.last}

    def test_repeated_body_uses_actual_response_id(self):
        result = {'request_id': 'first', 'request_sha256': 'same'}
        self.assertIs(self.select(result, self.directory, self.hashes, self.ids), self.first)

    def test_saved_response_resolves_older_driver_records(self):
        (self.directory / 'baseline.json').write_text(json.dumps({'events': [{'id': 'first'}]}))
        result = {'label': 'baseline', 'request_sha256': 'same'}
        self.assertIs(self.select(result, self.directory, self.hashes, self.ids), self.first)

    def test_ambiguous_hash_alone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.select({'request_sha256': 'same'}, self.directory, self.hashes, self.ids)

    def test_response_label_cannot_escape_run_directory(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.select({'label': '../outside', 'request_sha256': 'same'},
                        self.directory, self.hashes, self.ids)

    def test_id_cannot_bind_a_different_body(self):
        with self.assertRaisesRegex(ValueError, 'body mismatch'):
            self.select({'request_id': 'first', 'request_sha256': 'different'},
                        self.directory, self.hashes, self.ids)


class PromptIdentityContract(unittest.TestCase):
    def setUp(self):
        import hashlib
        from validate_whole_core_service import validate_prompt_identity
        self.validate = validate_prompt_identity
        self.result = {'body': {'prompt': [1, 2, 3]}}
        self.witness = {'prompt': {'tokens': 3, 'token_ids_u32le_sha256':
            hashlib.sha256(bytes([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0])).hexdigest()}}

    def test_little_endian_client_token_identity(self):
        self.validate(self.result, self.witness)

    def test_silently_truncated_prompt_is_rejected(self):
        self.witness['prompt']['tokens'] = 2
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            self.validate(self.result, self.witness)

    def test_same_length_different_tokens_are_rejected(self):
        self.result['body']['prompt'][-1] = 4
        with self.assertRaisesRegex(ValueError, 'token identity mismatch'):
            self.validate(self.result, self.witness)


if __name__ == '__main__':
    unittest.main()
