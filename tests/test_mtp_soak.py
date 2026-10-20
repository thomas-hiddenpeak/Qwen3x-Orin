"""Host-only negative checks for sustained MTP validation."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/evaluation'))
import validate_mtp_service as mtp
sys.path.pop(0)


class SoakTest(unittest.TestCase):
    def run_panel(self, drift=None, growth=None, step=30):
        baseline = dict(body={}, endpoint='completions', request_sha256='same',
                        text='answer', usage=dict(prompt_tokens=513, completion_tokens=16,
                                                 total_tokens=529), finish='length', done=True)
        clock = [0.0]; count = [0]; samples = [0]
        class Client:
            def request(self, body, endpoint, label):
                clock[0] += step; count[0] += 1
                result = copy.deepcopy(baseline)
                if drift == 'text' and count[0] == 12: result['text'] = 'wrong'
                return result
            def health(self): return {}
        def resources(pid):
            samples[0] += 1
            r = dict(rss_kib=100, fds=10, threads=3)
            if growth and samples[0] > 1: r[growth[0]] += growth[1]
            return r
        def ids(result):
            return [2 if drift == 'ids' and count[0] == 12 else 1]*16
        with patch.object(mtp.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(mtp.time, 'sleep', side_effect=lambda s: clock.__setitem__(0, clock[0]+s)), \
             patch.object(mtp.q, 'resource_snapshot', side_effect=resources), \
             patch.object(mtp.q, 'save'), patch('builtins.print'):
            return mtp.soak_panel(Client(), baseline, ROOT / '.q3x-work', 1, 7200,
                                  lambda r: [{'prefill_s': 1}], ids)

    def test_complete(self):
        result = self.run_panel()
        self.assertGreaterEqual(result['requests'], 200)
        self.assertGreaterEqual(result['elapsed_s'], 7200)
        self.assertTrue(result['fresh_restart_required'])

    def test_text_drift(self):
        with self.assertRaisesRegex(RuntimeError, 'native mismatch text'): self.run_panel('text')

    def test_token_drift(self):
        with self.assertRaisesRegex(RuntimeError, 'token drift'): self.run_panel('ids')

    def test_insufficient_reuse(self):
        with self.assertRaisesRegex(RuntimeError, 'insufficient sustained'): self.run_panel(step=100)

    def test_resource_growth(self):
        for key, amount, message in [('rss_kib', 262145, 'RSS'), ('fds', 5, 'FD'), ('threads', 2, 'thread')]:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, message):
                self.run_panel(growth=(key, amount))


class RestartTest(unittest.TestCase):
    def prior(self, change=None):
        values = {'cleanup.json': dict(success=True, returncode=0, errors=[]),
                  'soak-summary.json': dict(complete=True, elapsed_s=7200,
                      requests=200, generated_token_ids=[1]*16),
                  'audited-metrics.json': [{}]*209,
                  'identity.json': dict(panel='sustained', draft_length=2, binary_sha256='hash')}
        if change: values[change[0]][change[1]] = change[2]
        prior = MagicMock()
        prior.__truediv__.side_effect = lambda name: MagicMock(read_text=lambda: json.dumps(values[name]))
        with patch.object(mtp.q, 'digest', return_value='hash'):
            return mtp.completed_soak(prior, ROOT / 'unused', 2)

    def test_completed_prior(self):
        self.assertEqual(self.prior()['requests'], 200)

    def test_incomplete_or_wrong_prior(self):
        for change in [('cleanup.json', 'success', False),
                       ('soak-summary.json', 'elapsed_s', 7199),
                       ('soak-summary.json', 'requests', 199),
                       ('identity.json', 'draft_length', 3),
                       ('identity.json', 'binary_sha256', 'wrong')]:
            with self.subTest(change=change), self.assertRaises(RuntimeError): self.prior(change)

    def test_restart_output(self):
        baseline = dict(body={}, endpoint='completions', request_sha256='same',
                        text='answer', usage={}, finish='length', done=True)
        client = MagicMock(); client.request.return_value = copy.deepcopy(baseline)
        with patch.object(mtp.q, 'save'):
            mtp.restart_panel(client, baseline, dict(generated_token_ids=[1]*16), ROOT,
                              lambda r: [], lambda r: [1]*16)
            with self.assertRaisesRegex(RuntimeError, 'token drift'):
                mtp.restart_panel(client, baseline, dict(generated_token_ids=[1]*16), ROOT,
                                  lambda r: [], lambda r: [2]*16)
            client.request.return_value['text'] = 'wrong'
            with self.assertRaisesRegex(RuntimeError, 'native mismatch text'):
                mtp.restart_panel(client, baseline, dict(generated_token_ids=[1]*16), ROOT,
                                  lambda r: [], lambda r: [1]*16)


if __name__ == '__main__': unittest.main()
