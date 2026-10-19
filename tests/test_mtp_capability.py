"""Host-only fail-closed checks for the MTP public capability panel."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / 'tools' / 'evaluation'
sys.path.insert(0, str(EVALUATION))
try:
    import validate_mtp_service as mtp
finally:
    sys.path.remove(str(EVALUATION))


class FakeClient:
    def __init__(self, results):
        self.results = results
        self.records = []

    def request(self, body, endpoint, label):
        result = copy.deepcopy(self.results[int(label.split('-')[-1])])
        self.records.append(result)
        return result


class CapabilityContractTest(unittest.TestCase):
    def setUp(self):
        self.cases = [{'request': {}, 'target': 'A'} for _ in range(98)]
        self.baseline = [{'label': f'capability-{i}', 'request_sha256': str(i),
                          'text': '答案：A', 'usage': {'prompt_tokens': 20, 'completion_tokens': 4},
                          'finish': 'stop', 'done': True} for i in range(98)]
        self.reference = [{'request_sha256': str(i), 'answer': 'A',
                           'usage': {'prompt_tokens': 20}} for i in range(98)]

    def run_panel(self, results=None):
        client = FakeClient(results or self.baseline)
        with patch.object(mtp.time, 'sleep'), patch.object(mtp.q, 'save'), patch('builtins.print'):
            result = mtp.capability_panel(client, self.cases, self.reference,
                                          self.baseline, ROOT / '.q3x-work', lambda result: None)
        return result

    def test_complete_panel(self):
        self.assertEqual(self.run_panel()['native_responses_equal'], 98)

    def test_same_answer_different_text_is_rejected(self):
        results = copy.deepcopy(self.baseline)
        results[0]['text'] = '额外内容。答案：A'
        with self.assertRaisesRegex(RuntimeError, 'native baseline mismatch.*text'):
            self.run_panel(results)

    def test_reference_identity_is_required(self):
        self.reference[0]['request_sha256'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'reference request identity'):
            self.run_panel()

    def test_answer_regression_is_rejected(self):
        self.reference[0]['answer'] = 'B'
        with self.assertRaisesRegex(RuntimeError, 'reference answer mismatch'):
            self.run_panel()

    def test_incomplete_panel_is_rejected(self):
        self.cases.pop()
        with self.assertRaisesRegex(RuntimeError, '98-question'):
            self.run_panel()

    def test_length_finish_is_not_scored_as_valid(self):
        self.baseline[0]['finish'] = 'length'
        with self.assertRaisesRegex(RuntimeError, 'invalid capability answer'):
            self.run_panel()


if __name__ == '__main__':
    unittest.main()
