"""Preparation must not manufacture workload coverage or promotion authority."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preparation", ROOT / "tools/evaluation/prepare_prefill_mainline_validation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparationTest(unittest.TestCase):
    def test_invalid_or_foreign_token_ids(self):
        for ids in ([], [True], [-1], [248320], [1.5], "text"):
            with self.assertRaises(ValueError):
                MODULE.validate_prompt({"prompt": ids})

    def test_capacity_boundary_panel(self):
        for n in (63, 64, 65, 511, 512, 513, 7999, 8000, 8001,
                  8191, 8192, 8193, 40000, 40001, 44095):
            self.assertIn(n, MODULE.CONTEXTS)

    def test_no_padding_no_promotion_no_overwrite(self):
        work = ROOT / ".q3x-work"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temporary:
            directory = Path(temporary)
            source, probe = directory / "source.json", directory / "probe"
            source.write_text('{"prompt":[123,456,789]}')
            probe.write_text("host probe fixture")
            audit = {"scope": "host-admission-only", "cases": [
                {"prompt_tokens": p, "output_tokens": o, "surface": "token_ids",
                 "stream": True, "include_usage": True, "admitted": True}
                for p in MODULE.CONTEXTS for o in MODULE.OUTPUTS
                if p + o - 1 <= 44095]}
            with patch.object(MODULE.subprocess, "check_output",
                              side_effect=[json.dumps(audit), "fixture-revision"]):
                result = MODULE.prepare(source, probe, directory / "requests")
            self.assertFalse(result["promotion_qualified"])
            self.assertTrue(result["admission_ready"])
            self.assertTrue(result["missing_real_payloads"])
            self.assertEqual(len(result["requests"]), 4)
            for case in result["requests"]:
                raw = (directory / "requests" / case["path"]).read_bytes()
                self.assertEqual(json.loads(raw)["prompt"], [123])
                self.assertEqual(MODULE.sha(raw), case["body_sha256"])
            with patch.object(MODULE.subprocess, "check_output", return_value=json.dumps(audit)):
                with self.assertRaises(FileExistsError):
                    MODULE.prepare(source, probe, directory / "requests")
            long_source = directory / "long.json"
            long_source.write_text(json.dumps({"prompt": [456] * 44095}))
            with patch.object(MODULE.subprocess, "check_output",
                              side_effect=[json.dumps(audit), "fixture-revision"]):
                complete = MODULE.prepare(source, probe, directory / "complete", long_source)
            self.assertFalse(complete["missing_real_payloads"])
            self.assertFalse(complete["promotion_qualified"])
            self.assertEqual(json.loads((directory / "complete/p1-o1.json").read_text())["prompt"], [123])
            boundary = json.loads((directory / "complete/p44095-o1.json").read_text())
            self.assertEqual(len(boundary["prompt"]), 44095)
            self.assertEqual(boundary["prompt"][0], 456)
            audit["cases"].pop()
            with patch.object(MODULE.subprocess, "check_output", return_value=json.dumps(audit)):
                with self.assertRaisesRegex(ValueError, "omitted"):
                    MODULE.prepare(source, probe, directory / "incomplete")

    def test_reject_output_outside_workspace(self):
        with self.assertRaisesRegex(ValueError, "inside .q3x-work"):
            MODULE.prepare(Path("unused"), Path("unused"), ROOT / "not-generated-here")


if __name__ == "__main__":
    unittest.main()
