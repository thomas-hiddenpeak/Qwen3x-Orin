#!/usr/bin/env python3
"""Host-only tests for the fail-closed long-context EvalScope harness."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from typing import Any


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "tools/evaluation/run_native_long_context_matrix.sh"
SCHEMA = (
    REPOSITORY
    / "benchmarks/evalscope/qwen36-long-context-prefill-v1.manifest.schema.json"
)
MODEL = "qwen3.6-27b-nvfp4"
SELECTION = (
    "complete natural conversation prefixes; no padding, repetition, "
    "truncation, concatenation, or synthetic text"
)
WINDOWS = {
    "p8k": (7_168, 8_191),
    "p16k": (14_336, 16_383),
    "p40k": (39_000, 40_000),
}
PHASES = {
    "p8k": {"prefill1": 1},
    "p16k": {"prefill1": 1},
    "p40k": {"prefill1": 1, "cold16": 16},
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.model_dir = root / "model"
        self.corpus_dir = root / "corpora"
        self.manifest = root / "manifest.json"
        self.server = root / "fake-server"
        self.output = root / "output"
        self.model_dir.mkdir()
        self.corpus_dir.mkdir()
        tokenizer_hashes: dict[str, str] = {}
        for key, filename in (
            ("tokenizer_json_sha256", "tokenizer.json"),
            ("tokenizer_config_sha256", "tokenizer_config.json"),
            ("chat_template_jinja_sha256", "chat_template.jinja"),
        ):
            path = self.model_dir / filename
            path.write_text(f"host fixture {filename}\n", encoding="utf-8")
            tokenizer_hashes[key] = file_sha256(path)
        fake_server = (
            "#!/bin/sh\n"
            "# Q3X_RUN_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION\n"
            "# Q3X_FULL_ATTENTION_FLASHINFER_DIRECT\n"
            + "# " + ("x" * 262_144) + "\n"
            "exit 0\n"
        )
        self.server.write_text(fake_server, encoding="utf-8")
        self.server.chmod(0o755)

        buckets: dict[str, Any] = {}
        for bucket, phase_contract in PHASES.items():
            lower, upper = WINDOWS[bucket]
            runs: dict[str, Any] = {}
            for phase, max_tokens in phase_contract.items():
                request_lines: list[bytes] = []
                cases: list[dict[str, Any]] = []
                for ordinal in range(5):
                    prompt = [ordinal + 1] * (lower + ordinal)
                    request = {
                        "model": MODEL,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "seed": 42,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    encoded = canonical_json(request)
                    request_lines.append(encoded + b"\n")
                    locator = f"host-smoke-only:{bucket}:{ordinal}"
                    cases.append(
                        {
                            "ordinal": ordinal,
                            "role": "warmup" if ordinal == 0 else "measured",
                            "source_id": "host-smoke",
                            "source_locator": locator,
                            "source_record_sha256": sha256_bytes(
                                locator.encode("utf-8")
                            ),
                            "prompt_tokens": len(prompt),
                            "prompt_sha256": sha256_bytes(canonical_json(prompt)),
                            "request_sha256": sha256_bytes(encoded),
                        }
                    )
                corpus_name = f"host-{bucket}-{phase}.jsonl"
                corpus = b"".join(request_lines)
                (self.corpus_dir / corpus_name).write_bytes(corpus)
                runs[phase] = {
                    "max_tokens": max_tokens,
                    "corpus_file": corpus_name,
                    "corpus_sha256": sha256_bytes(corpus),
                    "count": 5,
                    "cases": cases,
                }
            buckets[bucket] = {
                "token_window": [lower, upper],
                "runs": runs,
            }
        manifest = {
            "schema_version": 1,
            "artifact": "qwen36_evalscope_long_context_prefill_v1",
            "sources": {
                "host-smoke": {
                    "dataset": "host-smoke-fixture-not-performance-data",
                    "revision": "unit-test",
                    "source_artifact_sha256": "1" * 64,
                    "license_or_terms": "host-only test fixture",
                    "benchmark_authorization": (
                        "host-only test; not performance evidence"
                    ),
                    "selection": SELECTION,
                }
            },
            "tokenizer": {
                **tokenizer_hashes,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            "request_contract": {
                "model": MODEL,
                "endpoint": "/v1/completions",
                "temperature": 0.0,
                "seed": 42,
                "stream": True,
                "include_usage": True,
                "parallel": 1,
                "warmup": 1,
                "measured": 4,
                "prefix_cache": False,
                "mtp": False,
            },
            "buckets": buckets,
        }
        self.manifest.write_bytes(canonical_json(manifest) + b"\n")
        self.manifest_sha256 = file_sha256(self.manifest)

    def command(self, *selectors: str) -> list[str]:
        return [
            str(RUNNER),
            str(self.server),
            str(self.model_dir),
            str(self.manifest),
            self.manifest_sha256,
            str(self.corpus_dir),
            str(self.output),
            *selectors,
        ]

    def run(self, *selectors: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["Q3X_LONG_EVAL_DRY_RUN"] = "1"
        return subprocess.run(
            self.command(*selectors),
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )


class LongContextEvalScopeHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(pathlib.Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_schema_is_checked_in_json(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["artifact"]["const"],
            "qwen36_evalscope_long_context_prefill_v1",
        )

    def test_shell_syntax_and_full_capacity_matrix(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)], check=False, capture_output=True, text=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        result = self.fixture.run("all", "both")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("runs=4 dry_run=1", result.stdout)
        self.assertIn("--max-sequence-length 8192", result.stdout)
        self.assertIn("--request-max-arena-bytes 705593344", result.stdout)
        self.assertIn("--max-sequence-length 16384", result.stdout)
        self.assertIn("--request-max-arena-bytes 1245085696", result.stdout)
        self.assertEqual(result.stdout.count("--max-sequence-length 40960"), 2)
        self.assertEqual(result.stdout.count("--request-max-arena-bytes 3221225472"), 2)
        self.assertIn("phase=cold16", result.stdout)
        self.assertIn("--max-output-tokens 16", result.stdout)
        self.assertIn("server_readiness_route=http://127.0.0.1:", result.stdout)
        self.assertIn("dry_run_complete=1 performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_wrong_manifest_hash_is_rejected(self) -> None:
        command = self.fixture.command("p8k", "prefill1")
        command[4] = "0" * 64
        environment = os.environ.copy()
        environment["Q3X_LONG_EVAL_DRY_RUN"] = "1"
        result = subprocess.run(
            command, env=environment, check=False, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("manifest SHA256 mismatch", result.stderr)

    def test_changed_corpus_is_rejected_before_run(self) -> None:
        corpus = self.fixture.corpus_dir / "host-p8k-prefill1.jsonl"
        corpus.write_bytes(corpus.read_bytes() + b"\n")
        result = self.fixture.run("p8k", "prefill1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("corpus SHA256 mismatch", result.stderr)
        self.assertFalse(self.fixture.output.exists())

    def test_missing_authorized_corpus_is_rejected(self) -> None:
        (self.fixture.corpus_dir / "host-p40k-cold16.jsonl").unlink()
        result = self.fixture.run("p40k", "cold16")
        self.assertEqual(result.returncode, 2)
        self.assertIn("authorized corpus is missing", result.stderr)

    def test_cold16_on_short_bucket_is_rejected(self) -> None:
        result = self.fixture.run("p8k", "cold16")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cold16 is defined only for p40k", result.stderr)

    def test_binary_without_attention_build_marker_is_rejected(self) -> None:
        self.fixture.server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.fixture.server.chmod(0o755)
        result = self.fixture.run("p8k", "prefill1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("long-context attention BUILD admission", result.stderr)


if __name__ == "__main__":
    unittest.main()
