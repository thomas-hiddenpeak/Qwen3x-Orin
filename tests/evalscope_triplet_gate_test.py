#!/usr/bin/env python3
"""Host-only tests for the external EvalScope direction gate."""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import pathlib
import pickle
import sqlite3
import statistics
import sys
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "tools/evaluation/validate_evalscope_triplet.py"
SPEC = importlib.util.spec_from_file_location("evalscope_triplet_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def request_for(
    prompt: list[int], max_tokens: int = 2
) -> dict[str, object]:
    return {
        "model": "test-model",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def response_payload(prefix: str, max_tokens: int = 2) -> str:
    messages = [
        {
            "id": "test",
            "object": "text_completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "text": prefix,
                    "logprobs": None,
                    "finish_reason": "length" if max_tokens == 1 else None,
                }
            ],
        },
    ]
    if max_tokens == 2:
        messages.append({
            "id": "test",
            "object": "text_completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "text": "!",
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
        })
    return base64.b64encode(pickle.dumps(messages)).decode("ascii")


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def build_manifest(
    path: pathlib.Path, max_tokens: int = 2
) -> dict[str, object]:
    cases = []
    for ordinal, prompt in enumerate(([100], [101, 102], [103, 104, 105])):
        request = request_for(list(prompt), max_tokens)
        cases.append(
            {
                "ordinal": ordinal,
                "prompt_tokens": len(prompt),
                "prompt_sha256": GATE.sha256_bytes(
                    GATE.canonical_json(list(prompt))
                ),
                "request_sha256": GATE.sha256_bytes(
                    GATE.canonical_json(request)
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact": "test_evalscope_manifest",
        "count": len(cases),
        "corpus_sha256": "0" * 64,
        "request_contract": {
            "model": "test-model",
            "endpoint": "/v1/completions",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "stream": True,
            "include_usage": True,
        },
        "cases": cases,
    }
    write_json(path, manifest)
    return manifest


def build_run(
    directory: pathlib.Path,
    name: str,
    ttfts: tuple[float, float],
    output_prefixes: tuple[str, str] = ("A", "B"),
    max_tokens: int = 2,
) -> None:
    directory.mkdir(parents=True)
    arguments = {
        "model": "test-model",
        "api": "openai",
        "url": f"http://127.0.0.1:18{name[-1]}/v1/completions",
        "number": 2,
        "parallel": 1,
        "warmup_num": 1.0,
        "num_workers": 1,
        "dataset": "line_by_line",
        "dataset_path": "/tmp/test-corpus.jsonl",
        "data_source": "local",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "tokenize_prompt": True,
        "no_test_connection": True,
        "apply_chat_template": False,
        "name": name,
        "outputs_dir": str(directory),
    }
    write_json(directory / "benchmark_args.json", arguments)

    database = sqlite3.connect(directory / "benchmark_data.db")
    database.execute(
        """CREATE TABLE result(
          request TEXT, start_time REAL, inter_token_latencies TEXT,
          success INTEGER, response_messages TEXT, completed_time REAL,
          latency REAL, first_chunk_latency REAL, prompt_tokens INTEGER,
          completion_tokens INTEGER, max_gpu_memory_cost REAL,
          time_per_output_token REAL, request_id TEXT
        )"""
    )
    starts = (100.0, 101.5)
    prompts = ([101, 102], [103, 104, 105])
    rows = []
    for index, (prompt, start, ttft, prefix) in enumerate(
        zip(prompts, starts, ttfts, output_prefixes)
    ):
        tpot = 0.0 if max_tokens == 1 else 0.1
        latency = ttft + tpot
        rows.append(
            (
                json.dumps(request_for(list(prompt), max_tokens)),
                start,
                json.dumps([] if max_tokens == 1 else [tpot]),
                1,
                response_payload(prefix, max_tokens),
                start + latency,
                latency,
                ttft,
                len(prompt),
                max_tokens,
                0.0,
                tpot,
                f"request-{index}",
            )
        )
    database.executemany(
        "INSERT INTO result VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    database.commit()
    database.close()

    latencies = [ttft + (0.0 if max_tokens == 1 else 0.1) for ttft in ttfts]
    wall_time = starts[-1] + latencies[-1] - starts[0]
    prompt_tokens = sum(len(prompt) for prompt in prompts)
    completion_tokens = 2 * max_tokens
    mean_ttft_ms = statistics.fmean(ttfts) * 1000.0
    summary = {
        "Test Duration (s)": wall_time,
        "Total Requests": 2,
        "Success Requests": 2,
        "Failed Requests": 0,
        "Avg Latency (s)": statistics.fmean(latencies),
        "Avg Input Tokens": prompt_tokens / 2,
        "Avg Output Tokens": float(max_tokens),
        "TTFT (ms)": mean_ttft_ms,
        "TPOT (ms)": 0.0 if max_tokens == 1 else 100.0,
        "ITL (ms)": 0.0 if max_tokens == 1 else 100.0,
        "Total Throughput (tok/s)": (
            prompt_tokens + completion_tokens
        )
        / wall_time,
    }
    write_json(directory / "benchmark_summary.json", summary)
    sorted_ttft = sorted(value * 1000.0 for value in ttfts)
    write_json(
        directory / "benchmark_percentile.json",
        [
            {"Percentiles": "50%", "TTFT (ms)": sorted_ttft[0]},
            {"Percentiles": "99%", "TTFT (ms)": sorted_ttft[1]},
        ],
    )
    write_json(
        directory / "workload_throughput.json",
        {
            "n_samples": 2,
            "wall_time_s": wall_time,
            "rows": [
                {
                    "metric": "Total Prompt tok/s",
                    "overall": prompt_tokens / wall_time,
                }
            ],
        },
    )


class EvalScopeTripletGateTest(unittest.TestCase):
    def run_gate(
        self,
        root: pathlib.Path,
        candidate_ttfts: tuple[float, float],
        candidate_outputs: tuple[str, str] = ("A", "B"),
        allow_output_change: bool = False,
        include_reference: bool = True,
        max_tokens: int = 2,
        candidate_tpot_override: float | None = None,
    ) -> tuple[int, dict[str, object]]:
        manifest = root / "manifest.json"
        build_manifest(manifest, max_tokens)
        build_run(
            root / "baseline", "baseline-0", (0.8, 1.2),
            max_tokens=max_tokens,
        )
        build_run(
            root / "candidate",
            "candidate-1",
            candidate_ttfts,
            candidate_outputs,
            max_tokens,
        )
        if candidate_tpot_override is not None:
            database = sqlite3.connect(root / "candidate/benchmark_data.db")
            database.execute(
                "UPDATE result SET time_per_output_token = ?",
                (candidate_tpot_override,),
            )
            database.commit()
            database.close()
        build_run(
            root / "reference", "reference-2", (0.4, 0.6),
            max_tokens=max_tokens,
        )
        stdout = io.StringIO()
        arguments = [
            "--manifest",
            str(manifest),
            "--baseline",
            str(root / "baseline"),
            "--candidate",
            str(root / "candidate"),
        ]
        if include_reference:
            arguments.extend(["--reference", str(root / "reference")])
        if allow_output_change:
            arguments.append("--allow-native-output-change")
        with contextlib.redirect_stdout(stdout):
            status = GATE.main(arguments)
        return status, json.loads(stdout.getvalue())

    def test_positive_native_direction_advances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary), (0.7, 1.1)
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            report["decision"]["candidate"], "advance_to_internal_validation"
        )
        self.assertTrue(
            report["decision"]["external_reference_is_decision_ineligible"]
        )

    def test_negative_native_direction_is_exit_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary), (0.9, 1.3)
            )
        self.assertEqual(status, 3)
        self.assertEqual(report["decision"]["candidate"], "reject_direction")

    def test_pair_gate_uses_frozen_reference_without_rerunning_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary),
                (0.7, 1.1),
                include_reference=False,
            )
        self.assertEqual(status, 0)
        self.assertNotIn("external_reference", report["runs"])
        self.assertIn("frozen", report["validation"]["external_reference"])

    def test_output_change_cannot_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary), (0.7, 1.1), ("changed", "B")
            )
        self.assertEqual(status, 3)
        self.assertIn("changed generated output", report["decision"]["reason"])

    def test_declared_throughput_mode_defers_output_to_capability_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary),
                (0.7, 1.1),
                ("changed", "B"),
                allow_output_change=True,
            )
        self.assertEqual(status, 0)
        self.assertTrue(report["decision"]["public_capability_validation_required"])

    def test_restricted_unpickler_rejects_globals(self) -> None:
        malicious = base64.b64encode(pickle.dumps(eval)).decode("ascii")
        with self.assertRaises(GATE.EvidenceError):
            GATE.decode_response_messages(malicious, "malicious")

    def test_one_token_zero_tpot_and_empty_itl_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary),
                (0.7, 1.1),
                max_tokens=1,
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            report["runs"]["native_candidate"]["metrics"]["mean_tpot_ms"],
            0.0,
        )
        self.assertEqual(
            report["runs"]["native_candidate"]["metrics"]["mean_itl_ms"],
            0.0,
        )

    def test_one_token_nonzero_tpot_is_invalid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, report = self.run_gate(
                pathlib.Path(temporary),
                (0.7, 1.1),
                max_tokens=1,
                candidate_tpot_override=0.1,
            )
        self.assertEqual(status, 1)
        self.assertEqual(report["decision"]["candidate"], "invalid_evidence")
        self.assertIn("must have TPOT=0", report["decision"]["reason"])


if __name__ == "__main__":
    unittest.main()
