#!/usr/bin/env python3
"""Validate a baseline/candidate/vLLM EvalScope direction triplet.

The validator deliberately separates two decisions:

* the candidate advances only when it improves the native baseline; and
* the external reference reports the remaining product gap but can never
  reject an incremental native improvement.

EvalScope stores response chunks as base64-encoded Python pickles.  This tool
uses a restricted unpickler that rejects every global/class lookup and accepts
only the built-in container/scalar opcodes used by EvalScope's JSON response
objects.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import pathlib
import pickle
import sqlite3
import statistics
import tempfile
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FILES = (
    "benchmark_args.json",
    "benchmark_data.db",
    "benchmark_percentile.json",
    "benchmark_summary.json",
    "workload_throughput.json",
)
EXPECTED_RESULT_COLUMNS = (
    "request",
    "start_time",
    "inter_token_latencies",
    "success",
    "response_messages",
    "completed_time",
    "latency",
    "first_chunk_latency",
    "prompt_tokens",
    "completion_tokens",
    "max_gpu_memory_cost",
    "time_per_output_token",
    "request_id",
)
ALLOWED_ARGUMENT_DIFFERENCES = frozenset({"name", "outputs_dir", "url"})


class EvidenceError(RuntimeError):
    """The supplied artifacts cannot support a benchmark decision."""


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickle built-in literals while rejecting executable globals."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"global pickle reference is forbidden: {module}.{name}"
        )


@dataclass(frozen=True)
class RequestResult:
    prompt_sha256: str
    request_sha256: str
    output_sha256: str
    prompt_tokens: int
    completion_tokens: int
    start_time: float
    completed_time: float
    latency_s: float
    ttft_s: float
    tpot_s: float
    inter_token_latencies_s: tuple[float, ...]


@dataclass(frozen=True)
class RunEvidence:
    role: str
    result_dir: pathlib.Path
    arguments: Mapping[str, Any]
    requests: tuple[RequestResult, ...]
    metrics: Mapping[str, float | int]
    artifact_sha256: Mapping[str, str]
    prompt_set_sha256: str
    output_set_sha256: str


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def load_json(path: pathlib.Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def require_int(value: Any, label: str) -> int:
    number = require_number(value, label)
    integer = int(number)
    if number != integer:
        raise EvidenceError(f"{label} must be integral")
    return integer


def assert_close(
    actual: float, expected: float, tolerance: float, label: str
) -> None:
    if abs(actual - expected) > tolerance:
        raise EvidenceError(
            f"{label} mismatch: expected {expected}, got {actual} "
            f"(tolerance {tolerance})"
        )


def nearest_observed(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceError("cannot compute a percentile from no values")
    index = round((len(ordered) - 1) * percentile / 100.0)
    return ordered[index]


def decode_response_messages(encoded: str, label: str) -> list[Any]:
    if not isinstance(encoded, str):
        raise EvidenceError(f"{label} response_messages is not text")
    try:
        raw = base64.b64decode(encoded, validate=True)
        value = RestrictedUnpickler(io.BytesIO(raw)).load()
    except (ValueError, pickle.UnpicklingError, EOFError) as error:
        raise EvidenceError(
            f"{label} response_messages is not a safe EvalScope payload: {error}"
        ) from error
    if not isinstance(value, list):
        raise EvidenceError(f"{label} response_messages must decode to a list")
    return value


def extract_output(
    messages: Sequence[Any], completion_tokens: int, label: str
) -> str:
    if len(messages) != completion_tokens:
        raise EvidenceError(
            f"{label} has {len(messages)} response chunks for "
            f"{completion_tokens} completion tokens"
        )
    pieces: list[str] = []
    for chunk_index, message in enumerate(messages):
        message = require_mapping(message, f"{label} chunk {chunk_index}")
        choices = message.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise EvidenceError(
                f"{label} chunk {chunk_index} must contain exactly one choice"
            )
        choice = require_mapping(
            choices[0], f"{label} chunk {chunk_index} choice"
        )
        if choice.get("index") != 0 or not isinstance(choice.get("text"), str):
            raise EvidenceError(
                f"{label} chunk {chunk_index} has an invalid completion choice"
            )
        finish_reason = choice.get("finish_reason")
        if chunk_index + 1 < len(messages) and finish_reason is not None:
            raise EvidenceError(
                f"{label} chunk {chunk_index} finishes before the last token"
            )
        if chunk_index + 1 == len(messages) and finish_reason not in {
            "length",
            "stop",
        }:
            raise EvidenceError(
                f"{label} final chunk has invalid finish_reason {finish_reason!r}"
            )
        pieces.append(choice["text"])
    return "".join(pieces)


def validate_result_dir(path: pathlib.Path, role: str) -> pathlib.Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise EvidenceError(f"{role} result directory does not exist: {resolved}")
    missing = [name for name in REQUIRED_FILES if not (resolved / name).is_file()]
    if missing:
        raise EvidenceError(
            f"{role} result directory is missing: {', '.join(missing)}"
        )
    return resolved


def validate_argument_contract(
    arguments: Mapping[str, Any], manifest: Mapping[str, Any], role: str
) -> int:
    contract = require_mapping(
        manifest.get("request_contract"), "manifest request_contract"
    )
    warmup = require_int(arguments.get("warmup_num"), f"{role} warmup_num")
    number = require_int(arguments.get("number"), f"{role} number")
    expected = {
        "model": contract.get("model"),
        "api": "openai",
        "dataset": "line_by_line",
        "data_source": "local",
        "number": number,
        "parallel": 1,
        "warmup_num": warmup,
        "num_workers": 1,
        "max_tokens": contract.get("max_tokens"),
        "temperature": contract.get("temperature"),
        "seed": contract.get("seed"),
        "stream": contract.get("stream"),
        "tokenize_prompt": True,
        "no_test_connection": True,
        "apply_chat_template": False,
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise EvidenceError(
                f"{role} argument {key!r} mismatch: expected {value!r}, "
                f"got {arguments.get(key)!r}"
            )
    if warmup < 0 or number <= 0:
        raise EvidenceError(f"{role} has invalid warmup/measurement counts")
    endpoint = contract.get("endpoint")
    url = arguments.get("url")
    if not isinstance(endpoint, str) or not isinstance(url, str):
        raise EvidenceError(f"{role} has no valid endpoint URL")
    if urllib.parse.urlparse(url).path != endpoint:
        raise EvidenceError(
            f"{role} endpoint path does not match manifest: {url!r}"
        )
    return warmup


def validate_run(
    result_dir: pathlib.Path,
    role: str,
    manifest: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
) -> RunEvidence:
    result_dir = validate_result_dir(result_dir, role)
    arguments = require_mapping(
        load_json(result_dir / "benchmark_args.json"),
        f"{role} benchmark_args",
    )
    summary = require_mapping(
        load_json(result_dir / "benchmark_summary.json"),
        f"{role} benchmark_summary",
    )
    percentiles = load_json(result_dir / "benchmark_percentile.json")
    workload = require_mapping(
        load_json(result_dir / "workload_throughput.json"),
        f"{role} workload_throughput",
    )
    validate_argument_contract(arguments, manifest, role)

    database = result_dir / "benchmark_data.db"
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise EvidenceError(f"cannot open {role} database: {error}") from error
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise EvidenceError(
                f"{role} database integrity_check failed: {integrity!r}"
            )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if tables != {"result"}:
            raise EvidenceError(
                f"{role} database tables differ from EvalScope contract: {tables}"
            )
        columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(result)")
        )
        if columns != EXPECTED_RESULT_COLUMNS:
            raise EvidenceError(
                f"{role} result schema mismatch: expected "
                f"{EXPECTED_RESULT_COLUMNS}, got {columns}"
            )
        rows = connection.execute(
            "SELECT request, start_time, inter_token_latencies, success, "
            "response_messages, completed_time, latency, first_chunk_latency, "
            "prompt_tokens, completion_tokens, time_per_output_token "
            "FROM result ORDER BY start_time"
        ).fetchall()
    except sqlite3.Error as error:
        raise EvidenceError(f"cannot read {role} database: {error}") from error
    finally:
        connection.close()

    if len(rows) != len(expected_cases):
        raise EvidenceError(
            f"{role} has {len(rows)} rows; expected {len(expected_cases)}"
        )
    seen_requests: set[str] = set()
    request_results: list[RequestResult] = []
    for index, (row, case) in enumerate(zip(rows, expected_cases), start=1):
        label = f"{role} request {index}"
        if not isinstance(case, dict):
            raise EvidenceError(f"manifest case {index} is not an object")
        try:
            request = json.loads(row[0])
            inter_token_latencies = json.loads(row[2])
        except (TypeError, json.JSONDecodeError) as error:
            raise EvidenceError(f"{label} contains invalid JSON: {error}") from error
        request = require_mapping(request, f"{label} body")
        prompt = request.get("prompt")
        if not isinstance(prompt, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in prompt
        ):
            raise EvidenceError(f"{label} has an invalid token-id prompt")
        request_sha = sha256_bytes(canonical_json(request))
        prompt_sha = sha256_bytes(canonical_json(prompt))
        if request_sha != case.get("request_sha256"):
            raise EvidenceError(f"{label} request hash does not match the manifest")
        if prompt_sha != case.get("prompt_sha256"):
            raise EvidenceError(f"{label} prompt hash does not match the manifest")
        if request_sha in seen_requests:
            raise EvidenceError(f"{label} duplicates a prior measured request")
        seen_requests.add(request_sha)

        success = require_int(row[3], f"{label} success")
        if success != 1:
            raise EvidenceError(f"{label} failed")
        prompt_tokens = require_int(row[8], f"{label} prompt_tokens")
        completion_tokens = require_int(
            row[9], f"{label} completion_tokens"
        )
        if prompt_tokens != len(prompt) or prompt_tokens != case.get(
            "prompt_tokens"
        ):
            raise EvidenceError(f"{label} prompt token count is inconsistent")
        max_tokens = require_int(request.get("max_tokens"), f"{label} max_tokens")
        if completion_tokens != max_tokens:
            raise EvidenceError(
                f"{label} is incomplete: {completion_tokens}/{max_tokens} tokens"
            )
        if not isinstance(inter_token_latencies, list) or len(
            inter_token_latencies
        ) != max(0, completion_tokens - 1):
            raise EvidenceError(f"{label} has an invalid ITL sequence")
        itls = tuple(
            require_number(value, f"{label} ITL")
            for value in inter_token_latencies
        )
        if any(value < 0 for value in itls):
            raise EvidenceError(f"{label} has a negative ITL")

        start_time = require_number(row[1], f"{label} start_time")
        completed_time = require_number(row[5], f"{label} completed_time")
        latency = require_number(row[6], f"{label} latency")
        ttft = require_number(row[7], f"{label} first_chunk_latency")
        tpot = require_number(row[10], f"{label} time_per_output_token")
        if not (0 < ttft <= latency):
            raise EvidenceError(f"{label} has invalid latency components")
        if completion_tokens == 1:
            if tpot != 0.0 or itls:
                raise EvidenceError(
                    f"{label} one-token result must have TPOT=0 and no ITL"
                )
        elif not (0 < tpot <= latency):
            raise EvidenceError(f"{label} has invalid TPOT")
        assert_close(
            completed_time - start_time,
            latency,
            0.002,
            f"{label} elapsed latency",
        )
        messages = decode_response_messages(row[4], label)
        output = extract_output(messages, completion_tokens, label)
        request_results.append(
            RequestResult(
                prompt_sha256=prompt_sha,
                request_sha256=request_sha,
                output_sha256=sha256_bytes(output.encode("utf-8")),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                start_time=start_time,
                completed_time=completed_time,
                latency_s=latency,
                ttft_s=ttft,
                tpot_s=tpot,
                inter_token_latencies_s=itls,
            )
        )

    latencies = [request.latency_s for request in request_results]
    ttfts_ms = [request.ttft_s * 1000.0 for request in request_results]
    tpots_ms = [request.tpot_s * 1000.0 for request in request_results]
    itls_ms = [
        value * 1000.0
        for request in request_results
        for value in request.inter_token_latencies_s
    ]
    prompt_tokens_total = sum(r.prompt_tokens for r in request_results)
    completion_tokens_total = sum(r.completion_tokens for r in request_results)
    wall_time = max(r.completed_time for r in request_results) - min(
        r.start_time for r in request_results
    )
    metrics: dict[str, float | int] = {
        "requests": len(request_results),
        "success_requests": len(request_results),
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total,
        "wall_time_s": wall_time,
        "mean_latency_s": statistics.fmean(latencies),
        "mean_ttft_ms": statistics.fmean(ttfts_ms),
        "p50_ttft_ms": nearest_observed(ttfts_ms, 50.0),
        "p99_ttft_ms": nearest_observed(ttfts_ms, 99.0),
        "mean_tpot_ms": statistics.fmean(tpots_ms),
        # EvalScope correctly has no inter-token interval for a one-token
        # completion and reports both TPOT and ITL as zero. Preserve that
        # defined boundary rather than rejecting the row or averaging an empty
        # sequence.
        "mean_itl_ms": statistics.fmean(itls_ms) if itls_ms else 0.0,
        "prompt_throughput_tok_s": prompt_tokens_total / wall_time,
        "completion_throughput_tok_s": completion_tokens_total / wall_time,
        "total_throughput_tok_s": (
            prompt_tokens_total + completion_tokens_total
        )
        / wall_time,
    }

    summary_checks = {
        "Total Requests": (len(request_results), 0.0),
        "Success Requests": (len(request_results), 0.0),
        "Failed Requests": (0, 0.0),
        "Test Duration (s)": (wall_time, 0.001),
        "Avg Latency (s)": (metrics["mean_latency_s"], 0.00011),
        "Avg Input Tokens": (
            prompt_tokens_total / len(request_results),
            0.0001,
        ),
        "Avg Output Tokens": (
            completion_tokens_total / len(request_results),
            0.0001,
        ),
        "TTFT (ms)": (metrics["mean_ttft_ms"], 0.011),
        "TPOT (ms)": (metrics["mean_tpot_ms"], 0.011),
        "ITL (ms)": (metrics["mean_itl_ms"], 0.011),
        "Total Throughput (tok/s)": (
            metrics["total_throughput_tok_s"],
            0.00011,
        ),
    }
    for key, (expected, tolerance) in summary_checks.items():
        actual = require_number(summary.get(key), f"{role} summary {key}")
        assert_close(actual, float(expected), tolerance, f"{role} summary {key}")

    if not isinstance(percentiles, list):
        raise EvidenceError(f"{role} percentile report must be a JSON array")
    percentile_map = {
        row.get("Percentiles"): row
        for row in percentiles
        if isinstance(row, dict)
    }
    for percentile_name, percentile_value in (("50%", 50.0), ("99%", 99.0)):
        row = require_mapping(
            percentile_map.get(percentile_name),
            f"{role} percentile {percentile_name}",
        )
        assert_close(
            require_number(row.get("TTFT (ms)"), f"{role} {percentile_name} TTFT"),
            nearest_observed(ttfts_ms, percentile_value),
            0.011,
            f"{role} {percentile_name} TTFT",
        )

    if require_int(workload.get("n_samples"), f"{role} workload samples") != len(
        request_results
    ):
        raise EvidenceError(f"{role} workload sample count mismatch")
    assert_close(
        require_number(workload.get("wall_time_s"), f"{role} workload wall time"),
        wall_time,
        0.001,
        f"{role} workload wall time",
    )
    workload_rows = workload.get("rows")
    if not isinstance(workload_rows, list):
        raise EvidenceError(f"{role} workload rows must be a list")
    workload_by_metric = {
        row.get("metric"): row
        for row in workload_rows
        if isinstance(row, dict)
    }
    prompt_row = require_mapping(
        workload_by_metric.get("Total Prompt tok/s"),
        f"{role} prompt throughput row",
    )
    assert_close(
        require_number(prompt_row.get("overall"), f"{role} prompt throughput"),
        float(metrics["prompt_throughput_tok_s"]),
        1e-9,
        f"{role} prompt throughput",
    )

    artifact_hashes = {
        name: file_sha256(result_dir / name) for name in REQUIRED_FILES
    }
    prompt_set_sha = sha256_bytes(
        canonical_json([request.prompt_sha256 for request in request_results])
    )
    output_set_sha = sha256_bytes(
        canonical_json([request.output_sha256 for request in request_results])
    )
    return RunEvidence(
        role=role,
        result_dir=result_dir,
        arguments=arguments,
        requests=tuple(request_results),
        metrics=metrics,
        artifact_sha256=artifact_hashes,
        prompt_set_sha256=prompt_set_sha,
        output_set_sha256=output_set_sha,
    )


def compare_arguments(runs: Sequence[RunEvidence]) -> None:
    if not runs:
        raise EvidenceError("no runs supplied")
    reference = {
        key: value
        for key, value in runs[0].arguments.items()
        if key not in ALLOWED_ARGUMENT_DIFFERENCES
    }
    for run in runs[1:]:
        current = {
            key: value
            for key, value in run.arguments.items()
            if key not in ALLOWED_ARGUMENT_DIFFERENCES
        }
        if current != reference:
            differing = sorted(
                key
                for key in set(reference) | set(current)
                if reference.get(key) != current.get(key)
            )
            raise EvidenceError(
                f"{run.role} benchmark arguments differ in: {', '.join(differing)}"
            )


def output_match(left: RunEvidence, right: RunEvidence) -> Mapping[str, int]:
    left_by_prompt = {
        request.prompt_sha256: request.output_sha256 for request in left.requests
    }
    right_by_prompt = {
        request.prompt_sha256: request.output_sha256 for request in right.requests
    }
    if left_by_prompt.keys() != right_by_prompt.keys():
        raise EvidenceError(
            f"{left.role} and {right.role} measured different prompt sets"
        )
    exact = sum(
        left_by_prompt[prompt] == right_by_prompt[prompt]
        for prompt in left_by_prompt
    )
    return {"exact": exact, "total": len(left_by_prompt)}


def ttft_length_buckets(
    baseline: RunEvidence, candidate: RunEvidence
) -> list[Mapping[str, float | int | str]]:
    if len(baseline.requests) != len(candidate.requests):
        raise EvidenceError("native runs have different request counts")
    definitions = (
        ("1-128", 1, 128),
        ("129-512", 129, 512),
        ("513-1024", 513, 1024),
        ("1025+", 1025, None),
    )
    rows: list[Mapping[str, float | int | str]] = []
    for label, minimum, maximum in definitions:
        deltas = [
            (candidate_request.ttft_s - baseline_request.ttft_s) * 1000.0
            for baseline_request, candidate_request in zip(
                baseline.requests, candidate.requests
            )
            if baseline_request.prompt_tokens >= minimum
            and (maximum is None or baseline_request.prompt_tokens <= maximum)
        ]
        if not deltas:
            continue
        rows.append(
            {
                "prompt_tokens": label,
                "requests": len(deltas),
                "mean_candidate_minus_baseline_ttft_ms": statistics.fmean(
                    deltas
                ),
                "candidate_faster_requests": sum(delta < 0 for delta in deltas),
                "candidate_slower_requests": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
            }
        )
    return rows


def metric_ratio(numerator: float | int, denominator: float | int) -> float:
    denominator = float(denominator)
    if denominator == 0:
        raise EvidenceError("cannot compute a metric ratio with denominator zero")
    return float(numerator) / denominator


def run_to_json(run: RunEvidence) -> Mapping[str, Any]:
    return {
        "metrics": dict(run.metrics),
        "prompt_set_sha256": run.prompt_set_sha256,
        "output_set_sha256": run.output_set_sha256,
        "artifacts": {
            name: {"sha256": digest}
            for name, digest in run.artifact_sha256.items()
        },
    }


def build_report(
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    baseline: RunEvidence,
    candidate: RunEvidence,
    reference: RunEvidence | None,
    minimum_ttft_improvement_ms: float,
    minimum_prompt_throughput_improvement_tok_s: float,
    require_exact_native_output: bool,
) -> tuple[Mapping[str, Any], int]:
    runs_to_compare = (baseline, candidate) + (
        (reference,) if reference is not None else ()
    )
    compare_arguments(runs_to_compare)
    baseline_candidate_outputs = output_match(baseline, candidate)

    baseline_ttft = float(baseline.metrics["mean_ttft_ms"])
    candidate_ttft = float(candidate.metrics["mean_ttft_ms"])
    improvement = baseline_ttft - candidate_ttft
    prompt_throughput_improvement = float(
        candidate.metrics["prompt_throughput_tok_s"]
    ) - float(baseline.metrics["prompt_throughput_tok_s"])
    outputs_complete = (
        baseline_candidate_outputs["exact"] == baseline_candidate_outputs["total"]
    )
    outputs_admissible = outputs_complete or not require_exact_native_output
    advances = (
        improvement > minimum_ttft_improvement_ms
        and prompt_throughput_improvement
        > minimum_prompt_throughput_improvement_tok_s
        and outputs_admissible
    )
    exit_code = 0 if advances else 3
    decision = "advance_to_internal_validation" if advances else "reject_direction"
    reason = (
        (
            "candidate improves native mean TTFT and preserves every generated output"
            if outputs_complete
            else "candidate improves native mean TTFT under the declared throughput-mode output policy; public capability validation remains mandatory"
        )
        if advances
        else (
            "candidate changed generated output; capability validation is required"
            if not outputs_complete
            else "candidate does not improve both native mean TTFT and prompt throughput by the declared margins"
        )
    )

    runs: dict[str, Any] = {
        "native_baseline": run_to_json(baseline),
        "native_candidate": run_to_json(candidate),
    }
    comparisons: dict[str, Any] = {
        "candidate_vs_native_baseline": {
            "mean_ttft_delta_ms": candidate_ttft - baseline_ttft,
            "mean_ttft_improvement_ms": improvement,
            "mean_ttft_ratio": metric_ratio(candidate_ttft, baseline_ttft),
            "p50_ttft_delta_ms": float(candidate.metrics["p50_ttft_ms"])
            - float(baseline.metrics["p50_ttft_ms"]),
            "p99_ttft_delta_ms": float(candidate.metrics["p99_ttft_ms"])
            - float(baseline.metrics["p99_ttft_ms"]),
            "prompt_throughput_delta_tok_s": float(
                prompt_throughput_improvement
            ),
            "exact_outputs": baseline_candidate_outputs,
            "prompt_length_buckets": ttft_length_buckets(
                baseline, candidate
            ),
        }
    }
    if reference is not None:
        reference_ttft = float(reference.metrics["mean_ttft_ms"])
        baseline_reference_outputs = output_match(baseline, reference)
        runs["external_reference"] = run_to_json(reference)
        comparisons["native_baseline_vs_external_reference"] = {
            "mean_ttft_ratio": metric_ratio(baseline_ttft, reference_ttft),
            "p50_ttft_ratio": metric_ratio(
                baseline.metrics["p50_ttft_ms"],
                reference.metrics["p50_ttft_ms"],
            ),
            "p99_ttft_ratio": metric_ratio(
                baseline.metrics["p99_ttft_ms"],
                reference.metrics["p99_ttft_ms"],
            ),
            "prompt_throughput_fraction": metric_ratio(
                baseline.metrics["prompt_throughput_tok_s"],
                reference.metrics["prompt_throughput_tok_s"],
            ),
            "exact_outputs": baseline_reference_outputs,
        }

    report = {
        "schema_version": 1,
        "artifact": "qwen36_evalscope_external_direction",
        "authority": "external_t3_single_round_direction_only",
        "manifest": {
            "sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest.get("corpus_sha256"),
            "declared_cases": manifest.get("count"),
            "measured_cases": len(baseline.requests),
        },
        "validation": {
            "database_integrity": "passed",
            "request_manifest_match": "passed",
            "completion_chunk_completeness": "passed",
            "usage_chunk_validation": (
                "not_observable_in_evalscope_1_9_1_benchmark_database; "
                "retain a separate raw-SSE protocol audit"
            ),
            "summary_recomputation": "passed",
            "benchmark_arguments_comparable": "passed",
            "external_reference": (
                "validated_raw_artifacts"
                if reference is not None
                else "not_run; use the separately frozen market reference"
            ),
            "allowed_argument_differences": sorted(
                ALLOWED_ARGUMENT_DIFFERENCES
            ),
        },
        "runs": runs,
        "comparisons": comparisons,
        "decision": {
            "candidate": decision,
            "reason": reason,
            "exit_code": exit_code,
            "minimum_mean_ttft_improvement_ms": minimum_ttft_improvement_ms,
            "minimum_prompt_throughput_improvement_tok_s": (
                minimum_prompt_throughput_improvement_tok_s
            ),
            "exact_native_output_required": require_exact_native_output,
            "external_reference_is_decision_ineligible": True,
            "positive_direction_requires_followup_validation": True,
            "public_capability_validation_required": not outputs_complete,
        },
    }
    return report, exit_code


def atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a native EvalScope baseline/candidate pair; optionally "
            "audit a raw external reference without making it a retention gate"
        )
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        help="Optional raw external-reference result; normally use the frozen record",
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        help="Optionally verify the ephemeral corpus file against the manifest",
    )
    parser.add_argument(
        "--minimum-mean-ttft-improvement-ms",
        type=float,
        default=0.0,
        help="Strict native-baseline direction margin; default: 0 ms",
    )
    parser.add_argument(
        "--minimum-prompt-throughput-improvement-tok-s",
        type=float,
        default=0.0,
        help="Strict native-baseline prompt-throughput margin; default: 0 tok/s",
    )
    parser.add_argument(
        "--allow-native-output-change",
        action="store_true",
        help=(
            "Allow a declared throughput-mode candidate to advance on "
            "performance direction; this never waives the public capability gate"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing --output"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not math.isfinite(args.minimum_mean_ttft_improvement_ms):
            raise EvidenceError("TTFT improvement margin must be finite")
        if not math.isfinite(
            args.minimum_prompt_throughput_improvement_tok_s
        ):
            raise EvidenceError("prompt-throughput improvement margin must be finite")
        manifest_path = args.manifest.resolve()
        manifest = require_mapping(load_json(manifest_path), "manifest")
        cases = manifest.get("cases")
        if not isinstance(cases, list):
            raise EvidenceError("manifest cases must be a JSON array")
        if require_int(manifest.get("count"), "manifest count") != len(cases):
            raise EvidenceError("manifest count does not match its cases")
        if args.corpus is not None:
            corpus = args.corpus.resolve()
            expected_corpus_sha = manifest.get("corpus_sha256")
            if not corpus.is_file() or file_sha256(corpus) != expected_corpus_sha:
                raise EvidenceError("corpus SHA256 does not match the manifest")

        argument_paths = {
            "native_baseline": validate_result_dir(args.baseline, "native_baseline")
            / "benchmark_args.json",
            "native_candidate": validate_result_dir(
                args.candidate, "native_candidate"
            )
            / "benchmark_args.json",
        }
        if args.reference is not None:
            argument_paths["external_reference"] = validate_result_dir(
                args.reference, "external_reference"
            ) / "benchmark_args.json"
        argument_objects = {
            role: require_mapping(load_json(path), f"{role} benchmark_args")
            for role, path in argument_paths.items()
        }
        warmups = {
            role: validate_argument_contract(arguments, manifest, role)
            for role, arguments in argument_objects.items()
        }
        if len(set(warmups.values())) != 1:
            raise EvidenceError("benchmark warmup counts differ")
        warmup = next(iter(warmups.values()))
        measured_count = require_int(
            argument_objects["native_baseline"].get("number"),
            "native_baseline number",
        )
        if warmup + measured_count > len(cases):
            raise EvidenceError("manifest does not contain the measured cases")
        expected_cases = cases[warmup : warmup + measured_count]

        baseline = validate_run(
            args.baseline, "native_baseline", manifest, expected_cases
        )
        candidate = validate_run(
            args.candidate, "native_candidate", manifest, expected_cases
        )
        reference = (
            validate_run(
                args.reference, "external_reference", manifest, expected_cases
            )
            if args.reference is not None
            else None
        )
        report, exit_code = build_report(
            manifest_path,
            manifest,
            baseline,
            candidate,
            reference,
            args.minimum_mean_ttft_improvement_ms,
            args.minimum_prompt_throughput_improvement_tok_s,
            not args.allow_native_output_change,
        )
        if args.output is not None:
            output = args.output.resolve()
            if output.exists() and not args.force:
                raise EvidenceError(
                    f"refusing to replace {output}; pass --force explicitly"
                )
            atomic_write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return exit_code
    except EvidenceError as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact": "qwen36_evalscope_external_direction",
                    "decision": {
                        "candidate": "invalid_evidence",
                        "reason": str(error),
                        "exit_code": 1,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
