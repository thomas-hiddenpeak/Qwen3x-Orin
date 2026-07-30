#!/usr/bin/env python3
"""Fail-closed validation for the external long-context EvalScope corpus.

This validator intentionally has no corpus builder.  Long-context performance
evidence is accepted only when an externally materialized, authorized natural
corpus and its immutable manifest agree byte-for-byte and request-for-request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any


MODEL = "qwen3.6-27b-nvfp4"
ARTIFACT = "qwen36_evalscope_long_context_prefill_v1"
SELECTION_POLICY = (
    "complete natural conversation prefixes; no padding, repetition, "
    "truncation, concatenation, or synthetic text"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class BucketContract:
    token_window: tuple[int, int]
    max_sequence_length: int
    request_max_arena_bytes: int
    planned_arena_bytes: int
    phases: tuple[str, ...]


BUCKETS = {
    "p8k": BucketContract(
        token_window=(7_168, 8_191),
        max_sequence_length=8_192,
        request_max_arena_bytes=873_365_504,
        planned_arena_bytes=873_365_504,
        phases=("prefill1",),
    ),
    "p16k": BucketContract(
        token_window=(14_336, 16_383),
        max_sequence_length=16_384,
        request_max_arena_bytes=1_580_630_016,
        planned_arena_bytes=1_580_630_016,
        phases=("prefill1",),
    ),
    "p40k": BucketContract(
        token_window=(39_000, 40_000),
        max_sequence_length=40_960,
        request_max_arena_bytes=3_703_209_984,
        planned_arena_bytes=3_703_209_984,
        phases=("prefill1", "cold16"),
    ),
}
PHASE_MAX_TOKENS = {"prefill1": 1, "cold16": 16}
TOKENIZER_FILES = {
    "tokenizer_json_sha256": "tokenizer.json",
    "tokenizer_config_sha256": "tokenizer_config.json",
    "chat_template_jinja_sha256": "chat_template.jinja",
}


class ContractError(RuntimeError):
    """An input does not satisfy the retained long-context contract."""


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()),
            f"{label} must be a non-empty string")
    require("\x00" not in value and "\n" not in value and "\r" not in value,
            f"{label} contains a forbidden control character")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(actual == expected,
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(actual)}")


def exact_scalar(value: Any, expected: Any) -> bool:
    return type(value) is type(expected) and value == expected


def safe_corpus_name(value: Any, label: str) -> str:
    name = require_string(value, label)
    path = pathlib.PurePosixPath(name)
    require(len(path.parts) == 1 and path.name == name and name not in {".", ".."},
            f"{label} must be a plain filename")
    require("\t" not in name and "\\" not in name,
            f"{label} contains a forbidden filename character")
    return name


def load_manifest(path: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    require(path.is_file(), f"manifest does not exist: {path}")
    require_sha256(expected_sha256, "manifest SHA256 argument")
    actual = file_sha256(path)
    require(actual == expected_sha256,
            f"manifest SHA256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"manifest is not valid UTF-8 JSON: {error}") from error
    return require_object(value, "manifest")


def validate_source(value: Any, label: str) -> None:
    source = require_object(value, label)
    require_exact_keys(
        source,
        {
            "dataset",
            "revision",
            "source_artifact_sha256",
            "license_or_terms",
            "benchmark_authorization",
            "selection",
        },
        label,
    )
    require_string(source["dataset"], f"{label}.dataset")
    require_string(source["revision"], f"{label}.revision")
    require_sha256(source["source_artifact_sha256"],
                   f"{label}.source_artifact_sha256")
    require_string(source["license_or_terms"], f"{label}.license_or_terms")
    require_string(source["benchmark_authorization"],
                   f"{label}.benchmark_authorization")
    require(source["selection"] == SELECTION_POLICY,
            f"{label}.selection does not forbid synthetic/stitched prompts")


def validate_sources(value: Any) -> set[str]:
    sources = require_object(value, "sources")
    require(bool(sources), "sources must contain at least one authorized source")
    checked: set[str] = set()
    for source_id, source in sources.items():
        require_string(source_id, "sources key")
        require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", source_id) is not None,
                f"invalid source id: {source_id!r}")
        validate_source(source, f"sources.{source_id}")
        checked.add(source_id)
    return checked


def validate_tokenizer(value: Any, tokenizer_dir: pathlib.Path) -> None:
    tokenizer = require_object(value, "tokenizer")
    require_exact_keys(
        tokenizer,
        set(TOKENIZER_FILES) | {"add_generation_prompt", "enable_thinking"},
        "tokenizer",
    )
    require(tokenizer["add_generation_prompt"] is True,
            "tokenizer.add_generation_prompt must be true")
    require(tokenizer["enable_thinking"] is False,
            "tokenizer.enable_thinking must be false")
    for manifest_key, filename in TOKENIZER_FILES.items():
        expected = require_sha256(tokenizer[manifest_key],
                                  f"tokenizer.{manifest_key}")
        path = tokenizer_dir / filename
        require(path.is_file(), f"tokenizer file does not exist: {path}")
        actual = file_sha256(path)
        require(actual == expected,
                f"{filename} SHA256 mismatch: expected {expected}, got {actual}")


def validate_request_contract(value: Any) -> None:
    contract = require_object(value, "request_contract")
    expected = {
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
    }
    require_exact_keys(contract, set(expected), "request_contract")
    for key, expected_value in expected.items():
        require(exact_scalar(contract[key], expected_value),
                f"request_contract.{key} must be {expected_value!r}")


def validate_case_shape(
    value: Any,
    *,
    bucket: str,
    phase: str,
    ordinal: int,
    token_window: tuple[int, int],
    source_ids: set[str],
) -> dict[str, Any]:
    case = require_object(value, f"{bucket}.{phase}.cases[{ordinal}]")
    require_exact_keys(
        case,
        {
            "ordinal",
            "role",
            "source_id",
            "source_locator",
            "source_record_sha256",
            "prompt_tokens",
            "prompt_sha256",
            "request_sha256",
        },
        f"{bucket}.{phase}.cases[{ordinal}]",
    )
    require(exact_scalar(case["ordinal"], ordinal),
            f"{bucket}.{phase} case ordinal {ordinal} is not exact")
    expected_role = "warmup" if ordinal == 0 else "measured"
    require(case["role"] == expected_role,
            f"{bucket}.{phase} case {ordinal} role must be {expected_role}")
    source_id = require_string(
        case["source_id"], f"{bucket}.{phase} case {ordinal} source_id"
    )
    require(source_id in source_ids,
            f"{bucket}.{phase} case {ordinal} has an unknown source_id")
    require_string(case["source_locator"],
                   f"{bucket}.{phase} case {ordinal} source_locator")
    require_sha256(case["source_record_sha256"],
                   f"{bucket}.{phase} case {ordinal} source_record_sha256")
    prompt_tokens = case["prompt_tokens"]
    require(isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool),
            f"{bucket}.{phase} case {ordinal} prompt_tokens must be an integer")
    require(token_window[0] <= prompt_tokens <= token_window[1],
            f"{bucket}.{phase} case {ordinal} is outside {token_window}")
    require_sha256(case["prompt_sha256"],
                   f"{bucket}.{phase} case {ordinal} prompt_sha256")
    require_sha256(case["request_sha256"],
                   f"{bucket}.{phase} case {ordinal} request_sha256")
    return case


def validate_run_manifest(
    value: Any,
    *,
    bucket: str,
    phase: str,
    token_window: tuple[int, int],
    source_ids: set[str],
) -> dict[str, Any]:
    run = require_object(value, f"buckets.{bucket}.runs.{phase}")
    require_exact_keys(run, {"max_tokens", "corpus_file", "corpus_sha256",
                             "count", "cases"},
                       f"buckets.{bucket}.runs.{phase}")
    require(exact_scalar(run["max_tokens"], PHASE_MAX_TOKENS[phase]),
            f"{bucket}.{phase}.max_tokens is not exact")
    safe_corpus_name(run["corpus_file"], f"{bucket}.{phase}.corpus_file")
    require_sha256(run["corpus_sha256"], f"{bucket}.{phase}.corpus_sha256")
    require(exact_scalar(run["count"], 5),
            f"{bucket}.{phase}.count must be 5")
    cases = run["cases"]
    require(isinstance(cases, list) and len(cases) == 5,
            f"{bucket}.{phase}.cases must contain 5 cases")
    checked = [
        validate_case_shape(case, bucket=bucket, phase=phase, ordinal=ordinal,
                            token_window=token_window, source_ids=source_ids)
        for ordinal, case in enumerate(cases)
    ]
    locators = [
        (case["source_id"], case["source_locator"]) for case in checked
    ]
    prompt_hashes = [case["prompt_sha256"] for case in checked]
    require(len(set(locators)) == len(locators),
            f"{bucket}.{phase} repeats a source locator")
    require(len(set(prompt_hashes)) == len(prompt_hashes),
            f"{bucket}.{phase} repeats a prompt")
    return run


def validate_manifest_structure(manifest: dict[str, Any]) -> None:
    require_exact_keys(
        manifest,
        {"schema_version", "artifact", "sources", "tokenizer",
         "request_contract", "buckets"},
        "manifest",
    )
    require(exact_scalar(manifest["schema_version"], 1),
            "schema_version must be 1")
    require(manifest["artifact"] == ARTIFACT,
            f"artifact must be {ARTIFACT}")
    source_ids = validate_sources(manifest["sources"])
    validate_request_contract(manifest["request_contract"])
    buckets = require_object(manifest["buckets"], "buckets")
    require(set(buckets) == set(BUCKETS),
            f"buckets must be exactly {sorted(BUCKETS)}")
    for bucket_name, contract in BUCKETS.items():
        bucket = require_object(buckets[bucket_name], f"buckets.{bucket_name}")
        require_exact_keys(bucket, {"token_window", "runs"},
                           f"buckets.{bucket_name}")
        require(bucket["token_window"] == list(contract.token_window),
                f"buckets.{bucket_name}.token_window is not exact")
        runs = require_object(bucket["runs"], f"buckets.{bucket_name}.runs")
        require(set(runs) == set(contract.phases),
                f"buckets.{bucket_name}.runs must be {list(contract.phases)}")
        checked_runs = {
            phase: validate_run_manifest(
                runs[phase], bucket=bucket_name, phase=phase,
                token_window=contract.token_window, source_ids=source_ids
            )
            for phase in contract.phases
        }
        if bucket_name == "p40k":
            prefill_cases = checked_runs["prefill1"]["cases"]
            cold_cases = checked_runs["cold16"]["cases"]
            for ordinal, (prefill_case, cold_case) in enumerate(
                zip(prefill_cases, cold_cases)
            ):
                for key in ("source_id", "source_locator", "source_record_sha256",
                            "prompt_tokens", "prompt_sha256"):
                    require(prefill_case[key] == cold_case[key],
                            f"p40k phases differ at case {ordinal} field {key}")


def validate_request(
    value: Any,
    case: dict[str, Any],
    *,
    bucket: str,
    phase: str,
    ordinal: int,
) -> None:
    request = require_object(value, f"corpus request {ordinal}")
    require_exact_keys(
        request,
        {"model", "prompt", "max_tokens", "temperature", "seed", "stream",
         "stream_options"},
        f"corpus request {ordinal}",
    )
    require(request["model"] == MODEL, f"request {ordinal} model mismatch")
    require(exact_scalar(request["max_tokens"], PHASE_MAX_TOKENS[phase]),
            f"request {ordinal} max_tokens mismatch")
    require(exact_scalar(request["temperature"], 0.0),
            f"request {ordinal} temperature mismatch")
    require(exact_scalar(request["seed"], 42),
            f"request {ordinal} seed mismatch")
    require(request["stream"] is True, f"request {ordinal} stream mismatch")
    require(isinstance(request["stream_options"], dict) and
            set(request["stream_options"]) == {"include_usage"} and
            request["stream_options"]["include_usage"] is True,
            f"request {ordinal} stream_options mismatch")
    prompt = request["prompt"]
    require(isinstance(prompt, list) and bool(prompt),
            f"request {ordinal} prompt must be a non-empty token-ID array")
    require(all(isinstance(token, int) and not isinstance(token, bool) and
                0 <= token <= 0xFFFFFFFF for token in prompt),
            f"request {ordinal} prompt contains a non-uint32 token ID")
    require(len(prompt) == case["prompt_tokens"],
            f"request {ordinal} prompt length mismatch")
    prompt_sha = sha256_bytes(canonical_json(prompt))
    require(prompt_sha == case["prompt_sha256"],
            f"request {ordinal} prompt SHA256 mismatch")
    request_sha = sha256_bytes(canonical_json(request))
    require(request_sha == case["request_sha256"],
            f"request {ordinal} canonical request SHA256 mismatch")


def validate_selected_corpus(
    manifest: dict[str, Any],
    corpus_dir: pathlib.Path,
    bucket: str,
    phase: str,
) -> tuple[pathlib.Path, dict[str, Any]]:
    contract = BUCKETS[bucket]
    require(phase in contract.phases,
            f"phase {phase} is not defined for bucket {bucket}")
    run = manifest["buckets"][bucket]["runs"][phase]
    corpus_path = corpus_dir / run["corpus_file"]
    require(corpus_path.is_file(),
            f"authorized corpus is missing: {corpus_path}")
    actual_sha = file_sha256(corpus_path)
    require(actual_sha == run["corpus_sha256"],
            f"corpus SHA256 mismatch: expected {run['corpus_sha256']}, "
            f"got {actual_sha}")
    try:
        raw_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ContractError(f"corpus is not valid UTF-8: {error}") from error
    require(len(raw_lines) == run["count"] and all(raw_lines),
            f"corpus must contain exactly {run['count']} non-empty lines")
    for ordinal, (line, case) in enumerate(zip(raw_lines, run["cases"])):
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"corpus line {ordinal + 1} is not JSON: {error}"
            ) from error
        validate_request(request, case, bucket=bucket, phase=phase,
                         ordinal=ordinal)
    return corpus_path, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one long-context EvalScope bucket/phase"
    )
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bucket", choices=sorted(BUCKETS), required=True)
    parser.add_argument("--phase", choices=sorted(PHASE_MAX_TOKENS), required=True)
    parser.add_argument("--format", choices=("json", "nul"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest, args.manifest_sha256)
        validate_manifest_structure(manifest)
        validate_tokenizer(manifest["tokenizer"], args.tokenizer_dir)
        corpus_path, run = validate_selected_corpus(
            manifest, args.corpus_dir, args.bucket, args.phase
        )
        contract = BUCKETS[args.bucket]
        result = {
            "bucket": args.bucket,
            "phase": args.phase,
            "corpus_path": str(corpus_path),
            "corpus_sha256": run["corpus_sha256"],
            "max_tokens": PHASE_MAX_TOKENS[args.phase],
            "number": 4,
            "warmup": 1,
            "max_sequence_length": contract.max_sequence_length,
            "request_max_arena_bytes": contract.request_max_arena_bytes,
            "planned_arena_bytes": contract.planned_arena_bytes,
            "token_window_min": contract.token_window[0],
            "token_window_max": contract.token_window[1],
        }
        if args.format == "json":
            print(json.dumps(result, sort_keys=True))
        else:
            for key in (
                "bucket", "phase", "corpus_path", "corpus_sha256",
                "max_tokens", "number", "warmup", "max_sequence_length",
                "request_max_arena_bytes", "planned_arena_bytes",
                "token_window_min", "token_window_max",
            ):
                sys.stdout.buffer.write(str(result[key]).encode("utf-8") + b"\0")
    except ContractError as error:
        print(f"long-context manifest rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
