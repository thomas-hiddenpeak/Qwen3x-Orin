#!/usr/bin/env python3
"""Reproduce the pinned Qwen3.6-27B NVFP4 greedy oracle on an Orin.

This is an independent reference utility, not part of the C++ runtime.  It
loads the exact token IDs stored in the fixture so tokenizer behavior cannot
hide a model/runtime mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPOSITORY_ROOT / "tests/fixtures/qwen36-27b-nvfp4-greedy.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and compare the pinned vLLM greedy fixture."
    )
    parser.add_argument("model_dir", type=Path, help="materialized checkpoint")
    parser.add_argument(
        "--fixture", type=Path, default=DEFAULT_FIXTURE, help="oracle JSON"
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="sequential deterministic runs"
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="SHA-256 every checkpoint file recorded by the fixture",
    )
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="run despite a different vLLM version (comparison remains strict)",
    )
    parser.add_argument(
        "--output", type=Path, help="also write the normalized result JSON"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        help="override the fixture KV cache dtype (for policy comparisons)",
    )
    parser.add_argument(
        "--mamba-cache-dtype",
        help="override the fixture causal-conv cache dtype",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        help="override the fixture recurrent-state cache dtype",
    )
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read fixture {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"unsupported fixture schema in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(model_dir: Path, fixture: dict[str, Any]) -> None:
    for name, identity in fixture["source"]["files"].items():
        path = model_dir / name
        try:
            stat = path.stat()
        except OSError as error:
            raise RuntimeError(f"cannot stat {path}: {error}") from error
        expected_size = int(identity["size_bytes"])
        if not path.is_file() or stat.st_size != expected_size:
            raise RuntimeError(
                f"file identity mismatch for {path}: expected {expected_size} bytes, "
                f"got {stat.st_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != identity["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {identity['sha256']}, "
                f"got {actual_hash}"
            )


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise RuntimeError("--repeats must be positive")
    fixture = load_fixture(args.fixture.resolve())
    model_dir = args.model_dir.resolve()
    if args.verify_files:
        verify_files(model_dir, fixture)

    # These controls must be set before importing vLLM.
    for name, value in fixture["oracle"]["environment"].items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    expected_version = fixture["oracle"]["vllm"]
    if vllm_version != expected_version and not args.allow_version_drift:
        raise RuntimeError(
            f"vLLM version mismatch: expected {expected_version}, got "
            f"{vllm_version}; pass --allow-version-drift to collect diagnostics"
        )

    engine = fixture["engine"]
    started = time.monotonic()
    cache_options = {
        name: engine[name]
        for name in (
            "kv_cache_dtype",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
        )
        if name in engine
    }
    for name in (
        "kv_cache_dtype",
        "mamba_cache_dtype",
        "mamba_ssm_cache_dtype",
    ):
        override = getattr(args, name)
        if override is not None:
            cache_options[name] = override
    llm = LLM(
        model=str(model_dir),
        trust_remote_code=False,
        language_model_only=bool(engine["language_model_only"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        enforce_eager=bool(engine["enforce_eager"]),
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        disable_log_stats=True,
        mm_processor_cache_gb=float(engine["mm_processor_cache_gb"]),
        **cache_options,
    )
    initialized = time.monotonic()

    sampling = fixture["sampling"]
    params = SamplingParams(
        temperature=float(sampling["temperature"]),
        max_tokens=int(sampling["max_tokens"]),
        logprobs=int(sampling["logprobs"]),
        seed=int(sampling["seed"]),
    )
    prompt_ids = [int(token) for token in fixture["prompt"]["token_ids"]]
    expected = fixture["expected"]
    runs: list[dict[str, Any]] = []
    passed = True
    for run_index in range(args.repeats):
        run_started = time.monotonic()
        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False
        )[0].outputs[0]
        elapsed = time.monotonic() - run_started
        token_ids = [int(token) for token in result.token_ids]
        logprobs = [
            float(step[token].logprob)
            for token, step in zip(result.token_ids, result.logprobs or [])
        ]
        matches = (
            token_ids == expected["token_ids"]
            and result.text == expected["text"]
            and result.finish_reason == expected["finish_reason"]
        )
        passed = passed and matches
        runs.append(
            {
                "run": run_index + 1,
                "token_ids": token_ids,
                "text": result.text,
                "finish_reason": result.finish_reason,
                "chosen_logprobs": logprobs,
                "seconds": elapsed,
                "matches_expected": matches,
            }
        )

    repeatable = all(
        run["token_ids"] == runs[0]["token_ids"]
        and run["text"] == runs[0]["text"]
        and run["chosen_logprobs"] == runs[0]["chosen_logprobs"]
        for run in runs[1:]
    )
    passed = passed and repeatable
    report = {
        "fixture": str(args.fixture.resolve()),
        "model_dir": str(model_dir),
        "vllm": vllm_version,
        "engine_init_seconds": initialized - started,
        "cache_options": cache_options,
        "repeatable": repeatable,
        "passed": passed,
        "runs": runs,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.write(serialized)
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
