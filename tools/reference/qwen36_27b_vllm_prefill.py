#!/usr/bin/env python3
"""Measure matched batch-one vLLM Prefill latency from raw token IDs.

This is an offline reference utility.  It does not import the native Q3X
runtime, mutate a checkpoint, or require a tokenizer round trip.  The built-in
P65/P129/P257/P513/P1025 profiles reproduce the repository's raw-token prompt
construction contract; caller-supplied token IDs are passed through unchanged.

The primary timing is RequestOutput.metrics.first_token_ts minus scheduled_ts.
Those fields are populated by vLLM's engine core when log statistics are
enabled.  If a runtime does not expose a coherent pair, the field is emitted as
null and the record explicitly falls back to the wall time of the batch-one,
one-output-token LLM.generate() call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


PINNED_VLLM_VERSION = "0.23.1rc1.dev269+gccd49f682"
PROFILE_TOKEN_COUNTS = (65, 129, 257, 513, 1025)
PROFILE_HASHES = {
    "P65": "b4c8a28798e5246ccac2ce86d1baeed937cc1438daee0eaefa960b9e5bb51e0b",
    "P129": "1e8bccd68253050cbfc90cd7062f8dc012f74c74bc13f8b19349be2e66d144de",
    "P257": "e24c73f282d6dabf42b48ea8fb9b71a452cdbaadc9fc0d4eebf7672ee584e2ce",
    "P513": "45ae21468cb8a0b0ac64566985daba4c0b78e429b99cac3453a98664587efd6d",
    "P1025": "b7cd0078f52252ab75b095679c8f757c2c6de19d66692764d65cb39c5a8743e8",
}

# For a requested length P, this is four fixed opening tokens, P-13 copies of
# token 23066, and nine fixed closing tokens.  P65/P129/P513 are bit-for-bit
# identical to benchmarks/qwen36-27b-sm87-prefill-prompts-v1.json.
PROFILE_PREFIX = (248045, 846, 198, 14556)
PROFILE_BODY_TOKEN = 23066
PROFILE_SUFFIX = (248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271)


@dataclass(frozen=True)
class PromptCase:
    label: str
    token_ids: tuple[int, ...]
    construction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline batch-one/output-one vLLM Prefill alignment using exact "
            "raw token IDs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: /home/rm01/setup/.venv/bin/python "
            "tools/reference/qwen36_27b_vllm_prefill.py "
            "--model /absolute/checkpoint --tokenizer /absolute/tokenizer "
            "--profiles P65,P513 --attention-backend FLASHINFER "
            "--output /tmp/vllm-prefill.jsonl"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="exact local model/checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="exact local tokenizer directory (may equal --model)",
    )
    parser.add_argument(
        "--profiles",
        help="comma-separated subset of P65,P129,P257,P513,P1025",
    )
    parser.add_argument(
        "--prompt-ids",
        action="append",
        default=[],
        metavar="[LABEL=]ID,ID,...",
        help="raw token-ID prompt; repeat for multiple prompts",
    )
    parser.add_argument(
        "--prompt-ids-file",
        type=Path,
        help=(
            "JSON mapping LABEL to ID arrays, or a JSON list of records with "
            "label and prompt_token_ids"
        ),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "FLASHINFER"),
        default="auto",
        help="explicit FLASHINFER request is validated against worker truth",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "float16", "bfloat16", "fp8", "fp8_e4m3"),
        default="bfloat16",
    )
    parser.add_argument(
        "--mamba-cache-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.8
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="defaults to the longest selected prompt plus one output token",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="defaults to --max-model-len; must fit a whole unchunked prompt",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graphs/compiled graph capture",
    )
    parser.add_argument(
        "--expected-vllm-version",
        default=PINNED_VLLM_VERSION,
        help="checked against vllm.__version__",
    )
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="run with a different vLLM version while recording it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output file; stdout when omitted",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return sha256_bytes(",".join(str(token) for token in token_ids).encode("ascii"))


def require_directory(path: Path, name: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cannot resolve {name} directory {path}: {error}"
        ) from error
    if not resolved.is_dir():
        raise RuntimeError(f"{name} path is not a directory: {resolved}")
    return resolved


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from error


def parse_token_array(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty JSON token-ID array")
    token_ids: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise RuntimeError(
                f"{context}[{index}] must be a non-negative integer, got {token!r}"
            )
        token_ids.append(token)
    return tuple(token_ids)


def parse_inline_prompt(value: str, index: int) -> PromptCase:
    if "=" in value:
        label, encoded = value.split("=", 1)
        label = label.strip()
    else:
        label = f"raw{index}"
        encoded = value
    if not label:
        raise RuntimeError(f"--prompt-ids entry {index} has an empty label")
    fields = [field for field in re.split(r"[\s,]+", encoded.strip()) if field]
    if not fields:
        raise RuntimeError(f"--prompt-ids entry {index} is empty")
    try:
        token_ids = tuple(int(field, 10) for field in fields)
    except ValueError as error:
        raise RuntimeError(
            f"--prompt-ids entry {index} contains a non-decimal token ID"
        ) from error
    if any(token < 0 for token in token_ids):
        raise RuntimeError(f"--prompt-ids entry {index} contains a negative token ID")
    return PromptCase(label=label, token_ids=token_ids, construction="raw-cli")


def parse_prompt_file(path: Path) -> list[PromptCase]:
    value = load_json(path.expanduser().resolve(strict=True))
    cases: list[PromptCase] = []
    if isinstance(value, dict):
        items = list(value.items())
        for label, token_ids in items:
            if not isinstance(label, str) or not label:
                raise RuntimeError("prompt file mapping keys must be non-empty labels")
            cases.append(
                PromptCase(
                    label=label,
                    token_ids=parse_token_array(token_ids, f"prompt {label!r}"),
                    construction="raw-json-mapping",
                )
            )
        return cases
    if not isinstance(value, list) or not value:
        raise RuntimeError("prompt file must be a non-empty JSON object or list")
    for index, record in enumerate(value, start=1):
        if not isinstance(record, dict):
            raise RuntimeError(f"prompt file record {index} must be an object")
        label = record.get("label", f"raw{index}")
        if not isinstance(label, str) or not label:
            raise RuntimeError(f"prompt file record {index} has an invalid label")
        cases.append(
            PromptCase(
                label=label,
                token_ids=parse_token_array(
                    record.get("prompt_token_ids"), f"prompt file record {index}"
                ),
                construction="raw-json-record",
            )
        )
    return cases


def make_profile(label: str) -> PromptCase:
    try:
        token_count = int(label.removeprefix("P"), 10)
    except ValueError as error:
        raise RuntimeError(f"invalid built-in profile {label!r}") from error
    if token_count not in PROFILE_TOKEN_COUNTS:
        raise RuntimeError(f"unsupported built-in profile {label!r}")
    token_ids = (
        PROFILE_PREFIX
        + (PROFILE_BODY_TOKEN,) * (token_count - 13)
        + PROFILE_SUFFIX
    )
    if len(token_ids) != token_count:
        raise AssertionError("built-in profile token-count construction drift")
    actual_hash = token_ids_sha256(token_ids)
    if actual_hash != PROFILE_HASHES[label]:
        raise AssertionError(
            f"built-in profile {label} identity drift: got {actual_hash}"
        )
    return PromptCase(
        label=label,
        token_ids=token_ids,
        construction="q3x-fixed-open/body-23066/fixed-close-v1",
    )


def resolve_prompts(args: argparse.Namespace) -> list[PromptCase]:
    has_raw = bool(args.prompt_ids or args.prompt_ids_file)
    if args.profiles and has_raw:
        raise RuntimeError("--profiles cannot be combined with raw prompt inputs")
    cases: list[PromptCase] = []
    if has_raw:
        cases.extend(
            parse_inline_prompt(value, index)
            for index, value in enumerate(args.prompt_ids, start=1)
        )
        if args.prompt_ids_file is not None:
            cases.extend(parse_prompt_file(args.prompt_ids_file))
    else:
        selected = (
            [field.strip().upper() for field in args.profiles.split(",")]
            if args.profiles
            else [f"P{count}" for count in PROFILE_TOKEN_COUNTS]
        )
        if not selected or any(not label for label in selected):
            raise RuntimeError("--profiles must contain at least one profile")
        cases.extend(make_profile(label) for label in selected)
    if not cases:
        raise RuntimeError("no prompts selected")
    labels = [case.label for case in cases]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"prompt labels must be unique, got {labels!r}")
    return cases


def model_vocab_size(model_config: dict[str, Any]) -> int:
    candidates = [model_config]
    text_config = model_config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config)
    for candidate in candidates:
        value = candidate.get("vocab_size")
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("model config does not expose a positive vocab_size")


def validate_prompt_ids(cases: list[PromptCase], vocab_size: int) -> None:
    for case in cases:
        if not case.token_ids:
            raise RuntimeError(f"prompt {case.label} is empty")
        maximum = max(case.token_ids)
        if maximum >= vocab_size:
            raise RuntimeError(
                f"prompt {case.label} token ID {maximum} is outside vocab_size "
                f"{vocab_size}"
            )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).removeprefix("torch.")


def effective_engine_config(llm: Any) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    model = config.model_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    attention = config.attention_config
    compilation = config.compilation_config
    return {
        "dtype": json_scalar(model.dtype),
        "quantization": json_scalar(model.quantization),
        "max_model_len": int(model.max_model_len),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
        "enable_prefix_caching": bool(cache.enable_prefix_caching),
        "kv_cache_dtype": json_scalar(cache.cache_dtype),
        "mamba_cache_dtype": json_scalar(cache.mamba_cache_dtype),
        "mamba_ssm_cache_dtype": json_scalar(cache.mamba_ssm_cache_dtype),
        "speculative_config": (
            None if config.speculative_config is None else "enabled"
        ),
        "attention_backend_config": json_scalar(attention.backend),
        "enforce_eager": bool(model.enforce_eager),
        "compilation_mode": json_scalar(getattr(compilation, "mode", None)),
        "cudagraph_mode": json_scalar(
            getattr(compilation, "cudagraph_mode", None)
        ),
        "language_model_only": bool(model.multimodal_config.language_model_only),
    }


def validate_effective_engine_config(config: dict[str, Any]) -> None:
    failures: list[str] = []
    if config["max_num_seqs"] != 1:
        failures.append(f"max_num_seqs={config['max_num_seqs']}")
    if config["enable_chunked_prefill"]:
        failures.append("chunked Prefill is enabled")
    if config["enable_prefix_caching"]:
        failures.append("prefix caching is enabled")
    if config["speculative_config"] is not None:
        failures.append("speculation/MTP is enabled")
    if failures:
        raise RuntimeError("unsafe comparison engine config: " + "; ".join(failures))


def backend_identity(backend: type[Any]) -> tuple[str, str]:
    full_name = backend.full_cls_name()
    if isinstance(full_name, tuple) and len(full_name) == 2:
        class_path = f"{full_name[0]}.{full_name[1]}"
    else:
        class_path = f"{backend.__module__}.{backend.__qualname__}"
    try:
        name = str(backend.get_name())
    except Exception:
        name = backend.__name__
    return name, class_path


def actual_attention_backends(llm: Any) -> dict[str, Any]:
    try:
        worker = llm.llm_engine.model_executor.driver_worker.worker
        runner = worker.model_runner
        nested_groups = runner.attn_groups
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "cannot inspect actual attention backends; this tool requires the "
            "local single-process vLLM worker"
        ) from error
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for groups in nested_groups:
        for group in groups:
            name, class_path = backend_identity(group.backend)
            key = (name, class_path)
            entry = aggregated.setdefault(
                key,
                {
                    "name": name,
                    "class_path": class_path,
                    "layer_count": 0,
                    "example_layers": [],
                },
            )
            layer_names = [str(layer) for layer in group.layer_names]
            entry["layer_count"] += len(layer_names)
            remaining = 4 - len(entry["example_layers"])
            if remaining > 0:
                entry["example_layers"].extend(layer_names[:remaining])
    if not aggregated:
        raise RuntimeError("loaded worker exposes no actual attention backend groups")
    return {
        "source": "driver_worker.worker.model_runner.attn_groups",
        "groups": sorted(aggregated.values(), key=lambda item: item["class_path"]),
    }


def validate_requested_backend(
    requested: str, actual_backends: dict[str, Any]
) -> None:
    if requested != "FLASHINFER":
        return
    groups = actual_backends["groups"]
    if not any("flashinfer" in group["class_path"].lower() for group in groups):
        raise RuntimeError(
            "FLASHINFER was requested but no loaded attention group uses a "
            "FlashInfer backend"
        )


def trusted_scheduled_span_ms(
    request_output: Any, outer_wall_ms: float
) -> tuple[float | None, str]:
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return None, "RequestOutput.metrics is null"
    scheduled = getattr(metrics, "scheduled_ts", None)
    first_token = getattr(metrics, "first_token_ts", None)
    if not isinstance(scheduled, (int, float)) or not isinstance(
        first_token, (int, float)
    ):
        return None, "scheduled_ts/first_token_ts are not numeric"
    if not math.isfinite(scheduled) or not math.isfinite(first_token):
        return None, "scheduled_ts/first_token_ts are non-finite"
    if scheduled <= 0.0 or first_token < scheduled:
        return None, "scheduled_ts/first_token_ts are unset or reversed"
    span_ms = (first_token - scheduled) * 1000.0
    # The outer call includes scheduling and frontend work, so a substantially
    # larger engine-core span means the fields cannot be trusted as one request.
    tolerance_ms = max(5.0, outer_wall_ms * 0.10)
    if span_ms > outer_wall_ms + tolerance_ms:
        return None, "engine-core span exceeds enclosing LLM.generate wall time"
    generation_tokens = getattr(metrics, "num_generation_tokens", None)
    if generation_tokens not in (None, 1):
        return None, f"metrics report {generation_tokens} generation tokens"
    return span_ms, "trusted RequestOutput.metrics monotonic engine-core timestamps"


def run_once(
    llm: Any,
    tokens_prompt_type: type[Any],
    sampling_params: Any,
    case: PromptCase,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    outputs = llm.generate(
        [tokens_prompt_type(prompt_token_ids=list(case.token_ids))],
        sampling_params,
        use_tqdm=False,
    )
    finished_ns = time.perf_counter_ns()
    outer_wall_ms = (finished_ns - started_ns) / 1_000_000.0
    if len(outputs) != 1:
        raise RuntimeError(
            f"prompt {case.label}: vLLM returned {len(outputs)} requests"
        )
    request_output = outputs[0]
    reported_prompt_ids = getattr(request_output, "prompt_token_ids", None)
    if reported_prompt_ids is None or list(reported_prompt_ids) != list(case.token_ids):
        raise RuntimeError(
            f"prompt {case.label}: vLLM-reported prompt token IDs differ from input"
        )
    completions = getattr(request_output, "outputs", None)
    if not isinstance(completions, list) or len(completions) != 1:
        raise RuntimeError(f"prompt {case.label}: expected one completion")
    output_token_ids = list(completions[0].token_ids)
    if len(output_token_ids) != 1:
        raise RuntimeError(
            f"prompt {case.label}: expected one output token, got {output_token_ids!r}"
        )
    cached_tokens = getattr(request_output, "num_cached_tokens", None)
    if cached_tokens not in (None, 0):
        raise RuntimeError(
            f"prompt {case.label}: expected zero cached tokens, got {cached_tokens}"
        )
    scheduled_ms, metric_reason = trusted_scheduled_span_ms(
        request_output, outer_wall_ms
    )
    effective_ms = scheduled_ms if scheduled_ms is not None else outer_wall_ms
    effective_source = (
        "request_output.metrics.scheduled_to_first_token"
        if scheduled_ms is not None
        else "engine_call_wall_ttft_fallback"
    )
    return {
        "prompt_tokens": len(reported_prompt_ids),
        "reported_prompt_tokens": len(reported_prompt_ids),
        "scheduled_to_first_token_ms": scheduled_ms,
        "scheduled_metric_trusted": scheduled_ms is not None,
        "scheduled_metric_reason": metric_reason,
        "outer_wall_ms": outer_wall_ms,
        "effective_ttft_ms": effective_ms,
        "effective_ttft_source": effective_source,
        "scheduled_prefill_tokens_per_second": (
            len(reported_prompt_ids) * 1000.0 / scheduled_ms
            if scheduled_ms is not None and scheduled_ms > 0.0
            else None
        ),
        "effective_prefill_tokens_per_second": (
            len(reported_prompt_ids) * 1000.0 / effective_ms
            if effective_ms > 0.0
            else None
        ),
        "first_token_id": int(output_token_ids[0]),
        "num_cached_tokens": cached_tokens,
    }


def output_stream(path: Path | None, protected: tuple[Path, ...]) -> TextIO:
    if path is None:
        return sys.stdout
    resolved = path.expanduser().resolve()
    if any(resolved == root or resolved.is_relative_to(root) for root in protected):
        raise RuntimeError(
            f"refusing to write output inside model/tokenizer path: {resolved}"
        )
    if not resolved.parent.is_dir():
        raise RuntimeError(f"output parent directory does not exist: {resolved.parent}")
    return resolved.open("w", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.measurements < 1:
        raise RuntimeError("--warmups must be non-negative and --measurements positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise RuntimeError("--gpu-memory-utilization must be between zero and one")

    model_dir = require_directory(args.model, "model")
    tokenizer_dir = require_directory(args.tokenizer, "tokenizer")
    model_config_path = model_dir / "config.json"
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if not model_config_path.is_file():
        raise RuntimeError(f"missing model config: {model_config_path}")
    if not tokenizer_json_path.is_file():
        raise RuntimeError(f"missing tokenizer JSON: {tokenizer_json_path}")
    model_config_json = load_json(model_config_path)
    if not isinstance(model_config_json, dict):
        raise RuntimeError(f"model config is not a JSON object: {model_config_path}")

    cases = resolve_prompts(args)
    vocab_size = model_vocab_size(model_config_json)
    validate_prompt_ids(cases, vocab_size)
    longest_prompt = max(len(case.token_ids) for case in cases)
    max_model_len = args.max_model_len or longest_prompt + 1
    if max_model_len < longest_prompt + 1:
        raise RuntimeError(
            f"--max-model-len {max_model_len} cannot fit P={longest_prompt} plus output"
        )
    max_num_batched_tokens = args.max_num_batched_tokens or max_model_len
    if max_num_batched_tokens < longest_prompt:
        raise RuntimeError(
            "--max-num-batched-tokens must fit the longest whole prompt when "
            "chunked Prefill is disabled"
        )

    # Set controls before importing vLLM.  Single-process V1 lets us inspect
    # the same worker/backend instances that execute the benchmark.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    if (
        vllm_version != args.expected_vllm_version
        and not args.allow_version_drift
    ):
        raise RuntimeError(
            f"vLLM version mismatch: expected {args.expected_vllm_version}, got "
            f"{vllm_version}; pass --allow-version-drift for diagnostics"
        )

    attention_config = (
        {"backend": "FLASHINFER"}
        if args.attention_backend == "FLASHINFER"
        else None
    )
    engine_started_ns = time.perf_counter_ns()
    llm = LLM(
        model=str(model_dir),
        tokenizer=str(tokenizer_dir),
        trust_remote_code=False,
        language_model_only=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        speculative_config=None,
        spec_method=None,
        spec_model=None,
        spec_tokens=None,
        disable_log_stats=False,
        mm_processor_cache_gb=0.0,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_cache_dtype=args.mamba_cache_dtype,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        attention_config=attention_config,
        seed=0,
    )
    engine_init_ms = (time.perf_counter_ns() - engine_started_ns) / 1_000_000.0
    engine_config = effective_engine_config(llm)
    validate_effective_engine_config(engine_config)
    backends = actual_attention_backends(llm)
    validate_requested_backend(args.attention_backend, backends)

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        n=1,
        seed=0,
        ignore_eos=True,
        detokenize=False,
    )
    runtime_versions = {
        "python": sys.version.split()[0],
        "vllm": vllm_version,
        "vllm_distribution": package_version("vllm"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer_python": package_version("flashinfer-python"),
        "transformers": package_version("transformers"),
        "tokenizers": package_version("tokenizers"),
    }
    identity = {
        "model_path": str(model_dir),
        "model_config_sha256": sha256_file(model_config_path),
        "tokenizer_path": str(tokenizer_dir),
        "tokenizer_json_sha256": sha256_file(tokenizer_json_path),
        "vocab_size": vocab_size,
    }
    requested_config = {
        "batch_size": 1,
        "output_tokens": 1,
        "temperature": 0.0,
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "speculation_mtp": False,
        "attention_backend": args.attention_backend,
        "warmups_per_prompt": args.warmups,
        "measurements_per_prompt": args.measurements,
    }

    sink = output_stream(args.output, (model_dir, tokenizer_dir))
    close_sink = sink is not sys.stdout
    try:
        for case in cases:
            stable_first_token: int | None = None
            for _ in range(args.warmups):
                warmup = run_once(llm, TokensPrompt, sampling, case)
                first_token = warmup["first_token_id"]
                if stable_first_token is None:
                    stable_first_token = first_token
                elif first_token != stable_first_token:
                    raise RuntimeError(
                        f"prompt {case.label}: greedy first token changed during warmup"
                    )
            for measurement_index in range(1, args.measurements + 1):
                measured = run_once(llm, TokensPrompt, sampling, case)
                first_token = measured["first_token_id"]
                if stable_first_token is None:
                    stable_first_token = first_token
                elif first_token != stable_first_token:
                    raise RuntimeError(
                        f"prompt {case.label}: greedy first token is not repeatable"
                    )
                record = {
                    "schema_version": 1,
                    "record_type": "vllm_prefill_measurement",
                    "profile": case.label,
                    "measurement_index": measurement_index,
                    "prompt_token_ids_csv_sha256": token_ids_sha256(case.token_ids),
                    "prompt_construction": case.construction,
                    **measured,
                    "engine_init_ms": engine_init_ms,
                    "runtime_versions": runtime_versions,
                    "model_identity": identity,
                    "requested_config": requested_config,
                    "effective_engine_config": engine_config,
                    "actual_backends": backends,
                    "metric_semantics": {
                        "scheduled_to_first_token_ms": (
                            "RequestOutput.metrics.first_token_ts - scheduled_ts; "
                            "both engine-core monotonic timestamps"
                        ),
                        "outer_wall_ms": (
                            "wall time enclosing one batch-one/output-one "
                            "LLM.generate call"
                        ),
                        "token_rate_numerator": "all prompt tokens P",
                    },
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
    finally:
        if close_sink:
            sink.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
