#!/usr/bin/env python3
"""Capture compact layer-boundary oracles from the pinned vLLM runtime.

The capture contains hashes, statistics, and fixed samples rather than full
activation vectors. It is intended to locate the first divergent native layer;
the separate greedy fixture remains the normative end-to-end gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GREEDY_FIXTURE = (
    REPOSITORY_ROOT / "tests/fixtures/qwen36-27b-nvfp4-greedy.json"
)
SAMPLE_INDICES = [
    0,
    1,
    2,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    3071,
    4095,
    5119,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact Qwen3.6-27B layer-boundary oracles."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--greedy-fixture", type=Path, default=DEFAULT_GREEDY_FIXTURE
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-version-drift", action="store_true")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--mamba-cache-dtype")
    parser.add_argument("--mamba-ssm-cache-dtype")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise RuntimeError(f"hook output has no tensor: {type(value)!r}")


def last_row_clone(value: Any) -> Any:
    tensor = first_tensor(value)
    if tensor.ndim == 0 or tensor.shape[-1] == 0:
        raise RuntimeError(f"unexpected hook tensor shape {tuple(tensor.shape)}")
    flattened = tensor.reshape(-1, tensor.shape[-1])
    return flattened[-1].detach().contiguous().clone()


def summarize_tensor(tensor: Any, sample_indices: list[int]) -> dict[str, Any]:
    import torch

    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    fp32 = cpu.float()
    finite = torch.isfinite(fp32)
    finite_count = int(finite.sum().item())
    if finite_count != fp32.numel():
        raise RuntimeError("non-finite value in layer oracle")
    values = fp32.tolist()
    mean = float(fp32.mean().item())
    rms = float(torch.sqrt(torch.mean(fp32 * fp32)).item())
    return {
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "length": int(cpu.numel()),
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "mean": mean,
        "rms": rms,
        "minimum": float(fp32.min().item()),
        "maximum": float(fp32.max().item()),
        # Indices are recorded once in report.semantics.samples. Keeping only
        # the values here makes the checked-in 64-layer fixture compact.
        "samples": [
            float(values[index]) for index in sample_indices if index < len(values)
        ],
    }


def resolve_causal_model(runner_model: Any) -> Any:
    if hasattr(runner_model, "language_model"):
        candidate = runner_model.language_model
        if hasattr(candidate, "model") and hasattr(candidate, "logits_processor"):
            return candidate
    if hasattr(runner_model, "model") and hasattr(
        runner_model, "logits_processor"
    ):
        return runner_model
    raise RuntimeError(
        f"cannot locate Qwen causal model below {type(runner_model).__name__}"
    )


def main() -> int:
    args = parse_args()
    golden = load_json(args.greedy_fixture.resolve())
    for name, value in golden["oracle"]["environment"].items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.inputs import TokensPrompt

    expected_version = golden["oracle"]["vllm"]
    if vllm_version != expected_version and not args.allow_version_drift:
        raise RuntimeError(
            f"vLLM version mismatch: expected {expected_version}, got "
            f"{vllm_version}"
        )

    engine = golden["engine"]
    initialized_at = time.monotonic()
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
        model=str(args.model_dir.resolve()),
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
    engine_ready_at = time.monotonic()

    worker = llm.llm_engine.model_executor.driver_worker.worker
    runner_model = worker.model_runner.model
    causal = resolve_causal_model(runner_model)
    transformer = causal.model
    layers = list(transformer.layers)
    if len(layers) != 64:
        raise RuntimeError(f"expected 64 layers, got {len(layers)}")

    embeddings: list[Any] = []
    layer_outputs: list[list[tuple[Any, Any]]] = [[] for _ in layers]
    final_norms: list[Any] = []
    logits: list[Any] = []
    handles = []

    def embedding_hook(_module: Any, _inputs: Any, output: Any) -> None:
        embeddings.append(last_row_clone(output))

    def make_layer_hook(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index} did not return a pair")
            layer_outputs[layer_index].append(
                (last_row_clone(output[0]), last_row_clone(output[1]))
            )

        return hook

    def norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        final_norms.append(last_row_clone(output))

    def logits_hook(_module: Any, _inputs: Any, output: Any) -> None:
        logits.append(last_row_clone(output))

    handles.append(transformer.embed_tokens.register_forward_hook(embedding_hook))
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(index)))
    handles.append(transformer.norm.register_forward_hook(norm_hook))
    handles.append(causal.logits_processor.register_forward_hook(logits_hook))

    prompt_ids = [int(token) for token in golden["prompt"]["token_ids"]]
    params = SamplingParams(temperature=0.0, max_tokens=2, logprobs=5, seed=0)
    generation_started = time.monotonic()
    result = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False
    )[0].outputs[0]
    generation_finished = time.monotonic()
    for handle in handles:
        handle.remove()

    if list(result.token_ids) != golden["expected"]["token_ids"][:2]:
        raise RuntimeError(
            f"capture generation drifted: got {list(result.token_ids)!r}"
        )
    capture_count = 2
    if not (
        len(embeddings) == capture_count
        and len(final_norms) == capture_count
        and len(logits) == capture_count
        and all(len(outputs) == capture_count for outputs in layer_outputs)
    ):
        raise RuntimeError(
            "unexpected hook counts: "
            f"embeddings={len(embeddings)}, final_norms={len(final_norms)}, "
            f"logits={len(logits)}, layer_counts="
            f"{[len(outputs) for outputs in layer_outputs]}"
        )

    phases = []
    phase_names = ["prefill_last_prompt_token", "decode_after_token_77517"]
    phase_inputs = [prompt_ids[-1], int(result.token_ids[0])]
    phase_positions = [len(prompt_ids) - 1, len(prompt_ids)]
    for phase_index, phase_name in enumerate(phase_names):
        logit = logits[phase_index].float().cpu()
        top_values, top_indices = torch.topk(logit, k=20)
        logsumexp = float(torch.logsumexp(logit, dim=-1).item())
        phases.append(
            {
                "name": phase_name,
                "input_token_id": phase_inputs[phase_index],
                "position": phase_positions[phase_index],
                "predicted_token_id": int(result.token_ids[phase_index]),
                "embedding": summarize_tensor(
                    embeddings[phase_index], SAMPLE_INDICES
                ),
                "layers": [
                    {
                        "index": layer_index,
                        "hidden": summarize_tensor(
                            layer_outputs[layer_index][phase_index][0],
                            SAMPLE_INDICES,
                        ),
                        "residual": summarize_tensor(
                            layer_outputs[layer_index][phase_index][1],
                            SAMPLE_INDICES,
                        ),
                    }
                    for layer_index in range(len(layers))
                ],
                "final_norm": summarize_tensor(
                    final_norms[phase_index], SAMPLE_INDICES
                ),
                "logits": {
                    "dtype": str(logits[phase_index].dtype).removeprefix(
                        "torch."
                    ),
                    "length": int(logits[phase_index].numel()),
                    "logsumexp": logsumexp,
                    "top20": [
                        {"token_id": int(token), "logit": float(value)}
                        for token, value in zip(
                            top_indices.tolist(), top_values.tolist()
                        )
                    ],
                    "chosen_logprob": float(
                        logit[int(result.token_ids[phase_index])].item()
                        - logsumexp
                    ),
                },
            }
        )

    report = {
        "schema_version": 1,
        "fixture": "qwen36-27b-nvfp4-layer-boundaries",
        "source_greedy_fixture": args.greedy_fixture.name,
        "source_revision": golden["source"]["revision"],
        "oracle": {
            "vllm": vllm_version,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "engine_init_seconds": engine_ready_at - initialized_at,
            "generation_seconds": generation_finished - generation_started,
            "cache_options": cache_options,
        },
        "semantics": {
            "layer_hidden": "BF16 MLP output returned by the decoder layer",
            "layer_residual": "BF16 residual accumulator returned by the decoder layer",
            "hash": "SHA-256 of contiguous raw tensor bytes on this oracle",
            "samples": SAMPLE_INDICES,
            "timings_are_normative": False,
        },
        "phases": phases,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if not math.isfinite(sum(phase["logits"]["logsumexp"] for phase in phases)):
        raise RuntimeError("non-finite logit summary")
    sys.stdout.write(serialized)
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
