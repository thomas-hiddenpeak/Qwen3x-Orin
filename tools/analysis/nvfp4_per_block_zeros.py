#!/usr/bin/env python3
"""Count zero-valued weights in canonical ModelOpt NVFP4 quantization blocks.

The script reads safetensors payloads through ``numpy.memmap`` and never
materializes a complete model tensor.  A quantization block is one output row
and 16 consecutive K values (eight packed bytes and one E4M3 block scale).

Two zero definitions are reported:

* ``e2m1_zero`` counts packed E2M1 codes 0x0 (+0) and 0x8 (-0).
* ``dequant_zero`` follows the repository's FP32 dequantization formula and
  therefore also catches a zero scale or FP32 underflow.

For the pinned NVIDIA Qwen3.6-27B-NVFP4 checkpoint these definitions are
expected to agree, but keeping both makes the analysis safe for other ModelOpt
NVFP4 checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment diagnostic
    raise SystemExit("nvfp4_per_block_zeros.py requires numpy") from error


GROUP_SIZE = 16
PACKED_BYTES_PER_BLOCK = GROUP_SIZE // 2
HISTOGRAM_BINS = GROUP_SIZE + 1
EXPECTED_PINNED_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
LAYER_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)$"
)
E2M1_VALUES = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    byte_size: int


@dataclasses.dataclass
class Stats:
    scope_type: str
    scope: str
    tensor_count: int
    blocks: int
    values: int
    e2m1_hist: np.ndarray
    dequant_hist: np.ndarray
    nibble_hist: np.ndarray
    scale_hist: np.ndarray

    def add(self, other: "Stats") -> None:
        self.tensor_count += other.tensor_count
        self.blocks += other.blocks
        self.values += other.values
        self.e2m1_hist += other.e2m1_hist
        self.dequant_hist += other.dequant_hist
        self.nibble_hist += other.nibble_hist
        self.scale_hist += other.scale_hist

    @property
    def e2m1_zeros(self) -> int:
        return _histogram_weighted_sum(self.e2m1_hist)

    @property
    def dequant_zeros(self) -> int:
        return _histogram_weighted_sum(self.dequant_hist)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or 8 + header_bytes > path.stat().st_size:
            raise ValueError(f"invalid safetensors header length in {path}")
        payload = stream.read(header_bytes)
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header: {path}") from error
    return 8 + header_bytes, header


def _build_tensor_specs(
    model_dir: Path, weight_map: dict[str, str]
) -> tuple[dict[str, TensorSpec], dict[str, int]]:
    specs: dict[str, TensorSpec] = {}
    shard_sizes: dict[str, int] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = model_dir / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"missing shard: {shard}")
        shard_sizes[shard_name] = shard.stat().st_size
        data_base, header = _read_safetensors_header(shard)
        for name, raw in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"invalid tensor metadata for {name}")
            start, end = (int(value) for value in raw["data_offsets"])
            if start < 0 or end < start or data_base + end > shard.stat().st_size:
                raise ValueError(f"invalid data offsets for {name}")
            if weight_map.get(name) != shard_name:
                raise ValueError(f"index/header shard mismatch for {name}")
            specs[name] = TensorSpec(
                name=name,
                shard=shard,
                dtype=str(raw["dtype"]),
                shape=tuple(int(value) for value in raw["shape"]),
                data_offset=data_base + start,
                byte_size=end - start,
            )
    missing = sorted(set(weight_map) - set(specs))
    extra = sorted(set(specs) - set(weight_map))
    if missing or extra:
        raise ValueError(
            f"index/header tensor mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    return specs, shard_sizes


def _read_f32_scalar(spec: TensorSpec) -> float:
    if spec.dtype != "F32" or spec.shape or spec.byte_size != 4:
        raise ValueError(f"expected scalar F32 tensor: {spec.name}")
    with spec.shard.open("rb") as stream:
        stream.seek(spec.data_offset)
        payload = stream.read(4)
    if len(payload) != 4:
        raise ValueError(f"truncated scalar tensor: {spec.name}")
    return struct.unpack("<f", payload)[0]


def _decode_e4m3fn_table() -> np.ndarray:
    result = np.empty(256, dtype=np.float32)
    for bits in range(256):
        magnitude = bits & 0x7F
        exponent = (magnitude >> 3) & 0x0F
        mantissa = magnitude & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            value = math.nan
        elif exponent == 0:
            value = math.ldexp(float(mantissa), -9)
        else:
            value = math.ldexp(1.0 + float(mantissa) / 8.0, exponent - 7)
        result[bits] = np.float32(-value if bits & 0x80 else value)
    return result


def _e2m1_pair_zero_lut() -> np.ndarray:
    result = np.empty(256, dtype=np.uint8)
    for packed in range(256):
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        result[packed] = int((low & 0x07) == 0) + int((high & 0x07) == 0)
    return result


def _dequant_pair_zero_lut(scale_2: float) -> np.ndarray:
    scales = _decode_e4m3fn_table()
    with np.errstate(invalid="ignore", under="ignore"):
        first_product = np.multiply(
            scales[:, None], E2M1_VALUES[None, :], dtype=np.float32
        )
        values = np.multiply(first_product, np.float32(scale_2), dtype=np.float32)
    code_is_zero = np.equal(values, np.float32(0.0)).astype(np.uint8)
    packed = np.arange(256, dtype=np.uint16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return code_is_zero[:, low] + code_is_zero[:, high]


E2M1_PAIR_ZERO_LUT = _e2m1_pair_zero_lut()


def _nibble_histogram(byte_hist: np.ndarray) -> np.ndarray:
    result = np.zeros(16, dtype=np.int64)
    for packed, count in enumerate(byte_hist.tolist()):
        result[packed & 0x0F] += count
        result[(packed >> 4) & 0x0F] += count
    return result


def _histogram_weighted_sum(histogram: np.ndarray) -> int:
    return sum(index * int(count) for index, count in enumerate(histogram))


def _projection_and_layer(target: str) -> tuple[str, str]:
    match = LAYER_PATTERN.fullmatch(target)
    if match:
        return match.group(2).removesuffix("_proj"), f"layer_{int(match.group(1)):02d}"
    if target == "lm_head":
        return "lm_head", "lm_head"
    raise ValueError(f"unexpected NVFP4 target name: {target}")


def _empty_stats(scope_type: str, scope: str) -> Stats:
    return Stats(
        scope_type=scope_type,
        scope=scope,
        tensor_count=0,
        blocks=0,
        values=0,
        e2m1_hist=np.zeros(HISTOGRAM_BINS, dtype=np.int64),
        dequant_hist=np.zeros(HISTOGRAM_BINS, dtype=np.int64),
        nibble_hist=np.zeros(16, dtype=np.int64),
        scale_hist=np.zeros(256, dtype=np.int64),
    )


def _scan_tensor(
    target: str,
    specs: dict[str, TensorSpec],
    chunk_blocks: int,
) -> tuple[Stats, dict[str, Any]]:
    weight = specs[f"{target}.weight"]
    scale = specs[f"{target}.weight_scale"]
    scale_2_spec = specs[f"{target}.weight_scale_2"]
    if weight.dtype != "U8" or len(weight.shape) != 2:
        raise ValueError(f"invalid NVFP4 packed weight: {weight.name}")
    if scale.dtype != "F8_E4M3" or len(scale.shape) != 2:
        raise ValueError(f"invalid NVFP4 block scale: {scale.name}")
    if weight.byte_size != _product(weight.shape):
        raise ValueError(f"unexpected U8 byte size: {weight.name}")
    if scale.byte_size != _product(scale.shape):
        raise ValueError(f"unexpected E4M3 byte size: {scale.name}")

    rows, packed_columns = weight.shape
    scale_rows, scale_columns = scale.shape
    logical_columns = packed_columns * 2
    if rows != scale_rows or logical_columns % GROUP_SIZE:
        raise ValueError(f"incompatible NVFP4 shapes for {target}")
    if scale_columns != logical_columns // GROUP_SIZE:
        raise ValueError(f"scale shape does not match group size for {target}")
    blocks = rows * scale_columns
    if weight.byte_size != blocks * PACKED_BYTES_PER_BLOCK:
        raise ValueError(f"weight payload is not block-contiguous for {target}")

    scale_2 = _read_f32_scalar(scale_2_spec)
    dequant_lut = _dequant_pair_zero_lut(scale_2)
    packed_view = np.memmap(
        weight.shard,
        dtype=np.uint8,
        mode="r",
        offset=weight.data_offset,
        shape=(weight.byte_size,),
    )
    scale_view = np.memmap(
        scale.shard,
        dtype=np.uint8,
        mode="r",
        offset=scale.data_offset,
        shape=(scale.byte_size,),
    )

    e2m1_hist = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    dequant_hist = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    byte_hist = np.zeros(256, dtype=np.int64)
    scale_hist = np.zeros(256, dtype=np.int64)
    for first in range(0, blocks, chunk_blocks):
        last = min(blocks, first + chunk_blocks)
        packed = np.asarray(
            packed_view[
                first * PACKED_BYTES_PER_BLOCK : last * PACKED_BYTES_PER_BLOCK
            ]
        ).reshape(-1, PACKED_BYTES_PER_BLOCK)
        scale_bits = np.asarray(scale_view[first:last])
        e2m1_counts = np.sum(
            E2M1_PAIR_ZERO_LUT[packed], axis=1, dtype=np.uint8
        )
        dequant_counts = np.sum(
            dequant_lut[scale_bits[:, None], packed], axis=1, dtype=np.uint8
        )
        e2m1_hist += np.bincount(e2m1_counts, minlength=HISTOGRAM_BINS)
        dequant_hist += np.bincount(dequant_counts, minlength=HISTOGRAM_BINS)
        byte_hist += np.bincount(packed.ravel(), minlength=256)
        scale_hist += np.bincount(scale_bits, minlength=256)

    del packed_view
    del scale_view
    nibble_hist = _nibble_histogram(byte_hist)
    projection, layer = _projection_and_layer(target)
    stats = Stats(
        scope_type="tensor",
        scope=target,
        tensor_count=1,
        blocks=blocks,
        values=rows * logical_columns,
        e2m1_hist=e2m1_hist,
        dequant_hist=dequant_hist,
        nibble_hist=nibble_hist,
        scale_hist=scale_hist,
    )
    if int(e2m1_hist.sum()) != blocks or int(dequant_hist.sum()) != blocks:
        raise AssertionError(f"block histogram mismatch for {target}")
    if int(nibble_hist.sum()) != stats.values:
        raise AssertionError(f"nibble histogram mismatch for {target}")
    if int(scale_hist.sum()) != blocks:
        raise AssertionError(f"scale histogram mismatch for {target}")
    if stats.e2m1_zeros != int(nibble_hist[0] + nibble_hist[8]):
        raise AssertionError(f"E2M1 zero totals disagree for {target}")

    detail = {
        "target": target,
        "projection": projection,
        "layer": layer,
        "weight_tensor": weight.name,
        "weight_shard": weight.shard.name,
        "packed_shape": list(weight.shape),
        "logical_shape": [rows, logical_columns],
        "scale_shape": list(scale.shape),
        "weight_scale_2": scale_2,
        "packed_byte_zero_count": int(byte_hist[0]),
    }
    return stats, detail


def _aggregate(
    records: Sequence[Stats], scope_type: str, scope: str
) -> Stats:
    result = _empty_stats(scope_type, scope)
    for record in records:
        result.add(record)
    return result


def _summary_row(stats: Stats) -> dict[str, Any]:
    dequant_zeros = stats.dequant_zeros
    e2m1_zeros = stats.e2m1_zeros
    zero_scale_blocks = int(stats.scale_hist[0] + stats.scale_hist[128])
    nan_scale_blocks = int(stats.scale_hist[127] + stats.scale_hist[255])
    negative_scale_blocks = int(stats.scale_hist[128:].sum())
    return {
        "scope_type": stats.scope_type,
        "scope": stats.scope,
        "tensor_count": stats.tensor_count,
        "blocks": stats.blocks,
        "logical_values": stats.values,
        "e2m1_zero_values": e2m1_zeros,
        "dequant_zero_values": dequant_zeros,
        "dequant_zero_pct": 100.0 * dequant_zeros / stats.values,
        "positive_zero_codes": int(stats.nibble_hist[0]),
        "negative_zero_codes": int(stats.nibble_hist[8]),
        "scale_induced_zero_values": dequant_zeros - e2m1_zeros,
        "zero_free_blocks": int(stats.dequant_hist[0]),
        "blocks_with_any_zero": stats.blocks - int(stats.dequant_hist[0]),
        "blocks_with_any_zero_pct": 100.0
        * (stats.blocks - int(stats.dequant_hist[0]))
        / stats.blocks,
        "all_zero_blocks": int(stats.dequant_hist[GROUP_SIZE]),
        "mean_zeros_per_block": dequant_zeros / stats.blocks,
        "zero_scale_blocks": zero_scale_blocks,
        "negative_scale_blocks": negative_scale_blocks,
        "nan_scale_blocks": nan_scale_blocks,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _histogram_rows(records: Sequence[Stats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for zero_count in range(HISTOGRAM_BINS):
            count = int(record.dequant_hist[zero_count])
            rows.append(
                {
                    "scope_type": record.scope_type,
                    "scope": record.scope,
                    "zero_count": zero_count,
                    "block_count": count,
                    "block_pct": 100.0 * count / record.blocks,
                    "e2m1_block_count": int(record.e2m1_hist[zero_count]),
                }
            )
    return rows


def _code_rows(records: Sequence[Stats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = E2M1_VALUES.tolist()
    for record in records:
        for code, count in enumerate(record.nibble_hist.tolist()):
            rows.append(
                {
                    "scope_type": record.scope_type,
                    "scope": record.scope,
                    "e2m1_code": f"0x{code:x}",
                    "decoded_value": float(values[code]),
                    "value_count": int(count),
                    "value_pct": 100.0 * int(count) / record.values,
                }
            )
    return rows


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _report_markdown(
    overall: Stats,
    projections: Sequence[Stats],
    layers: Sequence[Stats],
    source: dict[str, Any],
    scale_validation: dict[str, Any],
    elapsed_seconds: float,
) -> str:
    summary = _summary_row(overall)
    layer_rows = [_summary_row(layer) for layer in layers if layer.scope != "lm_head"]
    layer_rows.sort(key=lambda row: row["dequant_zero_pct"], reverse=True)
    scales_are_canonical = (
        scale_validation["positive_finite_block_scales"] == overall.blocks
        and scale_validation["positive_finite_tensor_scales"] == overall.tensor_count
        and summary["scale_induced_zero_values"] == 0
    )
    lines = [
        "# Qwen3.6-27B-NVFP4 per-block zero statistics",
        "",
        f"- Source: `{source['model_dir']}`",
        f"- Revision: `{source['revision']}`",
        f"- Quantization block: one output row × 16 consecutive K values",
        f"- NVFP4 tensors: {_format_integer(overall.tensor_count)}",
        f"- Blocks: {_format_integer(overall.blocks)}",
        f"- Logical NVFP4 values: {_format_integer(overall.values)}",
        f"- Scan time: {elapsed_seconds:.2f} seconds",
        "",
        "## Headline",
        "",
        f"- Strict dequantized zeros: {_format_integer(overall.dequant_zeros)} "
        f"({summary['dequant_zero_pct']:.10f}%)",
        f"- E2M1 +0 codes: {_format_integer(int(overall.nibble_hist[0]))}",
        f"- E2M1 -0 codes: {_format_integer(int(overall.nibble_hist[8]))}",
        f"- Blocks containing at least one zero: "
        f"{_format_integer(summary['blocks_with_any_zero'])} "
        f"({summary['blocks_with_any_zero_pct']:.10f}%)",
        f"- Fully zero blocks: {_format_integer(summary['all_zero_blocks'])}",
        f"- Positive, finite block scales: "
        f"{_format_integer(scale_validation['positive_finite_block_scales'])} / "
        f"{_format_integer(overall.blocks)}",
        "",
        (
            "All block scales and tensor-level scales are positive and finite, so "
            "strict dequantized zero equals E2M1 code 0x0/0x8 for this checkpoint."
            if scales_are_canonical
            else "Scale validation found non-canonical values; use the strict dequantized "
            "zero columns rather than assuming E2M1 code zeros are equivalent."
        ),
        "",
        "## By projection",
        "",
        "| Projection | Blocks | Zero values | Zero rate | Any-zero blocks | Fully zero |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for projection in projections:
        row = _summary_row(projection)
        lines.append(
            f"| {projection.scope} | {_format_integer(projection.blocks)} | "
            f"{_format_integer(row['dequant_zero_values'])} | "
            f"{row['dequant_zero_pct']:.8f}% | "
            f"{row['blocks_with_any_zero_pct']:.8f}% | "
            f"{_format_integer(row['all_zero_blocks'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-block zero-count histogram",
            "",
            "| Zeros in 16-value block | Blocks | Share |",
            "|---:|---:|---:|",
        ]
    )
    for zero_count, count in enumerate(overall.dequant_hist.tolist()):
        lines.append(
            f"| {zero_count} | {_format_integer(int(count))} | "
            f"{100.0 * int(count) / overall.blocks:.10f}% |"
        )
    lines.extend(
        [
            "",
            "## Transformer-layer range",
            "",
            "| Rank | Layer | Zero rate | Any-zero blocks |",
            "|---:|---|---:|---:|",
        ]
    )
    extremes = layer_rows[:5] + list(reversed(layer_rows[-5:]))
    for rank, row in enumerate(extremes, start=1):
        label = "high" if rank <= 5 else "low"
        lines.append(
            f"| {label} {rank if rank <= 5 else rank - 5} | {row['scope']} | "
            f"{row['dequant_zero_pct']:.8f}% | "
            f"{row['blocks_with_any_zero_pct']:.8f}% |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: overall, projection, layer, and tensor summaries.",
            "- `block_zero_histogram.csv`: zero-count histograms at all scopes.",
            "- `e2m1_code_histogram.csv`: packed E2M1 code distributions.",
            "- `stats.json`: full machine-readable result and provenance.",
            "",
            "The raw 1,149,009,920-row block table is intentionally not emitted; "
            "a uint8 zero count per block would still be about 1.15 GB.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_stats(record: Stats, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    result = _summary_row(record)
    result["e2m1_block_histogram_0_to_16"] = record.e2m1_hist.tolist()
    result["dequant_block_histogram_0_to_16"] = record.dequant_hist.tolist()
    result["e2m1_code_histogram_0_to_15"] = record.nibble_hist.tolist()
    result["weight_scale_code_histogram_0_to_255"] = record.scale_hist.tolist()
    if detail is not None:
        result.update(detail)
    return result


def _scale_validation(
    overall: Stats, tensor_details: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scale_2_values = [float(detail["weight_scale_2"]) for detail in tensor_details.values()]
    finite_scale_2 = [value for value in scale_2_values if math.isfinite(value)]
    positive_finite_scale_2 = [value for value in finite_scale_2 if value > 0.0]
    positive_finite_block_scales = int(overall.scale_hist[1:127].sum())
    observed_scale_codes = np.flatnonzero(overall.scale_hist)
    return {
        "positive_finite_block_scales": positive_finite_block_scales,
        "zero_block_scales": int(overall.scale_hist[0] + overall.scale_hist[128]),
        "negative_block_scales": int(overall.scale_hist[128:].sum()),
        "nan_block_scales": int(overall.scale_hist[127] + overall.scale_hist[255]),
        "observed_block_scale_code_min": int(observed_scale_codes.min()),
        "observed_block_scale_code_max": int(observed_scale_codes.max()),
        "positive_finite_tensor_scales": len(positive_finite_scale_2),
        "zero_tensor_scales": sum(value == 0.0 for value in finite_scale_2),
        "negative_tensor_scales": sum(value < 0.0 for value in finite_scale_2),
        "nonfinite_tensor_scales": len(scale_2_values) - len(finite_scale_2),
        "tensor_scale_min": min(finite_scale_2),
        "tensor_scale_max": max(finite_scale_2),
    }


def _validate_global(
    overall: Stats, targets: Sequence[str], quantized_layers: dict[str, Any]
) -> None:
    if overall.tensor_count != len(targets):
        raise AssertionError("global tensor count does not match NVFP4 target count")
    if overall.values != overall.blocks * GROUP_SIZE:
        raise AssertionError("global logical values do not match block geometry")
    if int(overall.dequant_hist.sum()) != overall.blocks:
        raise AssertionError("global dequant histogram does not sum to blocks")
    if int(overall.e2m1_hist.sum()) != overall.blocks:
        raise AssertionError("global E2M1 histogram does not sum to blocks")
    for target in targets:
        if int(quantized_layers[target].get("group_size", -1)) != GROUP_SIZE:
            raise AssertionError(f"unexpected group size for {target}")


def run(args: argparse.Namespace) -> dict[str, Path]:
    started = time.monotonic()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = [
        model_dir / "model.safetensors.index.json",
        model_dir / "hf_quant_config.json",
        model_dir / "config.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    index = _load_json(required[0])
    quant_config = _load_json(required[1])
    weight_map = index["weight_map"]
    quantized_layers = quant_config["quantization"]["quantized_layers"]
    targets = sorted(
        name
        for name, config in quantized_layers.items()
        if config.get("quant_algo") == "W4A16_NVFP4"
    )
    if not targets:
        raise ValueError("hf_quant_config.json contains no W4A16_NVFP4 targets")
    specs, shard_sizes = _build_tensor_specs(model_dir, weight_map)

    tensor_records: list[Stats] = []
    tensor_details: dict[str, dict[str, Any]] = {}
    for index_value, target in enumerate(targets, start=1):
        record, detail = _scan_tensor(target, specs, args.chunk_blocks)
        tensor_records.append(record)
        tensor_details[target] = detail
        if not args.quiet:
            row = _summary_row(record)
            print(
                f"[{index_value:3d}/{len(targets)}] {target}: "
                f"{row['dequant_zero_pct']:.6f}% zeros",
                flush=True,
            )

    by_projection: dict[str, list[Stats]] = defaultdict(list)
    by_layer: dict[str, list[Stats]] = defaultdict(list)
    for record in tensor_records:
        detail = tensor_details[record.scope]
        by_projection[detail["projection"]].append(record)
        by_layer[detail["layer"]].append(record)
    projection_order = ["gate", "up", "down", "lm_head"]
    projection_records = [
        _aggregate(by_projection[name], "projection", name)
        for name in projection_order
        if name in by_projection
    ]
    layer_records = [
        _aggregate(by_layer[name], "layer", name)
        for name in sorted(
            by_layer,
            key=lambda value: (value == "lm_head", value),
        )
    ]
    overall = _aggregate(tensor_records, "overall", "all_nvfp4")
    _validate_global(overall, targets, quantized_layers)
    scale_validation = _scale_validation(overall, tensor_details)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [overall, *projection_records, *layer_records, *tensor_records]
    summary_path = output_dir / "summary.csv"
    histogram_path = output_dir / "block_zero_histogram.csv"
    code_path = output_dir / "e2m1_code_histogram.csv"
    json_path = output_dir / "stats.json"
    report_path = output_dir / "README.md"
    _write_csv(summary_path, [_summary_row(record) for record in all_records])
    _write_csv(histogram_path, _histogram_rows(all_records))
    _write_csv(code_path, _code_rows(all_records))

    elapsed_seconds = time.monotonic() - started
    source = {
        "model_dir": str(model_dir),
        "repository": "nvidia/Qwen3.6-27B-NVFP4",
        "revision": args.revision,
        "index_metadata": index.get("metadata", {}),
        "shard_sizes": shard_sizes,
        "config_sha256": _sha256(required[2]),
        "quant_config_sha256": _sha256(required[1]),
        "index_sha256": _sha256(required[0]),
        "scanned_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "chunk_blocks": args.chunk_blocks,
    }
    result = {
        "schema_version": 1,
        "definitions": {
            "quant_block": "one output row x 16 consecutive K values",
            "e2m1_zero": "packed nibble code 0x0 or 0x8",
            "dequant_zero": (
                "float32(E2M1(code) * E4M3FN(weight_scale) * weight_scale_2) "
                "compares equal to 0.0"
            ),
        },
        "source": source,
        "scale_validation": scale_validation,
        "overall": _json_stats(overall),
        "projections": [_json_stats(record) for record in projection_records],
        "layers": [_json_stats(record) for record in layer_records],
        "tensors": [
            _json_stats(record, tensor_details[record.scope])
            for record in tensor_records
        ],
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    report_path.write_text(
        _report_markdown(
            overall,
            projection_records,
            layer_records,
            source,
            scale_validation,
            elapsed_seconds,
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "histogram": histogram_path,
        "codes": code_path,
        "json": json_path,
        "report": report_path,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvfp4-zero-stats"),
        help="directory for CSV, JSON, and Markdown outputs",
    )
    parser.add_argument(
        "--chunk-blocks",
        type=_positive_int,
        default=1 << 20,
        help="quantization blocks processed per NumPy chunk (default: 1048576)",
    )
    parser.add_argument(
        "--revision",
        default=EXPECTED_PINNED_REVISION,
        help="checkpoint revision recorded as provenance",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-tensor progress")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run(args)
    if not args.quiet:
        print("wrote:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
