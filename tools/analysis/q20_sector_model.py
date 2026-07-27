#!/usr/bin/env python3
"""Exact 32-byte transaction model for the test-only LM-head q20 screen.

This module is production-unreachable and never imports CUDA.  It separates
the expensive checkpoint/bound construction from the cheap structural test:
``simulate`` consumes conservative per-row checkpoint upper bounds and
incumbent lower bounds from an NPZ file, then applies the production LM-head
row ownership and counts the exact request sectors of the proposed layout.

The address contract mirrors
``nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel``:

* grid 64 x block 256, eight warps per CTA;
* one warp owns four consecutive rows and advances by 2,048 rows;
* each K256 phase reads 128 aligned packed-weight bytes per active row;
* the retained dual-phase scale load reads one aligned 32-byte sector per
  active row at the beginning of each K512 pair;
* the 19 upward-FP32 bounds are schedule-packed per 32-row CTA group as
  ``[group][checkpoint][local_row]``.  One cooperative preload is exactly
  2,432 bytes or 76 sectors, with no padding or cache-residency credit.

The model reports L1 request sectors.  Weight, scale, and sidecar regions are
separate 32-byte-aligned allocations, and no cross-request sector merging is
credited.  Common activation/output traffic cancels and is intentionally
excluded.  This is a structural traffic proof, not a latency claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "q20_sector_model.py requires numpy; use /home/rm01/setup/.venv/bin/python"
    ) from error


SECTOR_BYTES = 32
DEFAULT_GATE_BYTES = 55_000_000
FORMAL_EXPECTED_FIXTURES = 25
PROOF_SCHEMA_VERSION = 2
PROOF_CONTRACT_VERSION = "q20-directed-bf16-rne-rho-v1"
PRODUCTION_KERNEL_RELATIVE_PATH = Path("src/kernels/sm87/weight_only_gemv.cu")
PINNED_MODEL_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
PINNED_SHARD_NAME = "model-00003-of-00003.safetensors"
PINNED_SHARD_BYTES = 1_970_287_640
PINNED_HEADER_BYTES = 7_096
PINNED_FIXTURE_BYTES = 450_560
PINNED_FIXTURE_SHA256 = (
    "fe151ccec6e38e95c41975823eb66e08f5e5dc8fc51a16ea444c1ab5f3b10017"
)
GAMMA_48 = 0.0000028610311346944847
GAMMA_48_UPPER = math.nextafter(GAMMA_48, math.inf)
FP64_PROOF_MAX_PATH_OPS = 1_024
FP64_PROOF_RHO = 2.0 ** -40
FP64_GAMMA_PATH = (
    FP64_PROOF_MAX_PATH_OPS * (2.0 ** -53)
    / (1.0 - FP64_PROOF_MAX_PATH_OPS * (2.0 ** -53))
)
if not FP64_PROOF_RHO > FP64_GAMMA_PATH:  # pragma: no cover - constant contract
    raise AssertionError("FP64 proof envelope must exceed gamma_1024")


@dataclasses.dataclass(frozen=True)
class PinnedTensor:
    name: str
    absolute_offset: int
    size: int
    sha256: str | None
    dtype: str
    shape: tuple[int, ...]


PINNED_WEIGHT = PinnedTensor(
    "lm_head.weight",
    1_067_201_560,
    635_699_200,
    "746c1d13e9cf69bfca6f5901a7dec5a7f2b252359696644a1ee55953b9680205",
    "U8",
    (248_320, 2_560),
)
PINNED_SCALES = PinnedTensor(
    "lm_head.weight_scale",
    849_458_200,
    79_462_400,
    "e20faadf62bd2b3bf88f2fc9fbf4f42462fdfdd0f4bc9f23d7eaabcc1b697f9b",
    "F8_E4M3",
    (248_320, 320),
)
PINNED_SCALE2 = PinnedTensor(
    "lm_head.weight_scale_2", 7_108, 4, None, "F32", ()
)
PINNED_SCALE2_BITS = 0x390DB6DC


@dataclasses.dataclass(frozen=True)
class Topology:
    """Production LM-head ownership and payload geometry."""

    rows: int = 248_320
    columns: int = 5_120
    grid_blocks: int = 64
    warps_per_block: int = 8
    rows_per_warp: int = 4
    columns_per_checkpoint: int = 256
    nvfp4_values_per_byte: int = 2
    nvfp4_group_size: int = 16

    def validate(self) -> None:
        positive = dataclasses.asdict(self)
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"topology values must be positive: {positive}")
        if self.rows % self.rows_per_warp != 0:
            raise ValueError("rows must be divisible by rows_per_warp")
        if self.columns % self.columns_per_checkpoint != 0:
            raise ValueError("columns must be divisible by checkpoint width")
        if self.columns_per_checkpoint % self.nvfp4_values_per_byte != 0:
            raise ValueError("checkpoint width must pack to whole bytes")
        if self.columns_per_checkpoint % self.nvfp4_group_size != 0:
            raise ValueError("checkpoint width must contain whole scale groups")
        if self.checkpoints % 2 != 0:
            raise ValueError("dual-scale sector model requires an even checkpoint count")
        if self.rows % self.rows_per_cta_group:
            raise ValueError("rows must contain whole CTA sidecar groups")
        if self.weight_bytes_per_row_checkpoint % SECTOR_BYTES:
            raise ValueError("each K256 packed-weight span must be sector aligned")
        if 2 * self.scale_useful_bytes_per_row_checkpoint != SECTOR_BYTES:
            raise ValueError("each K512 dual-scale span must be exactly one sector")
        if self.rows_per_cta_group * self.bound_values_per_row * 4 % SECTOR_BYTES:
            raise ValueError("CTA sidecar group must be an exact sector multiple")

    @property
    def checkpoints(self) -> int:
        return self.columns // self.columns_per_checkpoint

    @property
    def bound_values_per_row(self) -> int:
        return self.checkpoints - 1

    @property
    def row_stride(self) -> int:
        return self.grid_blocks * self.warps_per_block * self.rows_per_warp

    @property
    def rows_per_cta_group(self) -> int:
        return self.warps_per_block * self.rows_per_warp

    @property
    def wave_count(self) -> int:
        return math.ceil(self.rows / self.row_stride)

    @property
    def packed_bytes_per_row(self) -> int:
        return self.columns // self.nvfp4_values_per_byte

    @property
    def scale_bytes_per_row(self) -> int:
        return self.columns // self.nvfp4_group_size

    @property
    def weight_bytes_per_row_checkpoint(self) -> int:
        return self.columns_per_checkpoint // self.nvfp4_values_per_byte

    @property
    def scale_useful_bytes_per_row_checkpoint(self) -> int:
        return self.columns_per_checkpoint // self.nvfp4_group_size

    @property
    def sidecar_bytes(self) -> int:
        return self.rows * self.bound_values_per_row * 4

    @property
    def sidecar_sectors_per_cta_group(self) -> int:
        return (
            self.rows_per_cta_group * self.bound_values_per_row * 4
        ) // SECTOR_BYTES

    @property
    def cta_group_count(self) -> int:
        return math.ceil(self.rows / self.rows_per_cta_group)

    @property
    def baseline_mandatory_bytes(self) -> int:
        return self.rows * (self.packed_bytes_per_row + self.scale_bytes_per_row)


PRODUCTION_TOPOLOGY = Topology()


def sectors_touched(address: int, size: int) -> tuple[int, ...]:
    """Return every 32-byte sector touched by one memory request."""

    if address < 0 or size <= 0:
        raise ValueError("address must be non-negative and size must be positive")
    first = address // SECTOR_BYTES
    last = (address + size - 1) // SECTOR_BYTES
    return tuple(range(first, last + 1))


def weight_address(topology: Topology, row: int, checkpoint: int) -> int:
    """Byte address relative to the aligned canonical packed-weight base."""

    _validate_row_checkpoint(topology, row, checkpoint)
    return (
        row * topology.packed_bytes_per_row
        + checkpoint * topology.weight_bytes_per_row_checkpoint
    )


def scale_pair_address(topology: Topology, row: int, checkpoint_pair: int) -> int:
    """Byte address of the retained aligned K512 dual-scale request."""

    if row < 0 or row >= topology.rows:
        raise IndexError(row)
    pair_count = topology.checkpoints // 2
    if checkpoint_pair < 0 or checkpoint_pair >= pair_count:
        raise IndexError(checkpoint_pair)
    return row * topology.scale_bytes_per_row + checkpoint_pair * SECTOR_BYTES


def sidecar_group_address(topology: Topology, group: int) -> int:
    """Byte address relative to the schedule-packed sidecar allocation."""

    if group < 0 or group >= topology.cta_group_count:
        raise IndexError(group)
    bytes_per_group = topology.sidecar_sectors_per_cta_group * SECTOR_BYTES
    return group * bytes_per_group


def _validate_row_checkpoint(
    topology: Topology, row: int, checkpoint: int
) -> None:
    if row < 0 or row >= topology.rows:
        raise IndexError(row)
    if checkpoint < 0 or checkpoint >= topology.checkpoints:
        raise IndexError(checkpoint)


def verify_address_contract(topology: Topology) -> dict[str, int]:
    """Prove the closed-form sector costs from concrete endpoint addresses."""

    topology.validate()
    sample_rows = sorted({0, min(1, topology.rows - 1), topology.rows - 1})
    weight_sectors = topology.weight_bytes_per_row_checkpoint // SECTOR_BYTES
    for row in sample_rows:
        for checkpoint in range(topology.checkpoints):
            address = weight_address(topology, row, checkpoint)
            if address % SECTOR_BYTES:
                raise AssertionError(f"unaligned weight address {address}")
            if len(sectors_touched(address, topology.weight_bytes_per_row_checkpoint)) != weight_sectors:
                raise AssertionError("weight sector formula disagrees with address span")
        for pair in range(topology.checkpoints // 2):
            address = scale_pair_address(topology, row, pair)
            if address % SECTOR_BYTES or len(sectors_touched(address, SECTOR_BYTES)) != 1:
                raise AssertionError("scale-pair address is not exactly one sector")
    bytes_per_group = topology.sidecar_sectors_per_cta_group * SECTOR_BYTES
    for group in sorted({0, max(0, topology.cta_group_count - 1)}):
        address = sidecar_group_address(topology, group)
        if address % SECTOR_BYTES:
            raise AssertionError("sidecar group is not sector aligned")
        if len(sectors_touched(address, bytes_per_group)) != topology.sidecar_sectors_per_cta_group:
            raise AssertionError("sidecar sector formula disagrees with address span")
    return {
        "weight_sectors_per_active_row_checkpoint": weight_sectors,
        "scale_sectors_per_active_row_k512_pair": 1,
        "sidecar_sectors_per_cta_group": topology.sidecar_sectors_per_cta_group,
    }


def _sha256_path(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_slice(
    path: Path, offset: int, size: int, chunk_bytes: int = 8 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ValueError(f"short read hashing {path} at {offset}+{size}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        raw_header = stream.read(header_bytes)
        if len(raw_header) != header_bytes:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid safetensors header") from error
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header_bytes, header


def _validate_pinned_tensor(
    header: dict[str, Any], data_base: int, expected: PinnedTensor
) -> dict[str, Any]:
    entry = header.get(expected.name)
    if not isinstance(entry, dict):
        raise ValueError(f"pinned tensor {expected.name!r} is absent")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
    ):
        raise ValueError(f"{expected.name}: invalid data_offsets")
    begin, end = offsets
    absolute_offset = data_base + begin
    size = end - begin
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    if dtype != expected.dtype or shape != list(expected.shape):
        raise ValueError(
            f"{expected.name}: dtype/shape {dtype!r}/{shape!r} does not match "
            f"pinned {expected.dtype!r}/{list(expected.shape)!r}"
        )
    if absolute_offset != expected.absolute_offset or size != expected.size:
        raise ValueError(
            f"{expected.name}: span {absolute_offset}+{size} does not match pinned "
            f"{expected.absolute_offset}+{expected.size}"
        )
    return {
        "name": expected.name,
        "dtype": dtype,
        "shape": shape,
        "absolute_offset": absolute_offset,
        "bytes": size,
    }


def validate_pinned_checkpoint(
    shard: Path, verify_hashes: bool
) -> dict[str, Any]:
    if not shard.is_file():
        raise ValueError(f"checkpoint shard does not exist: {shard}")
    if shard.stat().st_size != PINNED_SHARD_BYTES:
        raise ValueError(
            f"checkpoint bytes {shard.stat().st_size} != {PINNED_SHARD_BYTES}"
        )
    header_bytes, header = _read_safetensors_header(shard)
    if header_bytes != PINNED_HEADER_BYTES:
        raise ValueError(f"header bytes {header_bytes} != {PINNED_HEADER_BYTES}")
    data_base = 8 + header_bytes
    tensors = [
        _validate_pinned_tensor(header, data_base, expected)
        for expected in (PINNED_WEIGHT, PINNED_SCALES, PINNED_SCALE2)
    ]
    with shard.open("rb") as stream:
        stream.seek(PINNED_SCALE2.absolute_offset)
        scale2_raw = stream.read(4)
    if len(scale2_raw) != 4:
        raise ValueError("short lm_head.weight_scale_2 read")
    scale2_bits = struct.unpack("<I", scale2_raw)[0]
    if scale2_bits != PINNED_SCALE2_BITS:
        raise ValueError(
            f"weight_scale_2 bits 0x{scale2_bits:08x} != 0x{PINNED_SCALE2_BITS:08x}"
        )
    hashes: dict[str, str] = {}
    if verify_hashes:
        for expected in (PINNED_WEIGHT, PINNED_SCALES):
            actual = _sha256_slice(
                shard, expected.absolute_offset, expected.size
            )
            if actual != expected.sha256:
                raise ValueError(
                    f"{expected.name} sha256 {actual} != pinned {expected.sha256}"
                )
            hashes[expected.name] = actual
    return {
        "model_revision": PINNED_MODEL_REVISION,
        "shard": str(shard),
        "shard_bytes": shard.stat().st_size,
        "header_bytes": header_bytes,
        "data_base": data_base,
        "tensors": tensors,
        "weight_scale_2_bits": f"0x{scale2_bits:08x}",
        "payload_hashes_verified": verify_hashes,
        "payload_hashes": hashes,
    }


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    words = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32)


def load_pinned_finalnorm(
    path: Path,
    first_position: int,
    last_position: int,
    columns: int,
    verify_hash: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"finalnorm fixture does not exist: {path}")
    if path.stat().st_size != PINNED_FIXTURE_BYTES:
        raise ValueError(
            f"finalnorm bytes {path.stat().st_size} != {PINNED_FIXTURE_BYTES}"
        )
    if first_position < 0 or last_position < first_position:
        raise ValueError("invalid finalnorm position range")
    words = np.memmap(path, mode="r", dtype="<u2")
    if words.size % columns:
        raise ValueError("finalnorm fixture is not a whole number of vectors")
    vector_count = words.size // columns
    if last_position >= vector_count:
        raise ValueError(
            f"position {last_position} exceeds fixture vectors {vector_count}"
        )
    actual_hash = _sha256_path(path) if verify_hash else None
    if verify_hash and actual_hash != PINNED_FIXTURE_SHA256:
        raise ValueError(
            f"finalnorm sha256 {actual_hash} != pinned {PINNED_FIXTURE_SHA256}"
        )
    selected_bits = np.asarray(
        words.reshape(vector_count, columns)[first_position : last_position + 1]
    )
    activations = _bf16_bits_to_float32(selected_bits)
    labels = [f"P{position}" for position in range(first_position, last_position + 1)]
    return activations, labels, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_hash,
        "hash_verified": verify_hash,
        "vector_count": vector_count,
        "first_position": first_position,
        "last_position": last_position,
        "selected_count": len(labels),
    }


def _decode_e2m1_table() -> np.ndarray:
    table = np.empty(16, dtype=np.float64)
    for code in range(16):
        sign = -1.0 if code & 0x8 else 1.0
        magnitude = code & 0x7
        values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        table[code] = sign * values[magnitude]
    return table


def _decode_e4m3fn_table() -> np.ndarray:
    table = np.empty(256, dtype=np.float64)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        magnitude = code & 0x7F
        exponent = magnitude >> 3
        mantissa = magnitude & 0x7
        if magnitude == 0x7F:
            table[code] = math.nan
        elif exponent == 0:
            table[code] = sign * math.ldexp(float(mantissa), -9)
        else:
            table[code] = sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return table


E2M1_TABLE = _decode_e2m1_table()
E4M3FN_TABLE = _decode_e4m3fn_table()


def _dyadic_significand_bits(value: Fraction) -> int:
    numerator = abs(value.numerator)
    if numerator == 0:
        return 0
    while numerator & 1 == 0:
        numerator >>= 1
    return numerator.bit_length()


def verify_exact_decode_products(scale2: float) -> dict[str, int]:
    """Exhaustively prove exact FP32 decode and FP64 scaled coefficients."""

    scale2_fraction = Fraction.from_float(float(scale2))
    combinations = 0
    maximum_scaled_significand_bits = 0
    for weight in E2M1_TABLE:
        for block_scale in E4M3FN_TABLE:
            if not math.isfinite(float(block_scale)):
                continue
            exact_unscaled = Fraction.from_float(float(weight)) * Fraction.from_float(
                float(block_scale)
            )
            production_unscaled = float(
                np.float32(np.float32(weight) * np.float32(block_scale))
            )
            if Fraction.from_float(production_unscaled) != exact_unscaled:
                raise AssertionError("E2M1*E4M3 is not exact in production FP32")
            exact_scaled = exact_unscaled * scale2_fraction
            offline_scaled = float(weight) * float(block_scale) * float(scale2)
            if Fraction.from_float(offline_scaled) != exact_scaled:
                raise AssertionError("scaled decode coefficient is not exact in FP64")
            maximum_scaled_significand_bits = max(
                maximum_scaled_significand_bits,
                _dyadic_significand_bits(exact_scaled),
            )
            combinations += 1
    # A BF16 operand has at most eight significand bits, so every coefficient
    # times activation product remains exact in binary64 before reduction.
    maximum_term_significand_bits = maximum_scaled_significand_bits + 8
    if maximum_term_significand_bits > 53:
        raise AssertionError("decoded coefficient times BF16 exceeds FP64 precision")
    return {
        "finite_code_combinations": combinations,
        "maximum_scaled_coefficient_significand_bits": (
            maximum_scaled_significand_bits
        ),
        "maximum_coefficient_times_bf16_significand_bits": (
            maximum_term_significand_bits
        ),
    }


def _ceil_float32(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    rounded = source.astype(np.float32)
    finite = np.isfinite(source)
    needs_increment = finite & (rounded.astype(np.float64) < source)
    if np.any(needs_increment):
        rounded[needs_increment] = np.nextafter(
            rounded[needs_increment], np.float32(math.inf), dtype=np.float32
        )
    return rounded


def _floor_float32(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    rounded = source.astype(np.float32)
    finite = np.isfinite(source)
    needs_decrement = finite & (rounded.astype(np.float64) > source)
    if np.any(needs_decrement):
        rounded[needs_decrement] = np.nextafter(
            rounded[needs_decrement], np.float32(-math.inf), dtype=np.float32
        )
    return rounded


def _add_up_float64(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left, dtype=np.float64) + np.asarray(
            right, dtype=np.float64
        )
    return np.nextafter(result, np.float64(math.inf))


def _sub_down_float64(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left, dtype=np.float64) - np.asarray(
            right, dtype=np.float64
        )
    return np.nextafter(result, np.float64(-math.inf))


def _mul_up_nonnegative_float64(
    left: np.ndarray | float, right: np.ndarray | float
) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    if np.any(left64 < 0.0) or np.any(right64 < 0.0):
        raise ValueError("upward nonnegative multiply received a negative operand")
    with np.errstate(over="ignore", invalid="ignore"):
        result = left64 * right64
    return np.nextafter(result, np.float64(math.inf))


def _positive_reduction_upper(reduced: np.ndarray) -> np.ndarray:
    """Upper-bound a nonnegative FP64 reduction using rho > gamma_1024.

    Every NumPy/BLAS reduction used by this builder has at most 1,024 rounded
    binary64 operations on any input-to-output path.  With
    ``rho = 2^-40 > gamma_1024``, a computed nonnegative result ``hat`` obeys
    ``exact <= hat / (1-rho)``.  The division and return are rounded upward.
    """

    reduced64 = np.asarray(reduced, dtype=np.float64)
    if np.any(reduced64 < 0.0):
        raise ValueError("nonnegative reduction produced a negative value")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        upper = reduced64 / np.float64(1.0 - FP64_PROOF_RHO)
    return np.nextafter(upper, np.float64(math.inf))


def _signed_reduction_error(sum_abs_upper: np.ndarray) -> np.ndarray:
    """Return upward ``rho*S`` for a signed FP64 reduction error envelope."""

    return _mul_up_nonnegative_float64(FP64_PROOF_RHO, sum_abs_upper)


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    floats = np.asarray(values, dtype=np.float32)
    bits = floats.view(np.uint32)
    magnitude = bits & np.uint32(0x7FFFFFFF)
    nan_mask = magnitude > np.uint32(0x7F800000)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    result = (rounded >> 16).astype(np.uint16)
    if np.any(nan_mask):
        result[nan_mask] = ((bits[nan_mask] >> 16) | np.uint32(0x40)).astype(
            np.uint16
        )
    return result


def _upper_to_bf16_rne_image(values: np.ndarray) -> np.ndarray:
    """Map a real upper bound through the production BF16-RNE boundary.

    The returned numeric value may be below the real bound.  That is safe: the
    comparison target is the production BF16-RNE output, and round-to-nearest-
    even is monotone.  Applying BF16-RNE to an FP32 ceiling therefore preserves
    the upper-bound order without spending an extra BF16 ULP.
    """

    return _bf16_bits_to_float32(_float32_to_bf16_bits(_ceil_float32(values)))


def _lower_to_bf16_rne_image(values: np.ndarray) -> np.ndarray:
    """Map a real lower bound through monotone production BF16-RNE."""

    return _bf16_bits_to_float32(_float32_to_bf16_bits(_floor_float32(values)))


def _decode_nvfp4_block(
    packed: np.ndarray,
    raw_scales: np.ndarray,
    checkpoint: int,
    topology: Topology,
    scale2: float,
) -> np.ndarray:
    packed_begin = checkpoint * topology.weight_bytes_per_row_checkpoint
    packed_end = packed_begin + topology.weight_bytes_per_row_checkpoint
    codes = np.asarray(packed[:, packed_begin:packed_end], dtype=np.uint8)
    values = np.empty((codes.shape[0], topology.columns_per_checkpoint), dtype=np.float64)
    values[:, 0::2] = E2M1_TABLE[codes & np.uint8(0x0F)]
    values[:, 1::2] = E2M1_TABLE[codes >> np.uint8(4)]
    scales_per_checkpoint = (
        topology.columns_per_checkpoint // topology.nvfp4_group_size
    )
    scale_begin = checkpoint * scales_per_checkpoint
    scale_end = scale_begin + scales_per_checkpoint
    decoded_scales = E4M3FN_TABLE[
        np.asarray(raw_scales[:, scale_begin:scale_end], dtype=np.uint8)
    ]
    values *= np.repeat(decoded_scales, topology.nvfp4_group_size, axis=1)
    values *= scale2
    return values


def _directed_suffix_norms(block_squared_norms: np.ndarray) -> np.ndarray:
    reversed_cumulative = np.cumsum(
        block_squared_norms[:, ::-1], axis=1, dtype=np.float64
    )[:, ::-1]
    # The input block squares and this cumsum together stay below the declared
    # 1,024-operation FP64 path budget.  Inflate the ordinary NumPy reduction
    # before sqrt; a final nextafter then directs sqrt's one rounding upward.
    suffix_squared = _positive_reduction_upper(reversed_cumulative[:, 1:])
    suffix_norm = np.nextafter(
        np.sqrt(suffix_squared), np.float64(math.inf)
    )
    return _ceil_float32(suffix_norm)


def _pack_schedule_sidecar(
    sidecar: np.ndarray, topology: Topology
) -> np.ndarray:
    expected = (topology.rows, topology.bound_values_per_row)
    if sidecar.shape != expected:
        raise ValueError(f"sidecar shape {sidecar.shape} != {expected}")
    packed = np.ascontiguousarray(
        sidecar.reshape(
            topology.cta_group_count,
            topology.rows_per_cta_group,
            topology.bound_values_per_row,
        ).transpose(0, 2, 1),
        dtype="<f4",
    )
    if packed.nbytes != topology.sidecar_bytes:
        raise AssertionError("schedule sidecar byte count drift")
    return packed


def _schedule_sidecar_sha256(sidecar: np.ndarray, topology: Topology) -> str:
    packed = _pack_schedule_sidecar(sidecar, topology)
    return hashlib.sha256(packed.view(np.uint8)).hexdigest()


def save_schedule_sidecar(
    path: Path, sidecar: np.ndarray, topology: Topology
) -> dict[str, Any]:
    """Serialize canonical ``[group][checkpoint][local_row]`` little-endian f32."""

    packed = _pack_schedule_sidecar(sidecar, topology)
    with path.open("wb") as stream:
        packed.tofile(stream)
    if path.stat().st_size != topology.sidecar_bytes:
        raise AssertionError("serialized sidecar byte count drift")
    return {
        "path": str(path),
        "layout": "[cta_row_group][checkpoint][local_row]",
        "dtype": "<f4",
        "shape": list(packed.shape),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def build_directed_proofs(
    shard: Path,
    activations: np.ndarray,
    labels: Sequence[str],
    topology: Topology,
    chunk_rows: int,
    verify_hashes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode pinned NVFP4 bytes and construct conservative BF16 proof arrays."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if activations.shape != (len(labels), topology.columns):
        raise ValueError(
            f"activation shape {activations.shape} != {(len(labels), topology.columns)}"
        )
    checkpoint_manifest = validate_pinned_checkpoint(shard, verify_hashes)
    scale2 = struct.unpack("<f", struct.pack("<I", PINNED_SCALE2_BITS))[0]
    if not math.isfinite(scale2) or scale2 <= 0.0:
        raise ValueError("pinned weight_scale_2 must be positive and finite")
    exact_decode_contract = verify_exact_decode_products(scale2)
    packed = np.memmap(
        shard,
        mode="r",
        dtype=np.uint8,
        offset=PINNED_WEIGHT.absolute_offset,
        shape=(topology.rows, topology.packed_bytes_per_row),
    )
    raw_scales = np.memmap(
        shard,
        mode="r",
        dtype=np.uint8,
        offset=PINNED_SCALES.absolute_offset,
        shape=(topology.rows, topology.scale_bytes_per_row),
    )
    activation64 = np.asarray(activations, dtype=np.float64)
    if not np.all(np.isfinite(activation64)):
        raise ValueError("nonfinite finalnorm disables the q20 pruning proof")
    activation_abs = np.abs(activation64)

    activation_block_sq = np.empty(
        (len(labels), topology.checkpoints), dtype=np.float64
    )
    for checkpoint in range(topology.checkpoints):
        begin = checkpoint * topology.columns_per_checkpoint
        end = begin + topology.columns_per_checkpoint
        activation_block_sq[:, checkpoint] = np.sum(
            activation64[:, begin:end] ** 2, axis=1, dtype=np.float64
        )
    activation_suffix_norms = _directed_suffix_norms(activation_block_sq)

    upper_bounds = np.empty(
        (len(labels), topology.rows, topology.bound_values_per_row),
        dtype=np.float32,
    )
    incumbent_lowers = np.empty(
        (len(labels), topology.rows), dtype=np.float32
    )
    sidecar = np.empty(
        (topology.rows, topology.bound_values_per_row), dtype=np.float32
    )

    for row_begin in range(0, topology.rows, chunk_rows):
        row_end = min(row_begin + chunk_rows, topology.rows)
        packed_chunk = packed[row_begin:row_end]
        scale_chunk = raw_scales[row_begin:row_end]
        rows_in_chunk = row_end - row_begin
        block_dot = np.empty(
            (rows_in_chunk, len(labels), topology.checkpoints), dtype=np.float64
        )
        block_abs = np.empty_like(block_dot)
        block_sq = np.empty(
            (rows_in_chunk, topology.checkpoints), dtype=np.float64
        )
        for checkpoint in range(topology.checkpoints):
            weights = _decode_nvfp4_block(
                packed_chunk, scale_chunk, checkpoint, topology, scale2
            )
            begin = checkpoint * topology.columns_per_checkpoint
            end = begin + topology.columns_per_checkpoint
            block_dot[:, :, checkpoint] = weights @ activation64[:, begin:end].T
            block_abs[:, :, checkpoint] = (
                np.abs(weights) @ activation_abs[:, begin:end].T
            )
            block_sq[:, checkpoint] = np.sum(
                weights * weights, axis=1, dtype=np.float64
            )

        partial_hat = np.cumsum(block_dot, axis=2, dtype=np.float64)
        partial_abs_hat = np.cumsum(block_abs, axis=2, dtype=np.float64)
        partial_abs_upper = _positive_reduction_upper(partial_abs_hat)
        partial_fp64_error = _signed_reduction_error(partial_abs_upper)
        partial_upper = _add_up_float64(partial_hat, partial_fp64_error)
        partial_lower = _sub_down_float64(partial_hat, partial_fp64_error)
        suffix_norms = _directed_suffix_norms(block_sq)
        sidecar[row_begin:row_end] = suffix_norms
        suffix_bound = _mul_up_nonnegative_float64(
            suffix_norms[:, np.newaxis, :],
            activation_suffix_norms[np.newaxis, :, :],
        )
        total_abs_upper = _add_up_float64(
            partial_abs_upper[:, :, :-1], suffix_bound
        )
        production_fp32_error = _mul_up_nonnegative_float64(
            GAMMA_48_UPPER, total_abs_upper
        )
        exact_upper = _add_up_float64(
            _add_up_float64(partial_upper[:, :, :-1], suffix_bound),
            production_fp32_error,
        )
        final_fp32_error = _mul_up_nonnegative_float64(
            GAMMA_48_UPPER, partial_abs_upper[:, :, -1]
        )
        exact_lower = _sub_down_float64(
            partial_lower[:, :, -1], final_fp32_error
        )
        upper_chunk = _upper_to_bf16_rne_image(exact_upper).transpose(1, 0, 2)
        lower_chunk = _lower_to_bf16_rne_image(exact_lower).T
        # A single nonfinite checkpoint disables pruning for that complete
        # activation/row.  It may neither exit early nor publish an incumbent.
        ineligible = ~np.all(np.isfinite(upper_chunk), axis=2)
        ineligible |= ~np.isfinite(lower_chunk)
        upper_chunk[ineligible] = np.float32(math.inf)
        lower_chunk[ineligible] = np.float32(-math.inf)
        upper_bounds[:, row_begin:row_end, :] = upper_chunk
        incumbent_lowers[:, row_begin:row_end] = lower_chunk

    sidecar_sha256 = _schedule_sidecar_sha256(sidecar, topology)
    manifest = {
        "contract_version": PROOF_CONTRACT_VERSION,
        "checkpoint": checkpoint_manifest,
        "weight_scale_2": scale2,
        "exact_decode_products": exact_decode_contract,
        "gamma_48": GAMMA_48,
        "gamma_48_upper_float64": GAMMA_48_UPPER,
        "gamma_48_operation_path": {
            "lane_fma": 40,
            "local_pair_tree_additions": 2,
            "warp_tree_additions": 5,
            "final_weight_scale_2_multiply": 1,
            "total": 48,
            "scale2_handling": (
                "positive pinned scale2 is absorbed algebraically into exact "
                "offline coefficients; gamma48 includes its final FP32 multiply"
            ),
        },
        "fp64_reduction_envelope": {
            "max_operations_per_input_output_path": FP64_PROOF_MAX_PATH_OPS,
            "gamma_1024": FP64_GAMMA_PATH,
            "rho": FP64_PROOF_RHO,
            "rho_exceeds_gamma_1024": FP64_PROOF_RHO > FP64_GAMMA_PATH,
            "nonnegative_rule": "exact <= hat / (1-rho), division rounded upward",
            "signed_rule": "abs(exact-hat) <= rho * upward_sum_abs",
            "underflow_contract": {
                "binary64_min_normal_exponent": -1022,
                "minimum_nonzero_scaled_weight_activation_exponent": -156,
                "minimum_nonzero_activation_square_exponent": -266,
                "minimum_nonzero_scaled_weight_square_exponent": -46,
                "reason": (
                    "all decoded inputs are exact dyadic E2M1/E4M3/F32/BF16; "
                    "their nonzero products and cancellations remain far above "
                    "binary64 underflow"
                ),
            },
        },
        "bound_contract": (
            "exact binary FP64 coefficients; rho-directed FP64 reductions; "
            "upward FP32 suffix norms; Cauchy suffix; gamma48 including final "
            "scale2 multiply; monotone FP32-ceil/floor to BF16-RNE image"
        ),
        "activation_suffix_norm": {
            "source": "complete BF16 CTA-staged activation vector",
            "new_global_traffic_bytes": 0,
            "runtime_compute_and_sync_counted_by_sector_gate": False,
            "construction": (
                "BF16 squares and suffix reductions use the rho-directed FP64 "
                "contract, upward sqrt, then upward finite FP32"
            ),
        },
        "sidecar_layout": "[cta_row_group][checkpoint][local_row]",
        "sidecar_bytes": topology.sidecar_bytes,
        "sidecar_sha256": sidecar_sha256,
        "ineligible_activation_rows": int(
            np.count_nonzero(~np.all(np.isfinite(upper_bounds), axis=2))
        ),
        "nonfinite_policy": (
            "any nonfinite upper checkpoint or final lower marks the complete "
            "activation/row ineligible: all uppers=+inf and lower=-inf"
        ),
        "activation_labels": list(labels),
        "chunk_rows": chunk_rows,
    }
    return upper_bounds, incumbent_lowers, sidecar, manifest


@dataclasses.dataclass(frozen=True)
class ProofInput:
    """Conservative BF16-domain pruning bounds for one activation."""

    upper_bounds: np.ndarray
    incumbent_lowers: np.ndarray
    label: str

    def validate(self, topology: Topology) -> None:
        expected_upper = (topology.rows, topology.bound_values_per_row)
        if self.upper_bounds.shape != expected_upper:
            raise ValueError(
                f"{self.label}: upper_bounds {self.upper_bounds.shape} != {expected_upper}"
            )
        if self.incumbent_lowers.shape != (topology.rows,):
            raise ValueError(
                f"{self.label}: incumbent_lowers {self.incumbent_lowers.shape} != "
                f"{(topology.rows,)}"
            )
        if self.upper_bounds.dtype.kind != "f" or self.incumbent_lowers.dtype.kind != "f":
            raise TypeError(f"{self.label}: proof arrays must be floating point")
        if np.any(np.isnan(self.upper_bounds)) or np.any(
            np.isnan(self.incumbent_lowers)
        ):
            raise ValueError(f"{self.label}: proof arrays must not contain NaN")
        finite_rows = np.all(np.isfinite(self.upper_bounds), axis=1) & np.isfinite(
            self.incumbent_lowers
        )
        disabled_rows = np.all(np.isposinf(self.upper_bounds), axis=1) & np.isneginf(
            self.incumbent_lowers
        )
        if not np.all(finite_rows | disabled_rows):
            raise ValueError(
                f"{self.label}: each row must be finite or canonically disabled"
            )
        if np.any(
            self.incumbent_lowers[finite_rows, np.newaxis]
            > self.upper_bounds[finite_rows]
        ):
            raise ValueError(
                f"{self.label}: incumbent lower exceeds a checkpoint upper"
            )


def _apply_incumbent(
    processed: np.ndarray,
    row_ids: np.ndarray,
    upper_bounds: np.ndarray,
    incumbent: float,
) -> None:
    """Apply a universal strict-upper-below-incumbent pruning rule in place."""

    if not math.isfinite(incumbent) or row_ids.size == 0:
        return
    # One nonfinite upper checkpoint invalidates the complete row certificate;
    # never allow a later finite checkpoint to revive it.
    alive = np.all(np.isfinite(upper_bounds[row_ids]), axis=1)
    for checkpoint in range(upper_bounds.shape[1]):
        newly_pruned = alive & (upper_bounds[row_ids, checkpoint] < incumbent)
        if np.any(newly_pruned):
            processed[row_ids[newly_pruned]] = checkpoint + 1
            alive[newly_pruned] = False
        if not np.any(alive):
            break


def schedule_completed_wave(
    proof: ProofInput, topology: Topology, incumbent_delta: float
) -> np.ndarray:
    """Idealized completed-2,048-row-wave incumbent model."""

    processed = np.full(topology.rows, topology.checkpoints, dtype=np.uint8)
    incumbent = -math.inf
    for wave in range(topology.wave_count):
        begin = wave * topology.row_stride
        end = min(begin + topology.row_stride, topology.rows)
        rows = np.arange(begin, end, dtype=np.int64)
        _apply_incumbent(
            processed, rows, proof.upper_bounds, incumbent + incumbent_delta
        )
        completed = rows[processed[rows] == topology.checkpoints]
        finite = proof.incumbent_lowers[completed]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            incumbent = max(incumbent, float(np.max(finite)))
    return processed


def schedule_per_cta(
    proof: ProofInput, topology: Topology, incumbent_delta: float
) -> np.ndarray:
    """Implementable local incumbent with no cross-CTA publication."""

    processed = np.full(topology.rows, topology.checkpoints, dtype=np.uint8)
    rows_per_group = topology.rows_per_cta_group
    for cta in range(topology.grid_blocks):
        incumbent = -math.inf
        base = cta * rows_per_group
        while base < topology.rows:
            end = min(base + rows_per_group, topology.rows)
            rows = np.arange(base, end, dtype=np.int64)
            _apply_incumbent(
                processed, rows, proof.upper_bounds, incumbent + incumbent_delta
            )
            completed = rows[processed[rows] == topology.checkpoints]
            finite = proof.incumbent_lowers[completed]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                incumbent = max(incumbent, float(np.max(finite)))
            base += topology.row_stride
    return processed


def schedule_bounded_lag_atomic(
    proof: ProofInput,
    topology: Topology,
    incumbent_delta: float,
    lag_waves: int,
) -> np.ndarray:
    """Model an externally proven global-incumbent visibility lag.

    An atomic update alone does not bound CTA scheduling skew.  This schedule
    is therefore diagnostic unless a separate execution/synchronization proof
    establishes ``lag_waves`` for the eventual kernel.
    """

    if lag_waves < 0:
        raise ValueError("lag_waves must be non-negative")
    processed = np.full(topology.rows, topology.checkpoints, dtype=np.uint8)
    wave_maxima = np.full(topology.wave_count, -math.inf, dtype=np.float64)
    for wave in range(topology.wave_count):
        visible_wave = wave - lag_waves - 1
        incumbent = (
            float(np.max(wave_maxima[: visible_wave + 1]))
            if visible_wave >= 0
            else -math.inf
        )
        begin = wave * topology.row_stride
        end = min(begin + topology.row_stride, topology.rows)
        rows = np.arange(begin, end, dtype=np.int64)
        _apply_incumbent(
            processed, rows, proof.upper_bounds, incumbent + incumbent_delta
        )
        completed = rows[processed[rows] == topology.checkpoints]
        finite = proof.incumbent_lowers[completed]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            wave_maxima[wave] = float(np.max(finite))
    return processed


def count_traffic(
    processed_blocks: np.ndarray,
    topology: Topology,
    gate_bytes: int = DEFAULT_GATE_BYTES,
) -> dict[str, Any]:
    """Count exact request sectors for one vector of per-row K256 depths."""

    processed = np.asarray(processed_blocks, dtype=np.int64)
    if processed.shape != (topology.rows,):
        raise ValueError(f"processed shape {processed.shape} != {(topology.rows,)}")
    if np.any(processed < 1) or np.any(processed > topology.checkpoints):
        raise ValueError("each row must process between one and all checkpoints")

    weight_sectors = int(np.sum(processed) * (
        topology.weight_bytes_per_row_checkpoint // SECTOR_BYTES
    ))
    # The production XOR-dual scale path fetches both K256 halves together.
    scale_sectors = int(np.sum((processed + 1) // 2))
    sidecar_sectors = topology.cta_group_count * topology.sidecar_sectors_per_cta_group
    candidate_sectors = weight_sectors + scale_sectors + sidecar_sectors
    candidate_sector_bytes = candidate_sectors * SECTOR_BYTES

    logical_weight_bytes = int(
        np.sum(processed) * topology.weight_bytes_per_row_checkpoint
    )
    logical_scale_bytes = int(
        np.sum(processed) * topology.scale_useful_bytes_per_row_checkpoint
    )
    logical_candidate_bytes = (
        logical_weight_bytes + logical_scale_bytes + topology.sidecar_bytes
    )
    transaction_waste_bytes = candidate_sector_bytes - logical_candidate_bytes
    if transaction_waste_bytes < 0:
        raise AssertionError("sector bytes cannot be below logical bytes")

    baseline_sectors = topology.baseline_mandatory_bytes // SECTOR_BYTES
    saved_bytes = topology.baseline_mandatory_bytes - candidate_sector_bytes
    return {
        "processed_fraction": float(np.sum(processed))
        / float(topology.rows * topology.checkpoints),
        "rows_pruned": int(np.count_nonzero(processed < topology.checkpoints)),
        "maximum_processed_blocks": int(np.max(processed)),
        "minimum_processed_blocks": int(np.min(processed)),
        "baseline_mandatory_bytes": topology.baseline_mandatory_bytes,
        "baseline_request_sectors": baseline_sectors,
        "weight_request_sectors": weight_sectors,
        "scale_request_sectors": scale_sectors,
        "sidecar_request_sectors": sidecar_sectors,
        "candidate_request_sectors": candidate_sectors,
        "candidate_sector_bytes": candidate_sector_bytes,
        "logical_candidate_bytes": logical_candidate_bytes,
        "transaction_waste_bytes": transaction_waste_bytes,
        "saved_request_sectors": baseline_sectors - candidate_sectors,
        "saved_bytes": saved_bytes,
        "gate_bytes": gate_bytes,
        "gate_pass": saved_bytes >= gate_bytes,
    }


def simulate_proof(
    proof: ProofInput,
    topology: Topology,
    schedules: Sequence[str],
    incumbent_deltas: Sequence[float],
    atomic_lags: Sequence[int],
    gate_bytes: int,
) -> list[dict[str, Any]]:
    proof.validate(topology)
    results: list[dict[str, Any]] = []
    for delta in incumbent_deltas:
        for schedule in schedules:
            if schedule == "completed-wave":
                processed = schedule_completed_wave(proof, topology, delta)
                variants = [(schedule, None, processed)]
            elif schedule == "per-cta":
                processed = schedule_per_cta(proof, topology, delta)
                variants = [(schedule, None, processed)]
            elif schedule == "bounded-lag-atomic":
                variants = [
                    (
                        schedule,
                        lag,
                        schedule_bounded_lag_atomic(
                            proof, topology, delta, lag
                        ),
                    )
                    for lag in atomic_lags
                ]
            else:
                raise ValueError(f"unknown schedule {schedule!r}")
            for name, lag, processed_variant in variants:
                result = count_traffic(processed_variant, topology, gate_bytes)
                result.update(
                    {
                        "fixture": proof.label,
                        "schedule": name,
                        "atomic_lag_waves": lag,
                        "incumbent_delta": delta,
                        "requires_external_lag_proof": name
                        == "bounded-lag-atomic",
                    }
                )
                results.append(result)
    return results


def load_proofs_npz(
    path: Path, topology: Topology
) -> tuple[list[ProofInput], dict[str, Any]]:
    """Load a production-unreachable proof bundle generated offline.

    Required arrays are ``upper_bounds[A, rows, checkpoints-1]`` and
    ``incumbent_lowers[A, rows]``.  Optional ``labels[A]`` provides stable
    fixture names.  Bounds must already include directed-rounding envelopes;
    this traffic-only tool never silently treats approximate logits as proof.
    """

    with np.load(path, allow_pickle=False) as bundle:
        if "upper_bounds" not in bundle or "incumbent_lowers" not in bundle:
            raise ValueError("NPZ requires upper_bounds and incumbent_lowers")
        if "metadata_json" not in bundle:
            raise ValueError("NPZ requires metadata_json from build-proof")
        upper = np.asarray(bundle["upper_bounds"])
        lower = np.asarray(bundle["incumbent_lowers"])
        if upper.ndim == 2:
            upper = upper[np.newaxis, ...]
        if lower.ndim == 1:
            lower = lower[np.newaxis, ...]
        if upper.shape[0] != lower.shape[0]:
            raise ValueError("activation counts differ between proof arrays")
        if upper.dtype != np.dtype(np.float32) or lower.dtype != np.dtype(np.float32):
            raise TypeError("canonical proof arrays must use float32")
        if "labels" in bundle:
            labels = [str(value) for value in bundle["labels"].tolist()]
        else:
            labels = [f"activation-{index}" for index in range(upper.shape[0])]
        if len(labels) != upper.shape[0]:
            raise ValueError(
                f"label count {len(labels)} != activation count {upper.shape[0]}"
            )
        raw_metadata = bundle["metadata_json"]
        if raw_metadata.shape != ():
            raise ValueError("metadata_json must be a scalar string")
        try:
            metadata = json.loads(str(raw_metadata.item()))
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("proof metadata is not an object")
    expected_upper_hash = metadata.get("upper_bounds_sha256")
    expected_lower_hash = metadata.get("incumbent_lowers_sha256")
    if expected_upper_hash != _array_sha256(upper):
        raise ValueError("upper_bounds array hash mismatch")
    if expected_lower_hash != _array_sha256(lower):
        raise ValueError("incumbent_lowers array hash mismatch")
    if metadata.get("labels_sha256") != _labels_sha256(labels):
        raise ValueError("labels hash mismatch")
    metadata_topology = metadata.get("topology")
    if not isinstance(metadata_topology, dict):
        raise ValueError("proof metadata lacks topology")
    for key, value in dataclasses.asdict(topology).items():
        if metadata_topology.get(key) != value:
            raise ValueError(f"proof topology mismatch for {key}")
    proof_metadata = metadata.get("proof")
    if not isinstance(proof_metadata, dict) or not proof_metadata.get(
        "bound_contract"
    ):
        raise ValueError("proof metadata lacks the directed bound contract")
    proofs = [
        ProofInput(upper[index], lower[index], labels[index])
        for index in range(upper.shape[0])
    ]
    for proof in proofs:
        proof.validate(topology)
    return proofs, metadata


def canonical_provenance_errors(
    metadata: dict[str, Any], proofs: Sequence[ProofInput], sidecar: Path
) -> list[str]:
    """Return every reason the bundle cannot enter the frozen formal gate."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expected_labels = [f"P{position}" for position in range(19, 44)]
    actual_labels = [proof.label for proof in proofs]
    require(metadata.get("schema_version") == PROOF_SCHEMA_VERSION, "schema version")
    require(
        metadata.get("artifact") == "q20-lm-head-directed-bound-proof-bundle",
        "artifact kind",
    )
    require(metadata.get("production_reachable") is False, "production flag")
    require(metadata.get("gpu_used") is False, "GPU flag")
    require(metadata.get("mtp_used") is False, "MTP flag")
    require(actual_labels == expected_labels, "labels must be exactly P19..P43")
    require(
        metadata.get("script_sha256") == _sha256_path(Path(__file__).resolve()),
        "builder script hash",
    )

    fixture = metadata.get("fixture")
    if not isinstance(fixture, dict):
        errors.append("fixture metadata")
    else:
        require(fixture.get("bytes") == PINNED_FIXTURE_BYTES, "fixture bytes")
        require(
            fixture.get("sha256") == PINNED_FIXTURE_SHA256,
            "fixture sha256",
        )
        require(fixture.get("hash_verified") is True, "fixture hash verification")
        require(fixture.get("first_position") == 19, "fixture first position")
        require(fixture.get("last_position") == 43, "fixture last position")
        require(
            fixture.get("selected_count") == FORMAL_EXPECTED_FIXTURES,
            "fixture count",
        )

    proof_metadata = metadata.get("proof")
    if not isinstance(proof_metadata, dict):
        errors.append("proof metadata")
        proof_metadata = {}
    require(
        proof_metadata.get("contract_version") == PROOF_CONTRACT_VERSION,
        "proof contract version",
    )
    require(
        proof_metadata.get("activation_labels") == expected_labels,
        "proof activation labels",
    )
    reduction = proof_metadata.get("fp64_reduction_envelope")
    if not isinstance(reduction, dict):
        errors.append("FP64 reduction envelope")
    else:
        require(
            reduction.get("max_operations_per_input_output_path")
            == FP64_PROOF_MAX_PATH_OPS,
            "FP64 path budget",
        )
        require(reduction.get("rho") == FP64_PROOF_RHO, "FP64 rho")
        require(
            reduction.get("rho_exceeds_gamma_1024") is True,
            "FP64 rho/gamma proof",
        )

    checkpoint = proof_metadata.get("checkpoint")
    if not isinstance(checkpoint, dict):
        errors.append("checkpoint metadata")
        checkpoint = {}
    require(
        checkpoint.get("model_revision") == PINNED_MODEL_REVISION,
        "model revision",
    )
    require(checkpoint.get("shard_bytes") == PINNED_SHARD_BYTES, "shard bytes")
    require(checkpoint.get("header_bytes") == PINNED_HEADER_BYTES, "header bytes")
    require(
        checkpoint.get("weight_scale_2_bits") == f"0x{PINNED_SCALE2_BITS:08x}",
        "scale2 bits",
    )
    require(
        checkpoint.get("payload_hashes_verified") is True,
        "checkpoint payload hash verification",
    )
    expected_payload_hashes = {
        PINNED_WEIGHT.name: PINNED_WEIGHT.sha256,
        PINNED_SCALES.name: PINNED_SCALES.sha256,
    }
    require(
        checkpoint.get("payload_hashes") == expected_payload_hashes,
        "checkpoint payload hashes",
    )
    expected_tensors = {
        expected.name: (
            expected.dtype,
            list(expected.shape),
            expected.absolute_offset,
            expected.size,
        )
        for expected in (PINNED_WEIGHT, PINNED_SCALES, PINNED_SCALE2)
    }
    observed_tensors: dict[str, tuple[Any, Any, Any, Any]] = {}
    tensor_entries = checkpoint.get("tensors")
    if isinstance(tensor_entries, list):
        for entry in tensor_entries:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                observed_tensors[entry["name"]] = (
                    entry.get("dtype"),
                    entry.get("shape"),
                    entry.get("absolute_offset"),
                    entry.get("bytes"),
                )
    require(observed_tensors == expected_tensors, "checkpoint tensor contracts")

    repository_root = Path(__file__).resolve().parents[2]
    kernel_path = repository_root / PRODUCTION_KERNEL_RELATIVE_PATH
    kernel_metadata = metadata.get("production_kernel_source")
    if not isinstance(kernel_metadata, dict) or not kernel_path.is_file():
        errors.append("production kernel source")
    else:
        require(
            kernel_metadata.get("relative_path")
            == str(PRODUCTION_KERNEL_RELATIVE_PATH),
            "production kernel path",
        )
        require(
            kernel_metadata.get("sha256") == _sha256_path(kernel_path),
            "production kernel hash",
        )

    sidecar_metadata = proof_metadata.get("sidecar_artifact")
    expected_sidecar_hash = proof_metadata.get("sidecar_sha256")
    if not isinstance(sidecar_metadata, dict):
        errors.append("sidecar artifact metadata")
        sidecar_metadata = {}
    require(sidecar_metadata.get("dtype") == "<f4", "sidecar dtype")
    require(
        sidecar_metadata.get("layout")
        == "[cta_row_group][checkpoint][local_row]",
        "sidecar layout",
    )
    require(
        sidecar_metadata.get("shape")
        == [
            PRODUCTION_TOPOLOGY.cta_group_count,
            PRODUCTION_TOPOLOGY.bound_values_per_row,
            PRODUCTION_TOPOLOGY.rows_per_cta_group,
        ],
        "sidecar shape",
    )
    require(sidecar_metadata.get("bytes") == PRODUCTION_TOPOLOGY.sidecar_bytes, "sidecar metadata bytes")
    require(sidecar_metadata.get("sha256") == expected_sidecar_hash, "sidecar metadata hash")
    if not sidecar.is_file():
        errors.append("sidecar file")
    else:
        require(sidecar.stat().st_size == PRODUCTION_TOPOLOGY.sidecar_bytes, "sidecar file bytes")
        require(_sha256_path(sidecar) == expected_sidecar_hash, "sidecar file hash")
    return errors


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _labels_sha256(labels: Sequence[str]) -> str:
    encoded = json.dumps(
        list(labels), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_proofs_npz(
    path: Path,
    upper_bounds: np.ndarray,
    incumbent_lowers: np.ndarray,
    labels: Sequence[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if path.suffix != ".npz":
        raise ValueError("proof output must use the .npz suffix")
    hashes = {
        "upper_bounds_sha256": _array_sha256(upper_bounds),
        "incumbent_lowers_sha256": _array_sha256(incumbent_lowers),
        "labels_sha256": _labels_sha256(labels),
    }
    complete_manifest = dict(manifest)
    complete_manifest.update(hashes)
    metadata_json = json.dumps(
        complete_manifest, sort_keys=True, separators=(",", ":")
    )
    np.savez(
        path,
        upper_bounds=upper_bounds,
        incumbent_lowers=incumbent_lowers,
        labels=np.asarray(labels, dtype="U16"),
        metadata_json=np.asarray(metadata_json),
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
        **hashes,
    }


def synthetic_self_test() -> dict[str, Any]:
    """Small deterministic test that distinguishes all schedule models."""

    topology = Topology(
        rows=32,
        columns=512,
        grid_blocks=2,
        warps_per_block=2,
        rows_per_warp=4,
        columns_per_checkpoint=256,
    )
    address_contract = verify_address_contract(topology)
    pinned_scale2 = struct.unpack(
        "<f", struct.pack("<I", PINNED_SCALE2_BITS)
    )[0]
    decode_contract = verify_exact_decode_products(pinned_scale2)
    if decode_contract["finite_code_combinations"] != 4_064:
        raise AssertionError("decode code-product coverage drift")
    upper = np.full((topology.rows, 1), 100.0, dtype=np.float32)
    upper[16:, 0] = 5.0
    lowers = np.zeros(topology.rows, dtype=np.float32)
    lowers[0] = 10.0
    proof = ProofInput(upper, lowers, "synthetic")
    proof.validate(topology)

    completed = schedule_completed_wave(proof, topology, 0.0)
    per_cta = schedule_per_cta(proof, topology, 0.0)
    lagged = schedule_bounded_lag_atomic(proof, topology, 0.0, 1)
    sensitive = schedule_completed_wave(proof, topology, -10.0)
    if int(np.count_nonzero(completed == 1)) != 16:
        raise AssertionError("completed-wave synthetic pruning mismatch")
    if int(np.count_nonzero(per_cta == 1)) != 8:
        raise AssertionError("per-CTA synthetic pruning mismatch")
    if int(np.count_nonzero(lagged == 1)) != 0:
        raise AssertionError("bounded-lag synthetic pruning mismatch")
    if int(np.count_nonzero(sensitive == 1)) != 0:
        raise AssertionError("sensitivity synthetic pruning mismatch")

    # Concrete aligned requests must match the closed form used by traffic.
    if sectors_touched(weight_address(topology, 0, 0), 128) != (0, 1, 2, 3):
        raise AssertionError("weight address mapping mismatch")
    if len(sectors_touched(scale_pair_address(topology, 15, 0), 32)) != 1:
        raise AssertionError("scale address mapping mismatch")
    completed_traffic = count_traffic(completed, topology, gate_bytes=0)
    expected_scale_overfetch = 16 * int(np.count_nonzero(completed == 1))
    if completed_traffic["transaction_waste_bytes"] != expected_scale_overfetch:
        raise AssertionError("odd-depth dual-scale overfetch accounting mismatch")
    PRODUCTION_TOPOLOGY.validate()
    if PRODUCTION_TOPOLOGY.baseline_mandatory_bytes != 715_161_600:
        raise AssertionError("production mandatory-byte contract drift")
    if PRODUCTION_TOPOLOGY.sidecar_bytes != 18_872_320:
        raise AssertionError("production q20 sidecar-byte contract drift")
    if PRODUCTION_TOPOLOGY.cta_group_count != 7_760:
        raise AssertionError("production CTA-group count drift")
    if PRODUCTION_TOPOLOGY.sidecar_sectors_per_cta_group != 76:
        raise AssertionError("production sidecar sector geometry drift")
    return {
        "self_test": "pass",
        "address_contract": address_contract,
        "exact_decode_products": decode_contract,
        "completed_rows_pruned": int(np.count_nonzero(completed == 1)),
        "per_cta_rows_pruned": int(np.count_nonzero(per_cta == 1)),
        "lag1_rows_pruned": int(np.count_nonzero(lagged == 1)),
        "minus10_rows_pruned": int(np.count_nonzero(sensitive == 1)),
        "production_baseline_mandatory_bytes":
        PRODUCTION_TOPOLOGY.baseline_mandatory_bytes,
        "production_sidecar_bytes": PRODUCTION_TOPOLOGY.sidecar_bytes,
        "production_cta_groups": PRODUCTION_TOPOLOGY.cta_group_count,
        "production_sidecar_sectors_per_group":
        PRODUCTION_TOPOLOGY.sidecar_sectors_per_cta_group,
    }


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def _position_range(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("positions must be FIRST:LAST")
    try:
        first, last = (int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError("positions must be integers") from error
    if first < 0 or last < first:
        raise argparse.ArgumentTypeError("positions must satisfy 0 <= FIRST <= LAST")
    return first, last


def _topology_json(topology: Topology) -> dict[str, Any]:
    data = dataclasses.asdict(topology)
    data.update(
        {
            "checkpoints": topology.checkpoints,
            "bound_values_per_row": topology.bound_values_per_row,
            "row_stride": topology.row_stride,
            "rows_per_cta_group": topology.rows_per_cta_group,
            "wave_count": topology.wave_count,
            "baseline_mandatory_bytes": topology.baseline_mandatory_bytes,
            "sidecar_bytes": topology.sidecar_bytes,
        }
    )
    return data


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    self_test = subparsers.add_parser(
        "self-test", help="run the small CPU-only synthetic schedule/sector test"
    )
    self_test.add_argument("--output", type=Path)

    build_proof = subparsers.add_parser(
        "build-proof",
        help="CPU-only directed-bound producer for the pinned checkpoint/fixture",
    )
    checkpoint = build_proof.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--model-dir",
        type=Path,
        help=f"directory containing pinned {PINNED_SHARD_NAME}",
    )
    checkpoint.add_argument("--checkpoint-shard", type=Path)
    build_proof.add_argument("--finalnorm", type=Path, required=True)
    build_proof.add_argument(
        "--positions", type=_position_range, default=_position_range("19:43")
    )
    build_proof.add_argument("--chunk-rows", type=int, default=4_096)
    build_proof.add_argument("--output-proof", type=Path, required=True)
    build_proof.add_argument(
        "--output-sidecar",
        type=Path,
        help=(
            "canonical [group][checkpoint][row] <f4 payload; defaults beside "
            "--output-proof"
        ),
    )
    build_proof.add_argument("--output-manifest", type=Path)
    build_proof.add_argument(
        "--skip-pinned-hashes",
        action="store_true",
        help="development only: spans/bits stay pinned but 715 MB payload hashes are skipped",
    )

    simulate = subparsers.add_parser(
        "simulate", help="apply exact sector accounting to a conservative proof NPZ"
    )
    simulate.add_argument("--proof-npz", type=Path, required=True)
    simulate.add_argument(
        "--sidecar",
        type=Path,
        required=True,
        help="canonical sidecar emitted by build-proof; size/hash are formal-gate inputs",
    )
    simulate.add_argument(
        "--schedules",
        type=_csv_strings,
        default=_csv_strings("completed-wave,per-cta,bounded-lag-atomic"),
    )
    simulate.add_argument(
        "--atomic-lag-waves", type=_csv_ints, default=_csv_ints("1")
    )
    simulate.add_argument(
        "--incumbent-deltas", type=_csv_floats, default=_csv_floats("0,-1")
    )
    simulate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "self-test":
        _write_json(synthetic_self_test(), arguments.output)
        return 0

    if arguments.command == "build-proof":
        topology = PRODUCTION_TOPOLOGY
        verify_address_contract(topology)
        shard = (
            arguments.checkpoint_shard
            if arguments.checkpoint_shard is not None
            else arguments.model_dir / PINNED_SHARD_NAME
        )
        first_position, last_position = arguments.positions
        verify_hashes = not arguments.skip_pinned_hashes
        activations, labels, fixture_manifest = load_pinned_finalnorm(
            arguments.finalnorm,
            first_position,
            last_position,
            topology.columns,
            verify_hashes,
        )
        upper, lower, sidecar, proof_manifest = build_directed_proofs(
            shard,
            activations,
            labels,
            topology,
            arguments.chunk_rows,
            verify_hashes,
        )
        output_sidecar = arguments.output_sidecar
        if output_sidecar is None:
            output_sidecar = arguments.output_proof.with_suffix(".sidecar.f32le")
        sidecar_manifest = save_schedule_sidecar(
            output_sidecar, sidecar, topology
        )
        proof_manifest["sidecar_artifact"] = sidecar_manifest
        repository_root = Path(__file__).resolve().parents[2]
        production_kernel_path = (
            repository_root / PRODUCTION_KERNEL_RELATIVE_PATH
        )
        manifest = {
            "schema_version": PROOF_SCHEMA_VERSION,
            "artifact": "q20-lm-head-directed-bound-proof-bundle",
            "production_reachable": False,
            "gpu_used": False,
            "mtp_used": False,
            "topology": _topology_json(topology),
            "fixture": fixture_manifest,
            "proof": proof_manifest,
            "script_sha256": _sha256_path(Path(__file__).resolve()),
            "production_kernel_source": {
                "relative_path": str(PRODUCTION_KERNEL_RELATIVE_PATH),
                "sha256": _sha256_path(production_kernel_path),
            },
        }
        bundle_manifest = save_proofs_npz(
            arguments.output_proof, upper, lower, labels, manifest
        )
        result = {**manifest, "bundle": bundle_manifest}
        output_manifest = arguments.output_manifest
        if output_manifest is None:
            output_manifest = arguments.output_proof.with_suffix(".json")
        _write_json(result, output_manifest)
        sys.stdout.write(
            json.dumps(
                {
                    "proof": str(arguments.output_proof),
                    "sidecar": str(output_sidecar),
                    "manifest": str(output_manifest),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 0

    topology = PRODUCTION_TOPOLOGY
    address_contract = verify_address_contract(topology)
    proofs, proof_metadata = load_proofs_npz(arguments.proof_npz, topology)
    provenance_errors = canonical_provenance_errors(
        proof_metadata, proofs, arguments.sidecar
    )
    results: list[dict[str, Any]] = []
    for proof in proofs:
        results.extend(
            simulate_proof(
                proof,
                topology,
                arguments.schedules,
                arguments.incumbent_deltas,
                arguments.atomic_lag_waves,
                DEFAULT_GATE_BYTES,
            )
        )
    payload = {
        "schema_version": 1,
        "artifact": "q20-lm-head-exact-32b-sector-simulation",
        "production_reachable": False,
        "gpu_used": False,
        "mtp_used": False,
        "input": {
            "path": str(arguments.proof_npz),
            "bytes": arguments.proof_npz.stat().st_size,
            "sha256": _sha256_path(arguments.proof_npz),
            "metadata": proof_metadata,
            "sidecar": {
                "path": str(arguments.sidecar),
                "exists": arguments.sidecar.is_file(),
                "bytes": (
                    arguments.sidecar.stat().st_size
                    if arguments.sidecar.is_file()
                    else None
                ),
                "sha256": (
                    _sha256_path(arguments.sidecar)
                    if arguments.sidecar.is_file()
                    else None
                ),
            },
        },
        "topology": _topology_json(topology),
        "address_contract": address_contract,
        "sidecar_layout": "[cta_row_group][checkpoint_0_to_18][local_row_0_to_31]",
        "scale_contract": "one aligned 32B dual-scale request per active row per K512 pair",
        "claim_limit": (
            "The NPZ producer must supply directed-round conservative BF16-domain "
            "bounds. This tool proves request-sector arithmetic, not bound validity or latency."
        ),
        "structural_requirements": {
            "per_row_predicated_weight_loads": True,
            "dual_scale_preload_before_each_k512_pair": True,
            "cta_group_sidecar_preload": True,
            "sidecar_preload_barrier_cost_included_in_traffic": False,
            "bound_predicate_and_incumbent_cost_included_in_traffic": False,
            "no_cache_residency_credit": True,
        },
        "results": results,
    }
    independent_rows = [
        result
        for result in results
        if result["schedule"] == "per-cta"
        and result["incumbent_delta"] == -1.0
    ]
    independent_gate = (
        not provenance_errors
        and
        len(proofs) == FORMAL_EXPECTED_FIXTURES
        and len(independent_rows) == len(proofs)
        and all(result["saved_bytes"] >= DEFAULT_GATE_BYTES for result in independent_rows)
    )
    payload["independent_admission_gate"] = {
        "schedule": "per-cta",
        "incumbent_delta": -1.0,
        "expected_fixtures": FORMAL_EXPECTED_FIXTURES,
        "observed_fixtures": len(proofs),
        "canonical_input_pass": not provenance_errors,
        "canonical_input_errors": provenance_errors,
        "all_saved_at_least_gate_bytes": independent_gate,
        "reason": (
            "per-CTA is the only modeled schedule requiring no global completion "
            "or bounded-atomic-lag assumption"
        ),
    }
    _write_json(payload, arguments.output)
    return 0 if independent_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
