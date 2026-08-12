#!/usr/bin/env python3
"""Derive a post-exit Jetson nvmap diagnosis from a frozen AOT probe.

This tool is deliberately CPU-only and observational.  It never launches the
probe, opens a CUDA device, clears the nvmap page pool, or drops host caches.
The source probe status is immutable: the output can only add a derived
``no_owner_leak`` or ``inconclusive`` classification for the narrow
post-destruction memory-recovery question.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK_ROOT = REPOSITORY_ROOT / ".q3x-work"
CANONICAL_PROC_ROOT = pathlib.Path("/proc")
CANONICAL_NVMAP_DEBUGFS_ROOT = pathlib.Path("/sys/kernel/debug/nvmap")
EXPECTED_ARTIFACT = "q3x_sm87_target_aot_real_checkpoint_preparation"
DERIVED_ARTIFACT = "q3x_sm87_target_aot_nvmap_recovery_diagnostic"
RECOVERY_TOLERANCE_BYTES = 256 * 1024 * 1024
MAX_DEBUGFS_CAPTURE_BYTES = 16 * 1024 * 1024
BOOT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
SIZE_TOKEN_PATTERN = re.compile(r"^[0-9]+(?:[KMGTP]i?B?|B)?$", re.IGNORECASE)
HEX_TOKEN_PATTERN = re.compile(r"^[0-9a-fA-F]+$")
ADDRESS_TOKEN_PATTERN = re.compile(r"^(?:[0-9a-fA-F]+|0[xX][0-9a-fA-F]+)$")
ALLOCATION_CLIENT_HEADER = ("CLIENT", "PROCESS", "PID", "SIZE")
ALLOCATION_DETAIL_HEADER = (
    "BASE",
    "SIZE",
    "FLAGS",
    "REFS",
    "DUPES",
    "PINS",
    "KMAPS",
    "UMAPS",
    "SHARE",
    "UID",
    "FROM_HUGETLBFS",
)
REQUIRED_CHECKS = (
    "source_build_provenance_valid",
    "binary_provenance_valid",
    "engine_created",
    "c512_competition_path_exercised",
    "during_total_matches_initial",
    "exact_target_inventory",
    "mutually_exclusive_prefill_sidecars_empty",
    "engine_destroy_completed",
    "final_total_matches_initial",
)
MEMORY_CHECK = "memory_recovered_after_destroy"
SCHEMA_V3_REQUIRED_CHECKS = (
    "execution_identity_valid",
    "layer0_m192_oracle_passed",
)
M192_TOKENS = 192
M192_FULL_TOKENS = 128
M192_TAIL_TOKENS = 64
M192_HIDDEN = 5_120
M192_INTERMEDIATE = 17_408
M192_DEVICE_BUFFERS = 11
M192_DEVICE_ARENA_BYTES = 9_625_927_680
M192_ARTIFACT_COUNT = 128
M192_BASELINE_ROUTE = (
    "canonical-sm87-nvfp4-m32x6-gate-up-silu-down-m32x6-residual"
)
M192_CANDIDATE_PLAN = (
    "source-private-layer0-m192-m128n256k64-persistent16-v1"
)
M192_CANDIDATE_LAYOUT = "consumer-n64-k16-lane-component-v1"
M192_CANDIDATE_PUBLICATION = (
    "gate-up-silu-bf16;down-plus-immutable-residual-to-independent-bf16"
)


class EvidenceError(RuntimeError):
    """The requested derived evidence cannot be collected safely."""


def _path_is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_output_path(path: pathlib.Path) -> pathlib.Path:
    repository = REPOSITORY_ROOT.resolve()
    work = WORK_ROOT.resolve()
    if not _path_is_within(work, repository):
        raise EvidenceError("repository .q3x-work resolves outside the repository")
    resolved = path.expanduser().resolve(strict=False)
    if not _path_is_within(resolved, work):
        raise EvidenceError("--output must resolve inside repository .q3x-work")
    if resolved == work or resolved.suffix.lower() != ".json":
        raise EvidenceError("--output must name one JSON file below .q3x-work")
    if resolved.exists():
        raise EvidenceError(f"refusing to replace existing output: {resolved}")
    return resolved


def validate_input_path(path: pathlib.Path) -> pathlib.Path:
    repository = REPOSITORY_ROOT.resolve()
    work = WORK_ROOT.resolve()
    if not _path_is_within(work, repository):
        raise EvidenceError("repository .q3x-work resolves outside the repository")
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise EvidenceError("--input must be a regular non-symlink file")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"cannot resolve --input {path}: {error}") from error
    if not _path_is_within(resolved, work):
        raise EvidenceError("--input must resolve inside repository .q3x-work")
    if not resolved.is_file():
        raise EvidenceError("--input must be a regular file")
    return resolved


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_frozen_probe(path: pathlib.Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read --input {path}: {error}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"--input is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError("--input root must be a JSON object")
    return value, raw, sha256_bytes(raw)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"source probe field {name!r} must be an object")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(
            f"source probe field {name!r} must be a non-negative integer"
        )
    return value


def _utc_datetime(value: Any, name: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"source probe field {name!r} must be UTC text")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise EvidenceError(
            f"source probe field {name!r} is not ISO-8601: {error}"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise EvidenceError(f"source probe field {name!r} must be UTC")
    return parsed.astimezone(dt.timezone.utc)


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(byte not in "0123456789abcdef" for byte in value)
    ):
        raise EvidenceError(f"source probe field {name!r} must be lower SHA256")
    return value


def _required_true(mapping: Mapping[str, Any], name: str) -> None:
    if mapping.get(name) is not True:
        raise EvidenceError(f"source probe Oracle strong gate {name!r} must be true")


def _exact_nonnegative_integer(
    mapping: Mapping[str, Any], field: str, expected: int, scope: str
) -> None:
    actual = _nonnegative_integer(mapping.get(field), f"{scope}.{field}")
    if actual != expected:
        raise EvidenceError(f"source probe Oracle {scope}.{field} mismatch")


def _validate_m192_fixture(value: Any, name: str, generator: str) -> None:
    fixture = _mapping(value, name)
    if fixture.get("generator") != generator:
        raise EvidenceError(f"source probe Oracle {name}.generator mismatch")
    if fixture.get("dtype") != "bfloat16-raw-little-endian":
        raise EvidenceError(f"source probe Oracle {name}.dtype mismatch")
    if fixture.get("shape") != [M192_TOKENS, M192_HIDDEN]:
        raise EvidenceError(f"source probe Oracle {name}.shape mismatch")
    expected_elements = M192_TOKENS * M192_HIDDEN
    if _nonnegative_integer(fixture.get("elements"), f"{name}.elements") != (
        expected_elements
    ):
        raise EvidenceError(f"source probe Oracle {name}.elements mismatch")
    if _nonnegative_integer(fixture.get("bytes"), f"{name}.bytes") != (
        expected_elements * 2
    ):
        raise EvidenceError(f"source probe Oracle {name}.bytes mismatch")
    _lower_sha256(fixture.get("sha256"), f"{name}.sha256")


def _validate_m192_artifact(
    value: Any,
    name: str,
    *,
    expected_role: int,
    expected_plan: int,
    expected_payload_bytes: int,
    receipt: Mapping[str, Any],
) -> None:
    artifact = _mapping(value, name)
    expected_exact = {
        "layer_index": 0,
        "role": expected_role,
        "plan_identity": expected_plan,
        "layout_identity": 1,
        "transform_identity": 1,
        "payload_bytes": expected_payload_bytes,
        "device_ordinal": receipt["device_ordinal"],
        "allocation_owner_identity": receipt["owner_identity"],
        "allocation_identity": receipt["allocation_identity"],
    }
    for field, expected in expected_exact.items():
        _exact_nonnegative_integer(artifact, field, expected, name)
    for field in (
        "artifact_identity",
        "source_inventory_identity",
        "upload_receipt_identity",
        "manifest_seal",
    ):
        if _nonnegative_integer(artifact.get(field), f"{name}.{field}") == 0:
            raise EvidenceError(f"source probe Oracle {name}.{field} must be nonzero")
    _nonnegative_integer(
        artifact.get("device_arena_offset"), f"{name}.device_arena_offset"
    )
    _lower_sha256(artifact.get("payload_sha256"), f"{name}.payload_sha256")
    _required_true(artifact, "exact")


def _validate_m192_receipt(value: Any) -> None:
    receipt = _mapping(value, "layer0_m192_oracle.result.receipt")
    exact_text = {
        "baseline_route": M192_BASELINE_ROUTE,
        "candidate_plan": M192_CANDIDATE_PLAN,
        "candidate_layout": M192_CANDIDATE_LAYOUT,
        "candidate_publication": M192_CANDIDATE_PUBLICATION,
    }
    for field, expected in exact_text.items():
        if receipt.get(field) != expected:
            raise EvidenceError(f"source probe Oracle receipt.{field} mismatch")
    for field in ("owner_identity", "allocation_identity"):
        if _nonnegative_integer(receipt.get(field), f"receipt.{field}") == 0:
            raise EvidenceError(f"source probe Oracle receipt.{field} must be nonzero")
    _exact_nonnegative_integer(
        receipt, "arena_bytes", M192_DEVICE_ARENA_BYTES, "receipt"
    )
    _exact_nonnegative_integer(
        receipt, "artifact_count", M192_ARTIFACT_COUNT, "receipt"
    )
    device_ordinal = _nonnegative_integer(
        receipt.get("device_ordinal"), "receipt.device_ordinal"
    )
    if device_ordinal != 0:
        raise EvidenceError("source probe Oracle receipt.device_ordinal mismatch")
    _lower_sha256(
        receipt.get("verified_payload_catalog_sha256"),
        "receipt.verified_payload_catalog_sha256",
    )
    for gate in (
        "attachment_exact",
        "complete_owner_allocation_ranges_checked",
        "complete_owner_allocation_exact_cover",
        "complete_owner_allocation_nonoverlap",
        "oracle_allocations_disjoint_from_owner",
    ):
        _required_true(receipt, gate)
    _validate_m192_artifact(
        receipt.get("gate_up"),
        "receipt.gate_up",
        expected_role=1,
        expected_plan=1,
        expected_payload_bytes=100_270_080,
        receipt=receipt,
    )
    _validate_m192_artifact(
        receipt.get("down"),
        "receipt.down",
        expected_role=2,
        expected_plan=2,
        expected_payload_bytes=50_135_040,
        receipt=receipt,
    )


def _validate_m192_resource(value: Any, name: str, *, gate_up: bool) -> None:
    resource = _mapping(value, name)
    expected = {
        "role": 1 if gate_up else 2,
        "binary_version": 87,
        "registers_per_thread": 246 if gate_up else 210,
        "static_shared_bytes": 0,
        "dynamic_shared_bytes": 76_800,
        "local_bytes": 0,
        "maximum_threads_per_block": 256,
        "active_blocks_per_sm": 1,
        "physical_ctas": 16,
        "device_ordinal": 0,
        "device_major": 8,
        "device_minor": 7,
        "device_sm_count": 16,
        "block_m": 128,
        "block_n": 256,
        "block_k": 64,
        "pipeline_stages": 3,
        "cta_threads": 256,
        "grid_m": 2,
        "grid_n": 68 if gate_up else 20,
        "full_logical_tasks": 68 if gate_up else 20,
        "tail_logical_tasks": 68 if gate_up else 20,
        "total_logical_tasks": 136 if gate_up else 40,
        "n_halves_per_logical_task": 2 if gate_up else 1,
        "n_half_features": 128 if gate_up else 256,
        "raster_group_m": 2 if gate_up else 1,
        "n_stationary": not gate_up,
        "logical_tasks_per_cta_min": 8 if gate_up else 2,
        "logical_tasks_per_cta_max": 9 if gate_up else 3,
        "exact_geometry_gate": True,
        "exact_resource_gate": True,
    }
    for field, expected_value in expected.items():
        if isinstance(expected_value, bool):
            if resource.get(field) is not expected_value:
                raise EvidenceError(f"source probe Oracle {name}.{field} mismatch")
        else:
            _exact_nonnegative_integer(resource, field, expected_value, name)


def _validate_m192_boundary(
    value: Any, name: str, *, features: int
) -> None:
    boundary = _mapping(value, name)
    expected_elements = M192_TOKENS * features
    expected_counts = {
        "elements": expected_elements,
        "full_elements": M192_FULL_TOKENS * features,
        "tail_elements": M192_TAIL_TOKENS * features,
        "baseline_candidate_full_mismatches": 0,
        "baseline_candidate_tail_mismatches": 0,
        "baseline_replay_full_mismatches": 0,
        "baseline_replay_tail_mismatches": 0,
        "candidate_replay_full_mismatches": 0,
        "candidate_replay_tail_mismatches": 0,
        "first_mismatch_index": 0,
        "first_mismatch_row": 0,
        "first_mismatch_column": 0,
        "first_mismatch_expected": 0,
        "first_mismatch_actual": 0,
    }
    for field, expected in expected_counts.items():
        _exact_nonnegative_integer(boundary, field, expected, name)
    if boundary.get("first_mismatch_present") is not False or boundary.get(
        "first_mismatch_pair"
    ) != "":
        raise EvidenceError(f"source probe Oracle {name} mismatch metadata present")
    digests = [
        _lower_sha256(boundary.get(field), f"{name}.{field}")
        for field in ("baseline_sha256", "candidate_sha256", "replay_sha256")
    ]
    if len(set(digests)) != 1:
        raise EvidenceError(f"source probe Oracle {name} digest mismatch")
    for gate in (
        "baseline_all_finite",
        "candidate_all_finite",
        "replay_all_finite",
        "baseline_complete_write",
        "candidate_complete_write",
        "replay_complete_write",
        "baseline_guards_intact",
        "candidate_guards_intact",
        "replay_guards_intact",
        "baseline_candidate_bitwise_exact",
        "baseline_replay_bitwise_exact",
        "candidate_replay_bitwise_exact",
        "bitwise_exact",
    ):
        _required_true(boundary, gate)


def _validate_m192_cleanup(value: Any) -> None:
    cleanup = _mapping(value, "layer0_m192_oracle.result.cleanup")
    _required_true(cleanup, "synchronize_attempted")
    _required_true(cleanup, "stream_destroy_attempted")
    _required_true(cleanup, "passed")
    expected = {
        "synchronize_cuda_error": 0,
        "device_frees_attempted": M192_DEVICE_BUFFERS,
        "device_frees_succeeded": M192_DEVICE_BUFFERS,
        "first_device_free_cuda_error": 0,
        "stream_destroy_cuda_error": 0,
    }
    for field, expected_value in expected.items():
        _exact_nonnegative_integer(cleanup, field, expected_value, "cleanup")


def _validate_passed_m192_result(result: Mapping[str, Any]) -> None:
    expected_geometry = {
        "layer_index": 0,
        "token_count": M192_TOKENS,
        "full_token_count": M192_FULL_TOKENS,
        "tail_token_count": M192_TAIL_TOKENS,
        "baseline_gate_launches": 6,
        "baseline_up_launches": 6,
        "baseline_down_launches": 6,
        "candidate_gate_up_launches": 1,
        "candidate_down_launches": 1,
        "replay_gate_up_launches": 1,
        "replay_down_launches": 1,
    }
    for field, expected in expected_geometry.items():
        _exact_nonnegative_integer(result, field, expected, "result")
    for gate in (
        "owner_attachment_authenticated",
        "canonical_layer0_weights",
        "activation_preserved_after_candidate",
        "residual_preserved_after_candidate",
        "activation_preserved_after_replay",
        "residual_preserved_after_replay",
        "passed",
    ):
        _required_true(result, gate)
    _validate_m192_fixture(
        result.get("activation_fixture"),
        "activation_fixture",
        "q3x.layer0-m192.activation-palette-row17-column29-shift5.v1",
    )
    _validate_m192_fixture(
        result.get("residual_fixture"),
        "residual_fixture",
        "q3x.layer0-m192.residual-palette-row23-column11-xor.v1",
    )
    _validate_m192_receipt(result.get("receipt"))
    _validate_m192_resource(
        result.get("gate_up_resources"), "gate_up_resources", gate_up=True
    )
    _validate_m192_resource(
        result.get("down_resources"), "down_resources", gate_up=False
    )
    _validate_m192_boundary(
        result.get("gate_up"), "gate_up", features=M192_INTERMEDIATE
    )
    _validate_m192_boundary(
        result.get("down_residual"), "down_residual", features=M192_HIDDEN
    )
    _validate_m192_cleanup(result.get("cleanup"))


def validate_probe_shape(probe: Mapping[str, Any]) -> None:
    schema_version = probe.get("schema_version")
    if schema_version not in (2, 3):
        raise EvidenceError("source probe schema_version must be 2 or 3")
    if probe.get("artifact") != EXPECTED_ARTIFACT:
        raise EvidenceError(f"source probe artifact must be {EXPECTED_ARTIFACT!r}")
    if probe.get("status") not in ("pass", "fail"):
        raise EvidenceError("source probe status must be 'pass' or 'fail'")

    if schema_version == 3:
        execution = _mapping(probe.get("execution_identity"), "execution_identity")
        boot_id = execution.get("boot_id")
        if not isinstance(boot_id, str) or not BOOT_ID_PATTERN.fullmatch(boot_id):
            raise EvidenceError(
                "source probe execution_identity.boot_id has invalid format"
            )
        child_pid = _nonnegative_integer(
            execution.get("child_pid"), "execution_identity.child_pid"
        )
        if child_pid == 0:
            raise EvidenceError(
                "source probe execution_identity.child_pid must be positive"
            )
        child_started = _utc_datetime(
            execution.get("child_started_at_utc"),
            "execution_identity.child_started_at_utc",
        )
        child_finished = _utc_datetime(
            execution.get("child_evidence_finished_at_utc"),
            "execution_identity.child_evidence_finished_at_utc",
        )
        if child_finished < child_started:
            raise EvidenceError(
                "source probe execution_identity timestamps are reversed"
            )
        if not isinstance(execution.get("valid"), bool):
            raise EvidenceError(
                "source probe execution_identity.valid must be boolean"
            )

    binary = _mapping(probe.get("binary"), "binary")
    binary_path = binary.get("path")
    binary_sha = binary.get("self_sha256")
    if not isinstance(binary_path, str) or not pathlib.Path(binary_path).is_absolute():
        raise EvidenceError("source probe binary.path must be absolute")
    if (
        not isinstance(binary_sha, str)
        or len(binary_sha) != 64
        or any(byte not in "0123456789abcdef" for byte in binary_sha)
    ):
        raise EvidenceError("source probe binary.self_sha256 must be lower SHA256")
    binary_bytes = _nonnegative_integer(binary.get("bytes"), "binary.bytes")
    if binary_bytes == 0:
        raise EvidenceError("source probe binary.bytes must be positive")

    checks = _mapping(probe.get("checks"), "checks")
    schema_required_checks = (
        SCHEMA_V3_REQUIRED_CHECKS if schema_version == 3 else ()
    )
    for name in (*REQUIRED_CHECKS, *schema_required_checks, MEMORY_CHECK):
        if not isinstance(checks.get(name), bool):
            raise EvidenceError(f"source probe checks.{name} must be boolean")
    for name, value in checks.items():
        if not isinstance(name, str) or not isinstance(value, bool):
            raise EvidenceError("source probe checks values must be booleans")

    attempted = _mapping(probe.get("attempted"), "attempted")
    if not attempted:
        raise EvidenceError("source probe attempted object must not be empty")
    for name, value in attempted.items():
        if not isinstance(name, str) or not isinstance(value, bool):
            raise EvidenceError("source probe attempted values must be booleans")

    cuda_errors = _mapping(probe.get("cuda_errors"), "cuda_errors")
    if not cuda_errors:
        raise EvidenceError("source probe cuda_errors object must not be empty")
    for name, value in cuda_errors.items():
        _nonnegative_integer(value, f"cuda_errors.{name}")

    device = _mapping(probe.get("device"), "device")
    for name in (
        "total_before_bytes",
        "total_after_destroy_bytes",
        "free_before_bytes",
        "free_after_destroy_bytes",
        "destroy_recovery_tolerance_bytes",
    ):
        _nonnegative_integer(device.get(name), f"device.{name}")
    if device["destroy_recovery_tolerance_bytes"] != RECOVERY_TOLERANCE_BYTES:
        raise EvidenceError(
            "source probe destroy recovery tolerance must equal fixed 256 MiB"
        )

    diagnostic = _mapping(probe.get("diagnostic"), "diagnostic")
    _nonnegative_integer(diagnostic.get("code"), "diagnostic.code")
    _nonnegative_integer(diagnostic.get("cuda_error"), "diagnostic.cuda_error")

    if schema_version == 3:
        execution = _mapping(probe["execution_identity"], "execution_identity")
        if checks["execution_identity_valid"] is not execution["valid"]:
            raise EvidenceError(
                "source probe execution identity validity fields disagree"
            )
        oracle = _mapping(probe.get("layer0_m192_oracle"), "layer0_m192_oracle")
        for name in ("attempted", "result_available", "passed"):
            if not isinstance(oracle.get(name), bool):
                raise EvidenceError(
                    f"source probe layer0_m192_oracle.{name} must be boolean"
                )
        if attempted.get("layer0_m192_oracle") is not oracle["attempted"]:
            raise EvidenceError(
                "source probe layer0 M192 attempted fields disagree"
            )
        if checks["layer0_m192_oracle_passed"] is not oracle["passed"]:
            raise EvidenceError(
                "source probe layer0 M192 passed fields disagree"
            )
        oracle_result = oracle.get("result")
        if oracle["result_available"]:
            result = _mapping(oracle_result, "layer0_m192_oracle.result")
            if not isinstance(result.get("passed"), bool):
                raise EvidenceError(
                    "source probe layer0_m192_oracle.result.passed must be boolean"
                )
            if result["passed"] is not oracle["passed"]:
                raise EvidenceError(
                    "source probe layer0 M192 result passed fields disagree"
                )
            if result["passed"]:
                _validate_passed_m192_result(result)
        elif oracle_result is not None:
            raise EvidenceError(
                "source probe unavailable layer0 M192 result must be null"
            )
        if oracle["result_available"] and not oracle["attempted"]:
            raise EvidenceError(
                "source probe layer0 M192 result cannot exist without attempt"
            )
        if oracle["passed"] and not oracle["result_available"]:
            raise EvidenceError(
                "source probe passed layer0 M192 oracle requires a result"
            )
        oracle_diagnostic = _mapping(
            oracle.get("diagnostic"), "layer0_m192_oracle.diagnostic"
        )
        oracle_diagnostic_code = _nonnegative_integer(
            oracle_diagnostic.get("code"), "layer0_m192_oracle.diagnostic.code"
        )
        oracle_cuda_error = _nonnegative_integer(
            oracle_diagnostic.get("cuda_error"),
            "layer0_m192_oracle.diagnostic.cuda_error",
        )
        if oracle["passed"] and (
            oracle_diagnostic_code != 0 or oracle_cuda_error != 0
        ):
            raise EvidenceError(
                "source probe passed layer0 M192 oracle requires zero diagnostic"
            )


def _read_limited(path: pathlib.Path, *, optional: bool = False) -> dict[str, Any]:
    try:
        with path.open("rb", buffering=0) as stream:
            payload = stream.read(MAX_DEBUGFS_CAPTURE_BYTES + 1)
    except FileNotFoundError:
        if optional:
            return {"present": False, "sha256": None, "text": None}
        raise EvidenceError(f"required nvmap debugfs node is missing: {path}")
    except OSError as error:
        raise EvidenceError(
            f"cannot read nvmap debugfs node {path}: {error}"
        ) from error
    if len(payload) > MAX_DEBUGFS_CAPTURE_BYTES:
        raise EvidenceError(f"nvmap debugfs node exceeds capture limit: {path}")
    return {
        "present": True,
        "sha256": sha256_bytes(payload),
        "text": payload.decode("utf-8", errors="replace"),
    }


def _read_decimal(path: pathlib.Path) -> int:
    captured = _read_limited(path)
    text = str(captured["text"]).strip()
    if not text or not text.isdecimal():
        raise EvidenceError(f"nvmap pagepool node is not decimal: {path}")
    return int(text)


def _orphan_data_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    first = lines[0].upper()
    if "BASE" in first and "SIZE" in first and "REFS" in first:
        return lines[1:]
    return lines


def _parse_client_rows(
    text: str,
) -> tuple[list[int], list[dict[str, Any]], list[str]]:
    pids: set[int] = set()
    rows: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("CLIENT ") or line.lower().startswith("total"):
            continue
        fields = line.split()
        if len(fields) >= 4 and fields[-2].isdecimal() and SIZE_TOKEN_PATTERN.fullmatch(
            fields[-1]
        ):
            pid = int(fields[-2])
            pids.add(pid)
            rows.append(
                {
                    "client": fields[0],
                    "process": " ".join(fields[1:-2]),
                    "pid": pid,
                    "size": fields[-1],
                }
            )
        else:
            unparsed.append(line)
    return sorted(pids), rows, unparsed


def _parse_allocation_client_rows(
    text: str,
) -> tuple[
    list[int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    pids: set[int] = set()
    client_rows: list[dict[str, Any]] = []
    allocation_detail_rows: list[dict[str, Any]] = []
    unparsed: list[str] = []
    summaries: list[str] = []
    saw_client_header = False
    saw_detail_header = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if tuple(fields) == ALLOCATION_CLIENT_HEADER:
            saw_client_header = True
            continue
        if tuple(fields) == ALLOCATION_DETAIL_HEADER:
            saw_detail_header = True
            continue
        if (
            saw_client_header
            and len(fields) == 2
            and fields[0].lower() == "total"
            and SIZE_TOKEN_PATTERN.fullmatch(fields[1])
        ):
            summaries.append(raw_line)
            continue
        detail_claimed = (
            saw_detail_header
            and (
                raw_line[:1].isspace()
                or ADDRESS_TOKEN_PATTERN.fullmatch(fields[0]) is not None
            )
        )
        if detail_claimed:
            if (
                len(fields) == len(ALLOCATION_DETAIL_HEADER)
                and HEX_TOKEN_PATTERN.fullmatch(fields[0])
                and SIZE_TOKEN_PATTERN.fullmatch(fields[1])
                and HEX_TOKEN_PATTERN.fullmatch(fields[2])
                and all(field.isdecimal() for field in fields[3:9])
                and HEX_TOKEN_PATTERN.fullmatch(fields[9])
                and fields[10] in ("0", "1")
            ):
                allocation_detail_rows.append(
                    {
                        "base": fields[0],
                        "size": fields[1],
                        "flags": fields[2],
                        "refs": int(fields[3]),
                        "dupes": int(fields[4]),
                        "pins": int(fields[5]),
                        "kmaps": int(fields[6]),
                        "umaps": int(fields[7]),
                        "share": int(fields[8]),
                        "uid": fields[9],
                        "from_hugetlbfs": int(fields[10]),
                        "raw_line": raw_line,
                    }
                )
            else:
                # Indentation or an address-like first token claims the detail
                # grammar.  It must never fall through and masquerade as a
                # four-column client row when SIZE or later columns are
                # truncated, unfamiliar, or malformed.
                unparsed.append(raw_line)
            continue
        if (
            saw_client_header
            and len(fields) == 4
            and fields[2].isdecimal()
            and SIZE_TOKEN_PATTERN.fullmatch(fields[3])
        ):
            pid = int(fields[2])
            pids.add(pid)
            client_rows.append(
                {
                    "client": fields[0],
                    "process": fields[1],
                    "pid": pid,
                    "size": fields[3],
                }
            )
            continue
        # Retain exact whitespace and every column.  Allocation-table formats
        # have changed between Jetson releases; silently dropping an
        # unfamiliar detail row could hide the probe's surviving owner.
        unparsed.append(raw_line)
    return (
        sorted(pids),
        client_rows,
        allocation_detail_rows,
        unparsed,
        summaries,
    )


def _capture_table(path: pathlib.Path, *, optional: bool = False) -> dict[str, Any]:
    captured = _read_limited(path, optional=optional)
    if not captured["present"]:
        return captured
    text = str(captured["text"])
    pids, rows, unparsed = _parse_client_rows(text)
    captured["client_pids"] = pids
    captured["client_rows"] = rows
    captured["unparsed_client_lines"] = unparsed
    return captured


def capture_nvmap(debugfs_root: pathlib.Path, page_size: int) -> dict[str, Any]:
    root = debugfs_root.resolve()
    pagepool_root = root / "pagepool"
    pagepool = {
        "page_pool_available_pages": _read_decimal(
            pagepool_root / "page_pool_available_pages"
        ),
        "page_pool_available_big_pages": _read_decimal(
            pagepool_root / "page_pool_available_big_pages"
        ),
        "page_pool_big_page_size": _read_decimal(
            pagepool_root / "page_pool_big_page_size"
        ),
        "page_pool_pages_to_zero": _read_decimal(
            pagepool_root / "page_pool_pages_to_zero"
        ),
    }
    # available_big_pages is a subset/capability view of available_pages on
    # this Jetson nvmap implementation.  Adding it would double count pages.
    pagepool["absolute_available_bytes"] = (
        pagepool["page_pool_available_pages"] * page_size
    )
    pagepool["absolute_available_bytes_formula"] = (
        "page_pool_available_pages * host_page_size_bytes; "
        "available_big_pages is not added"
    )

    spaces: dict[str, Any] = {}
    for space in ("iovmm", "fsi"):
        space_root = root / space
        orphan = _read_limited(space_root / "orphan_handles")
        orphan_lines = _orphan_data_lines(str(orphan["text"]))
        orphan["data_lines"] = orphan_lines
        orphan["empty"] = not orphan_lines

        clients = _capture_table(space_root / "clients")
        allocations = _read_limited(space_root / "allocations", optional=True)
        if allocations["present"]:
            (
                allocation_pids,
                allocation_rows,
                allocation_detail_rows,
                unparsed_allocation_lines,
                allocation_summary_lines,
            ) = _parse_allocation_client_rows(str(allocations["text"]))
            allocations["client_pids"] = allocation_pids
            allocations["client_rows"] = allocation_rows
            allocations["allocation_detail_rows"] = allocation_detail_rows
            allocations["unparsed_allocation_lines"] = (
                unparsed_allocation_lines
            )
            allocations["summary_lines"] = allocation_summary_lines
        else:
            allocations["client_pids"] = []
            allocations["client_rows"] = []
            allocations["allocation_detail_rows"] = []
            allocations["unparsed_allocation_lines"] = []
            allocations["summary_lines"] = []
        spaces[space] = {
            "orphan_handles": orphan,
            "clients": clients,
            "allocations": allocations,
        }

    handles_root = root / "handles_by_pid"
    try:
        entries = list(handles_root.iterdir())
    except OSError as error:
        raise EvidenceError(f"cannot enumerate {handles_root}: {error}") from error
    unexpected = sorted(entry.name for entry in entries if not entry.name.isdecimal())
    if unexpected:
        raise EvidenceError(
            "handles_by_pid contains non-numeric entries: " + ", ".join(unexpected)
        )
    handle_pids = sorted(int(entry.name) for entry in entries)
    handle_entries = {
        str(pid): _read_limited(handles_root / str(pid)) for pid in handle_pids
    }
    return {
        "debugfs_root": str(root),
        "pagepool": pagepool,
        "address_spaces": spaces,
        "handles_by_pid_pids": handle_pids,
        "handles_by_pid": handle_entries,
    }


def normalize_observed_nvmap_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive table rows from an already captured immutable raw snapshot.

    A lifecycle report intentionally retains both raw debugfs text and the
    parser output that existed when it was captured.  Offline reclassification
    must use the current strict parser against that raw text; otherwise a
    parser correction cannot repair a frozen report, and stale derived fields
    could be mistaken for current evidence.
    """

    normalized = copy.deepcopy(dict(snapshot))
    spaces = normalized.get("address_spaces")
    if not isinstance(spaces, dict):
        raise EvidenceError("observed nvmap address_spaces must be an object")
    for name in ("iovmm", "fsi"):
        space = spaces.get(name)
        if not isinstance(space, dict):
            raise EvidenceError(
                f"observed nvmap address_spaces.{name} must be an object"
            )

        orphan = space.get("orphan_handles")
        if not isinstance(orphan, dict) or orphan.get("present") is not True:
            raise EvidenceError(
                f"observed nvmap {name}.orphan_handles must be present"
            )
        orphan_text = orphan.get("text")
        if not isinstance(orphan_text, str):
            raise EvidenceError(
                f"observed nvmap {name}.orphan_handles.text must be text"
            )
        orphan_lines = _orphan_data_lines(orphan_text)
        orphan["data_lines"] = orphan_lines
        orphan["empty"] = not orphan_lines

        clients = space.get("clients")
        if not isinstance(clients, dict) or clients.get("present") is not True:
            raise EvidenceError(f"observed nvmap {name}.clients must be present")
        clients_text = clients.get("text")
        if not isinstance(clients_text, str):
            raise EvidenceError(
                f"observed nvmap {name}.clients.text must be text"
            )
        client_pids, client_rows, unparsed_client_lines = _parse_client_rows(
            clients_text
        )
        clients["client_pids"] = client_pids
        clients["client_rows"] = client_rows
        clients["unparsed_client_lines"] = unparsed_client_lines

        allocations = space.get("allocations")
        if not isinstance(allocations, dict) or not isinstance(
            allocations.get("present"), bool
        ):
            raise EvidenceError(
                f"observed nvmap {name}.allocations must record present"
            )
        if allocations["present"]:
            allocations_text = allocations.get("text")
            if not isinstance(allocations_text, str):
                raise EvidenceError(
                    f"observed nvmap {name}.allocations.text must be text"
                )
            (
                allocation_pids,
                allocation_rows,
                allocation_detail_rows,
                unparsed_allocation_lines,
                allocation_summary_lines,
            ) = _parse_allocation_client_rows(allocations_text)
            allocations["client_pids"] = allocation_pids
            allocations["client_rows"] = allocation_rows
            allocations["allocation_detail_rows"] = allocation_detail_rows
            allocations["unparsed_allocation_lines"] = (
                unparsed_allocation_lines
            )
            allocations["summary_lines"] = allocation_summary_lines
        else:
            allocations["client_pids"] = []
            allocations["client_rows"] = []
            allocations["allocation_detail_rows"] = []
            allocations["unparsed_allocation_lines"] = []
            allocations["summary_lines"] = []
    return normalized


def read_boot_id(proc_root: pathlib.Path) -> str:
    path = proc_root / "sys/kernel/random/boot_id"
    try:
        value = path.read_text(encoding="ascii").strip().lower()
    except (OSError, UnicodeError) as error:
        raise EvidenceError(f"cannot read boot_id from {path}: {error}") from error
    if not BOOT_ID_PATTERN.fullmatch(value):
        raise EvidenceError(f"boot_id has an invalid format: {value!r}")
    return value


def authenticate_probe_binary(binary: Mapping[str, Any]) -> dict[str, Any]:
    requested = pathlib.Path(str(binary["path"])).expanduser()
    if requested.is_symlink():
        raise EvidenceError("source probe binary.path is now a symlink")
    try:
        resolved = requested.resolve(strict=True)
        stat = resolved.stat()
    except OSError as error:
        raise EvidenceError(
            f"cannot resolve source probe binary {requested}: {error}"
        ) from error
    if not resolved.is_file():
        raise EvidenceError("source probe binary.path is not a regular file")
    if not _path_is_within(resolved, WORK_ROOT.resolve()):
        raise EvidenceError(
            "source probe binary.path must resolve inside repository .q3x-work"
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"cannot hash source probe binary: {error}") from error
    actual_sha256 = digest.hexdigest()
    expected_sha256 = str(binary["self_sha256"])
    expected_bytes = int(binary["bytes"])
    matches = actual_sha256 == expected_sha256 and stat.st_size == expected_bytes
    if not matches:
        raise EvidenceError(
            "source probe binary no longer matches its frozen size/SHA256"
        )
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "sha256": actual_sha256,
        "matches_frozen_identity": True,
    }


def _process_name_matches_binary(process_name: str, binary_path: str) -> bool:
    """Fail closed on the prefix produced by Jetson's fixed PROCESS column."""

    name = process_name.strip()
    binary_name = pathlib.Path(binary_path).name
    return len(name) >= 8 and (
        binary_name.startswith(name) or name.startswith(binary_name)
    )


def scan_matching_probe_processes(
    proc_root: pathlib.Path, expected_binary_path: str
) -> dict[str, Any]:
    expected = pathlib.Path(expected_binary_path).resolve(strict=False)
    matched: list[int] = []
    inaccessible: list[dict[str, Any]] = []
    scanned = 0
    try:
        entries = sorted(
            (entry for entry in proc_root.iterdir() if entry.name.isdecimal()),
            key=lambda entry: int(entry.name),
        )
    except OSError as error:
        raise EvidenceError(
            f"cannot enumerate proc root {proc_root}: {error}"
        ) from error
    for entry in entries:
        pid = int(entry.name)
        scanned += 1
        try:
            target = os.readlink(entry / "exe")
        except FileNotFoundError:
            continue
        except OSError as error:
            inaccessible.append(
                {"pid": pid, "errno": error.errno, "error": str(error)}
            )
            continue
        target = target.removesuffix(" (deleted)")
        if pathlib.Path(target).resolve(strict=False) == expected:
            matched.append(pid)
    return {
        "proc_root": str(proc_root.resolve()),
        "expected_probe_binary_path": str(expected),
        "scanned_numeric_process_entries": scanned,
        "matching_probe_elf_pids": matched,
        "inaccessible_executables": inaccessible,
        "scan_complete": not inaccessible,
    }


def _all_bool_values(mapping: Mapping[str, Any], expected: bool) -> bool:
    return bool(mapping) and all(value is expected for value in mapping.values())


def production_root_binding(
    *,
    debugfs_root: pathlib.Path,
    proc_root: pathlib.Path,
    observed_nvmap: Mapping[str, Any],
    observed_process_audit: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_debugfs = debugfs_root.resolve(strict=False)
    resolved_proc = proc_root.resolve(strict=False)
    observed_debugfs = pathlib.Path(
        str(observed_nvmap.get("debugfs_root", ""))
    ).resolve(strict=False)
    observed_proc = pathlib.Path(
        str(observed_process_audit.get("proc_root", ""))
    ).resolve(strict=False)
    criteria = {
        "requested_proc_root_is_canonical": (
            resolved_proc == CANONICAL_PROC_ROOT
        ),
        "requested_nvmap_debugfs_root_is_canonical": (
            resolved_debugfs == CANONICAL_NVMAP_DEBUGFS_ROOT
        ),
        "observed_proc_root_matches_request": observed_proc == resolved_proc,
        "observed_nvmap_debugfs_root_matches_request": (
            observed_debugfs == resolved_debugfs
        ),
    }
    return {
        "authority": (
            "canonical_jetson_host"
            if all(criteria.values())
            else "test_only_injected_roots"
        ),
        "requested_proc_root": str(resolved_proc),
        "requested_nvmap_debugfs_root": str(resolved_debugfs),
        "observed_proc_root": str(observed_proc),
        "observed_nvmap_debugfs_root": str(observed_debugfs),
        "criteria": criteria,
        "production_bound": all(criteria.values()),
    }


def derive_report(
    probe: Mapping[str, Any],
    *,
    input_path: pathlib.Path,
    input_sha256: str,
    debugfs_root: pathlib.Path,
    proc_root: pathlib.Path,
    captured_at: dt.datetime | None = None,
    page_size: int | None = None,
    observed_boot_id: str | None = None,
    observed_nvmap: Mapping[str, Any] | None = None,
    observed_binary_authentication: Mapping[str, Any] | None = None,
    observed_process_audit: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    validate_probe_shape(probe)
    if captured_at is None:
        captured_at = dt.datetime.now(dt.timezone.utc)
    if captured_at.tzinfo is None:
        raise EvidenceError("captured_at must be timezone-aware")
    if page_size is None:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise EvidenceError("host page size must be positive")

    boot_id = (
        read_boot_id(proc_root) if observed_boot_id is None else observed_boot_id
    )
    if not BOOT_ID_PATTERN.fullmatch(boot_id):
        raise EvidenceError("observed post-exit boot_id has invalid format")
    nvmap = (
        capture_nvmap(debugfs_root, page_size)
        if observed_nvmap is None
        else normalize_observed_nvmap_snapshot(observed_nvmap)
    )
    binary = _mapping(probe["binary"], "binary")
    binary_authentication = (
        authenticate_probe_binary(binary)
        if observed_binary_authentication is None
        else dict(observed_binary_authentication)
    )
    if (
        pathlib.Path(str(binary_authentication.get("path"))).resolve(strict=False)
        != pathlib.Path(str(binary["path"])).resolve(strict=False)
        or binary_authentication.get("bytes") != binary["bytes"]
        or binary_authentication.get("sha256") != binary["self_sha256"]
        or binary_authentication.get("matches_frozen_identity") is not True
    ):
        raise EvidenceError(
            "observed binary authentication does not match frozen source probe"
        )
    process = (
        scan_matching_probe_processes(
            proc_root, str(binary_authentication["path"])
        )
        if observed_process_audit is None
        else dict(observed_process_audit)
    )
    root_binding = production_root_binding(
        debugfs_root=debugfs_root,
        proc_root=proc_root,
        observed_nvmap=nvmap,
        observed_process_audit=process,
    )
    checks = _mapping(probe["checks"], "checks")
    attempted = _mapping(probe["attempted"], "attempted")
    cuda_errors = _mapping(probe["cuda_errors"], "cuda_errors")
    diagnostic = _mapping(probe["diagnostic"], "diagnostic")
    device = _mapping(probe["device"], "device")

    free_before = _nonnegative_integer(
        device["free_before_bytes"], "device.free_before_bytes"
    )
    free_after = _nonnegative_integer(
        device["free_after_destroy_bytes"], "device.free_after_destroy_bytes"
    )
    cuda_free_gap = max(0, free_before - free_after)
    pool_bytes = int(nvmap["pagepool"]["absolute_available_bytes"])
    residual = max(0, cuda_free_gap - pool_bytes)

    required_checks_true = all(checks[name] is True for name in REQUIRED_CHECKS)
    unknown_checks_true = all(
        value is True
        for name, value in checks.items()
        if name not in (*REQUIRED_CHECKS, MEMORY_CHECK)
    )
    source_exclusive_memory_failure = (
        probe["status"] == "fail"
        and required_checks_true
        and unknown_checks_true
        and checks[MEMORY_CHECK] is False
        and _all_bool_values(attempted, True)
        and all(value == 0 for value in cuda_errors.values())
        and diagnostic["code"] == 0
        and diagnostic["cuda_error"] == 0
    )

    spaces = nvmap["address_spaces"]
    orphan_tables_empty = all(
        spaces[name]["orphan_handles"]["empty"] for name in ("iovmm", "fsi")
    )
    client_parse_complete = all(
        not spaces[name]["clients"]["unparsed_client_lines"]
        for name in ("iovmm", "fsi")
    )
    allocation_parse_complete = all(
        not spaces[name]["allocations"].get(
            "unparsed_allocation_lines", []
        )
        for name in ("iovmm", "fsi")
    )
    client_pids = sorted(
        set(spaces["iovmm"]["clients"]["client_pids"])
        | set(spaces["fsi"]["clients"]["client_pids"])
    )
    client_rows = [
        row
        for name in ("iovmm", "fsi")
        for row in spaces[name]["clients"]["client_rows"]
    ]
    allocation_rows = [
        row
        for name in ("iovmm", "fsi")
        for row in spaces[name]["allocations"].get("client_rows", [])
    ]
    allocation_client_pids = sorted(
        set(spaces["iovmm"]["allocations"].get("client_pids", []))
        | set(spaces["fsi"]["allocations"].get("client_pids", []))
    )
    all_nvmap_client_pids = sorted(set(client_pids) | set(allocation_client_pids))
    probe_named_client_pids = sorted(
        {
            int(row["pid"])
            for row in (*client_rows, *allocation_rows)
            if _process_name_matches_binary(
                str(row["process"]), str(binary_authentication["path"])
            )
        }
    )
    matching_pids = list(process["matching_probe_elf_pids"])
    child_exited = bool(process["scan_complete"]) and not matching_pids
    matching_absent_from_clients = not probe_named_client_pids and not (
        set(matching_pids) & set(all_nvmap_client_pids)
    )
    unexplained_handle_pids = sorted(
        set(nvmap["handles_by_pid_pids"]) - set(all_nvmap_client_pids)
    )
    matching_absent_from_handles = not unexplained_handle_pids and not (
        set(matching_pids) & set(nvmap["handles_by_pid_pids"])
    )
    pagepool_explains_gap = residual <= RECOVERY_TOLERANCE_BYTES

    criteria: dict[str, bool] = {
        "source_status_is_fail": probe["status"] == "fail",
        "source_exclusive_failure_is_memory_recovery": (
            source_exclusive_memory_failure
        ),
        "all_source_cuda_errors_zero": all(
            value == 0 for value in cuda_errors.values()
        ),
        "source_diagnostic_error_zero": (
            diagnostic["code"] == 0 and diagnostic["cuda_error"] == 0
        ),
        "matching_probe_elf_process_has_exited": child_exited,
        "matching_probe_elf_absent_from_nvmap_clients": (
            matching_absent_from_clients
        ),
        "matching_probe_elf_absent_from_handles_by_pid": (
            matching_absent_from_handles
        ),
        "nvmap_client_tables_parse_completely": client_parse_complete,
        "nvmap_allocation_tables_parse_completely": allocation_parse_complete,
        "iovmm_and_fsi_orphan_tables_empty": orphan_tables_empty,
        "post_exit_pagepool_explains_cuda_free_gap_within_tolerance": (
            pagepool_explains_gap
        ),
        **root_binding["criteria"],
    }
    if probe["schema_version"] == 3:
        execution = _mapping(probe["execution_identity"], "execution_identity")
        child_finished = _utc_datetime(
            execution["child_evidence_finished_at_utc"],
            "execution_identity.child_evidence_finished_at_utc",
        )
        child_pid = int(execution["child_pid"])
        criteria.update(
            {
                "source_execution_identity_is_valid": (
                    execution["valid"] is True
                    and checks["execution_identity_valid"] is True
                ),
                "source_and_post_exit_boot_id_match": (
                    execution["boot_id"] == boot_id
                ),
                "source_evidence_finished_before_post_exit_capture": (
                    child_finished <= captured_at.astimezone(dt.timezone.utc)
                ),
                "source_child_pid_absent_from_post_exit_nvmap_clients": (
                    child_pid not in all_nvmap_client_pids
                ),
                "source_child_pid_absent_from_post_exit_handles_by_pid": (
                    child_pid not in nvmap["handles_by_pid_pids"]
                ),
            }
        )
    classified = all(criteria.values())
    rejection_reasons = [name for name, accepted in criteria.items() if not accepted]

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": DERIVED_ARTIFACT,
        "captured_at_utc": captured_at.astimezone(dt.timezone.utc).isoformat(),
        "claim_boundary": (
            "derived diagnostic only: classifies whether the post-exit Jetson "
            "nvmap page pool explains the source probe's in-process "
            "cudaMemGetInfo recovery gap; does not alter the source probe "
            "status and has no launcher, generation, numerical, performance, "
            "production-route, or leak-freedom authority beyond the observed "
            "owner tables; non-canonical proc/debugfs roots are test-only and "
            "force an inconclusive classification"
        ),
        "observation_authority": root_binding,
        "source_probe": {
            "input_path": str(input_path.resolve()),
            "input_sha256": input_sha256,
            "schema_version": probe["schema_version"],
            "artifact": probe["artifact"],
            "status": probe["status"],
            "status_preserved": True,
            "binary_path": binary["path"],
            "binary_bytes": binary["bytes"],
            "binary_self_sha256": binary["self_sha256"],
            "binary_reauthentication": binary_authentication,
        },
        "host_snapshot": {
            "boot_id": boot_id,
            "page_size_bytes": page_size,
            "hostname": os.uname().nodename,
            "kernel": os.uname().release,
            "machine": os.uname().machine,
            "uid": os.getuid(),
        },
        "nvmap_snapshot": nvmap,
        "probe_process_audit": process,
        "calculation": {
            "cuda_free_before_bytes": free_before,
            "cuda_free_after_destroy_bytes": free_after,
            "cuda_free_gap_bytes": cuda_free_gap,
            "absolute_post_exit_pagepool_bytes": pool_bytes,
            "pagepool_adjusted_residual_bytes": residual,
            "derived_tolerance_bytes": RECOVERY_TOLERANCE_BYTES,
            "formula": (
                "max(0, free_before-free_after) minus absolute post-exit "
                "page_pool_available_pages*host_page_size, floored at zero"
            ),
        },
        "observed_nvmap_pid_sets": {
            "client_pids": client_pids,
            "client_or_allocation_pids": all_nvmap_client_pids,
            "handles_by_pid_pids": nvmap["handles_by_pid_pids"],
            "probe_name_matching_client_or_allocation_pids": (
                probe_named_client_pids
            ),
            "handles_without_current_client_row_pids": unexplained_handle_pids,
            "allocation_client_pids": sorted(
                allocation_client_pids
            ),
        },
        "criteria": criteria,
        "classification": "no_owner_leak" if classified else "inconclusive",
        "rejection_reasons": rejection_reasons,
    }
    return report, classified


def publish_create_only(path: pathlib.Path, report: Mapping[str, Any]) -> None:
    path = validate_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _path_is_within(path.parent.resolve(), WORK_ROOT.resolve()):
        raise EvidenceError("resolved output parent escaped repository .q3x-work")
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.tmp.",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise EvidenceError(
                f"refusing to replace existing output: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name:
            try:
                pathlib.Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "derive a CPU-only post-exit nvmap recovery diagnosis without "
            "changing the frozen source probe status"
        )
    )
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--debugfs-root",
        type=pathlib.Path,
        default=pathlib.Path("/sys/kernel/debug/nvmap"),
        help=(
            "canonical nvmap debugfs root; an override is test-only and can "
            "never produce an accepted no-owner-leak classification"
        ),
    )
    parser.add_argument(
        "--proc-root",
        type=pathlib.Path,
        default=pathlib.Path("/proc"),
        help=(
            "canonical procfs root; an override is test-only and can never "
            "produce an accepted no-owner-leak classification"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = validate_output_path(args.output)
        input_path = validate_input_path(args.input)
        probe, _raw, input_sha256 = read_frozen_probe(input_path)
        report, classified = derive_report(
            probe,
            input_path=input_path,
            input_sha256=input_sha256,
            debugfs_root=args.debugfs_root,
            proc_root=args.proc_root,
        )
        publish_create_only(output, report)
    except EvidenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"classification={report['classification']}")
    print(f"source_probe_status={report['source_probe']['status']}")
    print(f"evidence={output}")
    return 0 if classified else 1


if __name__ == "__main__":
    raise SystemExit(main())
