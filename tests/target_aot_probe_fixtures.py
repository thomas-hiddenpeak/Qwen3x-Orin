"""Pure-data fixtures for target-AOT probe evidence validator tests."""

from __future__ import annotations


def _resource(*, gate_up: bool) -> dict[str, object]:
    return {
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


def _boundary(*, features: int, digest_byte: str) -> dict[str, object]:
    digest = digest_byte * 64
    return {
        "elements": 192 * features,
        "full_elements": 128 * features,
        "tail_elements": 64 * features,
        "baseline_candidate_full_mismatches": 0,
        "baseline_candidate_tail_mismatches": 0,
        "baseline_replay_full_mismatches": 0,
        "baseline_replay_tail_mismatches": 0,
        "candidate_replay_full_mismatches": 0,
        "candidate_replay_tail_mismatches": 0,
        "first_mismatch_present": False,
        "first_mismatch_index": 0,
        "first_mismatch_row": 0,
        "first_mismatch_column": 0,
        "first_mismatch_expected": 0,
        "first_mismatch_actual": 0,
        "first_mismatch_pair": "",
        "baseline_sha256": digest,
        "candidate_sha256": digest,
        "replay_sha256": digest,
        "baseline_all_finite": True,
        "candidate_all_finite": True,
        "replay_all_finite": True,
        "baseline_complete_write": True,
        "candidate_complete_write": True,
        "replay_complete_write": True,
        "baseline_guards_intact": True,
        "candidate_guards_intact": True,
        "replay_guards_intact": True,
        "baseline_candidate_bitwise_exact": True,
        "baseline_replay_bitwise_exact": True,
        "candidate_replay_bitwise_exact": True,
        "bitwise_exact": True,
    }


def _artifact(*, gate_up: bool) -> dict[str, object]:
    return {
        "layer_index": 0,
        "role": 1 if gate_up else 2,
        "artifact_identity": 101 if gate_up else 102,
        "source_inventory_identity": 201 if gate_up else 202,
        "upload_receipt_identity": 301 if gate_up else 302,
        "plan_identity": 1 if gate_up else 2,
        "layout_identity": 1,
        "transform_identity": 1,
        "device_arena_offset": 0 if gate_up else 100_270_080,
        "payload_bytes": 100_270_080 if gate_up else 50_135_040,
        "payload_sha256": ("c" if gate_up else "d") * 64,
        "manifest_seal": 401 if gate_up else 402,
        "allocation_owner_identity": 501,
        "allocation_identity": 502,
        "device_ordinal": 0,
        "exact": True,
    }


def valid_m192_oracle_result() -> dict[str, object]:
    fixture_elements = 192 * 5_120
    return {
        "layer_index": 0,
        "token_count": 192,
        "full_token_count": 128,
        "tail_token_count": 64,
        "baseline_gate_launches": 6,
        "baseline_up_launches": 6,
        "baseline_down_launches": 6,
        "candidate_gate_up_launches": 1,
        "candidate_down_launches": 1,
        "replay_gate_up_launches": 1,
        "replay_down_launches": 1,
        "owner_attachment_authenticated": True,
        "canonical_layer0_weights": True,
        "activation_preserved_after_candidate": True,
        "residual_preserved_after_candidate": True,
        "activation_preserved_after_replay": True,
        "residual_preserved_after_replay": True,
        "activation_fixture": {
            "generator": (
                "q3x.layer0-m192.activation-palette-row17-column29-shift5.v1"
            ),
            "dtype": "bfloat16-raw-little-endian",
            "shape": [192, 5_120],
            "elements": fixture_elements,
            "bytes": fixture_elements * 2,
            "sha256": "a" * 64,
        },
        "residual_fixture": {
            "generator": "q3x.layer0-m192.residual-palette-row23-column11-xor.v1",
            "dtype": "bfloat16-raw-little-endian",
            "shape": [192, 5_120],
            "elements": fixture_elements,
            "bytes": fixture_elements * 2,
            "sha256": "b" * 64,
        },
        "receipt": {
            "baseline_route": (
                "canonical-sm87-nvfp4-m32x6-gate-up-silu-down-m32x6-residual"
            ),
            "candidate_plan": (
                "source-private-layer0-m192-m128n256k64-persistent16-v1"
            ),
            "candidate_layout": "consumer-n64-k16-lane-component-v1",
            "candidate_publication": (
                "gate-up-silu-bf16;"
                "down-plus-immutable-residual-to-independent-bf16"
            ),
            "owner_identity": 501,
            "allocation_identity": 502,
            "arena_bytes": 9_625_927_680,
            "device_ordinal": 0,
            "artifact_count": 128,
            "verified_payload_catalog_sha256": "e" * 64,
            "attachment_exact": True,
            "complete_owner_allocation_ranges_checked": True,
            "complete_owner_allocation_exact_cover": True,
            "complete_owner_allocation_nonoverlap": True,
            "oracle_allocations_disjoint_from_owner": True,
            "gate_up": _artifact(gate_up=True),
            "down": _artifact(gate_up=False),
        },
        "gate_up_resources": _resource(gate_up=True),
        "down_resources": _resource(gate_up=False),
        "gate_up": _boundary(features=17_408, digest_byte="1"),
        "down_residual": _boundary(features=5_120, digest_byte="2"),
        "cleanup": {
            "synchronize_attempted": True,
            "synchronize_cuda_error": 0,
            "device_frees_attempted": 11,
            "device_frees_succeeded": 11,
            "first_device_free_cuda_error": 0,
            "stream_destroy_attempted": True,
            "stream_destroy_cuda_error": 0,
            "passed": True,
        },
        "passed": True,
    }
