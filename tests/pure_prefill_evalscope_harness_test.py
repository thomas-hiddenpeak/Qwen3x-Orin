#!/usr/bin/env python3
"""Host-only tests for the authenticated pure-Prefill EvalScope harness."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "tools/evaluation/run_native_pure_prefill_matrix.sh"
FRAGMENT_NATIVE_PAYLOAD_BYTES = 8_623_226_880
HYBRID_PAYLOAD_BYTES = 8_623_226_880
PROJECTION_MAJOR_PAYLOAD_BYTES = 8_623_226_880
ATTENTION_K256_PAYLOAD_BYTES = 12_353_536_000
ATTENTION_K256_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-attention-k256"
)
ATTENTION_K256_ALTERNATING_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "alternating-attention-k256"
)
ATTENTION_K256_ALTERNATING_DOWN_PAIRRING_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "alternating-down-pairring-attention-k256"
)
ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "alternating-down-16warp-pairring-attention-k256"
)
ATTENTION_K256_A_EXCHANGE_B4_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "alternating-down-16warp-pairring-attention-k256-a-exchange-b4"
)
ATTENTION_K256_LDMATRIX_PAIRFEED_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4"
)
K256_PAIRFEED_PACKAGE_MODE = (
    "cumulative-prefill-k256-m128n256-pairfeed-package-"
    "attention-k256-a-exchange-b4"
)
ATTENTION_K256_M32N512_OWNER_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m32n512-owner-k128-b4-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
ATTENTION_K256_SHAPE_SEPARATED_MARLIN_MODE = (
    "cumulative-prefill-current-best-mlp-k512-shape-separated-marlin-"
    "package-attention-k256-a-exchange-b4"
)
ATTENTION_K256_M128N128_A_EXCHANGE_B3_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-"
    "ldmatrix-pairfeed-down-16warp-pairring-attention-k256-"
    "m128n128-a-exchange-b3"
)
PROJECTION_MAJOR_MODE = (
    "cumulative-prefill-current-best-mlp-k512-projection-major-gateup-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
PAIRED_WARP_MODE = (
    "cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
ATTENTION_K256_M128N128_PROJECTION_SERIAL_MODE = (
    "cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
ATTENTION_K256_M128N64_SAME_CTA_MODE = (
    "cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
ATTENTION_K256_M128N512_FUSED_QUANTIZE_MODE = (
    "cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-"
    "down-16warp-pairring-attention-k256-a-exchange-b4"
)
GDN_PROMPT_SPAN_MACRO_MODE = (
    f"{ATTENTION_K256_A_EXCHANGE_B4_MODE}-gdn-prompt-span-macro"
)
GDN_PROMPT_SPAN_MACRO_SELECTOR = (
    "Q3X_RUN_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION"
)
GDN_PROMPT_SPAN_MACRO_MARKER = (
    "prefill_projection_span_linear_gdn_prompt_span_macro"
)
ATTENTION_K256_LAYOUT = (
    "sm87_s4_n64_packed_k64_scale_k256_consumer_v3"
)
ATTENTION_K256_INCUMBENT_SELECTOR = (
    "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION"
)
ATTENTION_K256_A_EXCHANGE_B4_SELECTOR = (
    "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION"
)
ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR = (
    "Q3X_RUN_A4W4_ATTENTION_K256_M128N128_A_EXCHANGE_B3_ADMISSION"
)
MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
    "ALTERNATING_ADMISSION"
)
MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
    "LDMATRIX_PAIRFEED_ADMISSION"
)
MLP_K512_M32N512_OWNER_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION"
)
MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR = (
    "Q3X_RUN_A4W4_MLP_K512_SHAPE_SEPARATED_MARLIN_PACKAGE_ADMISSION"
)
MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION"
)
MLP_K512_M128N64_SAME_CTA_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION"
)
MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION"
)
MLP_K512_PROJECTION_MAJOR_SELECTOR = (
    "Q3X_RUN_A4W4_MLP_K512_PROJECTION_MAJOR_GATEUP_CANONICAL_DOWN_ADMISSION"
)
MLP_K512_REGISTER_PIPELINE_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_K512_M64N128_REGISTER_PIPELINE_ADMISSION"
)
MLP_K512_PAIRED_MASTER_SELECTOR = (
    "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION"
)
MLP_K512_PAIRED_WARP_SELECTOR = (
    "Q3X_RUN_A4W4_GATEUP_K512_M64N8_PAIRED_WARP_REGISTER_PIPELINE_ADMISSION"
)
K256_PAIRFEED_PACKAGE_SELECTOR = (
    "Q3X_RUN_A4W4_MLP_K256_M128N256_PAIRFEED_PACKAGE_ADMISSION"
)
ATTENTION_K256_MARKERS = (
    "prefill_projection_span_linear_qkv_z_k256_m128n256",
    "prefill_projection_span_linear_output_k256_m128n256",
    "prefill_projection_span_full_q_k_v_k256_m128n256",
    "prefill_projection_span_full_output_k256_m128n256",
)
ATTENTION_K256_A_EXCHANGE_B4_MARKERS = tuple(
    f"{marker}_a_exchange_b4" for marker in ATTENTION_K256_MARKERS
)
ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS = (
    "prefill_projection_span_linear_qkv_z_k256_m128n128_a_exchange_b3",
    "prefill_projection_span_linear_output_k256_m128n128_a_exchange_b3",
    "prefill_projection_span_full_q_k_v_k256_m128n128_a_exchange_b3",
    "prefill_projection_span_full_output_k256_m128n128_a_exchange_b3",
)
MLP_K512_CURRENT_MODE = "cumulative-prefill-current-best-mlp-k512"
MLP_K512_V1_MODE = "cumulative-prefill-current-best-mlp-k512-v1"
MLP_K512_EDGE_MODE = "cumulative-prefill-current-best-mlp-k512-edge"
MLP_K512_EDGE_M128N64_MODE = (
    "cumulative-prefill-current-best-mlp-k512-edge-m128n64"
)
MLP_K512_DOWN_M16N64_V2_MODE = (
    "cumulative-prefill-current-best-mlp-k512-down-m16n64-v2"
)
MLP_K512_EDGE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge"
)
MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_"
    "m64n128_k256_alternating"
)
MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_"
    "m64n128_k256_ldmatrix_pairfeed"
)
MLP_K512_M32N512_OWNER_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_"
    "m32n512_owner_k128_b4"
)
MLP_K512_SHAPE_SEPARATED_MARLIN_GATE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_"
    "m64n256_marlin_k64_b3"
)
MLP_K512_SHAPE_SEPARATED_MARLIN_DOWN_MARKER = (
    "prefill_projection_span_mlp_k512_down_m64n256_16warp_pairring"
)
MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY = (
    "prefill_projection_span_mlp_k512_gateup_m128n128_"
    "projection_serial_primary"
)
MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY = (
    "prefill_projection_span_mlp_k512_gateup_m128n128_"
    "projection_serial_secondary"
)
MLP_K512_M128N64_SAME_CTA_PRIMARY = (
    "prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary"
)
MLP_K512_M128N64_SAME_CTA_SECONDARY = (
    "prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary"
)
MLP_K512_M128N512_FUSED_QUANTIZE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize"
)
MLP_K512_PRODUCT_QUANTIZE_MARKER = (
    "prefill_projection_span_mlp_k512_product_quantize"
)
MLP_K512_EDGE_M128N64_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_m128n64"
)
MLP_K512_DOWN_M16N64_V2_MARKER = (
    "prefill_projection_span_mlp_k512_down_m16n64_v2"
)
FRAGMENT_NATIVE_M64N128_1CTA_MODE = (
    "cumulative-prefill-current-best-mlp-k512-fragment-native-"
    "m64n128-1cta"
)
FRAGMENT_NATIVE_M64N128_1CTA_PRIMARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m64n128_1cta_gateup_primary"
)
FRAGMENT_NATIVE_M64N128_1CTA_SECONDARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m64n128_1cta_gateup_secondary"
)
FRAGMENT_NATIVE_M128N64_STAGED_MODE = (
    "cumulative-prefill-current-best-mlp-k512-fragment-native-"
    "m128n64-staged"
)
FRAGMENT_NATIVE_M128N64_STAGED_PRIMARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m128n64_staged_gateup_primary"
)
FRAGMENT_NATIVE_M128N64_STAGED_SECONDARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m128n64_staged_gateup_secondary"
)
HYBRID_GATE_MODE = (
    "cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256"
)
HYBRID_GATE_DOWN_PAIRRING_MODE = (
    "cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-"
    "attention-k256"
)
HYBRID_LAYOUT = (
    "sm87_s4_gateup_n64_paired_down_n64_canonical_scale_k512_"
    "mlp_hybrid_v1"
)
HYBRID_INPUT_MARKER = (
    "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_"
    "input_quantize"
)
HYBRID_GATE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_"
    "paired_ldmatrix"
)
HYBRID_CANONICAL_DOWN_MARKER = (
    "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_down"
)
HYBRID_PAIRRING_DOWN_MARKER = (
    "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring"
)
DOWN_16WARP_PAIRRING_MARKER = (
    "prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring"
)
DOWN_PAIRRING_SELECTOR = (
    "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION"
)
DOWN_16WARP_PAIRRING_SELECTOR = (
    "Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION"
)
K256_PAIRFEED_PACKAGE_MARKERS = (
    "prefill_projection_span_mlp_k256_input_quantize",
    "prefill_projection_span_mlp_k256_gateup_down_edge_"
    "m128n256_pairfeed",
    "prefill_projection_span_mlp_k256_down_m128n128_16warp_pairring",
)
PROJECTION_MAJOR_LAYOUT = (
    "sm87_s4_gateup_n64_projection_major_down_n64_canonical_scale_k512_"
    "mlp_hybrid_v2"
)
PROJECTION_MAJOR_INPUT_MARKER = (
    "prefill_projection_span_mlp_k512_projection_major_input_quantize"
)
PROJECTION_MAJOR_GATE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline"
)
PROJECTION_MAJOR_DOWN_16WARP_MARKER = (
    "prefill_projection_span_mlp_k512_projection_major_down_m128n128_"
    "16warp_pairring"
)
PAIRED_WARP_GATE_MARKER = (
    "prefill_projection_span_mlp_k512_gateup_m64n8_"
    "paired_warp_register_pipeline"
)
PAIRED_WARP_DOWN_16WARP_MARKER = (
    "prefill_projection_span_mlp_k512_paired_warp_down_m128n128_"
    "16warp_pairring"
)


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.server = root / "fake-server"
        self.model_dir = root / "model"
        self.corpus_dir = root / "corpus"
        self.output = root / "output"
        self.payload = root / "weights-a4-k64.bin"
        self.policy = root / "policy-1p0.json"
        self.receipt = root / "weights-a4-k64.bin.receipt.json"
        self.k256_payload = root / "weights-a4-k256.bin"
        self.k256_policy = root / "policy-k256-v3.json"
        self.k256_receipt = root / "weights-a4-k256.bin.receipt.json"
        self.k512_payload = root / "attention-o-k512.bin"
        self.k512_policy = root / "attention-o-k512-policy.json"
        self.k512_receipt = root / "attention-o-k512.bin.receipt.json"
        self.mlp_k512_payload = root / "mlp-k512.bin"
        self.mlp_k512_policy = root / "mlp-k512-policy.json"
        self.mlp_k512_receipt = root / "mlp-k512.bin.receipt.json"
        self.fragment_native_payload = root / "mlp-k512-fragment-native.bin"
        self.fragment_native_policy = (
            root / "mlp-k512-fragment-native-policy.json"
        )
        self.fragment_native_receipt = (
            root / "mlp-k512-fragment-native.bin.receipt.json"
        )
        self.hybrid_payload = (
            root / "mlp-k512-paired-gateup-canonical-down.bin"
        )
        self.hybrid_policy = (
            root / "mlp-k512-paired-gateup-canonical-down-policy.json"
        )
        self.hybrid_receipt = (
            root / "mlp-k512-paired-gateup-canonical-down.bin.receipt.json"
        )
        self.projection_major_payload = (
            root / "mlp-k512-projection-major-gateup-canonical-down.bin"
        )
        self.projection_major_policy = (
            root / "mlp-k512-projection-major-gateup-canonical-down-policy.json"
        )
        self.projection_major_receipt = (
            root
            / "mlp-k512-projection-major-gateup-canonical-down.bin.receipt.json"
        )
        self.fake_bin = root / "fake-bin"
        self.model_dir.mkdir()
        self.corpus_dir.mkdir()
        self.fake_bin.mkdir()
        self.server.write_text(
            "#!/bin/sh\n"
            "# Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION\n"
            "# Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION\n"
            f"# {GDN_PROMPT_SPAN_MACRO_SELECTOR}\n"
            "# Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION\n"
            "# Q3X_FULL_ATTENTION_FLASHINFER_DIRECT\n"
            "# Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION\n"
            "# Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_"
            "ADMISSION\n"
            f"# {ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR}\n"
            "# Q3X_RUN_A4W4_MLP_K512_ADMISSION\n"
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION\n"
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "ALTERNATING_ADMISSION\n"
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "LDMATRIX_PAIRFEED_ADMISSION\n"
            f"# {MLP_K512_M32N512_OWNER_SELECTOR}\n"
            f"# {MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR}\n"
            f"# {MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR}\n"
            f"# {MLP_K512_M128N64_SAME_CTA_SELECTOR}\n"
            f"# {MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR}\n"
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION\n"
            f"# {MLP_K512_PAIRED_MASTER_SELECTOR}\n"
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_"
            "LDMATRIX_ADMISSION\n"
            f"# {MLP_K512_PROJECTION_MAJOR_SELECTOR}\n"
            f"# {MLP_K512_REGISTER_PIPELINE_SELECTOR}\n"
            f"# {MLP_K512_PAIRED_WARP_SELECTOR}\n"
            f"# {K256_PAIRFEED_PACKAGE_SELECTOR}\n"
            "# Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_"
            "ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_"
            "ADMISSION\n"
            f"{MLP_K512_EDGE_MARKER}\n"
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER}\n"
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER}\n"
            f"{MLP_K512_M32N512_OWNER_MARKER}\n"
            f"{MLP_K512_SHAPE_SEPARATED_MARLIN_GATE_MARKER}\n"
            f"{MLP_K512_SHAPE_SEPARATED_MARLIN_DOWN_MARKER}\n"
            f"{MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY}\n"
            f"{MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY}\n"
            f"{MLP_K512_M128N64_SAME_CTA_PRIMARY}\n"
            f"{MLP_K512_M128N64_SAME_CTA_SECONDARY}\n"
            f"{MLP_K512_M128N512_FUSED_QUANTIZE_MARKER}\n"
            f"{MLP_K512_PRODUCT_QUANTIZE_MARKER}\n"
            f"{MLP_K512_EDGE_M128N64_MARKER}\n"
            f"{MLP_K512_DOWN_M16N64_V2_MARKER}\n"
            + "".join(f"{marker}\n" for marker in ATTENTION_K256_MARKERS)
            + "".join(
                f"{marker}\n" for marker in ATTENTION_K256_A_EXCHANGE_B4_MARKERS
            )
            + "".join(
                f"{marker}\n"
                for marker in ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS
            )
            + f"{HYBRID_INPUT_MARKER}\n"
            + f"{HYBRID_GATE_MARKER}\n"
            + f"{HYBRID_CANONICAL_DOWN_MARKER}\n"
            + f"{HYBRID_PAIRRING_DOWN_MARKER}\n"
            + f"{DOWN_16WARP_PAIRRING_MARKER}\n"
            + f"{PROJECTION_MAJOR_INPUT_MARKER}\n"
            + f"{PROJECTION_MAJOR_GATE_MARKER}\n"
            + f"{PROJECTION_MAJOR_DOWN_16WARP_MARKER}\n"
            + f"{PAIRED_WARP_GATE_MARKER}\n"
            + f"{PAIRED_WARP_DOWN_16WARP_MARKER}\n"
            + "".join(f"{marker}\n" for marker in K256_PAIRFEED_PACKAGE_MARKERS)
            + f"{GDN_PROMPT_SPAN_MACRO_MARKER}\n"
            + "# Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION\n"
            + "exit 0\n",
            encoding="utf-8",
        )
        self.server.chmod(0o755)
        self.payload.write_bytes(b"host-only payload fixture\n")
        self.policy.write_text("{}\n", encoding="utf-8")
        self.receipt.write_text("{}\n", encoding="utf-8")
        with self.k256_payload.open("wb") as stream:
            stream.truncate(ATTENTION_K256_PAYLOAD_BYTES)
        self.k256_policy.write_text("{}\n", encoding="utf-8")
        k256_policy_sha = hashlib.sha256(b"{}\n").hexdigest()
        zero_sha = "0" * 64
        one_sha = "1" * 64
        self.k256_receipt.write_text(
            json.dumps(
                {
                    "schema": "q3x.prefill.a4.publication-receipt",
                    "version": {"major": 3, "minor": 0},
                    "mode": "production_calibrated",
                    "production_residency_eligible": True,
                    "sidecar_kind": "a4_k256",
                    "packed_k_group_size": 64,
                    "scale_group_size": 256,
                    "physical_layout": ATTENTION_K256_LAYOUT,
                    "source_checkpoint_id": "host-checkpoint",
                    "source_config_sha256": "a" * 64,
                    "source_index_sha256": "b" * 64,
                    "payload_bytes": ATTENTION_K256_PAYLOAD_BYTES,
                    "projection_count": 400,
                    "manifest_sha256": one_sha,
                    "policy_sha256": k256_policy_sha,
                    "policy_bytes": len(b"{}\n"),
                    "payload_sha256": zero_sha,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.k512_payload.write_bytes(b"host-only K512 payload fixture\n")
        self.k512_policy.write_text("{}\n", encoding="utf-8")
        self.k512_receipt.write_text("{}\n", encoding="utf-8")
        self.mlp_k512_payload.write_bytes(
            b"host-only MLP K512 payload fixture\n"
        )
        self.mlp_k512_policy.write_text("{}\n", encoding="utf-8")
        self.mlp_k512_receipt.write_text(
            json.dumps(
                {
                    "schema": (
                        "q3x.prefill.mlp-k512.publication-receipt"
                    ),
                    "physical_layout": (
                        "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
                    ),
                    "payload_bytes": 8_623_226_880,
                    "projection_count": 192,
                    "manifest_sha256": "2" * 64,
                    "policy_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                    "required_base": {
                        "sidecar_kind": "a4_k256",
                        "physical_layout": ATTENTION_K256_LAYOUT,
                        "manifest_sha256": one_sha,
                        "policy_sha256": k256_policy_sha,
                        "payload_sha256": zero_sha,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.fragment_native_payload.open("wb") as stream:
            stream.truncate(FRAGMENT_NATIVE_PAYLOAD_BYTES)
        self.fragment_native_policy.write_text("{}\n", encoding="utf-8")
        self.fragment_native_receipt.write_text(
            json.dumps(
                {
                    "schema": (
                        "q3x.prefill.mlp-k512.fragment-native."
                        "publication-receipt"
                    ),
                    "physical_layout": (
                        "sm87_s4_gateup_n64_paired_down_n128_"
                        "fragment_native_scale_k512_mlp_v2"
                    ),
                    "payload_bytes": FRAGMENT_NATIVE_PAYLOAD_BYTES,
                    "payload_sha256": zero_sha,
                    "source_v1": {"policy_sha256": zero_sha},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.hybrid_payload.open("wb") as stream:
            stream.truncate(HYBRID_PAYLOAD_BYTES)
        self.hybrid_policy.write_text("{}\n", encoding="utf-8")
        hybrid_policy_sha = hashlib.sha256(b"{}\n").hexdigest()
        self.hybrid_receipt.write_text(
            json.dumps(
                {
                    "schema": (
                        "q3x.prefill.mlp-k512.paired-gateup-canonical-down."
                        "publication-receipt"
                    ),
                    "version": {"major": 1, "minor": 0},
                    "mode": "lossless_gateup_permutation_down_passthrough",
                    "production_residency_eligible": True,
                    "physical_layout": HYBRID_LAYOUT,
                    "gateup_physical_layout": (
                        "sm87_s4_gateup_n64_paired_fragment_register_v1"
                    ),
                    "down_physical_layout": (
                        "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
                    ),
                    "source_checkpoint_id": "host-checkpoint",
                    "source_config_sha256": "a" * 64,
                    "source_index_sha256": "b" * 64,
                    "required_base": {
                        "sidecar_kind": "a4_k256",
                        "physical_layout": ATTENTION_K256_LAYOUT,
                        "manifest_sha256": one_sha,
                        "policy_sha256": k256_policy_sha,
                        "payload_sha256": zero_sha,
                    },
                    "source_v1": {
                        "physical_layout": (
                            "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
                        ),
                        "receipt_sha256": "c" * 64,
                        "manifest_sha256": "d" * 64,
                        "policy_sha256": hybrid_policy_sha,
                        "policy_bytes": len(b"{}\n"),
                        "payload_sha256": "e" * 64,
                        "payload_bytes": HYBRID_PAYLOAD_BYTES,
                    },
                    "manifest_sha256": "f" * 64,
                    "payload_sha256": zero_sha,
                    "payload_bytes": HYBRID_PAYLOAD_BYTES,
                    "layer_count": 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.projection_major_payload.open("wb") as stream:
            stream.truncate(PROJECTION_MAJOR_PAYLOAD_BYTES)
        self.projection_major_policy.write_text("{}\n", encoding="utf-8")
        projection_major_policy_sha = hashlib.sha256(b"{}\n").hexdigest()
        self.projection_major_receipt.write_text(
            json.dumps(
                {
                    "schema": (
                        "q3x.prefill.mlp-k512.projection-major-gateup-"
                        "canonical-down.publication-receipt"
                    ),
                    "version": {"major": 2, "minor": 0},
                    "mode": (
                        "lossless_projection_major_gateup_permutation_"
                        "down_passthrough"
                    ),
                    "production_residency_eligible": True,
                    "physical_layout": PROJECTION_MAJOR_LAYOUT,
                    "gateup_physical_layout": (
                        "sm87_s4_gateup_n64_projection_major_fragment_"
                        "register_v3"
                    ),
                    "down_physical_layout": (
                        "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
                    ),
                    "source_checkpoint_id": "host-checkpoint",
                    "source_config_sha256": "a" * 64,
                    "source_index_sha256": "b" * 64,
                    "required_base": {
                        "sidecar_kind": "a4_k256",
                        "physical_layout": ATTENTION_K256_LAYOUT,
                        "manifest_sha256": one_sha,
                        "policy_sha256": k256_policy_sha,
                        "payload_sha256": zero_sha,
                    },
                    "source_v1": {
                        "physical_layout": (
                            "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
                        ),
                        "receipt_sha256": "5" * 64,
                        "manifest_sha256": "6" * 64,
                        "policy_sha256": projection_major_policy_sha,
                        "policy_bytes": len(b"{}\n"),
                        "payload_sha256": "7" * 64,
                        "payload_bytes": PROJECTION_MAJOR_PAYLOAD_BYTES,
                    },
                    "manifest_sha256": "8" * 64,
                    "payload_sha256": zero_sha,
                    "payload_bytes": PROJECTION_MAJOR_PAYLOAD_BYTES,
                    "layer_count": 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        real_sha256sum = shutil.which("sha256sum")
        if real_sha256sum is None:
            raise RuntimeError("sha256sum is required by the harness tests")
        fake_sha256sum = self.fake_bin / "sha256sum"
        fake_sha256sum.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *weights-a4-k256.bin)\n"
            "    printf '%064d  %s\\n' 0 \"$1\"\n"
            "    exit 0\n"
            "    ;;\n"
            "  *mlp-k512-fragment-native*)\n"
            "    printf '%064d  %s\\n' 0 \"$1\"\n"
            "    exit 0\n"
            "    ;;\n"
            "  *mlp-k512-paired-gateup-canonical-down.bin)\n"
            "    printf '%064d  %s\\n' 0 \"$1\"\n"
            "    exit 0\n"
            "    ;;\n"
            "  *mlp-k512-projection-major-gateup-canonical-down.bin)\n"
            "    printf '%064d  %s\\n' 0 \"$1\"\n"
            "    exit 0\n"
            "    ;;\n"
            "esac\n"
            f"exec {shlex.quote(real_sha256sum)} \"$@\"\n",
            encoding="utf-8",
        )
        fake_sha256sum.chmod(0o755)
        for bucket in ("p512", "p1k", "p2k", "p4k"):
            (self.corpus_dir / f"q3x-sharegpt-prefill-{bucket}-5.jsonl").write_text(
                '{"host_only":true}\n', encoding="utf-8"
            )

    def command(
        self,
        *extra: str,
        bucket: str = "p2k",
        prefill_payload: pathlib.Path | None = None,
        prefill_policy: pathlib.Path | None = None,
        prefill_receipt: pathlib.Path | None = None,
    ) -> list[str]:
        return [
            str(RUNNER),
            "--dry-run",
            "--prefill-a4-payload",
            str(prefill_payload or self.payload),
            "--prefill-a4-policy",
            str(prefill_policy or self.policy),
            "--prefill-a4-receipt",
            str(prefill_receipt or self.receipt),
            *extra,
            str(self.server),
            str(self.model_dir),
            str(self.corpus_dir),
            str(self.output),
            bucket,
        ]

    def run(
        self,
        *extra: str,
        bucket: str = "p2k",
        prefill_payload: pathlib.Path | None = None,
        prefill_policy: pathlib.Path | None = None,
        prefill_receipt: pathlib.Path | None = None,
        eval_number: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        if eval_number is None:
            environment.pop("Q3X_EVAL_NUMBER", None)
        else:
            environment["Q3X_EVAL_NUMBER"] = eval_number
        environment["Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION"] = "1"
        environment["Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION"] = "1"
        environment["Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION"] = "1"
        environment[GDN_PROMPT_SPAN_MACRO_SELECTOR] = "1"
        environment["Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION"] = "1"
        environment["Q3X_FULL_ATTENTION_FLASHINFER_DIRECT"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION"] = "1"
        environment[
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION"
        ] = "1"
        environment["Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION"] = "1"
        environment[
            "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION"
        ] = "1"
        environment[ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR] = "1"
        environment["Q3X_RUN_A4W4_MLP_K512_ADMISSION"] = "1"
        environment[
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "ALTERNATING_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "LDMATRIX_PAIRFEED_ADMISSION"
        ] = "1"
        environment[MLP_K512_M32N512_OWNER_SELECTOR] = "1"
        environment[MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR] = "1"
        environment[MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR] = "1"
        environment[MLP_K512_M128N64_SAME_CTA_SELECTOR] = "1"
        environment[MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR] = "1"
        environment[
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_"
            "LDMATRIX_ADMISSION"
        ] = "1"
        environment[MLP_K512_PROJECTION_MAJOR_SELECTOR] = "1"
        environment[MLP_K512_REGISTER_PIPELINE_SELECTOR] = "1"
        environment[MLP_K512_PAIRED_WARP_SELECTOR] = "1"
        environment[K256_PAIRFEED_PACKAGE_SELECTOR] = "1"
        environment[
            "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION"
        ] = "1"
        environment[
            "Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION"
        ] = "1"
        environment["Q3X_GDN_CHUNK64_PROFILE_CANDIDATE"] = "1"
        environment["PATH"] = (
            f"{self.fake_bin}{os.pathsep}{environment['PATH']}"
        )
        return subprocess.run(
            self.command(
                *extra,
                bucket=bucket,
                prefill_payload=prefill_payload,
                prefill_policy=prefill_policy,
                prefill_receipt=prefill_receipt,
            ),
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )

    def run_k512(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-attention-o-k512-payload",
            str(self.k512_payload),
            "--prefill-attention-o-k512-policy",
            str(self.k512_policy),
            "--prefill-attention-o-k512-receipt",
            str(self.k512_receipt),
            "--mode",
            "cumulative-prefill-current-best-k512",
            *extra,
        )

    def run_mlp_k512(
        self, mode: str, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-mlp-k512-payload",
            str(self.mlp_k512_payload),
            "--prefill-mlp-k512-policy",
            str(self.mlp_k512_policy),
            "--prefill-mlp-k512-receipt",
            str(self.mlp_k512_receipt),
            "--mode",
            mode,
            *extra,
        )

    def run_mlp_k512_edge(
        self, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run_mlp_k512(MLP_K512_EDGE_MODE, *extra)

    def run_attention_k256(
        self,
        *,
        bucket: str = "p2k",
        mode: str = ATTENTION_K256_MODE,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-mlp-k512-payload",
            str(self.mlp_k512_payload),
            "--prefill-mlp-k512-policy",
            str(self.mlp_k512_policy),
            "--prefill-mlp-k512-receipt",
            str(self.mlp_k512_receipt),
            "--mode",
            mode,
            bucket=bucket,
            prefill_payload=self.k256_payload,
            prefill_policy=self.k256_policy,
            prefill_receipt=self.k256_receipt,
        )

    def run_attention_k256_alternating(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket, mode=ATTENTION_K256_ALTERNATING_MODE
        )

    def run_attention_k256_alternating_down_pairring(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_ALTERNATING_DOWN_PAIRRING_MODE,
        )

    def run_attention_k256_alternating_down_16warp_pairring(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE,
        )

    def run_attention_k256_a_exchange_b4(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_A_EXCHANGE_B4_MODE,
        )

    def run_attention_k256_ldmatrix_pairfeed(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_LDMATRIX_PAIRFEED_MODE,
        )

    def run_k256_pairfeed_package(
        self, *extra: str, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--mode",
            K256_PAIRFEED_PACKAGE_MODE,
            *extra,
            bucket=bucket,
            prefill_payload=self.k256_payload,
            prefill_policy=self.k256_policy,
            prefill_receipt=self.k256_receipt,
        )

    def run_attention_k256_m32n512_owner(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_M32N512_OWNER_MODE,
        )

    def run_attention_k256_shape_separated_marlin(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_SHAPE_SEPARATED_MARLIN_MODE,
        )

    def run_attention_k256_m128n128_a_exchange_b3(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_M128N128_A_EXCHANGE_B3_MODE,
        )

    def run_attention_k256_m128n128_projection_serial(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_M128N128_PROJECTION_SERIAL_MODE,
        )

    def run_attention_k256_m128n64_same_cta(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_M128N64_SAME_CTA_MODE,
        )

    def run_attention_k256_m128n512_fused_quantize(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=ATTENTION_K256_M128N512_FUSED_QUANTIZE_MODE,
        )

    def run_gdn_prompt_span_macro(
        self, *, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run_attention_k256(
            bucket=bucket,
            mode=GDN_PROMPT_SPAN_MACRO_MODE,
        )

    def run_mlp_k512_edge_m128n64(
        self, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run_mlp_k512(MLP_K512_EDGE_M128N64_MODE, *extra)

    def run_hybrid(
        self,
        *,
        down_pairring: bool = False,
        bucket: str = "p2k",
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            str(self.hybrid_payload),
            "--prefill-mlp-k512-paired-gateup-canonical-down-policy",
            str(self.hybrid_policy),
            "--prefill-mlp-k512-paired-gateup-canonical-down-receipt",
            str(self.hybrid_receipt),
            "--mode",
            (
                HYBRID_GATE_DOWN_PAIRRING_MODE
                if down_pairring
                else HYBRID_GATE_MODE
            ),
            bucket=bucket,
            prefill_payload=self.k256_payload,
            prefill_policy=self.k256_policy,
            prefill_receipt=self.k256_receipt,
        )

    def run_projection_major(
        self, *extra: str, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-payload",
            str(self.projection_major_payload),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-policy",
            str(self.projection_major_policy),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-receipt",
            str(self.projection_major_receipt),
            "--mode",
            PROJECTION_MAJOR_MODE,
            *extra,
            bucket=bucket,
            prefill_payload=self.k256_payload,
            prefill_policy=self.k256_policy,
            prefill_receipt=self.k256_receipt,
        )

    def run_paired_warp(
        self, *extra: str, bucket: str = "p2k"
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            str(self.hybrid_payload),
            "--prefill-mlp-k512-paired-gateup-canonical-down-policy",
            str(self.hybrid_policy),
            "--prefill-mlp-k512-paired-gateup-canonical-down-receipt",
            str(self.hybrid_receipt),
            "--mode",
            PAIRED_WARP_MODE,
            *extra,
            bucket=bucket,
            prefill_payload=self.k256_payload,
            prefill_policy=self.k256_policy,
            prefill_receipt=self.k256_receipt,
        )

    def enable_fragment_native_m64n128_1cta(
        self, *extra_stage_markers: str
    ) -> None:
        with self.server.open("a", encoding="utf-8") as stream:
            stream.write("Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION\n")
            stream.write(f"{FRAGMENT_NATIVE_M64N128_1CTA_PRIMARY}\n")
            stream.write(f"{FRAGMENT_NATIVE_M64N128_1CTA_SECONDARY}\n")
            stream.write(
                "prefill_projection_span_mlp_k512_fragment_native_down\n"
            )
            for marker in extra_stage_markers:
                stream.write(f"{marker}\n")

    def run_fragment_native_m64n128_1cta(
        self,
        *extra: str,
        extra_stage_markers: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        self.enable_fragment_native_m64n128_1cta(*extra_stage_markers)
        return self.run(
            "--prefill-mlp-k512-fragment-native-payload",
            str(self.fragment_native_payload),
            "--prefill-mlp-k512-fragment-native-policy",
            str(self.fragment_native_policy),
            "--prefill-mlp-k512-fragment-native-receipt",
            str(self.fragment_native_receipt),
            "--mode",
            FRAGMENT_NATIVE_M64N128_1CTA_MODE,
            *extra,
        )

    def enable_fragment_native_m128n64_staged(
        self, *extra_stage_markers: str
    ) -> None:
        with self.server.open("a", encoding="utf-8") as stream:
            stream.write("Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION\n")
            stream.write(f"{FRAGMENT_NATIVE_M128N64_STAGED_PRIMARY}\n")
            stream.write(f"{FRAGMENT_NATIVE_M128N64_STAGED_SECONDARY}\n")
            stream.write(
                "prefill_projection_span_mlp_k512_fragment_native_down\n"
            )
            for marker in extra_stage_markers:
                stream.write(f"{marker}\n")

    def run_fragment_native_m128n64_staged(
        self,
        *extra: str,
        extra_stage_markers: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        self.enable_fragment_native_m128n64_staged(*extra_stage_markers)
        return self.run(
            "--prefill-mlp-k512-fragment-native-payload",
            str(self.fragment_native_payload),
            "--prefill-mlp-k512-fragment-native-policy",
            str(self.fragment_native_policy),
            "--prefill-mlp-k512-fragment-native-receipt",
            str(self.fragment_native_receipt),
            "--mode",
            FRAGMENT_NATIVE_M128N64_STAGED_MODE,
            *extra,
        )


class PurePrefillEvalScopeHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(pathlib.Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shell_syntax_and_exact_command_contract(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=exact dry_run=1", result.stdout)
        self.assertIn("--prefill-a4-payload", result.stdout)
        self.assertIn(str(self.fixture.payload), result.stdout)
        self.assertIn("--prefill-a4-policy", result.stdout)
        self.assertIn("--prefill-a4-receipt", result.stdout)
        self.assertIn("-u Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION", result.stdout)
        self.assertIn("-u Q3X_GDN_CHUNK64_PROFILE_CANDIDATE", result.stdout)
        self.assertIn(
            "-u Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION", result.stdout
        )
        self.assertIn("-u Q3X_FULL_ATTENTION_FLASHINFER_DIRECT", result.stdout)
        self.assertIn(
            "-u Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION", result.stdout
        )
        self.assertIn(
            "-u Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
            result.stdout,
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION", result.stdout
        )
        self.assertNotIn("Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1", result.stdout)
        self.assertNotIn(
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1", result.stdout
        )
        self.assertNotIn(
            "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION=1", result.stdout
        )
        self.assertNotIn(
            "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION=1", result.stdout
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_eval_number_defaults_to_four(self) -> None:
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("eval_number=4", result.stdout)
        self.assertIn(
            "evalscope_request_plan measured=4 warmup=1 "
            "result_leaf=parallel_1_number_4",
            result.stdout,
        )

    def test_eval_number_one_is_accepted_for_direction_gate(self) -> None:
        result = self.fixture.run(eval_number="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("eval_number=1", result.stdout)
        self.assertIn(
            "evalscope_request_plan measured=1 warmup=1 "
            "result_leaf=parallel_1_number_1",
            result.stdout,
        )

    def test_eval_number_rejects_every_value_except_one_or_four(self) -> None:
        for value in ("", "0", "2", "04", "four"):
            with self.subTest(value=value):
                result = self.fixture.run(eval_number=value)
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "Q3X_EVAL_NUMBER must be exactly 1 or 4",
                    result.stderr,
                )
                self.assertFalse(self.fixture.output.exists())

    def test_evalscope_command_remains_external_openai_api(self) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "uvx --from 'evalscope[perf]==1.9.1' evalscope perf",
            contents,
        )
        self.assertIn("--model qwen3.6-27b-nvfp4 --api openai", contents)
        self.assertIn('/v1/completions"', contents)
        self.assertIn('--number "${eval_number}" --parallel 1', contents)
        self.assertIn("--warmup-num 1 --num-workers 1", contents)
        self.assertIn(
            'result_leaf="${run_dir}/${result_name}/'
            'parallel_1_number_${eval_number}"',
            contents,
        )
        self.assertIn(
            "printf 'evalscope_measured_requests=%s\\n'",
            contents,
        )

    def test_readiness_contract_allows_diagnostic_fields(self) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "prefill_chunk_size=512 .*readiness_route=/healthz", contents
        )

    def test_native_gdn_mode_adds_only_declared_selector(self) -> None:
        result = self.fixture.run("--mode", "native-gdn")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=native-gdn", result.stdout)
        self.assertIn("Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1", result.stdout)
        self.assertIn(
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1", result.stdout
        )

    def test_cumulative_prefill_mode_adds_only_declared_bundle(self) -> None:
        result = self.fixture.run("--mode", "cumulative-prefill")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=cumulative-prefill", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
            },
        )
        self.assertIn("-u Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION", startup)
        self.assertIn("-u Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION", startup)
        self.assertNotIn("Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION=1", startup)
        self.assertNotIn(
            "Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION=1", startup
        )

    def test_cumulative_down_mode_adds_only_retained_down_cell(self) -> None:
        result = self.fixture.run("--mode", "cumulative-prefill-down")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=cumulative-prefill-down", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION",
            },
        )
        self.assertIn("-u Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION", startup)
        self.assertIn("-u Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION", startup)
        self.assertNotIn("Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION=1", startup)
        self.assertNotIn("Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION=1", startup)

    def test_cumulative_down_requires_compiled_selector(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(
                "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION\n", ""
            ),
            encoding="utf-8",
        )
        result = self.fixture.run("--mode", "cumulative-prefill-down")
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the cumulative-prefill-down selector: "
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION",
            result.stderr,
        )

    def test_cumulative_attention_down_mode_is_exact_bundle(self) -> None:
        result = self.fixture.run(
            "--mode", "cumulative-prefill-attention-down"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            },
        )
        self.assertIn("-u Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION", startup)
        self.assertIn("-u Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION", startup)

    def test_cumulative_attention_down_requires_flashinfer_direct(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace("# Q3X_FULL_ATTENTION_FLASHINFER_DIRECT\n", ""),
            encoding="utf-8",
        )
        result = self.fixture.run(
            "--mode", "cumulative-prefill-attention-down"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the cumulative-prefill-attention-down "
            "selector: Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            result.stderr,
        )

    def test_cumulative_current_best_mode_is_exact_eight_selector_bundle(
        self,
    ) -> None:
        result = self.fixture.run(
            "--mode", "cumulative-prefill-current-best"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "mode=cumulative-prefill-current-best dry_run=1", result.stdout
        )
        self.assertIn("selector_count=8", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        expected = {
            "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
            "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
        }
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)), expected
        )
        metadata = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("selector_metadata")
        )
        self.assertIn("selector_count=8", metadata)
        self.assertEqual(
            set(re.findall(r"Q3X_[A-Z0-9_]+", metadata)), expected
        )
        for rejected in (
            "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
            "Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION",
            "Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION",
            "Q3X_GDN_CHUNK64_PROFILE_CANDIDATE",
        ):
            self.assertIn(f"-u {rejected}", startup)
            self.assertNotIn(f"{rejected}=1", startup)

    def test_cumulative_current_best_requires_every_new_selector(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        new_selectors = (
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
        )
        for selector in new_selectors:
            with self.subTest(selector=selector):
                self.fixture.server.write_text(
                    contents.replace(f"# {selector}\n", ""), encoding="utf-8"
                )
                result = self.fixture.run(
                    "--mode", "cumulative-prefill-current-best"
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "server does not contain the "
                    f"cumulative-prefill-current-best selector: {selector}",
                    result.stderr,
                )
        self.fixture.server.write_text(contents, encoding="utf-8")

    def test_cumulative_current_best_k512_is_exact_nine_selector_bundle(
        self,
    ) -> None:
        result = self.fixture.run_k512()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "mode=cumulative-prefill-current-best-k512 dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=9", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        expected = {
            "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
            "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION",
        }
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)), expected
        )
        self.assertIn("--prefill-attention-o-k512-payload", startup)
        self.assertIn(str(self.fixture.k512_payload), startup)
        self.assertIn("--prefill-attention-o-k512-policy", startup)
        self.assertIn(str(self.fixture.k512_policy), startup)
        self.assertIn("--prefill-attention-o-k512-receipt", startup)
        self.assertIn(str(self.fixture.k512_receipt), startup)
        self.assertIn(
            "prefill_attention_o_k512_authenticated_64_of_64",
            result.stdout,
        )
        self.assertIn(
            "prefill_attention_o_k512_payload_sha256", result.stdout
        )

    def test_cumulative_current_best_k512_requires_all_overlay_paths(
        self,
    ) -> None:
        result = self.fixture.run(
            "--prefill-attention-o-k512-payload",
            str(self.fixture.k512_payload),
            "--prefill-attention-o-k512-policy",
            str(self.fixture.k512_policy),
            "--mode",
            "cumulative-prefill-current-best-k512",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "missing required Prefill Attention-O K512 receipt", result.stderr
        )
        self.assertFalse(self.fixture.output.exists())

    def test_cumulative_current_best_k512_requires_compiled_selector(
        self,
    ) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(
                "# Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION\n", ""
            ),
            encoding="utf-8",
        )
        result = self.fixture.run_k512()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the "
            "cumulative-prefill-current-best-k512 selector: "
            "Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION",
            result.stderr,
        )

    def test_k512_readiness_contract_proves_overlay_and_payload_digest(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "prefill_attention_o_k512_requested=1 .*"
            "prefill_attention_o_k512_enabled=1 .*"
            "prefill_attention_o_k512_projections=64",
            contents,
        )
        self.assertIn(
            "prefill_attention_o_k512_payload_sha256=[0-9a-f]{64}",
            contents,
        )

    def test_attention_k256_mode_uses_exact_authenticated_bundle(self) -> None:
        result = self.fixture.run_attention_k256()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"mode={ATTENTION_K256_MODE} dry_run=1", result.stdout)
        self.assertIn("selector_count=9", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
                "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_ADMISSION",
                "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION",
            },
        )
        self.assertIn(str(self.fixture.k256_payload), startup)
        self.assertIn(str(self.fixture.k256_policy), startup)
        self.assertIn(str(self.fixture.k256_receipt), startup)
        self.assertIn("-u Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION", startup)
        self.assertIn("-u Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION", startup)
        self.assertNotIn("Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1", startup)
        self.assertNotIn("Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION=1", startup)
        self.assertIn(
            f"layout={ATTENTION_K256_LAYOUT} "
            f"payload_bytes={ATTENTION_K256_PAYLOAD_BYTES}",
            result.stdout,
        )
        self.assertIn("expected_launch_hits=128", result.stdout)
        self.assertIn("expected_logical_projections=208", result.stdout)
        for marker in ATTENTION_K256_MARKERS:
            self.assertIn(marker, result.stdout)

    def test_attention_k256_mode_admits_external_p512_bucket(self) -> None:
        result = self.fixture.run_attention_k256(bucket="p512")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "pure_prefill_corpus bucket=p512", result.stdout
        )
        self.assertIn(
            str(
                self.fixture.corpus_dir
                / "q3x-sharegpt-prefill-p512-5.jsonl"
            ),
            result.stdout,
        )

    def test_attention_k256_alternating_mode_replaces_only_gateup_edge(
        self,
    ) -> None:
        result = self.fixture.run_attention_k256_alternating()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_ALTERNATING_MODE} dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=9", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
                "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_ADMISSION",
                (
                    "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
                    "ALTERNATING_ADMISSION"
                ),
            },
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1", startup
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION", startup
        )
        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER in line
        )
        self.assertIn(
            f"required={MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER}",
            stage_contract,
        )
        self.assertIn(f"excluded={MLP_K512_EDGE_MARKER},", stage_contract)
        self.assertIn(
            "retained=prefill_projection_span_mlp_k512_input_quantize,"
            "prefill_projection_span_mlp_k512_down",
            stage_contract,
        )
        self.assertIn("expected_request_launch_hits=64", stage_contract)
        self.assertIn(
            "gateup_alternating_launch_hits_64_per_request", result.stdout
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_attention_k256_alternating_mode_requires_selector(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        selector = (
            "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "ALTERNATING_ADMISSION\n"
        )
        self.fixture.server.write_text(
            contents.replace(selector, ""), encoding="utf-8"
        )
        result = self.fixture.run_attention_k256_alternating()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_ALTERNATING_MODE} selector: "
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "ALTERNATING_ADMISSION",
            result.stderr,
        )

    def test_attention_k256_alternating_down_pairring_uses_v1_triplet(
        self,
    ) -> None:
        result = self.fixture.run_attention_k256_alternating_down_pairring()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_ALTERNATING_DOWN_PAIRRING_MODE} "
            "dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=10", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        selectors = set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup))
        self.assertIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_"
            "ALTERNATING_ADMISSION",
            selectors,
        )
        self.assertIn(
            "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION",
            selectors,
        )
        self.assertIn("--prefill-mlp-k512-payload", startup)
        self.assertIn(str(self.fixture.mlp_k512_payload), startup)
        self.assertNotIn(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            startup,
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_"
            "ADMISSION=1",
            startup,
        )
        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER in line
            and HYBRID_PAIRRING_DOWN_MARKER in line
        )
        self.assertIn(
            f"required={MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER},"
            f"{HYBRID_PAIRRING_DOWN_MARKER}",
            stage_contract,
        )
        self.assertIn(
            "excluded=prefill_projection_span_mlp_k512_gateup_down_edge,",
            stage_contract,
        )
        self.assertIn(
            "prefill_projection_span_mlp_k512_down,", stage_contract
        )
        self.assertIn(
            "retained=prefill_projection_span_mlp_k512_input_quantize",
            stage_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate:64,down:64",
            stage_contract,
        )
        self.assertIn(
            "down_m128n128_ldmatrix_pairring_launch_hits_64_per_request",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_attention_k256_alternating_down_16warp_pairring_is_one_selector_delta(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE,
            help_result.stderr,
        )

        baseline = self.fixture.run_attention_k256_alternating_down_pairring()
        candidate = (
            self.fixture.run_attention_k256_alternating_down_16warp_pairring()
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE} "
            "dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {DOWN_PAIRRING_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {DOWN_16WARP_PAIRRING_SELECTOR},
        )
        self.assertIn(f"-u {DOWN_PAIRRING_SELECTOR}", candidate_startup)
        self.assertIn(
            f"-u {DOWN_16WARP_PAIRRING_SELECTOR}", candidate_startup
        )
        self.assertNotIn(f"{DOWN_PAIRRING_SELECTOR}=1", candidate_startup)
        self.assertIn(
            f"{DOWN_16WARP_PAIRRING_SELECTOR}=1", candidate_startup
        )

        deltas = [
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        ]
        self.assertEqual(len(deltas), 1, deltas)
        delta = deltas[0]
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_ALTERNATING_DOWN_PAIRRING_MODE}",
            delta,
        )
        self.assertIn(f"removed_selector={DOWN_PAIRRING_SELECTOR}", delta)
        self.assertIn(
            f"added_selector={DOWN_16WARP_PAIRRING_SELECTOR}", delta
        )

        stage_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and DOWN_16WARP_PAIRRING_MARKER in line
        )
        self.assertIn(
            f"required={MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER},"
            f"{DOWN_16WARP_PAIRRING_MARKER}",
            stage_contract,
        )
        self.assertIn(HYBRID_PAIRRING_DOWN_MARKER, stage_contract)
        self.assertIn(
            "expected_request_launch_hits=gate:64,down_incumbent:0,"
            "down_candidate:64",
            stage_contract,
        )
        startup_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("startup_contract")
        )
        self.assertIn(DOWN_16WARP_PAIRRING_MARKER, startup_contract)
        self.assertIn(
            "down_m128n128_ldmatrix_pairring_launch_hits_0_per_request",
            startup_contract,
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_launch_hits_64_per_request",
            startup_contract,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_attention_k256_alternating_down_16warp_pairring_fails_closed(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(f"# {DOWN_16WARP_PAIRRING_SELECTOR}\n", ""),
            encoding="utf-8",
        )
        missing_selector = (
            self.fixture.run_attention_k256_alternating_down_16warp_pairring()
        )
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE} "
            f"selector: {DOWN_16WARP_PAIRRING_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(f"{DOWN_16WARP_PAIRRING_MARKER}\n", ""),
            encoding="utf-8",
        )
        missing_marker = (
            self.fixture.run_attention_k256_alternating_down_16warp_pairring()
        )
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the M128N128 16-warp pair-ring Down "
            f"stage: {DOWN_16WARP_PAIRRING_MARKER}",
            missing_marker.stderr,
        )

    def test_attention_k256_alternating_down_16warp_pairring_records_runtime_proof(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "down_m128n128_ldmatrix_pairring_expected_hits=0", contents
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_expected_hits=64", contents
        )
        self.assertIn(
            "down_m128n128_ldmatrix_pairring_launch_hits="
            "${down_m128n128_ldmatrix_pairring_expected_hits}",
            contents,
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_launch_hits="
            "${down_m128n128_16warp_pairring_expected_hits}",
            contents,
        )
        self.assertIn(
            "v1_16warp_pairring_down_runtime_contract bucket=%s requests=%s "
            "incumbent_launch_hits_per_request=%s "
            "candidate_launch_hits_per_request=%s status=passed",
            contents,
        )
        self.assertIn(
            "experiment_baseline_mode="
            f"{ATTENTION_K256_ALTERNATING_DOWN_PAIRRING_MODE}",
            contents,
        )
        self.assertIn(
            f"experiment_removed_selector={DOWN_PAIRRING_SELECTOR}",
            contents,
        )
        self.assertIn(
            f"experiment_added_selector={DOWN_16WARP_PAIRRING_SELECTOR}",
            contents,
        )
        self.assertIn(
            f"required_runtime_stage={DOWN_16WARP_PAIRRING_MARKER}",
            contents,
        )

    def test_attention_k256_a_exchange_b4_is_one_attention_selector_delta(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(ATTENTION_K256_A_EXCHANGE_B4_MODE, help_result.stderr)

        baseline = (
            self.fixture.run_attention_k256_alternating_down_16warp_pairring()
        )
        candidate = self.fixture.run_attention_k256_a_exchange_b4()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_A_EXCHANGE_B4_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {ATTENTION_K256_INCUMBENT_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {ATTENTION_K256_A_EXCHANGE_B4_SELECTOR},
        )
        self.assertIn(
            f"-u {ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            baseline_startup,
        )
        self.assertNotIn(
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}=1", baseline_startup
        )
        self.assertIn(
            f"-u {ATTENTION_K256_INCUMBENT_SELECTOR}", candidate_startup
        )
        self.assertIn(
            f"-u {ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            candidate_startup,
        )
        self.assertNotIn(
            f"{ATTENTION_K256_INCUMBENT_SELECTOR}=1", candidate_startup
        )
        self.assertIn(
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}=1",
            candidate_startup,
        )
        self.assertIn(
            f"{DOWN_16WARP_PAIRRING_SELECTOR}=1", candidate_startup
        )

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            "baseline_mode="
            f"{ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE}",
            delta,
        )
        self.assertIn(
            f"removed_selector={ATTENTION_K256_INCUMBENT_SELECTOR}", delta
        )
        self.assertIn(
            f"added_selector={ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            delta,
        )

        attention_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and ATTENTION_K256_A_EXCHANGE_B4_MARKERS[0] in line
        )
        self.assertIn(
            f"required={','.join(ATTENTION_K256_A_EXCHANGE_B4_MARKERS)}",
            attention_contract,
        )
        self.assertIn(
            f"excluded={','.join(ATTENTION_K256_MARKERS)},",
            attention_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=attention_incumbent:0,"
            "attention_candidate:128",
            attention_contract,
        )
        self.assertIn(
            "expected_request_logical_projections=attention_incumbent:0,"
            "attention_candidate:208",
            attention_contract,
        )

        candidate_metadata = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("attention_k256_publication_metadata")
        )
        self.assertIn("incumbent_expected_launch_hits=0", candidate_metadata)
        self.assertIn(
            "incumbent_expected_logical_projections=0", candidate_metadata
        )
        self.assertIn(
            "a_exchange_b4_expected_launch_hits=128", candidate_metadata
        )
        self.assertIn(
            "a_exchange_b4_expected_logical_projections=208",
            candidate_metadata,
        )
        baseline_metadata = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("attention_k256_publication_metadata")
        )
        self.assertIn("incumbent_expected_launch_hits=128", baseline_metadata)
        self.assertIn(
            "incumbent_expected_logical_projections=208", baseline_metadata
        )
        self.assertIn(
            "a_exchange_b4_expected_launch_hits=0", baseline_metadata
        )
        self.assertIn(
            "a_exchange_b4_expected_logical_projections=0", baseline_metadata
        )

        down_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and DOWN_16WARP_PAIRRING_MARKER in line
        )
        self.assertIn(
            "expected_request_launch_hits=gate:64,down_incumbent:0,"
            "down_candidate:64",
            down_contract,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_k256_pairfeed_package_is_base_only_production_bundle(self) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(K256_PAIRFEED_PACKAGE_MODE, help_result.stderr)

        result = self.fixture.run_k256_pairfeed_package()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={K256_PAIRFEED_PACKAGE_MODE} dry_run=1", result.stdout
        )
        self.assertIn("selector_count=8", result.stdout)

        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        selectors = set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup))
        self.assertEqual(
            selectors,
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
                ATTENTION_K256_A_EXCHANGE_B4_SELECTOR,
                K256_PAIRFEED_PACKAGE_SELECTOR,
            },
        )
        self.assertNotIn("--prefill-mlp-k512-payload", startup)
        self.assertNotIn("--prefill-mlp-k512-policy", startup)
        self.assertNotIn("--prefill-mlp-k512-receipt", startup)
        self.assertIn(
            "prefill_mlp_k256_implementation=m128n256_pairfeed_package "
            "source_publication=base_a4_k256 overlay=none",
            result.stdout,
        )
        self.assertIn(
            "attention_k256_mlp_binding_metadata "
            "layout=base-a4-k256-only payload_bytes=0",
            result.stdout,
        )

        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and K256_PAIRFEED_PACKAGE_MARKERS[0] in line
        )
        self.assertIn(
            f"required={','.join(K256_PAIRFEED_PACKAGE_MARKERS)}",
            stage_contract,
        )
        for old_stage in (
            "prefill_projection_span_mlp_k512_input_quantize",
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER,
            DOWN_16WARP_PAIRRING_MARKER,
            MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER,
            MLP_K512_EDGE_MARKER,
            "prefill_projection_span_mlp_k512_down",
        ):
            self.assertIn(old_stage, stage_contract)
        self.assertIn(
            "expected_request_launch_hits=package:64,gate_alternating:0,"
            "gate_pairfeed:0,gate_projection_serial:0,gate_same_cta:0,"
            "gate_fused_quantize:0,gate_paired_ldmatrix:0,"
            "gate_projection_major:0,gate_paired_warp:0,down_ldmatrix:0,"
            "down_16warp:0",
            stage_contract,
        )
        startup_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("startup_contract")
        )
        self.assertIn(
            "mlp_k256_m128n256_pairfeed_package_launch_hits_64_per_request",
            startup_contract,
        )
        self.assertIn(
            "all_k512_mlp_launch_hits_0_per_request", startup_contract
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

        runner_contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            '" mlp_k256_m128n256_pairfeed_package_launch_hits='
            '${mlp_k256_pairfeed_package_expected_hits}([[:space:]]|$)"',
            runner_contents,
        )
        for old_counter in (
            "gateup_alternating_expected_hits=0",
            "gateup_ldmatrix_pairfeed_expected_hits=0",
            "gateup_m128n128_projection_serial_expected_hits=0",
            "gateup_m128n64_same_cta_expected_hits=0",
            "gateup_m128n512_fused_quantize_expected_hits=0",
            "gateup_m128n512_paired_ldmatrix_expected_hits=0",
            "gateup_m64n128_register_pipeline_expected_hits=0",
            "gateup_m64n8_paired_warp_register_pipeline_expected_hits=0",
            "down_m128n128_ldmatrix_pairring_expected_hits=0",
            "down_m128n128_16warp_pairring_expected_hits=0",
        ):
            self.assertIn(old_counter, runner_contents)

    def test_k256_pairfeed_package_rejects_overlay_and_missing_stage(self) -> None:
        overlay_contracts = (
            (
                "--prefill-mlp-k512-payload",
                self.fixture.mlp_k512_payload,
                "--prefill-mlp-k512-policy",
                self.fixture.mlp_k512_policy,
                "--prefill-mlp-k512-receipt",
                self.fixture.mlp_k512_receipt,
            ),
            (
                "--prefill-mlp-k512-fragment-native-payload",
                self.fixture.fragment_native_payload,
                "--prefill-mlp-k512-fragment-native-policy",
                self.fixture.fragment_native_policy,
                "--prefill-mlp-k512-fragment-native-receipt",
                self.fixture.fragment_native_receipt,
            ),
            (
                "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
                self.fixture.hybrid_payload,
                "--prefill-mlp-k512-paired-gateup-canonical-down-policy",
                self.fixture.hybrid_policy,
                "--prefill-mlp-k512-paired-gateup-canonical-down-receipt",
                self.fixture.hybrid_receipt,
            ),
            (
                "--prefill-mlp-k512-projection-major-gateup-canonical-down-payload",
                self.fixture.projection_major_payload,
                "--prefill-mlp-k512-projection-major-gateup-canonical-down-policy",
                self.fixture.projection_major_policy,
                "--prefill-mlp-k512-projection-major-gateup-canonical-down-receipt",
                self.fixture.projection_major_receipt,
            ),
        )
        for overlay_contract in overlay_contracts:
            with self.subTest(overlay=overlay_contract[0]):
                overlay = self.fixture.run_k256_pairfeed_package(
                    *(str(argument) for argument in overlay_contract)
                )
                self.assertEqual(overlay.returncode, 2)
                self.assertIn(
                    "K256 MLP package mode uses only the base A4 K256 "
                    "publication; K512 MLP overlay arguments are forbidden",
                    overlay.stderr,
                )

        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(f"{K256_PAIRFEED_PACKAGE_MARKERS[1]}\n", ""),
            encoding="utf-8",
        )
        missing_stage = self.fixture.run_k256_pairfeed_package()
        self.assertEqual(missing_stage.returncode, 2)
        self.assertIn(
            "server does not prove the K256 M128N256 pair-feed MLP package "
            f"stage: {K256_PAIRFEED_PACKAGE_MARKERS[1]}",
            missing_stage.stderr,
        )

    def test_ldmatrix_pairfeed_is_one_gate_selector_delta(self) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_LDMATRIX_PAIRFEED_MODE, help_result.stderr
        )

        baseline = self.fixture.run_attention_k256_a_exchange_b4()
        candidate = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR},
        )
        self.assertIn(
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}=1",
            candidate_startup,
        )
        self.assertNotIn(
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR}=1",
            candidate_startup,
        )

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_A_EXCHANGE_B4_MODE}", delta
        )
        self.assertIn(
            "removed_selector="
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR}",
            delta,
        )
        self.assertIn(
            "added_selector="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            delta,
        )

        gate_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER in line
        )
        self.assertIn(
            "required="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER},"
            f"{DOWN_16WARP_PAIRRING_MARKER}",
            gate_contract,
        )
        self.assertIn(
            "excluded="
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER},",
            gate_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate_incumbent:0,"
            "gate_candidate:64,down_incumbent:0,down_candidate:64",
            gate_contract,
        )
        self.assertIn(
            "gateup_alternating_launch_hits_64_per_request,"
            "gateup_ldmatrix_pairfeed_launch_hits_0_per_request",
            baseline.stdout,
        )
        self.assertIn(
            "gateup_alternating_launch_hits_0_per_request,"
            "gateup_ldmatrix_pairfeed_launch_hits_64_per_request",
            candidate.stdout,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_m32n512_owner_is_one_child_selector_delta(self) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(ATTENTION_K256_M32N512_OWNER_MODE, help_result.stderr)

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = self.fixture.run_attention_k256_m32n512_owner()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_M32N512_OWNER_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=11", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(baseline_selectors - candidate_selectors, set())
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_M32N512_OWNER_SELECTOR},
        )
        self.assertIn(
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}=1",
            candidate_startup,
        )
        self.assertIn(
            f"{MLP_K512_M32N512_OWNER_SELECTOR}=1", candidate_startup
        )

        deltas = [
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        ]
        self.assertEqual(len(deltas), 1, deltas)
        delta = deltas[0]
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}",
            delta,
        )
        self.assertIn(
            "retained_selectors="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR},"
            f"{DOWN_16WARP_PAIRRING_SELECTOR}",
            delta,
        )
        self.assertIn(
            f"added_selector={MLP_K512_M32N512_OWNER_SELECTOR}", delta
        )

        gate_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_M32N512_OWNER_MARKER in line
        )
        self.assertIn(
            "required="
            f"{MLP_K512_M32N512_OWNER_MARKER},"
            f"{DOWN_16WARP_PAIRRING_MARKER}",
            gate_contract,
        )
        self.assertIn(
            f"excluded={MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER},",
            gate_contract,
        )
        self.assertIn(
            "prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,",
            gate_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate_incumbent:0,"
            "gate_candidate:64,down_incumbent:0,down_candidate:64",
            gate_contract,
        )
        for launch_contract in (
            "gateup_alternating_launch_hits_0_per_request",
            "gateup_ldmatrix_pairfeed_launch_hits_64_per_request",
            "gateup_m128n128_projection_serial_launch_hits_0_per_request",
            "gateup_m128n64_same_cta_launch_hits_0_per_request",
            "gateup_m128n512_fused_quantize_launch_hits_0_per_request",
            "gateup_m128n512_paired_ldmatrix_launch_hits_0_per_request",
            "gateup_m64n128_register_pipeline_launch_hits_0_per_request",
            "gateup_m64n8_paired_warp_register_pipeline_launch_hits_0_per_request",
            "down_m128n128_ldmatrix_pairring_launch_hits_0_per_request",
            "down_m128n128_16warp_pairring_launch_hits_64_per_request",
        ):
            self.assertIn(launch_contract, candidate.stdout)
        runner_contents = RUNNER.read_text(encoding="utf-8")
        owner_accounting = re.search(
            r'elif \[\[ "\$\{mlp_k512_m32n512_owner_mode\}" == 1 '
            r'\]\]; then\n(?P<body>.*?)'
            r'elif \[\[ "\$\{mlp_k512_edge_m64n128_k256_'
            r'ldmatrix_pairfeed_mode\}" == 1 \]\]; then',
            runner_contents,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(owner_accounting)
        assert owner_accounting is not None
        for expected_assignment in (
            "gateup_alternating_expected_hits=0",
            "gateup_ldmatrix_pairfeed_expected_hits=64",
            "gateup_m128n128_projection_serial_expected_hits=0",
            "gateup_m128n64_same_cta_expected_hits=0",
            "gateup_m128n512_fused_quantize_expected_hits=0",
            "gateup_m128n512_paired_ldmatrix_expected_hits=0",
            "gateup_m64n128_register_pipeline_expected_hits=0",
            "gateup_m64n8_paired_warp_register_pipeline_expected_hits=0",
        ):
            self.assertIn(expected_assignment, owner_accounting.group("body"))
        self.assertIn(
            'if [[ -n "${gateup_alternating_expected_hits}" ||',
            runner_contents,
        )
        self.assertIn(
            '" gateup_ldmatrix_pairfeed_launch_hits='
            '${gateup_ldmatrix_pairfeed_expected_hits}',
            runner_contents,
        )
        self.assertIn(
            "prefill_mlp_k512_gateup_implementation="
            "m32n512_owner_k128_b4",
            candidate.stdout,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                f"# {MLP_K512_M32N512_OWNER_SELECTOR}\n", ""
            ),
            encoding="utf-8",
        )
        missing_selector = self.fixture.run_attention_k256_m32n512_owner()
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_M32N512_OWNER_MODE} selector: "
            f"{MLP_K512_M32N512_OWNER_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(f"{MLP_K512_M32N512_OWNER_MARKER}\n", ""),
            encoding="utf-8",
        )
        missing_marker = self.fixture.run_attention_k256_m32n512_owner()
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the M32N512 owner K128/B4 Gate+Up "
            f"production stage: {MLP_K512_M32N512_OWNER_MARKER}",
            missing_marker.stderr,
        )
        self.fixture.server.write_text(original, encoding="utf-8")

    def test_shape_separated_marlin_is_one_package_selector_delta(self) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_SHAPE_SEPARATED_MARLIN_MODE,
            help_result.stderr,
        )

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = self.fixture.run_attention_k256_shape_separated_marlin()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_SHAPE_SEPARATED_MARLIN_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=11", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(baseline_selectors - candidate_selectors, set())
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR},
        )
        for retained in (
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR,
            DOWN_16WARP_PAIRRING_SELECTOR,
        ):
            self.assertIn(f"{retained}=1", candidate_startup)

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}",
            delta,
        )
        self.assertIn(
            f"added_selector={MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR}",
            delta,
        )
        gate_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_SHAPE_SEPARATED_MARLIN_GATE_MARKER in line
        )
        self.assertIn(
            "required="
            f"{MLP_K512_SHAPE_SEPARATED_MARLIN_GATE_MARKER},"
            f"{MLP_K512_SHAPE_SEPARATED_MARLIN_DOWN_MARKER}",
            gate_contract,
        )
        self.assertIn(
            "excluded="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER},"
            f"{DOWN_16WARP_PAIRRING_MARKER},",
            gate_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate_incumbent:0,"
            "gate_candidate:64,down_incumbent:0,down_candidate:64",
            gate_contract,
        )
        for binding in (
            "prefill_mlp_k512_implementation=shape_separated_marlin",
            "prefill_mlp_k512_gateup_implementation=m64n256_marlin_k64_b3",
            "prefill_mlp_k512_down_implementation=m64n256_16warp_pairring",
            "gateup_ldmatrix_pairfeed_launch_hits_64_per_request",
            "down_m128n128_16warp_pairring_launch_hits_64_per_request",
            "performance_evidence=0",
        ):
            self.assertIn(binding, candidate.stdout)

        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                f"# {MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR}\n", ""
            ),
            encoding="utf-8",
        )
        missing_selector = (
            self.fixture.run_attention_k256_shape_separated_marlin()
        )
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            f"selector: {MLP_K512_SHAPE_SEPARATED_MARLIN_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(
                f"{MLP_K512_SHAPE_SEPARATED_MARLIN_DOWN_MARKER}\n", ""
            ),
            encoding="utf-8",
        )
        missing_marker = self.fixture.run_attention_k256_shape_separated_marlin()
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "shape-separated Marlin MLP package stage: "
            f"{MLP_K512_SHAPE_SEPARATED_MARLIN_DOWN_MARKER}",
            missing_marker.stderr,
        )
        self.fixture.server.write_text(original, encoding="utf-8")

    def test_attention_k256_m128n128_a_exchange_b3_is_one_modifier_delta(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_M128N128_A_EXCHANGE_B3_MODE,
            help_result.stderr,
        )

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = (
            self.fixture.run_attention_k256_m128n128_a_exchange_b3()
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_M128N128_A_EXCHANGE_B3_MODE} "
            "dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=11", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(baseline_selectors - candidate_selectors, set())
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR},
        )
        self.assertIn(
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}=1",
            baseline_startup,
        )
        self.assertIn(
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}=1",
            candidate_startup,
        )
        self.assertIn(
            f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR}=1",
            candidate_startup,
        )

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}",
            delta,
        )
        self.assertIn(
            f"retained_selector={ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            delta,
        )
        self.assertIn(
            "added_selector="
            f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR}",
            delta,
        )

        attention_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS[0] in line
        )
        self.assertIn(
            "required="
            f"{','.join(ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS)}",
            attention_contract,
        )
        self.assertIn(
            f"excluded={','.join(ATTENTION_K256_MARKERS)},",
            attention_contract,
        )
        self.assertIn(
            f",{','.join(ATTENTION_K256_A_EXCHANGE_B4_MARKERS)},",
            attention_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=attention_incumbent:0,"
            "attention_candidate:128",
            attention_contract,
        )
        self.assertIn(
            "expected_request_logical_projections=attention_incumbent:0,"
            "attention_candidate:208",
            attention_contract,
        )

        down_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and DOWN_16WARP_PAIRRING_MARKER in line
        )
        self.assertIn(
            "expected_request_launch_hits=gate_incumbent:0,"
            "gate_candidate:64,down_incumbent:0,down_candidate:64",
            down_contract,
        )
        self.assertIn(
            "prefill_attention_k256_implementation="
            "a_exchange_b3_m128n128",
            RUNNER.read_text(encoding="utf-8"),
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                f"# {ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR}\n",
                "",
            ),
            encoding="utf-8",
        )
        missing_selector = (
            self.fixture.run_attention_k256_m128n128_a_exchange_b3()
        )
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_MODE} selector: "
            f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(
                f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS[-1]}\n",
                "",
            ),
            encoding="utf-8",
        )
        missing_marker = (
            self.fixture.run_attention_k256_m128n128_a_exchange_b3()
        )
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the K256 Attention production stage: "
            f"{ATTENTION_K256_M128N128_A_EXCHANGE_B3_MARKERS[-1]}",
            missing_marker.stderr,
        )
        self.fixture.server.write_text(original, encoding="utf-8")

    def test_ldmatrix_pairfeed_fails_closed_on_selector_or_marker(self) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                f"# {MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}\n",
                "",
            ),
            encoding="utf-8",
        )
        missing_selector = (
            self.fixture.run_attention_k256_ldmatrix_pairfeed()
        )
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_LDMATRIX_PAIRFEED_MODE} selector: "
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(
                f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER}\n",
                "",
            ),
            encoding="utf-8",
        )
        missing_marker = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the LDSM pair-feed M64N128 K256 "
            "Gate+Up production stage: "
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER}",
            missing_marker.stderr,
        )

    def test_m128n128_projection_serial_replaces_only_pairfeed_gate(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_M128N128_PROJECTION_SERIAL_MODE,
            help_result.stderr,
        )

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = (
            self.fixture.run_attention_k256_m128n128_projection_serial()
        )
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_M128N128_PROJECTION_SERIAL_MODE} "
            "dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR},
        )
        self.assertNotIn(
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR}=1",
            candidate_startup,
        )
        self.assertNotIn(
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}=1",
            candidate_startup,
        )

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}",
            delta,
        )
        self.assertIn(
            "removed_selector="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            delta,
        )
        self.assertIn(
            "added_selector="
            f"{MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR}",
            delta,
        )

        contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY in line
        )
        for required in (
            MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY,
            MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY,
            MLP_K512_PRODUCT_QUANTIZE_MARKER,
            DOWN_16WARP_PAIRRING_MARKER,
        ):
            self.assertIn(required, contract)
        self.assertIn(
            f"excluded={MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER},",
            contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate_alternating:0,"
            "gate_pairfeed:0,gate_candidate:64,down_incumbent:0,"
            "down_candidate:64",
            contract,
        )
        self.assertIn(
            "gateup_m128n128_projection_serial_launch_hits_64_per_request",
            candidate.stdout,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_m128n128_projection_serial_fails_closed_and_accounts_requests(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        removals = (
            (
                f"# {MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR}\n",
                "server does not contain the "
                f"{ATTENTION_K256_M128N128_PROJECTION_SERIAL_MODE} "
                "selector: "
                f"{MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR}",
            ),
            (
                f"{MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY}\n",
                "server does not prove the M128N128 projection-serial "
                "Gate+Up production stage: "
                f"{MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY}",
            ),
            (
                f"{MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY}\n",
                "server does not prove the M128N128 projection-serial "
                "Gate+Up production stage: "
                f"{MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY}",
            ),
            (
                f"{MLP_K512_PRODUCT_QUANTIZE_MARKER}\n",
                "server does not prove the M128N128 projection-serial "
                "Gate+Up production stage: "
                f"{MLP_K512_PRODUCT_QUANTIZE_MARKER}",
            ),
        )
        for removed, error in removals:
            with self.subTest(removed=removed):
                self.fixture.server.write_text(
                    original.replace(removed, ""), encoding="utf-8"
                )
                result = (
                    self.fixture.run_attention_k256_m128n128_projection_serial()
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "gateup_m128n128_projection_serial_expected_hits=64", contents
        )
        self.assertIn(
            "gateup_m128n128_projection_serial_expected_hits=0", contents
        )
        self.assertIn(
            "gateup_m128n128_projection_serial_launch_hits="
            "${gateup_m128n128_projection_serial_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_m128n128_projection_serial_runtime_contract "
            "bucket=%s requests=%s launch_hits_per_request=%s status=passed",
            contents,
        )

    def test_m128n64_same_cta_replaces_only_pairfeed_gate(self) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(ATTENTION_K256_M128N64_SAME_CTA_MODE, help_result.stderr)

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = self.fixture.run_attention_k256_m128n64_same_cta()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_M128N64_SAME_CTA_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_M128N64_SAME_CTA_SELECTOR},
        )
        self.assertIn(f"{MLP_K512_M128N64_SAME_CTA_SELECTOR}=1", candidate_startup)

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}", delta
        )
        self.assertIn(
            f"removed_selector={MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            delta,
        )
        self.assertIn(
            f"added_selector={MLP_K512_M128N64_SAME_CTA_SELECTOR}", delta
        )

        contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_M128N64_SAME_CTA_PRIMARY in line
        )
        for required in (
            MLP_K512_M128N64_SAME_CTA_PRIMARY,
            MLP_K512_M128N64_SAME_CTA_SECONDARY,
            MLP_K512_PRODUCT_QUANTIZE_MARKER,
            DOWN_16WARP_PAIRRING_MARKER,
        ):
            self.assertIn(required, contract)
        self.assertIn(
            "gate_alternating:0,gate_pairfeed:0,gate_projection_serial:0,"
            "gate_same_cta:64,gate_fused_quantize:0,"
            "gate_paired_ldmatrix:0,gate_projection_major:0,"
            "gate_paired_warp:0",
            contract,
        )
        self.assertIn(
            "gateup_m128n64_same_cta_launch_hits_64_per_request",
            candidate.stdout,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_m128n64_same_cta_fails_closed_and_accounts_requests(self) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        removals = (
            (
                f"# {MLP_K512_M128N64_SAME_CTA_SELECTOR}\n",
                "server does not contain the "
                f"{ATTENTION_K256_M128N64_SAME_CTA_MODE} selector: "
                f"{MLP_K512_M128N64_SAME_CTA_SELECTOR}",
            ),
            (
                f"{MLP_K512_M128N64_SAME_CTA_PRIMARY}\n",
                "server does not prove the M128N64 same-CTA Gate+Up "
                f"production stage: {MLP_K512_M128N64_SAME_CTA_PRIMARY}",
            ),
            (
                f"{MLP_K512_M128N64_SAME_CTA_SECONDARY}\n",
                "server does not prove the M128N64 same-CTA Gate+Up "
                f"production stage: {MLP_K512_M128N64_SAME_CTA_SECONDARY}",
            ),
            (
                f"{MLP_K512_PRODUCT_QUANTIZE_MARKER}\n",
                "server does not prove the M128N64 same-CTA Gate+Up "
                f"production stage: {MLP_K512_PRODUCT_QUANTIZE_MARKER}",
            ),
        )
        for removed, error in removals:
            with self.subTest(removed=removed):
                self.fixture.server.write_text(
                    original.replace(removed, ""), encoding="utf-8"
                )
                result = self.fixture.run_attention_k256_m128n64_same_cta()
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn("gateup_m128n64_same_cta_expected_hits=64", contents)
        self.assertIn(
            "gateup_m128n64_same_cta_launch_hits="
            "${gateup_m128n64_same_cta_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_m128n64_same_cta_runtime_contract bucket=%s requests=%s "
            "launch_hits_per_request=%s status=passed",
            contents,
        )
        for zero_leaf in (
            "gateup_alternating_expected_hits=0",
            "gateup_ldmatrix_pairfeed_expected_hits=0",
            "gateup_m128n128_projection_serial_expected_hits=0",
            "gateup_m128n512_fused_quantize_expected_hits=0",
            "gateup_m128n512_paired_ldmatrix_expected_hits=0",
            "gateup_m64n128_register_pipeline_expected_hits=0",
            "gateup_m64n8_paired_warp_register_pipeline_expected_hits=0",
        ):
            self.assertIn(zero_leaf, contents)

    def test_m128n512_fused_quantize_replaces_only_pairfeed_gate(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(
            ATTENTION_K256_M128N512_FUSED_QUANTIZE_MODE,
            help_result.stderr,
        )

        baseline = self.fixture.run_attention_k256_ldmatrix_pairfeed()
        candidate = self.fixture.run_attention_k256_m128n512_fused_quantize()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={ATTENTION_K256_M128N512_FUSED_QUANTIZE_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=10", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(
            baseline_selectors - candidate_selectors,
            {MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR},
        )
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR},
        )
        for excluded in (
            MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR,
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR,
            MLP_K512_M128N128_PROJECTION_SERIAL_SELECTOR,
        ):
            self.assertNotIn(f"{excluded}=1", candidate_startup)

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}",
            delta,
        )
        self.assertIn(
            "removed_selector="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            delta,
        )
        self.assertIn(
            "added_selector="
            f"{MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR}",
            delta,
        )

        contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and MLP_K512_M128N512_FUSED_QUANTIZE_MARKER in line
        )
        for required in (
            MLP_K512_M128N512_FUSED_QUANTIZE_MARKER,
            DOWN_16WARP_PAIRRING_MARKER,
            "retained=prefill_projection_span_mlp_k512_input_quantize",
        ):
            self.assertIn(required, contract)
        for excluded in (
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER,
            MLP_K512_M128N128_PROJECTION_SERIAL_PRIMARY,
            MLP_K512_M128N128_PROJECTION_SERIAL_SECONDARY,
            MLP_K512_PRODUCT_QUANTIZE_MARKER,
        ):
            self.assertIn(excluded, contract)
        self.assertIn(
            "expected_request_launch_hits=gate_alternating:0,"
            "gate_pairfeed:0,gate_projection_serial:0,gate_candidate:64,"
            "down_incumbent:0,down_candidate:64",
            contract,
        )
        self.assertIn(
            "gateup_m128n512_fused_quantize_launch_hits_64_per_request",
            candidate.stdout,
        )
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_m128n512_fused_quantize_fails_closed_and_accounts_requests(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        removals = (
            (
                f"# {MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR}\n",
                "server does not contain the "
                f"{ATTENTION_K256_M128N512_FUSED_QUANTIZE_MODE} selector: "
                f"{MLP_K512_M128N512_FUSED_QUANTIZE_SELECTOR}",
            ),
            (
                f"{MLP_K512_M128N512_FUSED_QUANTIZE_MARKER}\n",
                "server does not prove the M128N512 fused-quantize Gate+Up "
                "production stage: "
                f"{MLP_K512_M128N512_FUSED_QUANTIZE_MARKER}",
            ),
        )
        for removed, error in removals:
            with self.subTest(removed=removed):
                self.fixture.server.write_text(
                    original.replace(removed, ""), encoding="utf-8"
                )
                result = self.fixture.run_attention_k256_m128n512_fused_quantize()
                self.assertEqual(result.returncode, 2)
                self.assertIn(error, result.stderr)

        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "gateup_m128n512_fused_quantize_expected_hits=64", contents
        )
        self.assertIn(
            "gateup_m128n512_fused_quantize_expected_hits=0", contents
        )
        self.assertIn(
            "gateup_m128n512_fused_quantize_launch_hits="
            "${gateup_m128n512_fused_quantize_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_m128n512_fused_quantize_runtime_contract "
            "bucket=%s requests=%s launch_hits_per_request=%s status=passed",
            contents,
        )

    def test_attention_k256_a_exchange_b4_fails_closed(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                f"# {ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}\n", ""
            ),
            encoding="utf-8",
        )
        missing_selector = self.fixture.run_attention_k256_a_exchange_b4()
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{ATTENTION_K256_A_EXCHANGE_B4_MODE} selector: "
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(
                f"{ATTENTION_K256_A_EXCHANGE_B4_MARKERS[-1]}\n", ""
            ),
            encoding="utf-8",
        )
        missing_marker = self.fixture.run_attention_k256_a_exchange_b4()
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the K256 Attention production stage: "
            f"{ATTENTION_K256_A_EXCHANGE_B4_MARKERS[-1]}",
            missing_marker.stderr,
        )

    def test_attention_k256_a_exchange_b4_records_independent_runtime_proof(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        expected_variables = (
            "attention_k256_incumbent_expected_launch_hits",
            "attention_k256_incumbent_expected_logical_hits",
            "attention_k256_a_exchange_b4_expected_launch_hits",
            "attention_k256_a_exchange_b4_expected_logical_hits",
        )
        for variable in expected_variables:
            self.assertIn(variable, contents)
        request_fields = (
            "attention_k256_m128n256_incumbent_launch_hits",
            "attention_k256_m128n256_incumbent_logical_projection_hits",
            "attention_k256_m128n256_a_exchange_b4_launch_hits",
            "attention_k256_m128n256_a_exchange_b4_logical_projection_hits",
        )
        for field in request_fields:
            self.assertIn(field, contents)
        self.assertIn(
            "attention_k256_runtime_contract bucket=%s requests=%s "
            "incumbent_launch_hits_per_request=%s "
            "incumbent_logical_projections_per_request=%s "
            "candidate_launch_hits_per_request=%s "
            "candidate_logical_projections_per_request=%s status=passed",
            contents,
        )
        self.assertIn(
            "experiment_baseline_mode="
            f"{ATTENTION_K256_ALTERNATING_DOWN_16WARP_PAIRRING_MODE}",
            contents,
        )
        self.assertIn(
            f"experiment_removed_selector={ATTENTION_K256_INCUMBENT_SELECTOR}",
            contents,
        )
        self.assertIn(
            "experiment_added_selector="
            f"{ATTENTION_K256_A_EXCHANGE_B4_SELECTOR}",
            contents,
        )

    def test_gdn_prompt_span_macro_is_one_selector_delta_from_current_best(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(GDN_PROMPT_SPAN_MACRO_MODE, help_result.stderr)

        baseline = self.fixture.run_attention_k256_a_exchange_b4()
        candidate = self.fixture.run_gdn_prompt_span_macro()
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(candidate.returncode, 0, candidate.stderr)
        self.assertIn(
            f"mode={GDN_PROMPT_SPAN_MACRO_MODE} dry_run=1",
            candidate.stdout,
        )
        self.assertIn("selector_count=11", candidate.stdout)

        baseline_startup = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        candidate_startup = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        baseline_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", baseline_startup)
        )
        candidate_selectors = set(
            re.findall(r"(Q3X_[A-Z0-9_]+)=1", candidate_startup)
        )
        self.assertEqual(baseline_selectors - candidate_selectors, set())
        self.assertEqual(
            candidate_selectors - baseline_selectors,
            {GDN_PROMPT_SPAN_MACRO_SELECTOR},
        )
        self.assertIn(
            f"-u {GDN_PROMPT_SPAN_MACRO_SELECTOR}", baseline_startup
        )
        self.assertNotIn(
            f"{GDN_PROMPT_SPAN_MACRO_SELECTOR}=1", baseline_startup
        )
        self.assertIn(
            f"-u {GDN_PROMPT_SPAN_MACRO_SELECTOR}", candidate_startup
        )
        self.assertIn(
            f"{GDN_PROMPT_SPAN_MACRO_SELECTOR}=1", candidate_startup
        )

        delta = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_A_EXCHANGE_B4_MODE}", delta
        )
        self.assertIn(
            "retained_selector=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
            delta,
        )
        self.assertIn(
            f"added_selector={GDN_PROMPT_SPAN_MACRO_SELECTOR}", delta
        )
        gdn_contract = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("stage_contract")
            and GDN_PROMPT_SPAN_MACRO_MARKER in line
        )
        self.assertIn(
            f"required={GDN_PROMPT_SPAN_MACRO_MARKER}", gdn_contract
        )
        self.assertIn(
            "expected_request_launch_hits=native:0,macro:48", gdn_contract
        )
        self.assertIn(
            "expected_request_logical_tokens=native:0,"
            "macro:48*prompt_tokens",
            gdn_contract,
        )
        baseline_metadata = next(
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("gdn_prompt_span_accounting_metadata")
        )
        candidate_metadata = next(
            line
            for line in candidate.stdout.splitlines()
            if line.startswith("gdn_prompt_span_accounting_metadata")
        )
        self.assertIn("route=native-c512", baseline_metadata)
        self.assertIn(
            "native_launch_formula=48*ceil(prompt_tokens/512)",
            baseline_metadata,
        )
        self.assertIn("macro_launch_formula=0", baseline_metadata)
        self.assertIn("route=prompt-span-macro", candidate_metadata)
        self.assertIn("native_launch_formula=0", candidate_metadata)
        self.assertIn("macro_launch_formula=48", candidate_metadata)
        self.assertIn("performance_evidence=0", candidate.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_gdn_prompt_span_macro_fails_closed_on_selector_or_stage(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(f"# {GDN_PROMPT_SPAN_MACRO_SELECTOR}\n", ""),
            encoding="utf-8",
        )
        missing_selector = self.fixture.run_gdn_prompt_span_macro()
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "server does not contain the "
            f"{GDN_PROMPT_SPAN_MACRO_MODE} selector: "
            f"{GDN_PROMPT_SPAN_MACRO_SELECTOR}",
            missing_selector.stderr,
        )

        self.fixture.server.write_text(
            original.replace(f"{GDN_PROMPT_SPAN_MACRO_MARKER}\n", ""),
            encoding="utf-8",
        )
        missing_marker = self.fixture.run_gdn_prompt_span_macro()
        self.assertEqual(missing_marker.returncode, 2)
        self.assertIn(
            "server does not prove the GDN prompt-span macro production "
            f"stage: {GDN_PROMPT_SPAN_MACRO_MARKER}",
            missing_marker.stderr,
        )

    def test_gdn_prompt_span_runtime_contract_is_dynamic_and_provenanced(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        for field in (
            "gdn_chunk64_native_launch_hits",
            "gdn_chunk64_native_logical_token_hits",
            "gdn_prompt_span_macro_launch_hits",
            "gdn_prompt_span_macro_logical_token_hits",
        ):
            self.assertIn(field, contents)
        self.assertIn(
            "expected_gdn_logical_token_hits=$((48 * prompt_tokens))",
            contents,
        )
        self.assertIn(
            "expected_gdn_native_launch_hits=$((48 * "
            "((prompt_tokens + 511) / 512)))",
            contents,
        )
        self.assertIn("expected_gdn_macro_launch_hits=48", contents)
        self.assertIn(
            "gdn_prompt_span_runtime_contract bucket=%s requests=%s "
            "route=%s dynamic_prompt_token_accounting=passed status=passed",
            contents,
        )
        self.assertIn(
            "gdn_chunk64_native_expected_launch_formula=%s", contents
        )
        self.assertIn(
            "gdn_prompt_span_macro_expected_logical_token_formula=%s",
            contents,
        )

        expected = {
            1804: (192, 86_592),
            1853: (192, 88_944),
            1792: (192, 86_016),
            2148: (240, 103_104),
            1906: (192, 91_488),
        }
        for prompt_tokens, (native_launches, logical_tokens) in expected.items():
            with self.subTest(prompt_tokens=prompt_tokens):
                self.assertEqual(
                    48 * ((prompt_tokens + 511) // 512), native_launches
                )
                self.assertEqual(48 * prompt_tokens, logical_tokens)

    def test_attention_k256_request_window_preserves_four_independent_counts(
        self,
    ) -> None:
        awk_program = """
            /^evaluation request .* prompt_tokens=/ {
              successes += 1
              if (successes > skip) {
                print
              }
            }
        """
        historical = [
            "evaluation request old prompt_tokens=512 "
            "attention_k256_m128n256_incumbent_launch_hits=128 "
            "attention_k256_m128n256_incumbent_logical_projection_hits=208 "
            "attention_k256_m128n256_a_exchange_b4_launch_hits=0 "
            "attention_k256_m128n256_a_exchange_b4_logical_projection_hits=0"
        ]
        current = [
            f"evaluation request {kind} prompt_tokens=2048 "
            "attention_k256_m128n256_incumbent_launch_hits=0 "
            "attention_k256_m128n256_incumbent_logical_projection_hits=0 "
            "attention_k256_m128n256_a_exchange_b4_launch_hits=128 "
            "attention_k256_m128n256_a_exchange_b4_logical_projection_hits=208"
            for kind in ("warmup", "measured-1", "measured-2")
        ]
        log = "\n".join([*historical, "evaluation request failed", *current])
        selected = subprocess.run(
            ["awk", "-v", "skip=1", awk_program],
            input=log + "\n",
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual(selected, current)
        for request in selected:
            self.assertRegex(
                request,
                r" attention_k256_m128n256_incumbent_launch_hits=0 ",
            )
            self.assertRegex(
                request,
                r" attention_k256_m128n256_incumbent_logical_projection_hits=0 ",
            )
            self.assertRegex(
                request,
                r" attention_k256_m128n256_a_exchange_b4_launch_hits=128 ",
            )
            self.assertRegex(
                request,
                r" attention_k256_m128n256_a_exchange_b4_"
                r"logical_projection_hits=208$",
            )

    def test_attention_k256_alternating_mode_requires_stage_marker(
        self,
    ) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(
                f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER}\n", ""
            ),
            encoding="utf-8",
        )
        result = self.fixture.run_attention_k256_alternating()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not prove the alternating M64N128 K256 Gate+Up "
            f"stage: {MLP_K512_EDGE_M64N128_K256_ALTERNATING_MARKER}",
            result.stderr,
        )

    def test_attention_modes_require_request_local_gateup_hit_proof(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn("gateup_alternating_expected_hits=64", contents)
        self.assertIn("gateup_alternating_expected_hits=0", contents)
        self.assertIn("gateup_ldmatrix_pairfeed_expected_hits=64", contents)
        self.assertIn("gateup_ldmatrix_pairfeed_expected_hits=0", contents)
        self.assertIn("expected_request_logs=$((eval_number + 1))", contents)
        self.assertIn(
            "gateup_alternating_launch_hits=${gateup_alternating_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_alternating_runtime_contract bucket=%s requests=%s "
            "launch_hits_per_request=%s status=passed",
            contents,
        )
        self.assertIn(
            "gateup_ldmatrix_pairfeed_launch_hits="
            "${gateup_ldmatrix_pairfeed_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_ldmatrix_pairfeed_runtime_contract bucket=%s "
            "requests=%s launch_hits_per_request=%s status=passed",
            contents,
        )
        self.assertIn(
            "ldmatrix_pairfeed_baseline_mode="
            f"{ATTENTION_K256_A_EXCHANGE_B4_MODE}",
            contents,
        )
        self.assertIn(
            "experiment_removed_selector="
            f"{MLP_K512_EDGE_M64N128_K256_ALTERNATING_SELECTOR}",
            contents,
        )
        self.assertIn(
            "experiment_added_selector="
            f"{MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR}",
            contents,
        )

    def test_attention_k256_receipt_and_overlay_binding_fail_closed(
        self,
    ) -> None:
        base_receipt = json.loads(
            self.fixture.k256_receipt.read_text(encoding="utf-8")
        )
        overlay_receipt = json.loads(
            self.fixture.mlp_k512_receipt.read_text(encoding="utf-8")
        )
        mutations = (
            (
                "wrong base layout",
                lambda base, overlay: base.__setitem__(
                    "physical_layout",
                    "sm87_s4_n64_packed_k64_scale_k128_consumer_v2",
                ),
                "K256 Prefill A4 publication receipt does not match",
            ),
            (
                "wrong payload digest",
                lambda base, overlay: base.__setitem__(
                    "payload_sha256", "f" * 64
                ),
                "K256 Prefill A4 publication receipt does not match",
            ),
            (
                "wrong policy digest",
                lambda base, overlay: base.__setitem__(
                    "policy_sha256", "f" * 64
                ),
                "K256 Prefill A4 publication receipt does not match",
            ),
            (
                "K128-bound MLP overlay",
                lambda base, overlay: overlay["required_base"].update(
                    {
                        "sidecar_kind": "a4_k128",
                        "physical_layout": (
                            "sm87_s4_n64_packed_k64_scale_k128_consumer_v2"
                        ),
                    }
                ),
                "K512 MLP overlay receipt is not bound",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label):
                selected_base = json.loads(json.dumps(base_receipt))
                selected_overlay = json.loads(json.dumps(overlay_receipt))
                mutate(selected_base, selected_overlay)
                self.fixture.k256_receipt.write_text(
                    json.dumps(selected_base) + "\n", encoding="utf-8"
                )
                self.fixture.mlp_k512_receipt.write_text(
                    json.dumps(selected_overlay) + "\n", encoding="utf-8"
                )
                result = self.fixture.run_attention_k256()
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
        self.fixture.k256_receipt.write_text(
            json.dumps(base_receipt) + "\n", encoding="utf-8"
        )
        self.fixture.mlp_k512_receipt.write_text(
            json.dumps(overlay_receipt) + "\n", encoding="utf-8"
        )

    def test_attention_k256_requires_every_compiled_stage_marker(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(f"{ATTENTION_K256_MARKERS[-1]}\n", ""),
            encoding="utf-8",
        )
        result = self.fixture.run_attention_k256()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not prove the K256 Attention production stage: "
            f"{ATTENTION_K256_MARKERS[-1]}",
            result.stderr,
        )

    def test_attention_k256_readiness_checks_exact_publications(self) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "server readiness did not prove the exact authenticated K256 "
            "publication",
            contents,
        )
        self.assertIn(
            "server readiness did not prove the exact K256-bound K512 MLP "
            "SHA chain",
            contents,
        )

    def test_mlp_k512_edge_mode_uses_v1_payload_and_exact_stage_bundle(
        self,
    ) -> None:
        result = self.fixture.run_mlp_k512_edge()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"mode={MLP_K512_EDGE_MODE} dry_run=1", result.stdout)
        self.assertIn("selector_count=8", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_ADMISSION",
                "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION",
            },
        )
        self.assertIn("--prefill-mlp-k512-payload", startup)
        self.assertIn(str(self.fixture.mlp_k512_payload), startup)
        self.assertIn("--prefill-mlp-k512-policy", startup)
        self.assertIn(str(self.fixture.mlp_k512_policy), startup)
        self.assertIn("--prefill-mlp-k512-receipt", startup)
        self.assertIn(str(self.fixture.mlp_k512_receipt), startup)
        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
        )
        self.assertIn(f"required={MLP_K512_EDGE_MARKER}", stage_contract)
        for excluded in (
            "prefill_projection_span_mlp_k512_gate_up_primary",
            "prefill_projection_span_mlp_k512_gate_up_secondary",
            "prefill_projection_span_mlp_k512_product_quantize",
        ):
            self.assertIn(excluded, stage_contract)
        self.assertIn(
            "prefill_mlp_k512_authenticated_192_of_192", result.stdout
        )
        self.assertIn("old_gateup_split_stages_excluded", result.stdout)
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_mlp_k512_edge_mode_requires_compiled_stage_marker(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(f"{MLP_K512_EDGE_MARKER}\n", ""),
            encoding="utf-8",
        )
        result = self.fixture.run_mlp_k512_edge()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not prove the Gate+Up-to-Down K512 edge stage: "
            f"{MLP_K512_EDGE_MARKER}",
            result.stderr,
        )
        self.assertFalse(self.fixture.output.exists())

    def test_mlp_k512_edge_m128n64_uses_exact_real_api_bundle(self) -> None:
        result = self.fixture.run_mlp_k512_edge_m128n64()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={MLP_K512_EDGE_M128N64_MODE} dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=8", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_ADMISSION",
                "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION",
            },
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1", startup
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION=1", startup
        )
        stage_contracts = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
        ]
        self.assertEqual(len(stage_contracts), 1)
        stage_contract = stage_contracts[0]
        self.assertIn(
            f"required={MLP_K512_EDGE_M128N64_MARKER}", stage_contract
        )
        for excluded in (
            MLP_K512_EDGE_MARKER,
            "prefill_projection_span_mlp_k512_gate_up_primary",
            "prefill_projection_span_mlp_k512_gate_up_secondary",
            "prefill_projection_span_mlp_k512_product_quantize",
            MLP_K512_DOWN_M16N64_V2_MARKER,
        ):
            self.assertIn(excluded, stage_contract)
        self.assertIn(
            "retained=prefill_projection_span_mlp_k512_input_quantize,"
            "prefill_projection_span_mlp_k512_down",
            stage_contract,
        )
        self.assertIn("legacy_edge_stage_excluded", result.stdout)
        self.assertIn("down_m16n64_v2_stage_excluded", result.stdout)
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_mlp_k512_edge_m128n64_requires_compiled_stage_marker(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(f"{MLP_K512_EDGE_M128N64_MARKER}\n", ""),
            encoding="utf-8",
        )
        result = self.fixture.run_mlp_k512_edge_m128n64()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not prove the M128N64 Gate+Up-to-Down K512 "
            f"edge stage: {MLP_K512_EDGE_M128N64_MARKER}",
            result.stderr,
        )
        self.assertFalse(self.fixture.output.exists())

    def test_mlp_k512_current_best_promotes_edge_stage(self) -> None:
        result = self.fixture.run_mlp_k512(MLP_K512_CURRENT_MODE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={MLP_K512_CURRENT_MODE} dry_run=1", result.stdout
        )
        self.assertIn("selector_count=8", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1", startup
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION=1",
            startup,
        )
        self.assertIn(
            f"stage_contract required={MLP_K512_EDGE_MARKER}",
            result.stdout,
        )

    def test_mlp_k512_v1_mode_freezes_pre_edge_comparator(self) -> None:
        result = self.fixture.run_mlp_k512(MLP_K512_V1_MODE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"mode={MLP_K512_V1_MODE} dry_run=1", result.stdout)
        self.assertIn("selector_count=7", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertNotIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1", startup
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION", startup
        )
        self.assertNotIn("stage_contract", result.stdout)

    def test_mlp_k512_down_m16n64_v2_keeps_retained_edge(self) -> None:
        result = self.fixture.run_mlp_k512(
            MLP_K512_DOWN_M16N64_V2_MODE
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={MLP_K512_DOWN_M16N64_V2_MODE} dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=9", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertIn(
            "Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION=1",
            startup,
        )
        self.assertIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1", startup
        )
        self.assertIn(
            f"required={MLP_K512_EDGE_MARKER},"
            f"{MLP_K512_DOWN_M16N64_V2_MARKER}",
            result.stdout,
        )
        self.assertIn(
            "excluded=prefill_projection_span_mlp_k512_down,",
            result.stdout,
        )
        stage_contracts = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
        ]
        self.assertEqual(len(stage_contracts), 1)
        self.assertNotIn(
            "retained=prefill_projection_span_mlp_k512_input_quantize,"
            "prefill_projection_span_mlp_k512_down",
            stage_contracts[0],
        )

    def test_mlp_k512_down_m16n64_v2_requires_stage_marker(
        self,
    ) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(f"{MLP_K512_DOWN_M16N64_V2_MARKER}\n", ""),
            encoding="utf-8",
        )
        result = self.fixture.run_mlp_k512(
            MLP_K512_DOWN_M16N64_V2_MODE
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not prove the Down K512 M16N64 v2 stage: "
            f"{MLP_K512_DOWN_M16N64_V2_MARKER}",
            result.stderr,
        )
        self.fixture.server.write_text(original, encoding="utf-8")

    def test_real_run_rejects_partial_evalscope_success(self) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn('summary.get("Total Requests")', contents)
        self.assertIn('summary.get("Success Requests")', contents)
        self.assertIn('summary.get("Failed Requests")', contents)
        self.assertIn(
            "expected = int(sys.argv[2])", contents
        )
        self.assertIn(
            "if total != expected or success != expected or failed != 0:",
            contents,
        )

    def test_hybrid_gate_mode_uses_exact_authenticated_real_api_route(
        self,
    ) -> None:
        result = self.fixture.run_hybrid()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"mode={HYBRID_GATE_MODE} dry_run=1", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
                "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_"
                "ADMISSION",
                "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_"
                "LDMATRIX_ADMISSION",
            },
        )
        self.assertIn(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            startup,
        )
        self.assertIn(str(self.fixture.hybrid_payload), startup)
        self.assertIn(
            "--prefill-mlp-k512-paired-gateup-canonical-down-policy",
            startup,
        )
        self.assertIn(str(self.fixture.hybrid_policy), startup)
        self.assertIn(
            "--prefill-mlp-k512-paired-gateup-canonical-down-receipt",
            startup,
        )
        self.assertIn(str(self.fixture.hybrid_receipt), startup)
        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and HYBRID_INPUT_MARKER in line
        )
        self.assertIn(f"required={HYBRID_INPUT_MARKER},{HYBRID_GATE_MARKER}", stage_contract)
        self.assertIn(f"retained={HYBRID_CANONICAL_DOWN_MARKER}", stage_contract)
        self.assertIn(f"excluded={HYBRID_PAIRRING_DOWN_MARKER},", stage_contract)
        for old_stage in (
            "prefill_projection_span_mlp_k512_input_quantize",
            "prefill_projection_span_mlp_k512_gate_up_primary",
            "prefill_projection_span_mlp_k512_gateup_down_edge",
            "prefill_projection_span_mlp_k512_down",
            "prefill_projection_span_mlp_k512_fragment_native_input_quantize",
            "prefill_projection_span_mlp_k512_fragment_native_down",
        ):
            self.assertIn(old_stage, stage_contract)
        self.assertIn("expected_request_launch_hits=gate:64,down:0", stage_contract)
        self.assertIn(f"layout={HYBRID_LAYOUT}", result.stdout)
        self.assertIn("complete_sha_chain", result.stdout)
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_hybrid_gate_down_pairring_mode_changes_only_down_route(
        self,
    ) -> None:
        result = self.fixture.run_hybrid(down_pairring=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={HYBRID_GATE_DOWN_PAIRRING_MODE} dry_run=1",
            result.stdout,
        )
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        selectors = set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup))
        self.assertIn(
            "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION",
            selectors,
        )
        self.assertEqual(len(selectors), 10)
        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and HYBRID_INPUT_MARKER in line
        )
        self.assertIn(
            f"required={HYBRID_INPUT_MARKER},{HYBRID_GATE_MARKER},"
            f"{HYBRID_PAIRRING_DOWN_MARKER}",
            stage_contract,
        )
        self.assertIn(f"excluded={HYBRID_CANONICAL_DOWN_MARKER},", stage_contract)
        self.assertNotIn(" retained=", stage_contract)
        self.assertIn("expected_request_launch_hits=gate:64,down:64", stage_contract)

    def test_hybrid_triplet_and_complete_receipt_chain_fail_closed(self) -> None:
        partial = self.fixture.run(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            str(self.fixture.hybrid_payload),
            "--prefill-mlp-k512-paired-gateup-canonical-down-policy",
            str(self.fixture.hybrid_policy),
            "--mode",
            HYBRID_GATE_MODE,
            prefill_payload=self.fixture.k256_payload,
            prefill_policy=self.fixture.k256_policy,
            prefill_receipt=self.fixture.k256_receipt,
        )
        self.assertEqual(partial.returncode, 2)
        self.assertIn(
            "paired-GateUp/canonical-Down MLP K512 payload, policy, and "
            "receipt are required together",
            partial.stderr,
        )

        original = json.loads(
            self.fixture.hybrid_receipt.read_text(encoding="utf-8")
        )
        mutations = (
            lambda receipt: receipt.__setitem__("physical_layout", "wrong"),
            lambda receipt: receipt.__setitem__("payload_sha256", "1" * 64),
            lambda receipt: receipt["source_v1"].__setitem__(
                "policy_sha256", "2" * 64
            ),
            lambda receipt: receipt["required_base"].__setitem__(
                "manifest_sha256", "3" * 64
            ),
            lambda receipt: receipt.__setitem__("unexpected", True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                selected = json.loads(json.dumps(original))
                mutate(selected)
                self.fixture.hybrid_receipt.write_text(
                    json.dumps(selected) + "\n", encoding="utf-8"
                )
                result = self.fixture.run_hybrid()
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "invalid paired-GateUp/canonical-Down K512 "
                    "publication chain",
                    result.stderr,
                )
        self.fixture.hybrid_receipt.write_text(
            json.dumps(original) + "\n", encoding="utf-8"
        )

    def test_hybrid_modes_require_new_compiled_selector_and_stage(self) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            original.replace(
                "# Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_"
                "LDMATRIX_ADMISSION\n",
                "",
            ),
            encoding="utf-8",
        )
        missing_selector = self.fixture.run_hybrid()
        self.assertEqual(missing_selector.returncode, 2)
        self.assertIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_"
            "LDMATRIX_ADMISSION",
            missing_selector.stderr,
        )
        self.fixture.server.write_text(
            original.replace(f"{HYBRID_PAIRRING_DOWN_MARKER}\n", ""),
            encoding="utf-8",
        )
        missing_stage = self.fixture.run_hybrid(down_pairring=True)
        self.assertEqual(missing_stage.returncode, 2)
        self.assertIn(
            "server does not prove the paired-GateUp/canonical-Down "
            f"production stage: {HYBRID_PAIRRING_DOWN_MARKER}",
            missing_stage.stderr,
        )

    def test_hybrid_modes_enforce_request_local_gate_and_down_hits(self) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn("gateup_m128n512_paired_ldmatrix_expected_hits=64", contents)
        self.assertIn("down_m128n128_ldmatrix_pairring_expected_hits=0", contents)
        self.assertIn("down_m128n128_ldmatrix_pairring_expected_hits=64", contents)
        self.assertIn(
            "gateup_m128n512_paired_ldmatrix_launch_hits="
            "${gateup_m128n512_paired_ldmatrix_expected_hits}",
            contents,
        )
        self.assertIn(
            "down_m128n128_ldmatrix_pairring_launch_hits="
            "${down_m128n128_ldmatrix_pairring_expected_hits}",
            contents,
        )
        self.assertIn(
            "hybrid_mlp_runtime_contract bucket=%s requests=%s "
            "gate_launch_hits_per_request=%s "
            "down_pairring_launch_hits_per_request=%s status=passed",
            contents,
        )

    def test_hybrid_request_window_skips_history_and_includes_warmup(
        self,
    ) -> None:
        awk_program = """
            /^evaluation request .* prompt_tokens=/ {
              successes += 1
              if (successes > skip) {
                print
              }
            }
        """
        for down_hits in (0, 64):
            with self.subTest(down_hits=down_hits):
                historical = [
                    "evaluation request 1 prompt_tokens=100 "
                    "gateup_m128n512_paired_ldmatrix_launch_hits=0 "
                    "down_m128n128_ldmatrix_pairring_launch_hits=0",
                    "evaluation request 2 prompt_tokens=200 "
                    "gateup_m128n512_paired_ldmatrix_launch_hits=0 "
                    "down_m128n128_ldmatrix_pairring_launch_hits=0",
                ]
                current = [
                    "evaluation request warmup prompt_tokens=1804 "
                    "gateup_m128n512_paired_ldmatrix_launch_hits=64 "
                    f"down_m128n128_ldmatrix_pairring_launch_hits={down_hits}",
                    *[
                        f"evaluation request measured-{index} "
                        "prompt_tokens=1853 "
                        "gateup_m128n512_paired_ldmatrix_launch_hits=64 "
                        "down_m128n128_ldmatrix_pairring_launch_hits="
                        f"{down_hits}"
                        for index in range(4)
                    ],
                ]
                log = "\n".join(
                    [*historical, "evaluation request failed", *current]
                )
                selected = subprocess.run(
                    ["awk", "-v", "skip=2", awk_program],
                    input=log + "\n",
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.splitlines()
                self.assertEqual(selected, current)
                self.assertEqual(len(selected), 5)
                for request in selected:
                    self.assertRegex(
                        request,
                        r" gateup_m128n512_paired_ldmatrix_launch_hits=64$|"
                        r" gateup_m128n512_paired_ldmatrix_launch_hits=64 ",
                    )
                    self.assertRegex(
                        request,
                        rf" down_m128n128_ldmatrix_pairring_launch_hits="
                        rf"{down_hits}$",
                    )

    def test_fragment_native_m64n128_1cta_mode_uses_authenticated_real_route(
        self,
    ) -> None:
        result = self.fixture.run_fragment_native_m64n128_1cta()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={FRAGMENT_NATIVE_M64N128_1CTA_MODE} dry_run=1",
            result.stdout,
        )
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION",
            },
        )
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-payload", startup
        )
        self.assertIn(str(self.fixture.fragment_native_payload), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-policy", startup
        )
        self.assertIn(str(self.fixture.fragment_native_policy), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-receipt", startup
        )
        self.assertIn(str(self.fixture.fragment_native_receipt), startup)
        self.assertIn(
            "gateup_variant=m64n128_1cta down_variant=m64n128",
            result.stdout,
        )
        self.assertIn(
            "prefill_mlp_k512_fragment_native_gateup_variant_m64n128_1cta",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_fragment_native_policy_must_match_publication_receipt(
        self,
    ) -> None:
        receipt = json.loads(
            self.fixture.fragment_native_receipt.read_text(encoding="utf-8")
        )
        receipt["source_v1"]["policy_sha256"] = "f" * 64
        self.fixture.fragment_native_receipt.write_text(
            json.dumps(receipt) + "\n", encoding="utf-8"
        )
        result = self.fixture.run_fragment_native_m64n128_1cta()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "fragment-native policy SHA256 does not match source_v1 receipt",
            result.stderr,
        )
        self.assertFalse(self.fixture.output.exists())

    def test_fragment_native_m128n64_staged_mode_uses_authenticated_real_route(
        self,
    ) -> None:
        result = self.fixture.run_fragment_native_m128n64_staged()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={FRAGMENT_NATIVE_M128N64_STAGED_MODE} dry_run=1",
            result.stdout,
        )
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION",
            },
        )
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-payload", startup
        )
        self.assertIn(str(self.fixture.fragment_native_payload), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-policy", startup
        )
        self.assertIn(str(self.fixture.fragment_native_policy), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-receipt", startup
        )
        self.assertIn(str(self.fixture.fragment_native_receipt), startup)
        self.assertIn(
            "gateup_variant=m128n64_staged down_variant=m64n128",
            result.stdout,
        )
        self.assertIn(
            "prefill_mlp_k512_fragment_native_gateup_variant_"
            "m128n64_staged",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_fragment_native_m128n64_staged_rejects_other_gateup_markers(
        self,
    ) -> None:
        rejected_markers = (
            "prefill_projection_span_mlp_k512_fragment_native_"
            "gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128n64_1cta_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m64n128_1cta_gateup_primary",
        )
        for marker in rejected_markers:
            with self.subTest(marker=marker):
                with tempfile.TemporaryDirectory() as root:
                    fixture = Fixture(pathlib.Path(root))
                    result = fixture.run_fragment_native_m128n64_staged(
                        extra_stage_markers=(marker,)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        "server contains the mutually exclusive "
                        f"fragment-native Gate+Up variant: {marker}",
                        result.stderr,
                    )

    def test_fragment_native_m64n128_1cta_rejects_all_old_gateup_markers(
        self,
    ) -> None:
        rejected_markers = (
            "prefill_projection_span_mlp_k512_fragment_native_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128n64_1cta_gateup_primary",
            FRAGMENT_NATIVE_M128N64_STAGED_PRIMARY,
        )
        for marker in rejected_markers:
            with self.subTest(marker=marker):
                with tempfile.TemporaryDirectory() as root:
                    fixture = Fixture(pathlib.Path(root))
                    result = fixture.run_fragment_native_m64n128_1cta(
                        extra_stage_markers=(marker,)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        "server contains the mutually exclusive "
                        f"fragment-native Gate+Up variant: {marker}",
                        result.stderr,
                    )

    def test_paired_warp_mode_uses_real_hybrid_publication_and_exact_leaves(
        self,
    ) -> None:
        result = self.fixture.run_paired_warp()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"mode={PAIRED_WARP_MODE} dry_run=1", result.stdout)
        self.assertIn("selector_count=10", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        selectors = set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup))
        self.assertIn(MLP_K512_PAIRED_MASTER_SELECTOR, selectors)
        self.assertIn(MLP_K512_PAIRED_WARP_SELECTOR, selectors)
        self.assertIn(DOWN_16WARP_PAIRRING_SELECTOR, selectors)
        self.assertNotIn(
            "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_"
            "PAIRED_LDMATRIX_ADMISSION",
            selectors,
        )
        self.assertIn(
            "--prefill-mlp-k512-paired-gateup-canonical-down-payload",
            startup,
        )
        self.assertIn(str(self.fixture.hybrid_payload), startup)
        self.assertNotIn("--prefill-mlp-k512-payload", startup)
        self.assertIn(PAIRED_WARP_GATE_MARKER, result.stdout)
        self.assertIn(PAIRED_WARP_DOWN_16WARP_MARKER, result.stdout)
        self.assertIn(
            "gateup_m128n512_paired_ldmatrix_launch_hits_0_per_request",
            result.stdout,
        )
        self.assertIn(
            "gateup_m64n8_paired_warp_register_pipeline_"
            "launch_hits_64_per_request",
            result.stdout,
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_launch_hits_64_per_request",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_projection_major_mode_replaces_only_the_mlp_publication_and_gate(
        self,
    ) -> None:
        help_result = subprocess.run(
            [str(RUNNER), "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn(PROJECTION_MAJOR_MODE, help_result.stderr)
        self.assertIn(
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-"
            "payload",
            help_result.stderr,
        )

        result = self.fixture.run_projection_major()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={PROJECTION_MAJOR_MODE} dry_run=1", result.stdout
        )
        self.assertIn("selector_count=10", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        selectors = set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup))
        self.assertEqual(
            selectors,
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
                ATTENTION_K256_A_EXCHANGE_B4_SELECTOR,
                MLP_K512_PROJECTION_MAJOR_SELECTOR,
                MLP_K512_REGISTER_PIPELINE_SELECTOR,
                DOWN_16WARP_PAIRRING_SELECTOR,
            },
        )
        for removed in (
            "Q3X_RUN_A4W4_MLP_K512_ADMISSION",
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR,
        ):
            self.assertIn(f"-u {removed}", startup)
            self.assertNotIn(f"{removed}=1", startup)
        for flag, path in (
            (
                "--prefill-mlp-k512-projection-major-gateup-canonical-"
                "down-payload",
                self.fixture.projection_major_payload,
            ),
            (
                "--prefill-mlp-k512-projection-major-gateup-canonical-"
                "down-policy",
                self.fixture.projection_major_policy,
            ),
            (
                "--prefill-mlp-k512-projection-major-gateup-canonical-"
                "down-receipt",
                self.fixture.projection_major_receipt,
            ),
        ):
            self.assertIn(flag, startup)
            self.assertIn(str(path), startup)
        self.assertNotIn("--prefill-mlp-k512-payload", startup)

        delta = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("candidate_delta")
        )
        self.assertIn(
            f"baseline_mode={ATTENTION_K256_LDMATRIX_PAIRFEED_MODE}", delta
        )
        self.assertIn(MLP_K512_PROJECTION_MAJOR_SELECTOR, delta)
        self.assertIn(MLP_K512_REGISTER_PIPELINE_SELECTOR, delta)
        self.assertIn(
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_SELECTOR, delta
        )

        stage_contract = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("stage_contract")
            and PROJECTION_MAJOR_GATE_MARKER in line
        )
        self.assertIn(
            f"required={PROJECTION_MAJOR_INPUT_MARKER},"
            f"{PROJECTION_MAJOR_GATE_MARKER},"
            f"{PROJECTION_MAJOR_DOWN_16WARP_MARKER}",
            stage_contract,
        )
        self.assertIn(
            MLP_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_MARKER,
            stage_contract,
        )
        self.assertIn(
            "expected_request_launch_hits=gate_incumbent:0,"
            "gate_candidate:64,down_incumbent:0,down_candidate:64",
            stage_contract,
        )
        self.assertIn(f"layout={PROJECTION_MAJOR_LAYOUT}", result.stdout)
        self.assertIn(
            "gateup_m64n128_register_pipeline_launch_hits_64_per_request",
            result.stdout,
        )
        self.assertIn(
            "gateup_ldmatrix_pairfeed_launch_hits_0_per_request",
            result.stdout,
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_launch_hits_64_per_request",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_projection_major_paths_and_publications_fail_closed(self) -> None:
        partial = self.fixture.run(
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-payload",
            str(self.fixture.projection_major_payload),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-policy",
            str(self.fixture.projection_major_policy),
            "--mode",
            PROJECTION_MAJOR_MODE,
            prefill_payload=self.fixture.k256_payload,
            prefill_policy=self.fixture.k256_policy,
            prefill_receipt=self.fixture.k256_receipt,
        )
        self.assertEqual(partial.returncode, 2)
        self.assertIn(
            "projection-major-GateUp/canonical-Down MLP K512 payload, policy, "
            "and receipt are required together",
            partial.stderr,
        )

        missing = self.fixture.root / "missing-projection-major.bin"
        missing_path = self.fixture.run(
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-payload",
            str(missing),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-policy",
            str(self.fixture.projection_major_policy),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-receipt",
            str(self.fixture.projection_major_receipt),
            "--mode",
            PROJECTION_MAJOR_MODE,
            prefill_payload=self.fixture.k256_payload,
            prefill_policy=self.fixture.k256_policy,
            prefill_receipt=self.fixture.k256_receipt,
        )
        self.assertEqual(missing_path.returncode, 2)
        self.assertIn(
            "missing required projection-major-GateUp/canonical-Down MLP "
            "K512 payload",
            missing_path.stderr,
        )

        alias = self.fixture.run(
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-payload",
            str(self.fixture.projection_major_payload),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-policy",
            str(self.fixture.projection_major_receipt),
            "--prefill-mlp-k512-projection-major-gateup-canonical-down-receipt",
            str(self.fixture.projection_major_receipt),
            "--mode",
            PROJECTION_MAJOR_MODE,
            prefill_payload=self.fixture.k256_payload,
            prefill_policy=self.fixture.k256_policy,
            prefill_receipt=self.fixture.k256_receipt,
        )
        self.assertEqual(alias.returncode, 2)
        self.assertIn("must be distinct files", alias.stderr)

        conflict = self.fixture.run_projection_major(
            "--prefill-mlp-k512-payload",
            str(self.fixture.mlp_k512_payload),
            "--prefill-mlp-k512-policy",
            str(self.fixture.mlp_k512_policy),
            "--prefill-mlp-k512-receipt",
            str(self.fixture.mlp_k512_receipt),
        )
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("are mutually exclusive", conflict.stderr)

    def test_projection_major_receipt_identity_is_strict(self) -> None:
        original = json.loads(
            self.fixture.projection_major_receipt.read_text(encoding="utf-8")
        )
        mutations = (
            lambda receipt: receipt.__setitem__("version", {"major": 1, "minor": 0}),
            lambda receipt: receipt.__setitem__("physical_layout", "wrong"),
            lambda receipt: receipt.__setitem__("payload_sha256", "9" * 64),
            lambda receipt: receipt["required_base"].__setitem__(
                "manifest_sha256", "9" * 64
            ),
            lambda receipt: receipt.__setitem__("unexpected", True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                selected = json.loads(json.dumps(original))
                mutate(selected)
                self.fixture.projection_major_receipt.write_text(
                    json.dumps(selected) + "\n", encoding="utf-8"
                )
                result = self.fixture.run_projection_major()
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "invalid projection-major-GateUp/canonical-Down K512 "
                    "publication chain",
                    result.stderr,
                )
        self.fixture.projection_major_receipt.write_text(
            json.dumps(original) + "\n", encoding="utf-8"
        )

    def test_projection_major_requires_both_selectors_and_every_stage(self) -> None:
        original = self.fixture.server.read_text(encoding="utf-8")
        for selector in (
            MLP_K512_PROJECTION_MAJOR_SELECTOR,
            MLP_K512_REGISTER_PIPELINE_SELECTOR,
        ):
            with self.subTest(selector=selector):
                self.fixture.server.write_text(
                    original.replace(f"# {selector}\n", ""),
                    encoding="utf-8",
                )
                result = self.fixture.run_projection_major()
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    f"server does not contain the {PROJECTION_MAJOR_MODE} "
                    f"selector: {selector}",
                    result.stderr,
                )
        for marker in (
            PROJECTION_MAJOR_INPUT_MARKER,
            PROJECTION_MAJOR_GATE_MARKER,
            PROJECTION_MAJOR_DOWN_16WARP_MARKER,
        ):
            with self.subTest(marker=marker):
                self.fixture.server.write_text(
                    original.replace(f"{marker}\n", ""), encoding="utf-8"
                )
                result = self.fixture.run_projection_major()
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "server does not prove the projection-major-GateUp "
                    f"production stage: {marker}",
                    result.stderr,
                )
        self.fixture.server.write_text(original, encoding="utf-8")

    def test_projection_major_readiness_and_request_hits_are_hard_gates(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "prefill_mlp_k512_projection_major_gateup_canonical_down_"
            "requested=1 .*prefill_mlp_k512_projection_major_gateup_"
            "canonical_down_enabled=1 .*prefill_mlp_k512_projection_major_"
            "gateup_canonical_down_layers=64 .*prefill_mlp_k512_projection_"
            "major_gateup_canonical_down_bytes=8623226880",
            contents,
        )
        for identity in (
            "layout",
            "manifest_sha256",
            "policy_sha256",
            "payload_sha256",
            "receipt_sha256",
            "source_v1_receipt_sha256",
        ):
            self.assertIn(
                "prefill_mlp_k512_projection_major_gateup_canonical_down_"
                f"{identity}=",
                contents,
            )
        self.assertIn(
            "gateup_m64n128_register_pipeline_launch_hits="
            "${gateup_m64n128_register_pipeline_expected_hits}",
            contents,
        )
        self.assertIn(
            "gateup_ldmatrix_pairfeed_launch_hits="
            "${gateup_ldmatrix_pairfeed_expected_hits}",
            contents,
        )
        self.assertIn(
            "down_m128n128_16warp_pairring_launch_hits="
            "${down_m128n128_16warp_pairring_expected_hits}",
            contents,
        )
        self.assertIn(
            "projection_major_mlp_runtime_contract bucket=%s requests=%s "
            "old_gate_launch_hits_per_request=%s "
            "gate_launch_hits_per_request=%s "
            "down_incumbent_launch_hits_per_request=%s "
            "down_candidate_launch_hits_per_request=%s status=passed",
            contents,
        )

    def test_cumulative_short_mode_adds_short_route_without_cells(self) -> None:
        result = self.fixture.run("--mode", "cumulative-prefill-short")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=cumulative-prefill-short", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)),
            {
                "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
                "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
                "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
                "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
            },
        )
        cleared = (
            "Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION",
            "Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION",
        )
        for selector in cleared:
            self.assertIn(f"-u {selector}", startup)
            self.assertNotIn(f"{selector}=1", startup)

    def test_cumulative_short_requires_compiled_selector(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(
                "# Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION\n", ""
            ),
            encoding="utf-8",
        )
        result = self.fixture.run("--mode", "cumulative-prefill-short")
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the cumulative-prefill-short selector: "
            "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
            result.stderr,
        )

    def test_missing_a4_payload_is_rejected(self) -> None:
        self.fixture.payload.unlink()
        result = self.fixture.run()
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing required Prefill A4 payload", result.stderr)
        self.assertFalse(self.fixture.output.exists())

    def test_modes_are_mutually_exclusive(self) -> None:
        result = self.fixture.run(
            "--mode", "exact", "--mode", "native-gdn"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--mode may be specified only once", result.stderr)

    def test_existing_output_is_never_overwritten(self) -> None:
        self.fixture.output.mkdir()
        marker = self.fixture.output / "historical.txt"
        marker.write_text("keep\n", encoding="utf-8")
        result = self.fixture.run()
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing to overwrite output root", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
