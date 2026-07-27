#!/usr/bin/env python3
"""CPU-only synthetic tests for q20_sector_model.py."""

from __future__ import annotations

import unittest

import numpy as np

import q20_sector_model as model


class Q20SectorModelTest(unittest.TestCase):
    def test_embedded_self_test(self) -> None:
        result = model.synthetic_self_test()
        self.assertEqual(result["self_test"], "pass")
        self.assertEqual(result["completed_rows_pruned"], 16)
        self.assertEqual(result["per_cta_rows_pruned"], 8)
        self.assertEqual(result["lag1_rows_pruned"], 0)

    def test_production_payload_geometry(self) -> None:
        topology = model.PRODUCTION_TOPOLOGY
        self.assertEqual(topology.baseline_mandatory_bytes, 715_161_600)
        self.assertEqual(topology.sidecar_bytes, 18_872_320)
        self.assertEqual(topology.cta_group_count, 7_760)
        self.assertEqual(topology.sidecar_sectors_per_cta_group, 76)
        self.assertEqual(topology.row_stride, 2_048)
        self.assertEqual(topology.wave_count, 122)

    def test_odd_checkpoint_scale_overfetch(self) -> None:
        topology = model.Topology(
            rows=32,
            columns=512,
            grid_blocks=1,
            warps_per_block=8,
            rows_per_warp=4,
            columns_per_checkpoint=256,
        )
        processed = np.ones(topology.rows, dtype=np.uint8)
        traffic = model.count_traffic(processed, topology, gate_bytes=0)
        self.assertEqual(traffic["transaction_waste_bytes"], 32 * 16)

    def test_bounded_lag_is_not_independent_evidence(self) -> None:
        topology = model.Topology(
            rows=32,
            columns=512,
            grid_blocks=2,
            warps_per_block=2,
            rows_per_warp=4,
            columns_per_checkpoint=256,
        )
        upper = np.full((32, 1), 100.0, dtype=np.float32)
        lower = np.zeros(32, dtype=np.float32)
        proof = model.ProofInput(upper, lower, "synthetic")
        results = model.simulate_proof(
            proof,
            topology,
            schedules=["bounded-lag-atomic"],
            incumbent_deltas=[-1.0],
            atomic_lags=[1],
            gate_bytes=0,
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["requires_external_lag_proof"])

    def test_decode_tables_and_outward_bf16(self) -> None:
        self.assertEqual(model.E2M1_TABLE.tolist(), [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ])
        self.assertEqual(model.E4M3FN_TABLE[1], 2.0 ** -9)
        self.assertEqual(model.E4M3FN_TABLE[0x7E], 448.0)
        self.assertTrue(np.isnan(model.E4M3FN_TABLE[0x7F]))
        values = np.asarray(
            [1.001, -1.001, 3.14159, -3.14159, 1.0e-40, -1.0e-40],
            dtype=np.float64,
        )
        upper = model._upper_to_bf16_rne_image(values)
        lower = model._lower_to_bf16_rne_image(values)
        nominal = model._bf16_bits_to_float32(
            model._float32_to_bf16_bits(values.astype(np.float32))
        )
        self.assertTrue(np.all(upper >= lower))
        self.assertTrue(np.all(upper >= nominal))
        self.assertTrue(np.all(lower <= nominal))

        midpoint = np.float32(1.0 + 2.0 ** -8)
        boundary_values = np.asarray(
            [
                -np.nextafter(midpoint, np.float32(-np.inf)),
                -midpoint,
                -np.nextafter(midpoint, np.float32(np.inf)),
                np.nextafter(midpoint, np.float32(-np.inf)),
                midpoint,
                np.nextafter(midpoint, np.float32(np.inf)),
            ],
            dtype=np.float32,
        )
        boundary_values.sort()
        rne = model._bf16_bits_to_float32(
            model._float32_to_bf16_bits(boundary_values)
        )
        self.assertTrue(np.all(rne[1:] >= rne[:-1]))

    def test_directed_suffix_norm(self) -> None:
        squared = np.asarray([[9.0, 16.0, 0.0]], dtype=np.float64)
        suffix = model._directed_suffix_norms(squared)
        self.assertEqual(suffix.shape, (1, 2))
        self.assertGreaterEqual(float(suffix[0, 0]), 4.0)
        # The proof deliberately rounds even an exact zero suffix upward.  This
        # keeps one uniform strict upper-bound contract at the cost of one
        # harmless FP32 subnormal.
        self.assertEqual(
            suffix[0, 1],
            np.nextafter(np.float32(0.0), np.float32(np.inf)),
        )

    def test_fp64_envelope_and_sidecar_layout(self) -> None:
        self.assertGreater(model.FP64_PROOF_RHO, model.FP64_GAMMA_PATH)
        reduced = np.asarray([1.0, 0.0], dtype=np.float64)
        upper = model._positive_reduction_upper(reduced)
        self.assertTrue(np.all(upper >= reduced))

        topology = model.Topology(
            rows=32,
            columns=1024,
            grid_blocks=1,
            warps_per_block=2,
            rows_per_warp=4,
            columns_per_checkpoint=256,
        )
        sidecar = np.arange(
            topology.rows * topology.bound_values_per_row, dtype=np.float32
        ).reshape(topology.rows, topology.bound_values_per_row)
        packed = model._pack_schedule_sidecar(sidecar, topology)
        self.assertEqual(
            packed.shape,
            (
                topology.cta_group_count,
                topology.bound_values_per_row,
                topology.rows_per_cta_group,
            ),
        )
        np.testing.assert_array_equal(packed[0, 1], sidecar[:8, 1])
        self.assertEqual(packed.dtype, np.dtype("<f4"))

    def test_nonfinite_upper_disables_complete_row(self) -> None:
        topology = model.Topology(
            rows=16,
            columns=1024,
            grid_blocks=1,
            warps_per_block=2,
            rows_per_warp=4,
            columns_per_checkpoint=256,
        )
        upper = np.zeros(
            (topology.rows, topology.bound_values_per_row), dtype=np.float32
        )
        upper[8:, -1] = np.float32(np.inf)
        lower = np.zeros(topology.rows, dtype=np.float32)
        lower[0] = 10.0
        proof = model.ProofInput(upper, lower, "nonfinite-row")
        processed = model.schedule_per_cta(proof, topology, 0.0)
        np.testing.assert_array_equal(
            processed[8:],
            np.full(8, topology.checkpoints, dtype=np.uint8),
        )

    def test_pruned_row_cannot_publish_offline_lower(self) -> None:
        topology = model.Topology(
            rows=24,
            columns=1024,
            grid_blocks=1,
            warps_per_block=2,
            rows_per_warp=4,
            columns_per_checkpoint=256,
        )
        upper = np.full(
            (topology.rows, topology.bound_values_per_row),
            100.0,
            dtype=np.float32,
        )
        upper[8:16, 0] = 5.0
        upper[16:, 0] = 50.0
        lower = np.zeros(topology.rows, dtype=np.float32)
        lower[0] = 10.0
        # Deliberately incoherent proof data: this pruned row must still be
        # unable to publish an offline-only score into the implementable model.
        lower[8] = 100.0
        proof = model.ProofInput(upper, lower, "no-pruned-oracle")
        processed = model.schedule_per_cta(proof, topology, 0.0)
        np.testing.assert_array_equal(
            processed[16:],
            np.full(8, topology.checkpoints, dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
