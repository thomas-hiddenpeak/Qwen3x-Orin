import unittest
from copy import deepcopy

import numpy as np

from tools.evaluation.audit_prefill_state import metrics, validate_manifests


class StateMetricTest(unittest.TestCase):
    def test_repeated_tensor_does_not_dilute_relative_error(self):
        a = np.array([1.0, -2.0, 3.0, -4.0])
        b = a * 1.25
        small = metrics(a, b)
        large = metrics(np.tile(a, 4096), np.tile(b, 4096))
        self.assertAlmostEqual(small["relative_l2"], 0.25)
        self.assertAlmostEqual(large["relative_l2"], 0.25)
        self.assertAlmostEqual(small["incorrect_rmse_over_l2"] /
                               large["incorrect_rmse_over_l2"], 64.0)

    def test_zero_reference_cannot_pass_as_small_relative_error(self):
        self.assertIsNone(metrics([0.0], [1.0])["relative_l2"])
        self.assertEqual(metrics([0.0], [0.0])["relative_l2"], 0.0)

    def test_invalid_payloads_are_rejected(self):
        for a, b in (([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])):
            with self.assertRaises(ValueError):
                metrics(a, b)

    def test_prefill_boundary_rejects_trajectory_and_partial_state(self):
        reference = {"prompt_ids_u32le_sha256": "same", "prompt_tokens": 8192,
                     "capture_boundary": "prefill_commit_O1_no_decode",
                     "generated_ids": [1], "steps": [{"sequence": 8192}],
                     "state_files": {"kv_sample_positions": 8192}}
        candidate = deepcopy(reference)
        candidate["generated_ids"] = [2]
        validate_manifests(reference, candidate)  # Different predictions are measured.
        for field, value in (("capture_boundary", "generation_return"),
                             ("generated_ids", [1, 2]),
                             ("steps", [{"sequence": 8193}]),
                             ("state_files", {"kv_sample_positions": 2048}),
                             ("prompt_ids_u32le_sha256", "different")):
            invalid = deepcopy(candidate)
            invalid[field] = value
            with self.assertRaises(ValueError):
                validate_manifests(reference, invalid)


if __name__ == "__main__":
    unittest.main()
