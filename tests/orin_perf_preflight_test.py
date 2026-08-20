#!/usr/bin/env python3
"""CPU-only parser and policy tests for the Orin performance preflight."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "tools/evaluation/orin_perf_preflight.py"
SPEC = importlib.util.spec_from_file_location("orin_perf_preflight", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def process(
    pid: int,
    ppid: int,
    ticks: int,
    start: int,
    cwd: str = "/outside",
) -> object:
    return PREFLIGHT.ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        comm=f"process-{pid}",
        cmdline=f"process-{pid} --work",
        cwd=cwd,
        cpu_ticks=ticks,
        start_time_ticks=start,
    )


class OrinPerfPreflightTest(unittest.TestCase):
    def test_tegrastats_gr3d_formats_are_strict(self) -> None:
        self.assertEqual(
            PREFLIGHT.parse_gr3d_percent(
                "RAM 123/1000MB GR3D_FREQ 99%@1300 EMC_FREQ 5%@204"
            ),
            99.0,
        )
        self.assertEqual(
            PREFLIGHT.parse_gr3d_percent("GR3D_FREQ 12.5%@[918,918]"),
            12.5,
        )
        self.assertIsNone(PREFLIGHT.parse_gr3d_percent("CPU [0%@0]"))
        with self.assertRaises(PREFLIGHT.ParseError):
            PREFLIGHT.parse_gr3d_percent(
                "GR3D_FREQ 0% GR3D_FREQ 1%@306"
            )
        with self.assertRaises(PREFLIGHT.ParseError):
            PREFLIGHT.parse_gr3d_percent("GR3D_FREQ 101%")

    def test_proc_stat_parser_handles_parentheses_and_spaces(self) -> None:
        fields = ["R"] + ["0"] * 49
        fields[1] = "7"
        fields[11] = "31"
        fields[12] = "11"
        fields[19] = "123456"
        parsed = PREFLIGHT.parse_proc_stat_line(
            f"42 (worker ) name) {' '.join(fields)}"
        )
        self.assertEqual(parsed.pid, 42)
        self.assertEqual(parsed.ppid, 7)
        self.assertEqual(parsed.comm, "worker ) name")
        self.assertEqual(parsed.cpu_ticks, 42)
        self.assertEqual(parsed.start_time_ticks, 123456)

    def test_gpu_device_classification_excludes_nvmap_alone(self) -> None:
        self.assertTrue(PREFLIGHT.is_gpu_specific_device("/dev/nvhost-gpu"))
        self.assertTrue(
            PREFLIGHT.is_gpu_specific_device("/dev/nvhost-ctrl-gpu")
        )
        self.assertTrue(PREFLIGHT.is_gpu_specific_device("/dev/nvidia0"))
        self.assertTrue(
            PREFLIGHT.is_gpu_specific_device("/dev/dri/renderD128")
        )
        self.assertFalse(
            PREFLIGHT.is_gpu_specific_device("/dev/nvhost-ctrl-nvdla0")
        )
        self.assertFalse(PREFLIGHT.is_gpu_specific_device("/dev/nvmap"))
        self.assertTrue(PREFLIGHT.is_gpu_auxiliary_device("/dev/nvmap"))

    def test_allowed_pid_expands_only_to_descendants(self) -> None:
        processes = {
            10: process(10, 1, 0, 1),
            11: process(11, 10, 0, 2),
            12: process(12, 11, 0, 3),
            20: process(20, 1, 0, 4),
        }
        self.assertEqual(
            PREFLIGHT.expand_allowed_pids(processes, [10]), {10, 11, 12}
        )

    def test_cpu_delta_is_percent_of_one_core(self) -> None:
        before = {20: process(20, 1, 100, 77)}
        after = {20: process(20, 1, 150, 77)}
        samples = PREFLIGHT.cpu_process_samples(before, after, 1.0, 100)
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].percent_of_one_core, 50.0)

    def test_policy_rejects_each_unexpected_consumer(self) -> None:
        cpu = PREFLIGHT.CpuProcessSample(process(30, 1, 1, 1), 80.0)
        holders = [{"pid": 40, "allowed": False}]
        reasons = PREFLIGHT.resource_rejection_reasons(
            75.0,
            0.0,
            holders,
            [cpu],
            set(),
            25.0,
            [50],
        )
        self.assertEqual(
            [reason["code"] for reason in reasons],
            [
                "gr3d_busy",
                "unexpected_gpu_fd_holders",
                "unexpected_cpu_consumers",
                "allowed_pid_not_running",
            ],
        )

    def test_policy_accepts_explicit_consumers_when_gr3d_is_idle(self) -> None:
        cpu = PREFLIGHT.CpuProcessSample(process(30, 1, 1, 1), 80.0)
        reasons = PREFLIGHT.resource_rejection_reasons(
            0.0,
            0.0,
            [{"pid": 30, "allowed": True}],
            [cpu],
            {30},
            25.0,
            [],
        )
        self.assertEqual(reasons, [])

    def test_output_must_resolve_under_repository_work_tree(self) -> None:
        accepted = PREFLIGHT.validate_output_path(
            REPOSITORY / ".q3x-work/preflight/evidence.json"
        )
        self.assertTrue(
            accepted.is_relative_to((REPOSITORY / ".q3x-work").resolve())
        )
        with self.assertRaises(PREFLIGHT.ConfigError):
            PREFLIGHT.validate_output_path(REPOSITORY / "evidence.json")


if __name__ == "__main__":
    unittest.main()
