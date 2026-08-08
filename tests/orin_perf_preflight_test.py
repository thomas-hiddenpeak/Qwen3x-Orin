#!/usr/bin/env python3
"""CPU-only parser and policy tests for the Orin performance preflight."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from unittest import mock


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

    def test_fd_permission_failure_makes_gpu_audit_incomplete(self) -> None:
        observed = process(41, 1, 0, 91)
        with (
            mock.patch.object(
                PREFLIGHT,
                "_retry_os_read",
                return_value=(None, PermissionError("denied")),
            ),
            mock.patch.object(
                PREFLIGHT,
                "_process_identity_status",
                return_value=("same", None),
            ),
        ):
            audit = PREFLIGHT.scan_gpu_fd_holders(
                pathlib.Path("/proc"), {41: observed}, set(), REPOSITORY
            )

        self.assertFalse(audit.complete)
        self.assertEqual(audit.ignored_races, [])
        self.assertEqual(audit.inaccessible[0]["pid"], 41)
        self.assertEqual(
            audit.inaccessible[0]["identity_after_error"], "same"
        )

    def test_unreadable_identity_does_not_turn_permission_failure_into_race(
        self,
    ) -> None:
        observed = process(42, 1, 0, 92)
        with (
            mock.patch.object(
                PREFLIGHT,
                "_retry_os_read",
                return_value=(None, PermissionError("fd denied")),
            ),
            mock.patch.object(
                PREFLIGHT,
                "_process_identity_status",
                return_value=("unreadable", "stat denied"),
            ),
        ):
            audit = PREFLIGHT.scan_gpu_fd_holders(
                pathlib.Path("/proc"), {42: observed}, set(), REPOSITORY
            )

        self.assertFalse(audit.complete)
        self.assertEqual(
            audit.inaccessible[0]["identity_after_snapshot"], "unreadable"
        )

    def test_confirmed_process_exit_race_keeps_gpu_audit_complete(self) -> None:
        observed = process(43, 1, 0, 93)
        with (
            mock.patch.object(
                PREFLIGHT,
                "_retry_os_read",
                return_value=(None, FileNotFoundError("gone")),
            ),
            mock.patch.object(
                PREFLIGHT,
                "_process_identity_status",
                return_value=("gone", None),
            ),
        ):
            audit = PREFLIGHT.scan_gpu_fd_holders(
                pathlib.Path("/proc"), {43: observed}, set(), REPOSITORY
            )

        self.assertTrue(audit.complete)
        self.assertEqual(audit.inaccessible, [])
        self.assertEqual(audit.ignored_races[0]["kind"], "process_exited")

    def test_pid_reuse_is_not_treated_as_safe_process_exit(self) -> None:
        observed = process(45, 1, 0, 95)
        with (
            mock.patch.object(
                PREFLIGHT,
                "_retry_os_read",
                return_value=(None, FileNotFoundError("gone")),
            ),
            mock.patch.object(
                PREFLIGHT,
                "_process_identity_status",
                return_value=("replaced", "PID reused"),
            ),
        ):
            audit = PREFLIGHT.scan_gpu_fd_holders(
                pathlib.Path("/proc"), {45: observed}, set(), REPOSITORY
            )

        self.assertFalse(audit.complete)
        self.assertEqual(audit.ignored_races, [])
        self.assertEqual(
            audit.inaccessible[0]["identity_after_snapshot"], "replaced"
        )

    def test_formal_preflight_rejects_incomplete_gpu_handle_audit(self) -> None:
        before = process(44, 1, 10, 94)
        after = process(44, 1, 10, 94)
        collections = [
            PREFLIGHT.ProcessCollection({44: before}, [], []),
            PREFLIGHT.ProcessCollection({44: after}, [], []),
        ]
        incomplete = PREFLIGHT.GpuFdAudit(
            [],
            [
                {
                    "pid": 44,
                    "stage": "fd_directory",
                    "error": "denied",
                    "identity_after_error": "same",
                }
            ],
            [],
            0,
            0,
        )
        config = PREFLIGHT.PreflightConfig(
            output=REPOSITORY / ".q3x-work/preflight/not-written.json",
            samples=3,
            interval_ms=100,
            max_gr3d_percent=0.0,
            max_unexpected_cpu_percent=25.0,
            allow_pids=(),
            force=False,
        )
        with (
            mock.patch.object(
                PREFLIGHT,
                "collect_system_cpu",
                side_effect=[(100, 90), (200, 180)],
            ),
            mock.patch.object(
                PREFLIGHT,
                "collect_process_snapshot",
                side_effect=collections,
            ),
            mock.patch.object(
                PREFLIGHT,
                "collect_tegrastats",
                return_value=(
                    {"maximum_gr3d_percent": 0.0, "samples": []},
                    [],
                ),
            ),
            mock.patch.object(
                PREFLIGHT, "scan_gpu_fd_holders", return_value=incomplete
            ),
            mock.patch.object(PREFLIGHT.time, "sleep"),
        ):
            report, status = PREFLIGHT.run_preflight(config)

        self.assertEqual(status, 1)
        self.assertFalse(report["decision"]["accepted"])
        self.assertEqual(report["decision"]["result"], "collection_error")
        self.assertFalse(report["gpu_device_fd_audit"]["complete"])
        self.assertTrue(
            any(
                "GPU handle audit incomplete" in error
                for error in report["decision"]["collection_errors"]
            )
        )

    def test_output_must_resolve_under_repository_work_tree(self) -> None:
        accepted = PREFLIGHT.validate_output_path(
            REPOSITORY / ".q3x-work/preflight/evidence.json"
        )
        self.assertTrue(
            accepted.is_relative_to((REPOSITORY / ".q3x-work").resolve())
        )
        with self.assertRaises(PREFLIGHT.ConfigError):
            PREFLIGHT.validate_output_path(REPOSITORY / "evidence.json")

    def test_source_does_not_use_jetson_incomplete_gpu_cli(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("nvidia" + "-smi", source)


if __name__ == "__main__":
    unittest.main()
