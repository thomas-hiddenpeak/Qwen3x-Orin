#!/usr/bin/env python3
"""Focused CPU-only tests for the real-P40 preflight launcher."""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
EVALUATION_TOOLS = REPOSITORY / "tools/evaluation"
MODULE_PATH = EVALUATION_TOOLS / "run_sm87_aot_real_p40_arithmetic_witness.py"
sys.path.insert(0, str(EVALUATION_TOOLS))
SPEC = importlib.util.spec_from_file_location("real_p40_witness_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


class RealP40WitnessLauncherTest(unittest.TestCase):
    def test_source_preflight_sanitization_discards_raw_and_cooling_fields(self) -> None:
        source = {
            "tegrastats": {
                "samples": [
                    {
                        "gr3d_percent": 0.0,
                        "raw": "RAM 1/2MB CPU 0% FAN 30% GR3D_FREQ 0%",
                    }
                ],
                "unparsed_lines": ["FAN 30%"],
                "fan_speed": 30,
            },
            "cpu_audit": {"top_processes": [{"cmdline": "worker --fan 30"}]},
        }

        sanitized, counts = LAUNCHER.sanitize_source_report(source)

        sample = sanitized["tegrastats"]["samples"][0]
        self.assertEqual(sample, {"gr3d_percent": 0.0})
        self.assertNotIn("unparsed_lines", sanitized["tegrastats"])
        self.assertNotIn("fan_speed", sanitized["tegrastats"])
        self.assertEqual(
            sanitized["cpu_audit"]["top_processes"][0]["cmdline"],
            "[discarded incidental cooling-controller telemetry]",
        )
        self.assertEqual(counts["raw_monitor_fields_discarded"], 2)
        self.assertEqual(counts["cooling_controller_fields_discarded"], 1)
        self.assertEqual(counts["cooling_controller_text_values_discarded"], 1)

    def test_unexpected_gpu_holder_is_material_contention_and_not_accepted(self) -> None:
        state = LAUNCHER.classify_source_preflight(
            {
                "decision": {
                    "accepted": False,
                    "result": "reject_busy",
                    "reasons": [{"code": "unexpected_gpu_fd_holders"}],
                },
                "gpu_device_fd_audit": {
                    "complete": True,
                    "holders": [{"pid": 7, "allowed": False}]
                },
            },
            3,
        )

        self.assertFalse(state["core_clear"])
        self.assertTrue(state["material_contention_detected"])
        self.assertEqual(state["unowned_gpu_handle_count"], 1)

    def test_source_auxiliary_collection_gap_keeps_core_clear(self) -> None:
        state = LAUNCHER.classify_source_preflight(
            {
                "decision": {
                    "accepted": False,
                    "result": "collection_error",
                    "reasons": [],
                    "collection_errors": ["tegrastats sample unavailable"],
                },
                "gpu_device_fd_audit": {"complete": True, "holders": []},
            },
            1,
        )

        self.assertTrue(state["core_clear"])
        self.assertTrue(state["auxiliary_collection_gap"])

    def test_create_only_writer_refuses_existing_file(self) -> None:
        LAUNCHER.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LAUNCHER.WORK_ROOT) as temporary:
            path = pathlib.Path(temporary) / "preflight.json"
            LAUNCHER.write_json_create_only(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            with self.assertRaises(FileExistsError):
                LAUNCHER.write_json_create_only(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})

    def test_output_must_stay_in_current_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outside = pathlib.Path(temporary) / "evidence.json"
            with self.assertRaises(LAUNCHER.LauncherError):
                LAUNCHER.resolve_create_only_output(outside)

    def test_probe_exit_three_is_preserved_and_arguments_are_exact(self) -> None:
        config = LAUNCHER.Config(
            probe=pathlib.Path("/work/probe"),
            model_directory=pathlib.Path("/model"),
            corpus_jsonl=pathlib.Path("/work/corpus.jsonl"),
            preflight_json=pathlib.Path("/work/preflight.json"),
            evidence_json=pathlib.Path("/work/evidence.json"),
        )
        completed = subprocess.CompletedProcess([], 3)
        with mock.patch.object(LAUNCHER.subprocess, "run", return_value=completed) as run:
            status = LAUNCHER.launch_probe(config)

        self.assertEqual(status, 3)
        self.assertEqual(
            run.call_args.args[0],
            [
                "/work/probe",
                "/model",
                "/work/corpus.jsonl",
                "/work/preflight.json",
                "/work/evidence.json",
            ],
        )
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertFalse(run.call_args.kwargs["check"])

    def test_source_preflight_uses_frozen_privileged_read_only_audit(self) -> None:
        config = LAUNCHER.Config(
            probe=pathlib.Path("/work/probe"),
            model_directory=pathlib.Path("/model"),
            corpus_jsonl=pathlib.Path("/work/corpus.jsonl"),
            preflight_json=pathlib.Path("/work/preflight.json"),
            evidence_json=pathlib.Path("/work/evidence.json"),
        )
        envelope = {
            "status": 0,
            "report": {
                "decision": {"accepted": True, "result": "pass", "reasons": []},
                "gpu_device_fd_audit": {"complete": True, "holders": []},
            },
        }
        completed = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(envelope), stderr=""
        )
        with (
            mock.patch.object(LAUNCHER.shutil, "which", return_value="/usr/bin/sudo"),
            mock.patch.object(LAUNCHER, "sha256_file", return_value="a" * 64),
            mock.patch.object(LAUNCHER.subprocess, "run", return_value=completed) as run,
        ):
            report, status, execution = LAUNCHER.run_source_preflight(config)

        self.assertEqual(status, 0)
        self.assertTrue(report["decision"]["accepted"])
        self.assertEqual(execution["returncode"], 0)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/usr/bin/sudo", "-n"])
        self.assertEqual(command[-4:], ["5", "200", "0.0", "5.0"])
        self.assertNotIn("--allow-pid", command)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_launcher_hard_stop_returns_one_without_launching_probe(self) -> None:
        config = mock.Mock(preflight_json=pathlib.Path("/work/preflight.json"))
        with (
            mock.patch.object(LAUNCHER, "parse_args", return_value=config),
            mock.patch.object(
                LAUNCHER,
                "build_preflight_record",
                return_value=({"hard_stop_clear": False}, [{"code": "busy"}]),
            ),
            mock.patch.object(LAUNCHER, "write_json_create_only") as write,
            mock.patch.object(LAUNCHER, "launch_probe") as launch,
            mock.patch.object(LAUNCHER.sys, "stderr", io.StringIO()),
        ):
            status = LAUNCHER.main([])

        self.assertEqual(status, 1)
        write.assert_called_once()
        launch.assert_not_called()

    def test_cache_drop_failure_is_recorded_not_raised(self) -> None:
        before = {"observed_at_unix_ns": 1, "MemTotal_bytes": 2, "MemAvailable_bytes": 1}
        after = {"observed_at_unix_ns": 2, "MemTotal_bytes": 2, "MemAvailable_bytes": 1}
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="not permitted")
        with (
            mock.patch.object(LAUNCHER, "collect_memory_snapshot", side_effect=[before, after]),
            mock.patch.object(LAUNCHER.os, "sync"),
            mock.patch.object(
                LAUNCHER.shutil,
                "which",
                side_effect=lambda name: f"/usr/bin/{name}",
            ),
            mock.patch.object(LAUNCHER.subprocess, "run", return_value=failed) as run,
        ):
            result = LAUNCHER.run_cache_protocol()

        self.assertTrue(result["sync"]["completed"])
        self.assertTrue(result["cache_drop"]["attempted"])
        self.assertFalse(result["cache_drop"]["succeeded"])
        self.assertEqual(result["cache_drop"]["returncode"], 1)
        self.assertEqual(
            run.call_args.args[0],
            ["/usr/bin/sudo", "-n", "/usr/bin/tee", "/proc/sys/vm/drop_caches"],
        )
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_cache_drop_and_missing_auxiliary_do_not_create_hard_stop(self) -> None:
        accepted_source = {
            "decision": {
                "accepted": True,
                "result": "pass",
                "reasons": [],
                "collection_errors": [],
            },
            "gpu_device_fd_audit": {"complete": True, "holders": []},
            "tegrastats": {
                "samples": [{"gr3d_percent": 0.0, "raw": "GR3D_FREQ 0%"}],
                "unparsed_lines": [],
            },
        }
        cache = {
            "memory_before": {"MemAvailable_bytes": 10},
            "sync": {"completed": True},
            "cache_drop": {"attempted": True, "succeeded": False},
            "memory_after": {"MemAvailable_bytes": 11},
        }
        with tempfile.TemporaryDirectory(dir=LAUNCHER.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            max_frequency = root / "max_freq"
            available_frequencies = root / "available_frequencies"
            max_frequency.write_text("1300500000\n", encoding="utf-8")
            available_frequencies.write_text(
                "306000000 1300500000\n", encoding="utf-8"
            )
            config = LAUNCHER.Config(
                probe=pathlib.Path("/bin/true").resolve(),
                model_directory=root,
                corpus_jsonl=max_frequency,
                preflight_json=root / "preflight.json",
                evidence_json=root / "evidence.json",
            )
            with (
                mock.patch.object(
                    LAUNCHER,
                    "collect_host_device_identity",
                    return_value={"hostname": "orin", "machine": "aarch64"},
                ),
                mock.patch.object(
                    LAUNCHER,
                    "collect_cpu_affinity",
                    return_value={"cpu_ids": [0], "cpu_list": "0"},
                ),
                mock.patch.object(
                    LAUNCHER,
                    "collect_device_clocks",
                    return_value={
                        "cpu_policies": [],
                        "emc": {"error": "unavailable"},
                        "collection_errors": [{"error": "unavailable"}],
                    },
                ),
                mock.patch.object(
                    LAUNCHER,
                    "collect_nvpmodel",
                    return_value=({"error": "unavailable"}, False, 0),
                ),
                mock.patch.object(
                    LAUNCHER,
                    "collect_temperatures",
                    return_value={
                        "samples": [],
                        "maximum_celsius": None,
                        "operational_stop": False,
                    },
                ),
                mock.patch.object(
                    LAUNCHER,
                    "run_source_preflight",
                    return_value=(accepted_source, 0, {"returncode": 0}),
                ) as source_run,
                mock.patch.object(LAUNCHER, "run_cache_protocol", return_value=cache),
                mock.patch.object(LAUNCHER, "GPU_MAX_FREQUENCY_PATH", max_frequency),
                mock.patch.object(
                    LAUNCHER,
                    "GPU_AVAILABLE_FREQUENCIES_PATH",
                    available_frequencies,
                ),
                mock.patch.dict(LAUNCHER.os.environ, {}, clear=True),
            ):
                record, hard_stops = LAUNCHER.build_preflight_record(config)

        self.assertEqual(hard_stops, [])
        self.assertTrue(record["hard_stop_clear"])
        self.assertTrue(record["cache_drop_attempted"])
        self.assertFalse(record["cache_drop_succeeded"])
        self.assertFalse(record["nvpmodel_recorded"])
        self.assertFalse(record["device_clocks_recorded"])
        self.assertFalse(record["temperature_envelope_recorded"])
        self.assertEqual(record["freshness_limit_seconds"], 120)
        self.assertEqual(record["gpu_max_frequency_hz"], 1_300_500_000)
        self.assertEqual(record["gpu_available_max_frequency_hz"], 1_300_500_000)
        source_run.assert_called_once_with(config)
        self.assertEqual(LAUNCHER.SOURCE_PREFLIGHT_SAMPLES, 5)
        self.assertEqual(LAUNCHER.SOURCE_PREFLIGHT_INTERVAL_MS, 200)
        self.assertEqual(LAUNCHER.FROZEN_MAX_GR3D_PERCENT, 0.0)
        self.assertEqual(LAUNCHER.FROZEN_MAX_UNEXPECTED_CPU_PERCENT, 5.0)
        self.assertEqual(
            record["probe_sha256"],
            LAUNCHER.sha256_file(pathlib.Path("/bin/true").resolve()),
        )
        self.assertRegex(record["producer_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            record["source_preflight_producer_sha256"], r"^[0-9a-f]{64}$"
        )

    def test_actual_temperature_above_ninety_is_a_hard_stop(self) -> None:
        source_state = {
            "core_clear": True,
            "decision": {"result": "collection_error"},
            "reason_codes": set(),
            "unowned_gpu_handle_count": 0,
            "material_contention_detected": False,
        }

        reasons = LAUNCHER.hard_stop_reasons(
            source_state,
            1,
            {"operational_stop": True, "maximum_celsius": 90.001},
            True,
            [],
        )

        self.assertEqual(
            reasons,
            [
                {
                    "code": "temperature_above_operational_limit",
                    "maximum_celsius": 90.001,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
