#!/usr/bin/env python3
"""Host-only tests for the fixed stock-vLLM P40 geometry harness."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "tools/evaluation/run_vllm_p40_geometry.py"
SPEC = importlib.util.spec_from_file_location("vllm_p40_geometry", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
GEOMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEOMETRY
SPEC.loader.exec_module(GEOMETRY)


def snapshot(
    *,
    prefill_count: int,
    prefill_seconds: float,
    prompt_tokens: int,
    cache_tokens: int = 0,
) -> object:
    local_compute = prompt_tokens - cache_tokens
    text = f"""
vllm:request_prefill_time_seconds_sum{{model_name="m",engine="0"}} {prefill_seconds}
vllm:request_prefill_time_seconds_count{{model_name="m",engine="0"}} {prefill_count}
vllm:request_prefill_kv_computed_tokens_sum{{model_name="m",engine="0"}} {local_compute}
vllm:request_prefill_kv_computed_tokens_count{{model_name="m",engine="0"}} {prefill_count}
vllm:request_prompt_tokens_sum{{model_name="m",engine="0"}} {prompt_tokens}
vllm:request_prompt_tokens_count{{model_name="m",engine="0"}} {prefill_count}
vllm:prompt_tokens_by_source_total{{model_name="m",engine="0",source="local_compute"}} {local_compute}
vllm:prompt_tokens_by_source_total{{model_name="m",engine="0",source="local_cache_hit"}} {cache_tokens}
vllm:prompt_tokens_by_source_total{{model_name="m",engine="0",source="external_kv_transfer"}} 0
vllm:prompt_tokens_cached_total{{model_name="m",engine="0"}} {cache_tokens}
vllm:request_success_total{{model_name="m",engine="0",finished_reason="length"}} {prefill_count}
vllm:generation_tokens_total{{model_name="m",engine="0"}} {prefill_count}
vllm:num_preemptions_total{{model_name="m",engine="0"}} 0
vllm:request_queue_time_seconds_sum{{model_name="m",engine="0"}} 0.01
"""
    return GEOMETRY.parse_prometheus(text)


class VllmP40GeometryTest(unittest.TestCase):
    def make_config(self) -> object:
        return GEOMETRY.GeometryConfig(
            output_dir=REPOSITORY / ".q3x-work/test-p40-geometry",
            model_dir=pathlib.Path("/readonly/model"),
            vllm_bin=pathlib.Path("/readonly/vllm"),
            uvx_bin=pathlib.Path("/usr/bin/uvx"),
            port=18093,
            allow_pids=(2599,),
            readiness_timeout_seconds=900.0,
            request_timeout_seconds=680.0,
            preflight_samples=5,
            preflight_interval_ms=500,
            dry_run=True,
        )

    def test_prometheus_parser_handles_labels_and_escapes(self) -> None:
        parsed = GEOMETRY.parse_prometheus(
            '# HELP x test\nmetric_total{source="a,b",path="a\\\\b"} 2.5e1\n'
        )
        self.assertEqual(
            GEOMETRY.metric_total(parsed, "metric_total", {"source": "a,b"}),
            25.0,
        )
        with self.assertRaises(GEOMETRY.GeometryError):
            GEOMETRY.parse_prometheus("broken metric line\n")

    def test_exact_no_cache_metric_delta_passes(self) -> None:
        before = snapshot(prefill_count=1, prefill_seconds=10.0, prompt_tokens=40_000)
        after = snapshot(prefill_count=2, prefill_seconds=18.0, prompt_tokens=80_000)
        result = GEOMETRY.validate_metric_delta(before, after)
        self.assertEqual(result["prefill_kv_tokens"], 40_000)
        self.assertEqual(result["local_cache_hit_tokens"], 0)
        self.assertEqual(result["server_prefill_phase_tokens_per_second"], 5_000)

    def test_cache_hit_metric_delta_fails_closed(self) -> None:
        before = snapshot(prefill_count=1, prefill_seconds=10.0, prompt_tokens=40_000)
        after = snapshot(
            prefill_count=2,
            prefill_seconds=18.0,
            prompt_tokens=80_000,
            cache_tokens=16,
        )
        with self.assertRaises(GEOMETRY.GeometryError):
            GEOMETRY.validate_metric_delta(before, after)

    def test_server_command_is_stock_and_fixed(self) -> None:
        command = GEOMETRY.build_server_command(self.make_config(), 8_192)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "8192")
        self.assertNotIn("--linear-backend", command)
        self.assertNotIn("--speculative-config", command)

    def test_preflight_transfers_root_evidence_before_hashing(self) -> None:
        events: list[str] = []

        def record_run(command: list[str], **_: object) -> object:
            events.append(pathlib.Path(command[2]).name)
            return subprocess.CompletedProcess(command, 0)

        def record_hash(_: pathlib.Path) -> str:
            events.append("hash")
            return "a" * 64

        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "preflight.json"
            output.write_text(
                json.dumps({"decision": {"accepted": True, "result": "pass"}}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(GEOMETRY, "_run_logged", return_value=0),
                mock.patch.object(GEOMETRY, "build_environment", return_value={}),
                mock.patch.object(
                    GEOMETRY, "collect_performance_lane_state", return_value={}
                ),
                mock.patch.object(GEOMETRY, "write_json"),
                mock.patch.object(GEOMETRY, "sha256_file", side_effect=record_hash),
                mock.patch.object(
                    GEOMETRY.subprocess, "run", side_effect=record_run
                ),
            ):
                report = GEOMETRY.run_preflight(
                    self.make_config(), output, allowed_pids=()
                )

        self.assertEqual(events, ["chown", "chmod", "hash"])
        self.assertEqual(report["raw_preflight_sha256"], "a" * 64)

    def test_evalscope_command_is_offline_and_pinned(self) -> None:
        command = GEOMETRY.build_evalscope_command(
            self.make_config(),
            pathlib.Path("/corpus.jsonl"),
            pathlib.Path("/results"),
            "p40-b8192",
        )
        self.assertIn("--offline", command)
        self.assertEqual(
            command[command.index("--from") + 1], GEOMETRY.EVALSCOPE_REQUIREMENT
        )
        self.assertEqual(command[command.index("--warmup-num") + 1], "0")
        self.assertEqual(command[command.index("--number") + 1], "1")
        direct = GEOMETRY.build_evalscope_command(
            self.make_config(),
            pathlib.Path("/corpus.jsonl"),
            pathlib.Path("/results"),
            "p40-b8192",
            pathlib.Path("/sealed/bin/evalscope"),
        )
        self.assertEqual(direct[0], "/sealed/bin/evalscope")
        self.assertNotIn("--from", direct)
        self.assertNotIn("--offline", direct)

    def test_prefill_metric_surfaces_are_not_conflated(self) -> None:
        result = {
            "evalscope": {
                "TTFT (ms)": 1000.0,
                "prompt_tokens_per_ttft_second": 40_000.0,
            },
            "server_metric_delta": {
                "prefill_phase_seconds": 8.0,
                "server_prefill_phase_tokens_per_second": 5_000.0,
            },
            "server": {
                "final_log_observations": {
                    "logger_interval_samples": [
                        {"prompt_tokens_per_second": 4_300.0}
                    ],
                    "logger_interval_prompt_throughput_maximum": 4_300.0,
                }
            },
        }
        surfaces = GEOMETRY.build_prefill_metric_surfaces(result)
        self.assertEqual(
            surfaces["logger_window"]["maximum_prompt_tokens_per_second"],
            4_300.0,
        )
        self.assertEqual(
            surfaces["logger_window"]["configured_interval_seconds"], None
        )
        self.assertEqual(
            surfaces["logger_window"]["authority"],
            "unbound_without_same_run_environment_receipt",
        )
        self.assertEqual(
            surfaces["logger_window"]["metric_contract"][
                "prompt_throughput_formula"
            ],
            "sum(iteration_stats.prompt_token_stats.computed) / "
            "(time.monotonic_now - last_log_time)",
        )
        self.assertEqual(
            surfaces["request_prefill"]["tokens_per_second"], 5_000.0
        )
        self.assertEqual(
            surfaces["external_ttft"]["prompt_tokens_per_ttft_second"],
            40_000.0,
        )
        self.assertNotEqual(
            surfaces["logger_window"]["authority"],
            surfaces["request_prefill"]["authority"],
        )

    def test_all_generated_environment_directories_stay_in_work_root(self) -> None:
        overrides = GEOMETRY.environment_overrides(8_192)
        for key in GEOMETRY.DIRECTORY_ENVIRONMENT_KEYS:
            self.assertIn(key, overrides)
            self.assertTrue(
                GEOMETRY.path_is_within(
                    pathlib.Path(overrides[key]), GEOMETRY.WORK_ROOT
                ),
                key,
            )
        rpc_base = pathlib.Path(overrides["VLLM_RPC_BASE_PATH"])
        rpc_probe = rpc_base / ("0" * GEOMETRY.ZMQ_UUID_TEXT_BYTES)
        self.assertEqual(rpc_base, GEOMETRY.WORK_ROOT.resolve())
        self.assertLessEqual(
            len(os.fsencode(str(rpc_probe))), GEOMETRY.ZMQ_IPC_PATH_MAX_BYTES
        )

    def test_route_and_python_environment_are_not_inherited(self) -> None:
        hostile = {
            "VLLM_LINEAR_BACKEND": "humming",
            "FLASHINFER_DISABLE_JIT": "1",
            "TRITON_CACHE_AUTOTUNING": "0",
            "PYTHONPATH": "/untrusted/python",
            "VIRTUAL_ENV": "/untrusted/venv",
            "HOME": "/untrusted/home",
            "VLLM_NO_USAGE_STATS": "0",
            "VLLM_DO_NOT_TRACK": "0",
            "VLLM_LOG_STATS_INTERVAL": "37.5",
            "FLASHINFER_NO_DOWNLOAD": "0",
        }
        with mock.patch.dict(os.environ, hostile):
            effective = GEOMETRY.build_environment(8_192, create=False)
        for key in (
            "VLLM_LINEAR_BACKEND",
            "FLASHINFER_DISABLE_JIT",
            "TRITON_CACHE_AUTOTUNING",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        ):
            self.assertNotIn(key, effective)
        self.assertEqual(effective["PATH"], GEOMETRY.CONTROLLED_PATH)
        self.assertEqual(effective["HOME"], str(GEOMETRY.WORK_ROOT / "home"))
        self.assertEqual(effective["VLLM_NO_USAGE_STATS"], "1")
        self.assertEqual(effective["VLLM_DO_NOT_TRACK"], "1")
        self.assertEqual(effective["VLLM_LOG_STATS_INTERVAL"], "10.0")
        self.assertEqual(effective["FLASHINFER_NO_DOWNLOAD"], "1")
        receipt = GEOMETRY.environment_receipt(8_192)
        self.assertEqual(
            receipt["controlled_overrides"]["VLLM_LOG_STATS_INTERVAL"],
            "10.0",
        )

    def test_alternate_pycache_prefix_must_remain_empty(self) -> None:
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=GEOMETRY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            observed = GEOMETRY.validate_empty_workspace_directory(
                root, "test bytecode prefix"
            )
            self.assertTrue(observed["empty"])
            (root / "stale.pyc").write_bytes(b"stale")
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "must be empty"):
                GEOMETRY.validate_empty_workspace_directory(
                    root, "test bytecode prefix"
                )

    def test_corpus_identity_and_contract_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "corpus.jsonl"
            request = {
                "model": GEOMETRY.MODEL_NAME,
                "prompt": list(range(GEOMETRY.PROMPT_TOKENS)),
                "max_tokens": 1,
                "temperature": 0.0,
                "seed": 42,
                "stream": True,
            }
            raw = (json.dumps(request) + "\n").encode()
            path.write_bytes(raw)
            result = GEOMETRY.validate_corpus(path, hashlib.sha256(raw).hexdigest())
            self.assertEqual(result["prompt_tokens"], 40_000)
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "hash mismatch"):
                GEOMETRY.validate_corpus(path, "0" * 64)

    def test_hashed_identity_rejects_symlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target"
            target.write_bytes(b"identity")
            link = root / "link"
            link.symlink_to(target)
            digest = hashlib.sha256(b"identity").hexdigest()
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "symlink"):
                GEOMETRY.validate_hashed_files(root, {"link": digest}, "test")

    def test_evalscope_summary_requires_one_exact_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "Total Requests": 1,
                        "Success Requests": 1,
                        "Failed Requests": 0,
                        "Avg Input Tokens": 40_000,
                        "Avg Output Tokens": 1,
                        "TTFT (ms)": 8_000.0,
                        "Avg Latency (s)": 8.1,
                    }
                )
            )
            result = GEOMETRY.validate_evalscope_summary(path)
            self.assertEqual(result["TTFT (ms)"], 8_000.0)
            bad = json.loads(path.read_text())
            bad["Success Requests"] = 0
            path.write_text(json.dumps(bad))
            with self.assertRaises(GEOMETRY.GeometryError):
                GEOMETRY.validate_evalscope_summary(path)

    def test_output_directory_must_stay_below_work_root(self) -> None:
        with self.assertRaises(GEOMETRY.GeometryError):
            GEOMETRY.validate_output_dir(pathlib.Path("/tmp/q3x-geometry"))
        accepted = GEOMETRY.validate_output_dir(
            REPOSITORY / ".q3x-work/new-geometry", may_exist=True
        )
        self.assertTrue(GEOMETRY.path_is_within(accepted, GEOMETRY.WORK_ROOT))

    def test_plan_starts_with_only_the_target_like_whole_prompt_budget(self) -> None:
        plan = GEOMETRY.build_plan(self.make_config())
        self.assertEqual(
            [entry["max_num_batched_tokens"] for entry in plan["budgets"]],
            [40_000],
        )
        self.assertEqual(
            [entry["planned_scheduled_forwards"] for entry in plan["budgets"]],
            [1],
        )
        self.assertEqual(
            plan["deferred_explanatory_budgets"], [8_192, 4_096, 2_048]
        )

    def test_vendored_deep_gemm_availability_probe_is_pinned(self) -> None:
        relative = (
            "vllm/third_party/deep_gemm/"
            "_C.cpython-313-aarch64-linux-gnu.so"
        )
        self.assertIn(relative, GEOMETRY.VLLM_RUNTIME_FILE_SHA256)
        self.assertIn(relative, GEOMETRY.VLLM_RUNTIME_BUILD_IDS)
        self.assertIn("utils/deep_gemm.py", GEOMETRY.VLLM_SOURCE_SHA256)
        self.assertIn(
            "third_party/deep_gemm/__init__.py", GEOMETRY.VLLM_SOURCE_SHA256
        )

    def test_logger_window_source_semantics_are_pinned(self) -> None:
        self.assertEqual(
            GEOMETRY.VLLM_SOURCE_SHA256["envs.py"],
            "e078b0acb8e658faa6a8144d13a9036e2a82af40b348f419489a2607247cd936",
        )
        self.assertEqual(
            GEOMETRY.VLLM_SOURCE_SHA256[
                "entrypoints/serve/utils/server_utils.py"
            ],
            "52a4e59852a5c478ddf9d4dec1737dd5c8a807000d646ee61c9828d49f2d8491",
        )
        contract = GEOMETRY.logger_window_contract()
        self.assertIsNone(contract["configured_interval_seconds"])
        self.assertEqual(
            contract["authority"],
            "unbound_without_same_run_environment_receipt",
        )
        self.assertFalse(
            contract["configuration"]["same_run_receipt_binding"][
                "same_run_environment_receipt_bound"
            ]
        )
        for relative in contract["source_bindings"]:
            self.assertIn(relative, GEOMETRY.VLLM_SOURCE_SHA256)
        self.assertIn(
            "time.monotonic()",
            contract["elapsed_time_denominator_semantics"],
        )
        self.assertEqual(
            contract["scope_exclusions"],
            ["request_elapsed_time", "pure_prefill_latency", "external_ttft"],
        )

    def test_logger_window_receipt_binding_is_strict(self) -> None:
        formal = GEOMETRY.logger_window_contract(
            GEOMETRY.environment_receipt(8_192)
        )
        self.assertEqual(formal["configured_interval_seconds"], 10.0)
        self.assertEqual(
            formal["authority"],
            "same_run_environment_receipt_bound_"
            "supporting_service_telemetry_only",
        )
        inherited = GEOMETRY.environment_receipt(8_192)
        inherited["base_allowlist"].append("VLLM_LOG_STATS_INTERVAL")
        with self.assertRaisesRegex(GEOMETRY.GeometryError, "inherits forbidden"):
            GEOMETRY.logger_window_contract(inherited)
        wrong = GEOMETRY.environment_receipt(8_192)
        wrong["controlled_overrides"]["VLLM_LOG_STATS_INTERVAL"] = "20.0"
        with self.assertRaisesRegex(GEOMETRY.GeometryError, "does not bind"):
            GEOMETRY.logger_window_contract(wrong)

    def test_only_unlinked_posix_semaphore_matches_deleted_mapping_rule(self) -> None:
        accepted = "/dev/shm/sem.5zLMSe (deleted)"
        self.assertIsNotNone(GEOMETRY.DELETED_POSIX_SEMAPHORE.fullmatch(accepted))
        self.assertIsNone(
            GEOMETRY.DELETED_POSIX_SEMAPHORE.fullmatch(
                "/cache/kernel.so (deleted)"
            )
        )

    def test_measured_tegrastats_envelope_is_windowed_and_strict(self) -> None:
        self.assertEqual(GEOMETRY.JETSON_MAX_TEMPERATURE_MILLIC, 85_000)
        self.assertEqual(
            GEOMETRY.JETSON_THROTTLE_RISK_TEMPERATURE_MILLIC, 90_000
        )
        cpu = ",".join(["0%@2201"] * 12)
        lines = [
            (
                f"t RAM 1/62878MB CPU [{cpu}] EMC_FREQ 0%@3200 "
                f"GR3D_FREQ {gr3d}%@[1300,1300] "
                "cpu@63C/63C gpu@55C/55C "
                "tj@63C/63C"
            )
            for gr3d in (0, 20, 80, 40, 0, 0)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "tegrastats.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            observed = GEOMETRY.validate_measurement_telemetry(path, 1, 4)
            self.assertEqual(observed["pre_request_sample_count"], 1)
            self.assertEqual(observed["in_request_sample_count"], 3)
            self.assertEqual(observed["post_request_sample_count"], 2)
            self.assertEqual(observed["maximum_gr3d_percent"], 80.0)
            self.assertTrue(
                observed["clock_contract"][
                    "cpu_and_emc_stable_for_every_sample"
                ]
            )
            path.write_text(
                ("\n".join(lines).replace("@[1300,1300]", "@[1294,1297]"))
                + "\n",
                encoding="utf-8",
            )
            effective_gpu = GEOMETRY.validate_measurement_telemetry(path, 1, 4)
            self.assertEqual(
                effective_gpu["clock_contract"]["gpu_effective_mhz_observed"],
                [1294, 1297],
            )
            self.assertEqual(
                effective_gpu["continuous_exact_clock_gate_roles"],
                ["cpu", "emc"],
            )
            self.assertTrue(
                effective_gpu["temperature_is_not_a_clock_or_throttle_proxy"]
            )
            path.write_text(
                ("\n".join(lines).replace("EMC_FREQ 0%@3200", "EMC_FREQ 0%@3199"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "clock/RAM drift"):
                GEOMETRY.validate_measurement_telemetry(path, 1, 4)
            path.write_text(
                ("\n".join(lines).replace("cpu@63C/63C", "cpu@63C/85C"))
                + "\n",
                encoding="utf-8",
            )
            exact_limit = GEOMETRY.validate_measurement_telemetry(path, 1, 4)
            self.assertEqual(exact_limit["maximum_temperature_c"]["cpu"], 85.0)
            path.write_text(
                ("\n".join(lines).replace("cpu@63C/63C", "cpu@63C/85.1C"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(GEOMETRY.GeometryError):
                GEOMETRY.validate_measurement_telemetry(path, 1, 4)

    def test_thermal_cooldown_requires_stable_hysteresis_samples(self) -> None:
        def thermal(value: int) -> dict:
            return {
                "zones": {
                    name: {"temperature_millic": value}
                    for name in ("cpu-thermal", "gpu-thermal", "tj-thermal")
                }
            }

        with (
            mock.patch.object(
                GEOMETRY,
                "collect_thermal_state",
                side_effect=(thermal(66_000), thermal(65_000), thermal(64_000)),
            ) as collect,
            mock.patch.object(GEOMETRY.time, "sleep") as sleep,
        ):
            receipt = GEOMETRY.wait_for_thermal_cooldown(
                "test",
                target_millic=65_000,
                stable_samples=2,
                interval_seconds=0.1,
                timeout_seconds=10.0,
            )
        self.assertEqual(len(receipt["samples"]), 3)
        self.assertEqual(
            receipt["samples"][-1]["consecutive_at_or_below_target"], 2
        )
        self.assertEqual(collect.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        with self.assertRaisesRegex(GEOMETRY.GeometryError, "configuration"):
            GEOMETRY.wait_for_thermal_cooldown(
                "invalid", target_millic=GEOMETRY.JETSON_MAX_TEMPERATURE_MILLIC
            )

    def test_formal_host_probe_is_bracketed_by_clean_admission(self) -> None:
        events: list[str] = []
        runtime_receipt = {
            "prefix": "p",
            "python": "py",
            "executable": "e",
            "site_packages": "s",
            "distribution_manifest": "d",
        }
        runtime = GEOMETRY.EvalScopeRuntime(
            pathlib.Path("/sealed"),
            pathlib.Path("/sealed/bin/python"),
            pathlib.Path("/sealed/bin/evalscope"),
            runtime_receipt,
        )
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=GEOMETRY.WORK_ROOT) as temporary:
            output = pathlib.Path(temporary) / "formal"
            config = dataclasses.replace(
                self.make_config(), output_dir=output, dry_run=False
            )

            def preflight(_config: object, path: pathlib.Path, _pids: object) -> dict:
                events.append(f"preflight:{path.name}")
                return {"decision": {"accepted": True}}

            def host_runtime(_config: object) -> dict:
                events.append("host-runtime")
                return {"valid": True}

            def budget(_config: object, _runtime: object, value: int) -> dict:
                events.append(f"budget:{value}")
                return {"budget": value, "valid": True}

            with (
                mock.patch.object(GEOMETRY, "git_identity", return_value={}),
                mock.patch.object(GEOMETRY, "validate_static_inputs", return_value={}),
                mock.patch.object(
                    GEOMETRY, "resolve_evalscope_runtime", return_value=runtime
                ),
                mock.patch.object(GEOMETRY, "build_plan", return_value={}),
                mock.patch.object(GEOMETRY, "run_preflight", side_effect=preflight),
                mock.patch.object(
                    GEOMETRY, "validate_host_runtime", side_effect=host_runtime
                ),
                mock.patch.object(GEOMETRY, "run_budget", side_effect=budget),
                mock.patch.object(
                    GEOMETRY,
                    "validate_evalscope_environment",
                    return_value=runtime_receipt,
                ),
                mock.patch.object(
                    GEOMETRY,
                    "validate_empty_pycache_prefix",
                    return_value={"empty": True},
                ),
            ):
                result = GEOMETRY.run(config)
        self.assertTrue(result["valid"])
        self.assertEqual(
            events[:3],
            [
                "preflight:preflight-before-host-runtime.json",
                "host-runtime",
                "preflight:preflight-after-host-runtime.json",
            ],
        )
        self.assertEqual(
            events[3:],
            ["budget:40000"],
        )

    def test_process_group_cleanup_reaps_leader_without_escalation(self) -> None:
        process = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        started = time.monotonic()
        cleanup = GEOMETRY.stop_process_group(process)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0)
        self.assertEqual(cleanup["signals"], ["SIGINT"])
        self.assertTrue(cleanup["group_empty"])

    def test_rpc_socket_cleanup_is_exact_and_leaves_no_entry(self) -> None:
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        path = GEOMETRY.WORK_ROOT / "00000000-0000-0000-0000-000000000001"
        self.assertFalse(path.exists())
        handle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            handle.bind(str(path))
            observed = GEOMETRY.collect_vllm_rpc_sockets()
            self.assertEqual([record["path"] for record in observed], [str(path)])
            cleanup = GEOMETRY.cleanup_vllm_rpc_sockets()
            self.assertEqual(cleanup["observed"], observed)
            self.assertEqual(cleanup["removed"], observed)
            self.assertEqual(cleanup["remaining"], [])
            self.assertFalse(path.exists())
        finally:
            handle.close()
            path.unlink(missing_ok=True)

    def test_uuid_named_non_socket_is_never_deleted(self) -> None:
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        path = GEOMETRY.WORK_ROOT / "00000000-0000-0000-0000-000000000002"
        self.assertFalse(path.exists())
        try:
            path.write_bytes(b"not a socket")
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "not a socket"):
                GEOMETRY.cleanup_vllm_rpc_sockets()
            self.assertEqual(path.read_bytes(), b"not a socket")
        finally:
            path.unlink(missing_ok=True)

    def test_logged_command_rejects_surviving_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "leak.log"
            child_code = "import time; time.sleep(30)"
            parent_code = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
                "time.sleep(0.1)"
            )
            started = time.monotonic()
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "live process-group"):
                GEOMETRY._run_logged(
                    [sys.executable, "-c", parent_code, child_code],
                    {},
                    log,
                    5.0,
                )
            self.assertLess(time.monotonic() - started, 5.0)

    def test_final_server_log_route_evidence_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "server.log"
            required = "\n".join(
                (
                    "08-12 05:00:00 [core.py:116] speculative_config=None, "
                    "quantization=modelopt_mixed,",
                    "enable_prefix_caching=False, enable_chunked_prefill=True,",
                    "non-default args: {'max_num_batched_tokens': 8192}, "
                    "kernel_config=KernelConfig(linear_backend='auto', "
                    "enable_flashinfer_autotune=True)",
                    "Selected MarlinFP8ScaledMMLinearKernel for "
                    "ModelOptFp8LinearMethod",
                    "Using AttentionBackendEnum.FLASHINFER backend.",
                    "Using Triton/FLA GDN prefill kernel "
                    "(requested=auto, head_k_dim=128).",
                    "Cache the graph of compile range (1, 8192) for later use",
                    "saved AOT compiled function to /workspace/cache/model",
                    "torch.compile took 12.5 s in total",
                    "Initial profiling/warmup run took 3.25 s",
                    'INFO: 127.0.0.1 - "POST /v1/completions HTTP/1.1" '
                    "200 OK",
                    "08-12 05:00:10 [jit_monitor.py:135] Kernel JIT "
                    "compilation during inference: warmup_kernel. This causes "
                    "a latency spike.",
                    "08-12 05:00:20 [loggers.py:310] Engine 000: Avg prompt "
                    "throughput: 4300.0 tokens/s, Avg generation throughput: "
                    "0.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV "
                    "cache usage: 0.0%, Prefix cache hit rate: 0.0%",
                    'INFO: 127.0.0.1 - "POST /v1/completions HTTP/1.1" '
                    "200 OK",
                )
            )
            log.write_text(required)
            observed = GEOMETRY.observe_server_log(
                log,
                8_192,
                same_run_environment_receipt=GEOMETRY.environment_receipt(
                    8_192
                ),
            )
            self.assertEqual(observed["schema_version"], 3)
            self.assertEqual(
                observed["parser_identity"],
                "q3x.vllm_server_log_observation.lf_physical_lines.v3",
            )
            self.assertTrue(all(observed["required_route_probe_hits"].values()))
            self.assertEqual(
                observed["logger_interval_prompt_throughput_maximum"], 4300.0
            )
            self.assertNotIn("ten_second_logger_samples", observed)
            self.assertEqual(
                observed["backend_evidence"]["quantization"], "modelopt_mixed"
            )
            self.assertEqual(
                observed["backend_evidence"]["requested_linear_backend"],
                "auto",
            )
            self.assertEqual(
                observed["logger_interval_samples"][0][
                    "position_by_fixed_harness_http_response_start_"
                    "access_log_order"
                ],
                "after_first_before_second_http_response_start_access_log_"
                "observation",
            )
            self.assertEqual(
                observed["logger_interval_samples"][0][
                    "configured_interval_seconds"
                ],
                10.0,
            )
            self.assertEqual(
                observed["logger_interval_samples"][0]["authority"],
                "same_run_environment_receipt_bound_"
                "supporting_service_telemetry_only",
            )
            self.assertEqual(
                len(observed["http_response_start_access_log_observations"]),
                2,
            )
            self.assertEqual(
                observed["http_response_start_access_log_observations"][0][
                    "semantics"
                ],
                "access_log_emitted_at_asgi_http_response_start",
            )
            self.assertEqual(
                observed["backend_evidence"]["nvfp4_linear_selection"][
                    "evidence_status"
                ],
                "source-resolved/runtime-hit-unproven",
            )
            self.assertNotIn("api_completion_responses", observed)
            self.assertEqual(
                observed["startup_and_compilation"]["runtime_jit_events"][0][
                    "kernel"
                ],
                "warmup_kernel",
            )
            self.assertEqual(
                observed["startup_and_compilation"]["runtime_jit_events"][0][
                    "http_response_start_access_log_observations_seen"
                ],
                1,
            )
            self.assertEqual(
                observed["startup_and_compilation"]["runtime_jit_events"][0][
                    "request_phase_from_http_response_start_access_log"
                ],
                "indeterminate_including_streaming_warmup",
            )
            self.assertNotIn(
                "completed_completion_responses_seen",
                observed["startup_and_compilation"]["runtime_jit_events"][0],
            )
            self.assertEqual(
                observed["startup_and_compilation"][
                    "torch_compile_total_seconds"
                ],
                12.5,
            )
            self.assertFalse(observed["authority"]["performance_promotion"])
            self.assertEqual(
                observed["authority"][
                    "request_phase_from_http_response_start_access_log"
                ],
                "not_established",
            )
            self.assertEqual(
                observed["authority"]["logger_configured_interval"],
                "same_run_environment_receipt_bound_"
                "supporting_service_telemetry_only",
            )
            self.assertEqual(
                observed["authority"][
                    "logger_actual_monotonic_elapsed_denominator"
                ],
                "not_exposed_by_printed_server_log",
            )
            self.assertEqual(
                observed["logger_window_contract"][
                    "prompt_token_numerator_semantics"
                ],
                "sum of iteration_stats.prompt_token_stats.computed recorded by "
                "the local LoggingStatLogger since its previous reset; cached "
                "and transferred prompt tokens are excluded",
            )
            self.assertIn(
                "not proof of client header receipt",
                observed["note"],
            )
            self.assertIn(
                "bound by the same-run environment receipt",
                observed["note"],
            )
            self.assertIn("warmup completion", observed["note"])
            log.write_text(
                required.replace(
                    "Using AttentionBackendEnum.FLASHINFER backend.\n", ""
                )
            )
            missing_route = GEOMETRY.observe_server_log(
                log,
                8_192,
                same_run_environment_receipt=GEOMETRY.environment_receipt(
                    8_192
                ),
            )
            self.assertFalse(missing_route["route_validation"]["passed"])
            self.assertIn(
                "flashinfer_attention",
                missing_route["route_validation"][
                    "missing_required_route_probes"
                ],
            )
            self.assertEqual(
                missing_route["logger_interval_prompt_throughput_maximum"],
                4300.0,
            )
            log.write_text(required + "\nUsing Humming")
            forbidden_route = GEOMETRY.observe_server_log(
                log,
                8_192,
                same_run_environment_receipt=GEOMETRY.environment_receipt(
                    8_192
                ),
            )
            self.assertFalse(forbidden_route["route_validation"]["passed"])
            self.assertIn(
                "Using Humming",
                forbidden_route["route_validation"][
                    "forbidden_route_matches"
                ],
            )

    def test_measured_telemetry_failure_is_returned_for_post_collection(
        self,
    ) -> None:
        primary = GEOMETRY.GeometryError("thermal envelope crossed")
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = pathlib.Path(temporary) / "measured-tegrastats.log"
            telemetry.write_text("invalid thermal sample\n", encoding="utf-8")
            monitor = mock.Mock()
            with (
                mock.patch.object(
                    GEOMETRY,
                    "start_telemetry_monitor",
                    return_value=(monitor, {"started": True}),
                ),
                mock.patch.object(GEOMETRY, "wait_for_telemetry_lines"),
                mock.patch.object(
                    GEOMETRY,
                    "_telemetry_complete_line_count",
                    side_effect=(1, 4),
                ),
                mock.patch.object(
                    GEOMETRY,
                    "run_evalscope",
                    return_value={"TTFT (ms)": 1.0},
                ),
                mock.patch.object(
                    GEOMETRY,
                    "stop_telemetry_monitor",
                    return_value={"group_empty": True},
                ),
                mock.patch.object(
                    GEOMETRY,
                    "validate_measurement_telemetry",
                    side_effect=primary,
                ),
            ):
                result, receipt, failure = GEOMETRY.run_measured_evalscope(
                    self.make_config(),
                    mock.Mock(),
                    40_000,
                    pathlib.Path(temporary) / "evalscope",
                    "p40",
                    pathlib.Path(temporary) / "evalscope.log",
                    telemetry,
                )

            self.assertEqual(result, {"TTFT (ms)": 1.0})
            self.assertIs(failure, primary)
            self.assertIn("thermal envelope crossed", receipt["validation_error"])
            self.assertTrue(telemetry.with_suffix(".json").is_file())

    def test_telemetry_receipt_write_failure_does_not_replace_primary(
        self,
    ) -> None:
        primary = GEOMETRY.GeometryError("thermal envelope crossed")
        original_write_json = GEOMETRY.write_json
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = pathlib.Path(temporary) / "measured-tegrastats.log"
            telemetry.write_text("invalid thermal sample\n", encoding="utf-8")

            def fail_telemetry_receipt(path: pathlib.Path, value: object) -> None:
                if path == telemetry.with_suffix(".json"):
                    raise OSError("telemetry receipt disk failure")
                original_write_json(path, value)

            with (
                mock.patch.object(
                    GEOMETRY,
                    "start_telemetry_monitor",
                    return_value=(mock.Mock(), {"started": True}),
                ),
                mock.patch.object(GEOMETRY, "wait_for_telemetry_lines"),
                mock.patch.object(
                    GEOMETRY,
                    "_telemetry_complete_line_count",
                    side_effect=(1, 4),
                ),
                mock.patch.object(
                    GEOMETRY,
                    "run_evalscope",
                    return_value={"TTFT (ms)": 1.0},
                ),
                mock.patch.object(
                    GEOMETRY,
                    "stop_telemetry_monitor",
                    return_value={"group_empty": True},
                ),
                mock.patch.object(
                    GEOMETRY,
                    "validate_measurement_telemetry",
                    side_effect=primary,
                ),
                mock.patch.object(
                    GEOMETRY, "write_json", side_effect=fail_telemetry_receipt
                ),
            ):
                result, receipt, failure = GEOMETRY.run_measured_evalscope(
                    self.make_config(),
                    mock.Mock(),
                    40_000,
                    pathlib.Path(temporary) / "evalscope",
                    "p40",
                    pathlib.Path(temporary) / "evalscope.log",
                    telemetry,
                )

            self.assertEqual(result, {"TTFT (ms)": 1.0})
            self.assertIs(failure, primary)
            self.assertIn("thermal envelope crossed", receipt["validation_error"])

    def run_mocked_invalid_budget(
        self,
        output: pathlib.Path,
        *,
        primary: BaseException,
        metric_failure: BaseException,
        write_failure_name: str | None = None,
    ) -> tuple[object, object]:
        cache = {"file_count": 0, "bytes": 0, "files": {}}
        cache_snapshot = mock.Mock(return_value=cache)
        fake_server = mock.Mock(pid=424242)
        termination = {"cleanup_started": False, "received": []}
        original_write_json = GEOMETRY.write_json

        def maybe_fail_write(path: pathlib.Path, value: object) -> None:
            if write_failure_name is not None and path.name == write_failure_name:
                raise OSError(f"injected {write_failure_name} write failure")
            original_write_json(path, value)

        geometry_patches = {
            "collect_vllm_rpc_sockets": mock.Mock(return_value=[]),
            "wait_for_thermal_cooldown": mock.Mock(return_value={}),
            "run_preflight": mock.Mock(
                return_value={"decision": {"accepted": True}}
            ),
            "install_termination_handlers": mock.Mock(
                return_value=(termination, {})
            ),
            "restore_termination_handlers": mock.Mock(),
            "wait_for_server": mock.Mock(),
            "run_evalscope": mock.Mock(return_value={}),
            "snapshot_compilation_caches": cache_snapshot,
            "collect_server_process_tree_runtime": mock.Mock(
                return_value={"processes": []}
            ),
            "run_measured_evalscope": mock.Mock(
                return_value=(
                    {},
                    {"validation_error": repr(primary)},
                    primary,
                )
            ),
            "collect_performance_lane_state": mock.Mock(
                return_value={"lane": "captured"}
            ),
            "fetch_text": mock.Mock(side_effect=("before", "after")),
            "parse_prometheus": mock.Mock(return_value=mock.Mock()),
            "validate_metric_delta": mock.Mock(side_effect=metric_failure),
            "stop_process_group": mock.Mock(return_value={"group_empty": True}),
            "cleanup_vllm_rpc_sockets": mock.Mock(return_value={}),
            "observe_server_log": mock.Mock(
                return_value={"route_validation": {"passed": True}}
            ),
            "write_json": mock.Mock(side_effect=maybe_fail_write),
        }
        config = dataclasses.replace(self.make_config(), output_dir=output)
        with (
            mock.patch.multiple(GEOMETRY, **geometry_patches),
            mock.patch.object(
                GEOMETRY.subprocess, "Popen", return_value=fake_server
            ),
            mock.patch.multiple(
                GEOMETRY.os,
                getpgid=mock.Mock(return_value=424242),
                getsid=mock.Mock(return_value=424242),
            ),
            mock.patch.multiple(
                GEOMETRY.time,
                monotonic=mock.Mock(side_effect=(0.0, 1.0, 21.0)),
                sleep=mock.Mock(),
            ),
        ):
            GEOMETRY.run_budget(config, mock.Mock(), 40_000)
        return cache_snapshot, geometry_patches["observe_server_log"]

    def test_post_and_cleanup_receipt_failures_do_not_replace_primary(
        self,
    ) -> None:
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        for receipt_name in ("post-request-evidence.json", "cleanup.json"):
            with self.subTest(receipt=receipt_name):
                primary = GEOMETRY.GeometryError("thermal envelope crossed")
                with tempfile.TemporaryDirectory(
                    dir=GEOMETRY.WORK_ROOT
                ) as temporary:
                    with self.assertRaisesRegex(
                        GEOMETRY.GeometryError, "thermal envelope crossed"
                    ):
                        self.run_mocked_invalid_budget(
                            pathlib.Path(temporary) / "formal",
                            primary=primary,
                            metric_failure=GEOMETRY.GeometryError(
                                "delta invalid"
                            ),
                            write_failure_name=receipt_name,
                        )

    def test_keyboard_interrupt_during_metric_collection_is_not_hidden(
        self,
    ) -> None:
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=GEOMETRY.WORK_ROOT) as temporary:
            with self.assertRaises(KeyboardInterrupt):
                self.run_mocked_invalid_budget(
                    pathlib.Path(temporary) / "formal",
                    primary=GEOMETRY.GeometryError("thermal envelope crossed"),
                    metric_failure=KeyboardInterrupt(),
                )
            run_dir = pathlib.Path(temporary) / "formal" / "b40000"
            self.assertTrue((run_dir / "server.log").is_file())
            self.assertTrue((run_dir / "server-log-observations.json").is_file())

    def test_primary_request_failure_still_collects_post_request_evidence(
        self,
    ) -> None:
        primary = GEOMETRY.GeometryError("thermal envelope crossed")
        cache = {"file_count": 0, "bytes": 0, "files": {}}
        fake_server = mock.Mock(pid=424242)
        GEOMETRY.WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=GEOMETRY.WORK_ROOT) as temporary:
            output = pathlib.Path(temporary) / "formal"
            config = dataclasses.replace(self.make_config(), output_dir=output)
            termination = {"cleanup_started": False, "received": []}
            geometry_patches = {
                "collect_vllm_rpc_sockets": mock.Mock(return_value=[]),
                "wait_for_thermal_cooldown": mock.Mock(return_value={}),
                "run_preflight": mock.Mock(
                    return_value={"decision": {"accepted": True}}
                ),
                "install_termination_handlers": mock.Mock(
                    return_value=(termination, {})
                ),
                "restore_termination_handlers": mock.Mock(),
                "wait_for_server": mock.Mock(),
                "run_evalscope": mock.Mock(return_value={}),
                "snapshot_compilation_caches": mock.Mock(return_value=cache),
                "collect_server_process_tree_runtime": mock.Mock(
                    return_value={"processes": []}
                ),
                "run_measured_evalscope": mock.Mock(
                    return_value=(
                        {},
                        {"validation_error": repr(primary)},
                        primary,
                    )
                ),
                "collect_performance_lane_state": mock.Mock(
                    return_value={"lane": "captured"}
                ),
                "fetch_text": mock.Mock(side_effect=("before", "after")),
                "parse_prometheus": mock.Mock(return_value=mock.Mock()),
                "validate_metric_delta": mock.Mock(
                    side_effect=GEOMETRY.GeometryError("delta invalid")
                ),
                "stop_process_group": mock.Mock(
                    return_value={"group_empty": True}
                ),
                "cleanup_vllm_rpc_sockets": mock.Mock(return_value={}),
                "observe_server_log": mock.Mock(
                    return_value={"route_validation": {"passed": True}}
                ),
            }
            with (
                mock.patch.multiple(GEOMETRY, **geometry_patches),
                mock.patch.object(
                    GEOMETRY.subprocess, "Popen", return_value=fake_server
                ),
                mock.patch.multiple(
                    GEOMETRY.os,
                    getpgid=mock.Mock(return_value=424242),
                    getsid=mock.Mock(return_value=424242),
                ),
                mock.patch.multiple(
                    GEOMETRY.time,
                    monotonic=mock.Mock(side_effect=(0.0, 1.0, 21.0)),
                    sleep=mock.Mock(),
                ),
            ):
                with self.assertRaisesRegex(
                    GEOMETRY.GeometryError, "thermal envelope crossed"
                ):
                    GEOMETRY.run_budget(config, mock.Mock(), 40_000)

            run_dir = output / "b40000"
            for name in (
                "server-runtime-after-request.json",
                "lane-state-after-request.json",
                "metrics-after-attempt-001.prom",
                "cache-after-measured.json",
                "post-request-evidence.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)
            partial = json.loads((run_dir / "partial-result.json").read_text())
            self.assertFalse(partial["valid"])
            self.assertEqual(
                partial["prefill_metric_surfaces"]["logger_window"][
                    "authority"
                ],
                "unbound_without_same_run_environment_receipt",
            )
            evidence = json.loads(
                (run_dir / "post-request-evidence.json").read_text()
            )
            self.assertIn(
                "thermal envelope crossed",
                evidence["primary_request_error"]["repr"],
            )
            self.assertEqual(
                evidence["collectors"]["prometheus_after"]["status"], "fail"
            )
            self.assertIn(
                "delta invalid",
                evidence["collectors"]["prometheus_after"]["attempts"][0][
                    "validation_error"
                ]["repr"],
            )

    def test_server_log_preserves_cr_progress_inside_lf_physical_lines(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = pathlib.Path(temporary) / "server.log"
            route_records = (
                b"speculative_config=None, quantization=modelopt_mixed,",
                b"enable_prefix_caching=False, enable_chunked_prefill=True,",
                b"non-default args: {'max_num_batched_tokens': 8192}, "
                b"kernel_config=KernelConfig(linear_backend='auto')",
                b"Using AttentionBackendEnum.FLASHINFER backend.",
                b"Using Triton/FLA GDN prefill kernel "
                b"(requested=auto, head_k_dim=128).",
            )
            logger_before_start = (
                b"progress=10%\r08-12 05:00:20 [loggers.py:310] Engine 000: "
                b"Avg prompt throughput: 100.0 tokens/s, Avg generation "
                b"throughput: 0.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
                b"GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%\r"
                b'INFO: 127.0.0.1 - "POST /v1/completions HTTP/1.1" 200 OK'
            )
            second_start_before_logger = (
                b"progress=90%\r"
                b'INFO: 127.0.0.1 - "POST /v1/completions HTTP/1.1" 200 OK\r'
                b"08-12 05:00:30 [loggers.py:310] Engine 000: Avg prompt "
                b"throughput: 200.0 tokens/s, Avg generation throughput: "
                b"0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV "
                b"cache usage: 0.0%, Prefix cache hit rate: 0.0%"
            )
            raw_log = b"\n".join(
                route_records + (logger_before_start, second_start_before_logger)
            )
            log.write_bytes(raw_log)

            observed = GEOMETRY.observe_server_log(log, 8_192)

            self.assertIsNone(
                observed["logger_window_contract"][
                    "configured_interval_seconds"
                ]
            )
            self.assertEqual(
                observed["logger_window_contract"]["authority"],
                "unbound_without_same_run_environment_receipt",
            )
            standalone_samples = observed["logger_interval_samples"]
            self.assertIsNone(
                standalone_samples[0]["configured_interval_seconds"]
            )
            self.assertEqual(
                standalone_samples[0]["authority"],
                "unbound_without_same_run_environment_receipt",
            )
            formal = GEOMETRY.observe_server_log(
                log,
                8_192,
                same_run_environment_receipt=GEOMETRY.environment_receipt(
                    8_192
                ),
            )
            self.assertEqual(
                formal["logger_window_contract"][
                    "configured_interval_seconds"
                ],
                10.0,
            )
            self.assertEqual(
                formal["logger_interval_samples"][0]["authority"],
                "same_run_environment_receipt_bound_"
                "supporting_service_telemetry_only",
            )

            self.assertEqual(
                observed["physical_line_contract"],
                {
                    "delimiter": "LF_byte_only",
                    "decoded_physical_line_count": 7,
                    "carriage_return_byte_count": 4,
                    "bare_carriage_return_byte_count": 4,
                    "carriage_returns_preserved_within_physical_lines": True,
                },
            )
            starts = observed["http_response_start_access_log_observations"]
            samples = observed["logger_interval_samples"]
            self.assertEqual([event["line"] for event in starts], [6, 7])
            self.assertEqual([sample["line"] for sample in samples], [6, 7])
            self.assertTrue(
                all(
                    event["physical_line_contains_carriage_return"]
                    for event in starts
                )
            )
            self.assertEqual(
                samples[0][
                    "http_response_start_access_log_observations_seen"
                ],
                0,
            )
            self.assertEqual(
                samples[0][
                    "position_by_fixed_harness_http_response_start_"
                    "access_log_order"
                ],
                "before_first_http_response_start_access_log_observation",
            )
            self.assertEqual(
                samples[1][
                    "http_response_start_access_log_observations_seen"
                ],
                2,
            )
            self.assertEqual(
                observed["logger_interval_prompt_throughput_tokens_per_second"],
                [100.0, 200.0],
            )
            self.assertEqual(
                observed["log_sha256"], hashlib.sha256(raw_log).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
