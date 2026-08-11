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
        self.assertEqual(effective["FLASHINFER_NO_DOWNLOAD"], "1")

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

    def test_plan_has_only_the_four_locked_budgets(self) -> None:
        plan = GEOMETRY.build_plan(self.make_config())
        self.assertEqual(
            [entry["max_num_batched_tokens"] for entry in plan["budgets"]],
            [2_048, 4_096, 8_192, 40_000],
        )
        self.assertEqual(
            [entry["planned_scheduled_forwards"] for entry in plan["budgets"]],
            [20, 10, 5, 1],
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

    def test_only_unlinked_posix_semaphore_matches_deleted_mapping_rule(self) -> None:
        accepted = "/dev/shm/sem.5zLMSe (deleted)"
        self.assertIsNotNone(GEOMETRY.DELETED_POSIX_SEMAPHORE.fullmatch(accepted))
        self.assertIsNone(
            GEOMETRY.DELETED_POSIX_SEMAPHORE.fullmatch(
                "/cache/kernel.so (deleted)"
            )
        )

    def test_measured_tegrastats_envelope_is_windowed_and_strict(self) -> None:
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
            self.assertTrue(observed["clock_contract"]["stable_for_every_sample"])
            path.write_text(
                ("\n".join(lines).replace("EMC_FREQ 0%@3200", "EMC_FREQ 0%@3199"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "clock/RAM drift"):
                GEOMETRY.validate_measurement_telemetry(path, 1, 4)
            path.write_text(
                ("\n".join(lines).replace("cpu@63C/63C", "cpu@63C/70C"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(GEOMETRY.GeometryError):
                GEOMETRY.validate_measurement_telemetry(path, 1, 4)

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
            [f"budget:{value}" for value in (2048, 4096, 8192, 40000)],
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
                    "speculative_config=None, quantization=modelopt,",
                    "enable_prefix_caching=False, enable_chunked_prefill=True,",
                    "non-default args: {'max_num_batched_tokens': 8192}",
                    "Using AttentionBackendEnum.FLASHINFER backend.",
                    "Using Triton/FLA GDN prefill kernel (requested=auto).",
                    "Avg prompt throughput: 4300.0 tokens/s,",
                )
            )
            log.write_text(required)
            observed = GEOMETRY.observe_server_log(log, 8_192)
            self.assertTrue(all(observed["required_route_probe_hits"].values()))
            self.assertEqual(
                observed["ten_second_logger_prompt_throughput_maximum"], 4300.0
            )
            log.write_text(
                required.replace(
                    "Using AttentionBackendEnum.FLASHINFER backend.\n", ""
                )
            )
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "route evidence"):
                GEOMETRY.observe_server_log(log, 8_192)
            log.write_text(required + "\nUsing Humming")
            with self.assertRaisesRegex(GEOMETRY.GeometryError, "forbidden route"):
                GEOMETRY.observe_server_log(log, 8_192)


if __name__ == "__main__":
    unittest.main()
