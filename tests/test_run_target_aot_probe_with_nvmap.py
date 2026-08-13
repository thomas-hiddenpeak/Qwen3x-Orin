#!/usr/bin/env python3
"""CPU-only fake tests for the target-AOT parent/child lifecycle wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from tests.target_aot_probe_fixtures import valid_m192_oracle_result


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
TOOLS = REPOSITORY / "tools/evaluation"
sys.path.insert(0, str(TOOLS))
import classify_target_aot_nvmap_recovery as RECOVERY  # noqa: E402

MODULE_PATH = TOOLS / "run_target_aot_probe_with_nvmap.py"
SPEC = importlib.util.spec_from_file_location("target_aot_lifecycle_wrapper", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
WRAPPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WRAPPER
SPEC.loader.exec_module(WRAPPER)


BOOT_ID = "12345678-1234-4abc-8def-123456789abc"
PAGE_SIZE = 4096
FREE_BEFORE = 60_375_707_648
FREE_AFTER = 52_034_760_704
AVAILABLE_PAGES = 2_012_087
BINARY_PAYLOAD = b"fake executable target AOT probe fixture\n"


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def make_fake_host(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    debugfs = root / "debugfs/nvmap"
    proc = root / "proc"
    write_text(proc / "sys/kernel/random/boot_id", BOOT_ID + "\n")
    pagepool = debugfs / "pagepool"
    write_text(pagepool / "page_pool_available_pages", f"{AVAILABLE_PAGES}\n")
    write_text(
        pagepool / "page_pool_available_big_pages", f"{AVAILABLE_PAGES}\n"
    )
    write_text(pagepool / "page_pool_big_page_size", "65536\n")
    write_text(pagepool / "page_pool_pages_to_zero", "0\n")
    orphan = "    BASE        SIZE USERFLAGS   REFS  KMAPS  UMAPS      UID\n"
    clients = (
        "CLIENT                        PROCESS      PID        SIZE\n"
        "user                             Xorg     2599      17854K\n"
        "total                                               17854K\n"
    )
    allocations = (
        "CLIENT                        PROCESS      PID        SIZE\n"
        "user                             Xorg     2599      17854K\n"
    )
    for space in ("iovmm", "fsi"):
        write_text(debugfs / space / "orphan_handles", orphan)
        write_text(debugfs / space / "clients", clients)
        write_text(debugfs / space / "allocations", allocations)
    write_text(debugfs / "handles_by_pid/2599", "header.version: 1\n")
    xorg = proc / "2599"
    xorg.mkdir(parents=True)
    os.symlink("/usr/bin/Xorg", xorg / "exe")
    return debugfs, proc


def source_probe(
    binary: dict[str, object],
    *,
    child_pid: int,
    child_started: str,
    child_finished: str,
    schema_version: int = 4,
    asset_mode: str = "online_prepare",
    bundle_path: str = "",
    expected_catalog: str = "",
) -> dict[str, object]:
    probe: dict[str, object] = {
        "schema_version": schema_version,
        "artifact": RECOVERY.EXPECTED_ARTIFACT,
        "status": "fail",
        "claim_boundary": "fake child source claim",
        "binary": {
            "path": binary["path"],
            "bytes": binary["bytes"],
            "self_sha256": binary["sha256"],
        },
        "device": {
            "ordinal": 0,
            "total_before_bytes": 65_932_095_488,
            "total_during_bytes": 65_932_095_488,
            "total_after_destroy_bytes": 65_932_095_488,
            "free_before_bytes": FREE_BEFORE,
            "free_during_bytes": 28_009_820_160,
            "free_after_destroy_bytes": FREE_AFTER,
            "destroy_recovery_tolerance_bytes": 256 * 1024 * 1024,
        },
        "cuda_errors": {
            "device_query": 0,
            "initial_mem_info": 0,
            "during_mem_info": 0,
            "synchronize_before_destroy": 0,
            "final_mem_info": 0,
        },
        "attempted": {
            "device_query": True,
            "initial_mem_info": True,
            "engine_create": True,
            "during_mem_info": True,
            "layer0_m192_oracle": True,
            "synchronize_before_destroy": True,
            "engine_destroy": True,
            "final_mem_info": True,
            "memory_recovery_audit": True,
        },
        "checks": {
            "source_build_provenance_valid": True,
            "execution_identity_valid": True,
            "binary_provenance_valid": True,
            "engine_created": True,
            "c512_competition_path_exercised": True,
            "during_total_matches_initial": True,
            "exact_target_inventory": True,
            "mutually_exclusive_prefill_sidecars_empty": True,
            "layer0_m192_oracle_passed": True,
            "engine_destroy_completed": True,
            "final_total_matches_initial": True,
            "memory_recovered_after_destroy": False,
        },
        "layer0_m192_oracle": {
            "attempted": True,
            "result_available": True,
            "passed": True,
            "result": valid_m192_oracle_result(),
            "diagnostic": {"code": 0, "cuda_error": 0},
        },
        "diagnostic": {
            "code": 0,
            "stage": "",
            "message": "",
            "context": "",
            "cuda_error": 0,
        },
    }
    if schema_version in (3, 4):
        probe["execution_identity"] = {
            "boot_id": BOOT_ID,
            "child_pid": child_pid,
            "child_started_at_utc": child_started,
            "child_evidence_finished_at_utc": child_finished,
            "valid": True,
        }
    if schema_version == 4:
        header_catalog = ""
        verified_catalog = "a" * 64
        loaded = False
        created = False
        file_bytes_read = 0
        host_authentication_passes = 0
        file_bytes_written = 0
        if asset_mode == "create":
            header_catalog = "b" * 64
            verified_catalog = expected_catalog
            created = True
            file_bytes_written = WRAPPER.TARGET_AOT_PERSISTENT_BUNDLE_BYTES
        elif asset_mode == "load":
            header_catalog = "b" * 64
            verified_catalog = expected_catalog
            loaded = True
            file_bytes_read = 2 * WRAPPER.TARGET_AOT_PERSISTENT_BUNDLE_BYTES
            host_authentication_passes = 2
        probe["checks"]["persistent_bundle_contract_exact"] = True
        probe["target_aot"] = {
            "asset_mode": asset_mode,
            "persistent_bundle_path": bundle_path,
            "expected_payload_catalog_sha256": expected_catalog,
            "persistent_bundle_contract_exact": True,
            "loaded_from_persisted_bundle": loaded,
            "persistent_bundle_file_bytes_read": file_bytes_read,
            "persistent_bundle_host_authentication_passes": (
                host_authentication_passes
            ),
            "persistent_bundle_created": created,
            "persistent_bundle_file_bytes_written": file_bytes_written,
            "persistent_record_header_catalog_sha256": header_catalog,
            "verified_payload_catalog_sha256": verified_catalog,
        }
    elif schema_version == 2:
        del probe["checks"]["execution_identity_valid"]
        del probe["checks"]["layer0_m192_oracle_passed"]
        del probe["attempted"]["layer0_m192_oracle"]
        del probe["layer0_m192_oracle"]
    return probe


class FakeProcess:
    def __init__(
        self,
        command: list[str],
        binary: dict[str, object],
        *,
        reported_pid: int,
        schema_version: int,
        source_pass: bool,
        reported_bundle_request: tuple[str, str, str],
    ) -> None:
        self.pid = 4242
        self.command = command
        self.binary = binary
        self.reported_pid = reported_pid
        self.schema_version = schema_version
        self.source_pass = source_pass
        self.reported_bundle_request = reported_bundle_request
        self.started = WRAPPER.utc_now()

    def wait(self) -> int:
        finished = WRAPPER.utc_now()
        evidence = source_probe(
            self.binary,
            child_pid=self.reported_pid,
            child_started=WRAPPER.format_utc(self.started),
            child_finished=WRAPPER.format_utc(finished),
            schema_version=self.schema_version,
            asset_mode=self.reported_bundle_request[0],
            bundle_path=self.reported_bundle_request[1],
            expected_catalog=self.reported_bundle_request[2],
        )
        if self.source_pass:
            evidence["status"] = "pass"
            evidence["checks"]["memory_recovered_after_destroy"] = True
        output = pathlib.Path(self.command[2])
        output.write_text(json.dumps(evidence), encoding="utf-8")
        return 0 if self.source_pass else 1


class TargetAotLifecycleWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RECOVERY.WORK_ROOT.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        root: pathlib.Path,
        *,
        reported_pid: int = 4242,
        schema_version: int = 4,
        live_authentication_failure: bool = False,
        replace_path_during_launch: bool = False,
        source_pass: bool = False,
        bundle_mode: str | None = None,
        bundle_path: pathlib.Path | None = None,
        expected_catalog: str | None = None,
        reported_bundle_request: tuple[str, str, str] | None = None,
    ) -> tuple[dict[str, object], bool, pathlib.Path]:
        debugfs, proc = make_fake_host(root)
        binary_path = root / "probe/q3x_target_aot_probe"
        write_text(binary_path, BINARY_PAYLOAD.decode("ascii"))
        binary_path.chmod(0o755)
        model = root / "model"
        model.mkdir()
        child_evidence = root / "child.json"
        output = root / "parent.json"
        binary = WRAPPER.authenticate_probe_executable(binary_path)

        def popen_factory(command: list[str], **kwargs: object) -> FakeProcess:
            self.last_popen_command = list(command)
            self.last_popen_kwargs = dict(kwargs)
            executable = str(kwargs["executable"])
            executable_fd = int(executable.rsplit("/", 1)[1])
            os.fstat(executable_fd)
            self.popen_fd_was_open = True
            command_bundle_request = (
                (command[3], command[4], command[5])
                if len(command) == 6
                else ("online_prepare", "", "")
            )
            process = FakeProcess(
                command,
                binary,
                reported_pid=reported_pid,
                schema_version=schema_version,
                source_pass=source_pass,
                reported_bundle_request=(
                    reported_bundle_request
                    if reported_bundle_request is not None
                    else command_bundle_request
                ),
            )
            if replace_path_during_launch:
                replacement = binary_path.with_name("replacement_probe")
                write_text(replacement, BINARY_PAYLOAD.decode("ascii"))
                replacement.chmod(0o755)
                os.replace(replacement, binary_path)
            return process

        def live_authenticator(
            pid: int, expected: dict[str, object], **_kwargs: object
        ) -> dict[str, object]:
            if live_authentication_failure:
                raise RECOVERY.EvidenceError("injected live authentication failure")
            return {
                "captured_at_utc": WRAPPER.format_utc(WRAPPER.utc_now()),
                "pid": pid,
                "proc_exe_link_target": expected["path"],
                "resolved_path": expected["path"],
                "path": expected["path"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
                "st_dev": expected["st_dev"],
                "st_ino": expected["st_ino"],
                "st_ctime_ns": expected["st_ctime_ns"],
                "proc_start_time_ticks": 123456,
                "path_matches_prelaunch": True,
                "matches_prelaunch_identity": True,
            }

        with mock.patch.object(os, "sysconf", return_value=PAGE_SIZE):
            report, accepted = WRAPPER.run_probe(
                probe_path=binary_path,
                model_directory=model,
                child_evidence_path=child_evidence,
                output_path=output,
                debugfs_root=debugfs,
                proc_root=proc,
                bundle_mode=bundle_mode,
                bundle_path=bundle_path,
                expected_catalog=expected_catalog,
                popen_factory=popen_factory,
                live_authenticator=live_authenticator,
            )
        return report, accepted, output

    def test_fake_roots_never_accept_even_with_bound_schema_v4_child(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, accepted, output = self.execute(pathlib.Path(temporary))
            published = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(accepted)
        self.assertFalse(report["combined_lifecycle_accepted"])
        self.assertEqual(report["classification"], "inconclusive")
        self.assertTrue(report["identity_binding"]["bound"])
        self.assertTrue(report["bundle_request_binding"]["bound"])
        self.assertEqual(
            report["bundle_request_binding"]["requested"],
            {
                "asset_mode": "online_prepare",
                "persistent_bundle_path": "",
                "expected_payload_catalog_sha256": "",
            },
        )
        self.assertEqual(
            report["observation_authority"]["authority"],
            "test_only_injected_roots",
        )
        self.assertFalse(report["observation_authority"]["production_bound"])
        self.assertFalse(
            report["combined_acceptance_criteria"]
            ["canonical_proc_and_nvmap_roots"]
        )
        self.assertEqual(report["source_probe"]["status"], "fail")
        self.assertTrue(report["source_probe"]["status_preserved"])
        self.assertEqual(report["child"]["exit_code"], 1)
        self.assertEqual(
            report["fixed_recovery_tolerance_bytes"], 256 * 1024 * 1024
        )
        self.assertIn("2599", report["pre_child_snapshot"]["nvmap"]["handles_by_pid"])
        self.assertIn(
            "2599",
            report["immediate_post_exit_snapshot"]["nvmap"]["handles_by_pid"],
        )
        self.assertEqual(published["source_probe"]["status"], "fail")
        self.assertFalse(published["combined_lifecycle_accepted"])
        executable = self.last_popen_kwargs["executable"]
        pass_fds = self.last_popen_kwargs["pass_fds"]
        self.assertEqual(
            self.last_popen_command[0], report["child"]["prelaunch_binary"]["path"]
        )
        self.assertRegex(executable, r"^/proc/self/fd/[0-9]+$")
        self.assertEqual(pass_fds, (int(executable.rsplit("/", 1)[1]),))
        self.assertTrue(self.popen_fd_was_open)

    def test_persistent_bundle_request_is_triplet_bound_and_workspace_local(
        self,
    ) -> None:
        catalog = "367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe"
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            create_path = root / "assets/target-aot.bundle"
            mode, resolved, observed_catalog = WRAPPER.validate_bundle_request(
                "create", create_path, catalog
            )
            self.assertEqual(mode, "create")
            self.assertEqual(resolved, create_path.resolve(strict=False))
            self.assertEqual(observed_catalog, catalog)
            self.assertTrue(create_path.parent.is_dir())

            with create_path.open("wb") as stream:
                stream.truncate(WRAPPER.TARGET_AOT_PERSISTENT_BUNDLE_BYTES)
            mode, resolved, observed_catalog = WRAPPER.validate_bundle_request(
                "load", create_path, catalog
            )
            self.assertEqual(mode, "load")
            self.assertEqual(resolved, create_path.resolve(strict=True))
            self.assertEqual(observed_catalog, catalog)

            with self.assertRaisesRegex(
                RECOVERY.EvidenceError, "required together"
            ):
                WRAPPER.validate_bundle_request("load", create_path, None)
            with self.assertRaisesRegex(
                RECOVERY.EvidenceError, "nonzero lowercase SHA-256"
            ):
                WRAPPER.validate_bundle_request(
                    "load", create_path, "0" * 64
                )

    def test_child_bundle_mode_path_and_catalog_must_match_command(self) -> None:
        catalog = "367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe"
        cases = (
            ("load", None, None),
            ("create", "different", None),
            ("create", None, "d" * 64),
        )
        for reported_mode, reported_path_case, reported_catalog_case in cases:
            with self.subTest(
                mode=reported_mode,
                path=reported_path_case,
                catalog=reported_catalog_case,
            ), tempfile.TemporaryDirectory(
                dir=RECOVERY.WORK_ROOT
            ) as temporary:
                root = pathlib.Path(temporary)
                command_path = root / "assets/target-aot.bundle"
                reported_path = (
                    root / "assets/different-target-aot.bundle"
                    if reported_path_case == "different"
                    else command_path
                )
                with self.assertRaisesRegex(
                    RECOVERY.EvidenceError,
                    "does not match the launched command",
                ):
                    self.execute(
                        root,
                        bundle_mode="create",
                        bundle_path=command_path,
                        expected_catalog=catalog,
                        reported_bundle_request=(
                            reported_mode,
                            str(reported_path.resolve(strict=False)),
                            reported_catalog_case or catalog,
                        ),
                    )

    def test_persistent_modes_bind_exact_child_request(self) -> None:
        catalog = "367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe"
        for mode in ("create", "load"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                dir=RECOVERY.WORK_ROOT
            ) as temporary:
                root = pathlib.Path(temporary)
                bundle = root / "assets/target-aot.bundle"
                if mode == "load":
                    bundle.parent.mkdir(parents=True)
                    with bundle.open("wb") as stream:
                        stream.truncate(
                            WRAPPER.TARGET_AOT_PERSISTENT_BUNDLE_BYTES
                        )
                report, accepted, _output = self.execute(
                    root,
                    bundle_mode=mode,
                    bundle_path=bundle,
                    expected_catalog=catalog,
                )
                expected_path = str(bundle.resolve(strict=mode == "load"))
                self.assertFalse(accepted)
                self.assertTrue(report["bundle_request_binding"]["bound"])
                self.assertEqual(
                    report["bundle_request_binding"]["requested"],
                    {
                        "asset_mode": mode,
                        "persistent_bundle_path": expected_path,
                        "expected_payload_catalog_sha256": catalog,
                    },
                )

    def test_fake_roots_cannot_accept_even_a_complete_source_pass(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, accepted, output = self.execute(
                pathlib.Path(temporary), source_pass=True
            )
            published = json.loads(output.read_text(encoding="utf-8"))

        self.assertFalse(accepted)
        self.assertTrue(report["source_pass_complete"])
        self.assertEqual(report["child"]["exit_code"], 0)
        self.assertFalse(report["observation_authority"]["production_bound"])
        self.assertTrue(
            report["combined_acceptance_criteria"]
            ["recovery_diagnostic_or_complete_source_pass"]
        )
        self.assertFalse(
            report["combined_acceptance_criteria"]
            ["canonical_proc_and_nvmap_roots"]
        )
        self.assertFalse(report["combined_lifecycle_accepted"])
        self.assertFalse(published["combined_lifecycle_accepted"])

    def test_launched_and_reported_child_pid_mismatch_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, accepted, output = self.execute(
                pathlib.Path(temporary), reported_pid=9999
            )
            output_exists = output.exists()

        self.assertFalse(accepted)
        self.assertTrue(output_exists)
        self.assertFalse(
            report["identity_binding"]["criteria"]
            ["source_child_pid_matches_launched_pid"]
        )
        self.assertFalse(report["combined_lifecycle_accepted"])
        self.assertEqual(report["source_probe"]["status"], "fail")

    def test_same_bytes_path_replacement_fails_inode_binding(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, accepted, output = self.execute(
                pathlib.Path(temporary), replace_path_during_launch=True
            )
            output_exists = output.exists()

        self.assertFalse(accepted)
        self.assertTrue(output_exists)
        prelaunch = report["child"]["prelaunch_binary"]
        postlaunch = report["child"]["postlaunch_binary"]
        self.assertEqual(postlaunch["sha256"], prelaunch["sha256"])
        self.assertEqual(postlaunch["bytes"], prelaunch["bytes"])
        self.assertNotEqual(postlaunch["st_ino"], prelaunch["st_ino"])
        self.assertFalse(postlaunch["matches_prelaunch_identity"])
        self.assertFalse(
            report["identity_binding"]["criteria"]
            ["postlaunch_elf_matches_prelaunch_identity"]
        )
        self.assertFalse(report["combined_lifecycle_accepted"])

    def test_live_authentication_failure_still_waits_and_publishes(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, accepted, output = self.execute(
                pathlib.Path(temporary), live_authentication_failure=True
            )
            child_evidence_exists = (pathlib.Path(temporary) / "child.json").exists()
            output_exists = output.exists()

        self.assertFalse(accepted)
        self.assertTrue(child_evidence_exists)
        self.assertTrue(output_exists)
        self.assertEqual(report["child"]["exit_code"], 1)
        self.assertIn("injected", report["child"]["live_binary"]["error"])
        self.assertFalse(report["identity_binding"]["bound"])

    def test_legacy_schema_child_is_rejected_by_parent_wrapper(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            with self.assertRaisesRegex(
                RECOVERY.EvidenceError, "schema-v4"
            ):
                self.execute(pathlib.Path(temporary), schema_version=3)

    def test_existing_parent_output_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            debugfs, proc = make_fake_host(root)
            binary_path = root / "probe/q3x_target_aot_probe"
            write_text(binary_path, BINARY_PAYLOAD.decode("ascii"))
            binary_path.chmod(0o755)
            model = root / "model"
            model.mkdir()
            output = root / "parent.json"
            output.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                RECOVERY.EvidenceError, "refusing to replace"
            ):
                WRAPPER.run_probe(
                    probe_path=binary_path,
                    model_directory=model,
                    child_evidence_path=root / "child.json",
                    output_path=output,
                    debugfs_root=debugfs,
                    proc_root=proc,
                    popen_factory=lambda *_args, **_kwargs: self.fail(
                        "child must not launch"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
