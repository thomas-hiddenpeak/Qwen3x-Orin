#!/usr/bin/env python3
"""Host-only tests for the post-exit target-AOT nvmap diagnosis."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

from tests.target_aot_probe_fixtures import valid_m192_oracle_result


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY / "tools/evaluation/classify_target_aot_nvmap_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("target_aot_nvmap_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
RECOVERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECOVERY
SPEC.loader.exec_module(RECOVERY)


BOOT_ID = "12345678-1234-4abc-8def-123456789abc"
CAPTURED_AT = dt.datetime(2026, 8, 12, 5, 0, tzinfo=dt.timezone.utc)
PAGE_SIZE = 4096
FREE_BEFORE = 60_375_707_648
FREE_AFTER = 52_034_760_704
AVAILABLE_PAGES = 2_012_087
BINARY_PAYLOAD = b"frozen target aot probe ELF fixture\n"


def valid_probe(binary_path: pathlib.Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact": RECOVERY.EXPECTED_ARTIFACT,
        "status": "fail",
        "claim_boundary": "frozen source claim",
        "binary": {
            "path": str(binary_path.resolve()),
            "bytes": len(BINARY_PAYLOAD),
            "self_sha256": hashlib.sha256(BINARY_PAYLOAD).hexdigest(),
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
            "synchronize_before_destroy": True,
            "engine_destroy": True,
            "final_mem_info": True,
            "memory_recovery_audit": True,
        },
        "checks": {
            "source_build_provenance_valid": True,
            "binary_provenance_valid": True,
            "engine_created": True,
            "c512_competition_path_exercised": True,
            "during_total_matches_initial": True,
            "exact_target_inventory": True,
            "mutually_exclusive_prefill_sidecars_empty": True,
            "engine_destroy_completed": True,
            "final_total_matches_initial": True,
            "memory_recovered_after_destroy": False,
        },
        "diagnostic": {
            "code": 0,
            "stage": "",
            "message": "",
            "context": "",
            "cuda_error": 0,
        },
    }


def valid_probe_v3(binary_path: pathlib.Path) -> dict[str, object]:
    probe = valid_probe(binary_path)
    probe["schema_version"] = 3
    probe["execution_identity"] = {
        "boot_id": BOOT_ID,
        "child_pid": 4242,
        "child_started_at_utc": "2026-08-12T04:45:00.000Z",
        "child_evidence_finished_at_utc": "2026-08-12T04:59:59.000Z",
        "valid": True,
    }
    probe["attempted"]["layer0_m192_oracle"] = True
    probe["checks"]["execution_identity_valid"] = True
    probe["checks"]["layer0_m192_oracle_passed"] = True
    probe["layer0_m192_oracle"] = {
        "attempted": True,
        "result_available": True,
        "passed": True,
        "result": valid_m192_oracle_result(),
        "diagnostic": {"code": 0, "cuda_error": 0},
    }
    return probe


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def make_fake_host(
    root: pathlib.Path,
    *,
    available_pages: int = AVAILABLE_PAGES,
    orphan_line: str | None = None,
    allocations: bool = True,
    allocation_payload_by_space: dict[str, str] | None = None,
    live_probe_pid: int | None = None,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    debugfs = root / "debugfs/nvmap"
    proc = root / "proc"
    binary_path = root / "sealed/q3x_target_aot_probe"
    write_text(binary_path, BINARY_PAYLOAD.decode("ascii"))
    write_text(proc / "sys/kernel/random/boot_id", BOOT_ID + "\n")

    pagepool = debugfs / "pagepool"
    write_text(pagepool / "page_pool_available_pages", f"{available_pages}\n")
    write_text(
        pagepool / "page_pool_available_big_pages", f"{available_pages}\n"
    )
    write_text(pagepool / "page_pool_big_page_size", "65536\n")
    write_text(pagepool / "page_pool_pages_to_zero", "0\n")

    orphan_header = "    BASE        SIZE USERFLAGS   REFS  KMAPS  UMAPS      UID\n"
    orphan_payload = orphan_header + ((orphan_line + "\n") if orphan_line else "")
    client_lines = [
        "CLIENT                        PROCESS      PID        SIZE",
        "user                             Xorg     2599      17854K",
    ]
    if live_probe_pid is not None:
        client_lines.append(
            f"user                 q3x_aot_probe     {live_probe_pid}       1024K"
        )
    client_lines.append("total                                               17854K")
    client_payload = "\n".join(client_lines) + "\n"

    for space in ("iovmm", "fsi"):
        write_text(debugfs / space / "orphan_handles", orphan_payload)
        write_text(debugfs / space / "clients", client_payload)
        if allocations:
            allocation_payload = (
                "CLIENT                        PROCESS      PID        SIZE\n"
                "user                             Xorg     2599      17854K\n"
            )
            if allocation_payload_by_space is not None:
                allocation_payload = allocation_payload_by_space.get(
                    space, allocation_payload
                )
            write_text(
                debugfs / space / "allocations",
                allocation_payload,
            )

    handles = debugfs / "handles_by_pid"
    write_text(handles / "2599", "header.version: 1\n")
    xorg_proc = proc / "2599"
    xorg_proc.mkdir(parents=True)
    os.symlink("/usr/bin/Xorg", xorg_proc / "exe")
    if live_probe_pid is not None:
        write_text(handles / str(live_probe_pid), "header.version: 1\n")
        probe_proc = proc / str(live_probe_pid)
        probe_proc.mkdir(parents=True)
        os.symlink(binary_path, probe_proc / "exe")
    return debugfs, proc, binary_path


class TargetAotNvmapRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RECOVERY.WORK_ROOT.mkdir(parents=True, exist_ok=True)

    def derive(
        self,
        root: pathlib.Path,
        probe: dict[str, object] | None = None,
        **host_options: object,
    ) -> tuple[dict[str, object], bool]:
        debugfs, proc, binary_path = make_fake_host(root, **host_options)
        source = valid_probe(binary_path) if probe is None else copy.deepcopy(probe)
        if probe is not None:
            source["binary"] = valid_probe(binary_path)["binary"]
        raw = json.dumps(source, sort_keys=True).encode("utf-8")
        return RECOVERY.derive_report(
            source,
            input_path=root / "frozen-probe.json",
            input_sha256=hashlib.sha256(raw).hexdigest(),
            debugfs_root=debugfs,
            proc_root=proc,
            captured_at=CAPTURED_AT,
            page_size=PAGE_SIZE,
        )

    def test_only_exact_canonical_proc_and_nvmap_roots_have_production_authority(
        self,
    ) -> None:
        fake_proc = RECOVERY.WORK_ROOT / "fake-proc"
        fake_nvmap = RECOVERY.WORK_ROOT / "fake-debugfs/nvmap"
        cases = (
            (fake_proc, fake_nvmap, False),
            (RECOVERY.CANONICAL_PROC_ROOT, fake_nvmap, False),
            (fake_proc, RECOVERY.CANONICAL_NVMAP_DEBUGFS_ROOT, False),
            (
                RECOVERY.CANONICAL_PROC_ROOT,
                RECOVERY.CANONICAL_NVMAP_DEBUGFS_ROOT,
                True,
            ),
        )
        for proc_root, debugfs_root, expected in cases:
            with self.subTest(proc=str(proc_root), debugfs=str(debugfs_root)):
                binding = RECOVERY.production_root_binding(
                    proc_root=proc_root,
                    debugfs_root=debugfs_root,
                    observed_nvmap={"debugfs_root": str(debugfs_root)},
                    observed_process_audit={"proc_root": str(proc_root)},
                )
                self.assertIs(binding["production_bound"], expected)
                self.assertEqual(
                    binding["authority"],
                    "canonical_jetson_host"
                    if expected
                    else "test_only_injected_roots",
                )

    def test_fake_geometry_is_test_only_even_when_owner_checks_hold(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(pathlib.Path(temporary))

        self.assertFalse(classified)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertEqual(
            report["observation_authority"]["authority"],
            "test_only_injected_roots",
        )
        self.assertFalse(report["observation_authority"]["production_bound"])
        self.assertFalse(report["criteria"]["requested_proc_root_is_canonical"])
        self.assertFalse(
            report["criteria"]["requested_nvmap_debugfs_root_is_canonical"]
        )
        self.assertTrue(
            report["criteria"]["nvmap_allocation_tables_parse_completely"]
        )
        self.assertEqual(report["source_probe"]["status"], "fail")
        self.assertTrue(report["source_probe"]["status_preserved"])
        self.assertTrue(
            report["source_probe"]["binary_reauthentication"]
            ["matches_frozen_identity"]
        )
        self.assertIn("derived diagnostic only", report["claim_boundary"])
        self.assertEqual(report["host_snapshot"]["boot_id"], BOOT_ID)
        self.assertEqual(report["host_snapshot"]["page_size_bytes"], PAGE_SIZE)
        self.assertEqual(
            report["calculation"]["cuda_free_gap_bytes"],
            FREE_BEFORE - FREE_AFTER,
        )
        self.assertEqual(
            report["calculation"]["absolute_post_exit_pagepool_bytes"],
            AVAILABLE_PAGES * PAGE_SIZE,
        )
        self.assertEqual(
            report["calculation"]["pagepool_adjusted_residual_bytes"],
            99_438_592,
        )
        self.assertEqual(
            report["nvmap_snapshot"]["handles_by_pid_pids"], [2599]
        )
        self.assertEqual(
            report["nvmap_snapshot"]["handles_by_pid"]["2599"]["text"],
            "header.version: 1\n",
        )
        self.assertTrue(
            report["nvmap_snapshot"]["address_spaces"]["iovmm"]
            ["allocations"]["present"]
        )
        self.assertEqual(
            report["observed_nvmap_pid_sets"]["allocation_client_pids"],
            [2599],
        )

    def test_insufficient_absolute_pagepool_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(
                pathlib.Path(temporary), available_pages=1
            )

        self.assertFalse(classified)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertIn(
            "post_exit_pagepool_explains_cuda_free_gap_within_tolerance",
            report["rejection_reasons"],
        )

    def test_any_orphan_handle_rejects_no_owner_leak(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(
                pathlib.Path(temporary),
                orphan_line="00000000 00001000 00000000 1 0 0 deadbeef",
            )

        self.assertFalse(classified)
        self.assertFalse(
            report["criteria"]["iovmm_and_fsi_orphan_tables_empty"]
        )

    def test_unparsed_allocation_rows_are_preserved_and_fail_closed(self) -> None:
        safe = (
            "CLIENT                        PROCESS      PID        SIZE\n"
            "user                             Xorg     2599      17854K\n"
        )
        cases = (
            (
                "multi_column",
                "                                             0       1280K "
                "10120001      6      1      0      0      0      3 "
                "ffff000085af9800      0",
            ),
            (
                "format_change",
                "CLIENT PROCESS PID SIZE FLAGS",
            ),
            (
                "residual_allocation",
                "residual allocation pid=4242 bytes=4096",
            ),
        )
        for space in ("iovmm", "fsi"):
            for name, unparsed_line in cases:
                with self.subTest(space=space, case=name):
                    payloads = {"iovmm": safe, "fsi": safe}
                    payloads[space] = safe + unparsed_line + "\n"
                    with tempfile.TemporaryDirectory(
                        dir=RECOVERY.WORK_ROOT
                    ) as temporary:
                        report, classified = self.derive(
                            pathlib.Path(temporary),
                            allocation_payload_by_space=payloads,
                        )

                    self.assertFalse(classified)
                    self.assertFalse(
                        report["criteria"]
                        ["nvmap_allocation_tables_parse_completely"]
                    )
                    allocations = (
                        report["nvmap_snapshot"]["address_spaces"][space]
                        ["allocations"]
                    )
                    self.assertEqual(
                        allocations["unparsed_allocation_lines"],
                        [unparsed_line],
                    )
                    other = "fsi" if space == "iovmm" else "iovmm"
                    self.assertEqual(
                        report["nvmap_snapshot"]["address_spaces"][other]
                        ["allocations"]["unparsed_allocation_lines"],
                        [],
                    )

    def test_live_same_probe_elf_is_rejected_and_visible_in_owner_tables(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(
                pathlib.Path(temporary), live_probe_pid=777
            )

        self.assertFalse(classified)
        self.assertEqual(
            report["probe_process_audit"]["matching_probe_elf_pids"], [777]
        )
        self.assertFalse(
            report["criteria"]["matching_probe_elf_process_has_exited"]
        )
        self.assertFalse(
            report["criteria"]["matching_probe_elf_absent_from_nvmap_clients"]
        )
        self.assertFalse(
            report["criteria"]["matching_probe_elf_absent_from_handles_by_pid"]
        )

    def test_other_source_failure_or_cuda_error_cannot_be_reclassified(self) -> None:
        cases: list[dict[str, object]] = []
        failed_check = valid_probe(pathlib.Path("/placeholder"))
        failed_check["checks"]["exact_target_inventory"] = False
        cases.append(failed_check)
        cuda_failure = valid_probe(pathlib.Path("/placeholder"))
        cuda_failure["cuda_errors"]["final_mem_info"] = 2
        cases.append(cuda_failure)

        for index, probe in enumerate(cases):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory(
                    dir=RECOVERY.WORK_ROOT
                ) as temporary:
                    report, classified = self.derive(
                        pathlib.Path(temporary), probe=probe
                    )
                self.assertFalse(classified)
                self.assertFalse(
                    report["criteria"]
                    ["source_exclusive_failure_is_memory_recovery"]
                )

    def test_missing_optional_allocations_is_recorded_but_fake_root_is_test_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(
                pathlib.Path(temporary), allocations=False
            )

        self.assertFalse(classified)
        self.assertTrue(
            report["criteria"]["nvmap_allocation_tables_parse_completely"]
        )
        for space in ("iovmm", "fsi"):
            self.assertFalse(
                report["nvmap_snapshot"]["address_spaces"][space]
                ["allocations"]["present"]
            )

    def test_cli_publishes_create_only_inside_repository_work_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            debugfs, proc, binary_path = make_fake_host(root)
            source = valid_probe(binary_path)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            output = root / "derived.json"
            stdout = StringIO()
            stderr = StringIO()
            with (
                mock.patch.object(
                    RECOVERY.os, "sysconf", return_value=PAGE_SIZE
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                arguments = [
                    "--input",
                    str(source_path),
                    "--output",
                    str(output),
                    "--debugfs-root",
                    str(debugfs),
                    "--proc-root",
                    str(proc),
                ]
                first = RECOVERY.main(arguments)
                second = RECOVERY.main(arguments)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertIn("classification=inconclusive", stdout.getvalue())
        self.assertIn("refusing to replace existing output", stderr.getvalue())
        self.assertEqual(payload["classification"], "inconclusive")
        self.assertEqual(
            payload["observation_authority"]["authority"],
            "test_only_injected_roots",
        )
        self.assertEqual(
            payload["source_probe"]["input_sha256"],
            source_sha256,
        )

    def test_output_outside_repository_work_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(RECOVERY.EvidenceError, "inside repository"):
            RECOVERY.validate_output_path(REPOSITORY / "outside.json")

    def test_input_outside_repository_work_root_is_rejected(self) -> None:
        with self.assertRaisesRegex(RECOVERY.EvidenceError, "inside repository"):
            RECOVERY.validate_input_path(REPOSITORY / "README.md")

    def test_binary_rehash_and_source_tolerance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            debugfs, proc, binary_path = make_fake_host(root)
            probe = valid_probe(binary_path)
            binary_path.write_bytes(b"changed")
            with self.assertRaisesRegex(
                RECOVERY.EvidenceError, "no longer matches"
            ):
                RECOVERY.derive_report(
                    probe,
                    input_path=root / "source.json",
                    input_sha256="b" * 64,
                    debugfs_root=debugfs,
                    proc_root=proc,
                    captured_at=CAPTURED_AT,
                    page_size=PAGE_SIZE,
                )

            binary_path.write_bytes(BINARY_PAYLOAD)
            probe["device"]["destroy_recovery_tolerance_bytes"] = 1
            with self.assertRaisesRegex(RECOVERY.EvidenceError, "256 MiB"):
                RECOVERY.derive_report(
                    probe,
                    input_path=root / "source.json",
                    input_sha256="b" * 64,
                    debugfs_root=debugfs,
                    proc_root=proc,
                    captured_at=CAPTURED_AT,
                    page_size=PAGE_SIZE,
                )

    def test_source_status_is_never_promoted(self) -> None:
        probe = valid_probe(pathlib.Path("/placeholder"))
        probe["status"] = "pass"
        probe["checks"]["memory_recovered_after_destroy"] = True
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            report, classified = self.derive(
                pathlib.Path(temporary), probe=copy.deepcopy(probe)
            )

        self.assertFalse(classified)
        self.assertEqual(report["source_probe"]["status"], "pass")
        self.assertEqual(report["classification"], "inconclusive")

    def test_schema_v3_execution_identity_is_bound_to_post_exit_host(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            debugfs, proc, binary_path = make_fake_host(root)
            probe = valid_probe_v3(binary_path)
            raw = json.dumps(probe, sort_keys=True).encode("utf-8")
            report, classified = RECOVERY.derive_report(
                probe,
                input_path=root / "source-v3.json",
                input_sha256=hashlib.sha256(raw).hexdigest(),
                debugfs_root=debugfs,
                proc_root=proc,
                captured_at=CAPTURED_AT,
                page_size=PAGE_SIZE,
            )

        self.assertFalse(classified)
        self.assertEqual(report["source_probe"]["schema_version"], 3)
        self.assertTrue(
            report["criteria"]["source_execution_identity_is_valid"]
        )
        self.assertTrue(report["criteria"]["source_and_post_exit_boot_id_match"])
        self.assertTrue(
            report["criteria"]
            ["source_child_pid_absent_from_post_exit_handles_by_pid"]
        )
        self.assertFalse(report["observation_authority"]["production_bound"])

    def test_schema_v3_boot_mismatch_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir=RECOVERY.WORK_ROOT) as temporary:
            root = pathlib.Path(temporary)
            debugfs, proc, binary_path = make_fake_host(root)
            probe = valid_probe_v3(binary_path)
            probe["execution_identity"]["boot_id"] = (
                "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
            )
            raw = json.dumps(probe, sort_keys=True).encode("utf-8")
            report, classified = RECOVERY.derive_report(
                probe,
                input_path=root / "source-v3.json",
                input_sha256=hashlib.sha256(raw).hexdigest(),
                debugfs_root=debugfs,
                proc_root=proc,
                captured_at=CAPTURED_AT,
                page_size=PAGE_SIZE,
            )

        self.assertFalse(classified)
        self.assertFalse(report["criteria"]["source_and_post_exit_boot_id_match"])

    def test_schema_v3_cannot_skip_oracle_or_execution_fields(self) -> None:
        probe = valid_probe_v3(pathlib.Path("/placeholder"))
        del probe["checks"]["layer0_m192_oracle_passed"]
        with self.assertRaisesRegex(
            RECOVERY.EvidenceError, "layer0_m192_oracle_passed"
        ):
            RECOVERY.validate_probe_shape(probe)

    def test_schema_v3_rejects_internally_inconsistent_oracle_evidence(self) -> None:
        cases: list[tuple[str, object]] = []

        attempted = valid_probe_v3(pathlib.Path("/placeholder"))
        attempted["layer0_m192_oracle"]["attempted"] = False
        cases.append(("attempted fields disagree", attempted))

        check_disagrees = valid_probe_v3(pathlib.Path("/placeholder"))
        check_disagrees["checks"]["layer0_m192_oracle_passed"] = False
        cases.append(("passed fields disagree", check_disagrees))

        unavailable = valid_probe_v3(pathlib.Path("/placeholder"))
        unavailable["layer0_m192_oracle"]["result_available"] = False
        cases.append(("unavailable layer0 M192 result must be null", unavailable))

        result_disagrees = valid_probe_v3(pathlib.Path("/placeholder"))
        result_disagrees["layer0_m192_oracle"]["result"]["passed"] = False
        cases.append(("result passed fields disagree", result_disagrees))

        diagnostic = valid_probe_v3(pathlib.Path("/placeholder"))
        diagnostic["layer0_m192_oracle"]["diagnostic"]["cuda_error"] = 700
        cases.append(("requires zero diagnostic", diagnostic))

        diagnostic_code = valid_probe_v3(pathlib.Path("/placeholder"))
        diagnostic_code["layer0_m192_oracle"]["diagnostic"]["code"] = 3
        cases.append(("requires zero diagnostic", diagnostic_code))

        for message, probe in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RECOVERY.EvidenceError, message):
                    RECOVERY.validate_probe_shape(probe)

    def test_schema_v3_closes_every_passed_oracle_strong_gate_class(self) -> None:
        mutations: tuple[tuple[tuple[str, ...], object], ...] = (
            (("cleanup", "passed"), False),
            (("cleanup", "device_frees_succeeded"), 10),
            (("gate_up", "bitwise_exact"), False),
            (("gate_up", "candidate_complete_write"), False),
            (("gate_up", "replay_guards_intact"), False),
            (("gate_up", "baseline_all_finite"), False),
            (("down_residual", "baseline_candidate_full_mismatches"), 1),
            (("down_residual", "candidate_sha256"), "3" * 64),
            (("receipt", "attachment_exact"), False),
            (("receipt", "complete_owner_allocation_ranges_checked"), False),
            (("receipt", "complete_owner_allocation_exact_cover"), False),
            (("receipt", "complete_owner_allocation_nonoverlap"), False),
            (("receipt", "oracle_allocations_disjoint_from_owner"), False),
            (("receipt", "gate_up", "exact"), False),
            (("receipt", "down", "exact"), False),
            (("gate_up_resources", "exact_geometry_gate"), False),
            (("gate_up_resources", "exact_resource_gate"), False),
            (("gate_up_resources", "grid_m"), 1),
            (("down_resources", "registers_per_thread"), 209),
            (("baseline_gate_launches",), 5),
            (("candidate_down_launches",), 2),
            (("candidate_gate_up_launches",), True),
            (("token_count",), 191),
            (("full_token_count",), 127),
            (("tail_token_count",), 65),
            (("owner_attachment_authenticated",), False),
            (("canonical_layer0_weights",), False),
            (("activation_preserved_after_candidate",), False),
            (("residual_preserved_after_replay",), False),
            (("activation_fixture", "shape"), [191, 5_120]),
            (("residual_fixture", "sha256"), "short"),
        )
        for path, replacement in mutations:
            with self.subTest(path=".".join(path)):
                probe = valid_probe_v3(pathlib.Path("/placeholder"))
                cursor = probe["layer0_m192_oracle"]["result"]
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = replacement
                with self.assertRaisesRegex(RECOVERY.EvidenceError, "source probe"):
                    RECOVERY.validate_probe_shape(probe)


if __name__ == "__main__":
    unittest.main()
