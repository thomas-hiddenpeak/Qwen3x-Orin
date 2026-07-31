#!/usr/bin/env python3
"""Host-only tests for the authenticated pure-Prefill EvalScope harness."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "tools/evaluation/run_native_pure_prefill_matrix.sh"


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
        self.model_dir.mkdir()
        self.corpus_dir.mkdir()
        self.server.write_text(
            "#!/bin/sh\n"
            "# Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION\n"
            "# Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION\n"
            "# Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION\n"
            "# Q3X_FULL_ATTENTION_FLASHINFER_DIRECT\n"
            "# Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION\n"
            "exit 0\n",
            encoding="utf-8",
        )
        self.server.chmod(0o755)
        self.payload.write_bytes(b"host-only payload fixture\n")
        self.policy.write_text("{}\n", encoding="utf-8")
        self.receipt.write_text("{}\n", encoding="utf-8")
        for bucket in ("p512", "p1k", "p2k", "p4k"):
            (self.corpus_dir / f"q3x-sharegpt-prefill-{bucket}-5.jsonl").write_text(
                '{"host_only":true}\n', encoding="utf-8"
            )

    def command(self, *extra: str) -> list[str]:
        return [
            str(RUNNER),
            "--dry-run",
            "--prefill-a4-payload",
            str(self.payload),
            "--prefill-a4-policy",
            str(self.policy),
            "--prefill-a4-receipt",
            str(self.receipt),
            *extra,
            str(self.server),
            str(self.model_dir),
            str(self.corpus_dir),
            str(self.output),
            "p2k",
        ]

    def run(self, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION"] = "1"
        environment["Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION"] = "1"
        environment["Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION"] = "1"
        environment["Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION"] = "1"
        environment["Q3X_FULL_ATTENTION_FLASHINFER_DIRECT"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION"] = "1"
        environment["Q3X_GDN_CHUNK64_PROFILE_CANDIDATE"] = "1"
        return subprocess.run(
            self.command(*extra),
            env=environment,
            check=False,
            text=True,
            capture_output=True,
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
