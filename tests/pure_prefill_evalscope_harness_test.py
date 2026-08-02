#!/usr/bin/env python3
"""Host-only tests for the authenticated pure-Prefill EvalScope harness."""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "tools/evaluation/run_native_pure_prefill_matrix.sh"
FRAGMENT_NATIVE_PAYLOAD_BYTES = 8_623_226_880
FRAGMENT_NATIVE_M64N128_1CTA_MODE = (
    "cumulative-prefill-current-best-mlp-k512-fragment-native-"
    "m64n128-1cta"
)
FRAGMENT_NATIVE_M64N128_1CTA_PRIMARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m64n128_1cta_gateup_primary"
)
FRAGMENT_NATIVE_M64N128_1CTA_SECONDARY = (
    "prefill_projection_span_mlp_k512_fragment_native_"
    "m64n128_1cta_gateup_secondary"
)


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
        self.k512_payload = root / "attention-o-k512.bin"
        self.k512_policy = root / "attention-o-k512-policy.json"
        self.k512_receipt = root / "attention-o-k512.bin.receipt.json"
        self.fragment_native_payload = root / "mlp-k512-fragment-native.bin"
        self.fragment_native_policy = (
            root / "mlp-k512-fragment-native-policy.json"
        )
        self.fragment_native_receipt = (
            root / "mlp-k512-fragment-native.bin.receipt.json"
        )
        self.fake_bin = root / "fake-bin"
        self.model_dir.mkdir()
        self.corpus_dir.mkdir()
        self.fake_bin.mkdir()
        self.server.write_text(
            "#!/bin/sh\n"
            "# Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION\n"
            "# Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION\n"
            "# Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION\n"
            "# Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION\n"
            "# Q3X_FULL_ATTENTION_FLASHINFER_DIRECT\n"
            "# Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION\n"
            "# Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION\n"
            "# Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION\n"
            "# Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION\n"
            "exit 0\n",
            encoding="utf-8",
        )
        self.server.chmod(0o755)
        self.payload.write_bytes(b"host-only payload fixture\n")
        self.policy.write_text("{}\n", encoding="utf-8")
        self.receipt.write_text("{}\n", encoding="utf-8")
        self.k512_payload.write_bytes(b"host-only K512 payload fixture\n")
        self.k512_policy.write_text("{}\n", encoding="utf-8")
        self.k512_receipt.write_text("{}\n", encoding="utf-8")
        with self.fragment_native_payload.open("wb") as stream:
            stream.truncate(FRAGMENT_NATIVE_PAYLOAD_BYTES)
        self.fragment_native_policy.write_text("{}\n", encoding="utf-8")
        zero_sha = "0" * 64
        self.fragment_native_receipt.write_text(
            json.dumps(
                {
                    "schema": (
                        "q3x.prefill.mlp-k512.fragment-native."
                        "publication-receipt"
                    ),
                    "physical_layout": (
                        "sm87_s4_gateup_n64_paired_down_n128_"
                        "fragment_native_scale_k512_mlp_v2"
                    ),
                    "payload_bytes": FRAGMENT_NATIVE_PAYLOAD_BYTES,
                    "payload_sha256": zero_sha,
                    "source_v1": {"policy_sha256": zero_sha},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        real_sha256sum = shutil.which("sha256sum")
        if real_sha256sum is None:
            raise RuntimeError("sha256sum is required by the harness tests")
        fake_sha256sum = self.fake_bin / "sha256sum"
        fake_sha256sum.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *mlp-k512-fragment-native*)\n"
            "    printf '%064d  %s\\n' 0 \"$1\"\n"
            "    exit 0\n"
            "    ;;\n"
            "esac\n"
            f"exec {shlex.quote(real_sha256sum)} \"$@\"\n",
            encoding="utf-8",
        )
        fake_sha256sum.chmod(0o755)
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
        environment["Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION"] = "1"
        environment["Q3X_FULL_ATTENTION_FLASHINFER_DIRECT"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION"] = "1"
        environment["Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION"] = "1"
        environment[
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION"
        ] = "1"
        environment["Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION"] = "1"
        environment["Q3X_GDN_CHUNK64_PROFILE_CANDIDATE"] = "1"
        environment["PATH"] = (
            f"{self.fake_bin}{os.pathsep}{environment['PATH']}"
        )
        return subprocess.run(
            self.command(*extra),
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )

    def run_k512(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return self.run(
            "--prefill-attention-o-k512-payload",
            str(self.k512_payload),
            "--prefill-attention-o-k512-policy",
            str(self.k512_policy),
            "--prefill-attention-o-k512-receipt",
            str(self.k512_receipt),
            "--mode",
            "cumulative-prefill-current-best-k512",
            *extra,
        )

    def enable_fragment_native_m64n128_1cta(
        self, *extra_stage_markers: str
    ) -> None:
        with self.server.open("a", encoding="utf-8") as stream:
            stream.write("Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION\n")
            stream.write(f"{FRAGMENT_NATIVE_M64N128_1CTA_PRIMARY}\n")
            stream.write(f"{FRAGMENT_NATIVE_M64N128_1CTA_SECONDARY}\n")
            stream.write(
                "prefill_projection_span_mlp_k512_fragment_native_down\n"
            )
            for marker in extra_stage_markers:
                stream.write(f"{marker}\n")

    def run_fragment_native_m64n128_1cta(
        self,
        *extra: str,
        extra_stage_markers: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        self.enable_fragment_native_m64n128_1cta(*extra_stage_markers)
        return self.run(
            "--prefill-mlp-k512-fragment-native-payload",
            str(self.fragment_native_payload),
            "--prefill-mlp-k512-fragment-native-policy",
            str(self.fragment_native_policy),
            "--prefill-mlp-k512-fragment-native-receipt",
            str(self.fragment_native_receipt),
            "--mode",
            FRAGMENT_NATIVE_M64N128_1CTA_MODE,
            *extra,
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
        self.assertIn(
            "-u Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
            result.stdout,
        )
        self.assertIn(
            "-u Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION", result.stdout
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

    def test_cumulative_current_best_mode_is_exact_eight_selector_bundle(
        self,
    ) -> None:
        result = self.fixture.run(
            "--mode", "cumulative-prefill-current-best"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "mode=cumulative-prefill-current-best dry_run=1", result.stdout
        )
        self.assertIn("selector_count=8", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        expected = {
            "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
            "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
        }
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)), expected
        )
        metadata = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("selector_metadata")
        )
        self.assertIn("selector_count=8", metadata)
        self.assertEqual(
            set(re.findall(r"Q3X_[A-Z0-9_]+", metadata)), expected
        )
        for rejected in (
            "Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION",
            "Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION",
            "Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION",
            "Q3X_GDN_CHUNK64_PROFILE_CANDIDATE",
        ):
            self.assertIn(f"-u {rejected}", startup)
            self.assertNotIn(f"{rejected}=1", startup)

    def test_cumulative_current_best_requires_every_new_selector(self) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        new_selectors = (
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
        )
        for selector in new_selectors:
            with self.subTest(selector=selector):
                self.fixture.server.write_text(
                    contents.replace(f"# {selector}\n", ""), encoding="utf-8"
                )
                result = self.fixture.run(
                    "--mode", "cumulative-prefill-current-best"
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "server does not contain the "
                    f"cumulative-prefill-current-best selector: {selector}",
                    result.stderr,
                )
        self.fixture.server.write_text(contents, encoding="utf-8")

    def test_cumulative_current_best_k512_is_exact_nine_selector_bundle(
        self,
    ) -> None:
        result = self.fixture.run_k512()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "mode=cumulative-prefill-current-best-k512 dry_run=1",
            result.stdout,
        )
        self.assertIn("selector_count=9", result.stdout)
        startup = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith("server_startup_command")
        )
        expected = {
            "Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION",
            "Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION",
            "Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION",
            "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION",
            "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
            "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
            "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
            "Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION",
        }
        self.assertEqual(
            set(re.findall(r"(Q3X_[A-Z0-9_]+)=1", startup)), expected
        )
        self.assertIn("--prefill-attention-o-k512-payload", startup)
        self.assertIn(str(self.fixture.k512_payload), startup)
        self.assertIn("--prefill-attention-o-k512-policy", startup)
        self.assertIn(str(self.fixture.k512_policy), startup)
        self.assertIn("--prefill-attention-o-k512-receipt", startup)
        self.assertIn(str(self.fixture.k512_receipt), startup)
        self.assertIn(
            "prefill_attention_o_k512_authenticated_64_of_64",
            result.stdout,
        )
        self.assertIn(
            "prefill_attention_o_k512_payload_sha256", result.stdout
        )

    def test_cumulative_current_best_k512_requires_all_overlay_paths(
        self,
    ) -> None:
        result = self.fixture.run(
            "--prefill-attention-o-k512-payload",
            str(self.fixture.k512_payload),
            "--prefill-attention-o-k512-policy",
            str(self.fixture.k512_policy),
            "--mode",
            "cumulative-prefill-current-best-k512",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "missing required Prefill Attention-O K512 receipt", result.stderr
        )
        self.assertFalse(self.fixture.output.exists())

    def test_cumulative_current_best_k512_requires_compiled_selector(
        self,
    ) -> None:
        contents = self.fixture.server.read_text(encoding="utf-8")
        self.fixture.server.write_text(
            contents.replace(
                "# Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION\n", ""
            ),
            encoding="utf-8",
        )
        result = self.fixture.run_k512()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "server does not contain the "
            "cumulative-prefill-current-best-k512 selector: "
            "Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION",
            result.stderr,
        )

    def test_k512_readiness_contract_proves_overlay_and_payload_digest(
        self,
    ) -> None:
        contents = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            "prefill_attention_o_k512_requested=1 .*"
            "prefill_attention_o_k512_enabled=1 .*"
            "prefill_attention_o_k512_projections=64",
            contents,
        )
        self.assertIn(
            "prefill_attention_o_k512_payload_sha256=[0-9a-f]{64}",
            contents,
        )

    def test_fragment_native_m64n128_1cta_mode_uses_authenticated_real_route(
        self,
    ) -> None:
        result = self.fixture.run_fragment_native_m64n128_1cta()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"mode={FRAGMENT_NATIVE_M64N128_1CTA_MODE} dry_run=1",
            result.stdout,
        )
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
                "Q3X_FULL_ATTENTION_FLASHINFER_DIRECT",
                "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION",
                "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION",
                "Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION",
            },
        )
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-payload", startup
        )
        self.assertIn(str(self.fixture.fragment_native_payload), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-policy", startup
        )
        self.assertIn(str(self.fixture.fragment_native_policy), startup)
        self.assertIn(
            "--prefill-mlp-k512-fragment-native-receipt", startup
        )
        self.assertIn(str(self.fixture.fragment_native_receipt), startup)
        self.assertIn(
            "gateup_variant=m64n128_1cta down_variant=m64n128",
            result.stdout,
        )
        self.assertIn(
            "prefill_mlp_k512_fragment_native_gateup_variant_m64n128_1cta",
            result.stdout,
        )
        self.assertIn("performance_evidence=0", result.stdout)
        self.assertFalse(self.fixture.output.exists())

    def test_fragment_native_policy_must_match_publication_receipt(
        self,
    ) -> None:
        receipt = json.loads(
            self.fixture.fragment_native_receipt.read_text(encoding="utf-8")
        )
        receipt["source_v1"]["policy_sha256"] = "f" * 64
        self.fixture.fragment_native_receipt.write_text(
            json.dumps(receipt) + "\n", encoding="utf-8"
        )
        result = self.fixture.run_fragment_native_m64n128_1cta()
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "fragment-native policy SHA256 does not match source_v1 receipt",
            result.stderr,
        )
        self.assertFalse(self.fixture.output.exists())

    def test_fragment_native_m64n128_1cta_rejects_all_old_gateup_markers(
        self,
    ) -> None:
        rejected_markers = (
            "prefill_projection_span_mlp_k512_fragment_native_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128_gateup_primary",
            "prefill_projection_span_mlp_k512_fragment_native_"
            "m128n64_1cta_gateup_primary",
        )
        for marker in rejected_markers:
            with self.subTest(marker=marker):
                with tempfile.TemporaryDirectory() as root:
                    fixture = Fixture(pathlib.Path(root))
                    result = fixture.run_fragment_native_m64n128_1cta(
                        extra_stage_markers=(marker,)
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        "server contains the mutually exclusive "
                        f"fragment-native Gate+Up variant: {marker}",
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
