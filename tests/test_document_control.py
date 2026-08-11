#!/usr/bin/env python3
"""Host-only tests for the tracked Markdown document-control validator."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "tools/docs/validate_document_control.py"
SPEC = importlib.util.spec_from_file_location("document_control", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {MODULE_PATH}")
CONTROL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTROL
SPEC.loader.exec_module(CONTROL)


def control_header(
    document_id: str, document_class: str, status: str = "active"
) -> str:
    return f"""---
q3x_document:
  id: {document_id}
  class: {document_class}
  status: {status}
  owner: test-maintainers
  authority: test fixture
  effective: 2026-08-12
  last_reviewed: 2026-08-12
  supersedes: []
  superseded_by: []
  ssot_for: test fixture
  review_trigger: fixture change
---

# Fixture
"""


def write_file(root: pathlib.Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def initialize_git_index(root: pathlib.Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "add", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )


class DocumentControlTest(unittest.TestCase):
    def test_current_repository_passes(self) -> None:
        result = CONTROL.validate_repository(REPOSITORY)
        self.assertTrue(result.passed, CONTROL.format_result(result))
        self.assertEqual(result.tracked_markdown, result.registry_entries)
        self.assertGreater(result.required_headers, 0)
        self.assertGreater(result.local_links_checked, 0)

    def test_inventory_rejects_missing_extra_and_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_file(
                root,
                "AGENTS.md",
                control_header("fixture-agents", "local-work-package"),
            )
            registry = control_header("fixture-registry", "procedure") + """
## Local work package (2)

| Path | Role | Lifecycle | Authority |
| --- | --- | --- | --- |
| `AGENTS.md` | `work` | active | fixture |
| `AGENTS.md` | `duplicate` | active | fixture |

## Procedures (1)

| Path | Role | Lifecycle | Authority |
| --- | --- | --- | --- |
| `docs/EXTRA.md` | `extra` | active | fixture |
"""
            write_file(root, "docs/DOCUMENT_REGISTRY.md", registry)
            initialize_git_index(root)

            result = CONTROL.validate_repository(root)
            rendered = CONTROL.format_result(result)
            self.assertFalse(result.passed)
            self.assertIn("registry: duplicate path AGENTS.md", rendered)
            self.assertIn(
                "registry: missing tracked path: docs/DOCUMENT_REGISTRY.md",
                rendered,
            )
            self.assertIn("registry: extra untracked path: docs/EXTRA.md", rendered)

    def test_first_party_headers_are_strict_but_evidence_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_file(
                root,
                "AGENTS.md",
                control_header("fixture-agents", "local-work-package"),
            )
            write_file(root, "docs/ACTIVE.md", "# Transitional content\n")
            write_file(root, "docs/EVIDENCE.md", "# Frozen evidence\n")
            write_file(root, "third_party/README.md", "# Vendor notice\n")
            entries = (
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("AGENTS.md"),
                    "local-work-package",
                    "work",
                    "active",
                    1,
                ),
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("docs/ACTIVE.md"),
                    "active",
                    "design",
                    "active",
                    2,
                ),
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("docs/EVIDENCE.md"),
                    "evidence",
                    "record",
                    "frozen",
                    3,
                ),
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("third_party/README.md"),
                    "third-party",
                    "notice",
                    "frozen",
                    4,
                ),
            )

            required, checked, errors = CONTROL.validate_headers(root, entries)
            self.assertEqual(required, 2)
            self.assertEqual(checked, 1)
            self.assertEqual(
                errors,
                ("docs/ACTIVE.md: required q3x_document header is missing",),
            )

    def test_header_class_status_and_id_must_match_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_file(
                root,
                "AGENTS.md",
                control_header("shared-id", "active", status="draft"),
            )
            write_file(
                root,
                "docs/README.md",
                control_header("shared-id", "procedure"),
            )
            entries = (
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("AGENTS.md"),
                    "local-work-package",
                    "work",
                    "active",
                    1,
                ),
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("docs/README.md"),
                    "procedure",
                    "index",
                    "active",
                    2,
                ),
            )

            _, _, errors = CONTROL.validate_headers(root, entries)
            rendered = "\n".join(errors)
            self.assertIn("header class 'active' does not match", rendered)
            self.assertIn("header status 'draft' does not match", rendered)
            self.assertIn("duplicate q3x_document id 'shared-id'", rendered)

    def test_supersession_references_must_exist_and_be_reciprocal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            successor = control_header("successor", "active").replace(
                "  supersedes: []", "  supersedes: [docs/OLD.md]"
            )
            predecessor = control_header(
                "predecessor", "historical", status="historical"
            )
            write_file(root, "docs/NEW.md", successor)
            write_file(root, "docs/OLD.md", predecessor)
            entries = (
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("docs/NEW.md"),
                    "active",
                    "design",
                    "active",
                    1,
                ),
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("docs/OLD.md"),
                    "historical",
                    "history",
                    "historical",
                    2,
                ),
            )

            _, _, errors = CONTROL.validate_headers(root, entries)
            rendered = "\n".join(errors)
            self.assertIn("is not reciprocated", rendered)

            write_file(
                root,
                "docs/OLD.md",
                predecessor.replace(
                    "  superseded_by: []", "  superseded_by: [docs/NEW.md]"
                ),
            )
            _, _, errors = CONTROL.validate_headers(root, entries)
            self.assertEqual(errors, ())

    def test_missing_supersession_field_reports_error_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_file(
                root,
                "AGENTS.md",
                control_header("fixture-agents", "local-work-package").replace(
                    "  supersedes: []\n", ""
                ),
            )
            entries = (
                CONTROL.RegistryEntry(
                    pathlib.PurePosixPath("AGENTS.md"),
                    "local-work-package",
                    "work",
                    "active",
                    1,
                ),
            )

            _, _, errors = CONTROL.validate_headers(root, entries)
            self.assertIn("AGENTS.md: missing header field supersedes", errors)

    def test_link_check_resolves_local_targets_and_ignores_code_and_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_file(root, "docs/target.md", "# Target\n")
            write_file(root, "docs/space name.md", "# Encoded\n")
            write_file(
                root,
                "docs/source.md",
                """# Source

[target](target.md#section)
[encoded](space%20name.md)
[external](https://example.com/missing.md)
[email](mailto:test@example.com)
[anchor](#source)
`[inline code](not-real.md)`

```text
[fenced code](also-not-real.md)
```

[missing](missing.md)
[escape](../../outside.md)
[reference]: target.md
""",
            )

            checked, errors = CONTROL.validate_local_links(
                root, (pathlib.PurePosixPath("docs/source.md"),)
            )
            self.assertEqual(checked, 5)
            self.assertEqual(len(errors), 2)
            rendered = "\n".join(errors)
            self.assertIn("missing local link target: missing.md", rendered)
            self.assertIn("local link escapes repository: ../../outside.md", rendered)
            self.assertNotIn("not-real.md", rendered)


if __name__ == "__main__":
    unittest.main()
