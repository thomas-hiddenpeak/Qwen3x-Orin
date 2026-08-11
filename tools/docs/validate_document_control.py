#!/usr/bin/env python3
"""Validate the tracked Markdown control plane without modifying the tree.

The validator deliberately depends only on the Python standard library and
Git.  It compares the literal registry inventory with the Git index, checks
the established first-party control-header floor, and resolves repository-
local Markdown links.  It is safe to run from CI or from any working
directory: no files, index entries, or repository configuration are changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import pathlib
import re
import subprocess
import sys
import urllib.parse
from collections.abc import Iterable, Sequence


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = pathlib.PurePosixPath("docs/DOCUMENT_REGISTRY.md")

SECTION_CLASSES = {
    "Local work package": "local-work-package",
    "Normative": "normative",
    "Active first-party documents": "active",
    "Contracts": "contract",
    "Procedures": "procedure",
    "Historical": "historical",
    "Evidence": "evidence",
    "External reference documents": "external",
    "Third-party documents": "third-party",
}

# Every currently registered first-party non-evidence document has crossed
# the standard-header transition.  Keeping this class floor explicit makes a
# removed header fail instead of disappearing from a dynamically discovered
# list.  Frozen evidence remains optional and vendored documents are outside
# the first-party header contract.
REQUIRED_HEADER_CLASSES = frozenset(
    {
        "local-work-package",
        "normative",
        "active",
        "contract",
        "procedure",
        "historical",
        "external",
    }
)
VALID_HEADER_CLASSES = REQUIRED_HEADER_CLASSES | {"evidence"}
REQUIRED_HEADER_FIELDS = (
    "id",
    "class",
    "status",
    "owner",
    "authority",
    "effective",
    "last_reviewed",
    "supersedes",
    "superseded_by",
    "ssot_for",
    "review_trigger",
)
VALID_STATUSES = frozenset(
    {"draft", "active", "frozen", "historical", "superseded"}
)
DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SECTION_PATTERN = re.compile(r"^##\s+(.+?)(?:\s+\([0-9]+\))?\s*$")
REGISTRY_PATH_PATTERN = re.compile(r"^`([^`]+\.md)`$")
HEADER_FIELD_PATTERN = re.compile(r"^  ([a-z][a-z0-9_]*):(?:\s*(.*))?$")
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
REFERENCE_DEFINITION_PATTERN = re.compile(
    r"^ {0,3}\[[^\]\n]+\]:\s*(?:<([^>\n]+)>|(\S+))"
)


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    path: pathlib.PurePosixPath
    document_class: str
    role: str
    lifecycle: str
    line: int


@dataclasses.dataclass(frozen=True)
class RegistryInventory:
    entries: tuple[RegistryEntry, ...]
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class HeaderParse:
    fields: dict[str, str] | None
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    tracked_markdown: int
    registry_entries: int
    required_headers: int
    headers_checked: int
    local_links_checked: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


class GitInventoryError(RuntimeError):
    """Git could not provide the tracked Markdown inventory."""


def _unquote_scalar(value: str) -> str:
    stripped = value.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1]
        and stripped[0] in {"'", '"'}
    ):
        return stripped[1:-1]
    return stripped


def _section_name(heading: str) -> str:
    match = SECTION_PATTERN.match(heading)
    return match.group(1) if match else ""


def parse_registry_text(text: str) -> RegistryInventory:
    """Parse literal path rows and their registry class/lifecycle."""

    entries: list[RegistryEntry] = []
    errors: list[str] = []
    current_class: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            current_class = SECTION_CLASSES.get(_section_name(line))
            continue
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        path_match = REGISTRY_PATH_PATTERN.fullmatch(cells[0])
        if path_match is None:
            continue
        path_text = path_match.group(1)
        path = pathlib.PurePosixPath(path_text)
        if current_class is None:
            errors.append(
                f"registry:{line_number}: path row is outside a recognized class "
                f"section: {path_text}"
            )
            continue
        if len(cells) < 3:
            errors.append(
                f"registry:{line_number}: path row has no role/lifecycle: {path_text}"
            )
            continue
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            errors.append(
                f"registry:{line_number}: invalid repository-relative Markdown "
                f"path: {path_text}"
            )
            continue
        role = _unquote_scalar(cells[1])
        lifecycle = _unquote_scalar(cells[2])
        if not role or not lifecycle:
            errors.append(
                f"registry:{line_number}: empty role/lifecycle: {path_text}"
            )
            continue
        entries.append(
            RegistryEntry(
                path=path,
                document_class=current_class,
                role=role,
                lifecycle=lifecycle,
                line=line_number,
            )
        )

    by_path: dict[pathlib.PurePosixPath, list[int]] = {}
    for entry in entries:
        by_path.setdefault(entry.path, []).append(entry.line)
    for path, lines in sorted(by_path.items(), key=lambda item: str(item[0])):
        if len(lines) > 1:
            rendered_lines = ",".join(str(line) for line in lines)
            errors.append(
                f"registry: duplicate path {path} at lines {rendered_lines}"
            )
    if not entries:
        errors.append("registry: no literal Markdown path entries found")

    return RegistryInventory(tuple(entries), tuple(sorted(set(errors))))


def parse_registry(path: pathlib.Path) -> RegistryInventory:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return RegistryInventory((), (f"registry: cannot read {REGISTRY_PATH}: {error}",))
    return parse_registry_text(text)


def tracked_markdown_paths(root: pathlib.Path) -> tuple[pathlib.PurePosixPath, ...]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", "*.md"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as error:
        raise GitInventoryError(f"cannot execute git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise GitInventoryError(f"git ls-files failed: {detail}")
    paths = []
    for value in completed.stdout.splitlines():
        path = pathlib.PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise GitInventoryError(f"git returned an invalid Markdown path: {value}")
        paths.append(path)
    return tuple(sorted(paths, key=str))


def _parse_iso_date(value: str, label: str, errors: list[str]) -> None:
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError:
        errors.append(f"{label} must be an ISO YYYY-MM-DD date: {value!r}")
        return
    if parsed.isoformat() != value:
        errors.append(f"{label} must use canonical YYYY-MM-DD form: {value!r}")


def _parse_header_list(value: str) -> tuple[str, ...]:
    """Parse the bounded inline-list representation accepted above."""

    inner = value[1:-1].strip()
    if not inner:
        return ()
    return tuple(
        _unquote_scalar(item.strip()) for item in inner.split(",") if item.strip()
    )


def parse_control_header(path: pathlib.Path, display_path: str) -> HeaderParse:
    """Parse and validate the standard q3x_document YAML subset."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return HeaderParse(None, (f"{display_path}: cannot read Markdown: {error}",))
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return HeaderParse(None, ())
    try:
        end = lines.index("---", 1)
    except ValueError:
        return HeaderParse(
            None, (f"{display_path}: unterminated YAML front matter",)
        )
    front_matter = lines[1:end]
    if not front_matter or front_matter[0] != "q3x_document:":
        return HeaderParse(None, ())

    fields: dict[str, str] = {}
    errors: list[str] = []
    active_block_field: str | None = None
    block_items: list[str] = []

    def close_block() -> None:
        nonlocal active_block_field, block_items
        if active_block_field is not None:
            fields[active_block_field] = "[" + ", ".join(block_items) + "]"
            active_block_field = None
            block_items = []

    for offset, line in enumerate(front_matter[1:], start=3):
        if not line.strip():
            continue
        item_match = re.fullmatch(r"    -\s+(.+)", line)
        if item_match is not None and active_block_field is not None:
            block_items.append(item_match.group(1).strip())
            continue
        close_block()
        field_match = HEADER_FIELD_PATTERN.fullmatch(line)
        if field_match is None:
            errors.append(
                f"{display_path}:{offset}: invalid q3x_document field indentation"
            )
            continue
        name = field_match.group(1)
        value = (field_match.group(2) or "").strip()
        if name in fields:
            errors.append(f"{display_path}:{offset}: duplicate header field {name}")
            continue
        if value:
            fields[name] = _unquote_scalar(value)
        elif name in {"supersedes", "superseded_by"}:
            active_block_field = name
        else:
            fields[name] = ""
    close_block()

    present = set(fields)
    required = set(REQUIRED_HEADER_FIELDS)
    for name in sorted(required - present):
        errors.append(f"{display_path}: missing header field {name}")
    for name in sorted(present - required):
        errors.append(f"{display_path}: unknown header field {name}")

    for name in (
        "id",
        "class",
        "status",
        "owner",
        "authority",
        "effective",
        "last_reviewed",
        "ssot_for",
        "review_trigger",
    ):
        if name in fields and not fields[name]:
            errors.append(f"{display_path}: header field {name} is empty")
    if fields.get("id") and not DOCUMENT_ID_PATTERN.fullmatch(fields["id"]):
        errors.append(
            f"{display_path}: header id is not stable kebab-case: {fields['id']!r}"
        )
    if fields.get("class") and fields["class"] not in VALID_HEADER_CLASSES:
        errors.append(
            f"{display_path}: invalid first-party header class: {fields['class']!r}"
        )
    if fields.get("status") and fields["status"] not in VALID_STATUSES:
        errors.append(
            f"{display_path}: invalid header status: {fields['status']!r}"
        )
    for name in ("effective", "last_reviewed"):
        if fields.get(name):
            _parse_iso_date(fields[name], f"{display_path}: {name}", errors)
    for name in ("supersedes", "superseded_by"):
        value = fields.get(name)
        if value is not None and not (value.startswith("[") and value.endswith("]")):
            errors.append(
                f"{display_path}: header field {name} must be a YAML list"
            )

    return HeaderParse(fields, tuple(sorted(set(errors))))


def validate_headers(
    root: pathlib.Path, entries: Iterable[RegistryEntry]
) -> tuple[int, int, tuple[str, ...]]:
    required_count = 0
    checked_count = 0
    errors: list[str] = []
    seen_ids: dict[str, pathlib.PurePosixPath] = {}
    headers_by_path: dict[pathlib.PurePosixPath, dict[str, str]] = {}

    for entry in sorted(entries, key=lambda item: str(item.path)):
        if entry.document_class == "third-party":
            continue
        required = entry.document_class in REQUIRED_HEADER_CLASSES
        if required:
            required_count += 1
        parsed = parse_control_header(root / entry.path, str(entry.path))
        errors.extend(parsed.errors)
        if parsed.fields is None:
            if required:
                errors.append(f"{entry.path}: required q3x_document header is missing")
            continue
        checked_count += 1
        fields = parsed.fields
        headers_by_path[entry.path] = fields
        if fields.get("class") != entry.document_class:
            errors.append(
                f"{entry.path}: header class {fields.get('class')!r} does not match "
                f"registry class {entry.document_class!r}"
            )
        if fields.get("status") != entry.lifecycle:
            errors.append(
                f"{entry.path}: header status {fields.get('status')!r} does not match "
                f"registry lifecycle {entry.lifecycle!r}"
            )
        document_id = fields.get("id")
        if document_id:
            previous = seen_ids.get(document_id)
            if previous is not None:
                errors.append(
                    f"{entry.path}: duplicate q3x_document id {document_id!r}; "
                    f"already used by {previous}"
                )
            else:
                seen_ids[document_id] = entry.path

    def resolve_reference(reference: str) -> pathlib.PurePosixPath | None:
        path_reference = pathlib.PurePosixPath(reference)
        if path_reference in headers_by_path:
            return path_reference
        return seen_ids.get(reference)

    reciprocal_fields = {
        "supersedes": "superseded_by",
        "superseded_by": "supersedes",
    }
    for source, fields in sorted(headers_by_path.items(), key=lambda item: str(item[0])):
        for field, reciprocal in reciprocal_fields.items():
            for reference in _parse_header_list(fields.get(field, "[]")):
                target = resolve_reference(reference)
                if target is None:
                    errors.append(
                        f"{source}: {field} references unknown document "
                        f"{reference!r}"
                    )
                    continue
                target_fields = headers_by_path[target]
                reciprocal_targets = {
                    resolve_reference(value)
                    for value in _parse_header_list(
                        target_fields.get(reciprocal, "[]")
                    )
                }
                if source not in reciprocal_targets:
                    errors.append(
                        f"{source}: {field} reference {reference!r} is not "
                        f"reciprocated by {target}:{reciprocal}"
                    )

    return required_count, checked_count, tuple(sorted(set(errors)))


def _mask_inline_code(line: str) -> str:
    """Replace complete CommonMark-style backtick spans with spaces."""

    characters = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        run_end = index
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        marker = line[index:run_end]
        close = line.find(marker, run_end)
        if close < 0:
            index = run_end
            continue
        for masked in range(index, close + len(marker)):
            characters[masked] = " "
        index = close + len(marker)
    return "".join(characters)


def _inline_destinations(line: str) -> Iterable[str]:
    """Yield destinations from inline links/images on one non-code line."""

    index = 0
    while True:
        close_label = line.find("](", index)
        if close_label < 0:
            return
        open_label = line.rfind("[", 0, close_label)
        if open_label < 0 or (
            close_label > 0 and line[close_label - 1] == "\\"
        ):
            index = close_label + 2
            continue
        cursor = close_label + 2
        depth = 1
        escaped = False
        while cursor < len(line):
            character = line[cursor]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        if depth != 0:
            return
        raw = line[close_label + 2 : cursor].strip()
        if raw.startswith("<"):
            end_angle = raw.find(">", 1)
            if end_angle >= 0:
                yield raw[1:end_angle]
        elif raw:
            # An unescaped whitespace begins an optional Markdown title.
            destination: list[str] = []
            escaped = False
            for character in raw:
                if escaped:
                    destination.append(character)
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character.isspace():
                    break
                else:
                    destination.append(character)
            if escaped:
                destination.append("\\")
            if destination:
                yield "".join(destination)
        index = cursor + 1


def markdown_destinations(text: str) -> Iterable[tuple[int, str]]:
    """Yield line-numbered inline and reference Markdown destinations."""

    fence_character: str | None = None
    fence_length = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        fence = FENCE_PATTERN.match(line)
        if fence is not None:
            marker = fence.group(1)
            if fence_character is None:
                fence_character = marker[0]
                fence_length = len(marker)
                continue
            if marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue

        masked = _mask_inline_code(line)
        reference = REFERENCE_DEFINITION_PATTERN.match(masked)
        if reference is not None:
            yield line_number, reference.group(1) or reference.group(2)
        for destination in _inline_destinations(masked):
            yield line_number, destination


def _local_link_target(
    root: pathlib.Path,
    source: pathlib.PurePosixPath,
    destination: str,
) -> pathlib.Path | None:
    value = destination.strip()
    if not value or value.startswith("#") or value.startswith("//"):
        return None
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    decoded = urllib.parse.unquote(parsed.path)
    if decoded.startswith("/"):
        candidate = root / decoded.lstrip("/")
    else:
        candidate = root / source.parent / decoded
    return candidate.resolve(strict=False)


def validate_local_links(
    root: pathlib.Path, tracked: Iterable[pathlib.PurePosixPath]
) -> tuple[int, tuple[str, ...]]:
    root_resolved = root.resolve()
    checked = 0
    errors: list[str] = []
    for source in sorted(tracked, key=str):
        source_path = root / source
        try:
            text = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"{source}: cannot read Markdown for link check: {error}")
            continue
        for line, destination in markdown_destinations(text):
            target = _local_link_target(root_resolved, source, destination)
            if target is None:
                continue
            checked += 1
            try:
                target.relative_to(root_resolved)
            except ValueError:
                errors.append(
                    f"{source}:{line}: local link escapes repository: {destination}"
                )
                continue
            if not target.exists():
                errors.append(
                    f"{source}:{line}: missing local link target: {destination}"
                )
    return checked, tuple(sorted(set(errors)))


def validate_repository(root: pathlib.Path) -> ValidationResult:
    root = root.resolve()
    errors: list[str] = []
    try:
        tracked = tracked_markdown_paths(root)
    except GitInventoryError as error:
        tracked = ()
        errors.append(f"git: {error}")

    inventory = parse_registry(root / REGISTRY_PATH)
    errors.extend(inventory.errors)

    tracked_set = set(tracked)
    registry_set = {entry.path for entry in inventory.entries}
    for path in sorted(tracked_set - registry_set, key=str):
        errors.append(f"registry: missing tracked path: {path}")
    for path in sorted(registry_set - tracked_set, key=str):
        errors.append(f"registry: extra untracked path: {path}")

    required_headers, headers_checked, header_errors = validate_headers(
        root, inventory.entries
    )
    errors.extend(header_errors)
    local_links_checked, link_errors = validate_local_links(root, tracked)
    errors.extend(link_errors)

    return ValidationResult(
        tracked_markdown=len(tracked),
        registry_entries=len(inventory.entries),
        required_headers=required_headers,
        headers_checked=headers_checked,
        local_links_checked=local_links_checked,
        errors=tuple(sorted(set(errors))),
    )


def format_result(result: ValidationResult) -> str:
    lines = [
        f"document-control: {'PASS' if result.passed else 'FAIL'}",
        f"tracked_markdown: {result.tracked_markdown}",
        f"registry_entries: {result.registry_entries}",
        f"required_headers: {result.required_headers}",
        f"headers_checked: {result.headers_checked}",
        f"local_links_checked: {result.local_links_checked}",
        f"errors: {len(result.errors)}",
    ]
    lines.extend(f"error: {error}" for error in result.errors)
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Qwen3x-Orin tracked Markdown document control."
    )
    parser.add_argument(
        "--repository",
        type=pathlib.Path,
        default=REPOSITORY_ROOT,
        help="repository root (default: inferred from this script)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(sys.argv[1:] if argv is None else argv)
    result = validate_repository(arguments.repository)
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
