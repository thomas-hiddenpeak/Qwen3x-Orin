#!/usr/bin/env python3
"""Run the target-AOT probe under strict parent/child nvmap observation.

The wrapper launches exactly one already-built probe.  It authenticates the
ELF before launch and while the child is alive, captures Jetson nvmap state
immediately before launch and immediately after exit, and binds those parent
observations to the probe's schema-v3 evidence.  It never clears a pool or
cache and never rewrites the child's source status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import classify_target_aot_nvmap_recovery as recovery


PARENT_ARTIFACT = "q3x_sm87_target_aot_parent_child_nvmap_lifecycle"
MAX_CHILD_STREAM_BYTES = 16 * 1024 * 1024


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def format_utc(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise recovery.EvidenceError("parent timestamp must be timezone-aware")
    return (
        value.astimezone(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc(value: str, name: str) -> dt.datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise recovery.EvidenceError(f"{name} is not ISO-8601: {error}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise recovery.EvidenceError(f"{name} must be UTC")
    return parsed.astimezone(dt.timezone.utc)


def _sha256_file(path: pathlib.Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise recovery.EvidenceError(f"cannot hash {path}: {error}") from error
    return size, digest.hexdigest()


def _sha256_fd(fd: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    offset = 0
    try:
        while True:
            chunk = os.pread(fd, 1024 * 1024, offset)
            if not chunk:
                break
            offset += len(chunk)
            digest.update(chunk)
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot hash authenticated probe fd {fd}: {error}"
        ) from error
    return offset, digest.hexdigest()


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _same_binary_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return all(
        actual.get(name) == expected.get(name)
        for name in (
            "path",
            "bytes",
            "sha256",
            "st_dev",
            "st_ino",
            "st_ctime_ns",
        )
    )


def open_authenticated_probe(
    path: pathlib.Path,
) -> tuple[int, dict[str, Any]]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise recovery.EvidenceError("--probe must be a regular non-symlink ELF")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise recovery.EvidenceError(f"cannot resolve --probe {path}: {error}") from error
    if not recovery._path_is_within(resolved, recovery.WORK_ROOT.resolve()):
        raise recovery.EvidenceError(
            "--probe must resolve inside repository .q3x-work"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError as error:
        raise recovery.EvidenceError(f"cannot open --probe {resolved}: {error}") from error
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise recovery.EvidenceError("--probe must be a regular file")
        if opened_stat.st_mode & 0o111 == 0:
            raise recovery.EvidenceError("--probe must be executable")
        try:
            path_stat = os.stat(resolved, follow_symlinks=False)
        except OSError as error:
            raise recovery.EvidenceError(
                f"cannot restat authenticated --probe {resolved}: {error}"
            ) from error
        if _stat_identity(path_stat) != _stat_identity(opened_stat):
            raise recovery.EvidenceError(
                "--probe path changed while opening its authenticated fd"
            )
        size, digest = _sha256_fd(fd)
        if size == 0 or size != opened_stat.st_size:
            raise recovery.EvidenceError("--probe size changed while hashing")
        record: dict[str, Any] = {
            "path": str(resolved),
            "bytes": size,
            "sha256": digest,
            **_stat_identity(opened_stat),
            "authentication_source": "opened_read_only_fd",
            "matches_frozen_identity": True,
        }
        return fd, record
    except BaseException:
        os.close(fd)
        raise


def authenticate_probe_executable(path: pathlib.Path) -> dict[str, Any]:
    fd, record = open_authenticated_probe(path)
    try:
        return record
    finally:
        os.close(fd)


def authenticate_retained_probe_fd(
    fd: int, expected: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        opened_stat = os.fstat(fd)
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot restat retained probe fd {fd}: {error}"
        ) from error
    size, digest = _sha256_fd(fd)
    record: dict[str, Any] = {
        "path": str(expected["path"]),
        "bytes": size,
        "sha256": digest,
        **_stat_identity(opened_stat),
        "authentication_source": "retained_prelaunch_fd_post_exit",
    }
    record["matches_prelaunch_identity"] = _same_binary_identity(record, expected)
    # The child schema freezes path, size and SHA256 but not inode metadata.
    # Preserve that narrower source-schema authentication separately from the
    # stricter parent pre/live/post inode binding.
    record["matches_frozen_identity"] = all(
        record.get(name) == expected.get(name)
        for name in ("path", "bytes", "sha256")
    )
    return record


def authenticate_probe_path_against(
    path: pathlib.Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = authenticate_probe_executable(path)
    actual["matches_prelaunch_identity"] = _same_binary_identity(actual, expected)
    return actual


def validate_model_directory(path: pathlib.Path) -> pathlib.Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot resolve --model-directory {path}: {error}"
        ) from error
    if not resolved.is_dir():
        raise recovery.EvidenceError("--model-directory must be a directory")
    return resolved


def capture_parent_snapshot(
    *,
    debugfs_root: pathlib.Path,
    proc_root: pathlib.Path,
    page_size: int,
    expected_binary_path: str,
) -> dict[str, Any]:
    started = utc_now()
    boot_id = recovery.read_boot_id(proc_root)
    nvmap = recovery.capture_nvmap(debugfs_root, page_size)
    process = recovery.scan_matching_probe_processes(
        proc_root, expected_binary_path
    )
    finished = utc_now()
    return {
        "capture_started_at_utc": format_utc(started),
        "capture_finished_at_utc": format_utc(finished),
        "boot_id": boot_id,
        "page_size_bytes": page_size,
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "machine": os.uname().machine,
        "uid": os.getuid(),
        "nvmap": nvmap,
        "probe_process_audit": process,
    }


def authenticate_live_child(
    child_pid: int,
    expected: Mapping[str, Any],
    *,
    proc_root: pathlib.Path,
) -> dict[str, Any]:
    if child_pid <= 0:
        raise recovery.EvidenceError("launched child PID must be positive")
    proc_entry = proc_root / str(child_pid)
    exe = proc_entry / "exe"
    try:
        link_target = os.readlink(exe).removesuffix(" (deleted)")
        resolved_target = pathlib.Path(link_target).resolve(strict=False)
        stat_text = (proc_entry / "stat").read_text(encoding="utf-8")
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot authenticate live child PID {child_pid}: {error}"
        ) from error
    closing = stat_text.rfind(")")
    if closing < 0:
        raise recovery.EvidenceError("live child /proc stat has no command terminator")
    stat_fields = stat_text[closing + 1 :].split()
    if len(stat_fields) <= 19 or not stat_fields[19].isdecimal():
        raise recovery.EvidenceError("live child /proc stat has invalid starttime")
    try:
        executable_stat = os.stat(exe)
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot stat live child executable for PID {child_pid}: {error}"
        ) from error
    size, digest = _sha256_file(exe)
    path_matches = resolved_target == pathlib.Path(str(expected["path"]))
    record: dict[str, Any] = {
        "captured_at_utc": format_utc(utc_now()),
        "pid": child_pid,
        "proc_exe_link_target": link_target,
        "resolved_path": str(resolved_target),
        "path": str(expected["path"]),
        "bytes": size,
        "sha256": digest,
        **_stat_identity(executable_stat),
        "proc_start_time_ticks": int(stat_fields[19]),
        "path_matches_prelaunch": path_matches,
    }
    record["matches_prelaunch_identity"] = (
        path_matches and _same_binary_identity(record, expected)
    )
    return record


def _stream_record(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise recovery.EvidenceError(
            f"cannot read retained child stream {path}: {error}"
        ) from error
    if len(payload) > MAX_CHILD_STREAM_BYTES:
        raise recovery.EvidenceError(
            f"child stream exceeds {MAX_CHILD_STREAM_BYTES} bytes"
        )
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "utf8_text": payload.decode("utf-8", errors="replace"),
    }


def _source_exit_contract(probe: Mapping[str, Any], exit_code: int) -> bool:
    return (probe["status"] == "pass" and exit_code == 0) or (
        probe["status"] == "fail" and exit_code == 1
    )


def _complete_source_pass(probe: Mapping[str, Any]) -> bool:
    if probe["status"] != "pass":
        return False
    checks = recovery._mapping(probe["checks"], "checks")
    attempted = recovery._mapping(probe["attempted"], "attempted")
    cuda_errors = recovery._mapping(probe["cuda_errors"], "cuda_errors")
    diagnostic = recovery._mapping(probe["diagnostic"], "diagnostic")
    return (
        all(value is True for value in checks.values())
        and all(value is True for value in attempted.values())
        and all(value == 0 for value in cuda_errors.values())
        and diagnostic["code"] == 0
        and diagnostic["cuda_error"] == 0
    )


def bind_child_identity(
    probe: Mapping[str, Any],
    *,
    launched_pid: int,
    child_exit_code: int,
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    prelaunch_binary: Mapping[str, Any],
    live_binary: Mapping[str, Any],
    postexit_retained_binary: Mapping[str, Any],
    postlaunch_binary: Mapping[str, Any],
    launch_requested_at: dt.datetime,
    child_wait_finished_at: dt.datetime,
) -> tuple[dict[str, bool], bool]:
    execution = recovery._mapping(
        probe.get("execution_identity"), "execution_identity"
    )
    binary = recovery._mapping(probe.get("binary"), "binary")
    child_started = parse_utc(
        str(execution["child_started_at_utc"]), "child_started_at_utc"
    )
    child_finished = parse_utc(
        str(execution["child_evidence_finished_at_utc"]),
        "child_evidence_finished_at_utc",
    )
    pre_finished = parse_utc(
        str(pre_snapshot["capture_finished_at_utc"]),
        "pre capture finished_at",
    )
    post_started = parse_utc(
        str(post_snapshot["capture_started_at_utc"]),
        "post capture started_at",
    )
    source_path = pathlib.Path(str(binary["path"])).resolve(strict=False)
    expected_path = pathlib.Path(str(prelaunch_binary["path"]))
    source_binary_matches = (
        source_path == expected_path
        and binary["bytes"] == prelaunch_binary["bytes"]
        and binary["self_sha256"] == prelaunch_binary["sha256"]
    )
    post_process = recovery._mapping(
        post_snapshot["probe_process_audit"], "post probe_process_audit"
    )
    criteria = {
        "source_probe_schema_is_v3": probe.get("schema_version") == 3,
        "source_execution_identity_declared_valid": execution.get("valid") is True,
        "source_child_pid_matches_launched_pid": (
            execution.get("child_pid") == launched_pid
        ),
        "pre_source_post_boot_ids_match": (
            pre_snapshot["boot_id"]
            == execution.get("boot_id")
            == post_snapshot["boot_id"]
        ),
        "source_binary_matches_prelaunch_identity": source_binary_matches,
        "live_child_elf_matches_prelaunch_identity": (
            live_binary.get("pid") == launched_pid
            and live_binary.get("matches_prelaunch_identity") is True
        ),
        "live_child_start_time_captured": (
            isinstance(live_binary.get("proc_start_time_ticks"), int)
            and not isinstance(live_binary.get("proc_start_time_ticks"), bool)
            and int(live_binary["proc_start_time_ticks"]) > 0
        ),
        "postexit_retained_fd_matches_prelaunch_identity": (
            postexit_retained_binary.get("matches_prelaunch_identity") is True
        ),
        "postlaunch_elf_matches_prelaunch_identity": (
            postlaunch_binary.get("matches_prelaunch_identity") is True
        ),
        "source_status_matches_child_exit_code": _source_exit_contract(
            probe, child_exit_code
        ),
        "pre_snapshot_finished_before_launch_request": (
            pre_finished <= launch_requested_at
        ),
        "child_started_after_launch_request": launch_requested_at <= child_started,
        "child_evidence_finished_after_child_start": child_started <= child_finished,
        "child_evidence_finished_before_wait_returned": (
            child_finished <= child_wait_finished_at
        ),
        "post_snapshot_started_after_wait_returned": (
            child_wait_finished_at <= post_started
        ),
        "post_exit_probe_elf_absent": (
            post_process.get("scan_complete") is True
            and not post_process.get("matching_probe_elf_pids")
        ),
    }
    return criteria, all(criteria.values())


def run_probe(
    *,
    probe_path: pathlib.Path,
    model_directory: pathlib.Path,
    child_evidence_path: pathlib.Path,
    output_path: pathlib.Path,
    debugfs_root: pathlib.Path,
    proc_root: pathlib.Path,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    live_authenticator: Callable[..., Mapping[str, Any]] = authenticate_live_child,
    snapshotter: Callable[..., Mapping[str, Any]] = capture_parent_snapshot,
) -> tuple[dict[str, Any], bool]:
    child_output = recovery.validate_output_path(child_evidence_path)
    parent_output = recovery.validate_output_path(output_path)
    if child_output == parent_output:
        raise recovery.EvidenceError(
            "--child-evidence and --output must name different files"
        )
    model = validate_model_directory(model_directory)
    presnapshot_binary = authenticate_probe_executable(probe_path)
    child_output.parent.mkdir(parents=True, exist_ok=True)
    parent_output.parent.mkdir(parents=True, exist_ok=True)
    if not recovery._path_is_within(
        child_output.parent.resolve(), recovery.WORK_ROOT.resolve()
    ) or not recovery._path_is_within(
        parent_output.parent.resolve(), recovery.WORK_ROOT.resolve()
    ):
        raise recovery.EvidenceError("evidence parent escaped repository .q3x-work")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise recovery.EvidenceError("host page size must be positive")

    pre_snapshot = dict(
        snapshotter(
            debugfs_root=debugfs_root,
            proc_root=proc_root,
            page_size=page_size,
            expected_binary_path=str(presnapshot_binary["path"]),
        )
    )
    pre_process = recovery._mapping(
        pre_snapshot["probe_process_audit"], "pre probe_process_audit"
    )
    if pre_process.get("scan_complete") is not True:
        raise recovery.EvidenceError(
            "pre-launch probe ELF process scan was incomplete"
        )
    if pre_process.get("matching_probe_elf_pids"):
        raise recovery.EvidenceError(
            "another process is already executing the authenticated probe ELF"
        )

    probe_fd, prelaunch_binary = open_authenticated_probe(probe_path)
    if not _same_binary_identity(prelaunch_binary, presnapshot_binary):
        os.close(probe_fd)
        raise recovery.EvidenceError(
            "--probe identity changed between pre-snapshot and launch fd authentication"
        )
    command = [str(prelaunch_binary["path"]), str(model), str(child_output)]
    stdout_name = ""
    stderr_name = ""
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=parent_output.parent,
                prefix=".target-aot-child-stdout.",
                delete=False,
            ) as stdout_stream,
            tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=parent_output.parent,
                prefix=".target-aot-child-stderr.",
                delete=False,
            ) as stderr_stream,
        ):
            stdout_name = stdout_stream.name
            stderr_name = stderr_stream.name
            launch_requested_at = utc_now()
            child = popen_factory(
                command,
                executable=f"/proc/self/fd/{probe_fd}",
                pass_fds=(probe_fd,),
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                close_fds=True,
            )
            launched_pid = int(child.pid)
            try:
                live_binary = dict(
                    live_authenticator(
                        launched_pid, prelaunch_binary, proc_root=proc_root
                    )
                )
            except recovery.EvidenceError as error:
                # Never abandon a successfully launched probe because its
                # live /proc authentication failed.  Wait for it, capture the
                # immediate post-exit state, and publish an unbound result.
                live_binary = {
                    "captured_at_utc": format_utc(utc_now()),
                    "pid": launched_pid,
                    "matches_prelaunch_identity": False,
                    "error": str(error),
                }
            child_exit_code = int(child.wait())
            child_wait_finished_at = utc_now()
            post_snapshot = dict(
                snapshotter(
                    debugfs_root=debugfs_root,
                    proc_root=proc_root,
                    page_size=page_size,
                    expected_binary_path=str(prelaunch_binary["path"]),
                )
            )
            stdout_stream.flush()
            stderr_stream.flush()

        child_stdout = _stream_record(pathlib.Path(stdout_name))
        child_stderr = _stream_record(pathlib.Path(stderr_name))
        postexit_retained_binary = authenticate_retained_probe_fd(
            probe_fd, prelaunch_binary
        )
        postlaunch_binary = authenticate_probe_path_against(
            probe_path, prelaunch_binary
        )
        frozen_child_path = recovery.validate_input_path(child_output)
        probe, _raw, child_sha256 = recovery.read_frozen_probe(frozen_child_path)
        recovery.validate_probe_shape(probe)
        if probe.get("schema_version") != 3:
            raise recovery.EvidenceError(
                "parent/child lifecycle wrapper requires schema-v3 child evidence"
            )

        post_captured_at = parse_utc(
            str(post_snapshot["capture_finished_at_utc"]),
            "post capture finished_at",
        )
        derived, classified = recovery.derive_report(
            probe,
            input_path=frozen_child_path,
            input_sha256=child_sha256,
            debugfs_root=debugfs_root,
            proc_root=proc_root,
            captured_at=post_captured_at,
            page_size=page_size,
            observed_boot_id=str(post_snapshot["boot_id"]),
            observed_nvmap=recovery._mapping(post_snapshot["nvmap"], "post nvmap"),
            observed_binary_authentication=postexit_retained_binary,
            observed_process_audit=recovery._mapping(
                post_snapshot["probe_process_audit"],
                "post probe_process_audit",
            ),
        )
        binding_criteria, identity_bound = bind_child_identity(
            probe,
            launched_pid=launched_pid,
            child_exit_code=child_exit_code,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            prelaunch_binary=prelaunch_binary,
            live_binary=live_binary,
            postexit_retained_binary=postexit_retained_binary,
            postlaunch_binary=postlaunch_binary,
            launch_requested_at=launch_requested_at,
            child_wait_finished_at=child_wait_finished_at,
        )
        source_pass_complete = _complete_source_pass(probe)
        observation_authority = recovery._mapping(
            derived["observation_authority"], "observation_authority"
        )
        production_observation_bound = (
            observation_authority.get("production_bound") is True
            and observation_authority.get("authority")
            == "canonical_jetson_host"
        )
        recovery_or_source_accepted = classified or (
            source_pass_complete and child_exit_code == 0
        )
        combined_acceptance_criteria = {
            "canonical_proc_and_nvmap_roots": production_observation_bound,
            "parent_child_identity_bound": identity_bound,
            "recovery_diagnostic_or_complete_source_pass": (
                recovery_or_source_accepted
            ),
        }
        accepted = all(combined_acceptance_criteria.values())
        report: dict[str, Any] = {
            "schema_version": 1,
            "artifact": PARENT_ARTIFACT,
            "captured_at_utc": format_utc(utc_now()),
            "claim_boundary": (
                "strict parent/child lifecycle and narrow post-exit nvmap "
                "diagnostic only; preserves the child source status and has "
                "no timing/performance, generation, public-launcher, "
                "production-route, or general leak-freedom authority; "
                "non-canonical proc/debugfs roots are test-only and can never "
                "set combined_lifecycle_accepted"
            ),
            "fixed_recovery_tolerance_bytes": recovery.RECOVERY_TOLERANCE_BYTES,
            "command": {
                "probe": str(prelaunch_binary["path"]),
                "model_directory": str(model),
                "child_evidence": str(child_output),
                "shell": False,
                "execution_source": "retained_authenticated_fd",
                "argv0_preserved_as_probe_path": True,
            },
            "parent": {
                "pid": os.getpid(),
                "launch_requested_at_utc": format_utc(launch_requested_at),
                "child_wait_finished_at_utc": format_utc(child_wait_finished_at),
            },
            "child": {
                "pid": launched_pid,
                "exit_code": child_exit_code,
                "prelaunch_binary": prelaunch_binary,
                "live_binary": live_binary,
                "postexit_retained_binary": postexit_retained_binary,
                "postlaunch_binary": postlaunch_binary,
                "stdout": child_stdout,
                "stderr": child_stderr,
            },
            "pre_child_snapshot": pre_snapshot,
            "immediate_post_exit_snapshot": post_snapshot,
            "source_probe": {
                "path": str(frozen_child_path),
                "sha256": child_sha256,
                "schema_version": probe["schema_version"],
                "artifact": probe["artifact"],
                "status": probe["status"],
                "status_preserved": True,
            },
            "identity_binding": {
                "criteria": binding_criteria,
                "bound": identity_bound,
            },
            "observation_authority": dict(observation_authority),
            "combined_acceptance_criteria": combined_acceptance_criteria,
            "recovery_diagnostic": derived,
            "classification": derived["classification"],
            "source_pass_complete": source_pass_complete,
            "combined_lifecycle_accepted": accepted,
        }
        recovery.publish_create_only(parent_output, report)
        return report, accepted
    finally:
        os.close(probe_fd)
        for name in (stdout_name, stderr_name):
            if name:
                try:
                    pathlib.Path(name).unlink(missing_ok=True)
                except OSError:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "launch one schema-v3 target-AOT probe with immediate pre/post "
            "Jetson nvmap lifecycle capture"
        )
    )
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--model-directory", type=pathlib.Path, required=True)
    parser.add_argument("--child-evidence", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--debugfs-root",
        type=pathlib.Path,
        default=pathlib.Path("/sys/kernel/debug/nvmap"),
        help=(
            "canonical nvmap debugfs root; overrides are test-only and can "
            "never set combined_lifecycle_accepted"
        ),
    )
    parser.add_argument(
        "--proc-root",
        type=pathlib.Path,
        default=pathlib.Path("/proc"),
        help=(
            "canonical procfs root; overrides are test-only and can never "
            "set combined_lifecycle_accepted"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, accepted = run_probe(
            probe_path=args.probe,
            model_directory=args.model_directory,
            child_evidence_path=args.child_evidence,
            output_path=args.output,
            debugfs_root=args.debugfs_root,
            proc_root=args.proc_root,
        )
    except recovery.EvidenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"classification={report['classification']}")
    print(f"source_probe_status={report['source_probe']['status']}")
    print(f"identity_bound={str(report['identity_binding']['bound']).lower()}")
    print(f"evidence={args.output.resolve(strict=False)}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
