#!/usr/bin/env python3
"""Create a fresh Orin preflight and immediately launch the real-P40 witness.

The launcher is intentionally narrow: it never changes cooling controls or
other workloads and never overwrites evidence.  Launcher configuration errors
return 2, launcher collection or hard-stop failures return 1, and only an
actually launched probe has its 0/1/2/3 status returned unchanged.  The source
process/GPU audit comes from
``orin_perf_preflight.py`` but is sanitized in memory before it is retained.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

sys.dont_write_bytecode = True

import orin_perf_preflight as source_preflight


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK_ROOT = REPOSITORY_ROOT / ".q3x-work"
PRODUCER_PATH = pathlib.Path(__file__).resolve()
SOURCE_PREFLIGHT_PATH = pathlib.Path(source_preflight.__file__).resolve()
SCHEMA = "q3x.sm87.aot-system-v1.real-p40-arithmetic-witness.preflight.v1"
DECISION_UNIT = "bmma-static-support-k16-parent-zero-fill-v2"
WORK_PACKAGE_ID = "P3"
ARCHITECTURE_CANDIDATE_ID = "AC-PREFILL-SM87-AOT-SYSTEM-v1"
FRESHNESS_LIMIT_SECONDS = 120
SOURCE_PREFLIGHT_SAMPLES = 5
SOURCE_PREFLIGHT_INTERVAL_MS = 200
FROZEN_MAX_GR3D_PERCENT = 0.0
FROZEN_MAX_UNEXPECTED_CPU_PERCENT = (
    source_preflight.DEFAULT_MAX_UNEXPECTED_CPU_PERCENT
)
MAX_PREFLIGHT_BYTES = 1024 * 1024
MAX_RETAINED_STRING_CHARACTERS = 4096
GPU_DEVFREQ_ROOT = pathlib.Path("/sys/class/devfreq/17000000.gpu")
GPU_MAX_FREQUENCY_PATH = GPU_DEVFREQ_ROOT / "max_freq"
GPU_AVAILABLE_FREQUENCIES_PATH = GPU_DEVFREQ_ROOT / "available_frequencies"
EMC_DEVFREQ_ROOT = pathlib.Path("/sys/class/devfreq/bwmgr")
FAN_TEXT_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])fan(?:[^a-z0-9]|$)")
ROOT_SOURCE_PREFLIGHT_HELPER = r"""
import json
import pathlib
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import orin_perf_preflight as preflight
config = preflight.PreflightConfig(
    output=pathlib.Path(sys.argv[2]),
    samples=int(sys.argv[3]),
    interval_ms=int(sys.argv[4]),
    max_gr3d_percent=float(sys.argv[5]),
    max_unexpected_cpu_percent=float(sys.argv[6]),
    allow_pids=(),
    force=False,
)
report, status = preflight.run_preflight(config)
json.dump({"report": report, "status": status}, sys.stdout,
          ensure_ascii=False, separators=(",", ":"), sort_keys=True)
"""


class LauncherError(RuntimeError):
    """The requested run cannot produce authenticated create-only evidence."""


@dataclasses.dataclass(frozen=True)
class Config:
    probe: pathlib.Path
    model_directory: pathlib.Path
    corpus_jsonl: pathlib.Path
    preflight_json: pathlib.Path
    evidence_json: pathlib.Path


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_input(path: pathlib.Path, *, directory: bool, executable: bool) -> pathlib.Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise LauncherError(f"input does not resolve: {path}: {error}") from error
    if directory:
        if not resolved.is_dir():
            raise LauncherError(f"input is not a directory: {resolved}")
    elif not resolved.is_file():
        raise LauncherError(f"input is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise LauncherError(f"probe is not executable: {resolved}")
    return resolved


def resolve_create_only_output(path: pathlib.Path) -> pathlib.Path:
    work_root = WORK_ROOT.resolve(strict=True)
    requested = path.expanduser()
    if not requested.is_absolute():
        requested = pathlib.Path.cwd() / requested
    preliminary = requested.resolve(strict=False)
    if not _is_within(preliminary, work_root) or preliminary == work_root:
        raise LauncherError(
            f"output must resolve below the current worktree .q3x-work: {path}"
        )
    if requested.name in {"", ".", ".."}:
        raise LauncherError("output must name a JSON file")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    if not _is_within(parent, work_root):
        raise LauncherError("output parent escaped the current worktree .q3x-work")
    resolved = parent / requested.name
    try:
        os.lstat(resolved)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise LauncherError(f"cannot inspect output path {resolved}: {error}") from error
    else:
        raise LauncherError(f"refusing to replace existing output: {resolved}")
    return resolved


def write_json_create_only(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    if len(payload) > MAX_PREFLIGHT_BYTES:
        raise LauncherError("sanitized preflight exceeds the probe's 1 MiB limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("create-only evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _read_text(path: pathlib.Path, maximum_bytes: int = 4096) -> str:
    with path.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise LauncherError(f"telemetry field is unexpectedly large: {path}")
    return raw.rstrip(b"\0\r\n").decode("utf-8", errors="replace")


def _read_uint(path: pathlib.Path) -> int:
    text = _read_text(path).strip()
    try:
        value = int(text, 10)
    except ValueError as error:
        raise LauncherError(f"telemetry field is not an integer: {path}") from error
    if value < 0:
        raise LauncherError(f"telemetry field is negative: {path}")
    return value


def _read_frequency_list(path: pathlib.Path) -> list[int]:
    values: list[int] = []
    for token in _read_text(path, 32_768).split():
        try:
            value = int(token, 10)
        except ValueError as error:
            raise LauncherError(f"invalid frequency list: {path}") from error
        if value <= 0:
            raise LauncherError(f"non-positive frequency in {path}")
        values.append(value)
    if not values:
        raise LauncherError(f"empty frequency list: {path}")
    return values


def _format_cpu_set(cpus: set[int]) -> str:
    if not cpus:
        return ""
    ranges: list[str] = []
    ordered = sorted(cpus)
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = cpu
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def collect_cpu_affinity() -> dict[str, Any]:
    cpus = set(os.sched_getaffinity(0))
    if not cpus:
        raise LauncherError("launcher CPU affinity is empty")
    return {
        "launcher_pid": os.getpid(),
        "cpu_ids": sorted(cpus),
        "cpu_list": _format_cpu_set(cpus),
        "cpu_count": len(cpus),
    }


def collect_host_device_identity() -> dict[str, Any]:
    uname = os.uname()
    device_tree_root = pathlib.Path("/proc/device-tree")
    model_path = device_tree_root / "model"
    compatible_path = device_tree_root / "compatible"
    model = _read_text(model_path) if model_path.is_file() else None
    compatible: list[str] = []
    if compatible_path.is_file():
        raw = compatible_path.read_bytes()[:16_384]
        compatible = [
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        ]
    return {
        "hostname": uname.nodename,
        "kernel_release": uname.release,
        "machine": uname.machine,
        "uid": os.getuid(),
        "device_tree_model": model,
        "device_tree_compatible": compatible,
    }


def collect_memory_snapshot() -> dict[str, Any]:
    selected = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SReclaimable",
        "Shmem",
        "SwapTotal",
        "SwapFree",
    }
    values: dict[str, int] = {}
    with pathlib.Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            key = fields[0].removesuffix(":") if fields else ""
            if key not in selected or len(fields) < 2:
                continue
            try:
                raw = int(fields[1], 10)
            except ValueError as error:
                raise LauncherError(f"invalid /proc/meminfo field: {key}") from error
            multiplier = 1024 if len(fields) >= 3 and fields[2] == "kB" else 1
            values[f"{key}_bytes"] = raw * multiplier
    if "MemTotal_bytes" not in values or "MemAvailable_bytes" not in values:
        raise LauncherError("/proc/meminfo lacks required memory fields")
    return {"observed_at_unix_ns": time.time_ns(), **values}


def collect_temperatures() -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for zone in sorted(pathlib.Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            zone_type = _read_text(zone / "type").strip()
            # Thermal-zone values are temperature sensors, not cooling controls.
            # A cooling-controller-named zone is nevertheless excluded entirely.
            if FAN_TEXT_PATTERN.search(zone_type):
                continue
            millidegrees = _read_uint(zone / "temp")
        except (OSError, LauncherError) as error:
            errors.append({"zone": zone.name, "error": str(error)})
            continue
        celsius = millidegrees / 1000.0
        if not math.isfinite(celsius) or celsius < -100.0 or celsius > 250.0:
            errors.append({"zone": zone.name, "error": "temperature out of range"})
            continue
        samples.append(
            {
                "zone": zone.name,
                "type": zone_type,
                "millidegrees_celsius": millidegrees,
                "celsius": celsius,
            }
        )
    maximum = max((sample["celsius"] for sample in samples), default=None)
    return {
        "observed_at_unix_ns": time.time_ns(),
        "samples": samples,
        "unavailable_zones": errors,
        "maximum_celsius": maximum,
        "normal_through_celsius": 85.0,
        "operational_stop_above_celsius": 90.0,
        "above_normal_requires_clock_overcurrent_throttle_context": (
            maximum is not None and 85.0 < maximum <= 90.0
        ),
        "operational_stop": maximum is not None and maximum > 90.0,
    }


def _collect_devfreq(root: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {"canonical_path": str(root.resolve(strict=True))}
    for name in ("name", "governor"):
        path = root / name
        result[name] = _read_text(path).strip() if path.is_file() else None
    for name in ("cur_freq", "min_freq", "max_freq"):
        result[f"{name}_hz"] = _read_uint(root / name)
    available_path = root / "available_frequencies"
    result["available_frequencies_hz"] = _read_frequency_list(available_path)
    return result


def collect_device_clocks() -> dict[str, Any]:
    cpu_policies: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for policy in sorted(pathlib.Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        item: dict[str, Any] = {"policy": policy.name}
        for name in (
            "scaling_cur_freq",
            "scaling_min_freq",
            "scaling_max_freq",
            "cpuinfo_min_freq",
            "cpuinfo_max_freq",
        ):
            path = policy / name
            if path.is_file():
                try:
                    item[f"{name}_hz"] = _read_uint(path) * 1000
                except (OSError, LauncherError) as error:
                    errors.append({"path": str(path), "error": str(error)})
        governor = policy / "scaling_governor"
        try:
            item["scaling_governor"] = (
                _read_text(governor).strip() if governor.is_file() else None
            )
        except OSError as error:
            item["scaling_governor"] = None
            errors.append({"path": str(governor), "error": str(error)})
        cpu_policies.append(item)
    try:
        gpu = _collect_devfreq(GPU_DEVFREQ_ROOT)
    except (OSError, LauncherError) as error:
        gpu = {"error": str(error)}
        errors.append({"path": str(GPU_DEVFREQ_ROOT), "error": str(error)})
    try:
        emc = _collect_devfreq(EMC_DEVFREQ_ROOT)
    except (OSError, LauncherError) as error:
        emc = {"error": str(error)}
        errors.append({"path": str(EMC_DEVFREQ_ROOT), "error": str(error)})
    return {
        "observed_at_unix_ns": time.time_ns(),
        "cpu_policies": cpu_policies,
        "gpu": gpu,
        "emc": emc,
        "collection_errors": errors,
    }


def _sanitize_text_lines(text: str) -> tuple[str, int]:
    kept: list[str] = []
    discarded = 0
    for line in text.splitlines():
        if FAN_TEXT_PATTERN.search(line):
            discarded += 1
        else:
            kept.append(line)
    return "\n".join(kept), discarded


def collect_nvpmodel() -> tuple[dict[str, Any], bool, int]:
    executable_text = shutil.which("nvpmodel")
    if executable_text is None:
        return {"error": "nvpmodel executable not found"}, False, 0
    executable = pathlib.Path(executable_text).resolve()
    command = [str(executable), "-q"]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
            text=True,
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}, False, 0
    stdout, stdout_discarded = _sanitize_text_lines(completed.stdout)
    stderr, stderr_discarded = _sanitize_text_lines(completed.stderr)
    recorded = completed.returncode == 0 and bool(stdout.strip())
    return (
        {
            "command": command,
            "executable_sha256": sha256_file(executable),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        },
        recorded,
        stdout_discarded + stderr_discarded,
    )


def run_source_preflight(
    config: Config,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Run the read-only ownership audit with enough privilege to inspect /proc."""

    sudo_text = shutil.which("sudo")
    if sudo_text is None:
        raise LauncherError("sudo is unavailable for the exhaustive GPU fd audit")
    sudo = pathlib.Path(sudo_text).resolve()
    python = pathlib.Path(sys.executable).resolve()
    command = [
        str(sudo),
        "-n",
        str(python),
        "-B",
        "-c",
        ROOT_SOURCE_PREFLIGHT_HELPER,
        str(SOURCE_PREFLIGHT_PATH.parent),
        str(config.preflight_json),
        str(SOURCE_PREFLIGHT_SAMPLES),
        str(SOURCE_PREFLIGHT_INTERVAL_MS),
        str(FROZEN_MAX_GR3D_PERCENT),
        str(FROZEN_MAX_UNEXPECTED_CPU_PERCENT),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30.0,
            text=True,
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LauncherError(f"source preflight execution failed: {error}") from error
    stderr, discarded_lines = _sanitize_text_lines(completed.stderr)
    execution = {
        "privilege_command": str(sudo),
        "privilege_command_sha256": sha256_file(sudo),
        "python_executable": str(python),
        "python_executable_sha256": sha256_file(python),
        "helper_sha256": hashlib.sha256(
            ROOT_SOURCE_PREFLIGHT_HELPER.encode("utf-8")
        ).hexdigest(),
        "returncode": completed.returncode,
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr": stderr,
        "cooling_controller_stderr_lines_discarded": discarded_lines,
    }
    if completed.returncode != 0:
        raise LauncherError(
            f"privileged source preflight helper exited {completed.returncode}"
        )
    if len(completed.stdout.encode("utf-8")) > 2 * MAX_PREFLIGHT_BYTES:
        raise LauncherError("source preflight helper output exceeds 2 MiB")
    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise LauncherError("source preflight helper returned invalid JSON") from error
    if not isinstance(envelope, Mapping) or not isinstance(
        envelope.get("report"), Mapping
    ):
        raise LauncherError("source preflight helper returned an invalid envelope")
    try:
        status = int(envelope["status"])
    except (KeyError, TypeError, ValueError) as error:
        raise LauncherError("source preflight helper omitted its status") from error
    return dict(envelope["report"]), status, execution


def sanitize_source_report(value: Any) -> tuple[Any, dict[str, int]]:
    """Drop raw monitor lines and any incidental cooling-controller field."""

    counts = {
        "raw_monitor_fields_discarded": 0,
        "cooling_controller_fields_discarded": 0,
        "cooling_controller_text_values_discarded": 0,
        "long_text_values_truncated": 0,
    }

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            output: dict[str, Any] = {}
            for key, child in item.items():
                folded = str(key).casefold()
                if folded in {"raw", "unparsed_lines"}:
                    counts["raw_monitor_fields_discarded"] += 1
                    continue
                if "fan" in folded:
                    counts["cooling_controller_fields_discarded"] += 1
                    continue
                output[str(key)] = visit(child)
            return output
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str) and FAN_TEXT_PATTERN.search(item):
            counts["cooling_controller_text_values_discarded"] += 1
            return "[discarded incidental cooling-controller telemetry]"
        if isinstance(item, str) and len(item) > MAX_RETAINED_STRING_CHARACTERS:
            counts["long_text_values_truncated"] += 1
            return item[:MAX_RETAINED_STRING_CHARACTERS] + "[truncated]"
        return item

    return visit(value), counts


def classify_source_preflight(
    report: Mapping[str, Any], source_status: int
) -> dict[str, Any]:
    decision = report.get("decision", {})
    reasons = decision.get("reasons", []) if isinstance(decision, Mapping) else []
    reason_codes = {
        str(reason.get("code"))
        for reason in reasons
        if isinstance(reason, Mapping)
    }
    gpu_audit = report.get("gpu_device_fd_audit", {})
    holders = gpu_audit.get("holders", []) if isinstance(gpu_audit, Mapping) else []
    gpu_ownership_audit_complete = bool(
        gpu_audit.get("complete") if isinstance(gpu_audit, Mapping) else False
    )
    unowned_handles = sum(
        1
        for holder in holders
        if isinstance(holder, Mapping) and not bool(holder.get("allowed"))
    )
    material_contention = bool(
        reason_codes
        & {
            "gr3d_busy",
            "unexpected_cpu_consumers",
            "unexpected_gpu_fd_holders",
        }
    )
    unknown_reason_codes = reason_codes - {
        "gr3d_busy",
        "unexpected_cpu_consumers",
        "unexpected_gpu_fd_holders",
    }
    decision_pass = (
        source_status == 0
        and isinstance(decision, Mapping)
        and bool(decision.get("accepted"))
        and decision.get("result") == "pass"
    )
    auxiliary_collection_gap = (
        source_status == 1
        and isinstance(decision, Mapping)
        and not bool(decision.get("accepted"))
        and decision.get("result") == "collection_error"
    )
    core_clear = (
        gpu_ownership_audit_complete
        and unowned_handles == 0
        and not material_contention
        and not unknown_reason_codes
        and (decision_pass or auxiliary_collection_gap)
    )
    return {
        "decision": decision,
        "reason_codes": reason_codes,
        "unowned_gpu_handle_count": unowned_handles,
        "gpu_ownership_audit_complete": gpu_ownership_audit_complete,
        "material_contention_detected": material_contention,
        "source_status": source_status,
        "core_clear": core_clear,
        "auxiliary_collection_gap": auxiliary_collection_gap and core_clear,
    }


def hard_stop_reasons(
    source_state: Mapping[str, Any],
    source_status: int,
    temperature_envelope: Mapping[str, Any],
    sync_completed: bool,
    ambient_q3x_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Return only core/safety stops for this ordinary direction screen."""

    reasons: list[dict[str, Any]] = []
    decision = source_state.get("decision", {})
    if not bool(source_state.get("core_clear")):
        reasons.append(
            {
                "code": "source_preflight_core_rejected",
                "source_status": source_status,
                "source_result": (
                    decision.get("result") if isinstance(decision, Mapping) else None
                ),
            }
        )
    unowned_handles = int(source_state.get("unowned_gpu_handle_count", 0))
    if unowned_handles:
        reasons.append({"code": "unowned_gpu_handles", "count": unowned_handles})
    if bool(source_state.get("material_contention_detected")):
        reason_codes = source_state.get("reason_codes", set())
        reasons.append(
            {"code": "material_contention", "source_codes": sorted(reason_codes)}
        )
    if bool(temperature_envelope.get("operational_stop")):
        reasons.append(
            {
                "code": "temperature_above_operational_limit",
                "maximum_celsius": temperature_envelope.get("maximum_celsius"),
            }
        )
    if not sync_completed:
        reasons.append({"code": "sync_failed"})
    if ambient_q3x_names:
        reasons.append(
            {
                "code": "ambient_q3x_environment",
                "variable_names": list(ambient_q3x_names),
            }
        )
    return reasons


def run_cache_protocol() -> dict[str, Any]:
    memory_before = collect_memory_snapshot()
    sync_started = time.time_ns()
    sync_completed = False
    sync_error: str | None = None
    try:
        os.sync()
        sync_completed = True
    except OSError as error:
        sync_error = str(error)
    sync_finished = time.time_ns()

    sudo = shutil.which("sudo")
    tee = shutil.which("tee")
    command = [sudo or "sudo", "-n", tee or "tee", "/proc/sys/vm/drop_caches"]
    drop_started = time.time_ns()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    error_text: str | None = None
    try:
        completed = subprocess.run(
            command,
            input="3\n",
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
            text=True,
            close_fds=True,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        error_text = str(error)
    drop_finished = time.time_ns()
    memory_after = collect_memory_snapshot()
    return {
        "memory_before": memory_before,
        "sync": {
            "started_at_unix_ns": sync_started,
            "finished_at_unix_ns": sync_finished,
            "completed": sync_completed,
            "error": sync_error,
        },
        "cache_drop": {
            "command": command,
            "input": "3\\n",
            "attempted": True,
            "started_at_unix_ns": drop_started,
            "finished_at_unix_ns": drop_finished,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": error_text,
            "succeeded": returncode == 0,
        },
        "memory_after": memory_after,
    }


def build_preflight_record(config: Config) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    host = collect_host_device_identity()
    affinity = collect_cpu_affinity()
    clocks = collect_device_clocks()
    nvpmodel, nvpmodel_recorded, nvpmodel_discarded = collect_nvpmodel()

    # This witness invocation admits only the source helper itself (added internally
    # by orin_perf_preflight) and its descendants.  An external PID allowlist
    # would let a caller relabel a foreign GPU owner as trusted.
    raw_source_report, source_status, source_execution = run_source_preflight(config)
    raw_source_sha256 = canonical_json_sha256(raw_source_report)
    sanitized_source, sanitization = sanitize_source_report(raw_source_report)
    sanitization["nvpmodel_lines_discarded"] = nvpmodel_discarded
    sanitized_source_sha256 = canonical_json_sha256(sanitized_source)

    cache = run_cache_protocol()
    temperatures = collect_temperatures()
    source_state = classify_source_preflight(sanitized_source, source_status)
    unowned_handles = source_state["unowned_gpu_handle_count"]
    material_contention = source_state["material_contention_detected"]
    temperature_stop = bool(temperatures["operational_stop"])
    temperature_recorded = bool(temperatures["samples"])
    gpu_max_path = GPU_MAX_FREQUENCY_PATH.resolve(strict=True)
    gpu_available_path = GPU_AVAILABLE_FREQUENCIES_PATH.resolve(strict=True)
    gpu_max_hz = _read_uint(gpu_max_path)
    gpu_available = _read_frequency_list(gpu_available_path)
    clocks_recorded = (
        bool(clocks["cpu_policies"])
        and not bool(clocks.get("collection_errors"))
    )
    ambient_q3x_names = sorted(name for name in os.environ if name.startswith("Q3X_"))

    hard_stops = hard_stop_reasons(
        source_state,
        source_status,
        temperatures,
        bool(cache["sync"]["completed"]),
        ambient_q3x_names,
    )

    producer_sha256 = sha256_file(PRODUCER_PATH)
    source_preflight_producer_sha256 = sha256_file(SOURCE_PREFLIGHT_PATH)
    probe_sha256 = sha256_file(config.probe)
    created_at_unix_ns = time.time_ns()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "decision_unit": DECISION_UNIT,
        "work_package_id": WORK_PACKAGE_ID,
        "architecture_candidate_id": ARCHITECTURE_CANDIDATE_ID,
        "created_at_utc": dt.datetime.fromtimestamp(
            created_at_unix_ns / 1_000_000_000, tz=dt.timezone.utc
        ).isoformat(),
        "created_at_unix_ns": created_at_unix_ns,
        "freshness_limit_seconds": FRESHNESS_LIMIT_SECONDS,
        "launcher_pid": os.getpid(),
        "hostname": host["hostname"],
        "machine": host["machine"],
        "producer_path": str(PRODUCER_PATH),
        "producer_sha256": producer_sha256,
        "source_preflight_producer_path": str(SOURCE_PREFLIGHT_PATH),
        "source_preflight_producer_sha256": source_preflight_producer_sha256,
        "source_preflight_report_sha256": sanitized_source_sha256,
        "source_preflight_unsanitized_report_sha256": raw_source_sha256,
        "probe_sha256": probe_sha256,
        "gpu_max_frequency_canonical_path": str(gpu_max_path),
        "gpu_max_frequency_hz": gpu_max_hz,
        "gpu_available_frequencies_canonical_path": str(gpu_available_path),
        "gpu_available_max_frequency_hz": max(gpu_available),
        "hard_stop_clear": not hard_stops,
        "fan_fields_sanitized": True,
        "material_contention_detected": material_contention,
        "safety_stop": temperature_stop,
        "temperature_operational_stop": temperature_stop,
        "sync_completed": bool(cache["sync"]["completed"]),
        "cache_drop_attempted": bool(cache["cache_drop"]["attempted"]),
        "cache_drop_succeeded": bool(cache["cache_drop"]["succeeded"]),
        "cpu_affinity_recorded": bool(affinity["cpu_ids"]),
        "nvpmodel_recorded": nvpmodel_recorded,
        "device_clocks_recorded": clocks_recorded,
        "temperature_envelope_recorded": temperature_recorded,
        "unowned_gpu_handle_count": unowned_handles,
        "ambient_q3x_environment_clean": not ambient_q3x_names,
        "host_device_identity": host,
        "cpu_affinity": affinity,
        "nvpmodel": nvpmodel,
        "device_clocks": clocks,
        "temperature_envelope": temperatures,
        "cache_protocol": cache,
        "sanitization": sanitization,
        "source_preflight": sanitized_source,
        "source_preflight_execution": source_execution,
        "hard_stop_reasons": hard_stops,
        "launch_contract": {
            "probe": str(config.probe),
            "model_directory": str(config.model_directory),
            "corpus_jsonl": str(config.corpus_jsonl),
            "preflight_json": str(config.preflight_json),
            "evidence_json": str(config.evidence_json),
            "probe_sha256": probe_sha256,
            "probe_exit_status_preserved": [0, 1, 2, 3],
        },
    }
    return record, hard_stops


def launch_probe(config: Config) -> int:
    command = [
        str(config.probe),
        str(config.model_directory),
        str(config.corpus_jsonl),
        str(config.preflight_json),
        str(config.evidence_json),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            check=False,
            close_fds=True,
        )
    except OSError as error:
        print(
            json.dumps(
                {"result": "probe_launch_failed", "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    if completed.returncode in {0, 1, 2, 3}:
        return completed.returncode
    print(
        json.dumps(
            {
                "result": "unexpected_probe_status",
                "probe_returncode": completed.returncode,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Create an authenticated real-P40 preflight and immediately run "
            "the SM87 arithmetic witness"
        )
    )
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--model-directory", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-jsonl", type=pathlib.Path, required=True)
    parser.add_argument("--preflight-json", type=pathlib.Path, required=True)
    parser.add_argument("--evidence-json", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    preflight_json = resolve_create_only_output(args.preflight_json)
    evidence_json = resolve_create_only_output(args.evidence_json)
    if preflight_json == evidence_json:
        raise LauncherError("preflight and probe evidence paths must differ")
    return Config(
        probe=resolve_input(args.probe, directory=False, executable=True),
        model_directory=resolve_input(
            args.model_directory, directory=True, executable=False
        ),
        corpus_jsonl=resolve_input(
            args.corpus_jsonl, directory=False, executable=False
        ),
        preflight_json=preflight_json,
        evidence_json=evidence_json,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
    except LauncherError as error:
        print(
            json.dumps(
                {"result": "launcher_configuration_error", "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        record, hard_stops = build_preflight_record(config)
        write_json_create_only(config.preflight_json, record)
    except (LauncherError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"result": "launcher_collection_error", "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    if hard_stops:
        print(
            json.dumps(
                {
                    "result": "preflight_hard_stop",
                    "preflight": str(config.preflight_json),
                    "reasons": hard_stops,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    return launch_probe(config)


if __name__ == "__main__":
    raise SystemExit(main())
