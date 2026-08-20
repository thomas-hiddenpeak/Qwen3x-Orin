#!/usr/bin/env python3
"""Fail closed when an Orin performance lane is not exclusively available.

GR3D utilization comes only from tegrastats.  The complementary /proc audits
find visible processes that retain GPU device descriptors and processes that
consume a material fraction of one CPU core.  This tool is observational: it
never signals, stops, renices, or otherwise mutates another process.
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
import selectors
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK_ROOT = REPOSITORY_ROOT / ".q3x-work"
TEGRASTATS_PATH = pathlib.Path("/usr/bin/tegrastats")
GR3D_PATTERN = re.compile(r"(?:^|\s)GR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%")
GPU_NVHOST_PATTERN = re.compile(r"^/dev/nvhost-[^/]*gpu(?:$|[-_])")
GPU_NVIDIA_PATTERN = re.compile(
    r"^/dev/nvidia(?:[0-9]+|ctl|uvm(?:-tools)?|modeset)$"
)
GPU_RENDER_PATTERN = re.compile(r"^/dev/dri/renderD[0-9]+$")


class ConfigError(RuntimeError):
    """The requested preflight cannot produce repository-owned evidence."""


class ParseError(ValueError):
    """One operating-system telemetry record is malformed or ambiguous."""


@dataclasses.dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int
    comm: str
    cmdline: str
    cwd: str
    cpu_ticks: int
    start_time_ticks: int


@dataclasses.dataclass(frozen=True)
class CpuProcessSample:
    process: ProcessSnapshot
    percent_of_one_core: float


@dataclasses.dataclass(frozen=True)
class PreflightConfig:
    output: pathlib.Path
    samples: int
    interval_ms: int
    max_gr3d_percent: float
    max_unexpected_cpu_percent: float
    allow_pids: tuple[int, ...]
    force: bool


def parse_gr3d_percent(line: str) -> float | None:
    """Return the single GR3D percentage in one tegrastats line."""

    matches = GR3D_PATTERN.findall(line)
    if not matches:
        return None
    if len(matches) != 1:
        raise ParseError("tegrastats line contains multiple GR3D fields")
    value = float(matches[0])
    if not math.isfinite(value) or value < 0.0 or value > 100.0:
        raise ParseError(f"GR3D percentage is outside [0,100]: {value}")
    return value


def parse_proc_stat_line(line: str) -> ProcessSnapshot:
    """Parse the identity and CPU fields from /proc/PID/stat."""

    open_paren = line.find("(")
    close_paren = line.rfind(")")
    if open_paren <= 0 or close_paren <= open_paren:
        raise ParseError("process stat has no bounded command name")
    try:
        pid = int(line[:open_paren].strip())
    except ValueError as error:
        raise ParseError("process stat PID is not an integer") from error
    fields = line[close_paren + 1 :].strip().split()
    # fields[0] is field 3 (state); starttime is field 22.
    if len(fields) < 20:
        raise ParseError("process stat is missing required CPU fields")
    try:
        ppid = int(fields[1])
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        start_time_ticks = int(fields[19])
    except ValueError as error:
        raise ParseError("process stat CPU fields are not integers") from error
    if pid <= 0 or ppid < 0 or user_ticks < 0 or system_ticks < 0:
        raise ParseError("process stat contains an invalid negative field")
    return ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        comm=line[open_paren + 1 : close_paren],
        cmdline="",
        cwd="",
        cpu_ticks=user_ticks + system_ticks,
        start_time_ticks=start_time_ticks,
    )


def parse_system_cpu_line(line: str) -> tuple[int, int]:
    fields = line.split()
    if len(fields) < 5 or fields[0] != "cpu":
        raise ParseError("/proc/stat does not begin with aggregate CPU data")
    try:
        counters = [int(value) for value in fields[1:]]
    except ValueError as error:
        raise ParseError("aggregate CPU counters are not integers") from error
    if any(value < 0 for value in counters):
        raise ParseError("aggregate CPU counters contain a negative value")
    total = sum(counters)
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    return total, idle


def is_gpu_specific_device(target: str) -> bool:
    normalized = target.removesuffix(" (deleted)")
    return bool(
        GPU_NVHOST_PATTERN.match(normalized)
        or GPU_NVIDIA_PATTERN.match(normalized)
        or GPU_RENDER_PATTERN.match(normalized)
    )


def is_gpu_auxiliary_device(target: str) -> bool:
    return target.removesuffix(" (deleted)") == "/dev/nvmap"


def _read_process_snapshot(path: pathlib.Path) -> ProcessSnapshot:
    parsed = parse_proc_stat_line((path / "stat").read_text(encoding="utf-8"))
    try:
        raw_cmdline = (path / "cmdline").read_bytes()[:16_384]
        cmdline = raw_cmdline.replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
    except OSError:
        cmdline = ""
    try:
        cwd = os.readlink(path / "cwd")
    except OSError:
        cwd = ""
    return dataclasses.replace(parsed, cmdline=cmdline, cwd=cwd)


def collect_process_snapshot(
    proc_root: pathlib.Path,
) -> tuple[dict[int, ProcessSnapshot], list[dict[str, Any]]]:
    processes: dict[int, ProcessSnapshot] = {}
    skipped: list[dict[str, Any]] = []
    try:
        entries = sorted(
            (entry for entry in proc_root.iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
    except OSError as error:
        raise RuntimeError(f"cannot enumerate {proc_root}: {error}") from error
    for entry in entries:
        try:
            snapshot = _read_process_snapshot(entry)
        except FileNotFoundError:
            continue
        except (OSError, ParseError) as error:
            skipped.append({"pid": int(entry.name), "error": str(error)})
            continue
        if snapshot.pid != int(entry.name):
            skipped.append(
                {"pid": int(entry.name), "error": "PID changed during scan"}
            )
            continue
        processes[snapshot.pid] = snapshot
    return processes, skipped


def collect_system_cpu(proc_root: pathlib.Path) -> tuple[int, int]:
    with (proc_root / "stat").open("r", encoding="utf-8") as stream:
        return parse_system_cpu_line(stream.readline())


def cpu_process_samples(
    first: Mapping[int, ProcessSnapshot],
    second: Mapping[int, ProcessSnapshot],
    elapsed_seconds: float,
    clock_ticks_per_second: int,
) -> list[CpuProcessSample]:
    if elapsed_seconds <= 0.0 or clock_ticks_per_second <= 0:
        raise ValueError("CPU sampling interval and clock rate must be positive")
    samples: list[CpuProcessSample] = []
    for pid, after in second.items():
        before = first.get(pid)
        if before is None or before.start_time_ticks != after.start_time_ticks:
            continue
        delta = after.cpu_ticks - before.cpu_ticks
        if delta < 0:
            continue
        percent = (
            float(delta)
            / float(clock_ticks_per_second)
            / elapsed_seconds
            * 100.0
        )
        samples.append(CpuProcessSample(after, percent))
    samples.sort(key=lambda sample: (-sample.percent_of_one_core, sample.process.pid))
    return samples


def expand_allowed_pids(
    processes: Mapping[int, ProcessSnapshot], roots: Sequence[int]
) -> set[int]:
    allowed = set(roots)
    changed = True
    while changed:
        changed = False
        for process in processes.values():
            if process.pid not in allowed and process.ppid in allowed:
                allowed.add(process.pid)
                changed = True
    return allowed


def path_is_within(path: str | pathlib.Path, root: pathlib.Path) -> bool:
    if not path:
        return False
    try:
        pathlib.Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def process_is_project_owned(
    process: ProcessSnapshot, repository_root: pathlib.Path
) -> bool:
    if path_is_within(process.cwd, repository_root):
        return True
    root_text = str(repository_root.resolve())
    return root_text in process.cmdline


def scan_gpu_fd_holders(
    proc_root: pathlib.Path,
    processes: dict[int, ProcessSnapshot],
    allowed_pids: set[int],
    repository_root: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    holders: list[dict[str, Any]] = []
    inaccessible: list[dict[str, Any]] = []
    scanned_fds = 0
    for pid in sorted(processes):
        fd_root = proc_root / str(pid) / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as error:
            inaccessible.append({"pid": pid, "error": str(error)})
            continue
        specific: set[str] = set()
        auxiliary: set[str] = set()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            except OSError as error:
                inaccessible.append(
                    {"pid": pid, "fd": descriptor.name, "error": str(error)}
                )
                continue
            scanned_fds += 1
            if is_gpu_specific_device(target):
                specific.add(target.removesuffix(" (deleted)"))
            elif is_gpu_auxiliary_device(target):
                auxiliary.add(target.removesuffix(" (deleted)"))
        # nvmap is shared by non-GPU engines; report it only when a process
        # also proves GPU ownership through a GPU-specific descriptor.
        if not specific:
            continue
        process = processes[pid]
        holders.append(
            {
                "pid": pid,
                "ppid": process.ppid,
                "comm": process.comm,
                "cmdline": process.cmdline,
                "cwd": process.cwd,
                "device_fds": sorted(specific | auxiliary),
                "allowed": pid in allowed_pids,
                "project_owned": process_is_project_owned(
                    process, repository_root
                ),
            }
        )
    return holders, inaccessible, scanned_fds


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_tegrastats(
    sample_count: int, interval_ms: int
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    resolved = TEGRASTATS_PATH.resolve()
    evidence: dict[str, Any] = {
        "binary": str(resolved),
        "binary_sha256": None,
        "command": [str(resolved), "--interval", str(interval_ms)],
        "requested_samples": sample_count,
        "samples": [],
        "unparsed_lines": [],
    }
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        errors.append(f"tegrastats is not executable: {resolved}")
        return evidence, errors
    try:
        evidence["binary_sha256"] = file_sha256(resolved)
    except OSError as error:
        errors.append(f"cannot hash tegrastats: {error}")
        return evidence, errors

    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    raw_buffer = b""
    deadline = time.monotonic() + max(
        5.0, sample_count * interval_ms / 1000.0 + 3.0
    )
    try:
        process = subprocess.Popen(
            evidence["command"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        assert process.stdout is not None
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while len(evidence["samples"]) < sample_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                errors.append("timed out waiting for tegrastats GR3D samples")
                break
            events = selector.select(min(remaining, 0.5))
            if not events:
                if process.poll() is not None:
                    errors.append(
                        f"tegrastats exited before enough samples: "
                        f"status={process.returncode}"
                    )
                    break
                continue
            try:
                chunk = os.read(process.stdout.fileno(), 65_536)
            except BlockingIOError:
                continue
            if not chunk:
                if process.poll() is not None:
                    errors.append(
                        f"tegrastats closed output before enough samples: "
                        f"status={process.returncode}"
                    )
                    break
                continue
            raw_buffer += chunk
            while b"\n" in raw_buffer:
                raw_line, raw_buffer = raw_buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    gr3d = parse_gr3d_percent(line)
                except ParseError as error:
                    errors.append(str(error))
                    evidence["unparsed_lines"].append(line)
                    continue
                if gr3d is None:
                    if len(evidence["unparsed_lines"]) < 20:
                        evidence["unparsed_lines"].append(line)
                    continue
                evidence["samples"].append(
                    {
                        "observed_at_unix_ns": time.time_ns(),
                        "gr3d_percent": gr3d,
                        "raw": line,
                    }
                )
                if len(evidence["samples"]) >= sample_count:
                    break
    except OSError as error:
        errors.append(f"cannot execute tegrastats: {error}")
    finally:
        if selector is not None:
            selector.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process is not None:
            evidence["collector_returncode"] = process.returncode

    values = [sample["gr3d_percent"] for sample in evidence["samples"]]
    evidence["collected_samples"] = len(values)
    evidence["maximum_gr3d_percent"] = max(values) if values else None
    evidence["mean_gr3d_percent"] = (
        sum(values) / len(values) if values else None
    )
    if len(values) != sample_count and not errors:
        errors.append(
            f"expected {sample_count} GR3D samples, collected {len(values)}"
        )
    return evidence, errors


def _process_json(
    sample: CpuProcessSample,
    allowed_pids: set[int],
    repository_root: pathlib.Path,
) -> dict[str, Any]:
    process = sample.process
    return {
        "pid": process.pid,
        "ppid": process.ppid,
        "comm": process.comm,
        "cmdline": process.cmdline,
        "cwd": process.cwd,
        "percent_of_one_core": sample.percent_of_one_core,
        "allowed": process.pid in allowed_pids,
        "project_owned": process_is_project_owned(process, repository_root),
    }


def resource_rejection_reasons(
    maximum_gr3d_percent: float | None,
    max_gr3d_percent: float,
    gpu_holders: Sequence[Mapping[str, Any]],
    cpu_samples: Sequence[CpuProcessSample],
    allowed_pids: set[int],
    max_unexpected_cpu_percent: float,
    missing_allowed_pids: Sequence[int],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if (
        maximum_gr3d_percent is not None
        and maximum_gr3d_percent > max_gr3d_percent
    ):
        reasons.append(
            {
                "code": "gr3d_busy",
                "observed_percent": maximum_gr3d_percent,
                "allowed_percent": max_gr3d_percent,
            }
        )
    unexpected_holders = [
        holder for holder in gpu_holders if not bool(holder.get("allowed"))
    ]
    if unexpected_holders:
        reasons.append(
            {
                "code": "unexpected_gpu_fd_holders",
                "pids": [int(holder["pid"]) for holder in unexpected_holders],
            }
        )
    unexpected_cpu = [
        sample
        for sample in cpu_samples
        if sample.process.pid not in allowed_pids
        and sample.percent_of_one_core >= max_unexpected_cpu_percent
    ]
    if unexpected_cpu:
        reasons.append(
            {
                "code": "unexpected_cpu_consumers",
                "pids": [sample.process.pid for sample in unexpected_cpu],
                "threshold_percent_of_one_core": max_unexpected_cpu_percent,
            }
        )
    if missing_allowed_pids:
        reasons.append(
            {
                "code": "allowed_pid_not_running",
                "pids": list(missing_allowed_pids),
            }
        )
    return reasons


def validate_output_path(path: pathlib.Path) -> pathlib.Path:
    repository = REPOSITORY_ROOT.resolve()
    work = WORK_ROOT.resolve()
    if not path_is_within(work, repository):
        raise ConfigError("repository .q3x-work resolves outside the repository")
    resolved = path.expanduser().resolve()
    if not path_is_within(resolved, work):
        raise ConfigError("--output must resolve inside repository .q3x-work")
    if resolved == work or resolved.name in {"", ".", ".."}:
        raise ConfigError("--output must name one JSON evidence file")
    if resolved.exists() and resolved.is_dir():
        raise ConfigError("--output names a directory")
    return resolved


def write_json_atomic(
    path: pathlib.Path, value: Mapping[str, Any], force: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise ConfigError(f"refusing to replace {path}; pass --force explicitly")
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary_name, path)
            temporary_name = ""
        else:
            try:
                os.link(temporary_name, path)
            except FileExistsError as error:
                raise ConfigError(
                    f"refusing to replace {path}; pass --force explicitly"
                ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name:
            try:
                pathlib.Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def run_preflight(config: PreflightConfig) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    collection_errors: list[str] = []
    proc_root = pathlib.Path("/proc")
    try:
        first_cpu_total = collect_system_cpu(proc_root)
        first_processes, first_skipped = collect_process_snapshot(proc_root)
    except (OSError, ParseError, RuntimeError) as error:
        first_cpu_total = (0, 0)
        first_processes = {}
        first_skipped = []
        collection_errors.append(f"initial /proc sample failed: {error}")

    cpu_sample_started = time.monotonic()
    tegra, tegra_errors = collect_tegrastats(config.samples, config.interval_ms)
    collection_errors.extend(tegra_errors)
    elapsed_so_far = time.monotonic() - cpu_sample_started
    if elapsed_so_far < 0.25:
        time.sleep(0.25 - elapsed_so_far)

    try:
        second_cpu_total = collect_system_cpu(proc_root)
        second_processes, second_skipped = collect_process_snapshot(proc_root)
    except (OSError, ParseError, RuntimeError) as error:
        second_cpu_total = (0, 0)
        second_processes = {}
        second_skipped = []
        collection_errors.append(f"final /proc sample failed: {error}")
    cpu_elapsed = time.monotonic() - cpu_sample_started

    missing_allowed = [
        pid for pid in config.allow_pids if pid not in second_processes
    ]
    allowed_pids = expand_allowed_pids(second_processes, config.allow_pids)
    allowed_pids.add(os.getpid())

    try:
        holders, inaccessible_fds, scanned_fds = scan_gpu_fd_holders(
            proc_root,
            second_processes,
            allowed_pids,
            REPOSITORY_ROOT,
        )
    except RuntimeError as error:
        holders, inaccessible_fds, scanned_fds = [], [], 0
        collection_errors.append(f"GPU fd scan failed: {error}")

    try:
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
        cpu_samples = cpu_process_samples(
            first_processes, second_processes, cpu_elapsed, clock_ticks
        )
    except (OSError, ValueError) as error:
        clock_ticks = 0
        cpu_samples = []
        collection_errors.append(f"process CPU calculation failed: {error}")

    aggregate_cpu_percent: float | None = None
    total_delta = second_cpu_total[0] - first_cpu_total[0]
    idle_delta = second_cpu_total[1] - first_cpu_total[1]
    if total_delta > 0 and 0 <= idle_delta <= total_delta:
        aggregate_cpu_percent = (total_delta - idle_delta) / total_delta * 100.0
    elif not collection_errors:
        collection_errors.append("aggregate CPU counters did not advance safely")

    maximum_gr3d = tegra.get("maximum_gr3d_percent")
    reasons = resource_rejection_reasons(
        float(maximum_gr3d) if isinstance(maximum_gr3d, (int, float)) else None,
        config.max_gr3d_percent,
        holders,
        cpu_samples,
        allowed_pids,
        config.max_unexpected_cpu_percent,
        missing_allowed,
    )

    if collection_errors:
        result = "collection_error"
        accepted = False
        exit_status = 1
    elif reasons:
        result = "reject_busy"
        accepted = False
        exit_status = 3
    else:
        result = "pass"
        accepted = True
        exit_status = 0

    top_cpu = cpu_samples[:64]
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "q3x_orin_perf_preflight",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "host": {
            "hostname": os.uname().nodename,
            "kernel": os.uname().release,
            "machine": os.uname().machine,
            "uid": os.getuid(),
        },
        "configuration": {
            "samples": config.samples,
            "interval_ms": config.interval_ms,
            "max_gr3d_percent": config.max_gr3d_percent,
            "max_unexpected_cpu_percent_of_one_core": (
                config.max_unexpected_cpu_percent
            ),
            "requested_allowed_pids": list(config.allow_pids),
            "expanded_allowed_pids": sorted(allowed_pids - {os.getpid()}),
        },
        "tegrastats": tegra,
        "gpu_device_fd_audit": {
            "scanned_fds": scanned_fds,
            "holders": holders,
            "inaccessible": inaccessible_fds,
            "matching_rule": (
                "GPU-specific nvhost/nvidia/render descriptors; nvmap only "
                "when paired with a GPU-specific descriptor"
            ),
        },
        "cpu_audit": {
            "interval_seconds": cpu_elapsed,
            "clock_ticks_per_second": clock_ticks,
            "aggregate_busy_percent": aggregate_cpu_percent,
            "top_processes": [
                _process_json(sample, allowed_pids, REPOSITORY_ROOT)
                for sample in top_cpu
            ],
            "initial_process_scan_skips": first_skipped,
            "final_process_scan_skips": second_skipped,
        },
        "decision": {
            "accepted": accepted,
            "result": result,
            "reasons": reasons,
            "collection_errors": collection_errors,
        },
        "collector_elapsed_seconds": time.monotonic() - started,
    }
    return report, exit_status


def parse_args(argv: Sequence[str] | None = None) -> PreflightConfig:
    parser = argparse.ArgumentParser(
        description="Fail closed unless an Orin performance lane is exclusive"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--max-gr3d-percent", type=float, default=0.0)
    parser.add_argument(
        "--max-unexpected-cpu-percent", type=float, default=25.0
    )
    parser.add_argument(
        "--allow-pid",
        action="append",
        default=[],
        type=int,
        help="Allow this expected PID and all descendants; may be repeated",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 3 or args.samples > 100:
        raise ConfigError("--samples must be in [3,100]")
    if args.interval_ms < 100 or args.interval_ms > 5_000:
        raise ConfigError("--interval-ms must be in [100,5000]")
    if not math.isfinite(args.max_gr3d_percent) or not (
        0.0 <= args.max_gr3d_percent <= 100.0
    ):
        raise ConfigError("--max-gr3d-percent must be finite and in [0,100]")
    if not math.isfinite(args.max_unexpected_cpu_percent) or not (
        0.0 < args.max_unexpected_cpu_percent <= 10_000.0
    ):
        raise ConfigError(
            "--max-unexpected-cpu-percent must be finite and in (0,10000]"
        )
    allow_pids = tuple(sorted(set(args.allow_pid)))
    if any(pid <= 0 for pid in allow_pids):
        raise ConfigError("--allow-pid values must be positive")
    return PreflightConfig(
        output=validate_output_path(args.output),
        samples=args.samples,
        interval_ms=args.interval_ms,
        max_gr3d_percent=args.max_gr3d_percent,
        max_unexpected_cpu_percent=args.max_unexpected_cpu_percent,
        allow_pids=allow_pids,
        force=args.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        if config.output.exists() and not config.force:
            raise ConfigError(
                f"refusing to replace {config.output}; pass --force explicitly"
            )
    except ConfigError as error:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "result": "configuration_error",
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    report, status = run_preflight(config)
    try:
        write_json_atomic(config.output, report, config.force)
    except (ConfigError, OSError) as error:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "result": "evidence_write_error",
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "accepted": report["decision"]["accepted"],
                "result": report["decision"]["result"],
                "reasons": report["decision"]["reasons"],
                "collection_errors": report["decision"]["collection_errors"],
                "evidence": str(config.output),
            },
            sort_keys=True,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
