#!/usr/bin/env python3
"""Audit BF16 state with size-invariant metrics; never grants qualification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(reference, candidate):
    a = np.asarray(reference, dtype=np.float64).ravel()
    b = np.asarray(candidate, dtype=np.float64).ravel()
    if a.size == 0 or a.shape != b.shape:
        raise ValueError("empty or unequal state shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite state")
    delta = a - b
    aa, bb, dd = float(a @ a), float(b @ b), float(delta @ delta)
    # RMSE(error) / RMS(reference) == L2(error) / L2(reference).
    # RMSE(error) / L2(reference) incorrectly shrinks by sqrt(element count).
    relative = np.sqrt(dd / aa) if aa else (0.0 if dd == 0 else None)
    return {
        "elements": int(a.size),
        "value_equal": bool(np.array_equal(a, b)),
        "rmse": float(np.sqrt(dd / a.size)),
        "reference_rms": float(np.sqrt(aa / a.size)),
        "relative_l2": None if relative is None else float(relative),
        "cosine": float((a @ b) / np.sqrt(aa * bb)) if aa and bb else None,
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "incorrect_rmse_over_l2": (
            float(np.sqrt(dd / a.size) / np.sqrt(aa)) if aa else None
        ),
    }


def bf16(words):
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_manifests(a, b, historical=False):
    if (a["prompt_ids_u32le_sha256"] != b["prompt_ids_u32le_sha256"]
            or a["prompt_tokens"] != b["prompt_tokens"]
            or a["state_files"]["kv_sample_positions"] != b["state_files"]["kv_sample_positions"]):
        raise ValueError("different input tokens or capture coverage")
    if not historical and any(
        m.get("capture_boundary") != "prefill_commit_O1_no_decode"
        or len(m["generated_ids"]) != 1
        or len(m["steps"]) != 1
        or m["steps"][0]["sequence"] != m["prompt_tokens"]
        or m["state_files"]["kv_sample_positions"] != m["prompt_tokens"]
        for m in (a, b)
    ):
        raise ValueError("requires O1 Prefill boundary and complete KV captures")


def audit(baseline, candidate, historical=False):
    manifests = [json.loads(p.read_text()) for p in (baseline, candidate)]
    a, b = manifests
    validate_manifests(a, b, historical)
    # Pinned Qwen3.6-27B capture layout, matching request_state.h and the C++
    # capture. Two equally truncated files must never count as full state.
    expected_bytes = {"gdn": 75497472, "conv": 2949120, "logits": 248320 * 2,
                      "kv": 32 * a["state_files"]["kv_sample_positions"] * 4 * 256 * 2}
    result = {
        "scope": "historical diagnostic" if historical else "matched Prefill boundary diagnostic",
        "formula": "sqrt(sum((candidate-reference)^2)/sum(reference^2))",
        "qualification_granted": False,
        "manifests": [{"path": str(p), "sha256": digest(p)} for p in (baseline, candidate)],
        "regions": {},
    }
    for role, blocks in (("gdn", 48), ("conv", 48), ("kv", 32), ("logits", 1)):
        paths = [p.parent / m["state_files"][role]
                 for p, m in zip((baseline, candidate), manifests)
                 if m["state_files"].get(role)]
        if len(paths) != 2:
            if role == "logits" and historical:
                continue
            raise ValueError("missing " + role)
        if any(p.stat().st_size != expected_bytes[role] for p in paths):
            raise ValueError("incomplete or incompatible " + role + " payload")
        arrays = [np.memmap(p, dtype="<u2", mode="r") for p in paths]
        if arrays[0].shape != arrays[1].shape or arrays[0].size % blocks:
            raise ValueError("invalid " + role + " sizes")
        rows = []
        for slot, (x, y) in enumerate(zip(*(v.reshape(blocks, -1) for v in arrays))):
            row = metrics(bf16(x), bf16(y))
            row["block"] = slot
            row["bitwise_equal"] = bool(np.array_equal(x, y))
            if role == "kv":
                row.update(layer=4 * (slot // 2) + 3, kind="key" if slot % 2 == 0 else "value")
            elif role in ("gdn", "conv"):
                row["layer"] = 4 * (slot // 3) + slot % 3
            if role == "logits":
                x, y = bf16(x).astype(np.float64), bf16(y).astype(np.float64)
                logp = x - np.max(x); logp -= np.log(np.exp(logp).sum())
                logq = y - np.max(y); logq -= np.log(np.exp(logq).sum())
                row.update(kl_reference_to_candidate=float(np.exp(logp) @ (logp - logq)),
                           reference_argmax=int(np.argmax(x)), candidate_argmax=int(np.argmax(y)))
                if not historical and (
                    row["reference_argmax"] != a["generated_ids"][0]
                    or row["candidate_argmax"] != b["generated_ids"][0]
                ):
                    raise ValueError("logits do not reproduce the committed first token")
            rows.append(row)
        result["regions"][role] = {
            "files": [{"path": str(p), "sha256": digest(p)} for p in paths],
            "blocks": rows,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.baseline, args.candidate, args.historical)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
