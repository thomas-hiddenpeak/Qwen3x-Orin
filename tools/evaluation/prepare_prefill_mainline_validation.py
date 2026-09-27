#!/usr/bin/env python3
"""Prepare pinned requests and audit admission; never grants promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = (1, 19, 43, 44, 63, 64, 65, 511, 512, 513, 1089,
            7999, 8000, 8001, 8191, 8192, 8193, 16000, 32000,
            39999, 40000, 40001, 44094, 44095)
OUTPUTS = (1, 16, 256, 4096)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_prompt(value):
    ids = value.get("prompt") if isinstance(value, dict) else None
    if not isinstance(ids, list) or not ids or any(
        type(token) is not int or not 0 <= token < 248320 for token in ids
    ):
        raise ValueError("source must contain nonempty pinned-model token IDs")
    return ids


def prepare(source, probe, output, long_source=None):
    output = output.resolve()
    if not output.is_relative_to((ROOT / ".q3x-work").resolve()):
        raise ValueError("generated requests must stay inside .q3x-work")
    raw = source.read_bytes()
    ids = validate_prompt(json.loads(raw))
    long_raw = long_source.read_bytes() if long_source else None
    long_ids = validate_prompt(json.loads(long_raw)) if long_raw else []
    audit = json.loads(subprocess.check_output(
        [str(probe.resolve()), "--service-coverage"], text=True))
    if audit.get("scope") != "host-admission-only":
        raise ValueError("unexpected coverage probe schema")
    expected = {(p, o) for p in CONTEXTS for o in OUTPUTS if p + o - 1 <= 44095}
    observed = {(c["prompt_tokens"], c["output_tokens"]) for c in audit["cases"]
                if c["surface"] == "token_ids" and c["stream"] and c["include_usage"]}
    if expected != observed:
        raise ValueError("probe omitted or changed a required capacity case")
    # Refuse to overwrite a frozen request set from an earlier invocation.
    output.mkdir(parents=True, exist_ok=False)
    requests, missing = [], []
    for p, o in sorted(expected):
        name = f"p{p}-o{o}"
        selected_ids = ids if p <= len(ids) else long_ids
        if p > len(selected_ids):
            missing.append({"case": name, "reason": "real prompt source too short"})
            continue
        body = {"model": "qwen3.6-27b-nvfp4", "prompt": selected_ids[:p],
                "max_tokens": o, "temperature": 0, "seed": 42,
                "stream": True, "stream_options": {"include_usage": True}}
        payload = encode(body)
        (output / f"{name}.json").write_bytes(payload)
        requests.append({"case": name, "path": f"{name}.json",
                         "body_sha256": sha(payload),
                         "source": "primary" if selected_ids is ids else "long",
                         "prompt_ids_u32le_sha256": sha(struct.pack(f"<{p}I", *selected_ids[:p]))})
    rejected = [c for c in audit["cases"] if not c["admitted"]]
    result = {"schema_version": 1, "scope": "preparation-not-qualification",
              "source_sha256": sha(raw), "source_tokens": len(ids),
              "long_source_sha256": sha(long_raw) if long_raw else None,
              "long_source_tokens": len(long_ids),
              "probe_sha256": sha(probe.read_bytes()),
              "source_revision": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "promotion_qualified": False,
              "admission_ready": not rejected,
              "blocked_admission_cases": rejected,
              "missing_real_payloads": missing, "requests": requests,
              "pending_surfaces": ["raw text and chat with tokenizer-bound counts",
                                   "nonstreaming", "streaming without usage"],
              "performance_subset": ["p1089-o256", "p8192-o256", "p40000-o256"],
              "note": "Prefixes are performance/state fixtures, not capability answers; "
                      "never pad/repeat tokens to fill missing long inputs. "
                      "O1 has no subsequent-Decode throughput."}
    (output / "coverage.json").write_bytes(encode(audit))
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-request", type=Path, required=True)
    parser.add_argument("--coverage-probe", type=Path, required=True)
    parser.add_argument("--long-prompt-request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.prompt_request, args.coverage_probe, args.output,
                     args.long_prompt_request)
    print(json.dumps({"prepared_requests": len(result["requests"]),
                      "blocked_admission_cases": len(result["blocked_admission_cases"]),
                      "missing_real_payloads": len(result["missing_real_payloads"]),
                      "promotion_qualified": False}))
    # Successful preparation is not a successful service-readiness gate.
    return 3 if result["blocked_admission_cases"] or result["missing_real_payloads"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
