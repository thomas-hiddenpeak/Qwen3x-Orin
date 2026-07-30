#!/usr/bin/env python3
"""Materialize real-prompt, one-token EvalScope Prefill length buckets.

The repository retains only hashes and source coordinates.  Token-id corpora
stay outside Git because they are reversible to source text.  Every selected
case is one complete natural ShareGPT conversation prefix; this builder never
pads, repeats, truncates, or concatenates token sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Any


SOURCE_SHA256 = (
    "a78dfae704e5ca29f18b881b0d298f4bf401093bf33d4ea993dd5728e591c565"
)
SOURCE_REVISION = "0cccec87492d6e36056c017734c112f92fbc1efd"
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_CONFIG_SHA256 = (
    "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
)
CHAT_TEMPLATE_SHA256 = (
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
)

# One warmup followed by four measured prompts in each bucket.  These are the
# first complete natural conversations in the declared token windows.
BUCKETS = {
    "p512": {
        "window": [448, 576],
        "source_lines": [0, 2, 3, 11, 19],
        "corpus_sha256": "ef783790ade41aac3fd91e5c6e8131e2cdf49e1d79508b31aafcd5c700228143",
    },
    "p1k": {
        "window": [896, 1152],
        "source_lines": [4, 16, 33, 55, 57],
        "corpus_sha256": "3b63431127b9376159ba96cef1f96d33ccd88bfaee391c00c0e77cc7d5b67578",
    },
    "p2k": {
        "window": [1792, 2304],
        "source_lines": [582, 1286, 1660, 2391, 2394],
        "corpus_sha256": "41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af",
    },
}


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify(path: pathlib.Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )


def messages_from_conversation(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        human = turn.get("human", "")
        assistant = turn.get("assistant", "")
        human = human.strip() if isinstance(human, str) else ""
        assistant = assistant.strip() if isinstance(assistant, str) else ""
        if not human:
            continue
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages


def atomic_write(path: pathlib.Path, payload: bytes, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"refusing to replace {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".q3x-")
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
        pathlib.Path(temporary).replace(path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen3.6-27b-nvfp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify(args.source, SOURCE_SHA256, "ShareGPT source")
    verify(
        args.tokenizer_dir / "tokenizer.json",
        TOKENIZER_JSON_SHA256,
        "tokenizer.json",
    )
    verify(
        args.tokenizer_dir / "tokenizer_config.json",
        TOKENIZER_CONFIG_SHA256,
        "tokenizer_config.json",
    )
    verify(
        args.tokenizer_dir / "chat_template.jinja",
        CHAT_TEMPLATE_SHA256,
        "chat_template.jinja",
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to materialize corpora") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, trust_remote_code=False
    )

    wanted = {
        line
        for bucket in BUCKETS.values()
        for line in bucket["source_lines"]
    }
    rows: dict[int, dict[str, Any]] = {}
    with args.source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line_number in wanted:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"source line {line_number} is not an object")
                rows[line_number] = value
            if len(rows) == len(wanted):
                break
    if set(rows) != wanted:
        raise RuntimeError("source does not contain every pinned line")

    matrix_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "qwen36_evalscope_pure_prefill_length_matrix",
        "source": {
            "dataset": "swift/sharegpt",
            "file": args.source.name,
            "revision": SOURCE_REVISION,
            "sha256": SOURCE_SHA256,
            "selection": "complete natural conversations; no padding, repetition, truncation, or concatenation",
        },
        "tokenizer": {
            "directory_name": args.tokenizer_dir.name,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "chat_template_jinja_sha256": CHAT_TEMPLATE_SHA256,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "request_contract": {
            "model": args.model,
            "endpoint": "/v1/completions",
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": args.seed,
            "stream": True,
            "include_usage": True,
            "parallel": 1,
            "warmup": 1,
            "measured": 4,
            "prefix_cache": False,
            "mtp": False,
        },
        "buckets": {},
    }

    for bucket_name, bucket in BUCKETS.items():
        request_lines: list[bytes] = []
        cases: list[dict[str, Any]] = []
        lower, upper = bucket["window"]
        for ordinal, source_line in enumerate(bucket["source_lines"]):
            row = rows[source_line]
            messages = messages_from_conversation(row.get("conversation"))
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(token_ids, Mapping):
                token_ids = token_ids.get("input_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise RuntimeError(f"invalid tokenization at source line {source_line}")
            if not lower <= len(token_ids) <= upper:
                raise RuntimeError(
                    f"{bucket_name} source line {source_line} has "
                    f"{len(token_ids)} tokens outside [{lower}, {upper}]"
                )
            request = {
                "model": args.model,
                "prompt": token_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "seed": args.seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            encoded = canonical_json(request)
            request_lines.append(encoded + b"\n")
            conversation_id = row.get("conversation_id")
            cases.append(
                {
                    "ordinal": ordinal,
                    "role": "warmup" if ordinal == 0 else "measured",
                    "source_line": source_line,
                    "conversation_id": conversation_id
                    if isinstance(conversation_id, str)
                    else None,
                    "prompt_tokens": len(token_ids),
                    "prompt_sha256": sha256_bytes(canonical_json(token_ids)),
                    "request_sha256": sha256_bytes(encoded),
                }
            )
        corpus = b"".join(request_lines)
        corpus_sha = sha256_bytes(corpus)
        if corpus_sha != bucket["corpus_sha256"]:
            raise RuntimeError(
                f"{bucket_name} corpus SHA256 mismatch: expected "
                f"{bucket['corpus_sha256']}, got {corpus_sha}"
            )
        corpus_name = f"q3x-sharegpt-prefill-{bucket_name}-5.jsonl"
        atomic_write(args.output_dir / corpus_name, corpus, args.force)
        matrix_manifest["buckets"][bucket_name] = {
            "token_window": bucket["window"],
            "corpus_file": corpus_name,
            "corpus_sha256": corpus_sha,
            "count": len(cases),
            "cases": cases,
        }
        bucket_manifest = {
            "schema_version": 1,
            "artifact": f"qwen36_evalscope_pure_prefill_{bucket_name}",
            "source": matrix_manifest["source"],
            "tokenizer": matrix_manifest["tokenizer"],
            "request_contract": matrix_manifest["request_contract"],
            "token_window": bucket["window"],
            "corpus_file": corpus_name,
            "corpus_sha256": corpus_sha,
            "count": len(cases),
            "cases": cases,
        }
        atomic_write(
            args.output_dir
            / f"qwen36-sharegpt-prefill-{bucket_name}-v1.manifest.json",
            canonical_json(bucket_manifest) + b"\n",
            args.force,
        )

    manifest_payload = canonical_json(matrix_manifest) + b"\n"
    atomic_write(
        args.output_dir / "qwen36-sharegpt-prefill-length-matrix-v1.manifest.json",
        manifest_payload,
        args.force,
    )
    print(json.dumps(matrix_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
