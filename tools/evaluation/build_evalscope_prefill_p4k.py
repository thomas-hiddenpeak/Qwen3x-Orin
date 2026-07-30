#!/usr/bin/env python3
"""Materialize the natural ShareGPT P4K one-token EvalScope bucket.

The pinned public source is streamed and only the five declared complete
conversation prefixes are retained.  No source text or reversible token ids
are checked into Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Any


SOURCE_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
SOURCE_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
SOURCE_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILE}"
)
SOURCE_LFS_SHA256 = (
    "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
)
SOURCE_BYTES = 672_837_942
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
TOKENIZER_CONFIG_SHA256 = (
    "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
)
CHAT_TEMPLATE_SHA256 = (
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
)
SELECTED = {
    2382: "5qjPXGl_18",
    2610: "Ixh1Px9_14",
    3068: "JwplmNR_4",
    4475: "ulejue7_0",
    6154: "8DctRHd_0",
}
TOKEN_WINDOW = [3584, 4032]
CORPUS_SHA256 = (
    "fc01397e54ccf8f858e3854f924ddeeea9b563efd35841e6ea31166005c18767"
)


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
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"{label} is absent or does not match its SHA256")


def messages_from_row(row: Any) -> list[dict[str, str]]:
    if not isinstance(row, dict) or not isinstance(row.get("conversations"), list):
        return []
    messages: list[dict[str, str]] = []
    for turn in row["conversations"]:
        if not isinstance(turn, dict):
            continue
        source_role = turn.get("from")
        value = turn.get("value")
        value = value.strip() if isinstance(value, str) else ""
        if not value:
            continue
        if source_role == "human":
            messages.append({"role": "user", "content": value})
        elif source_role == "gpt":
            messages.append({"role": "assistant", "content": value})
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    return messages if messages and messages[-1]["role"] == "user" else []


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
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen3.6-27b-nvfp4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify(args.tokenizer_dir / "tokenizer.json", TOKENIZER_JSON_SHA256,
           "tokenizer.json")
    verify(args.tokenizer_dir / "tokenizer_config.json",
           TOKENIZER_CONFIG_SHA256, "tokenizer_config.json")
    verify(args.tokenizer_dir / "chat_template.jinja", CHAT_TEMPLATE_SHA256,
           "chat_template.jinja")
    try:
        import ijson
        import requests
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("ijson, requests, and transformers are required") from error

    head = requests.head(SOURCE_URL, allow_redirects=False, timeout=30)
    head.raise_for_status()
    linked_etag = head.headers.get("x-linked-etag", "").strip('"')
    linked_size = head.headers.get("x-linked-size", "")
    revision = head.headers.get("x-repo-commit", "")
    if (linked_etag != SOURCE_LFS_SHA256 or linked_size != str(SOURCE_BYTES)
            or revision != SOURCE_REVISION):
        raise RuntimeError("pinned ShareGPT source identity changed")

    response = requests.get(SOURCE_URL, stream=True, timeout=(30, 180))
    response.raise_for_status()
    response.raw.decode_content = True
    selected_rows: dict[int, dict[str, Any]] = {}
    try:
        for index, row in enumerate(ijson.items(response.raw, "item")):
            if index in SELECTED:
                if not isinstance(row, dict) or row.get("id") != SELECTED[index]:
                    raise RuntimeError(f"source identity mismatch at index {index}")
                selected_rows[index] = row
            if len(selected_rows) == len(SELECTED):
                break
    finally:
        response.close()
    if set(selected_rows) != set(SELECTED):
        raise RuntimeError("stream ended before every selected P4K row")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    request_lines: list[bytes] = []
    cases: list[dict[str, Any]] = []
    for ordinal, source_index in enumerate(SELECTED):
        row = selected_rows[source_index]
        messages = messages_from_row(row)
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get("input_ids")
        if not isinstance(token_ids, list) or not (
            TOKEN_WINDOW[0] <= len(token_ids) <= TOKEN_WINDOW[1]
        ):
            raise RuntimeError(f"selected P4K row {source_index} token mismatch")
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
        cases.append({
            "ordinal": ordinal,
            "role": "warmup" if ordinal == 0 else "measured",
            "source_index": source_index,
            "conversation_id": row["id"],
            "source_row_sha256": sha256_bytes(canonical_json(row)),
            "prompt_tokens": len(token_ids),
            "prompt_sha256": sha256_bytes(canonical_json(token_ids)),
            "request_sha256": sha256_bytes(encoded),
        })

    corpus = b"".join(request_lines)
    corpus_sha = sha256_bytes(corpus)
    if corpus_sha != CORPUS_SHA256:
        raise RuntimeError(
            f"P4K corpus SHA256 mismatch: expected {CORPUS_SHA256}, "
            f"got {corpus_sha}"
        )
    corpus_name = "q3x-sharegpt-prefill-p4k-5.jsonl"
    manifest = {
        "schema_version": 1,
        "artifact": "qwen36_evalscope_pure_prefill_p4k",
        "source": {
            "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "file": SOURCE_FILE,
            "revision": SOURCE_REVISION,
            "lfs_sha256": SOURCE_LFS_SHA256,
            "bytes": SOURCE_BYTES,
            "selection": "complete natural conversations; distinct base conversation IDs; no padding, repetition, truncation, or concatenation",
        },
        "tokenizer": {
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
        "token_window": TOKEN_WINDOW,
        "corpus_file": corpus_name,
        "corpus_sha256": corpus_sha,
        "count": len(cases),
        "cases": cases,
    }
    atomic_write(args.output_dir / corpus_name, corpus, args.force)
    atomic_write(
        args.output_dir / "qwen36-sharegpt-prefill-p4k-v1.manifest.json",
        canonical_json(manifest) + b"\n", args.force
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
