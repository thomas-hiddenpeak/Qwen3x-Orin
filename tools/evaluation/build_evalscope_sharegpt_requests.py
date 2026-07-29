#!/usr/bin/env python3
"""Materialize a hash-locked false-thinking ShareGPT EvalScope corpus.

The generated JSONL contains complete OpenAI /v1/completions request bodies
with exact token-id prompts. It is intentionally an ephemeral benchmark input:
token ids are reversible enough that the repository stores only the builder,
source identities, and a non-reversible manifest of per-request hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Any, Iterable


PINNED_SOURCE_SHA256 = (
    "a78dfae704e5ca29f18b881b0d298f4bf401093bf33d4ea993dd5728e591c565"
)
PINNED_SOURCE_REVISION = "0cccec87492d6e36056c017734c112f92fbc1efd"
PINNED_TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
PINNED_TOKENIZER_CONFIG_SHA256 = (
    "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
)
PINNED_CHAT_TEMPLATE_SHA256 = (
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
)
PINNED_CORPUS_SHA256 = (
    "bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989"
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


def messages_from_conversation(conversation: Any) -> list[dict[str, str]]:
    if not isinstance(conversation, list):
        return []
    messages: list[dict[str, str]] = []
    for turn in conversation:
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


def iter_source_rows(
    source: pathlib.Path, source_line_offset: int
) -> Iterable[tuple[int, dict[str, Any]]]:
    with source.open("r", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream):
            if source_line < source_line_offset:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid source JSON at line {source_line}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"source line {source_line} is not a JSON object"
                )
            yield source_line, row


def verify_file(path: pathlib.Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact false-thinking ShareGPT token-id requests for "
            "EvalScope line_by_line"
        )
    )
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--source-line-offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=33)
    parser.add_argument("--model", default="qwen3.6-27b-nvfp4")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-sha256", default=PINNED_SOURCE_SHA256
    )
    parser.add_argument(
        "--source-revision", default=PINNED_SOURCE_REVISION
    )
    parser.add_argument(
        "--tokenizer-json-sha256", default=PINNED_TOKENIZER_JSON_SHA256
    )
    parser.add_argument(
        "--tokenizer-config-sha256",
        default=PINNED_TOKENIZER_CONFIG_SHA256,
    )
    parser.add_argument(
        "--chat-template-sha256", default=PINNED_CHAT_TEMPLATE_SHA256
    )
    parser.add_argument(
        "--expected-corpus-sha256",
        default=PINNED_CORPUS_SHA256,
        help="Fail closed unless the complete generated JSONL has this SHA256",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing outputs"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_line_offset < 0:
        raise RuntimeError("--source-line-offset must be non-negative")
    if args.count <= 0 or args.max_tokens <= 0:
        raise RuntimeError("--count and --max-tokens must be positive")
    if args.seed < -(1 << 63) or args.seed > (1 << 63) - 1:
        raise RuntimeError("--seed must fit a signed 64-bit OpenAI integer")
    if not args.model:
        raise RuntimeError("--model must be non-empty")
    protected_inputs = {
        args.source.resolve(),
        (args.tokenizer_dir / "tokenizer.json").resolve(),
        (args.tokenizer_dir / "tokenizer_config.json").resolve(),
        (args.tokenizer_dir / "chat_template.jinja").resolve(),
    }
    destinations = {args.output.resolve(), args.manifest.resolve()}
    if len(destinations) != 2:
        raise RuntimeError("--output and --manifest must be different files")
    if destinations & protected_inputs:
        raise RuntimeError("outputs must not replace source or tokenizer files")
    for destination in (args.output, args.manifest):
        if destination.exists() and not args.force:
            raise RuntimeError(
                f"refusing to replace {destination}; pass --force explicitly"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

    source_sha = verify_file(
        args.source, args.source_sha256, "ShareGPT source"
    )
    tokenizer_json = args.tokenizer_dir / "tokenizer.json"
    tokenizer_config = args.tokenizer_dir / "tokenizer_config.json"
    chat_template = args.tokenizer_dir / "chat_template.jinja"
    tokenizer_json_sha = verify_file(
        tokenizer_json, args.tokenizer_json_sha256, "tokenizer.json"
    )
    tokenizer_config_sha = verify_file(
        tokenizer_config,
        args.tokenizer_config_sha256,
        "tokenizer_config.json",
    )
    chat_template_sha = verify_file(
        chat_template, args.chat_template_sha256, "chat_template.jinja"
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required only to materialize the pinned corpus"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )

    request_lines: list[bytes] = []
    cases: list[dict[str, Any]] = []
    for source_line, row in iter_source_rows(
        args.source, args.source_line_offset
    ):
        messages = messages_from_conversation(row.get("conversation"))
        if not messages:
            continue
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(token_ids, Mapping) and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(
                f"tokenizer returned no flat token list for source line {source_line}"
            )
        if any(
            not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        ):
            raise RuntimeError(
                f"tokenizer returned an invalid id for source line {source_line}"
            )

        request = {
            "model": args.model,
            "prompt": token_ids,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request_bytes = canonical_json(request)
        request_lines.append(request_bytes + b"\n")
        conversation_id = row.get("conversation_id")
        cases.append(
            {
                "ordinal": len(cases),
                "source_line": source_line,
                "conversation_id": (
                    conversation_id if isinstance(conversation_id, str) else None
                ),
                "prompt_tokens": len(token_ids),
                "prompt_sha256": sha256_bytes(canonical_json(token_ids)),
                "request_sha256": sha256_bytes(request_bytes),
            }
        )
        if len(cases) == args.count:
            break

    if len(cases) != args.count:
        raise RuntimeError(
            f"source exhausted after {len(cases)} usable rows; expected {args.count}"
        )

    corpus = b"".join(request_lines)
    corpus_sha = sha256_bytes(corpus)
    if (
        args.expected_corpus_sha256 is not None
        and corpus_sha != args.expected_corpus_sha256
    ):
        raise RuntimeError(
            "corpus SHA256 mismatch: expected "
            f"{args.expected_corpus_sha256}, got {corpus_sha}"
        )

    manifest = {
        "schema_version": 1,
        "artifact": "qwen36_evalscope_false_thinking_sharegpt_requests",
        "source": {
            "dataset": "swift/sharegpt",
            "file": args.source.name,
            "revision": args.source_revision,
            "sha256": source_sha,
            "source_line_offset": args.source_line_offset,
        },
        "tokenizer": {
            "directory_name": args.tokenizer_dir.name,
            "tokenizer_json_sha256": tokenizer_json_sha,
            "tokenizer_config_sha256": tokenizer_config_sha,
            "chat_template_jinja_sha256": chat_template_sha,
            "chat_template": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
        "request_contract": {
            "model": args.model,
            "endpoint": "/v1/completions",
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "stream": True,
            "include_usage": True,
        },
        "count": len(cases),
        "corpus_file": args.output.name,
        "corpus_sha256": corpus_sha,
        "cases": cases,
    }
    manifest_bytes = canonical_json(manifest) + b"\n"

    def atomic_write(destination: pathlib.Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    atomic_write(args.output, corpus)
    atomic_write(args.manifest, manifest_bytes)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                "count": len(cases),
                "corpus_sha256": corpus_sha,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
