# Pinned Qwen 3.6 tokenizer

`q3x::text` is the pure C++17 tokenizer path for the exact
`tokenizer.json` shipped by
`nvidia/Qwen3.6-27B-NVFP4@0893e1606ff3d5f97a441f405d5fc541a6bdf404`.
It uses ICU 74 for NFC normalization and the pinned Unicode regular
expression, then applies the GPT-2 byte alphabet and ranked BPE merges.

The loader fails closed unless all of the following agree:

- raw SHA-256
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`;
- NFC, Split/Regex, ByteLevel post-processor, and ByteLevel decoder schema;
- 248,044 contiguous base-vocabulary ids and 247,587 valid merge rules;
- the exact 26 `tokenizer.json` added-token records and ids 248044–248069.

The Transformers `from_pretrained` configuration adds seven audio/TTS tokens
from `tokenizer_config.json`, making `len(tokenizer) == 248077`. Those tokens
are not present in the pinned `tokenizer.json` and are intentionally outside
this text-only loader. The inference chat path only uses ids through 248069.

## Chat API scope

`Tokenizer::format_qwen36_chat` implements a named, fail-closed subset of the
pinned Qwen template. It accepts an optional leading text-only `system`
message followed by alternating `user` and plain `assistant` messages. It
supports `add_generation_prompt` and both values of `enable_thinking`, including
the exact `enable_thinking=false` suffix:

```text
<|im_start|>assistant
<think>

</think>

```

It is not a Jinja interpreter. Tools, tool responses, multimodal message
objects, unexpected roles, and assistant thinking-history markup return a
structured `kUnsupportedChat` error.

## Resource and error contract

Loading and runtime operations accept `TokenizerLimits`. Tokenizer bytes,
vocabulary entries, merges, added tokens, input bytes, output token count, and
chat message count are bounded, with additional absolute ceilings. Public load,
encode, decode, and chat entry points catch allocation and implementation
exceptions and return a `TokenizerError` instead of allowing exceptions from
untrusted input to cross the API boundary.

The differential test uses `tokenizers.Tokenizer.from_file(tokenizer.json)` as
the oracle for the raw artifact. This distinction matters: Transformers 5.12.1
reconstructs a Qwen2 fast backend from `tokenizer_config.json` and changes the
raw regex's combining-mark treatment. The C++ implementation deliberately
matches the pinned `tokenizer.json`, while all cases in the repository's
independent fixture are also identical under the Transformers path.
