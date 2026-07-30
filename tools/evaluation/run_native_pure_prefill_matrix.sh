#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 ELF MODEL_DIR CORPUS_DIR OUTPUT_ROOT [p512|p1k|p2k|p4k]" >&2
  exit 2
fi

server=$1
model_dir=$2
corpus_dir=$3
output_root=$4
only_bucket=${5:-}
port=${Q3X_EVAL_PORT:-18080}
server_log="${output_root}.server.log"

[[ -x "${server}" ]] || { echo "missing executable: ${server}" >&2; exit 2; }
[[ -d "${model_dir}" ]] || { echo "missing model: ${model_dir}" >&2; exit 2; }
[[ ! -e "${output_root}" ]] || {
  echo "refusing to overwrite output root: ${output_root}" >&2
  exit 2
}

declare -A corpus_sha=(
  [p512]=ef783790ade41aac3fd91e5c6e8131e2cdf49e1d79508b31aafcd5c700228143
  [p1k]=3b63431127b9376159ba96cef1f96d33ccd88bfaee391c00c0e77cc7d5b67578
  [p2k]=41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af
  [p4k]=fc01397e54ccf8f858e3854f924ddeeea9b563efd35841e6ea31166005c18767
)
buckets=(p512 p1k p2k p4k)
if [[ -n "${only_bucket}" ]]; then
  [[ -n "${corpus_sha[${only_bucket}]:-}" ]] || {
    echo "unknown bucket: ${only_bucket}" >&2
    exit 2
  }
  buckets=("${only_bucket}")
fi

for bucket in "${buckets[@]}"; do
  corpus="${corpus_dir}/q3x-sharegpt-prefill-${bucket}-5.jsonl"
  [[ -f "${corpus}" ]] || { echo "missing corpus: ${corpus}" >&2; exit 2; }
  actual=$(sha256sum "${corpus}" | awk '{print $1}')
  [[ "${actual}" == "${corpus_sha[${bucket}]}" ]] || {
    echo "corpus SHA256 mismatch for ${bucket}: ${actual}" >&2
    exit 2
  }
done

mkdir -p "${output_root}"
exec 9>/tmp/q3x-gpu-bench.lock
flock 9

server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# max_tokens=1 has no Decode step.  Decode-only sidecars and routes are
# deliberately removed from the inherited environment so this measures the
# Prefill production candidate without the 6.417-GB Gate/Up Decode duplicate.
env \
  -u Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION \
  -u Q3X_RUN_DECODE_DOWN_K512_CONSUMER_ORDER_ADMISSION \
  -u Q3X_RUN_DECODE_GATE_UP_COUPLED_FEED_ADMISSION \
  Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION=1 \
  Q3X_RUN_PREFILL_SINGLE_ARBITRARY_TILE_ADMISSION=1 \
  Q3X_RUN_NVFP4_MARLIN_PREFILL_ADMISSION=1 \
  Q3X_RUN_PREFILL_MARLIN_GATE_UP_EPILOGUE_ADMISSION=1 \
  Q3X_RUN_FP8_MARLIN_PREFILL_ADMISSION=1 \
  Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION=1 \
  Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1 \
  Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION=1 \
  Q3X_RUN_PREFILL_RESIDUAL_RMS_PROMPT_WIDE_ADMISSION=1 \
  Q3X_RUN_PREFILL_EMBEDDING_PROMPT_WIDE_ADMISSION=1 \
  Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1 \
  Q3X_FULL_ATTENTION_FLASHINFER_DIRECT=1 \
  "${server}" "${model_dir}" \
    --host 127.0.0.1 --port "${port}" \
    --model qwen3.6-27b-nvfp4 \
    --max-sequence-length 4096 --max-output-tokens 1 \
    --prefill-chunk-size 512 --projection-backend sm87 \
    >"${server_log}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 300); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}" || true
    echo "server exited before readiness; inspect ${server_log}" >&2
    exit 3
  fi
  if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready}" == 1 ]] || {
  echo "readiness timeout; inspect ${server_log}" >&2
  exit 4
}

for bucket in "${buckets[@]}"; do
  corpus="${corpus_dir}/q3x-sharegpt-prefill-${bucket}-5.jsonl"
  run_dir="${output_root}/${bucket}"
  mkdir -p "${run_dir}"
  uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
    --model qwen3.6-27b-nvfp4 --api openai \
    --url "http://127.0.0.1:${port}/v1/completions" \
    --tokenizer-path "${model_dir}" \
    --dataset line_by_line --data-source local \
    --dataset-path "${corpus}" \
    --number 4 --parallel 1 --warmup-num 1 --num-workers 1 \
    --max-tokens 1 --temperature 0 --seed 42 \
    --stream --tokenize-prompt --no-test-connection \
    --outputs-dir "${run_dir}" --name "pure-prefill-${bucket}" --no-timestamp \
    >"${run_dir}/evalscope.stdout" 2>&1
  find "${run_dir}" -name benchmark_summary.json -print \
    -exec sed -n '1,220p' {} \;
done
