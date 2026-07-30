#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "usage: $0 ELF MODEL_DIR MANIFEST MANIFEST_SHA256 CORPUS_DIR OUTPUT_ROOT [p8k|p16k|p40k|all] [prefill1|cold16|both]" >&2
  exit 2
fi

server=$1
model_dir=$2
manifest=$3
manifest_sha256=$4
corpus_dir=$5
output_root=$6
bucket_selector=${7:-all}
phase_selector=${8:-prefill1}
base_port=${Q3X_EVAL_PORT:-18080}
readiness_timeout=${Q3X_LONG_EVAL_READINESS_TIMEOUT_SECONDS:-600}
dry_run=${Q3X_LONG_EVAL_DRY_RUN:-0}

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
validator="${repository}/tools/evaluation/validate_long_prefill_manifest.py"
attention_gate=Q3X_RUN_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION
flashinfer_gate=Q3X_FULL_ATTENTION_FLASHINFER_DIRECT

positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

[[ -x "${server}" ]] || { echo "missing executable: ${server}" >&2; exit 2; }
[[ -d "${model_dir}" ]] || { echo "missing model: ${model_dir}" >&2; exit 2; }
[[ -f "${manifest}" ]] || { echo "missing manifest: ${manifest}" >&2; exit 2; }
[[ -d "${corpus_dir}" ]] || { echo "missing corpus directory: ${corpus_dir}" >&2; exit 2; }
[[ -x "${validator}" ]] || { echo "missing manifest validator: ${validator}" >&2; exit 2; }
[[ ! -e "${output_root}" ]] || {
  echo "refusing to overwrite output root: ${output_root}" >&2
  exit 2
}
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "Q3X_LONG_EVAL_DRY_RUN must be 0 or 1" >&2
  exit 2
}
positive_integer "${base_port}" && ((base_port <= 65535)) || {
  echo "Q3X_EVAL_PORT must be in [1,65535]" >&2
  exit 2
}
positive_integer "${readiness_timeout}" || {
  echo "Q3X_LONG_EVAL_READINESS_TIMEOUT_SECONDS must be positive" >&2
  exit 2
}

case "${bucket_selector}" in
  p8k|p16k|p40k|all) ;;
  *) echo "unknown bucket selector: ${bucket_selector}" >&2; exit 2 ;;
esac
case "${phase_selector}" in
  prefill1|cold16|both) ;;
  *) echo "unknown phase selector: ${phase_selector}" >&2; exit 2 ;;
esac
if [[ "${phase_selector}" != prefill1 &&
      "${bucket_selector}" != p40k &&
      "${bucket_selector}" != all ]]; then
  echo "cold16 is defined only for p40k" >&2
  exit 2
fi

if ! strings -a "${server}" | grep -F "${attention_gate}" >/dev/null; then
  echo "server does not contain the long-context attention BUILD admission" >&2
  exit 2
fi
if ! strings -a "${server}" | grep -F "${flashinfer_gate}" >/dev/null; then
  echo "server does not contain the FlashInfer direct attention BUILD admission" >&2
  exit 2
fi
if [[ "${dry_run}" == 0 ]] && ! readelf -h "${server}" >/dev/null 2>&1; then
  echo "performance execution requires an ELF server binary" >&2
  exit 2
fi

declare -a selected_runs=()
append_bucket_runs() {
  local bucket=$1
  case "${phase_selector}" in
    prefill1) selected_runs+=("${bucket}:prefill1") ;;
    cold16)
      if [[ "${bucket}" == p40k ]]; then
        selected_runs+=("p40k:cold16")
      fi
      ;;
    both)
      selected_runs+=("${bucket}:prefill1")
      if [[ "${bucket}" == p40k ]]; then
        selected_runs+=("p40k:cold16")
      fi
      ;;
  esac
}
if [[ "${bucket_selector}" == all ]]; then
  append_bucket_runs p8k
  append_bucket_runs p16k
  append_bucket_runs p40k
else
  append_bucket_runs "${bucket_selector}"
fi
[[ ${#selected_runs[@]} -gt 0 ]] || {
  echo "the selectors produced no valid runs" >&2
  exit 2
}
if ((base_port + ${#selected_runs[@]} - 1 > 65535)); then
  echo "Q3X_EVAL_PORT leaves too few consecutive ports for this matrix" >&2
  exit 2
fi

scratch_dir=$(mktemp -d /tmp/q3x-long-eval-contract.XXXXXX)
server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -rf -- "${scratch_dir}"
}
trap cleanup EXIT INT TERM

declare -a contract_files=()
for index in "${!selected_runs[@]}"; do
  IFS=: read -r bucket phase <<<"${selected_runs[index]}"
  contract_file="${scratch_dir}/contract-${index}.nul"
  python3 "${validator}" \
    --manifest "${manifest}" --manifest-sha256 "${manifest_sha256}" \
    --tokenizer-dir "${model_dir}" --corpus-dir "${corpus_dir}" \
    --bucket "${bucket}" --phase "${phase}" --format nul \
    >"${contract_file}"
  contract_files+=("${contract_file}")
done

server_sha256=$(sha256sum "${server}" | awk '{print $1}')
manifest_actual_sha256=$(sha256sum "${manifest}" | awk '{print $1}')
printf 'long_context_matrix server=%q server_sha256=%s manifest=%q manifest_sha256=%s runs=%s dry_run=%s\n' \
  "${server}" "${server_sha256}" "${manifest}" "${manifest_actual_sha256}" \
  "${#selected_runs[@]}" "${dry_run}"
printf 'long_context_attention build_marker=present run_admission=%s=1 flashinfer_direct=%s=1\n' \
  "${attention_gate}" "${flashinfer_gate}"

for index in "${!selected_runs[@]}"; do
  mapfile -d '' -t contract <"${contract_files[index]}"
  [[ ${#contract[@]} -eq 12 ]] || {
    echo "internal validator output contract changed" >&2
    exit 5
  }
  bucket=${contract[0]}
  phase=${contract[1]}
  corpus=${contract[2]}
  corpus_sha256=${contract[3]}
  max_tokens=${contract[4]}
  number=${contract[5]}
  warmup=${contract[6]}
  max_sequence_length=${contract[7]}
  request_max_arena_bytes=${contract[8]}
  planned_arena_bytes=${contract[9]}
  token_window_min=${contract[10]}
  token_window_max=${contract[11]}
  port=$((base_port + index))

  case "${bucket}" in
    p8k) default_read_timeout=300; default_total_timeout=600 ;;
    p16k) default_read_timeout=600; default_total_timeout=1200 ;;
    p40k) default_read_timeout=1800; default_total_timeout=3600 ;;
    *) echo "internal unknown bucket: ${bucket}" >&2; exit 5 ;;
  esac
  connect_timeout=${Q3X_LONG_EVAL_CONNECT_TIMEOUT_SECONDS:-30}
  read_timeout=${Q3X_LONG_EVAL_READ_TIMEOUT_SECONDS:-${default_read_timeout}}
  total_timeout=${Q3X_LONG_EVAL_TOTAL_TIMEOUT_SECONDS:-${default_total_timeout}}
  positive_integer "${connect_timeout}" || {
    echo "Q3X_LONG_EVAL_CONNECT_TIMEOUT_SECONDS must be positive" >&2
    exit 2
  }
  positive_integer "${read_timeout}" || {
    echo "Q3X_LONG_EVAL_READ_TIMEOUT_SECONDS must be positive" >&2
    exit 2
  }
  positive_integer "${total_timeout}" || {
    echo "Q3X_LONG_EVAL_TOTAL_TIMEOUT_SECONDS must be positive" >&2
    exit 2
  }
  ((total_timeout >= read_timeout)) || {
    echo "total client timeout must be at least the read timeout" >&2
    exit 2
  }

  readiness_route="http://127.0.0.1:${port}/healthz"
  completion_route="http://127.0.0.1:${port}/v1/completions"
  server_args=(
    "${server}" "${model_dir}"
    --host 127.0.0.1 --port "${port}"
    --model qwen3.6-27b-nvfp4
    --max-sequence-length "${max_sequence_length}"
    --max-output-tokens "${max_tokens}"
    --request-max-arena-bytes "${request_max_arena_bytes}"
    --prefill-chunk-size 512 --projection-backend sm87
  )
  printf 'long_context_run bucket=%s phase=%s token_window=%s..%s corpus=%q corpus_sha256=%s planned_arena_bytes=%s\n' \
    "${bucket}" "${phase}" "${token_window_min}" "${token_window_max}" \
    "${corpus}" "${corpus_sha256}" "${planned_arena_bytes}"
  printf 'server_startup_args'
  printf ' %q' "${server_args[@]}"
  printf '\n'
  printf 'client_args connect_timeout=%s read_timeout=%s total_timeout=%s max_tokens=%s number=%s warmup=%s\n' \
    "${connect_timeout}" "${read_timeout}" "${total_timeout}" \
    "${max_tokens}" "${number}" "${warmup}"
  printf 'server_readiness_route=%s completion_route=%s\n' \
    "${readiness_route}" "${completion_route}"
done

if [[ "${dry_run}" == 1 ]]; then
  echo "dry_run_complete=1 performance_evidence=0"
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v uvx >/dev/null || { echo "uvx is required" >&2; exit 2; }
mkdir -p "${output_root}"
exec 9>/tmp/q3x-gpu-bench.lock
flock 9

for index in "${!selected_runs[@]}"; do
  mapfile -d '' -t contract <"${contract_files[index]}"
  bucket=${contract[0]}
  phase=${contract[1]}
  corpus=${contract[2]}
  max_tokens=${contract[4]}
  number=${contract[5]}
  warmup=${contract[6]}
  max_sequence_length=${contract[7]}
  request_max_arena_bytes=${contract[8]}
  port=$((base_port + index))
  case "${bucket}" in
    p8k) default_read_timeout=300; default_total_timeout=600 ;;
    p16k) default_read_timeout=600; default_total_timeout=1200 ;;
    p40k) default_read_timeout=1800; default_total_timeout=3600 ;;
  esac
  connect_timeout=${Q3X_LONG_EVAL_CONNECT_TIMEOUT_SECONDS:-30}
  read_timeout=${Q3X_LONG_EVAL_READ_TIMEOUT_SECONDS:-${default_read_timeout}}
  total_timeout=${Q3X_LONG_EVAL_TOTAL_TIMEOUT_SECONDS:-${default_total_timeout}}
  readiness_route="http://127.0.0.1:${port}/healthz"
  completion_route="http://127.0.0.1:${port}/v1/completions"
  run_dir="${output_root}/${bucket}/${phase}"
  server_log="${run_dir}/server.log"
  mkdir -p "${run_dir}"

  env \
    -u Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION \
    -u Q3X_RUN_DECODE_DOWN_K512_CONSUMER_ORDER_ADMISSION \
    -u Q3X_RUN_DECODE_GATE_UP_COUPLED_FEED_ADMISSION \
    -u Q3X_RUN_LONG_PREFILL_LAYER_MAJOR_ADMISSION \
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
    Q3X_RUN_FULL_ATTENTION_LONG_CONTEXT_GROUP_Q64_ADMISSION=1 \
    Q3X_FULL_ATTENTION_FLASHINFER_DIRECT=1 \
    "${server}" "${model_dir}" \
      --host 127.0.0.1 --port "${port}" \
      --model qwen3.6-27b-nvfp4 \
      --max-sequence-length "${max_sequence_length}" \
      --max-output-tokens "${max_tokens}" \
      --request-max-arena-bytes "${request_max_arena_bytes}" \
      --prefill-chunk-size 512 --projection-backend sm87 \
      >"${server_log}" 2>&1 &
  server_pid=$!

  ready=0
  for ((attempt = 0; attempt < readiness_timeout; ++attempt)); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}" || true
      server_pid=
      echo "server exited before readiness; inspect ${server_log}" >&2
      exit 3
    fi
    if curl -fsS --max-time 5 "${readiness_route}" \
        >"${run_dir}/readiness.json" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "${ready}" == 1 ]] || {
    echo "readiness timeout at ${readiness_route}; inspect ${server_log}" >&2
    exit 4
  }
  if ! grep -Eq \
      "max_sequence_length=${max_sequence_length} maximum_output_tokens=${max_tokens} request_max_arena_bytes=${request_max_arena_bytes} prefill_chunk_size=512 readiness_route=/healthz" \
      "${server_log}"; then
    echo "server readiness did not echo the requested capacity contract" >&2
    exit 5
  fi
  if ! grep -Eq \
      'long_context_group_q64_compiled=1 .*long_context_group_q64_run_requested=1 .*long_context_group_q64_probe_selected=1' \
      "${server_log}"; then
    echo "server readiness did not prove BUILD+RUN long attention admission" >&2
    exit 5
  fi
  if ! grep -Eq \
      'long_prefill_run_requested=0 .*long_prefill_hidden_capacity=0' \
      "${server_log}"; then
    echo "server retained rejected layer-major hidden slabs in this run" >&2
    exit 5
  fi

  uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
    --model qwen3.6-27b-nvfp4 --api openai \
    --url "${completion_route}" \
    --tokenizer-path "${model_dir}" \
    --dataset line_by_line --data-source local \
    --dataset-path "${corpus}" \
    --number "${number}" --parallel 1 --warmup-num "${warmup}" --num-workers 1 \
    --max-tokens "${max_tokens}" --temperature 0 --seed 42 \
    --connect-timeout "${connect_timeout}" \
    --read-timeout "${read_timeout}" --total-timeout "${total_timeout}" \
    --stream --tokenize-prompt --no-test-connection \
    --outputs-dir "${run_dir}" --name "long-context-${bucket}-${phase}" \
    --no-timestamp >"${run_dir}/evalscope.stdout" 2>&1

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  server_pid=
  find "${run_dir}" -name benchmark_summary.json -print \
    -exec sed -n '1,220p' {} \;
done
