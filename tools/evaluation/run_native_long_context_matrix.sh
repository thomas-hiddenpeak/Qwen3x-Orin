#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_native_long_context_matrix.sh \
  --prefill-a4-payload FILE --prefill-a4-policy FILE \
  --prefill-a4-receipt FILE [--mode exact|native-gdn] [--dry-run] \
  ELF MODEL_DIR MANIFEST MANIFEST_SHA256 CORPUS_DIR OUTPUT_ROOT \
  [p8k|p16k|p40k|all] [prefill1|cold16|both]
EOF
}

mode=exact
mode_seen=0
dry_run=${Q3X_LONG_EVAL_DRY_RUN:-0}
prefill_a4_payload=
prefill_a4_policy=
prefill_a4_receipt=
declare -a positional=()
while (($# > 0)); do
  case "$1" in
    --prefill-a4-payload|--prefill-a4-policy|--prefill-a4-receipt|--mode)
      (($# >= 2)) || { echo "missing value for $1" >&2; usage; exit 2; }
      option=$1
      value=$2
      shift 2
      case "${option}" in
        --prefill-a4-payload) prefill_a4_payload=${value} ;;
        --prefill-a4-policy) prefill_a4_policy=${value} ;;
        --prefill-a4-receipt) prefill_a4_receipt=${value} ;;
        --mode)
          ((mode_seen == 0)) || {
            echo "--mode may be specified only once" >&2
            exit 2
          }
          mode=${value}
          mode_seen=1
          ;;
      esac
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      positional+=("$@")
      break
      ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done

[[ ${#positional[@]} -ge 6 && ${#positional[@]} -le 8 ]] || {
  usage
  exit 2
}
case "${mode}" in
  exact|native-gdn) ;;
  *) echo "--mode must be exact or native-gdn" >&2; exit 2 ;;
esac

server=${positional[0]}
model_dir=${positional[1]}
manifest=${positional[2]}
manifest_sha256=${positional[3]}
corpus_dir=${positional[4]}
output_root=${positional[5]}
bucket_selector=${positional[6]:-all}
phase_selector=${positional[7]:-prefill1}
base_port=${Q3X_EVAL_PORT:-18080}
readiness_timeout=${Q3X_LONG_EVAL_READINESS_TIMEOUT_SECONDS:-600}

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
validator="${repository}/tools/evaluation/validate_long_prefill_manifest.py"
native_gdn_selector=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION

positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

[[ -x "${server}" ]] || { echo "missing executable: ${server}" >&2; exit 2; }
[[ -d "${model_dir}" ]] || { echo "missing model: ${model_dir}" >&2; exit 2; }
[[ -f "${manifest}" ]] || { echo "missing manifest: ${manifest}" >&2; exit 2; }
[[ -d "${corpus_dir}" ]] || { echo "missing corpus directory: ${corpus_dir}" >&2; exit 2; }
[[ -x "${validator}" ]] || { echo "missing manifest validator: ${validator}" >&2; exit 2; }
[[ -f "${prefill_a4_payload}" ]] || {
  echo "missing required Prefill A4 payload: ${prefill_a4_payload:-<unset>}" >&2
  exit 2
}
[[ -f "${prefill_a4_policy}" ]] || {
  echo "missing required Prefill A4 policy: ${prefill_a4_policy:-<unset>}" >&2
  exit 2
}
[[ -f "${prefill_a4_receipt}" ]] || {
  echo "missing required Prefill A4 receipt: ${prefill_a4_receipt:-<unset>}" >&2
  exit 2
}
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

if [[ "${mode}" == native-gdn ]] &&
   ! grep -F "${native_gdn_selector}" < <(strings -a "${server}") >/dev/null; then
  echo "server does not contain the native-GDN runtime selector" >&2
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
# Keep the child process on production defaults. Remove inherited experiment
# selectors without maintaining another obsolete per-feature environment list;
# native-gdn adds back exactly its declared selector.
runtime_env=(env)
while IFS= read -r variable; do
  case "${variable}" in
    Q3X_RUN_*|Q3X_DISABLE_*|Q3X_GDN_*|Q3X_FULL_ATTENTION_FLASHINFER_DIRECT)
      runtime_env+=(-u "${variable}")
      ;;
  esac
done < <(compgen -e)
runtime_env+=(-u Q3X_DISABLE_OPTIMIZED_PREFILL)
if [[ "${mode}" == native-gdn ]]; then
  runtime_env+=(Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1)
fi

printf 'long_context_matrix server=%q server_sha256=%s manifest=%q manifest_sha256=%s runs=%s mode=%s dry_run=%s\n' \
  "${server}" "${server_sha256}" "${manifest}" "${manifest_actual_sha256}" \
  "${#selected_runs[@]}" "${mode}" "${dry_run}"
echo 'long_context_runtime_contract grouped_q64=compiled_and_probe_selected long_prefill=build_enabled_and_capacity_reserved optimized_prefill_disabled=0 verification=readiness_log'

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
    --prefill-a4-payload "${prefill_a4_payload}"
    --prefill-a4-policy "${prefill_a4_policy}"
    --prefill-a4-receipt "${prefill_a4_receipt}"
  )
  printf 'long_context_run bucket=%s phase=%s token_window=%s..%s corpus=%q corpus_sha256=%s planned_arena_bytes=%s\n' \
    "${bucket}" "${phase}" "${token_window_min}" "${token_window_max}" \
    "${corpus}" "${corpus_sha256}" "${planned_arena_bytes}"
  printf 'server_startup_args'
  printf ' %q' "${runtime_env[@]}" "${server_args[@]}"
  printf '\n'
  printf 'client_args connect_timeout=%s read_timeout=%s total_timeout=%s max_tokens=%s number=%s warmup=%s\n' \
    "${connect_timeout}" "${read_timeout}" "${total_timeout}" \
    "${max_tokens}" "${number}" "${warmup}"
  printf 'server_readiness_route=%s completion_route=%s\n' \
    "${readiness_route}" "${completion_route}"
done

if [[ "${dry_run}" == 1 ]]; then
  echo "dry_run_complete=1 performance_evidence=0 startup_contract_check=deferred"
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

  "${runtime_env[@]}" "${server}" "${model_dir}" \
      --host 127.0.0.1 --port "${port}" \
      --model qwen3.6-27b-nvfp4 \
      --max-sequence-length "${max_sequence_length}" \
      --max-output-tokens "${max_tokens}" \
      --request-max-arena-bytes "${request_max_arena_bytes}" \
      --prefill-chunk-size 512 --projection-backend sm87 \
      --prefill-a4-payload "${prefill_a4_payload}" \
      --prefill-a4-policy "${prefill_a4_policy}" \
      --prefill-a4-receipt "${prefill_a4_receipt}" \
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
      'long_context_group_q64_compiled=1 .*long_context_group_q64_probe_selected=1' \
      "${server_log}"; then
    echo "server readiness did not prove native grouped-Q64 availability and selection" >&2
    exit 5
  fi
  if ! grep -Eq \
      "optimized_prefill_disabled=0 .*long_prefill_build_enabled=1 .*long_prefill_hidden_capacity=${max_sequence_length} .*long_prefill_projection_span_capacity=[1-9][0-9]*" \
      "${server_log}"; then
    echo "server did not prove enabled full-capacity layer-major Prefill" >&2
    exit 5
  fi
  if ! grep -Eq \
      'prefill_a4_requested=1 .*prefill_a4_enabled=1 .*prefill_a4_projections=400([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated Prefill A4 400/400" >&2
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
    --outputs-dir "${run_dir}" --name "long-context-${bucket}-${phase}-${mode}" \
    --no-timestamp >"${run_dir}/evalscope.stdout" 2>&1

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  server_pid=
  find "${run_dir}" -name benchmark_summary.json -print \
    -exec sed -n '1,220p' {} \;
done
