#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_native_pure_prefill_matrix.sh \
  --prefill-a4-payload FILE --prefill-a4-policy FILE \
  --prefill-a4-receipt FILE [--mode exact|native-gdn] [--dry-run] \
  ELF MODEL_DIR CORPUS_DIR OUTPUT_ROOT [p512|p1k|p2k|p4k]
EOF
}

mode=exact
mode_seen=0
dry_run=${Q3X_PURE_PREFILL_DRY_RUN:-0}
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
    -*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

[[ ${#positional[@]} -ge 4 && ${#positional[@]} -le 5 ]] || {
  usage
  exit 2
}
case "${mode}" in
  exact|native-gdn) ;;
  *) echo "--mode must be exact or native-gdn" >&2; exit 2 ;;
esac
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "Q3X_PURE_PREFILL_DRY_RUN must be 0 or 1" >&2
  exit 2
}

server=${positional[0]}
model_dir=${positional[1]}
corpus_dir=${positional[2]}
output_root=${positional[3]}
only_bucket=${positional[4]:-}
port=${Q3X_EVAL_PORT:-18080}

[[ -x "${server}" ]] || { echo "missing executable: ${server}" >&2; exit 2; }
[[ -d "${model_dir}" ]] || { echo "missing model: ${model_dir}" >&2; exit 2; }
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
[[ "${port}" =~ ^[1-9][0-9]*$ ]] && ((port <= 65535)) || {
  echo "Q3X_EVAL_PORT must be in [1,65535]" >&2
  exit 2
}

if [[ "${mode}" == native-gdn ]]; then
  for selector in \
    Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION \
    Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION; do
    if ! grep -F "${selector}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not contain the native-GDN selector: ${selector}" >&2
      exit 2
    fi
  done
fi

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
  if [[ "${actual}" != "${corpus_sha[${bucket}]}" && "${dry_run}" == 0 ]]; then
    echo "corpus SHA256 mismatch for ${bucket}: ${actual}" >&2
    exit 2
  fi
  if [[ "${actual}" == "${corpus_sha[${bucket}]}" ]]; then
    corpus_status=verified
  else
    corpus_status=dry-run-unverified
  fi
  printf 'pure_prefill_corpus bucket=%s sha256=%s status=%s\n' \
    "${bucket}" "${actual}" "${corpus_status}"
done

# Remove inherited experiment selectors generically. The benchmark process gets
# production defaults plus, for the candidate mode, exactly the two selectors
# that define the measured native-GDN bundle. Harness-only Q3X_EVAL_* variables
# are consumed before this point.
runtime_env=(env)
sanitized=0
while IFS= read -r variable; do
  case "${variable}" in
    Q3X_RUN_*|Q3X_DISABLE_*|Q3X_GDN_*|Q3X_FULL_ATTENTION_FLASHINFER_DIRECT)
      runtime_env+=(-u "${variable}")
      ((sanitized += 1))
      ;;
  esac
done < <(compgen -e)
runtime_env+=(-u Q3X_DISABLE_OPTIMIZED_PREFILL)
if [[ "${mode}" == native-gdn ]]; then
  runtime_env+=(
    Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1
    Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1
  )
fi

server_args=(
  "${server}" "${model_dir}"
  --host 127.0.0.1 --port "${port}"
  --model qwen3.6-27b-nvfp4
  --max-sequence-length 4096 --max-output-tokens 1
  --prefill-chunk-size 512 --projection-backend sm87
  --prefill-a4-payload "${prefill_a4_payload}"
  --prefill-a4-policy "${prefill_a4_policy}"
  --prefill-a4-receipt "${prefill_a4_receipt}"
)

printf 'pure_prefill_matrix mode=%s dry_run=%s sanitized_experiment_env=%s\n' \
  "${mode}" "${dry_run}" "${sanitized}"
printf 'server_startup_command'
printf ' %q' "${runtime_env[@]}" "${server_args[@]}"
printf '\n'
printf 'startup_contract required=prefill_a4_authenticated_400_of_400,optimized_prefill_disabled_0\n'

if [[ "${dry_run}" == 1 ]]; then
  echo "dry_run_complete=1 performance_evidence=0 startup_contract_check=deferred"
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v uvx >/dev/null || { echo "uvx is required" >&2; exit 2; }
mkdir -p "${output_root}"
server_log="${output_root}/server.log"
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

"${runtime_env[@]}" "${server_args[@]}" >"${server_log}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}" || true
    server_pid=
    echo "server exited before readiness; inspect ${server_log}" >&2
    exit 3
  fi
  if curl -fsS "http://127.0.0.1:${port}/v1/models" \
      >"${output_root}/readiness.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[[ "${ready}" == 1 ]] || {
  echo "readiness timeout; inspect ${server_log}" >&2
  exit 4
}
if ! grep -Eq \
    'max_sequence_length=4096 maximum_output_tokens=1 .*prefill_chunk_size=512 readiness_route=/healthz' \
    "${server_log}"; then
  echo "server readiness did not echo the pure-Prefill capacity contract" >&2
  exit 5
fi
if ! grep -Eq 'optimized_prefill_disabled=0' "${server_log}"; then
  echo "server readiness did not prove optimized_prefill_disabled=0" >&2
  exit 5
fi
if ! grep -Eq \
    'prefill_a4_requested=1 .*prefill_a4_enabled=1 .*prefill_a4_projections=400([[:space:]]|$)' \
    "${server_log}"; then
  echo "server readiness did not prove authenticated Prefill A4 400/400" >&2
  exit 5
fi
echo "startup_contract_check=passed prefill_a4_authenticated=400/400 optimized_prefill_disabled=0"

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
    --outputs-dir "${run_dir}" --name "pure-prefill-${bucket}-${mode}" \
    --no-timestamp >"${run_dir}/evalscope.stdout" 2>&1
  find "${run_dir}" -name benchmark_summary.json -print \
    -exec sed -n '1,220p' {} \;
done
