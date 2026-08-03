#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_native_pure_prefill_matrix.sh \
  --prefill-a4-payload FILE --prefill-a4-policy FILE \
  --prefill-a4-receipt FILE \
  [--prefill-attention-o-k512-payload FILE \
   --prefill-attention-o-k512-policy FILE \
   --prefill-attention-o-k512-receipt FILE] \
  [--prefill-mlp-k512-payload FILE \
   --prefill-mlp-k512-policy FILE \
   --prefill-mlp-k512-receipt FILE] \
  [--prefill-mlp-k512-fragment-native-payload FILE \
   --prefill-mlp-k512-fragment-native-policy FILE \
   --prefill-mlp-k512-fragment-native-receipt FILE] \
  [--prefill-mlp-k512-paired-gateup-canonical-down-payload FILE \
   --prefill-mlp-k512-paired-gateup-canonical-down-policy FILE \
   --prefill-mlp-k512-paired-gateup-canonical-down-receipt FILE] \
  [--prefill-mlp-k512-projection-major-gateup-canonical-down-payload FILE \
   --prefill-mlp-k512-projection-major-gateup-canonical-down-policy FILE \
   --prefill-mlp-k512-projection-major-gateup-canonical-down-receipt FILE] \
  [--prefill-mlp-factorized-lane-r1-payload FILE \
   --prefill-mlp-factorized-lane-r1-policy FILE \
   --prefill-mlp-factorized-lane-r1-receipt FILE] \
  [--mode exact|native-gdn|cumulative-prefill|cumulative-prefill-down|cumulative-prefill-attention-down|cumulative-prefill-current-best|cumulative-prefill-current-best-k512|cumulative-prefill-current-best-mlp-k512|cumulative-prefill-current-best-mlp-k512-v1|cumulative-prefill-current-best-mlp-k512-edge|cumulative-prefill-current-best-mlp-k512-edge-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-k256-m128n256-pairfeed-package-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m32n512-owner-k128-b4-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-shape-separated-marlin-package-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4-l2-macro4x4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-m128n128-a-exchange-b3|cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4-gdn-prompt-span-macro|cumulative-prefill-current-best-mlp-k512-edge-m128n64|cumulative-prefill-current-best-mlp-k512-down-m16n64-v2|cumulative-prefill-current-best-mlp-k512-fragment-native|cumulative-prefill-current-best-mlp-k512-fragment-native-m128|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged|cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta|cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256|cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256|cumulative-prefill-short] \
  [--mode cumulative-prefill-factorized-r1-attention-k256-a-exchange-b4] \
  [--dry-run] \
  ELF MODEL_DIR CORPUS_DIR OUTPUT_ROOT [p512|p1k|p2k|p4k]
EOF
}

mode=exact
k256_pairfeed_package_mode=cumulative-prefill-k256-m128n256-pairfeed-package-attention-k256-a-exchange-b4
ldmatrix_pairfeed_baseline_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4
ldmatrix_pairfeed_candidate_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4
m32n512_owner_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
m32n512_owner_candidate_mode=cumulative-prefill-current-best-mlp-k512-edge-m32n512-owner-k128-b4-down-16warp-pairring-attention-k256-a-exchange-b4
shape_separated_marlin_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
shape_separated_marlin_candidate_mode=cumulative-prefill-current-best-mlp-k512-shape-separated-marlin-package-attention-k256-a-exchange-b4
l2_macro4x4_candidate_mode=${ldmatrix_pairfeed_candidate_mode}-l2-macro4x4
attention_b3_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
attention_b3_candidate_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-m128n128-a-exchange-b3
projection_major_candidate_mode=cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4
projection_major_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
paired_warp_candidate_mode=cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4
paired_warp_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
projection_serial_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
projection_serial_candidate_mode=cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-down-16warp-pairring-attention-k256-a-exchange-b4
same_cta_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
same_cta_candidate_mode=cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-down-16warp-pairring-attention-k256-a-exchange-b4
fused_quantize_baseline_mode=${ldmatrix_pairfeed_candidate_mode}
fused_quantize_candidate_mode=cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-down-16warp-pairring-attention-k256-a-exchange-b4
gdn_prompt_span_baseline_mode=${ldmatrix_pairfeed_baseline_mode}
gdn_prompt_span_candidate_mode=${gdn_prompt_span_baseline_mode}-gdn-prompt-span-macro
factorized_r1_mode_name=cumulative-prefill-factorized-r1-attention-k256-a-exchange-b4
mode_seen=0
dry_run=${Q3X_PURE_PREFILL_DRY_RUN:-0}
profile_request_index=${Q3X_EVAL_PROFILE_REQUEST_INDEX:-0}
nsys_output=${Q3X_EVAL_NSYS_OUTPUT:-}
eval_number=${Q3X_EVAL_NUMBER-4}
prefill_a4_payload=
prefill_a4_policy=
prefill_a4_receipt=
prefill_attention_o_k512_payload=
prefill_attention_o_k512_policy=
prefill_attention_o_k512_receipt=
prefill_mlp_k512_payload=
prefill_mlp_k512_policy=
prefill_mlp_k512_receipt=
prefill_mlp_k512_fragment_native_payload=
prefill_mlp_k512_fragment_native_policy=
prefill_mlp_k512_fragment_native_receipt=
prefill_mlp_k512_paired_gateup_canonical_down_payload=
prefill_mlp_k512_paired_gateup_canonical_down_policy=
prefill_mlp_k512_paired_gateup_canonical_down_receipt=
prefill_mlp_k512_projection_major_gateup_canonical_down_payload=
prefill_mlp_k512_projection_major_gateup_canonical_down_policy=
prefill_mlp_k512_projection_major_gateup_canonical_down_receipt=
prefill_mlp_factorized_lane_r1_payload=
prefill_mlp_factorized_lane_r1_policy=
prefill_mlp_factorized_lane_r1_receipt=
declare -a positional=()

while (($# > 0)); do
  case "$1" in
    --prefill-a4-payload|--prefill-a4-policy|--prefill-a4-receipt|\
    --prefill-attention-o-k512-payload|--prefill-attention-o-k512-policy|\
    --prefill-attention-o-k512-receipt|--prefill-mlp-k512-payload|\
    --prefill-mlp-k512-policy|--prefill-mlp-k512-receipt|\
    --prefill-mlp-k512-fragment-native-payload|\
    --prefill-mlp-k512-fragment-native-policy|\
    --prefill-mlp-k512-fragment-native-receipt|\
    --prefill-mlp-k512-paired-gateup-canonical-down-payload|\
    --prefill-mlp-k512-paired-gateup-canonical-down-policy|\
    --prefill-mlp-k512-paired-gateup-canonical-down-receipt|\
    --prefill-mlp-k512-projection-major-gateup-canonical-down-payload|\
    --prefill-mlp-k512-projection-major-gateup-canonical-down-policy|\
    --prefill-mlp-k512-projection-major-gateup-canonical-down-receipt|\
    --prefill-mlp-factorized-lane-r1-payload|\
    --prefill-mlp-factorized-lane-r1-policy|\
    --prefill-mlp-factorized-lane-r1-receipt|--mode)
      (($# >= 2)) || { echo "missing value for $1" >&2; usage; exit 2; }
      option=$1
      value=$2
      shift 2
      case "${option}" in
        --prefill-a4-payload) prefill_a4_payload=${value} ;;
        --prefill-a4-policy) prefill_a4_policy=${value} ;;
        --prefill-a4-receipt) prefill_a4_receipt=${value} ;;
        --prefill-attention-o-k512-payload)
          prefill_attention_o_k512_payload=${value}
          ;;
        --prefill-attention-o-k512-policy)
          prefill_attention_o_k512_policy=${value}
          ;;
        --prefill-attention-o-k512-receipt)
          prefill_attention_o_k512_receipt=${value}
          ;;
        --prefill-mlp-k512-payload)
          prefill_mlp_k512_payload=${value}
          ;;
        --prefill-mlp-k512-policy)
          prefill_mlp_k512_policy=${value}
          ;;
        --prefill-mlp-k512-receipt)
          prefill_mlp_k512_receipt=${value}
          ;;
        --prefill-mlp-k512-fragment-native-payload)
          prefill_mlp_k512_fragment_native_payload=${value}
          ;;
        --prefill-mlp-k512-fragment-native-policy)
          prefill_mlp_k512_fragment_native_policy=${value}
          ;;
        --prefill-mlp-k512-fragment-native-receipt)
          prefill_mlp_k512_fragment_native_receipt=${value}
          ;;
        --prefill-mlp-k512-paired-gateup-canonical-down-payload)
          prefill_mlp_k512_paired_gateup_canonical_down_payload=${value}
          ;;
        --prefill-mlp-k512-paired-gateup-canonical-down-policy)
          prefill_mlp_k512_paired_gateup_canonical_down_policy=${value}
          ;;
        --prefill-mlp-k512-paired-gateup-canonical-down-receipt)
          prefill_mlp_k512_paired_gateup_canonical_down_receipt=${value}
          ;;
        --prefill-mlp-k512-projection-major-gateup-canonical-down-payload)
          prefill_mlp_k512_projection_major_gateup_canonical_down_payload=${value}
          ;;
        --prefill-mlp-k512-projection-major-gateup-canonical-down-policy)
          prefill_mlp_k512_projection_major_gateup_canonical_down_policy=${value}
          ;;
        --prefill-mlp-k512-projection-major-gateup-canonical-down-receipt)
          prefill_mlp_k512_projection_major_gateup_canonical_down_receipt=${value}
          ;;
        --prefill-mlp-factorized-lane-r1-payload)
          prefill_mlp_factorized_lane_r1_payload=${value}
          ;;
        --prefill-mlp-factorized-lane-r1-policy)
          prefill_mlp_factorized_lane_r1_policy=${value}
          ;;
        --prefill-mlp-factorized-lane-r1-receipt)
          prefill_mlp_factorized_lane_r1_receipt=${value}
          ;;
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
  exact|native-gdn|cumulative-prefill|cumulative-prefill-down|cumulative-prefill-attention-down|cumulative-prefill-current-best|cumulative-prefill-current-best-k512|cumulative-prefill-current-best-mlp-k512|cumulative-prefill-current-best-mlp-k512-v1|cumulative-prefill-current-best-mlp-k512-edge|cumulative-prefill-current-best-mlp-k512-edge-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4-l2-macro4x4|cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-down-16warp-pairring-attention-k256-a-exchange-b4|cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4-gdn-prompt-span-macro|cumulative-prefill-current-best-mlp-k512-edge-m128n64|cumulative-prefill-current-best-mlp-k512-down-m16n64-v2|cumulative-prefill-current-best-mlp-k512-fragment-native|cumulative-prefill-current-best-mlp-k512-fragment-native-m128|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged|cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta|cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256|cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256|cumulative-prefill-short|"${k256_pairfeed_package_mode}"|"${attention_b3_candidate_mode}"|"${m32n512_owner_candidate_mode}"|"${shape_separated_marlin_candidate_mode}"|"${factorized_r1_mode_name}") ;;
  *)
    echo "--mode must be exact, native-gdn, cumulative-prefill, or" \
      "cumulative-prefill-down, cumulative-prefill-attention-down, or" \
      "cumulative-prefill-current-best, cumulative-prefill-current-best-k512," \
      "cumulative-prefill-current-best-mlp-k512," \
      "cumulative-prefill-current-best-mlp-k512-v1," \
      "cumulative-prefill-current-best-mlp-k512-edge," \
      "cumulative-prefill-current-best-mlp-k512-edge-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "${k256_pairfeed_package_mode}," \
      "cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-down-16warp-pairring-attention-k256-a-exchange-b4," \
      "cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4-gdn-prompt-span-macro," \
      "cumulative-prefill-current-best-mlp-k512-edge-m128n64," \
      "cumulative-prefill-current-best-mlp-k512-down-m16n64-v2," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta," \
      "cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256, or" \
      "${factorized_r1_mode_name}, or" \
      "cumulative-prefill-short" >&2
    exit 2
    ;;
esac
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "Q3X_PURE_PREFILL_DRY_RUN must be 0 or 1" >&2
  exit 2
}
case "${eval_number}" in
  1|4) ;;
  *)
    echo "Q3X_EVAL_NUMBER must be exactly 1 or 4" >&2
    exit 2
    ;;
esac
[[ "${profile_request_index}" =~ ^[0-9]+$ ]] || {
  echo "Q3X_EVAL_PROFILE_REQUEST_INDEX must be a non-negative integer" >&2
  exit 2
}
if [[ -n "${nsys_output}" ]]; then
  ((profile_request_index > 0)) || {
    echo "Q3X_EVAL_NSYS_OUTPUT requires Q3X_EVAL_PROFILE_REQUEST_INDEX > 0" >&2
    exit 2
  }
  command -v nsys >/dev/null || {
    echo "Q3X_EVAL_NSYS_OUTPUT requires nsys" >&2
    exit 2
  }
elif ((profile_request_index > 0)); then
  echo "Q3X_EVAL_PROFILE_REQUEST_INDEX > 0 requires Q3X_EVAL_NSYS_OUTPUT" >&2
  exit 2
fi

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
k512_mode=0
if [[ "${mode}" == cumulative-prefill-current-best-k512 ]]; then
  k512_mode=1
  [[ -f "${prefill_attention_o_k512_payload}" ]] || {
    echo "missing required Prefill Attention-O K512 payload: ${prefill_attention_o_k512_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_attention_o_k512_policy}" ]] || {
    echo "missing required Prefill Attention-O K512 policy: ${prefill_attention_o_k512_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_attention_o_k512_receipt}" ]] || {
    echo "missing required Prefill Attention-O K512 receipt: ${prefill_attention_o_k512_receipt:-<unset>}" >&2
    exit 2
  }
fi
mlp_k512_mode=0
mlp_k512_edge_mode=0
mlp_k512_edge_m64n128_k256_alternating_mode=0
mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode=0
mlp_k512_m32n512_owner_mode=0
mlp_k512_shape_separated_marlin_package_mode=0
mlp_k512_m128n128_projection_serial_mode=0
mlp_k512_m128n64_same_cta_mode=0
mlp_k512_m128n512_fused_quantize_mode=0
mlp_k512_v1_down_pairring_mode=0
mlp_k512_v1_down_16warp_pairring_mode=0
mlp_k512_edge_m128n64_mode=0
mlp_k512_down_m16n64_v2_mode=0
attention_k256_mode=0
attention_k256_a_exchange_b4_mode=0
attention_k256_a_exchange_b3_mode=0
l2_macro4x4_mode=0
gdn_prompt_span_accounting_mode=0
gdn_prompt_span_macro_mode=0
mlp_k512_hybrid_mode=0
mlp_k512_hybrid_down_pairring_mode=0
mlp_k512_projection_major_mode=0
mlp_k512_paired_warp_mode=0
k256_pairfeed_package_selected=0
factorized_r1_mode=0
if [[ "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
      "${mode}" == "${m32n512_owner_candidate_mode}" ||
      "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
      "${mode}" == "${l2_macro4x4_candidate_mode}" ||
      "${mode}" == "${attention_b3_candidate_mode}" ||
      "${mode}" == "${projection_major_candidate_mode}" ||
      "${mode}" == "${paired_warp_candidate_mode}" ||
      "${mode}" == "${projection_serial_candidate_mode}" ||
      "${mode}" == "${same_cta_candidate_mode}" ||
      "${mode}" == "${fused_quantize_candidate_mode}" ||
      "${mode}" == "${gdn_prompt_span_baseline_mode}" ||
      "${mode}" == "${gdn_prompt_span_candidate_mode}" ||
      "${mode}" == "${k256_pairfeed_package_mode}" ||
      "${mode}" == "${factorized_r1_mode_name}" ]]; then
  gdn_prompt_span_accounting_mode=1
  if [[ "${mode}" == "${gdn_prompt_span_candidate_mode}" ]]; then
    gdn_prompt_span_macro_mode=1
  fi
fi
if [[ "${mode}" == "${factorized_r1_mode_name}" ]]; then
  factorized_r1_mode=1
  attention_k256_mode=1
  attention_k256_a_exchange_b4_mode=1
fi
if [[ "${mode}" == "${k256_pairfeed_package_mode}" ]]; then
  k256_pairfeed_package_selected=1
  attention_k256_mode=1
  attention_k256_a_exchange_b4_mode=1
fi
if [[ "${mode}" == "${projection_major_candidate_mode}" ]]; then
  mlp_k512_projection_major_mode=1
  attention_k256_mode=1
  attention_k256_a_exchange_b4_mode=1
fi
if [[ "${mode}" == "${paired_warp_candidate_mode}" ]]; then
  mlp_k512_hybrid_mode=1
  mlp_k512_paired_warp_mode=1
  attention_k256_mode=1
  attention_k256_a_exchange_b4_mode=1
fi
if [[ "${mode}" == \
        cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256 ||
      "${mode}" == \
        cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256 ]]; then
  mlp_k512_hybrid_mode=1
  attention_k256_mode=1
  if [[ "${mode}" == \
      cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256 ]]; then
    mlp_k512_hybrid_down_pairring_mode=1
  fi
fi
if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-v1 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4 ||
      "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
      "${mode}" == "${m32n512_owner_candidate_mode}" ||
      "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
      "${mode}" == "${l2_macro4x4_candidate_mode}" ||
      "${mode}" == "${attention_b3_candidate_mode}" ||
      "${mode}" == "${projection_serial_candidate_mode}" ||
      "${mode}" == "${same_cta_candidate_mode}" ||
      "${mode}" == "${fused_quantize_candidate_mode}" ||
      "${mode}" == "${gdn_prompt_span_candidate_mode}" ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m128n64 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-down-m16n64-v2 ]]; then
  mlp_k512_mode=1
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-down-m16n64-v2 ]]; then
    mlp_k512_edge_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-down-m16n64-v2 ]]; then
    mlp_k512_down_m16n64_v2_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m128n64 ]]; then
    mlp_k512_edge_m128n64_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4 ||
        "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
        "${mode}" == "${m32n512_owner_candidate_mode}" ||
        "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
        "${mode}" == "${l2_macro4x4_candidate_mode}" ||
        "${mode}" == "${attention_b3_candidate_mode}" ||
        "${mode}" == "${projection_serial_candidate_mode}" ||
        "${mode}" == "${same_cta_candidate_mode}" ||
        "${mode}" == "${fused_quantize_candidate_mode}" ||
        "${mode}" == "${gdn_prompt_span_candidate_mode}" ]]; then
    attention_k256_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4 ||
        "${mode}" == "${gdn_prompt_span_candidate_mode}" ]]; then
    mlp_k512_edge_m64n128_k256_alternating_mode=1
  fi
  if [[ "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
        "${mode}" == "${m32n512_owner_candidate_mode}" ||
        "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
        "${mode}" == "${l2_macro4x4_candidate_mode}" ||
        "${mode}" == "${attention_b3_candidate_mode}" ]]; then
    mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode=1
  fi
  if [[ "${mode}" == "${m32n512_owner_candidate_mode}" ]]; then
    mlp_k512_m32n512_owner_mode=1
  fi
  if [[ "${mode}" == "${shape_separated_marlin_candidate_mode}" ]]; then
    mlp_k512_shape_separated_marlin_package_mode=1
  fi
  if [[ "${mode}" == "${projection_serial_candidate_mode}" ]]; then
    mlp_k512_m128n128_projection_serial_mode=1
  fi
  if [[ "${mode}" == "${same_cta_candidate_mode}" ]]; then
    mlp_k512_m128n64_same_cta_mode=1
  fi
  if [[ "${mode}" == "${fused_quantize_candidate_mode}" ]]; then
    mlp_k512_m128n512_fused_quantize_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256 ]]; then
    mlp_k512_v1_down_pairring_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256 ||
        "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4 ||
        "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
        "${mode}" == "${m32n512_owner_candidate_mode}" ||
        "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
        "${mode}" == "${l2_macro4x4_candidate_mode}" ||
        "${mode}" == "${attention_b3_candidate_mode}" ||
        "${mode}" == "${projection_serial_candidate_mode}" ||
        "${mode}" == "${same_cta_candidate_mode}" ||
        "${mode}" == "${fused_quantize_candidate_mode}" ||
        "${mode}" == "${gdn_prompt_span_candidate_mode}" ]]; then
    mlp_k512_v1_down_16warp_pairring_mode=1
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4 ||
        "${mode}" == "${ldmatrix_pairfeed_candidate_mode}" ||
        "${mode}" == "${m32n512_owner_candidate_mode}" ||
        "${mode}" == "${shape_separated_marlin_candidate_mode}" ||
        "${mode}" == "${l2_macro4x4_candidate_mode}" ||
        "${mode}" == "${attention_b3_candidate_mode}" ||
        "${mode}" == "${projection_serial_candidate_mode}" ||
        "${mode}" == "${same_cta_candidate_mode}" ||
        "${mode}" == "${fused_quantize_candidate_mode}" ||
        "${mode}" == "${gdn_prompt_span_candidate_mode}" ]]; then
    attention_k256_a_exchange_b4_mode=1
  fi
  if [[ "${mode}" == "${l2_macro4x4_candidate_mode}" ]]; then
    l2_macro4x4_mode=1
  fi
  if [[ "${mode}" == "${attention_b3_candidate_mode}" ]]; then
    attention_k256_a_exchange_b3_mode=1
  fi
  [[ -f "${prefill_mlp_k512_payload}" ]] || {
    echo "missing required Prefill MLP K512 payload: ${prefill_mlp_k512_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_policy}" ]] || {
    echo "missing required Prefill MLP K512 policy: ${prefill_mlp_k512_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_receipt}" ]] || {
    echo "missing required Prefill MLP K512 receipt: ${prefill_mlp_k512_receipt:-<unset>}" >&2
    exit 2
  }
fi
factorized_r1_layout=sm87_s4_n64_packed_k64_factorized_lane_mlp_v4
factorized_r1_payload_bytes=8568619008
factorized_r1_payload_sha256=
factorized_r1_policy_sha256=
factorized_r1_receipt_sha256=
factorized_r1_manifest_sha256=
factorized_r1_base_receipt_sha256=
if [[ "${factorized_r1_mode}" == 1 ]]; then
  [[ -f "${prefill_mlp_factorized_lane_r1_payload}" ]] || {
    echo "missing required factorized-lane R1 payload: ${prefill_mlp_factorized_lane_r1_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_factorized_lane_r1_policy}" ]] || {
    echo "missing required factorized-lane R1 policy: ${prefill_mlp_factorized_lane_r1_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_factorized_lane_r1_receipt}" ]] || {
    echo "missing required factorized-lane R1 receipt: ${prefill_mlp_factorized_lane_r1_receipt:-<unset>}" >&2
    exit 2
  }
  [[ $(stat -c '%s' "${prefill_mlp_factorized_lane_r1_payload}") == \
      "${factorized_r1_payload_bytes}" ]] || {
    echo "factorized-lane R1 payload is not exactly ${factorized_r1_payload_bytes} bytes" >&2
    exit 2
  }
  factorized_r1_payload_sha256=$(
    sha256sum "${prefill_mlp_factorized_lane_r1_payload}" | awk '{print $1}'
  )
  factorized_r1_policy_sha256=$(
    sha256sum "${prefill_mlp_factorized_lane_r1_policy}" | awk '{print $1}'
  )
  factorized_r1_receipt_sha256=$(
    sha256sum "${prefill_mlp_factorized_lane_r1_receipt}" | awk '{print $1}'
  )
  factorized_r1_base_receipt_sha256=$(
    sha256sum "${prefill_a4_receipt}" | awk '{print $1}'
  )
  factorized_r1_contract=$(
    python3 - "${prefill_mlp_factorized_lane_r1_receipt}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
base = receipt["required_base_k256"]
values = (
    receipt["schema"],
    receipt["mode"],
    str(receipt["production_residency_eligible"]).lower(),
    receipt["residency_eligibility_scope"],
    str(receipt["performance_upper_bound_only"]).lower(),
    str(receipt["quality_production_eligible"]).lower(),
    receipt["physical_layout"],
    str(receipt["lane_count"]),
    str(receipt["payload_bytes"]),
    str(receipt["projection_count"]),
    receipt["manifest_sha256"],
    receipt["policy_sha256"],
    receipt["payload_sha256"],
    base["physical_layout"],
    base["manifest_sha256"],
    base["policy_sha256"],
    base["payload_sha256"],
    base["receipt_sha256"],
)
if any("\t" in value or "\n" in value for value in values):
    raise ValueError("receipt contract values must be scalar")
print("\t".join(values))
PY
  ) || {
    echo "invalid factorized-lane R1 receipt JSON" >&2
    exit 2
  }
  IFS=$'\t' read -r factorized_r1_schema factorized_r1_receipt_mode \
    factorized_r1_residency factorized_r1_scope factorized_r1_upper_bound \
    factorized_r1_quality factorized_r1_receipt_layout \
    factorized_r1_lane_count factorized_r1_receipt_payload_bytes \
    factorized_r1_projection_count factorized_r1_manifest_sha256 \
    factorized_r1_receipt_policy_sha256 \
    factorized_r1_receipt_payload_sha256 factorized_r1_base_layout \
    factorized_r1_base_manifest_sha256 factorized_r1_base_policy_sha256 \
    factorized_r1_base_payload_sha256 factorized_r1_bound_base_receipt_sha256 \
    <<<"${factorized_r1_contract}"
  [[ "${factorized_r1_schema}" == q3x.prefill.mlp-factorized-r1.receipt &&
     "${factorized_r1_receipt_mode}" == performance_upper_bound_r1 &&
     "${factorized_r1_residency}" == false &&
     "${factorized_r1_scope}" == authenticated_abi_only &&
     "${factorized_r1_upper_bound}" == true &&
     "${factorized_r1_quality}" == false &&
     "${factorized_r1_receipt_layout}" == "${factorized_r1_layout}" &&
     "${factorized_r1_lane_count}" == 1 &&
     "${factorized_r1_receipt_payload_bytes}" == \
        "${factorized_r1_payload_bytes}" &&
     "${factorized_r1_projection_count}" == 192 &&
     "${factorized_r1_receipt_policy_sha256}" == \
        "${factorized_r1_policy_sha256}" &&
     "${factorized_r1_receipt_payload_sha256}" == \
        "${factorized_r1_payload_sha256}" &&
     "${factorized_r1_bound_base_receipt_sha256}" == \
        "${factorized_r1_base_receipt_sha256}" ]] || {
    echo "factorized-lane R1 publication receipt does not match the real files" >&2
    exit 2
  }
fi
attention_k256_layout=sm87_s4_n64_packed_k64_scale_k256_consumer_v3
attention_k256_payload_bytes=12353536000
attention_k256_payload_sha256=
attention_k256_policy_sha256=
attention_k256_receipt_sha256=
attention_k256_manifest_sha256=
attention_k256_overlay_layout=
attention_k256_overlay_payload_bytes=
attention_k256_overlay_manifest_sha256=
attention_k256_overlay_policy_sha256=
attention_k256_overlay_payload_sha256=
if [[ "${attention_k256_mode}" == 1 ]]; then
  [[ $(stat -c '%s' "${prefill_a4_payload}") == \
      "${attention_k256_payload_bytes}" ]] || {
    echo "K256 Prefill A4 payload is not exactly ${attention_k256_payload_bytes} bytes" >&2
    exit 2
  }
  command -v python3 >/dev/null || {
    echo "python3 is required to validate the K256 publication receipts" >&2
    exit 2
  }
  attention_k256_payload_sha256=$(
    sha256sum "${prefill_a4_payload}" | awk '{print $1}'
  )
  attention_k256_policy_sha256=$(
    sha256sum "${prefill_a4_policy}" | awk '{print $1}'
  )
  attention_k256_policy_bytes=$(stat -c '%s' "${prefill_a4_policy}")
  attention_k256_receipt_sha256=$(
    sha256sum "${prefill_a4_receipt}" | awk '{print $1}'
  )
  if [[ "${factorized_r1_mode}" == 1 ]]; then
    attention_k256_receipt_contract=$(
    python3 - "${prefill_a4_receipt}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    base = json.load(stream)
values = (
    base["schema"],
    base["sidecar_kind"],
    str(base["packed_k_group_size"]),
    str(base["scale_group_size"]),
    base["physical_layout"],
    str(base["payload_bytes"]),
    str(base["projection_count"]),
    base["manifest_sha256"],
    base["policy_sha256"],
    base["payload_sha256"],
)
print("\t".join(values))
PY
    ) || {
      echo "invalid K256 base publication receipt JSON" >&2
      exit 2
    }
    IFS=$'\t' read -r attention_k256_receipt_schema \
      attention_k256_receipt_kind attention_k256_receipt_packed_k \
      attention_k256_receipt_scale_k attention_k256_receipt_layout \
      attention_k256_receipt_payload_bytes \
      attention_k256_receipt_projection_count \
      attention_k256_manifest_sha256 \
      attention_k256_receipt_policy_sha256 \
      attention_k256_receipt_payload_sha256 \
      <<<"${attention_k256_receipt_contract}"
    [[ "${attention_k256_receipt_schema}" == \
        q3x.prefill.a4.publication-receipt &&
       "${attention_k256_receipt_kind}" == a4_k256 &&
       "${attention_k256_receipt_packed_k}" == 64 &&
       "${attention_k256_receipt_scale_k}" == 256 &&
       "${attention_k256_receipt_layout}" == "${attention_k256_layout}" &&
       "${attention_k256_receipt_payload_bytes}" == \
          "${attention_k256_payload_bytes}" &&
       "${attention_k256_receipt_projection_count}" == 400 &&
       "${attention_k256_receipt_policy_sha256}" == \
          "${attention_k256_policy_sha256}" &&
       "${attention_k256_receipt_payload_sha256}" == \
          "${attention_k256_payload_sha256}" &&
       "${factorized_r1_base_layout}" == "${attention_k256_layout}" &&
       "${factorized_r1_base_manifest_sha256}" == \
          "${attention_k256_manifest_sha256}" &&
       "${factorized_r1_base_policy_sha256}" == \
          "${attention_k256_policy_sha256}" &&
       "${factorized_r1_base_payload_sha256}" == \
          "${attention_k256_payload_sha256}" ]] || {
      echo "factorized-lane R1 is not bound to the selected real K256 base" >&2
      exit 2
    }
    attention_k256_overlay_layout=${factorized_r1_layout}
    attention_k256_overlay_payload_bytes=${factorized_r1_payload_bytes}
    attention_k256_overlay_manifest_sha256=${factorized_r1_manifest_sha256}
    attention_k256_overlay_policy_sha256=${factorized_r1_policy_sha256}
    attention_k256_overlay_payload_sha256=${factorized_r1_payload_sha256}
  elif [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
    attention_k256_receipt_contract=$(
    python3 - "${prefill_a4_receipt}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    base = json.load(stream)
values = (
    base["schema"],
    base["sidecar_kind"],
    str(base["packed_k_group_size"]),
    str(base["scale_group_size"]),
    base["physical_layout"],
    str(base["payload_bytes"]),
    str(base["projection_count"]),
    base["manifest_sha256"],
    base["policy_sha256"],
    base["payload_sha256"],
)
if any("\t" in value or "\n" in value for value in values):
    raise ValueError("receipt contract values must be scalar")
print("\t".join(values))
PY
    ) || {
      echo "invalid K256 base publication receipt JSON" >&2
      exit 2
    }
    IFS=$'\t' read -r attention_k256_receipt_schema \
      attention_k256_receipt_kind attention_k256_receipt_packed_k \
      attention_k256_receipt_scale_k attention_k256_receipt_layout \
      attention_k256_receipt_payload_bytes \
      attention_k256_receipt_projection_count \
      attention_k256_manifest_sha256 \
      attention_k256_receipt_policy_sha256 \
      attention_k256_receipt_payload_sha256 \
      <<<"${attention_k256_receipt_contract}"
    [[ "${attention_k256_receipt_schema}" == \
        q3x.prefill.a4.publication-receipt &&
       "${attention_k256_receipt_kind}" == a4_k256 &&
       "${attention_k256_receipt_packed_k}" == 64 &&
       "${attention_k256_receipt_scale_k}" == 256 &&
       "${attention_k256_receipt_layout}" == "${attention_k256_layout}" &&
       "${attention_k256_receipt_payload_bytes}" == \
          "${attention_k256_payload_bytes}" &&
       "${attention_k256_receipt_projection_count}" == 400 &&
       "${attention_k256_receipt_policy_sha256}" == \
          "${attention_k256_policy_sha256}" &&
       "${attention_k256_receipt_payload_sha256}" == \
          "${attention_k256_payload_sha256}" ]] || {
      echo "K256 Prefill A4 publication receipt does not match the real files" >&2
      exit 2
    }
    attention_k256_overlay_layout=base-a4-k256-only
    attention_k256_overlay_payload_bytes=0
    attention_k256_overlay_manifest_sha256=none
    attention_k256_overlay_policy_sha256=none
    attention_k256_overlay_payload_sha256=none
  elif [[ "${mlp_k512_hybrid_mode}" == 0 &&
        "${mlp_k512_projection_major_mode}" == 0 ]]; then
    attention_k256_receipt_contract=$(
    python3 - "${prefill_a4_receipt}" "${prefill_mlp_k512_receipt}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    base = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    overlay = json.load(stream)
values = (
    base["schema"],
    base["sidecar_kind"],
    str(base["packed_k_group_size"]),
    str(base["scale_group_size"]),
    base["physical_layout"],
    str(base["payload_bytes"]),
    str(base["projection_count"]),
    base["manifest_sha256"],
    base["policy_sha256"],
    base["payload_sha256"],
    overlay["required_base"]["sidecar_kind"],
    overlay["required_base"]["physical_layout"],
    overlay["required_base"]["manifest_sha256"],
    overlay["required_base"]["policy_sha256"],
    overlay["required_base"]["payload_sha256"],
    overlay["schema"],
    overlay["physical_layout"],
    str(overlay["payload_bytes"]),
    str(overlay["projection_count"]),
    overlay["manifest_sha256"],
    overlay["policy_sha256"],
    overlay["payload_sha256"],
)
if any("\t" in value or "\n" in value for value in values):
    raise ValueError("receipt contract values must be scalar")
print("\t".join(values))
PY
    ) || {
      echo "invalid K256 base or K512 overlay publication receipt JSON" >&2
      exit 2
    }
    IFS=$'\t' read -r attention_k256_receipt_schema \
    attention_k256_receipt_kind attention_k256_receipt_packed_k \
    attention_k256_receipt_scale_k attention_k256_receipt_layout \
    attention_k256_receipt_payload_bytes \
    attention_k256_receipt_projection_count \
    attention_k256_manifest_sha256 \
    attention_k256_receipt_policy_sha256 \
    attention_k256_receipt_payload_sha256 overlay_base_kind \
    overlay_base_layout overlay_base_manifest_sha256 \
    overlay_base_policy_sha256 overlay_base_payload_sha256 \
    attention_k256_overlay_schema attention_k256_overlay_layout \
    attention_k256_overlay_payload_bytes \
    attention_k256_overlay_projection_count \
    attention_k256_overlay_manifest_sha256 \
    attention_k256_overlay_policy_sha256 \
    attention_k256_overlay_payload_sha256 \
      <<<"${attention_k256_receipt_contract}"
    [[ "${attention_k256_receipt_schema}" == \
        q3x.prefill.a4.publication-receipt &&
     "${attention_k256_receipt_kind}" == a4_k256 &&
     "${attention_k256_receipt_packed_k}" == 64 &&
     "${attention_k256_receipt_scale_k}" == 256 &&
     "${attention_k256_receipt_layout}" == "${attention_k256_layout}" &&
     "${attention_k256_receipt_payload_bytes}" == \
        "${attention_k256_payload_bytes}" &&
     "${attention_k256_receipt_projection_count}" == 400 &&
     "${attention_k256_receipt_policy_sha256}" == \
        "${attention_k256_policy_sha256}" &&
     "${attention_k256_receipt_payload_sha256}" == \
        "${attention_k256_payload_sha256}" ]] || {
    echo "K256 Prefill A4 publication receipt does not match the real files" >&2
    exit 2
  }
    [[ "${overlay_base_kind}" == a4_k256 &&
     "${overlay_base_layout}" == "${attention_k256_layout}" &&
     "${overlay_base_manifest_sha256}" == \
        "${attention_k256_manifest_sha256}" &&
     "${overlay_base_policy_sha256}" == \
        "${attention_k256_policy_sha256}" &&
     "${overlay_base_payload_sha256}" == \
        "${attention_k256_payload_sha256}" ]] || {
    echo "K512 MLP overlay receipt is not bound to the selected K256 base" >&2
    exit 2
  }
    [[ "${attention_k256_overlay_schema}" == \
        q3x.prefill.mlp-k512.publication-receipt &&
     "${attention_k256_overlay_layout}" == \
        sm87_s4_n64_packed_k64_scale_k512_mlp_v1 &&
     "${attention_k256_overlay_payload_bytes}" == 8623226880 &&
     "${attention_k256_overlay_projection_count}" == 192 ]] || {
    echo "K512 MLP overlay receipt has the wrong production contract" >&2
      exit 2
    }
  fi
fi
mlp_k512_args=0
if [[ -n "${prefill_mlp_k512_payload}" ]]; then
  ((mlp_k512_args += 1))
fi
if [[ -n "${prefill_mlp_k512_policy}" ]]; then
  ((mlp_k512_args += 1))
fi
if [[ -n "${prefill_mlp_k512_receipt}" ]]; then
  ((mlp_k512_args += 1))
fi
((mlp_k512_args == 0 || mlp_k512_args == 3)) || {
  echo "Prefill MLP K512 v1 payload, policy, and receipt are required together" >&2
  exit 2
}
mlp_k512_fragment_native_args=0
if [[ -n "${prefill_mlp_k512_fragment_native_payload}" ]]; then
  ((mlp_k512_fragment_native_args += 1))
fi
if [[ -n "${prefill_mlp_k512_fragment_native_policy}" ]]; then
  ((mlp_k512_fragment_native_args += 1))
fi
if [[ -n "${prefill_mlp_k512_fragment_native_receipt}" ]]; then
  ((mlp_k512_fragment_native_args += 1))
fi
((mlp_k512_fragment_native_args == 0 ||
  mlp_k512_fragment_native_args == 3)) || {
  echo "Prefill fragment-native MLP K512 payload, policy, and receipt are required together" >&2
  exit 2
}
mlp_k512_hybrid_args=0
if [[ -n "${prefill_mlp_k512_paired_gateup_canonical_down_payload}" ]]; then
  ((mlp_k512_hybrid_args += 1))
fi
if [[ -n "${prefill_mlp_k512_paired_gateup_canonical_down_policy}" ]]; then
  ((mlp_k512_hybrid_args += 1))
fi
if [[ -n "${prefill_mlp_k512_paired_gateup_canonical_down_receipt}" ]]; then
  ((mlp_k512_hybrid_args += 1))
fi
((mlp_k512_hybrid_args == 0 || mlp_k512_hybrid_args == 3)) || {
  echo "Prefill paired-GateUp/canonical-Down MLP K512 payload, policy, and receipt are required together" >&2
  exit 2
}
mlp_k512_projection_major_args=0
if [[ -n "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}" ]]; then
  ((mlp_k512_projection_major_args += 1))
fi
if [[ -n "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}" ]]; then
  ((mlp_k512_projection_major_args += 1))
fi
if [[ -n "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}" ]]; then
  ((mlp_k512_projection_major_args += 1))
fi
((mlp_k512_projection_major_args == 0 ||
  mlp_k512_projection_major_args == 3)) || {
  echo "Prefill projection-major-GateUp/canonical-Down MLP K512 payload, policy, and receipt are required together" >&2
  exit 2
}
factorized_r1_args=0
if [[ -n "${prefill_mlp_factorized_lane_r1_payload}" ]]; then
  ((factorized_r1_args += 1))
fi
if [[ -n "${prefill_mlp_factorized_lane_r1_policy}" ]]; then
  ((factorized_r1_args += 1))
fi
if [[ -n "${prefill_mlp_factorized_lane_r1_receipt}" ]]; then
  ((factorized_r1_args += 1))
fi
((factorized_r1_args == 0 || factorized_r1_args == 3)) || {
  echo "Prefill factorized-lane R1 payload, policy, and receipt are required together" >&2
  exit 2
}
if [[ "${factorized_r1_mode}" == 1 && "${factorized_r1_args}" != 3 ]]; then
  echo "factorized-lane R1 mode requires its complete publication triplet" >&2
  exit 2
fi
if [[ "${factorized_r1_mode}" == 0 && "${factorized_r1_args}" != 0 ]]; then
  echo "factorized-lane R1 publication arguments require factorized-lane R1 mode" >&2
  exit 2
fi
if [[ "${k256_pairfeed_package_selected}" == 1 ]] &&
   ((mlp_k512_args != 0 ||
     mlp_k512_fragment_native_args != 0 ||
     mlp_k512_hybrid_args != 0 ||
     mlp_k512_projection_major_args != 0)); then
  echo "K256 MLP package mode uses only the base A4 K256 publication; K512 MLP overlay arguments are forbidden" >&2
  exit 2
fi
if [[ "${k256_pairfeed_package_selected}" == 1 ]] &&
   ((factorized_r1_args != 0)); then
  echo "K256 MLP package mode cannot use the factorized-lane R1 publication" >&2
  exit 2
fi
selected_mlp_k512_publications=0
((mlp_k512_args == 0)) || ((selected_mlp_k512_publications += 1))
((mlp_k512_fragment_native_args == 0)) || \
  ((selected_mlp_k512_publications += 1))
((mlp_k512_hybrid_args == 0)) || ((selected_mlp_k512_publications += 1))
((mlp_k512_projection_major_args == 0)) || \
  ((selected_mlp_k512_publications += 1))
((factorized_r1_args == 0)) || ((selected_mlp_k512_publications += 1))
if ((selected_mlp_k512_publications > 1)); then
  echo "Prefill MLP K512 publications and factorized-lane R1 are mutually exclusive" >&2
  exit 2
fi
mlp_k512_fragment_native_mode=0
mlp_k512_fragment_native_gateup_variant=none
mlp_k512_fragment_native_down_variant=none
mlp_k512_fragment_native_layout=sm87_s4_gateup_n64_paired_down_n128_fragment_native_scale_k512_mlp_v2
mlp_k512_fragment_native_payload_bytes=8623226880
mlp_k512_fragment_native_payload_sha256=
mlp_k512_fragment_native_policy_sha256=
mlp_k512_fragment_native_receipt_sha256=
if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta ]]; then
  mlp_k512_fragment_native_mode=1
  mlp_k512_fragment_native_down_variant=m64n128
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128 ]]; then
    mlp_k512_fragment_native_gateup_variant=m128n32
  elif [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged ]]; then
    mlp_k512_fragment_native_gateup_variant=m128n64_staged
  elif [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta ]]; then
    mlp_k512_fragment_native_gateup_variant=m64n128_1cta
  elif [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta ||
          "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta ]]; then
    mlp_k512_fragment_native_gateup_variant=m128n64_1cta
  else
    mlp_k512_fragment_native_gateup_variant=m64n64
  fi
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta ]]; then
    mlp_k512_fragment_native_down_variant=m128n256_1cta
  fi
  [[ -f "${prefill_mlp_k512_fragment_native_payload}" ]] || {
    echo "missing required fragment-native MLP K512 payload: ${prefill_mlp_k512_fragment_native_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_fragment_native_policy}" ]] || {
    echo "missing required fragment-native MLP K512 policy: ${prefill_mlp_k512_fragment_native_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_fragment_native_receipt}" ]] || {
    echo "missing required fragment-native MLP K512 receipt: ${prefill_mlp_k512_fragment_native_receipt:-<unset>}" >&2
    exit 2
  }
  [[ $(stat -c '%s' "${prefill_mlp_k512_fragment_native_payload}") == "${mlp_k512_fragment_native_payload_bytes}" ]] || {
    echo "fragment-native MLP K512 payload is not exactly ${mlp_k512_fragment_native_payload_bytes} bytes" >&2
    exit 2
  }
  mlp_k512_fragment_native_policy_sha256=$(
    sha256sum "${prefill_mlp_k512_fragment_native_policy}" | awk '{print $1}'
  )
  mlp_k512_fragment_native_receipt_sha256=$(
    sha256sum "${prefill_mlp_k512_fragment_native_receipt}" | awk '{print $1}'
  )
  command -v python3 >/dev/null || {
    echo "python3 is required to validate the fragment-native receipt" >&2
    exit 2
  }
  fragment_receipt_contract=$(
    python3 - "${prefill_mlp_k512_fragment_native_receipt}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    receipt = json.load(stream)
values = (
    receipt["schema"],
    receipt["physical_layout"],
    str(receipt["payload_bytes"]),
    receipt["payload_sha256"],
    receipt["source_v1"]["policy_sha256"],
)
if any("\t" in value or "\n" in value for value in values):
    raise ValueError("receipt contract values must be scalar")
print("\t".join(values))
PY
  ) || {
    echo "invalid fragment-native publication receipt JSON" >&2
    exit 2
  }
  IFS=$'\t' read -r fragment_receipt_schema fragment_receipt_layout \
    fragment_receipt_payload_bytes fragment_receipt_payload_sha256 \
    fragment_receipt_policy_sha256 <<<"${fragment_receipt_contract}"
  [[ "${fragment_receipt_schema}" == \
      q3x.prefill.mlp-k512.fragment-native.publication-receipt &&
     "${fragment_receipt_layout}" == \
      "${mlp_k512_fragment_native_layout}" &&
     "${fragment_receipt_payload_bytes}" == \
      "${mlp_k512_fragment_native_payload_bytes}" ]] || {
    echo "fragment-native publication receipt contract mismatch" >&2
    exit 2
  }
  [[ "${fragment_receipt_policy_sha256}" == \
      "${mlp_k512_fragment_native_policy_sha256}" ]] || {
    echo "fragment-native policy SHA256 does not match source_v1 receipt" >&2
    exit 2
  }
  mlp_k512_fragment_native_payload_sha256=$(
    sha256sum "${prefill_mlp_k512_fragment_native_payload}" | awk '{print $1}'
  )
  [[ "${fragment_receipt_payload_sha256}" == \
      "${mlp_k512_fragment_native_payload_sha256}" ]] || {
    echo "fragment-native payload SHA256 does not match receipt" >&2
    exit 2
  }
fi
mlp_k512_hybrid_layout=sm87_s4_gateup_n64_paired_down_n64_canonical_scale_k512_mlp_hybrid_v1
mlp_k512_hybrid_payload_bytes=8623226880
mlp_k512_hybrid_payload_sha256=
mlp_k512_hybrid_policy_sha256=
mlp_k512_hybrid_receipt_sha256=
mlp_k512_hybrid_manifest_sha256=
mlp_k512_hybrid_source_v1_receipt_sha256=
mlp_k512_hybrid_source_v1_manifest_sha256=
mlp_k512_hybrid_source_v1_payload_sha256=
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  [[ -f "${prefill_mlp_k512_paired_gateup_canonical_down_payload}" ]] || {
    echo "missing required paired-GateUp/canonical-Down MLP K512 payload: ${prefill_mlp_k512_paired_gateup_canonical_down_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_paired_gateup_canonical_down_policy}" ]] || {
    echo "missing required paired-GateUp/canonical-Down MLP K512 policy: ${prefill_mlp_k512_paired_gateup_canonical_down_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_paired_gateup_canonical_down_receipt}" ]] || {
    echo "missing required paired-GateUp/canonical-Down MLP K512 receipt: ${prefill_mlp_k512_paired_gateup_canonical_down_receipt:-<unset>}" >&2
    exit 2
  }
  [[ $(stat -c '%s' \
      "${prefill_mlp_k512_paired_gateup_canonical_down_payload}") == \
      "${mlp_k512_hybrid_payload_bytes}" ]] || {
    echo "paired-GateUp/canonical-Down MLP K512 payload is not exactly ${mlp_k512_hybrid_payload_bytes} bytes" >&2
    exit 2
  }
  mlp_k512_hybrid_payload_sha256=$(
    sha256sum \
      "${prefill_mlp_k512_paired_gateup_canonical_down_payload}" |
      awk '{print $1}'
  )
  mlp_k512_hybrid_policy_sha256=$(
    sha256sum \
      "${prefill_mlp_k512_paired_gateup_canonical_down_policy}" |
      awk '{print $1}'
  )
  mlp_k512_hybrid_receipt_sha256=$(
    sha256sum \
      "${prefill_mlp_k512_paired_gateup_canonical_down_receipt}" |
      awk '{print $1}'
  )
  mlp_k512_hybrid_policy_bytes=$(stat -c '%s' \
    "${prefill_mlp_k512_paired_gateup_canonical_down_policy}")
  hybrid_receipt_contract=$(
    python3 - \
      "${prefill_a4_receipt}" \
      "${prefill_mlp_k512_paired_gateup_canonical_down_receipt}" \
      "${attention_k256_payload_sha256}" \
      "${attention_k256_policy_sha256}" \
      "${attention_k256_policy_bytes}" \
      "${mlp_k512_hybrid_payload_sha256}" \
      "${mlp_k512_hybrid_policy_sha256}" \
      "${mlp_k512_hybrid_policy_bytes}" <<'PY'
import json
import re
import sys

(
    base_path,
    hybrid_path,
    base_payload_sha256,
    base_policy_sha256,
    base_policy_bytes_text,
    hybrid_payload_sha256,
    hybrid_policy_sha256,
    hybrid_policy_bytes_text,
) = sys.argv[1:]
hybrid_policy_bytes = int(hybrid_policy_bytes_text)
base_policy_bytes = int(base_policy_bytes_text)
with open(base_path, encoding="utf-8") as stream:
    base = json.load(stream)
with open(hybrid_path, encoding="utf-8") as stream:
    hybrid = json.load(stream)

sha256 = re.compile(r"[0-9a-f]{64}").fullmatch


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} has the wrong strict JSON fields")


base_keys = (
    "schema", "version", "mode", "production_residency_eligible",
    "sidecar_kind", "packed_k_group_size", "scale_group_size",
    "physical_layout", "source_checkpoint_id", "source_config_sha256",
    "source_index_sha256", "manifest_sha256", "policy_sha256",
    "policy_bytes", "payload_sha256", "payload_bytes", "projection_count",
)
hybrid_keys = (
    "schema", "version", "mode", "production_residency_eligible",
    "physical_layout", "gateup_physical_layout", "down_physical_layout",
    "source_checkpoint_id", "source_config_sha256", "source_index_sha256",
    "required_base", "source_v1", "manifest_sha256", "payload_sha256",
    "payload_bytes", "layer_count",
)
version_keys = ("major", "minor")
base_binding_keys = (
    "sidecar_kind", "physical_layout", "manifest_sha256",
    "policy_sha256", "payload_sha256",
)
source_keys = (
    "physical_layout", "receipt_sha256", "manifest_sha256",
    "policy_sha256", "policy_bytes", "payload_sha256", "payload_bytes",
)
exact_keys(base, base_keys, "K256 base receipt")
exact_keys(hybrid, hybrid_keys, "hybrid receipt")
exact_keys(base["version"], version_keys, "K256 base version")
exact_keys(hybrid["version"], version_keys, "hybrid version")
exact_keys(hybrid["required_base"], base_binding_keys, "required_base")
exact_keys(hybrid["source_v1"], source_keys, "source_v1")

if not (
    base["schema"] == "q3x.prefill.a4.publication-receipt"
    and base["version"] == {"major": 3, "minor": 0}
    and base["mode"] == "production_calibrated"
    and base["production_residency_eligible"] is True
    and base["sidecar_kind"] == "a4_k256"
    and base["packed_k_group_size"] == 64
    and base["scale_group_size"] == 256
    and base["physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k256_consumer_v3"
    and isinstance(base["source_checkpoint_id"], str)
    and bool(base["source_checkpoint_id"])
    and sha256(base["source_config_sha256"])
    and sha256(base["source_index_sha256"])
    and sha256(base["manifest_sha256"])
    and base["policy_sha256"] == base_policy_sha256
    and base["policy_bytes"] == base_policy_bytes
    and base["payload_sha256"] == base_payload_sha256
    and base["payload_bytes"] == 12_353_536_000
    and base["projection_count"] == 400
):
    raise ValueError("K256 base receipt does not bind the selected real files")

required_base = hybrid["required_base"]
source_v1 = hybrid["source_v1"]
if not (
    hybrid["schema"]
        == "q3x.prefill.mlp-k512.paired-gateup-canonical-down."
           "publication-receipt"
    and hybrid["version"] == {"major": 1, "minor": 0}
    and hybrid["mode"] == "lossless_gateup_permutation_down_passthrough"
    and hybrid["production_residency_eligible"] is True
    and hybrid["physical_layout"]
        == "sm87_s4_gateup_n64_paired_down_n64_canonical_scale_k512_"
           "mlp_hybrid_v1"
    and hybrid["gateup_physical_layout"]
        == "sm87_s4_gateup_n64_paired_fragment_register_v1"
    and hybrid["down_physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
    and hybrid["source_checkpoint_id"] == base["source_checkpoint_id"]
    and hybrid["source_config_sha256"] == base["source_config_sha256"]
    and hybrid["source_index_sha256"] == base["source_index_sha256"]
    and required_base["sidecar_kind"] == base["sidecar_kind"]
    and required_base["physical_layout"] == base["physical_layout"]
    and required_base["manifest_sha256"] == base["manifest_sha256"]
    and required_base["policy_sha256"] == base["policy_sha256"]
    and required_base["payload_sha256"] == base["payload_sha256"]
    and source_v1["physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
    and sha256(source_v1["receipt_sha256"])
    and sha256(source_v1["manifest_sha256"])
    and source_v1["policy_sha256"] == hybrid_policy_sha256
    and source_v1["policy_bytes"] == hybrid_policy_bytes
    and sha256(source_v1["payload_sha256"])
    and source_v1["payload_bytes"] == 8_623_226_880
    and sha256(hybrid["manifest_sha256"])
    and hybrid["payload_sha256"] == hybrid_payload_sha256
    and hybrid["payload_bytes"] == 8_623_226_880
    and hybrid["layer_count"] == 64
):
    raise ValueError("hybrid receipt does not bind the complete real-file chain")

values = (
    base["manifest_sha256"],
    hybrid["manifest_sha256"],
    source_v1["receipt_sha256"],
    source_v1["manifest_sha256"],
    source_v1["payload_sha256"],
)
print("\t".join(values))
PY
  ) || {
    echo "invalid paired-GateUp/canonical-Down K512 publication chain" >&2
    exit 2
  }
  IFS=$'\t' read -r attention_k256_manifest_sha256 \
    mlp_k512_hybrid_manifest_sha256 \
    mlp_k512_hybrid_source_v1_receipt_sha256 \
    mlp_k512_hybrid_source_v1_manifest_sha256 \
    mlp_k512_hybrid_source_v1_payload_sha256 \
    <<<"${hybrid_receipt_contract}"
  attention_k256_overlay_layout=${mlp_k512_hybrid_layout}
  attention_k256_overlay_payload_bytes=${mlp_k512_hybrid_payload_bytes}
  attention_k256_overlay_manifest_sha256=${mlp_k512_hybrid_manifest_sha256}
  attention_k256_overlay_policy_sha256=${mlp_k512_hybrid_policy_sha256}
  attention_k256_overlay_payload_sha256=${mlp_k512_hybrid_payload_sha256}
fi
mlp_k512_projection_major_layout=sm87_s4_gateup_n64_projection_major_down_n64_canonical_scale_k512_mlp_hybrid_v2
mlp_k512_projection_major_payload_bytes=8623226880
mlp_k512_projection_major_payload_sha256=
mlp_k512_projection_major_policy_sha256=
mlp_k512_projection_major_receipt_sha256=
mlp_k512_projection_major_manifest_sha256=
mlp_k512_projection_major_source_v1_receipt_sha256=
mlp_k512_projection_major_source_v1_manifest_sha256=
mlp_k512_projection_major_source_v1_payload_sha256=
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  [[ -f "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}" ]] || {
    echo "missing required projection-major-GateUp/canonical-Down MLP K512 payload: ${prefill_mlp_k512_projection_major_gateup_canonical_down_payload:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}" ]] || {
    echo "missing required projection-major-GateUp/canonical-Down MLP K512 policy: ${prefill_mlp_k512_projection_major_gateup_canonical_down_policy:-<unset>}" >&2
    exit 2
  }
  [[ -f "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}" ]] || {
    echo "missing required projection-major-GateUp/canonical-Down MLP K512 receipt: ${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt:-<unset>}" >&2
    exit 2
  }
  projection_major_payload_identity=$(stat -Lc '%d:%i' \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}")
  projection_major_policy_identity=$(stat -Lc '%d:%i' \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}")
  projection_major_receipt_identity=$(stat -Lc '%d:%i' \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}")
  if [[ "${projection_major_payload_identity}" == "${projection_major_policy_identity}" ||
        "${projection_major_payload_identity}" == "${projection_major_receipt_identity}" ||
        "${projection_major_policy_identity}" == "${projection_major_receipt_identity}" ]]; then
    echo "projection-major-GateUp/canonical-Down MLP K512 payload, policy, and receipt must be distinct files" >&2
    exit 2
  fi
  [[ $(stat -c '%s' \
      "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}") == \
      "${mlp_k512_projection_major_payload_bytes}" ]] || {
    echo "projection-major-GateUp/canonical-Down MLP K512 payload is not exactly ${mlp_k512_projection_major_payload_bytes} bytes" >&2
    exit 2
  }
  mlp_k512_projection_major_payload_sha256=$(sha256sum \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}" |
    awk '{print $1}')
  mlp_k512_projection_major_policy_sha256=$(sha256sum \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}" |
    awk '{print $1}')
  mlp_k512_projection_major_receipt_sha256=$(sha256sum \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}" |
    awk '{print $1}')
  mlp_k512_projection_major_policy_bytes=$(stat -c '%s' \
    "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}")
  projection_major_receipt_contract=$(
    python3 - \
      "${prefill_a4_receipt}" \
      "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}" \
      "${attention_k256_payload_sha256}" \
      "${attention_k256_policy_sha256}" \
      "${attention_k256_policy_bytes}" \
      "${mlp_k512_projection_major_payload_sha256}" \
      "${mlp_k512_projection_major_policy_sha256}" \
      "${mlp_k512_projection_major_policy_bytes}" <<'PY'
import json
import re
import sys

(
    base_path,
    publication_path,
    base_payload_sha256,
    base_policy_sha256,
    base_policy_bytes_text,
    publication_payload_sha256,
    publication_policy_sha256,
    publication_policy_bytes_text,
) = sys.argv[1:]
base_policy_bytes = int(base_policy_bytes_text)
publication_policy_bytes = int(publication_policy_bytes_text)
with open(base_path, encoding="utf-8") as stream:
    base = json.load(stream)
with open(publication_path, encoding="utf-8") as stream:
    publication = json.load(stream)

sha256 = re.compile(r"[0-9a-f]{64}").fullmatch


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} has the wrong strict JSON fields")


base_keys = (
    "schema", "version", "mode", "production_residency_eligible",
    "sidecar_kind", "packed_k_group_size", "scale_group_size",
    "physical_layout", "source_checkpoint_id", "source_config_sha256",
    "source_index_sha256", "manifest_sha256", "policy_sha256",
    "policy_bytes", "payload_sha256", "payload_bytes", "projection_count",
)
publication_keys = (
    "schema", "version", "mode", "production_residency_eligible",
    "physical_layout", "gateup_physical_layout", "down_physical_layout",
    "source_checkpoint_id", "source_config_sha256", "source_index_sha256",
    "required_base", "source_v1", "manifest_sha256", "payload_sha256",
    "payload_bytes", "layer_count",
)
version_keys = ("major", "minor")
base_binding_keys = (
    "sidecar_kind", "physical_layout", "manifest_sha256",
    "policy_sha256", "payload_sha256",
)
source_keys = (
    "physical_layout", "receipt_sha256", "manifest_sha256",
    "policy_sha256", "policy_bytes", "payload_sha256", "payload_bytes",
)
exact_keys(base, base_keys, "K256 base receipt")
exact_keys(publication, publication_keys, "projection-major receipt")
exact_keys(base["version"], version_keys, "K256 base version")
exact_keys(publication["version"], version_keys, "publication version")
exact_keys(publication["required_base"], base_binding_keys, "required_base")
exact_keys(publication["source_v1"], source_keys, "source_v1")

if not (
    base["schema"] == "q3x.prefill.a4.publication-receipt"
    and base["version"] == {"major": 3, "minor": 0}
    and base["mode"] == "production_calibrated"
    and base["production_residency_eligible"] is True
    and base["sidecar_kind"] == "a4_k256"
    and base["packed_k_group_size"] == 64
    and base["scale_group_size"] == 256
    and base["physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k256_consumer_v3"
    and isinstance(base["source_checkpoint_id"], str)
    and bool(base["source_checkpoint_id"])
    and sha256(base["source_config_sha256"])
    and sha256(base["source_index_sha256"])
    and sha256(base["manifest_sha256"])
    and base["policy_sha256"] == base_policy_sha256
    and base["policy_bytes"] == base_policy_bytes
    and base["payload_sha256"] == base_payload_sha256
    and base["payload_bytes"] == 12_353_536_000
    and base["projection_count"] == 400
):
    raise ValueError("K256 base receipt does not bind the selected real files")

required_base = publication["required_base"]
source_v1 = publication["source_v1"]
if not (
    publication["schema"]
        == "q3x.prefill.mlp-k512.projection-major-gateup-canonical-down."
           "publication-receipt"
    and publication["version"] == {"major": 2, "minor": 0}
    and publication["mode"]
        == "lossless_projection_major_gateup_permutation_down_passthrough"
    and publication["production_residency_eligible"] is True
    and publication["physical_layout"]
        == "sm87_s4_gateup_n64_projection_major_down_n64_canonical_"
           "scale_k512_mlp_hybrid_v2"
    and publication["gateup_physical_layout"]
        == "sm87_s4_gateup_n64_projection_major_fragment_register_v3"
    and publication["down_physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
    and publication["source_checkpoint_id"] == base["source_checkpoint_id"]
    and publication["source_config_sha256"] == base["source_config_sha256"]
    and publication["source_index_sha256"] == base["source_index_sha256"]
    and required_base["sidecar_kind"] == base["sidecar_kind"]
    and required_base["physical_layout"] == base["physical_layout"]
    and required_base["manifest_sha256"] == base["manifest_sha256"]
    and required_base["policy_sha256"] == base["policy_sha256"]
    and required_base["payload_sha256"] == base["payload_sha256"]
    and source_v1["physical_layout"]
        == "sm87_s4_n64_packed_k64_scale_k512_mlp_v1"
    and sha256(source_v1["receipt_sha256"])
    and sha256(source_v1["manifest_sha256"])
    and source_v1["policy_sha256"] == publication_policy_sha256
    and source_v1["policy_bytes"] == publication_policy_bytes
    and sha256(source_v1["payload_sha256"])
    and source_v1["payload_bytes"] == 8_623_226_880
    and sha256(publication["manifest_sha256"])
    and publication["payload_sha256"] == publication_payload_sha256
    and publication["payload_bytes"] == 8_623_226_880
    and publication["layer_count"] == 64
):
    raise ValueError("projection-major receipt does not bind the real-file chain")

print("\t".join((
    base["manifest_sha256"],
    publication["manifest_sha256"],
    source_v1["receipt_sha256"],
    source_v1["manifest_sha256"],
    source_v1["payload_sha256"],
)))
PY
  ) || {
    echo "invalid projection-major-GateUp/canonical-Down K512 publication chain" >&2
    exit 2
  }
  IFS=$'\t' read -r attention_k256_manifest_sha256 \
    mlp_k512_projection_major_manifest_sha256 \
    mlp_k512_projection_major_source_v1_receipt_sha256 \
    mlp_k512_projection_major_source_v1_manifest_sha256 \
    mlp_k512_projection_major_source_v1_payload_sha256 \
    <<<"${projection_major_receipt_contract}"
  attention_k256_overlay_layout=${mlp_k512_projection_major_layout}
  attention_k256_overlay_payload_bytes=${mlp_k512_projection_major_payload_bytes}
  attention_k256_overlay_manifest_sha256=${mlp_k512_projection_major_manifest_sha256}
  attention_k256_overlay_policy_sha256=${mlp_k512_projection_major_policy_sha256}
  attention_k256_overlay_payload_sha256=${mlp_k512_projection_major_payload_sha256}
fi
[[ ! -e "${output_root}" ]] || {
  echo "refusing to overwrite output root: ${output_root}" >&2
  exit 2
}
[[ "${port}" =~ ^[1-9][0-9]*$ ]] && ((port <= 65535)) || {
  echo "Q3X_EVAL_PORT must be in [1,65535]" >&2
  exit 2
}

declare -a candidate_selectors=()
case "${mode}" in
  native-gdn)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
    )
    ;;
  cumulative-prefill)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
    )
    ;;
  cumulative-prefill-down)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION
    )
    ;;
  cumulative-prefill-attention-down)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
    )
    ;;
  cumulative-prefill-current-best)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-k512)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512|\
  cumulative-prefill-current-best-mlp-k512-edge)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-v1)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4|"${m32n512_owner_candidate_mode}"|"${shape_separated_marlin_candidate_mode}"|"${l2_macro4x4_candidate_mode}"|"${attention_b3_candidate_mode}")
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  "${k256_pairfeed_package_mode}")
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K256_M128N256_PAIRFEED_PACKAGE_ADMISSION
    )
    ;;
  "${factorized_r1_mode_name}")
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_FACTORIZED_LANE_R1_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_PROJECTION_MAJOR_GATEUP_CANONICAL_DOWN_ADMISSION
      Q3X_RUN_A4W4_GATEUP_K512_M64N128_REGISTER_PIPELINE_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION
      Q3X_RUN_A4W4_GATEUP_K512_M64N8_PAIRED_WARP_REGISTER_PIPELINE_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-m128n128-projection-serial-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-m128n64-same-cta-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-m128n512-fused-quantize-down-16warp-pairring-attention-k256-a-exchange-b4)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4-gdn-prompt-span-macro)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION
      Q3X_RUN_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-edge-m128n64)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-down-m16n64-v2)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-hybrid-gate-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-hybrid-gate-down-pairring-attention-k256)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
      Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION
      Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION
      Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION
    )
    ;;
  cumulative-prefill-current-best-mlp-k512-fragment-native|\
  cumulative-prefill-current-best-mlp-k512-fragment-native-m128|\
  cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta|\
  cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged|\
  cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta|\
  cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_FULL_ATTENTION_FLASHINFER_DIRECT
      Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION
      Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION
      Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION
    )
    ;;
  cumulative-prefill-short)
    candidate_selectors=(
      Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION
      Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION
      Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION
      Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION
    )
    ;;
esac
if [[ "${l2_macro4x4_mode}" == 1 ]]; then
  candidate_selectors+=(
    Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_L2_MACRO4X4_ADMISSION
    Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_L2_MACRO4X4_ADMISSION
  )
fi
if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
  candidate_selectors+=(
    Q3X_RUN_A4W4_ATTENTION_K256_M128N128_A_EXCHANGE_B3_ADMISSION
  )
fi
if [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
  candidate_selectors+=(
    Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION
  )
fi
if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
  candidate_selectors+=(
    Q3X_RUN_A4W4_MLP_K512_SHAPE_SEPARATED_MARLIN_PACKAGE_ADMISSION
  )
fi
if ((${#candidate_selectors[@]} > 0)); then
  for selector in "${candidate_selectors[@]}"; do
    if ! grep -F "${selector}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not contain the ${mode} selector: ${selector}" >&2
      exit 2
    fi
  done
fi
gdn_prompt_span_macro_marker=prefill_projection_span_linear_gdn_prompt_span_macro
if [[ "${gdn_prompt_span_macro_mode}" == 1 ]] &&
   ! grep -Fx "${gdn_prompt_span_macro_marker}" \
      < <(strings -a "${server}") >/dev/null; then
  echo "server does not prove the GDN prompt-span macro production stage: ${gdn_prompt_span_macro_marker}" >&2
  exit 2
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  attention_k256_incumbent_markers=(
    prefill_projection_span_linear_qkv_z_k256_m128n256
    prefill_projection_span_linear_output_k256_m128n256
    prefill_projection_span_full_q_k_v_k256_m128n256
    prefill_projection_span_full_output_k256_m128n256
  )
  attention_k256_a_exchange_b4_markers=(
    prefill_projection_span_linear_qkv_z_k256_m128n256_a_exchange_b4
    prefill_projection_span_linear_output_k256_m128n256_a_exchange_b4
    prefill_projection_span_full_q_k_v_k256_m128n256_a_exchange_b4
    prefill_projection_span_full_output_k256_m128n256_a_exchange_b4
  )
  attention_k256_a_exchange_b4_l2_macro4x4_markers=(
    prefill_projection_span_linear_qkv_z_k256_m128n256_a_exchange_b4_l2_macro4x4
    prefill_projection_span_linear_output_k256_m128n256_a_exchange_b4_l2_macro4x4
    prefill_projection_span_full_q_k_v_k256_m128n256_a_exchange_b4_l2_macro4x4
    prefill_projection_span_full_output_k256_m128n256_a_exchange_b4_l2_macro4x4
  )
  attention_k256_a_exchange_b3_markers=(
    prefill_projection_span_linear_qkv_z_k256_m128n128_a_exchange_b3
    prefill_projection_span_linear_output_k256_m128n128_a_exchange_b3
    prefill_projection_span_full_q_k_v_k256_m128n128_a_exchange_b3
    prefill_projection_span_full_output_k256_m128n128_a_exchange_b3
  )
  if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
    attention_k256_markers=(
      "${attention_k256_a_exchange_b3_markers[@]}"
    )
  elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
    attention_k256_markers=(
      "${attention_k256_a_exchange_b4_l2_macro4x4_markers[@]}"
    )
  elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
    attention_k256_markers=(
      "${attention_k256_a_exchange_b4_markers[@]}"
    )
  else
    attention_k256_markers=("${attention_k256_incumbent_markers[@]}")
  fi
  for marker in "${attention_k256_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the K256 Attention production stage: ${marker}" >&2
      exit 2
    fi
  done
fi
k256_pairfeed_input_marker=prefill_projection_span_mlp_k256_input_quantize
k256_pairfeed_gateup_marker=prefill_projection_span_mlp_k256_gateup_down_edge_m128n256_pairfeed
k256_pairfeed_down_marker=prefill_projection_span_mlp_k256_down_m128n128_16warp_pairring
k256_pairfeed_required_markers=(
  "${k256_pairfeed_input_marker}"
  "${k256_pairfeed_gateup_marker}"
  "${k256_pairfeed_down_marker}"
)
if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  for marker in "${k256_pairfeed_required_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the K256 M128N256 pair-feed MLP package stage: ${marker}" >&2
      exit 2
    fi
  done
fi
factorized_r1_required_markers=(
  prefill_projection_span_factorized_lane_r1_input_quantize
  prefill_projection_span_factorized_lane_r1_gateup
  prefill_projection_span_factorized_lane_r1_product_quantize
  prefill_projection_span_factorized_lane_r1_down
)
if [[ "${factorized_r1_mode}" == 1 ]]; then
  for marker in "${factorized_r1_required_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the factorized-lane R1 stage: ${marker}" >&2
      exit 2
    fi
  done
fi
if [[ "${mlp_k512_edge_mode}" == 1 ]]; then
  edge_marker=prefill_projection_span_mlp_k512_gateup_down_edge
  if ! grep -Fx "${edge_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the Gate+Up-to-Down K512 edge stage: ${edge_marker}" >&2
    exit 2
  fi
fi
if [[ "${mlp_k512_edge_m64n128_k256_alternating_mode}" == 1 ]]; then
  alternating_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating
  if ! grep -Fx "${alternating_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the alternating M64N128 K256 Gate+Up stage: ${alternating_marker}" >&2
    exit 2
  fi
fi
ldmatrix_pairfeed_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_ldmatrix_pairfeed
m32n512_owner_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m32n512_owner_k128_b4
shape_separated_marlin_gate_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m64n256_marlin_k64_b3
shape_separated_marlin_down_marker=prefill_projection_span_mlp_k512_down_m64n256_16warp_pairring
if [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 &&
      "${mlp_k512_m32n512_owner_mode}" == 0 ]] &&
   ! grep -Fx "${ldmatrix_pairfeed_marker}" \
      < <(strings -a "${server}") >/dev/null; then
  echo "server does not prove the LDSM pair-feed M64N128 K256 Gate+Up production stage: ${ldmatrix_pairfeed_marker}" >&2
  exit 2
fi
if [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]] &&
   ! grep -Fx "${m32n512_owner_marker}" \
      < <(strings -a "${server}") >/dev/null; then
  echo "server does not prove the M32N512 owner K128/B4 Gate+Up production stage: ${m32n512_owner_marker}" >&2
  exit 2
fi
if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
  for marker in "${shape_separated_marlin_gate_marker}" \
                "${shape_separated_marlin_down_marker}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the shape-separated Marlin MLP package stage: ${marker}" >&2
      exit 2
    fi
  done
fi
projection_major_input_marker=prefill_projection_span_mlp_k512_projection_major_input_quantize
projection_major_gate_marker=prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline
projection_major_down_16warp_marker=prefill_projection_span_mlp_k512_projection_major_down_m128n128_16warp_pairring
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  projection_major_required_markers=(
    "${projection_major_input_marker}"
    "${projection_major_gate_marker}"
    "${projection_major_down_16warp_marker}"
  )
  for marker in "${projection_major_required_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the projection-major-GateUp production stage: ${marker}" >&2
      exit 2
    fi
  done
fi
paired_warp_gate_marker=prefill_projection_span_mlp_k512_gateup_m64n8_paired_warp_register_pipeline
if [[ "${mlp_k512_paired_warp_mode}" == 1 ]] &&
   ! grep -Fx "${paired_warp_gate_marker}" \
      < <(strings -a "${server}") >/dev/null; then
  echo "server does not prove the M64N8 paired-warp GateUp stage: ${paired_warp_gate_marker}" >&2
  exit 2
fi
projection_serial_primary_marker=prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_primary
projection_serial_secondary_marker=prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_secondary
projection_serial_quantize_marker=prefill_projection_span_mlp_k512_product_quantize
projection_serial_markers=(
  "${projection_serial_primary_marker}"
  "${projection_serial_secondary_marker}"
  "${projection_serial_quantize_marker}"
)
if [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
  for marker in "${projection_serial_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the M128N128 projection-serial Gate+Up production stage: ${marker}" >&2
      exit 2
    fi
  done
fi
same_cta_primary_marker=prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary
same_cta_secondary_marker=prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary
same_cta_markers=(
  "${same_cta_primary_marker}"
  "${same_cta_secondary_marker}"
  "${projection_serial_quantize_marker}"
)
if [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
  for marker in "${same_cta_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the M128N64 same-CTA Gate+Up production stage: ${marker}" >&2
      exit 2
    fi
  done
fi
fused_quantize_marker=prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize
if [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]] &&
   ! grep -Fx "${fused_quantize_marker}" \
      < <(strings -a "${server}") >/dev/null; then
  echo "server does not prove the M128N512 fused-quantize Gate+Up production stage: ${fused_quantize_marker}" >&2
  exit 2
fi
pairring_down_marker=prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring
if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
  if ! grep -Fx "${pairring_down_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the M128N128 LDSM pair-ring Down stage: ${pairring_down_marker}" >&2
    exit 2
  fi
fi
if [[ "${l2_macro4x4_mode}" == 1 ]]; then
  pairring_down_16warp_marker=prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring_l2_macro4x4
else
  pairring_down_16warp_marker=prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring
fi
paired_warp_down_16warp_marker=prefill_projection_span_mlp_k512_paired_warp_down_m128n128_16warp_pairring
if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
  if ! grep -Fx "${pairring_down_16warp_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the M128N128 16-warp pair-ring Down stage: ${pairring_down_16warp_marker}" >&2
    exit 2
  fi
fi
if [[ "${mlp_k512_edge_m128n64_mode}" == 1 ]]; then
  edge_m128n64_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m128n64
  if ! grep -Fx "${edge_m128n64_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the M128N64 Gate+Up-to-Down K512 edge stage: ${edge_m128n64_marker}" >&2
    exit 2
  fi
fi
if [[ "${mlp_k512_down_m16n64_v2_mode}" == 1 ]]; then
  down_m16n64_v2_marker=prefill_projection_span_mlp_k512_down_m16n64_v2
  if ! grep -Fx "${down_m16n64_v2_marker}" < <(strings -a "${server}") >/dev/null; then
    echo "server does not prove the Down K512 M16N64 v2 stage: ${down_m16n64_v2_marker}" >&2
    exit 2
  fi
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  hybrid_input_marker=prefill_projection_span_mlp_k512_paired_gateup_canonical_down_input_quantize
  hybrid_gate_marker=prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix
  hybrid_canonical_down_marker=prefill_projection_span_mlp_k512_paired_gateup_canonical_down_down
  hybrid_pairring_down_marker=${pairring_down_marker}
  hybrid_required_markers=("${hybrid_input_marker}")
  if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    hybrid_required_markers+=(
      "${paired_warp_gate_marker}"
      "${paired_warp_down_16warp_marker}"
    )
  elif [[ "${mlp_k512_hybrid_down_pairring_mode}" == 1 ]]; then
    hybrid_required_markers+=("${hybrid_gate_marker}")
    hybrid_required_markers+=("${hybrid_pairring_down_marker}")
  else
    hybrid_required_markers+=("${hybrid_gate_marker}")
    hybrid_required_markers+=("${hybrid_canonical_down_marker}")
  fi
  for marker in "${hybrid_required_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the paired-GateUp/canonical-Down production stage: ${marker}" >&2
      exit 2
    fi
  done
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  declare -a fragment_gateup_markers=()
  declare -a fragment_gateup_rejected_markers=()
  case "${mlp_k512_fragment_native_gateup_variant}" in
    m64n128_1cta)
      fragment_gateup_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_secondary
      )
      fragment_gateup_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary
      )
      ;;
    m128n64_staged)
      fragment_gateup_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_secondary
      )
      fragment_gateup_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary
      )
      ;;
    m128n64_1cta)
      fragment_gateup_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_secondary
      )
      fragment_gateup_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary
      )
      ;;
    m128n32)
      fragment_gateup_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_secondary
      )
      fragment_gateup_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary
      )
      ;;
    m64n64)
      fragment_gateup_markers=(
        prefill_projection_span_mlp_k512_fragment_native_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_gateup_secondary
      )
      fragment_gateup_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary
        prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary
      )
      ;;
  esac
  for marker in "${fragment_gateup_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the ${mlp_k512_fragment_native_gateup_variant} Gate+Up variant: ${marker}" >&2
      exit 2
    fi
  done
  for marker in "${fragment_gateup_rejected_markers[@]}"; do
    if grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server contains the mutually exclusive fragment-native Gate+Up variant: ${marker}" >&2
      exit 2
    fi
  done
  declare -a fragment_down_markers=()
  declare -a fragment_down_rejected_markers=()
  case "${mlp_k512_fragment_native_down_variant}" in
    m128n256_1cta)
      fragment_down_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128n256_1cta_down
      )
      fragment_down_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_down
      )
      ;;
    m64n128)
      fragment_down_markers=(
        prefill_projection_span_mlp_k512_fragment_native_down
      )
      fragment_down_rejected_markers=(
        prefill_projection_span_mlp_k512_fragment_native_m128n256_1cta_down
      )
      ;;
  esac
  for marker in "${fragment_down_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the ${mlp_k512_fragment_native_down_variant} Down variant: ${marker}" >&2
      exit 2
    fi
  done
  for marker in "${fragment_down_rejected_markers[@]}"; do
    if grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server contains the mutually exclusive fragment-native Down variant: ${marker}" >&2
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
if [[ ("${mlp_k512_mode}" == 1 && "${attention_k256_mode}" == 0) ||
      "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  for bucket in "${buckets[@]}"; do
    if [[ "${bucket}" == p512 ]]; then
      echo "Prefill MLP K512 is not admissible for p512; select p1k, p2k, or p4k explicitly" >&2
      exit 2
    fi
  done
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
  printf 'pure_prefill_corpus bucket=%s sha256=%s status=%s path=%q\n' \
    "${bucket}" "${actual}" "${corpus_status}" "${corpus}"
done

# Remove inherited experiment selectors generically. The benchmark process gets
# production defaults plus exactly the selectors declared by the selected
# candidate bundle. Harness-only Q3X_EVAL_* variables are consumed before this
# point.
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
for selector in "${candidate_selectors[@]}"; do
  runtime_env+=("${selector}=1")
done

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
if [[ "${k512_mode}" == 1 ]]; then
  server_args+=(
    --prefill-attention-o-k512-payload \
      "${prefill_attention_o_k512_payload}"
    --prefill-attention-o-k512-policy \
      "${prefill_attention_o_k512_policy}"
    --prefill-attention-o-k512-receipt \
      "${prefill_attention_o_k512_receipt}"
  )
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  server_args+=(
    --prefill-mlp-k512-paired-gateup-canonical-down-payload \
      "${prefill_mlp_k512_paired_gateup_canonical_down_payload}"
    --prefill-mlp-k512-paired-gateup-canonical-down-policy \
      "${prefill_mlp_k512_paired_gateup_canonical_down_policy}"
    --prefill-mlp-k512-paired-gateup-canonical-down-receipt \
      "${prefill_mlp_k512_paired_gateup_canonical_down_receipt}"
  )
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  server_args+=(
    --prefill-mlp-k512-projection-major-gateup-canonical-down-payload \
      "${prefill_mlp_k512_projection_major_gateup_canonical_down_payload}"
    --prefill-mlp-k512-projection-major-gateup-canonical-down-policy \
      "${prefill_mlp_k512_projection_major_gateup_canonical_down_policy}"
    --prefill-mlp-k512-projection-major-gateup-canonical-down-receipt \
      "${prefill_mlp_k512_projection_major_gateup_canonical_down_receipt}"
  )
fi
if [[ "${mlp_k512_mode}" == 1 ]]; then
  server_args+=(
    --prefill-mlp-k512-payload "${prefill_mlp_k512_payload}"
    --prefill-mlp-k512-policy "${prefill_mlp_k512_policy}"
    --prefill-mlp-k512-receipt "${prefill_mlp_k512_receipt}"
  )
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  server_args+=(
    --prefill-mlp-k512-fragment-native-payload \
      "${prefill_mlp_k512_fragment_native_payload}"
    --prefill-mlp-k512-fragment-native-policy \
      "${prefill_mlp_k512_fragment_native_policy}"
    --prefill-mlp-k512-fragment-native-receipt \
      "${prefill_mlp_k512_fragment_native_receipt}"
  )
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  server_args+=(
    --prefill-mlp-factorized-lane-r1-payload \
      "${prefill_mlp_factorized_lane_r1_payload}"
    --prefill-mlp-factorized-lane-r1-policy \
      "${prefill_mlp_factorized_lane_r1_policy}"
    --prefill-mlp-factorized-lane-r1-receipt \
      "${prefill_mlp_factorized_lane_r1_receipt}"
  )
fi
declare -a profiler_prefix=()
if [[ -n "${nsys_output}" ]]; then
  profiler_prefix=(
    nsys profile
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --stats=false
    --capture-range=cudaProfilerApi --capture-range-end=stop
    --force-overwrite=true --output "${nsys_output}"
  )
  server_args+=(--profile-request-index "${profile_request_index}")
fi
gateup_alternating_expected_hits=
gateup_ldmatrix_pairfeed_expected_hits=
gateup_m128n128_projection_serial_expected_hits=
gateup_m128n64_same_cta_expected_hits=
gateup_m128n512_fused_quantize_expected_hits=
gateup_m128n512_paired_ldmatrix_expected_hits=
gateup_m64n128_register_pipeline_expected_hits=
gateup_m64n8_paired_warp_register_pipeline_expected_hits=
down_m128n128_ldmatrix_pairring_expected_hits=
down_m128n128_16warp_pairring_expected_hits=
mlp_k256_pairfeed_package_expected_hits=
factorized_lane_r1_package_expected_hits=
attention_k256_incumbent_expected_launch_hits=
attention_k256_incumbent_expected_logical_hits=
attention_k256_a_exchange_b4_expected_launch_hits=
attention_k256_a_exchange_b4_expected_logical_hits=
gdn_chunk64_native_expected_launch_formula=
gdn_chunk64_native_expected_logical_formula=
gdn_prompt_span_macro_expected_launch_formula=
gdn_prompt_span_macro_expected_logical_formula=
if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
  gdn_chunk64_native_expected_launch_formula='48*ceil(prompt_tokens/512)'
  gdn_chunk64_native_expected_logical_formula='48*prompt_tokens'
  gdn_prompt_span_macro_expected_launch_formula=0
  gdn_prompt_span_macro_expected_logical_formula=0
  if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
    gdn_chunk64_native_expected_launch_formula=0
    gdn_chunk64_native_expected_logical_formula=0
    gdn_prompt_span_macro_expected_launch_formula=48
    gdn_prompt_span_macro_expected_logical_formula='48*prompt_tokens'
  fi
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  attention_k256_incumbent_expected_launch_hits=128
  attention_k256_incumbent_expected_logical_hits=208
  attention_k256_a_exchange_b4_expected_launch_hits=0
  attention_k256_a_exchange_b4_expected_logical_hits=0
  if [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
    attention_k256_incumbent_expected_launch_hits=0
    attention_k256_incumbent_expected_logical_hits=0
    attention_k256_a_exchange_b4_expected_launch_hits=128
    attention_k256_a_exchange_b4_expected_logical_hits=208
  fi
fi
if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n64_same_cta_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=0
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=0
  mlp_k256_pairfeed_package_expected_hits=64
elif [[ "${factorized_r1_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n64_same_cta_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=0
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=0
  mlp_k256_pairfeed_package_expected_hits=0
  factorized_lane_r1_package_expected_hits=64
elif [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=64
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=64
elif [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=64
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=64
elif [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n64_same_cta_expected_hits=64
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=0
elif [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=64
elif [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=0
  gateup_m128n128_projection_serial_expected_hits=64
  gateup_m128n512_fused_quantize_expected_hits=0
elif [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=64
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=64
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
elif [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=64
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n64_same_cta_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=0
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=64
elif [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=64
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n64_same_cta_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
  gateup_m128n512_paired_ldmatrix_expected_hits=0
  gateup_m64n128_register_pipeline_expected_hits=0
  gateup_m64n8_paired_warp_register_pipeline_expected_hits=0
elif [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=0
  gateup_ldmatrix_pairfeed_expected_hits=64
  gateup_m128n128_projection_serial_expected_hits=0
  gateup_m128n512_fused_quantize_expected_hits=0
elif [[ "${mlp_k512_edge_m64n128_k256_alternating_mode}" == 1 ]]; then
  gateup_alternating_expected_hits=64
  if [[ "${mode}" == "${ldmatrix_pairfeed_baseline_mode}" ]]; then
    gateup_ldmatrix_pairfeed_expected_hits=0
  fi
elif [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ]]; then
  gateup_alternating_expected_hits=0
fi
if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
  down_m128n128_ldmatrix_pairring_expected_hits=64
fi
if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
  down_m128n128_ldmatrix_pairring_expected_hits=0
  down_m128n128_16warp_pairring_expected_hits=64
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    gateup_m128n512_paired_ldmatrix_expected_hits=0
  else
    gateup_m128n512_paired_ldmatrix_expected_hits=64
  fi
  if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    down_m128n128_ldmatrix_pairring_expected_hits=0
    down_m128n128_16warp_pairring_expected_hits=64
  elif [[ "${mlp_k512_hybrid_down_pairring_mode}" == 1 ]]; then
    down_m128n128_ldmatrix_pairring_expected_hits=64
  else
    down_m128n128_ldmatrix_pairring_expected_hits=0
  fi
fi

printf 'pure_prefill_matrix mode=%s dry_run=%s sanitized_experiment_env=%s selector_count=%s eval_number=%s\n' \
  "${mode}" "${dry_run}" "${sanitized}" "${#candidate_selectors[@]}" \
  "${eval_number}"
printf 'evalscope_request_plan measured=%s warmup=1 result_leaf=parallel_1_number_%s\n' \
  "${eval_number}" "${eval_number}"
server_elf_sha256=$(sha256sum "${server}" | awk '{print $1}')
printf 'server_metadata elf_sha256=%s evalscope_version=1.9.1\n' \
  "${server_elf_sha256}"
if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
  printf 'gdn_prompt_span_accounting_metadata route=%s native_launch_formula=%s native_logical_token_formula=%s macro_launch_formula=%s macro_logical_token_formula=%s\n' \
    "$([[ "${gdn_prompt_span_macro_mode}" == 1 ]] && \
       printf prompt-span-macro || printf native-c512)" \
    "${gdn_chunk64_native_expected_launch_formula}" \
    "${gdn_chunk64_native_expected_logical_formula}" \
    "${gdn_prompt_span_macro_expected_launch_formula}" \
    "${gdn_prompt_span_macro_expected_logical_formula}"
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  printf 'attention_k256_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s manifest_sha256=%s incumbent_expected_launch_hits=%s incumbent_expected_logical_projections=%s a_exchange_b4_expected_launch_hits=%s a_exchange_b4_expected_logical_projections=%s\n' \
    "${attention_k256_layout}" "${attention_k256_payload_bytes}" \
    "${attention_k256_payload_sha256}" \
    "${attention_k256_policy_sha256}" \
    "${attention_k256_receipt_sha256}" \
    "${attention_k256_manifest_sha256}" \
    "${attention_k256_incumbent_expected_launch_hits}" \
    "${attention_k256_incumbent_expected_logical_hits}" \
    "${attention_k256_a_exchange_b4_expected_launch_hits}" \
    "${attention_k256_a_exchange_b4_expected_logical_hits}"
  printf 'attention_k256_mlp_binding_metadata layout=%s payload_bytes=%s manifest_sha256=%s policy_sha256=%s payload_sha256=%s\n' \
    "${attention_k256_overlay_layout}" \
    "${attention_k256_overlay_payload_bytes}" \
    "${attention_k256_overlay_manifest_sha256}" \
    "${attention_k256_overlay_policy_sha256}" \
    "${attention_k256_overlay_payload_sha256}"
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  printf 'factorized_r1_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s manifest_sha256=%s base_receipt_sha256=%s production_residency_eligible=false quality_production_eligible=false performance_upper_bound_only=true\n' \
    "${factorized_r1_layout}" "${factorized_r1_payload_bytes}" \
    "${factorized_r1_payload_sha256}" \
    "${factorized_r1_policy_sha256}" \
    "${factorized_r1_receipt_sha256}" \
    "${factorized_r1_manifest_sha256}" \
    "${factorized_r1_base_receipt_sha256}"
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  printf 'fragment_native_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s gateup_variant=%s down_variant=%s\n' \
    "${mlp_k512_fragment_native_layout}" \
    "${mlp_k512_fragment_native_payload_bytes}" \
    "${mlp_k512_fragment_native_payload_sha256}" \
    "${mlp_k512_fragment_native_policy_sha256}" \
    "${mlp_k512_fragment_native_receipt_sha256}" \
    "${mlp_k512_fragment_native_gateup_variant}" \
    "${mlp_k512_fragment_native_down_variant}"
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  hybrid_down_variant=canonical
  if [[ "${mlp_k512_hybrid_down_pairring_mode}" == 1 ]]; then
    hybrid_down_variant=pairring
  elif [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    hybrid_down_variant=16warp-pairring
  fi
  printf 'hybrid_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s manifest_sha256=%s source_v1_receipt_sha256=%s source_v1_manifest_sha256=%s source_v1_payload_sha256=%s down_variant=%s\n' \
    "${mlp_k512_hybrid_layout}" \
    "${mlp_k512_hybrid_payload_bytes}" \
    "${mlp_k512_hybrid_payload_sha256}" \
    "${mlp_k512_hybrid_policy_sha256}" \
    "${mlp_k512_hybrid_receipt_sha256}" \
    "${mlp_k512_hybrid_manifest_sha256}" \
    "${mlp_k512_hybrid_source_v1_receipt_sha256}" \
    "${mlp_k512_hybrid_source_v1_manifest_sha256}" \
    "${mlp_k512_hybrid_source_v1_payload_sha256}" \
    "${hybrid_down_variant}"
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  printf 'projection_major_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s manifest_sha256=%s source_v1_receipt_sha256=%s source_v1_manifest_sha256=%s source_v1_payload_sha256=%s\n' \
    "${mlp_k512_projection_major_layout}" \
    "${mlp_k512_projection_major_payload_bytes}" \
    "${mlp_k512_projection_major_payload_sha256}" \
    "${mlp_k512_projection_major_policy_sha256}" \
    "${mlp_k512_projection_major_receipt_sha256}" \
    "${mlp_k512_projection_major_manifest_sha256}" \
    "${mlp_k512_projection_major_source_v1_receipt_sha256}" \
    "${mlp_k512_projection_major_source_v1_manifest_sha256}" \
    "${mlp_k512_projection_major_source_v1_payload_sha256}"
fi
printf 'profile_metadata enabled=%s request_index=%s nsys_output=%q trace=cuda,nvtx capture_range=cudaProfilerApi\n' \
  "$([[ -n "${nsys_output}" ]] && printf 1 || printf 0)" \
  "${profile_request_index}" "${nsys_output}"
printf 'selector_metadata mode=%s selector_count=%s' \
  "${mode}" "${#candidate_selectors[@]}"
for selector in "${candidate_selectors[@]}"; do
  printf ' %s' "${selector}"
done
printf '\n'
if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  printf 'prefill_mlp_k256_implementation=m128n256_pairfeed_package source_publication=base_a4_k256 overlay=none\n'
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  printf 'prefill_mlp_implementation=factorized_lane_r1 experiment_baseline_mode=%s publication=authenticated_upper_bound_only runtime_counter=factorized_lane_r1_package_launch_hits\n' \
    "${k256_pairfeed_package_mode}"
fi
if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
  printf 'prefill_mlp_k512_implementation=shape_separated_marlin\n'
  printf 'prefill_mlp_k512_gateup_implementation=m64n256_marlin_k64_b3\n'
  printf 'prefill_mlp_k512_down_implementation=m64n256_16warp_pairring\n'
  printf 'prefill_mlp_k512_parent_implementations=m64n128_k256_ldmatrix_pairfeed,m128n128_16warp_pairring\n'
fi
if [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
  printf 'prefill_mlp_k512_gateup_implementation=m32n512_owner_k128_b4 parent_implementation=m64n128_k256_ldmatrix_pairfeed\n'
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selector=Q3X_RUN_A4W4_MLP_K256_M128N256_PAIRFEED_PACKAGE_ADMISSION added_selector=Q3X_RUN_A4W4_FACTORIZED_LANE_R1_ADMISSION retained_selectors=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION,Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION,Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION,Q3X_FULL_ATTENTION_FLASHINFER_DIRECT,Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION,Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION,Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION publication_delta=replace_k256_mlp_with_authenticated_factorized_r1\n' \
    "${k256_pairfeed_package_mode}"
elif [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selectors=Q3X_RUN_A4W4_MLP_K512_ADMISSION,Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION added_selector=Q3X_RUN_A4W4_MLP_K256_M128N256_PAIRFEED_PACKAGE_ADMISSION retained_selectors=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION,Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION,Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION,Q3X_FULL_ATTENTION_FLASHINFER_DIRECT,Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION,Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION,Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION publication_delta=remove_k512_overlay_use_base_a4_k256_only\n' \
    "${ldmatrix_pairfeed_candidate_mode}"
elif [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selectors=Q3X_RUN_A4W4_MLP_K512_ADMISSION,Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION added_selectors=Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION,Q3X_RUN_A4W4_GATEUP_K512_M64N8_PAIRED_WARP_REGISTER_PIPELINE_ADMISSION retained_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n' \
    "${paired_warp_baseline_mode}"
elif [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selectors=Q3X_RUN_A4W4_MLP_K512_ADMISSION,Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION added_selectors=Q3X_RUN_A4W4_MLP_K512_PROJECTION_MAJOR_GATEUP_CANONICAL_DOWN_ADMISSION,Q3X_RUN_A4W4_GATEUP_K512_M64N128_REGISTER_PIPELINE_ADMISSION retained_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n' \
    "${projection_major_baseline_mode}"
elif [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s retained_selector=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION added_selector=Q3X_RUN_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION\n' \
    "${gdn_prompt_span_baseline_mode}"
elif [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION\n' \
    "${same_cta_baseline_mode}"
elif [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION\n' \
    "${fused_quantize_baseline_mode}"
elif [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION\n' \
    "${projection_serial_baseline_mode}"
elif [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s retained_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION added_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N128_A_EXCHANGE_B3_ADMISSION\n' \
    "${attention_b3_baseline_mode}"
elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s retained_selectors=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION added_selectors=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_L2_MACRO4X4_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_L2_MACRO4X4_ADMISSION\n' \
    "${ldmatrix_pairfeed_candidate_mode}"
elif [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s retained_selectors=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION added_selector=Q3X_RUN_A4W4_MLP_K512_SHAPE_SEPARATED_MARLIN_PACKAGE_ADMISSION replaced_runtime_stages=%s,%s\n' \
    "${shape_separated_marlin_baseline_mode}" \
    "${ldmatrix_pairfeed_marker}" \
    "${pairring_down_16warp_marker}"
elif [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s retained_selectors=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION added_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION\n' \
    "${m32n512_owner_baseline_mode}"
elif [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=%s removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION added_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n' \
    "${ldmatrix_pairfeed_baseline_mode}"
elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256 removed_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION added_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION\n'
elif [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
  printf 'candidate_delta baseline_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256 removed_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION added_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n'
fi
if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  printf 'stage_contract required=%s,%s,%s excluded=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_ldmatrix_pairfeed,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_down expected_request_launch_hits=package:64,gate_alternating:0,gate_pairfeed:0,gate_projection_serial:0,gate_same_cta:0,gate_fused_quantize:0,gate_paired_ldmatrix:0,gate_projection_major:0,gate_paired_warp:0,down_ldmatrix:0,down_16warp:0\n' \
    "${k256_pairfeed_input_marker}" \
    "${k256_pairfeed_gateup_marker}" \
    "${k256_pairfeed_down_marker}"
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  printf 'factorized_r1_layout=%s\n' "${factorized_r1_layout}"
  printf 'factorized_r1_payload_bytes=%s\n' "${factorized_r1_payload_bytes}"
  printf 'factorized_r1_payload_sha256=%s\n' "${factorized_r1_payload_sha256}"
  printf 'factorized_r1_policy_sha256=%s\n' "${factorized_r1_policy_sha256}"
  printf 'factorized_r1_receipt_sha256=%s\n' "${factorized_r1_receipt_sha256}"
  printf 'factorized_r1_manifest_sha256=%s\n' "${factorized_r1_manifest_sha256}"
  printf 'factorized_r1_base_receipt_sha256=%s\n' "${factorized_r1_base_receipt_sha256}"
  printf 'factorized_r1_min_prompt_tokens=480\n'
  printf 'factorized_r1_production_residency_eligible=false\n'
  printf 'factorized_r1_quality_production_eligible=false\n'
  printf 'factorized_r1_performance_upper_bound_only=true\n'
  printf 'stage_contract required=prefill_projection_span_factorized_lane_r1_input_quantize,prefill_projection_span_factorized_lane_r1_gateup,prefill_projection_span_factorized_lane_r1_product_quantize,prefill_projection_span_factorized_lane_r1_down expected_request_launch_hits=package:64,k256_package:0,gate_alternatives:0,down_alternatives:0 mtp=false\n'
fi
if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
  printf 'stage_contract required=%s,%s,%s excluded=%s,%s,%s,prefill_projection_span_mlp_k512_paired_gateup_canonical_down_down,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_ldmatrix_pairfeed expected_request_launch_hits=old_gate:0,new_gate:64,down_incumbent:0,down_candidate:64\n' \
    "${hybrid_input_marker}" \
    "${paired_warp_gate_marker}" \
    "${paired_warp_down_16warp_marker}" \
    "${hybrid_gate_marker}" \
    "${projection_major_gate_marker}" \
    "${projection_major_down_16warp_marker}"
fi
if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
  if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
    printf 'stage_contract required=%s retained=gdn-token-parallel-conv,gdn-native-preprocess expected_request_launch_hits=native:0,macro:48 expected_request_logical_tokens=native:0,macro:48*prompt_tokens\n' \
      "${gdn_prompt_span_macro_marker}"
  else
    printf 'stage_contract retained=gdn-native-c512 expected_request_launch_hits=native:48*ceil(prompt_tokens/512),macro:0 expected_request_logical_tokens=native:48*prompt_tokens,macro:0\n'
  fi
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
    printf 'stage_contract required=%s excluded=%s,%s,%s,prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix expected_request_launch_hits=attention_incumbent:0,attention_candidate:128 expected_request_logical_projections=attention_incumbent:0,attention_candidate:208\n' \
      "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b3_markers[*]}")" \
      "$(IFS=,; printf '%s' "${attention_k256_incumbent_markers[*]}")" \
      "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_markers[*]}")" \
      "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_l2_macro4x4_markers[*]}")"
  elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
    printf 'stage_contract required=%s excluded=%s,%s,prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix expected_request_launch_hits=attention_incumbent:0,attention_candidate:128 expected_request_logical_projections=attention_incumbent:0,attention_candidate:208\n' \
      "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_l2_macro4x4_markers[*]}")" \
      "$(IFS=,; printf '%s' "${attention_k256_incumbent_markers[*]}")" \
      "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_markers[*]}")"
  elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
    printf 'stage_contract required=prefill_projection_span_linear_qkv_z_k256_m128n256_a_exchange_b4,prefill_projection_span_linear_output_k256_m128n256_a_exchange_b4,prefill_projection_span_full_q_k_v_k256_m128n256_a_exchange_b4,prefill_projection_span_full_output_k256_m128n256_a_exchange_b4 excluded=prefill_projection_span_linear_qkv_z_k256_m128n256,prefill_projection_span_linear_output_k256_m128n256,prefill_projection_span_full_q_k_v_k256_m128n256,prefill_projection_span_full_output_k256_m128n256,prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix expected_request_launch_hits=attention_incumbent:0,attention_candidate:128 expected_request_logical_projections=attention_incumbent:0,attention_candidate:208\n'
  else
    printf 'stage_contract required=prefill_projection_span_linear_qkv_z_k256_m128n256,prefill_projection_span_linear_output_k256_m128n256,prefill_projection_span_full_q_k_v_k256_m128n256,prefill_projection_span_full_output_k256_m128n256 excluded=prefill_projection_span_linear_qkv_z_k256_m128n256_a_exchange_b4,prefill_projection_span_linear_output_k256_m128n256_a_exchange_b4,prefill_projection_span_full_q_k_v_k256_m128n256_a_exchange_b4,prefill_projection_span_full_output_k256_m128n256_a_exchange_b4,prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix expected_request_launch_hits=attention_incumbent:128,attention_candidate:0 expected_request_logical_projections=attention_incumbent:208,attention_candidate:0\n'
  fi
fi
if [[ "${mlp_k512_edge_mode}" == 1 &&
      "${mlp_k512_down_m16n64_v2_mode}" == 0 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge excluded=prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize retained=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
fi
if [[ "${mlp_k512_edge_m64n128_k256_alternating_mode}" == 1 ]]; then
  if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
    printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring excluded=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate:64,down_incumbent:0,down_candidate:64\n'
  elif [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
    printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring excluded=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2 retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate:64,down:64\n'
  else
    printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating excluded=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down_m16n64_v2 retained=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down expected_request_launch_hits=64\n'
  fi
fi
if [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 ]]; then
  if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
    printf 'stage_contract required=%s,%s excluded=%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_primary,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_secondary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary,prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix,prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline,prefill_projection_span_mlp_k512_gateup_m64n8_paired_warp_register_pipeline,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize,parent_selectors:Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION+Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION expected_request_launch_hits=gate_incumbent:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
      "${shape_separated_marlin_gate_marker}" \
      "${shape_separated_marlin_down_marker}" \
      "${ldmatrix_pairfeed_marker}" \
      "${pairring_down_16warp_marker}"
  elif [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
    printf 'stage_contract required=%s,%s excluded=%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_primary,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_secondary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary,prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix,prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline,prefill_projection_span_mlp_k512_gateup_m64n8_paired_warp_register_pipeline,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize,parent_selector:Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION expected_request_launch_hits=gate_incumbent:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
      "${m32n512_owner_marker}" \
      "${pairring_down_16warp_marker}" \
      "${ldmatrix_pairfeed_marker}"
  else
    printf 'stage_contract required=%s,%s excluded=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate_incumbent:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
      "${ldmatrix_pairfeed_marker}" \
      "${pairring_down_16warp_marker}"
  fi
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  printf 'stage_contract required=%s,%s,%s excluded=%s,prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring expected_request_launch_hits=gate_incumbent:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
    "${projection_major_input_marker}" \
    "${projection_major_gate_marker}" \
    "${projection_major_down_16warp_marker}" \
    "${ldmatrix_pairfeed_marker}"
fi
if [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
  printf 'stage_contract required=%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring excluded=%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate_alternating:0,gate_pairfeed:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
    "${projection_serial_primary_marker}" \
    "${projection_serial_secondary_marker}" \
    "${projection_serial_quantize_marker}" \
    "${ldmatrix_pairfeed_marker}"
fi
if [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
  printf 'stage_contract required=%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring excluded=%s,%s,%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix,%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate_alternating:0,gate_pairfeed:0,gate_projection_serial:0,gate_same_cta:64,gate_fused_quantize:0,gate_paired_ldmatrix:0,gate_projection_major:0,gate_paired_warp:0,down_incumbent:0,down_candidate:64\n' \
    "${same_cta_primary_marker}" \
    "${same_cta_secondary_marker}" \
    "${projection_serial_quantize_marker}" \
    "${ldmatrix_pairfeed_marker}" \
    "${projection_serial_primary_marker}" \
    "${projection_serial_secondary_marker}" \
    "${fused_quantize_marker}" \
    "${projection_major_gate_marker}" \
    "${paired_warp_gate_marker}"
fi
if [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]]; then
  printf 'stage_contract required=%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring excluded=%s,%s,%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring retained=prefill_projection_span_mlp_k512_input_quantize expected_request_launch_hits=gate_alternating:0,gate_pairfeed:0,gate_projection_serial:0,gate_candidate:64,down_incumbent:0,down_candidate:64\n' \
    "${fused_quantize_marker}" \
    "${ldmatrix_pairfeed_marker}" \
    "${projection_serial_primary_marker}" \
    "${projection_serial_secondary_marker}" \
    "${projection_serial_quantize_marker}"
fi
if [[ "${mlp_k512_edge_m128n64_mode}" == 1 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge_m128n64 excluded=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down_m16n64_v2 retained=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
fi
if [[ "${mlp_k512_down_m16n64_v2_mode}" == 1 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_down_m16n64_v2 excluded=prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize retained=prefill_projection_span_mlp_k512_input_quantize\n'
fi
hybrid_old_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_ldmatrix_pairfeed,prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_fragment_native_input_quantize,prefill_projection_span_mlp_k512_fragment_native_gateup_primary,prefill_projection_span_mlp_k512_fragment_native_gateup_secondary,prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary,prefill_projection_span_mlp_k512_fragment_native_m128_gateup_secondary,prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary,prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_secondary,prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary,prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_secondary,prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary,prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_secondary,prefill_projection_span_mlp_k512_fragment_native_product_quantize,prefill_projection_span_mlp_k512_fragment_native_down,prefill_projection_span_mlp_k512_fragment_native_m128n256_1cta_down
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    printf 'stage_contract required=%s,%s,%s excluded=%s,%s,%s expected_request_launch_hits=old_gate:0,new_gate:64,down_incumbent:0,down_candidate:64\n' \
      "${hybrid_input_marker}" "${paired_warp_gate_marker}" \
      "${paired_warp_down_16warp_marker}" "${hybrid_gate_marker}" \
      "${hybrid_canonical_down_marker}" "${hybrid_pairring_down_marker}"
  elif [[ "${mlp_k512_hybrid_down_pairring_mode}" == 1 ]]; then
    printf 'stage_contract required=%s,%s,%s excluded=%s,%s expected_request_launch_hits=gate:64,down:64\n' \
      "${hybrid_input_marker}" "${hybrid_gate_marker}" \
      "${hybrid_pairring_down_marker}" \
      "${hybrid_canonical_down_marker}" "${hybrid_old_runtime_stages}"
  else
    printf 'stage_contract required=%s,%s retained=%s excluded=%s,%s expected_request_launch_hits=gate:64,down:0\n' \
      "${hybrid_input_marker}" "${hybrid_gate_marker}" \
      "${hybrid_canonical_down_marker}" \
      "${hybrid_pairring_down_marker}" "${hybrid_old_runtime_stages}"
  fi
fi
printf 'server_startup_command'
printf ' %q' "${runtime_env[@]}" "${profiler_prefix[@]}" "${server_args[@]}"
printf '\n'
printf 'startup_contract required=prefill_a4_authenticated_400_of_400,optimized_prefill_disabled_0'
if [[ "${k512_mode}" == 1 ]]; then
  printf ',prefill_attention_o_k512_authenticated_64_of_64,prefill_attention_o_k512_payload_sha256'
fi
if [[ "${mlp_k512_mode}" == 1 ]]; then
  printf ',prefill_mlp_k512_authenticated_192_of_192,prefill_mlp_k512_payload_sha256'
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  printf ',prefill_a4_k256_authenticated_400_of_400,attention_k256_incumbent_expected_launch_hits_%s,attention_k256_incumbent_expected_logical_projections_%s,attention_k256_a_exchange_b4_expected_launch_hits_%s,attention_k256_a_exchange_b4_expected_logical_projections_%s' \
    "${attention_k256_incumbent_expected_launch_hits}" \
    "${attention_k256_incumbent_expected_logical_hits}" \
    "${attention_k256_a_exchange_b4_expected_launch_hits}" \
    "${attention_k256_a_exchange_b4_expected_logical_hits}"
fi
if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
  printf ',%s,%s,%s,base_a4_k256_only,k512_mlp_overlay_forbidden,mlp_k256_m128n256_pairfeed_package_launch_hits_64_per_request,all_k512_mlp_launch_hits_0_per_request' \
    "${k256_pairfeed_input_marker}" \
    "${k256_pairfeed_gateup_marker}" \
    "${k256_pairfeed_down_marker}"
fi
if [[ "${mlp_k512_edge_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_gateup_down_edge,old_gateup_split_stages_excluded'
fi
if [[ "${mlp_k512_edge_m64n128_k256_alternating_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_64_per_request'
  if [[ "${mode}" == "${ldmatrix_pairfeed_baseline_mode}" ]]; then
    printf ',gateup_ldmatrix_pairfeed_launch_hits_0_per_request'
  fi
fi
if [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 ]]; then
  if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
    printf ',%s,%s,pairfeed_parent_selector_retained,down16_parent_selector_retained,parent_gateup_stage_excluded,parent_down_stage_excluded,gateup_sibling_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_64_per_request,gateup_m128n128_projection_serial_launch_hits_0_per_request,gateup_m128n64_same_cta_launch_hits_0_per_request,gateup_m128n512_fused_quantize_launch_hits_0_per_request,gateup_m128n512_paired_ldmatrix_launch_hits_0_per_request,gateup_m64n128_register_pipeline_launch_hits_0_per_request,gateup_m64n8_paired_warp_register_pipeline_launch_hits_0_per_request,down_m128n128_ldmatrix_pairring_launch_hits_0_per_request,down_m128n128_16warp_pairring_launch_hits_64_per_request' \
      "${shape_separated_marlin_gate_marker}" \
      "${shape_separated_marlin_down_marker}"
  elif [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
    printf ',%s,pairfeed_parent_selector_retained,pairfeed_gateup_stage_excluded,gateup_sibling_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_64_per_request,gateup_m128n128_projection_serial_launch_hits_0_per_request,gateup_m128n64_same_cta_launch_hits_0_per_request,gateup_m128n512_fused_quantize_launch_hits_0_per_request,gateup_m128n512_paired_ldmatrix_launch_hits_0_per_request,gateup_m64n128_register_pipeline_launch_hits_0_per_request,gateup_m64n8_paired_warp_register_pipeline_launch_hits_0_per_request' \
      "${m32n512_owner_marker}"
  else
    printf ',%s,alternating_gateup_stage_excluded,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_64_per_request,gateup_m128n128_projection_serial_launch_hits_0_per_request,gateup_m128n512_fused_quantize_launch_hits_0_per_request' \
      "${ldmatrix_pairfeed_marker}"
  fi
fi
if [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
  printf ',%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,pairfeed_gateup_stage_excluded,alternating_gateup_stage_excluded,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_0_per_request,gateup_m128n128_projection_serial_launch_hits_64_per_request,gateup_m128n512_fused_quantize_launch_hits_0_per_request' \
    "${projection_serial_primary_marker}" \
    "${projection_serial_secondary_marker}" \
    "${projection_serial_quantize_marker}"
fi
if [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
  printf ',%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,pairfeed_gateup_stage_excluded,projection_serial_gateup_stages_excluded,fused_quantize_gateup_stage_excluded,alternating_gateup_stage_excluded,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_0_per_request,gateup_m128n128_projection_serial_launch_hits_0_per_request,gateup_m128n64_same_cta_launch_hits_64_per_request,gateup_m128n512_fused_quantize_launch_hits_0_per_request,gateup_m128n512_paired_ldmatrix_launch_hits_0_per_request,gateup_m64n128_register_pipeline_launch_hits_0_per_request,gateup_m64n8_paired_warp_register_pipeline_launch_hits_0_per_request' \
    "${same_cta_primary_marker}" \
    "${same_cta_secondary_marker}" \
    "${projection_serial_quantize_marker}"
fi
if [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]]; then
  printf ',%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,pairfeed_gateup_stage_excluded,projection_serial_gateup_stages_excluded,standalone_product_quantize_stage_excluded,alternating_gateup_stage_excluded,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded,gateup_alternating_launch_hits_0_per_request,gateup_ldmatrix_pairfeed_launch_hits_0_per_request,gateup_m128n128_projection_serial_launch_hits_0_per_request,gateup_m128n512_fused_quantize_launch_hits_64_per_request' \
    "${fused_quantize_marker}"
fi
if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring,v1_down_stage_excluded,down_m128n128_ldmatrix_pairring_launch_hits_64_per_request'
fi
if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,v1_down_stage_excluded,incumbent_pairring_stage_excluded,down_m128n128_ldmatrix_pairring_launch_hits_0_per_request,down_m128n128_16warp_pairring_launch_hits_64_per_request'
fi
if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
  printf ',gdn_request_accounting_dynamic_by_prompt_tokens'
  if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
    printf ',%s,gdn_chunk64_native_launch_hits_0_per_request,gdn_prompt_span_macro_launch_hits_48_per_request' \
      "${gdn_prompt_span_macro_marker}"
  else
    printf ',gdn_chunk64_native_launch_hits_48_times_ceil_prompt_tokens_div_512,gdn_prompt_span_macro_launch_hits_0_per_request'
  fi
fi
if [[ "${mlp_k512_edge_m128n64_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,legacy_edge_stage_excluded,old_gateup_split_stages_excluded,down_m16n64_v2_stage_excluded'
fi
if [[ "${mlp_k512_down_m16n64_v2_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_down_m16n64_v2,v1_down_stage_excluded'
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  printf ',prefill_mlp_k512_fragment_native_authenticated_64_of_64,prefill_mlp_k512_fragment_native_gateup_variant_%s,prefill_mlp_k512_fragment_native_down_variant_%s,prefill_mlp_k512_fragment_native_layout,prefill_mlp_k512_fragment_native_payload_sha256,prefill_mlp_k512_fragment_native_policy_sha256,prefill_mlp_k512_fragment_native_receipt_sha256' \
    "${mlp_k512_fragment_native_gateup_variant}" \
    "${mlp_k512_fragment_native_down_variant}"
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  printf ',prefill_mlp_k512_paired_gateup_canonical_down_authenticated_64_of_64,prefill_mlp_k512_paired_gateup_canonical_down_layout,prefill_mlp_k512_paired_gateup_canonical_down_payload_sha256,prefill_mlp_k512_paired_gateup_canonical_down_policy_sha256,prefill_mlp_k512_paired_gateup_canonical_down_receipt_sha256,prefill_mlp_k512_paired_gateup_canonical_down_complete_sha_chain,gateup_m128n512_paired_ldmatrix_launch_hits_%s_per_request,down_m128n128_ldmatrix_pairring_launch_hits_%s_per_request' \
    "${gateup_m128n512_paired_ldmatrix_expected_hits}" \
    "${down_m128n128_ldmatrix_pairring_expected_hits}"
  if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
    printf ',%s,%s,gateup_m64n8_paired_warp_register_pipeline_launch_hits_64_per_request,down_m128n128_16warp_pairring_launch_hits_64_per_request' \
      "${paired_warp_gate_marker}" "${paired_warp_down_16warp_marker}"
  fi
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  printf ',prefill_mlp_k512_projection_major_gateup_canonical_down_authenticated_64_of_64,prefill_mlp_k512_projection_major_gateup_canonical_down_layout,prefill_mlp_k512_projection_major_gateup_canonical_down_complete_sha_chain,%s,%s,%s,gateup_ldmatrix_pairfeed_launch_hits_0_per_request,gateup_m64n128_register_pipeline_launch_hits_64_per_request,down_m128n128_ldmatrix_pairring_launch_hits_0_per_request,down_m128n128_16warp_pairring_launch_hits_64_per_request' \
    "${projection_major_input_marker}" \
    "${projection_major_gate_marker}" \
    "${projection_major_down_16warp_marker}"
fi
printf '\n'

if [[ "${dry_run}" == 1 ]]; then
  echo "dry_run_complete=1 performance_evidence=0 startup_contract_check=deferred"
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v uvx >/dev/null || { echo "uvx is required" >&2; exit 2; }
mkdir -p "${output_root}"
{
  printf 'mode=%s\n' "${mode}"
  printf 'server_elf_sha256=%s\n' "${server_elf_sha256}"
  printf 'evalscope_version=1.9.1\n'
  printf 'evalscope_measured_requests=%s\n' "${eval_number}"
  printf 'evalscope_warmup_requests=1\n'
  printf 'evalscope_result_leaf=parallel_1_number_%s\n' "${eval_number}"
  printf 'profile_request_index=%s\n' "${profile_request_index}"
  printf 'nsys_output=%s\n' "${nsys_output}"
  printf 'selectors='
  printf '%s ' "${candidate_selectors[@]}"
  printf '\n'
  if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
    printf 'prefill_mlp_k256_publication=base_a4_k256_only\n'
    printf 'prefill_mlp_k256_k512_overlay=forbidden\n'
    printf 'required_runtime_stages=%s,%s,%s\n' \
      "${k256_pairfeed_input_marker}" \
      "${k256_pairfeed_gateup_marker}" \
      "${k256_pairfeed_down_marker}"
    printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_ldmatrix_pairfeed,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_down\n'
    printf 'mlp_k256_m128n256_pairfeed_package_expected_launch_hits_per_request=64\n'
    printf 'all_k512_mlp_expected_launch_hits_per_request=0\n'
  fi
  if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
    printf 'gdn_prompt_span_route=%s\n' \
      "$([[ "${gdn_prompt_span_macro_mode}" == 1 ]] && \
         printf prompt-span-macro || printf native-c512)"
    printf 'gdn_chunk64_native_expected_launch_formula=%s\n' \
      "${gdn_chunk64_native_expected_launch_formula}"
    printf 'gdn_chunk64_native_expected_logical_token_formula=%s\n' \
      "${gdn_chunk64_native_expected_logical_formula}"
    printf 'gdn_prompt_span_macro_expected_launch_formula=%s\n' \
      "${gdn_prompt_span_macro_expected_launch_formula}"
    printf 'gdn_prompt_span_macro_expected_logical_token_formula=%s\n' \
      "${gdn_prompt_span_macro_expected_logical_formula}"
    if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
      printf 'experiment_baseline_mode=%s\n' \
        "${gdn_prompt_span_baseline_mode}"
      printf 'experiment_retained_selector=Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION\n'
      printf 'experiment_added_selector=Q3X_RUN_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION\n'
      printf 'required_runtime_stage=%s\n' \
        "${gdn_prompt_span_macro_marker}"
    fi
  fi
  if [[ "${attention_k256_mode}" == 1 ]]; then
    printf 'prefill_a4_k256_layout=%s\n' "${attention_k256_layout}"
    printf 'prefill_a4_k256_payload_bytes=%s\n' \
      "${attention_k256_payload_bytes}"
    printf 'prefill_a4_k256_payload_sha256=%s\n' \
      "${attention_k256_payload_sha256}"
    printf 'prefill_a4_k256_policy_sha256=%s\n' \
      "${attention_k256_policy_sha256}"
    printf 'prefill_a4_k256_receipt_sha256=%s\n' \
      "${attention_k256_receipt_sha256}"
    printf 'prefill_a4_k256_manifest_sha256=%s\n' \
      "${attention_k256_manifest_sha256}"
    if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
      printf 'prefill_attention_k256_implementation=a_exchange_b3_m128n128\n'
    elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
      printf 'prefill_attention_k256_implementation=a_exchange_b4_l2_macro4x4\n'
    elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
      printf 'prefill_attention_k256_implementation=a_exchange_b4\n'
    else
      printf 'prefill_attention_k256_implementation=incumbent\n'
    fi
    printf 'prefill_attention_k256_incumbent_expected_launch_hits=%s\n' \
      "${attention_k256_incumbent_expected_launch_hits}"
    printf 'prefill_attention_k256_incumbent_expected_logical_projections=%s\n' \
      "${attention_k256_incumbent_expected_logical_hits}"
    printf 'prefill_attention_k256_a_exchange_b4_expected_launch_hits=%s\n' \
      "${attention_k256_a_exchange_b4_expected_launch_hits}"
    printf 'prefill_attention_k256_a_exchange_b4_expected_logical_projections=%s\n' \
      "${attention_k256_a_exchange_b4_expected_logical_hits}"
    printf 'required_runtime_stages=%s\n' \
      "$(IFS=,; printf '%s' "${attention_k256_markers[*]}")"
    if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
      printf 'excluded_runtime_stages=%s,%s,%s,%s\n' \
        "$(IFS=,; printf '%s' "${attention_k256_incumbent_markers[*]}")" \
        "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_markers[*]}")" \
        "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_l2_macro4x4_markers[*]}")" \
        prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix
    elif [[ "${l2_macro4x4_mode}" == 1 ]]; then
      printf 'excluded_runtime_stages=%s,%s,%s\n' \
        "$(IFS=,; printf '%s' "${attention_k256_incumbent_markers[*]}")" \
        "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_markers[*]}")" \
        prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix
    elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
      printf 'excluded_runtime_stages=%s,%s\n' \
        "$(IFS=,; printf '%s' "${attention_k256_incumbent_markers[*]}")" \
        prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix
    else
      printf 'excluded_runtime_stages=%s,%s\n' \
        "$(IFS=,; printf '%s' "${attention_k256_a_exchange_b4_markers[*]}")" \
        prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix
    fi
    printf 'prefill_mlp_k512_k256_base_layout=%s\n' \
      "${attention_k256_overlay_layout}"
    printf 'prefill_mlp_k512_k256_base_payload_bytes=%s\n' \
      "${attention_k256_overlay_payload_bytes}"
    printf 'prefill_mlp_k512_k256_base_manifest_sha256=%s\n' \
      "${attention_k256_overlay_manifest_sha256}"
    printf 'prefill_mlp_k512_k256_base_policy_sha256=%s\n' \
      "${attention_k256_overlay_policy_sha256}"
    printf 'prefill_mlp_k512_k256_base_payload_sha256=%s\n' \
      "${attention_k256_overlay_payload_sha256}"
  fi
  if [[ "${mlp_k512_edge_mode}" == 1 &&
        "${mlp_k512_down_m16n64_v2_mode}" == 0 ]]; then
    printf 'required_runtime_stage=prefill_projection_span_mlp_k512_gateup_down_edge\n'
    printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize\n'
  fi
  if [[ "${attention_k256_a_exchange_b3_mode}" == 1 ]]; then
    printf 'required_runtime_stage=%s\n' "${ldmatrix_pairfeed_marker}"
    printf 'required_runtime_stage=prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'experiment_baseline_mode=%s\n' "${attention_b3_baseline_mode}"
    printf 'experiment_retained_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N128_A_EXCHANGE_B3_ADMISSION\n'
  fi
  if [[ "${mlp_k512_shape_separated_marlin_package_mode}" == 1 ]]; then
    printf 'prefill_mlp_k512_implementation=shape_separated_marlin\n'
    printf 'prefill_mlp_k512_gateup_implementation=m64n256_marlin_k64_b3\n'
    printf 'prefill_mlp_k512_down_implementation=m64n256_16warp_pairring\n'
    printf 'prefill_mlp_k512_parent_implementations=m64n128_k256_ldmatrix_pairfeed,m128n128_16warp_pairring\n'
    printf 'required_runtime_stages=%s,%s\n' \
      "${shape_separated_marlin_gate_marker}" \
      "${shape_separated_marlin_down_marker}"
    printf 'experiment_baseline_mode=%s\n' \
      "${shape_separated_marlin_baseline_mode}"
    printf 'experiment_retained_selectors=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_MLP_K512_SHAPE_SEPARATED_MARLIN_PACKAGE_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize\n'
    printf 'excluded_runtime_stages=%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_primary,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_secondary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary,prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix,prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline,prefill_projection_span_mlp_k512_gateup_m64n8_paired_warp_register_pipeline,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n' \
      "${ldmatrix_pairfeed_marker}" \
      "${pairring_down_16warp_marker}"
  fi
  if [[ "${mlp_k512_m32n512_owner_mode}" == 1 ]]; then
    printf 'prefill_mlp_k512_gateup_implementation=m32n512_owner_k128_b4\n'
    printf 'prefill_mlp_k512_gateup_parent_implementation=m64n128_k256_ldmatrix_pairfeed\n'
    printf 'required_runtime_stages=%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n' \
      "${m32n512_owner_marker}"
    printf 'experiment_baseline_mode=%s\n' "${m32n512_owner_baseline_mode}"
    printf 'experiment_retained_selectors=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION,Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'excluded_runtime_stages=%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gateup_down_edge_m128n64,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_primary,prefill_projection_span_mlp_k512_gateup_m128n128_projection_serial_secondary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_primary,prefill_projection_span_mlp_k512_gateup_m128n64_same_cta_secondary,prefill_projection_span_mlp_k512_gateup_m128n512_fused_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix,prefill_projection_span_mlp_k512_gateup_m64n128_register_pipeline,prefill_projection_span_mlp_k512_gateup_m64n8_paired_warp_register_pipeline,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n' \
      "${ldmatrix_pairfeed_marker}"
  fi
  if [[ "${mlp_k512_edge_m64n128_k256_ldmatrix_pairfeed_mode}" == 1 &&
        "${attention_k256_a_exchange_b3_mode}" == 0 &&
        "${mlp_k512_m32n512_owner_mode}" == 0 &&
        "${mlp_k512_shape_separated_marlin_package_mode}" == 0 ]]; then
    printf 'required_runtime_stage=%s\n' "${ldmatrix_pairfeed_marker}"
    printf 'required_runtime_stage=prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'experiment_baseline_mode=%s\n' \
      "${ldmatrix_pairfeed_baseline_mode}"
    printf 'experiment_removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize\n'
    printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n'
  fi
  if [[ "${mlp_k512_m128n128_projection_serial_mode}" == 1 ]]; then
    printf 'required_runtime_stages=%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n' \
      "${projection_serial_primary_marker}" \
      "${projection_serial_secondary_marker}" \
      "${projection_serial_quantize_marker}"
    printf 'experiment_baseline_mode=%s\n' \
      "${projection_serial_baseline_mode}"
    printf 'experiment_removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'excluded_runtime_stages=%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n' \
      "${ldmatrix_pairfeed_marker}"
  fi
  if [[ "${mlp_k512_m128n64_same_cta_mode}" == 1 ]]; then
    printf 'required_runtime_stages=%s,%s,%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n' \
      "${same_cta_primary_marker}" \
      "${same_cta_secondary_marker}" \
      "${projection_serial_quantize_marker}"
    printf 'experiment_baseline_mode=%s\n' "${same_cta_baseline_mode}"
    printf 'experiment_removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'excluded_runtime_stages=%s,%s,%s,%s,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n' \
      "${ldmatrix_pairfeed_marker}" \
      "${projection_serial_primary_marker}" \
      "${projection_serial_secondary_marker}" \
      "${fused_quantize_marker}"
  fi
  if [[ "${mlp_k512_m128n512_fused_quantize_mode}" == 1 ]]; then
    printf 'required_runtime_stages=%s,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n' \
      "${fused_quantize_marker}"
    printf 'experiment_baseline_mode=%s\n' \
      "${fused_quantize_baseline_mode}"
    printf 'experiment_removed_selector=Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
    printf 'experiment_added_selector=Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
    printf 'excluded_runtime_stages=%s,%s,%s,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating,prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n' \
      "${ldmatrix_pairfeed_marker}" \
      "${projection_serial_primary_marker}" \
      "${projection_serial_secondary_marker}"
  fi
  if [[ "${mlp_k512_edge_m64n128_k256_alternating_mode}" == 1 ]]; then
    printf 'required_runtime_stage=prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating\n'
    if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
      if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
        : # The GDN-only experiment provenance was recorded above.
      elif [[ "${attention_k256_a_exchange_b4_mode}" == 1 ]]; then
        printf 'experiment_baseline_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256\n'
        printf 'experiment_removed_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION\n'
        printf 'experiment_added_selector=Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION\n'
      else
        printf 'experiment_baseline_mode=cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256\n'
        printf 'experiment_removed_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION\n'
        printf 'experiment_added_selector=Q3X_RUN_A4W4_DOWN_K512_M128N128_16WARP_PAIRRING_ADMISSION\n'
      fi
      printf 'required_runtime_stage=prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n'
      printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize\n'
      printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2,prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n'
    elif [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
      printf 'required_runtime_stage=prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring\n'
      printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize\n'
      printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_down_m16n64_v2\n'
    else
      printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
      printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down_m16n64_v2\n'
    fi
  fi
  if [[ -n "${gateup_alternating_expected_hits}" ||
        -n "${gateup_ldmatrix_pairfeed_expected_hits}" ||
        -n "${gateup_m128n128_projection_serial_expected_hits}" ||
        -n "${gateup_m128n64_same_cta_expected_hits}" ||
        -n "${gateup_m128n512_fused_quantize_expected_hits}" ||
        -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ||
        -n "${gateup_m64n128_register_pipeline_expected_hits}" ||
        -n "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" ||
        -n "${factorized_lane_r1_package_expected_hits}" ]]; then
    printf 'gateup_alternating_expected_launch_hits_per_request=%s\n' \
      "${gateup_alternating_expected_hits}"
  fi
  if [[ -n "${gateup_ldmatrix_pairfeed_expected_hits}" ]]; then
    printf 'gateup_ldmatrix_pairfeed_expected_launch_hits_per_request=%s\n' \
      "${gateup_ldmatrix_pairfeed_expected_hits}"
  fi
  if [[ -n "${gateup_m128n128_projection_serial_expected_hits}" ]]; then
    printf 'gateup_m128n128_projection_serial_expected_launch_hits_per_request=%s\n' \
      "${gateup_m128n128_projection_serial_expected_hits}"
  fi
  if [[ -n "${gateup_m128n64_same_cta_expected_hits}" ]]; then
    printf 'gateup_m128n64_same_cta_expected_launch_hits_per_request=%s\n' \
      "${gateup_m128n64_same_cta_expected_hits}"
  fi
  if [[ -n "${gateup_m128n512_fused_quantize_expected_hits}" ]]; then
    printf 'gateup_m128n512_fused_quantize_expected_launch_hits_per_request=%s\n' \
      "${gateup_m128n512_fused_quantize_expected_hits}"
  fi
  if [[ -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ]]; then
    printf 'gateup_m128n512_paired_ldmatrix_expected_launch_hits_per_request=%s\n' \
      "${gateup_m128n512_paired_ldmatrix_expected_hits}"
  fi
  if [[ -n "${gateup_m64n128_register_pipeline_expected_hits}" ]]; then
    printf 'gateup_m64n128_register_pipeline_expected_launch_hits_per_request=%s\n' \
      "${gateup_m64n128_register_pipeline_expected_hits}"
  fi
  if [[ -n "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" ]]; then
    printf 'gateup_m64n8_paired_warp_register_pipeline_expected_launch_hits_per_request=%s\n' \
      "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}"
  fi
  if [[ -n "${factorized_lane_r1_package_expected_hits}" ]]; then
    printf 'factorized_lane_r1_package_expected_launch_hits_per_request=%s\n' \
      "${factorized_lane_r1_package_expected_hits}"
  fi
  if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
    printf 'down_m128n128_ldmatrix_pairring_expected_launch_hits_per_request=%s\n' \
      "${down_m128n128_ldmatrix_pairring_expected_hits}"
  fi
  if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ||
        "${mlp_k512_projection_major_mode}" == 1 ||
        "${mlp_k512_paired_warp_mode}" == 1 ||
        "${k256_pairfeed_package_selected}" == 1 ]]; then
    printf 'down_m128n128_ldmatrix_pairring_expected_launch_hits_per_request=%s\n' \
      "${down_m128n128_ldmatrix_pairring_expected_hits}"
    printf 'down_m128n128_16warp_pairring_expected_launch_hits_per_request=%s\n' \
      "${down_m128n128_16warp_pairring_expected_hits}"
  fi
  if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
    printf 'mlp_k256_m128n256_pairfeed_package_expected_launch_hits_per_request=%s\n' \
      "${mlp_k256_pairfeed_package_expected_hits}"
  fi
  if [[ "${mlp_k512_edge_m128n64_mode}" == 1 ]]; then
    printf 'required_runtime_stage=prefill_projection_span_mlp_k512_gateup_down_edge_m128n64\n'
    printf 'retained_runtime_stages=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
    printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down_m16n64_v2\n'
  fi
  if [[ "${mlp_k512_down_m16n64_v2_mode}" == 1 ]]; then
    printf 'required_runtime_stages=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_down_m16n64_v2\n'
    printf 'excluded_runtime_stages=prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize\n'
  fi
  if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
    printf 'prefill_mlp_k512_fragment_native_gateup_variant=%s\n' \
      "${mlp_k512_fragment_native_gateup_variant}"
    printf 'prefill_mlp_k512_fragment_native_down_variant=%s\n' \
      "${mlp_k512_fragment_native_down_variant}"
    printf 'prefill_mlp_k512_fragment_native_layout=%s\n' \
      "${mlp_k512_fragment_native_layout}"
    printf 'prefill_mlp_k512_fragment_native_payload_bytes=%s\n' \
      "${mlp_k512_fragment_native_payload_bytes}"
    printf 'prefill_mlp_k512_fragment_native_payload_sha256=%s\n' \
      "${mlp_k512_fragment_native_payload_sha256}"
    printf 'prefill_mlp_k512_fragment_native_policy_sha256=%s\n' \
      "${mlp_k512_fragment_native_policy_sha256}"
    printf 'prefill_mlp_k512_fragment_native_receipt_sha256=%s\n' \
      "${mlp_k512_fragment_native_receipt_sha256}"
  fi
  if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_layout=%s\n' \
      "${mlp_k512_hybrid_layout}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_payload_bytes=%s\n' \
      "${mlp_k512_hybrid_payload_bytes}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_payload_sha256=%s\n' \
      "${mlp_k512_hybrid_payload_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_policy_sha256=%s\n' \
      "${mlp_k512_hybrid_policy_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_receipt_sha256=%s\n' \
      "${mlp_k512_hybrid_receipt_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_manifest_sha256=%s\n' \
      "${mlp_k512_hybrid_manifest_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_source_v1_receipt_sha256=%s\n' \
      "${mlp_k512_hybrid_source_v1_receipt_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_source_v1_manifest_sha256=%s\n' \
      "${mlp_k512_hybrid_source_v1_manifest_sha256}"
    printf 'prefill_mlp_k512_paired_gateup_canonical_down_source_v1_payload_sha256=%s\n' \
      "${mlp_k512_hybrid_source_v1_payload_sha256}"
    if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
      printf 'experiment_baseline_mode=%s\n' "${paired_warp_baseline_mode}"
      printf 'experiment_removed_selectors=Q3X_RUN_A4W4_MLP_K512_ADMISSION,Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
      printf 'experiment_added_selectors=Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION,Q3X_RUN_A4W4_GATEUP_K512_M64N8_PAIRED_WARP_REGISTER_PIPELINE_ADMISSION\n'
      printf 'required_runtime_stages=%s,%s,%s\n' \
        "${hybrid_input_marker}" "${paired_warp_gate_marker}" \
        "${paired_warp_down_16warp_marker}"
      printf 'excluded_runtime_stages=%s,%s,%s\n' \
        "${hybrid_gate_marker}" "${hybrid_canonical_down_marker}" \
        "${hybrid_pairring_down_marker}"
    elif [[ "${mlp_k512_hybrid_down_pairring_mode}" == 1 ]]; then
      printf 'required_runtime_stages=%s,%s,%s\n' \
        "${hybrid_input_marker}" "${hybrid_gate_marker}" \
        "${hybrid_pairring_down_marker}"
      printf 'excluded_runtime_stages=%s,%s\n' \
        "${hybrid_canonical_down_marker}" "${hybrid_old_runtime_stages}"
    else
      printf 'required_runtime_stages=%s,%s\n' \
        "${hybrid_input_marker}" "${hybrid_gate_marker}"
      printf 'retained_runtime_stages=%s\n' \
        "${hybrid_canonical_down_marker}"
      printf 'excluded_runtime_stages=%s,%s\n' \
        "${hybrid_pairring_down_marker}" "${hybrid_old_runtime_stages}"
    fi
    printf 'gateup_m128n512_paired_ldmatrix_expected_launch_hits_per_request=%s\n' \
      "${gateup_m128n512_paired_ldmatrix_expected_hits}"
    printf 'down_m128n128_ldmatrix_pairring_expected_launch_hits_per_request=%s\n' \
      "${down_m128n128_ldmatrix_pairring_expected_hits}"
    if [[ "${mlp_k512_paired_warp_mode}" == 1 ]]; then
      printf 'gateup_m64n8_paired_warp_register_pipeline_expected_launch_hits_per_request=%s\n' \
        "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}"
      printf 'down_m128n128_16warp_pairring_expected_launch_hits_per_request=%s\n' \
        "${down_m128n128_16warp_pairring_expected_hits}"
    fi
  fi
  if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
    printf 'experiment_baseline_mode=%s\n' "${projection_major_baseline_mode}"
    printf 'experiment_removed_selectors=Q3X_RUN_A4W4_MLP_K512_ADMISSION,Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION\n'
    printf 'experiment_added_selectors=Q3X_RUN_A4W4_MLP_K512_PROJECTION_MAJOR_GATEUP_CANONICAL_DOWN_ADMISSION,Q3X_RUN_A4W4_GATEUP_K512_M64N128_REGISTER_PIPELINE_ADMISSION\n'
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_layout=%s\n' \
      "${mlp_k512_projection_major_layout}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_payload_bytes=%s\n' \
      "${mlp_k512_projection_major_payload_bytes}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_payload_sha256=%s\n' \
      "${mlp_k512_projection_major_payload_sha256}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_policy_sha256=%s\n' \
      "${mlp_k512_projection_major_policy_sha256}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_sha256=%s\n' \
      "${mlp_k512_projection_major_receipt_sha256}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_manifest_sha256=%s\n' \
      "${mlp_k512_projection_major_manifest_sha256}"
    printf 'prefill_mlp_k512_projection_major_gateup_canonical_down_source_v1_receipt_sha256=%s\n' \
      "${mlp_k512_projection_major_source_v1_receipt_sha256}"
    printf 'required_runtime_stages=%s,%s,%s\n' \
      "${projection_major_input_marker}" \
      "${projection_major_gate_marker}" \
      "${projection_major_down_16warp_marker}"
    printf 'excluded_runtime_stages=%s,prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down_m128n128_16warp_pairring\n' \
      "${ldmatrix_pairfeed_marker}"
  fi
  for bucket in "${buckets[@]}"; do
    printf 'corpus_%s_sha256=%s\n' \
      "${bucket}" "${corpus_sha[${bucket}]}"
  done
} >"${output_root}/provenance.txt"
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

"${runtime_env[@]}" "${profiler_prefix[@]}" "${server_args[@]}" \
  >"${server_log}" 2>&1 &
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
    'max_sequence_length=4096 maximum_output_tokens=1 .*prefill_chunk_size=512 .*readiness_route=/healthz' \
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
if [[ "${attention_k256_mode}" == 1 ]]; then
  if ! grep -F \
      "prefill_a4_bytes=${attention_k256_payload_bytes} prefill_a4_layout=${attention_k256_layout} prefill_a4_manifest_sha256=${attention_k256_manifest_sha256} prefill_a4_policy_sha256=${attention_k256_policy_sha256} prefill_a4_payload_sha256=${attention_k256_payload_sha256}" \
      "${server_log}" >/dev/null; then
    echo "server readiness did not prove the exact authenticated K256 publication" >&2
    exit 5
  fi
fi
if [[ "${k512_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_attention_o_k512_requested=1 .*prefill_attention_o_k512_enabled=1 .*prefill_attention_o_k512_projections=64([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated Prefill Attention-O K512 64/64" >&2
    exit 5
  fi
  if ! grep -Eq \
      'prefill_attention_o_k512_payload_sha256=[0-9a-f]{64}([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove the Prefill Attention-O K512 payload SHA-256" >&2
    exit 5
  fi
fi
if [[ "${mlp_k512_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_mlp_k512_requested=1 .*prefill_mlp_k512_enabled=1 .*prefill_mlp_k512_projections=192([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated Prefill MLP K512 192/192" >&2
    exit 5
  fi
  if ! grep -Eq \
      'prefill_mlp_k512_payload_sha256=[0-9a-f]{64}([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove the Prefill MLP K512 payload SHA-256" >&2
    exit 5
  fi
  if [[ "${attention_k256_mode}" == 1 ]] &&
     ! grep -F \
       "prefill_mlp_k512_bytes=${attention_k256_overlay_payload_bytes} prefill_mlp_k512_copy_chunks=" \
       "${server_log}" >/dev/null; then
    echo "server readiness did not prove the K256-bound K512 MLP payload size" >&2
    exit 5
  fi
  if [[ "${attention_k256_mode}" == 1 ]] &&
     ! grep -F \
       "prefill_mlp_k512_layout=${attention_k256_overlay_layout} prefill_mlp_k512_manifest_sha256=${attention_k256_overlay_manifest_sha256} prefill_mlp_k512_policy_sha256=${attention_k256_overlay_policy_sha256} prefill_mlp_k512_payload_sha256=${attention_k256_overlay_payload_sha256}" \
       "${server_log}" >/dev/null; then
    echo "server readiness did not prove the exact K256-bound K512 MLP SHA chain" >&2
    exit 5
  fi
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_mlp_factorized_lane_r1_requested=1 .*prefill_mlp_factorized_lane_r1_enabled=1 .*prefill_mlp_factorized_lane_r1_layers=64 .*prefill_mlp_factorized_lane_r1_bytes=8568619008([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated factorized-lane R1 64/64" >&2
    exit 5
  fi
  factorized_r1_readiness_bindings=(
    "prefill_mlp_factorized_lane_r1_layout=${factorized_r1_layout}"
    "prefill_mlp_factorized_lane_r1_manifest_sha256=${factorized_r1_manifest_sha256}"
    "prefill_mlp_factorized_lane_r1_policy_sha256=${factorized_r1_policy_sha256}"
    "prefill_mlp_factorized_lane_r1_payload_sha256=${factorized_r1_payload_sha256}"
    "prefill_mlp_factorized_lane_r1_receipt_sha256=${factorized_r1_receipt_sha256}"
    "prefill_mlp_factorized_lane_r1_base_receipt_sha256=${factorized_r1_base_receipt_sha256}"
    "prefill_mlp_factorized_lane_r1_min_prompt_tokens=480"
    "prefill_mlp_factorized_lane_r1_production_residency_eligible=false"
    "prefill_mlp_factorized_lane_r1_quality_production_eligible=false"
    "prefill_mlp_factorized_lane_r1_performance_upper_bound_only=true"
  )
  for binding in "${factorized_r1_readiness_bindings[@]}"; do
    if ! grep -F " ${binding}" "${server_log}" >/dev/null; then
      echo "server readiness did not prove factorized-lane R1 binding: ${binding}" >&2
      exit 5
    fi
  done
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_mlp_k512_fragment_native_requested=1 .*prefill_mlp_k512_fragment_native_enabled=1 .*prefill_mlp_k512_fragment_native_layers=64 prefill_mlp_k512_fragment_native_bytes=8623226880([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated fragment-native MLP K512 64/64" >&2
    exit 5
  fi
  if ! grep -F \
      " prefill_mlp_k512_fragment_native_layout=${mlp_k512_fragment_native_layout} " \
      "${server_log}" >/dev/null; then
    echo "server readiness did not prove the exact fragment-native MLP K512 layout" >&2
    exit 5
  fi
  if ! grep -F \
      " prefill_mlp_k512_fragment_native_payload_sha256=${mlp_k512_fragment_native_payload_sha256} " \
      "${server_log}" >/dev/null; then
    echo "server readiness payload SHA-256 does not match the real fragment-native file" >&2
    exit 5
  fi
  if ! grep -F \
      " prefill_mlp_k512_fragment_native_policy_sha256=${mlp_k512_fragment_native_policy_sha256} " \
      "${server_log}" >/dev/null; then
    echo "server readiness policy SHA-256 does not match the real source-v1 policy file" >&2
    exit 5
  fi
  if ! grep -F \
      " prefill_mlp_k512_fragment_native_receipt_sha256=${mlp_k512_fragment_native_receipt_sha256} " \
      "${server_log}" >/dev/null; then
    echo "server readiness receipt SHA-256 does not match the real v2 receipt file" >&2
    exit 5
  fi
  if ! grep -Eq \
      'prefill_mlp_k512_fragment_native_manifest_sha256=[0-9a-f]{64} .*prefill_mlp_k512_fragment_native_policy_sha256=[0-9a-f]{64} .*prefill_mlp_k512_fragment_native_payload_sha256=[0-9a-f]{64} .*prefill_mlp_k512_fragment_native_receipt_sha256=[0-9a-f]{64} .*prefill_mlp_k512_fragment_native_source_v1_receipt_sha256=[0-9a-f]{64}([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove the complete fragment-native SHA binding chain" >&2
    exit 5
  fi
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_mlp_k512_paired_gateup_canonical_down_requested=1 .*prefill_mlp_k512_paired_gateup_canonical_down_enabled=1 .*prefill_mlp_k512_paired_gateup_canonical_down_layers=64 .*prefill_mlp_k512_paired_gateup_canonical_down_bytes=8623226880([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated paired-GateUp/canonical-Down MLP K512 64/64" >&2
    exit 5
  fi
  hybrid_readiness_bindings=(
    "prefill_mlp_k512_paired_gateup_canonical_down_layout=${mlp_k512_hybrid_layout}"
    "prefill_mlp_k512_paired_gateup_canonical_down_manifest_sha256=${mlp_k512_hybrid_manifest_sha256}"
    "prefill_mlp_k512_paired_gateup_canonical_down_policy_sha256=${mlp_k512_hybrid_policy_sha256}"
    "prefill_mlp_k512_paired_gateup_canonical_down_payload_sha256=${mlp_k512_hybrid_payload_sha256}"
    "prefill_mlp_k512_paired_gateup_canonical_down_receipt_sha256=${mlp_k512_hybrid_receipt_sha256}"
    "prefill_mlp_k512_paired_gateup_canonical_down_source_v1_receipt_sha256=${mlp_k512_hybrid_source_v1_receipt_sha256}"
  )
  for binding in "${hybrid_readiness_bindings[@]}"; do
    if ! grep -F " ${binding}" "${server_log}" >/dev/null; then
      echo "server readiness did not prove the exact hybrid publication binding: ${binding}" >&2
      exit 5
    fi
  done
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  if ! grep -Eq \
      'prefill_mlp_k512_projection_major_gateup_canonical_down_requested=1 .*prefill_mlp_k512_projection_major_gateup_canonical_down_enabled=1 .*prefill_mlp_k512_projection_major_gateup_canonical_down_layers=64 .*prefill_mlp_k512_projection_major_gateup_canonical_down_bytes=8623226880([[:space:]]|$)' \
      "${server_log}"; then
    echo "server readiness did not prove authenticated projection-major-GateUp/canonical-Down MLP K512 64/64" >&2
    exit 5
  fi
  projection_major_readiness_bindings=(
    "prefill_mlp_k512_projection_major_gateup_canonical_down_layout=${mlp_k512_projection_major_layout}"
    "prefill_mlp_k512_projection_major_gateup_canonical_down_manifest_sha256=${mlp_k512_projection_major_manifest_sha256}"
    "prefill_mlp_k512_projection_major_gateup_canonical_down_policy_sha256=${mlp_k512_projection_major_policy_sha256}"
    "prefill_mlp_k512_projection_major_gateup_canonical_down_payload_sha256=${mlp_k512_projection_major_payload_sha256}"
    "prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_sha256=${mlp_k512_projection_major_receipt_sha256}"
    "prefill_mlp_k512_projection_major_gateup_canonical_down_source_v1_receipt_sha256=${mlp_k512_projection_major_source_v1_receipt_sha256}"
  )
  for binding in "${projection_major_readiness_bindings[@]}"; do
    if ! grep -F " ${binding}" "${server_log}" >/dev/null; then
      echo "server readiness did not prove the exact projection-major publication binding: ${binding}" >&2
      exit 5
    fi
  done
fi
printf 'startup_contract_check=passed prefill_a4_authenticated=400/400 optimized_prefill_disabled=0'
if [[ "${attention_k256_mode}" == 1 ]]; then
  printf ' prefill_a4_k256_layout=verified prefill_a4_k256_bytes=verified prefill_a4_k256_sha_chain=verified prefill_attention_k256_runtime_accounting=required'
fi
if [[ "${k512_mode}" == 1 ]]; then
  printf ' prefill_attention_o_k512_authenticated=64/64 prefill_attention_o_k512_payload_sha256=verified'
fi
if [[ "${mlp_k512_mode}" == 1 ]]; then
  printf ' prefill_mlp_k512_authenticated=192/192 prefill_mlp_k512_payload_sha256=verified'
fi
if [[ "${mlp_k512_fragment_native_mode}" == 1 ]]; then
  printf ' prefill_mlp_k512_fragment_native_authenticated=64/64 prefill_mlp_k512_fragment_native_gateup_variant=%s prefill_mlp_k512_fragment_native_down_variant=%s prefill_mlp_k512_fragment_native_layout=verified prefill_mlp_k512_fragment_native_payload_sha256=verified prefill_mlp_k512_fragment_native_policy_sha256=verified prefill_mlp_k512_fragment_native_receipt_sha256=verified' \
    "${mlp_k512_fragment_native_gateup_variant}" \
    "${mlp_k512_fragment_native_down_variant}"
fi
if [[ "${mlp_k512_hybrid_mode}" == 1 ]]; then
  printf ' prefill_mlp_k512_paired_gateup_canonical_down_authenticated=64/64 prefill_mlp_k512_paired_gateup_canonical_down_layout=verified prefill_mlp_k512_paired_gateup_canonical_down_payload_sha256=verified prefill_mlp_k512_paired_gateup_canonical_down_policy_sha256=verified prefill_mlp_k512_paired_gateup_canonical_down_receipt_sha256=verified prefill_mlp_k512_paired_gateup_canonical_down_complete_sha_chain=verified'
fi
if [[ "${mlp_k512_projection_major_mode}" == 1 ]]; then
  printf ' prefill_mlp_k512_projection_major_gateup_canonical_down_authenticated=64/64 prefill_mlp_k512_projection_major_gateup_canonical_down_layout=verified prefill_mlp_k512_projection_major_gateup_canonical_down_payload_sha256=verified prefill_mlp_k512_projection_major_gateup_canonical_down_policy_sha256=verified prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_sha256=verified prefill_mlp_k512_projection_major_gateup_canonical_down_complete_sha_chain=verified'
fi
if [[ "${factorized_r1_mode}" == 1 ]]; then
  printf ' prefill_mlp_factorized_lane_r1_authenticated=64/64 prefill_mlp_factorized_lane_r1_sha_chain=verified production_residency_eligible=false quality_production_eligible=false performance_upper_bound_only=true'
fi
printf '\n'

for bucket in "${buckets[@]}"; do
  corpus="${corpus_dir}/q3x-sharegpt-prefill-${bucket}-5.jsonl"
  run_dir="${output_root}/${bucket}"
  result_name="pure-prefill-${bucket}-${mode}"
  result_leaf="${run_dir}/${result_name}/parallel_1_number_${eval_number}"
  mkdir -p "${run_dir}"
  request_log_before=$(grep -Ec \
    '^evaluation request .* prompt_tokens=' "${server_log}" || true)
  uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
    --model qwen3.6-27b-nvfp4 --api openai \
    --url "http://127.0.0.1:${port}/v1/completions" \
    --tokenizer-path "${model_dir}" \
    --dataset line_by_line --data-source local \
    --dataset-path "${corpus}" \
    --number "${eval_number}" --parallel 1 --warmup-num 1 --num-workers 1 \
    --max-tokens 1 --temperature 0 --seed 42 \
    --stream --tokenize-prompt --no-test-connection \
    --outputs-dir "${run_dir}" --name "${result_name}" \
    --no-timestamp >"${run_dir}/evalscope.stdout" 2>&1
  if [[ -n "${gateup_alternating_expected_hits}" ||
        -n "${gateup_ldmatrix_pairfeed_expected_hits}" ||
        -n "${gateup_m128n128_projection_serial_expected_hits}" ||
        -n "${gateup_m128n64_same_cta_expected_hits}" ||
        -n "${gateup_m128n512_fused_quantize_expected_hits}" ||
        -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ||
        -n "${gateup_m64n128_register_pipeline_expected_hits}" ||
        -n "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" ||
        -n "${factorized_lane_r1_package_expected_hits}" ]]; then
    mapfile -t new_request_logs < <(
      awk -v skip="${request_log_before}" '
        /^evaluation request .* prompt_tokens=/ {
          successes += 1
          if (successes > skip) {
            print
          }
        }
      ' "${server_log}"
    )
    expected_request_logs=$((eval_number + 1))
    if [[ ${#new_request_logs[@]} -ne ${expected_request_logs} ]]; then
      echo "expected exactly ${expected_request_logs} successful API request logs for ${bucket}, found ${#new_request_logs[@]}" >&2
      exit 6
    fi
    for request_log in "${new_request_logs[@]}"; do
      if ! grep -Eq \
          " gateup_alternating_launch_hits=${gateup_alternating_expected_hits}([[:space:]]|$)" \
          <<<"${request_log}"; then
        echo "request did not prove gateup_alternating_launch_hits=${gateup_alternating_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_ldmatrix_pairfeed_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_ldmatrix_pairfeed_launch_hits=${gateup_ldmatrix_pairfeed_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_ldmatrix_pairfeed_launch_hits=${gateup_ldmatrix_pairfeed_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m128n128_projection_serial_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m128n128_projection_serial_launch_hits=${gateup_m128n128_projection_serial_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m128n128_projection_serial_launch_hits=${gateup_m128n128_projection_serial_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m128n64_same_cta_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m128n64_same_cta_launch_hits=${gateup_m128n64_same_cta_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m128n64_same_cta_launch_hits=${gateup_m128n64_same_cta_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m128n512_fused_quantize_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m128n512_fused_quantize_launch_hits=${gateup_m128n512_fused_quantize_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m128n512_fused_quantize_launch_hits=${gateup_m128n512_fused_quantize_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m128n512_paired_ldmatrix_launch_hits=${gateup_m128n512_paired_ldmatrix_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m128n512_paired_ldmatrix_launch_hits=${gateup_m128n512_paired_ldmatrix_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m64n128_register_pipeline_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m64n128_register_pipeline_launch_hits=${gateup_m64n128_register_pipeline_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m64n128_register_pipeline_launch_hits=${gateup_m64n128_register_pipeline_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" ]] &&
         ! grep -Eq \
           " gateup_m64n8_paired_warp_register_pipeline_launch_hits=${gateup_m64n8_paired_warp_register_pipeline_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove gateup_m64n8_paired_warp_register_pipeline_launch_hits=${gateup_m64n8_paired_warp_register_pipeline_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ||
            "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ||
            "${mlp_k512_projection_major_mode}" == 1 ||
            "${k256_pairfeed_package_selected}" == 1 ||
            "${factorized_r1_mode}" == 1 ]] &&
         ! grep -Eq \
           " down_m128n128_ldmatrix_pairring_launch_hits=${down_m128n128_ldmatrix_pairring_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove down_m128n128_ldmatrix_pairring_launch_hits=${down_m128n128_ldmatrix_pairring_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ||
            "${mlp_k512_projection_major_mode}" == 1 ||
            "${mlp_k512_paired_warp_mode}" == 1 ||
            "${k256_pairfeed_package_selected}" == 1 ||
            "${factorized_r1_mode}" == 1 ]] &&
         ! grep -Eq \
           " down_m128n128_16warp_pairring_launch_hits=${down_m128n128_16warp_pairring_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove down_m128n128_16warp_pairring_launch_hits=${down_m128n128_16warp_pairring_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ "${k256_pairfeed_package_selected}" == 1 ||
            "${factorized_r1_mode}" == 1 ]] &&
         ! grep -Eq \
           " mlp_k256_m128n256_pairfeed_package_launch_hits=${mlp_k256_pairfeed_package_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove mlp_k256_m128n256_pairfeed_package_launch_hits=${mlp_k256_pairfeed_package_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ -n "${factorized_lane_r1_package_expected_hits}" ]] &&
         ! grep -Eq \
           " factorized_lane_r1_package_launch_hits=${factorized_lane_r1_package_expected_hits}([[:space:]]|$)" \
           <<<"${request_log}"; then
        echo "request did not prove factorized_lane_r1_package_launch_hits=${factorized_lane_r1_package_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if [[ "${factorized_r1_mode}" == 1 ]] &&
         ! grep -Eq ' mtp=false([[:space:]]|$)' <<<"${request_log}"; then
        echo "request did not prove mtp=false: ${request_log}" >&2
        exit 6
      fi
    done
    printf 'gateup_alternating_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
      "${bucket}" "${expected_request_logs}" \
      "${gateup_alternating_expected_hits}"
    if [[ -n "${gateup_ldmatrix_pairfeed_expected_hits}" ]]; then
      printf 'gateup_ldmatrix_pairfeed_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_ldmatrix_pairfeed_expected_hits}"
    fi
    if [[ -n "${gateup_m128n128_projection_serial_expected_hits}" ]]; then
      printf 'gateup_m128n128_projection_serial_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_m128n128_projection_serial_expected_hits}"
    fi
    if [[ -n "${gateup_m128n64_same_cta_expected_hits}" ]]; then
      printf 'gateup_m128n64_same_cta_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_m128n64_same_cta_expected_hits}"
    fi
    if [[ -n "${gateup_m128n512_fused_quantize_expected_hits}" ]]; then
      printf 'gateup_m128n512_fused_quantize_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_m128n512_fused_quantize_expected_hits}"
    fi
    if [[ -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ]]; then
      printf 'gateup_m128n512_paired_ldmatrix_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_m128n512_paired_ldmatrix_expected_hits}"
    fi
    if [[ -n "${gateup_m64n128_register_pipeline_expected_hits}" ]]; then
      printf 'projection_major_mlp_runtime_contract bucket=%s requests=%s old_gate_launch_hits_per_request=%s gate_launch_hits_per_request=%s down_incumbent_launch_hits_per_request=%s down_candidate_launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_ldmatrix_pairfeed_expected_hits}" \
        "${gateup_m64n128_register_pipeline_expected_hits}" \
        "${down_m128n128_ldmatrix_pairring_expected_hits}" \
        "${down_m128n128_16warp_pairring_expected_hits}"
    fi
    if [[ -n "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" ]]; then
      printf 'paired_warp_mlp_runtime_contract bucket=%s requests=%s old_gate_launch_hits_per_request=%s gate_launch_hits_per_request=%s down_incumbent_launch_hits_per_request=%s down_candidate_launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${gateup_m128n512_paired_ldmatrix_expected_hits}" \
        "${gateup_m64n8_paired_warp_register_pipeline_expected_hits}" \
        "${down_m128n128_ldmatrix_pairring_expected_hits}" \
        "${down_m128n128_16warp_pairring_expected_hits}"
    fi
    if [[ "${mlp_k512_v1_down_pairring_mode}" == 1 ]]; then
      printf 'v1_pairring_down_runtime_contract bucket=%s requests=%s launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${down_m128n128_ldmatrix_pairring_expected_hits}"
    fi
    if [[ "${mlp_k512_v1_down_16warp_pairring_mode}" == 1 ]]; then
      printf 'v1_16warp_pairring_down_runtime_contract bucket=%s requests=%s incumbent_launch_hits_per_request=%s candidate_launch_hits_per_request=%s status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${down_m128n128_ldmatrix_pairring_expected_hits}" \
        "${down_m128n128_16warp_pairring_expected_hits}"
    fi
    if [[ "${k256_pairfeed_package_selected}" == 1 ]]; then
      printf 'mlp_k256_pairfeed_package_runtime_contract bucket=%s requests=%s package_launch_hits_per_request=%s gateup_alternating_launch_hits_per_request=0 gateup_ldmatrix_pairfeed_launch_hits_per_request=0 gateup_m128n128_projection_serial_launch_hits_per_request=0 gateup_m128n64_same_cta_launch_hits_per_request=0 gateup_m128n512_fused_quantize_launch_hits_per_request=0 gateup_m128n512_paired_ldmatrix_launch_hits_per_request=0 gateup_m64n128_register_pipeline_launch_hits_per_request=0 gateup_m64n8_paired_warp_register_pipeline_launch_hits_per_request=0 down_m128n128_ldmatrix_pairring_launch_hits_per_request=0 down_m128n128_16warp_pairring_launch_hits_per_request=0 status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${mlp_k256_pairfeed_package_expected_hits}"
    fi
    if [[ -n "${factorized_lane_r1_package_expected_hits}" ]]; then
      printf 'factorized_lane_r1_runtime_contract bucket=%s requests=%s package_launch_hits_per_request=%s mtp=false status=passed\n' \
        "${bucket}" "${expected_request_logs}" \
        "${factorized_lane_r1_package_expected_hits}"
    fi
  fi
  if [[ "${attention_k256_mode}" == 1 ]]; then
    mapfile -t attention_request_logs < <(
      awk -v skip="${request_log_before}" '
        /^evaluation request .* prompt_tokens=/ {
          successes += 1
          if (successes > skip) {
            print
          }
        }
      ' "${server_log}"
    )
    expected_request_logs=$((eval_number + 1))
    if [[ ${#attention_request_logs[@]} -ne ${expected_request_logs} ]]; then
      echo "expected exactly ${expected_request_logs} successful API request logs for ${bucket}, found ${#attention_request_logs[@]}" >&2
      exit 6
    fi
    for request_log in "${attention_request_logs[@]}"; do
      attention_runtime_contract=(
        "attention_k256_m128n256_incumbent_launch_hits=${attention_k256_incumbent_expected_launch_hits}"
        "attention_k256_m128n256_incumbent_logical_projection_hits=${attention_k256_incumbent_expected_logical_hits}"
        "attention_k256_m128n256_a_exchange_b4_launch_hits=${attention_k256_a_exchange_b4_expected_launch_hits}"
        "attention_k256_m128n256_a_exchange_b4_logical_projection_hits=${attention_k256_a_exchange_b4_expected_logical_hits}"
      )
      for binding in "${attention_runtime_contract[@]}"; do
        if ! grep -Eq " ${binding}([[:space:]]|$)" <<<"${request_log}"; then
          echo "request did not prove ${binding}: ${request_log}" >&2
          exit 6
        fi
      done
    done
    printf 'attention_k256_runtime_contract bucket=%s requests=%s incumbent_launch_hits_per_request=%s incumbent_logical_projections_per_request=%s candidate_launch_hits_per_request=%s candidate_logical_projections_per_request=%s status=passed\n' \
      "${bucket}" "${expected_request_logs}" \
      "${attention_k256_incumbent_expected_launch_hits}" \
      "${attention_k256_incumbent_expected_logical_hits}" \
      "${attention_k256_a_exchange_b4_expected_launch_hits}" \
      "${attention_k256_a_exchange_b4_expected_logical_hits}"
  fi
  if [[ "${gdn_prompt_span_accounting_mode}" == 1 ]]; then
    mapfile -t gdn_request_logs < <(
      awk -v skip="${request_log_before}" '
        /^evaluation request .* prompt_tokens=/ {
          successes += 1
          if (successes > skip) {
            print
          }
        }
      ' "${server_log}"
    )
    expected_request_logs=$((eval_number + 1))
    if [[ ${#gdn_request_logs[@]} -ne ${expected_request_logs} ]]; then
      echo "expected exactly ${expected_request_logs} successful API request logs for ${bucket}, found ${#gdn_request_logs[@]}" >&2
      exit 6
    fi
    for request_log in "${gdn_request_logs[@]}"; do
      if [[ ! "${request_log}" =~ (^|[[:space:]])prompt_tokens=([0-9]+)([[:space:]]|$) ]]; then
        echo "GDN request accounting could not parse prompt_tokens: ${request_log}" >&2
        exit 6
      fi
      prompt_tokens=${BASH_REMATCH[2]}
      expected_gdn_logical_token_hits=$((48 * prompt_tokens))
      expected_gdn_native_launch_hits=$((48 * ((prompt_tokens + 511) / 512)))
      expected_gdn_native_logical_token_hits=${expected_gdn_logical_token_hits}
      expected_gdn_macro_launch_hits=0
      expected_gdn_macro_logical_token_hits=0
      if [[ "${gdn_prompt_span_macro_mode}" == 1 ]]; then
        expected_gdn_native_launch_hits=0
        expected_gdn_native_logical_token_hits=0
        expected_gdn_macro_launch_hits=48
        expected_gdn_macro_logical_token_hits=${expected_gdn_logical_token_hits}
      fi
      gdn_runtime_contract=(
        "gdn_chunk64_native_launch_hits=${expected_gdn_native_launch_hits}"
        "gdn_chunk64_native_logical_token_hits=${expected_gdn_native_logical_token_hits}"
        "gdn_prompt_span_macro_launch_hits=${expected_gdn_macro_launch_hits}"
        "gdn_prompt_span_macro_logical_token_hits=${expected_gdn_macro_logical_token_hits}"
      )
      for binding in "${gdn_runtime_contract[@]}"; do
        if ! grep -Eq " ${binding}([[:space:]]|$)" <<<"${request_log}"; then
          echo "request did not prove ${binding}: ${request_log}" >&2
          exit 6
        fi
      done
    done
    printf 'gdn_prompt_span_runtime_contract bucket=%s requests=%s route=%s dynamic_prompt_token_accounting=passed status=passed\n' \
      "${bucket}" "${expected_request_logs}" \
      "$([[ "${gdn_prompt_span_macro_mode}" == 1 ]] && \
         printf prompt-span-macro || printf native-c512)"
  fi
  if [[ -n "${gateup_m128n512_paired_ldmatrix_expected_hits}" ]]; then
    mapfile -t new_request_logs < <(
      awk -v skip="${request_log_before}" '
        /^evaluation request .* prompt_tokens=/ {
          successes += 1
          if (successes > skip) {
            print
          }
        }
      ' "${server_log}"
    )
    expected_request_logs=$((eval_number + 1))
    if [[ ${#new_request_logs[@]} -ne ${expected_request_logs} ]]; then
      echo "expected exactly ${expected_request_logs} successful API request logs for ${bucket}, found ${#new_request_logs[@]}" >&2
      exit 6
    fi
    for request_log in "${new_request_logs[@]}"; do
      if ! grep -Eq \
          " gateup_m128n512_paired_ldmatrix_launch_hits=${gateup_m128n512_paired_ldmatrix_expected_hits}([[:space:]]|$)" \
          <<<"${request_log}"; then
        echo "request did not prove gateup_m128n512_paired_ldmatrix_launch_hits=${gateup_m128n512_paired_ldmatrix_expected_hits}: ${request_log}" >&2
        exit 6
      fi
      if ! grep -Eq \
          " down_m128n128_ldmatrix_pairring_launch_hits=${down_m128n128_ldmatrix_pairring_expected_hits}([[:space:]]|$)" \
          <<<"${request_log}"; then
        echo "request did not prove down_m128n128_ldmatrix_pairring_launch_hits=${down_m128n128_ldmatrix_pairring_expected_hits}: ${request_log}" >&2
        exit 6
      fi
    done
    printf 'hybrid_mlp_runtime_contract bucket=%s requests=%s gate_launch_hits_per_request=%s down_pairring_launch_hits_per_request=%s status=passed\n' \
      "${bucket}" "${expected_request_logs}" \
      "${gateup_m128n512_paired_ldmatrix_expected_hits}" \
      "${down_m128n128_ldmatrix_pairring_expected_hits}"
  fi
  mapfile -t summary_files < <(
    find "${run_dir}" -name benchmark_summary.json -type f -print
  )
  expected_summary="${result_leaf}/benchmark_summary.json"
  [[ ${#summary_files[@]} == 1 &&
     "${summary_files[0]}" == "${expected_summary}" ]] || {
    echo "expected exactly the EvalScope result leaf ${expected_summary}" >&2
    if ((${#summary_files[@]} > 0)); then
      printf 'found EvalScope summary: %s\n' "${summary_files[@]}" >&2
    fi
    exit 6
  }
  summary_file=${expected_summary}
  python3 - "${summary_file}" "${eval_number}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
expected = int(sys.argv[2])
total = summary.get("Total Requests")
success = summary.get("Success Requests")
failed = summary.get("Failed Requests")
if total != expected or success != expected or failed != 0:
    raise SystemExit(
        f"EvalScope request contract failed: expected={expected} total={total} "
        f"success={success} failed={failed}"
    )
PY
  printf 'evalscope_request_contract bucket=%s total=%s success=%s failed=0 result_leaf=%q summary=%q\n' \
    "${bucket}" "${eval_number}" "${eval_number}" "${result_leaf}" \
    "${summary_file}"
  printf '%s\n' "${summary_file}"
  sed -n '1,220p' "${summary_file}"
done
