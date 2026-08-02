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
  [--mode exact|native-gdn|cumulative-prefill|cumulative-prefill-down|cumulative-prefill-attention-down|cumulative-prefill-current-best|cumulative-prefill-current-best-k512|cumulative-prefill-current-best-mlp-k512|cumulative-prefill-current-best-mlp-k512-v1|cumulative-prefill-current-best-mlp-k512-edge|cumulative-prefill-current-best-mlp-k512-edge-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m128n64|cumulative-prefill-current-best-mlp-k512-down-m16n64-v2|cumulative-prefill-current-best-mlp-k512-fragment-native|cumulative-prefill-current-best-mlp-k512-fragment-native-m128|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged|cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta|cumulative-prefill-short] \
  [--dry-run] \
  ELF MODEL_DIR CORPUS_DIR OUTPUT_ROOT [p512|p1k|p2k|p4k]
EOF
}

mode=exact
mode_seen=0
dry_run=${Q3X_PURE_PREFILL_DRY_RUN:-0}
profile_request_index=${Q3X_EVAL_PROFILE_REQUEST_INDEX:-0}
nsys_output=${Q3X_EVAL_NSYS_OUTPUT:-}
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
declare -a positional=()

while (($# > 0)); do
  case "$1" in
    --prefill-a4-payload|--prefill-a4-policy|--prefill-a4-receipt|\
    --prefill-attention-o-k512-payload|--prefill-attention-o-k512-policy|\
    --prefill-attention-o-k512-receipt|--prefill-mlp-k512-payload|\
    --prefill-mlp-k512-policy|--prefill-mlp-k512-receipt|\
    --prefill-mlp-k512-fragment-native-payload|\
    --prefill-mlp-k512-fragment-native-policy|\
    --prefill-mlp-k512-fragment-native-receipt|--mode)
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
  exact|native-gdn|cumulative-prefill|cumulative-prefill-down|cumulative-prefill-attention-down|cumulative-prefill-current-best|cumulative-prefill-current-best-k512|cumulative-prefill-current-best-mlp-k512|cumulative-prefill-current-best-mlp-k512-v1|cumulative-prefill-current-best-mlp-k512-edge|cumulative-prefill-current-best-mlp-k512-edge-attention-k256|cumulative-prefill-current-best-mlp-k512-edge-m128n64|cumulative-prefill-current-best-mlp-k512-down-m16n64-v2|cumulative-prefill-current-best-mlp-k512-fragment-native|cumulative-prefill-current-best-mlp-k512-fragment-native-m128|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged|cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta|cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta|cumulative-prefill-short) ;;
  *)
    echo "--mode must be exact, native-gdn, cumulative-prefill, or" \
      "cumulative-prefill-down, cumulative-prefill-attention-down, or" \
      "cumulative-prefill-current-best, cumulative-prefill-current-best-k512," \
      "cumulative-prefill-current-best-mlp-k512," \
      "cumulative-prefill-current-best-mlp-k512-v1," \
      "cumulative-prefill-current-best-mlp-k512-edge," \
      "cumulative-prefill-current-best-mlp-k512-edge-attention-k256," \
      "cumulative-prefill-current-best-mlp-k512-edge-m128n64," \
      "cumulative-prefill-current-best-mlp-k512-down-m16n64-v2," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-staged," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m64n128-1cta," \
      "cumulative-prefill-current-best-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta, or" \
      "cumulative-prefill-short" >&2
    exit 2
    ;;
esac
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "Q3X_PURE_PREFILL_DRY_RUN must be 0 or 1" >&2
  exit 2
}
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
mlp_k512_edge_m128n64_mode=0
mlp_k512_down_m16n64_v2_mode=0
attention_k256_mode=0
if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-v1 ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge ||
      "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ||
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
  if [[ "${mode}" == cumulative-prefill-current-best-mlp-k512-edge-attention-k256 ]]; then
    attention_k256_mode=1
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
  attention_k256_receipt_sha256=$(
    sha256sum "${prefill_a4_receipt}" | awk '{print $1}'
  )
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
if ((mlp_k512_args != 0 && mlp_k512_fragment_native_args != 0)); then
  echo "Prefill MLP K512 v1 and fragment-native v2 are mutually exclusive" >&2
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
if ((${#candidate_selectors[@]} > 0)); then
  for selector in "${candidate_selectors[@]}"; do
    if ! grep -F "${selector}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not contain the ${mode} selector: ${selector}" >&2
      exit 2
    fi
  done
fi
if [[ "${attention_k256_mode}" == 1 ]]; then
  attention_k256_markers=(
    prefill_projection_span_linear_qkv_z_k256_m128n256
    prefill_projection_span_linear_output_k256_m128n256
    prefill_projection_span_full_q_k_v_k256_m128n256
    prefill_projection_span_full_output_k256_m128n256
  )
  for marker in "${attention_k256_markers[@]}"; do
    if ! grep -Fx "${marker}" < <(strings -a "${server}") >/dev/null; then
      echo "server does not prove the K256 Attention production stage: ${marker}" >&2
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

printf 'pure_prefill_matrix mode=%s dry_run=%s sanitized_experiment_env=%s selector_count=%s\n' \
  "${mode}" "${dry_run}" "${sanitized}" "${#candidate_selectors[@]}"
server_elf_sha256=$(sha256sum "${server}" | awk '{print $1}')
printf 'server_metadata elf_sha256=%s evalscope_version=1.9.1\n' \
  "${server_elf_sha256}"
if [[ "${attention_k256_mode}" == 1 ]]; then
  printf 'attention_k256_publication_metadata layout=%s payload_bytes=%s payload_sha256=%s policy_sha256=%s receipt_sha256=%s manifest_sha256=%s expected_launch_hits=128 expected_logical_projections=208\n' \
    "${attention_k256_layout}" "${attention_k256_payload_bytes}" \
    "${attention_k256_payload_sha256}" \
    "${attention_k256_policy_sha256}" \
    "${attention_k256_receipt_sha256}" \
    "${attention_k256_manifest_sha256}"
  printf 'attention_k256_mlp_binding_metadata layout=%s payload_bytes=%s manifest_sha256=%s policy_sha256=%s payload_sha256=%s\n' \
    "${attention_k256_overlay_layout}" \
    "${attention_k256_overlay_payload_bytes}" \
    "${attention_k256_overlay_manifest_sha256}" \
    "${attention_k256_overlay_policy_sha256}" \
    "${attention_k256_overlay_payload_sha256}"
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
printf 'profile_metadata enabled=%s request_index=%s nsys_output=%q trace=cuda,nvtx capture_range=cudaProfilerApi\n' \
  "$([[ -n "${nsys_output}" ]] && printf 1 || printf 0)" \
  "${profile_request_index}" "${nsys_output}"
printf 'selector_metadata mode=%s selector_count=%s' \
  "${mode}" "${#candidate_selectors[@]}"
for selector in "${candidate_selectors[@]}"; do
  printf ' %s' "${selector}"
done
printf '\n'
if [[ "${attention_k256_mode}" == 1 ]]; then
  printf 'stage_contract required=prefill_projection_span_linear_qkv_z_k256_m128n256,prefill_projection_span_linear_output_k256_m128n256,prefill_projection_span_full_q_k_v_k256_m128n256,prefill_projection_span_full_output_k256_m128n256 excluded=prefill_projection_span_linear_qkv_z_supermatrix,prefill_projection_span_linear_output_supermatrix,prefill_projection_span_full_q_k_v_supermatrix,prefill_projection_span_full_output_supermatrix\n'
fi
if [[ "${mlp_k512_edge_mode}" == 1 &&
      "${mlp_k512_down_m16n64_v2_mode}" == 0 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge excluded=prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize retained=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
fi
if [[ "${mlp_k512_edge_m128n64_mode}" == 1 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge_m128n64 excluded=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize,prefill_projection_span_mlp_k512_down_m16n64_v2 retained=prefill_projection_span_mlp_k512_input_quantize,prefill_projection_span_mlp_k512_down\n'
fi
if [[ "${mlp_k512_down_m16n64_v2_mode}" == 1 ]]; then
  printf 'stage_contract required=prefill_projection_span_mlp_k512_gateup_down_edge,prefill_projection_span_mlp_k512_down_m16n64_v2 excluded=prefill_projection_span_mlp_k512_down,prefill_projection_span_mlp_k512_gate_up_primary,prefill_projection_span_mlp_k512_gate_up_secondary,prefill_projection_span_mlp_k512_product_quantize retained=prefill_projection_span_mlp_k512_input_quantize\n'
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
  printf ',prefill_a4_k256_authenticated_400_of_400,attention_k256_expected_launch_hits_128,attention_k256_expected_logical_projections_208'
fi
if [[ "${mlp_k512_edge_mode}" == 1 ]]; then
  printf ',prefill_projection_span_mlp_k512_gateup_down_edge,old_gateup_split_stages_excluded'
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
  printf 'profile_request_index=%s\n' "${profile_request_index}"
  printf 'nsys_output=%s\n' "${nsys_output}"
  printf 'selectors='
  printf '%s ' "${candidate_selectors[@]}"
  printf '\n'
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
    printf 'prefill_attention_k256_expected_launch_hits=128\n'
    printf 'prefill_attention_k256_expected_logical_projections=208\n'
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
printf '\n'

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
  mapfile -t summary_files < <(
    find "${run_dir}" -name benchmark_summary.json -type f -print
  )
  [[ ${#summary_files[@]} == 1 ]] || {
    echo "expected exactly one EvalScope benchmark_summary.json under ${run_dir}" >&2
    exit 6
  }
  summary_file=${summary_files[0]}
  python3 - "${summary_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
total = summary.get("Total Requests")
success = summary.get("Success Requests")
failed = summary.get("Failed Requests")
if total != 4 or success != 4 or failed != 0:
    raise SystemExit(
        f"EvalScope request contract failed: total={total} "
        f"success={success} failed={failed}"
    )
PY
  printf 'evalscope_request_contract bucket=%s total=4 success=4 failed=0 summary=%q\n' \
    "${bucket}" "${summary_file}"
  printf '%s\n' "${summary_file}"
  sed -n '1,220p' "${summary_file}"
done
