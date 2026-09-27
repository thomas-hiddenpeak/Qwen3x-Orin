#include "q3x/io/json.h"
#include "q3x/server/openai_protocol.h"
#include <iostream>
int main() {
  q3x::server::TargetPrefillWitnessRecord record;
  record.prompt_tokens = 40000;
  record.consumed_prompt_tokens = 40000;
  record.completion_tokens = 256;
  record.prefill_route_evidence.valid = true;
  record.prefill_route_evidence.complete = true;
  record.prefill_route_evidence.operators[0].exact_fallback_hits = 16;
  const auto text = q3x::server::serialize_target_prefill_witness(record);
  if (!q3x::io::json::parse(text) ||
#if defined(Q3X_ENABLE_WHOLE_CORE_EXACT_DECODE_ADMISSION)
      text.find("target-prefill-witness-whole-core-exact-decode-admission-v2") == std::string::npos ||
      text.find("\"schema_version\":2") == std::string::npos ||
#else
      text.find("target-prefill-witness-fused-decode-admission-v7") == std::string::npos ||
      text.find("\"schema_version\":7") == std::string::npos ||
#endif
      text.find("ordered-pipeline-unqualified") == std::string::npos ||
      text.find("\"completed_fallback_hits_unqualified\":16") == std::string::npos ||
      text.find("completed_exact_fallback_hits") != std::string::npos ||
      text.find("\"scope\":\"architecture_candidate_unqualified\"") == std::string::npos ||
      text.find("\"approximate_numerics\":false") == std::string::npos) {
    std::cerr << "Admission witness falsely qualified a numerical fallback\n";
    return 1;
  }
  std::cout << "PASS unqualified fused witness preserves counts without exactness claims\n";
}
