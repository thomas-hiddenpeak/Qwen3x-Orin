// The kernel and standalone harness share the single private implementation
// translation unit. Keeping main here leaves test ownership explicit without
// duplicating candidate code or exposing it through the installed API.
int q3x_gdn_prefill_c16_norm_gate_sm87_candidate_main();

int main() {
  return q3x_gdn_prefill_c16_norm_gate_sm87_candidate_main();
}
