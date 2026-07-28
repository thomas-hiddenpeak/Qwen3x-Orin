#include "pinned_checkpoint.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace checkpoint = q3x::test::support;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

}  // namespace

int main(const int argc, char** argv) {
  TestContext test;
  const checkpoint::PinnedModelRevision& model =
      checkpoint::qwen36_27b_nvfp4_model_revision();
  const checkpoint::PinnedBundleDescriptor& bundle =
      checkpoint::qwen36_27b_nvfp4_layer0_mlp_bundle();

  test.expect(model.id == "nvidia-qwen3.6-27b-nvfp4@0893e160",
              "Qwen3.6 model descriptor id is pinned");
  test.expect(
      model.revision == "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
      "Qwen3.6 model revision is pinned");
  test.expect(model.metadata_files.size() == 3U && model.shards.size() == 1U,
              "metadata and layer-0 shard identities are pinned");
  test.expect(bundle.model == &model && bundle.tensors.size() == 9U &&
                  bundle.require_single_shard,
              "layer-0 Gate/Up/Down bundle has nine one-shard tensors");

  const checkpoint::PinnedTensor* const gate =
      checkpoint::find_pinned_tensor(bundle,
                                     checkpoint::kQwen36Layer0GateWeight);
  const checkpoint::PinnedTensor* const up =
      checkpoint::find_pinned_tensor(bundle,
                                     checkpoint::kQwen36Layer0UpWeight);
  const checkpoint::PinnedTensor* const down =
      checkpoint::find_pinned_tensor(bundle,
                                     checkpoint::kQwen36Layer0DownWeight);
  test.expect(gate != nullptr && up != nullptr && down != nullptr,
              "Gate/Up/Down weight pins are discoverable");
  if (gate != nullptr && up != nullptr && down != nullptr) {
    test.expect(down->file_end == gate->file_begin &&
                    gate->file_end == up->file_begin,
                "pinned layer-0 NVFP4 weights have exact adjacent ranges");
  }

  const checkpoint::PinnedCheckpointError sample_error{
      checkpoint::PinnedCheckpointErrorCode::kPayloadHashMismatch,
      "tensor.sha256", {}, "tensor", "expected", "actual", "mismatch"};
  const std::string described =
      checkpoint::describe_pinned_checkpoint_error(sample_error);
  test.expect(described.find("payload-hash-mismatch") != std::string::npos &&
                  described.find("tensor=tensor") != std::string::npos,
              "structured errors are printable without exiting");

  if (argc == 2 && argv[1] != nullptr && argv[1][0] != '\0') {
    checkpoint::PinnedBundleLoadOptions options;
    options.tensor_names = {
        std::string(checkpoint::kQwen36Layer0GateWeightScale2),
        std::string(checkpoint::kQwen36Layer0UpWeightScale2),
        std::string(checkpoint::kQwen36Layer0DownWeightScale2),
    };
    options.maximum_total_payload_bytes = 1U * 1024U * 1024U;
    const checkpoint::PinnedBundleLoadResult loaded =
        checkpoint::load_pinned_checkpoint_bundle(
            std::filesystem::path(argv[1]), bundle, options);
    if (!loaded) {
      test.expect(false, "real checkpoint scalar bundle loads");
      if (loaded.error.has_value()) {
        std::cerr << checkpoint::describe_pinned_checkpoint_error(
                         *loaded.error)
                  << '\n';
      }
    } else {
      test.expect(loaded.value->tensors.size() == 3U &&
                      loaded.value->metadata_files.size() == 3U &&
                      loaded.value->shards.size() == 1U,
                  "real checkpoint load records complete provenance");
      const checkpoint::PinnedTensorPayload* const gate_scale2 =
          loaded.value->find_tensor(
              checkpoint::kQwen36Layer0GateWeightScale2);
      const checkpoint::PinnedTensorPayload* const up_scale2 =
          loaded.value->find_tensor(
              checkpoint::kQwen36Layer0UpWeightScale2);
      const checkpoint::PinnedTensorPayload* const down_scale2 =
          loaded.value->find_tensor(
              checkpoint::kQwen36Layer0DownWeightScale2);
      test.expect(gate_scale2 != nullptr && up_scale2 != nullptr &&
                      down_scale2 != nullptr &&
                      gate_scale2->f32_bits == 0x391e79e8U &&
                      up_scale2->f32_bits == 0x391e79e8U &&
                      down_scale2->f32_bits == 0x39b61862U,
                  "real checkpoint scalar bits match all three exact pins");
    }
  }

  if (test.failures() != 0) {
    std::cerr << test.failures() << " pinned checkpoint support tests failed\n";
    return 1;
  }
  std::cout << "pinned checkpoint support tests passed\n";
  return 0;
}
