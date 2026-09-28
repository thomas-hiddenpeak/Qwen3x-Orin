#include "model/mtp_weights_internal.h"

#include <cstdlib>
#include <iostream>

namespace {
namespace mtp = q3x::model::mtp_detail;
namespace weights = q3x::model::weights;
int checks = 0;
void require(bool value) {
  ++checks;
  if (!value) {
    std::cerr << "MTP weights assertion " << checks << " failed\n";
    std::exit(1);
  }
}

weights::WeightManifest fixture() {
  weights::WeightManifest manifest;
  std::uint64_t offset = 8192;
  for (const auto& spec : mtp::kTensorSpecs) {
    weights::TensorLocator locator;
    locator.category = weights::TensorCategory::kMtp;
    locator.dtype = q3x::io::safetensors::DType::kBf16;
    locator.shape.push_back(spec.rows);
    if (spec.columns != 0) locator.shape.push_back(spec.columns);
    locator.byte_size = 2 * spec.rows * (spec.columns == 0 ? 1 : spec.columns);
    locator.file_begin = offset;
    offset += locator.byte_size;
    locator.file_end = offset;
    manifest.tensors.emplace(spec.name, std::move(locator));
  }
  return manifest;
}

void validate(const weights::WeightManifest& manifest) {
  const auto result = mtp::plan_weights(manifest);
  require(result.value.has_value() && result.error == mtp::PlanError::kNone);
  require(result.value->arena_bytes == 849398784);
  std::uint64_t end = 0;
  for (std::size_t i = 0; i < mtp::kTensorSpecs.size(); ++i) {
    const auto& entry = result.value->tensors[i];
    require(entry.arena_offset == end && entry.arena_offset % 256 == 0);
    require(entry.locator == &manifest.tensors.find(mtp::kTensorSpecs[i].name)->second);
    end += entry.locator->byte_size;
  }
  require(end == result.value->arena_bytes);
}

void synthetic() {
  validate(fixture());
  for (const auto& spec : mtp::kTensorSpecs) {
    for (unsigned mutation = 0; mutation < 7; ++mutation) {
      auto manifest = fixture();
      auto& entry = manifest.tensors.find(spec.name)->second;
      switch (mutation) {
        case 0: manifest.tensors.erase(std::string(spec.name)); break;
        case 1: entry.category = weights::TensorCategory::kText; break;
        case 2: entry.dtype = q3x::io::safetensors::DType::kF16; break;
        case 3: entry.shape[0] += 1; break;
        case 4: entry.shape.clear(); break;
        case 5: entry.byte_size += 1; break;
        case 6: entry.file_end = entry.file_begin - 1; break;
      }
      const auto result = mtp::plan_weights(manifest);
      require(!result.value.has_value() && result.error != mtp::PlanError::kNone);
    }
  }
  auto manifest = fixture();
  manifest.tensors.emplace("mtp.unexpected.weight", manifest.tensors.begin()->second);
  require(!mtp::plan_weights(manifest).value.has_value());
  manifest = fixture();
  auto node = manifest.tensors.extract(manifest.tensors.begin());
  node.key() = "mtp.other.weight";
  manifest.tensors.insert(std::move(node));
  require(!mtp::plan_weights(manifest).value.has_value());
  manifest = fixture();
  manifest.tensors.emplace("model.extra.weight", manifest.tensors.begin()->second);
  require(!mtp::plan_weights(manifest).value.has_value());
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 2) return 2;
  synthetic();
  if (argc == 2) {
    const auto manifest = weights::build_qwen36_27b_text_manifest(argv[1]);
    if (!manifest) {
      for (const auto& diagnostic : manifest.diagnostics) {
        std::cerr << diagnostic.context << ": " << diagnostic.message << '\n';
      }
      return 1;
    }
    validate(*manifest.value);
    const auto plan = mtp::plan_weights(*manifest.value);
    std::cout << "checkpoint_header_contract=pass\n"
                 "payload_authentication=not_performed\n"
                 "mtp_tensor_count=15\nmtp_arena_bytes="
              << plan.value->arena_bytes << '\n';
    for (std::size_t i = 0; i < mtp::kTensorSpecs.size(); ++i) {
      const auto& locator = *plan.value->tensors[i].locator;
      std::cout << mtp::kTensorSpecs[i].name << ' ' << locator.shard << ' '
                << locator.file_begin << ' ' << locator.file_end << ' '
                << plan.value->tensors[i].arena_offset << '\n';
    }
  }
  std::cout << "mtp_weights_checks=" << checks << '\n';
}
