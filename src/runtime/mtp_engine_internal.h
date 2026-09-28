#pragma once

namespace q3x::runtime {
class ReferenceEngine;
class ReferenceRunner;
class RequestState;
class ModelWeights;
namespace mtp_detail {
// Borrowed, serialized test access. Definitions exist only in internal-test
// builds. The engine must outlive every draft/transaction using these owners.
class EngineAccess {
 public:
  static ReferenceRunner* runner(ReferenceEngine&) noexcept;
  static RequestState* state(ReferenceEngine&) noexcept;
  static const ModelWeights* model(const ReferenceEngine&) noexcept;
};
}
}
