#pragma once

#include <string>

namespace q3x::kernels {

// Launches a tiny CUDA kernel and verifies a device-to-host result. This is a
// build/runtime smoke test, not a performance benchmark.
[[nodiscard]] bool RunDeviceSmokeTest(std::string* error_message = nullptr);

}  // namespace q3x::kernels
