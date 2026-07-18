#include "q3x/runtime/reference_engine.h"

#include <iostream>
#include <string_view>

int main() {
  const std::string_view error = q3x::runtime::to_string(
      q3x::runtime::ReferenceEngineError::kInvalidArgument);
  const std::string_view stop = q3x::runtime::to_string(
      q3x::runtime::ReferenceStopReason::kImEnd);
  if (error != "invalid_argument" || stop != "im_end") {
    return 1;
  }
  std::cout << "installed q3x::engine consumer linked successfully\n";
  return 0;
}
