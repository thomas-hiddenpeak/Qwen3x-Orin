#pragma once

#include "q3x/server/evaluation_server.h"

#include <string>

namespace q3x::server {

// Parses the evaluation-server command line without loading a model or
// touching CUDA. argv[0] is the executable name and argv[1] is MODEL_DIR.
// This narrow entry point exists so diagnostic-only switches can be covered
// by host tests without starting the server.
[[nodiscard]] bool parse_evaluation_server_arguments(
    int argc, const char* const* argv, EvaluationServerOptions& options,
    std::string& error);

}  // namespace q3x::server
