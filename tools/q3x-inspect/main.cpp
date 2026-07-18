#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/resident_weights.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <locale>
#include <map>
#include <new>
#include <ostream>
#include <string>
#include <string_view>

namespace {

namespace st = q3x::io::safetensors;
namespace checkpoint = q3x::model::checkpoint;
namespace weights = q3x::model::weights;
namespace runtime = q3x::runtime;

constexpr int kExitSuccess = 0;
constexpr int kExitUsage = 2;
constexpr int kExitIo = 3;
constexpr int kExitInvalidInput = 4;
constexpr int kExitInternal = 5;

void WriteQuoted(std::ostream& output, std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    output.put('"');
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        switch (byte) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (byte < 0x20U) {
                    output << "\\u00" << kHex[(byte >> 4U) & 0x0FU]
                           << kHex[byte & 0x0FU];
                } else {
                    output.put(static_cast<char>(byte));
                }
                break;
        }
    }
    output.put('"');
}

void PrintUsage(std::ostream& output) {
    output
        << "qwen3x-inspect: bounded safetensors metadata inspection\n\n"
        << "Usage:\n"
        << "  qwen3x-inspect header FILE   Validate and summarize one shard header\n"
        << "  qwen3x-inspect index FILE    Validate and summarize a shard index\n"
        << "  qwen3x-inspect checkpoint DIR [--require-shards]\n"
        << "                               Validate pinned model metadata; the\n"
        << "                               option strictly validates every shard\n"
        << "                               header and index tensor contract\n"
        << "  qwen3x-inspect manifest DIR  Validate the pinned Qwen3.6-27B\n"
        << "                               text-only tensor ABI and summarize\n"
        << "                               payload/arena requirements\n"
        << "  qwen3x-inspect load-plan DIR Validate the authenticated resident\n"
        << "                               layout without allocating GPU memory\n"
        << "  qwen3x-inspect help          Show this help\n\n"
        << "The header command reads only the safetensors prefix and JSON header; "
           "tensor payload bytes are not read. Tensor names are not printed.\n"
        << "Checkpoint inspection without --require-shards remains metadata-only: "
           "missing shards are warnings. Strict inspection reads every shard "
           "header but never its payload.\n\n"
        << "Exit status:\n"
        << "  0  success\n"
        << "  2  command-line usage error\n"
        << "  3  file open or I/O error\n"
        << "  4  malformed or unsupported checkpoint metadata\n"
        << "  5  internal or resource failure\n";
}

std::string_view ErrorCodeName(st::ErrorCode code) noexcept {
    switch (code) {
        case st::ErrorCode::kNone:
            return "none";
        case st::ErrorCode::kInvalidOption:
            return "invalid_option";
        case st::ErrorCode::kOpenFailed:
            return "open_failed";
        case st::ErrorCode::kIoFailure:
            return "io_failure";
        case st::ErrorCode::kFileTooSmall:
            return "file_too_small";
        case st::ErrorCode::kFileTooLarge:
            return "file_too_large";
        case st::ErrorCode::kHeaderTooLarge:
            return "header_too_large";
        case st::ErrorCode::kInvalidHeaderLength:
            return "invalid_header_length";
        case st::ErrorCode::kInvalidHeaderStart:
            return "invalid_header_start";
        case st::ErrorCode::kInvalidJson:
            return "invalid_json";
        case st::ErrorCode::kHeaderNotObject:
            return "header_not_object";
        case st::ErrorCode::kTooManyTensors:
            return "too_many_tensors";
        case st::ErrorCode::kTensorDescriptorNotObject:
            return "tensor_descriptor_not_object";
        case st::ErrorCode::kMissingTensorField:
            return "missing_tensor_field";
        case st::ErrorCode::kUnknownTensorField:
            return "unknown_tensor_field";
        case st::ErrorCode::kUnsupportedDType:
            return "unsupported_dtype";
        case st::ErrorCode::kInvalidShape:
            return "invalid_shape";
        case st::ErrorCode::kRankLimitExceeded:
            return "rank_limit_exceeded";
        case st::ErrorCode::kArithmeticOverflow:
            return "arithmetic_overflow";
        case st::ErrorCode::kInvalidDataOffsets:
            return "invalid_data_offsets";
        case st::ErrorCode::kMisalignedTensor:
            return "misaligned_tensor";
        case st::ErrorCode::kTensorSizeMismatch:
            return "tensor_size_mismatch";
        case st::ErrorCode::kDataOutOfRange:
            return "data_out_of_range";
        case st::ErrorCode::kOverlappingData:
            return "overlapping_data";
        case st::ErrorCode::kDataGap:
            return "data_gap";
        case st::ErrorCode::kDataNotFullyCovered:
            return "data_not_fully_covered";
        case st::ErrorCode::kInvalidMetadata:
            return "invalid_metadata";
        case st::ErrorCode::kIndexTooLarge:
            return "index_too_large";
        case st::ErrorCode::kIndexNotObject:
            return "index_not_object";
        case st::ErrorCode::kMissingWeightMap:
            return "missing_weight_map";
        case st::ErrorCode::kInvalidWeightMap:
            return "invalid_weight_map";
        case st::ErrorCode::kTooManyShards:
            return "too_many_shards";
        case st::ErrorCode::kUnsafeShardPath:
            return "unsafe_shard_path";
        case st::ErrorCode::kShardMissing:
            return "shard_missing";
        case st::ErrorCode::kShardNotRegular:
            return "shard_not_regular";
        case st::ErrorCode::kShardSetMismatch:
            return "shard_set_mismatch";
        case st::ErrorCode::kUnexpectedTensor:
            return "unexpected_tensor";
        case st::ErrorCode::kTensorInWrongShard:
            return "tensor_in_wrong_shard";
        case st::ErrorCode::kMissingIndexedTensor:
            return "missing_indexed_tensor";
        case st::ErrorCode::kMissingTotalSize:
            return "missing_total_size";
        case st::ErrorCode::kPayloadSizeMismatch:
            return "payload_size_mismatch";
        case st::ErrorCode::kAllocationFailure:
            return "allocation_failure";
    }
    return "unknown";
}

int ExitCodeFor(st::ErrorCode code) noexcept {
    switch (code) {
        case st::ErrorCode::kOpenFailed:
        case st::ErrorCode::kIoFailure:
            return kExitIo;
        case st::ErrorCode::kAllocationFailure:
            return kExitInternal;
        default:
            return kExitInvalidInput;
    }
}

void PrintError(const st::Error& error) {
    std::cerr << "error.code=" << ErrorCodeName(error.code) << '\n'
              << "error.message=";
    WriteQuoted(std::cerr, error.message());
    std::cerr << "\nerror.context=";
    WriteQuoted(std::cerr, error.context);
    std::cerr << "\nerror.offset=";
    if (error.offset == st::kUnknownOffset) {
        std::cerr << "unknown";
    } else {
        std::cerr << error.offset;
    }
    std::cerr << '\n';
}

void PrintInternalError(std::string_view message) {
    std::cerr << "error.code=internal_error\nerror.message=";
    WriteQuoted(std::cerr, message);
    std::cerr << "\nerror.context=\"\"\nerror.offset=unknown\n";
}

int RunHeader(const std::string& path) {
    const auto result = st::read_header(path);
    if (!result) {
        PrintError(result.error);
        return ExitCodeFor(result.error.code);
    }

    std::map<std::string_view, std::size_t, std::less<>> dtype_counts;
    for (const auto& tensor : result.value->tensors) {
        ++dtype_counts[st::to_string(tensor.second.dtype)];
    }

    std::cout << "kind=header\nfile=";
    WriteQuoted(std::cout, path);
    std::cout << "\nfile_size=" << result.value->file_size
              << "\nheader_size=" << result.value->header_size
              << "\ndata_offset=" << result.value->data_offset
              << "\ndata_size=" << result.value->data_size
              << "\ntensor_count=" << result.value->tensors.size()
              << "\ndtype_count=" << dtype_counts.size() << '\n';

    std::size_t index = 0;
    for (const auto& dtype : dtype_counts) {
        std::cout << "dtype[" << index << "].name=";
        WriteQuoted(std::cout, dtype.first);
        std::cout << "\ndtype[" << index << "].tensor_count=" << dtype.second
                  << '\n';
        ++index;
    }

    std::cout << "metadata_count=" << result.value->metadata.size() << '\n';
    index = 0;
    for (const auto& metadata : result.value->metadata) {
        std::cout << "metadata[" << index << "].key=";
        WriteQuoted(std::cout, metadata.first);
        std::cout << "\nmetadata[" << index << "].value=";
        WriteQuoted(std::cout, metadata.second);
        std::cout << '\n';
        ++index;
    }

    if (!std::cout) {
        PrintInternalError("failed to write summary");
        return kExitInternal;
    }
    return kExitSuccess;
}

int RunIndex(const std::string& path) {
    const auto result = st::read_index(path);
    if (!result) {
        PrintError(result.error);
        return ExitCodeFor(result.error.code);
    }

    std::cout << "kind=index\nfile=";
    WriteQuoted(std::cout, path);
    std::cout << "\ntensor_count=" << result.value->weight_map.size()
              << "\nshard_count=" << result.value->shards.size()
              << "\ntotal_size=";
    if (result.value->total_size.has_value()) {
        std::cout << *result.value->total_size;
    } else {
        std::cout << "unknown";
    }
    std::cout << "\nmetadata_count=" << result.value->metadata.size() << '\n';

    for (std::size_t index = 0; index < result.value->shards.size(); ++index) {
        std::cout << "shard[" << index << "]=";
        WriteQuoted(std::cout, result.value->shards[index]);
        std::cout << '\n';
    }

    if (!std::cout) {
        PrintInternalError("failed to write summary");
        return kExitInternal;
    }
    return kExitSuccess;
}

int ExitCodeFor(const checkpoint::InspectionStatus status) noexcept {
    switch (status) {
        case checkpoint::InspectionStatus::kMetadataCompatible:
            return kExitSuccess;
        case checkpoint::InspectionStatus::kMissingRequiredFile:
        case checkpoint::InspectionStatus::kIoError:
        case checkpoint::InspectionStatus::kMissingShard:
            return kExitIo;
        case checkpoint::InspectionStatus::kInvalidMetadata:
        case checkpoint::InspectionStatus::kUnknownRevision:
        case checkpoint::InspectionStatus::kUnsupportedArchitecture:
        case checkpoint::InspectionStatus::kUnsupportedQuantization:
            return kExitInvalidInput;
    }
    return kExitInternal;
}

void PrintDiagnostic(const checkpoint::Diagnostic& diagnostic,
                     const std::size_t index,
                     std::ostream& output) {
    const std::string prefix = "diagnostic[" + std::to_string(index) + "].";
    output << prefix << "severity="
           << (diagnostic.severity == checkpoint::Severity::kError ? "error"
                                                                    : "warning")
           << '\n'
           << prefix << "code=" << checkpoint::to_string(diagnostic.code)
           << '\n'
           << prefix << "source=";
    WriteQuoted(output, diagnostic.source);
    output << '\n' << prefix << "json_pointer=";
    WriteQuoted(output, diagnostic.json_pointer);
    output << '\n' << prefix << "message=";
    WriteQuoted(output, diagnostic.message);
    output << '\n' << prefix << "expected=";
    WriteQuoted(output, diagnostic.expected);
    output << '\n' << prefix << "actual=";
    WriteQuoted(output, diagnostic.actual);
    output << '\n';
}

int RunCheckpoint(const std::string& directory, const bool require_shards) {
    checkpoint::InspectionOptions options;
    options.require_shards = require_shards;
    checkpoint::InspectionResult result =
        checkpoint::inspect_directory(directory, options);
    if (!result) {
        std::cerr << "kind=checkpoint\nstatus="
                  << checkpoint::to_string(result.status)
                  << "\ndiagnostic_count=" << result.diagnostics.size() << '\n';
        for (std::size_t index = 0; index < result.diagnostics.size(); ++index) {
            PrintDiagnostic(result.diagnostics[index], index, std::cerr);
        }
        return ExitCodeFor(result.status);
    }

    const checkpoint::InspectionReport& report = *result.report;
    std::cout << "kind=checkpoint\nstatus="
              << checkpoint::to_string(report.status) << "\ndescriptor=";
    WriteQuoted(std::cout, report.checkpoint.id);
    std::cout << "\nrepository=";
    WriteQuoted(std::cout, report.checkpoint.repository);
    std::cout << "\nrevision=";
    WriteQuoted(std::cout, report.checkpoint.revision);
    std::cout << "\nmodel=" << q3x::model::to_string(report.checkpoint.model)
              << "\narchitecture=";
    WriteQuoted(std::cout, report.config.architecture);
    std::cout << "\ntopology="
              << (report.config.num_experts == 0U ? "dense" : "moe")
              << "\nlayers=" << report.config.num_hidden_layers
              << "\nhidden_size=" << report.config.hidden_size
              << "\nmodelopt_producer=";
    WriteQuoted(std::cout, report.quantization.producer_name);
    std::cout << "\nmodelopt_version=";
    WriteQuoted(std::cout, report.quantization.producer_version);
    std::cout << "\nquantized_module_count="
              << report.quantization.quantized_modules.size()
              << "\nfp8_module_count=" << report.quantization.fp8_count
              << "\nnvfp4_module_count=" << report.quantization.nvfp4_count
              << "\nindex_tensor_count=" << report.index.tensor_count
              << "\nindex_shard_count=" << report.index.shards.size()
              << "\npresent_shards=" << report.present_shards
              << "\nmissing_shards=" << report.missing_shards
              << "\nshard_contract_validated="
              << (report.shard_contract_validated ? "true" : "false")
              << "\nvalidated_shards=" << report.validated_shards
              << "\nvalidated_tensors=" << report.validated_tensors
              << "\nvalidated_payload_bytes="
              << report.validated_payload_bytes
              << "\nfile_evidence_count=" << report.files.size() << '\n';
    for (std::size_t index = 0; index < report.files.size(); ++index) {
        std::cout << "file[" << index << "].name=";
        WriteQuoted(std::cout, report.files[index].name);
        std::cout << "\nfile[" << index << "].size="
                  << report.files[index].size << "\nfile[" << index
                  << "].sha256=";
        WriteQuoted(std::cout, report.files[index].sha256);
        std::cout << '\n';
    }
    std::cout << "diagnostic_count=" << report.diagnostics.size() << '\n';
    for (std::size_t index = 0; index < report.diagnostics.size(); ++index) {
        PrintDiagnostic(report.diagnostics[index], index, std::cout);
    }
    if (!report.checkpoint.known_metadata_quirk.empty()) {
        std::cout << "known_metadata_quirk=";
        WriteQuoted(std::cout, report.checkpoint.known_metadata_quirk);
        std::cout << '\n';
    }

    if (!std::cout) {
        PrintInternalError("failed to write checkpoint summary");
        return kExitInternal;
    }
    return kExitSuccess;
}

int ExitCodeFor(const weights::ManifestErrorCode code) noexcept {
    switch (code) {
        case weights::ManifestErrorCode::kIoFailure:
            return kExitIo;
        case weights::ManifestErrorCode::kAllocationFailure:
            return kExitInternal;
        case weights::ManifestErrorCode::kNone:
            return kExitSuccess;
        default:
            return kExitInvalidInput;
    }
}

void PrintManifestDiagnostic(const weights::ManifestDiagnostic& diagnostic,
                             const std::size_t index,
                             std::ostream& output) {
    const std::string prefix = "diagnostic[" + std::to_string(index) + "].";
    output << prefix << "code=" << weights::to_string(diagnostic.code) << '\n'
           << prefix << "context=";
    WriteQuoted(output, diagnostic.context);
    output << '\n' << prefix << "message=";
    WriteQuoted(output, diagnostic.message);
    output << '\n' << prefix << "expected=";
    WriteQuoted(output, diagnostic.expected);
    output << '\n' << prefix << "actual=";
    WriteQuoted(output, diagnostic.actual);
    output << '\n';
}

int RunManifest(const std::string& directory) {
    weights::ManifestResult result =
        weights::build_qwen36_27b_text_manifest(directory);
    if (!result) {
        std::cerr << "kind=manifest\nstatus=invalid\ndiagnostic_count="
                  << result.diagnostics.size() << '\n';
        for (std::size_t index = 0; index < result.diagnostics.size(); ++index) {
            PrintManifestDiagnostic(result.diagnostics[index], index, std::cerr);
        }
        return result.diagnostics.empty()
                   ? kExitInternal
                   : ExitCodeFor(result.diagnostics.front().code);
    }

    const weights::WeightManifest& manifest = *result.value;
    const weights::WeightManifestSummary& summary = manifest.summary;
    std::cout << "kind=manifest\nstatus=compatible\ndescriptor=";
    WriteQuoted(std::cout, manifest.checkpoint.id);
    std::cout << "\nshard_count=" << summary.shard_count
              << "\ntensor_count=" << summary.tensor_count
              << "\ntext_tensor_count=" << summary.text_tensor_count
              << "\nvision_tensor_count=" << summary.vision_tensor_count
              << "\nmtp_tensor_count=" << summary.mtp_tensor_count
              << "\nfp8_module_count=" << summary.fp8_module_count
              << "\nnvfp4_module_count=" << summary.nvfp4_module_count
              << "\nraw_text_bytes=" << summary.raw_text_bytes
              << "\nvision_skip_bytes=" << summary.vision_bytes
              << "\nmtp_skip_bytes=" << summary.mtp_bytes
              << "\nskipped_bytes=" << summary.skipped_bytes
              << "\narena_alignment=" << summary.arena_alignment
              << "\nestimated_text_arena_bytes="
              << summary.estimated_text_arena_bytes << '\n';
    if (!std::cout) {
        PrintInternalError("failed to write manifest summary");
        return kExitInternal;
    }
    return kExitSuccess;
}

int RunLoadPlan(const std::string& directory) {
    weights::ManifestResult manifest =
        weights::build_qwen36_27b_text_manifest(directory);
    if (!manifest) {
        std::cerr << "kind=load-plan\nstatus=invalid\ndiagnostic_count="
                  << manifest.diagnostics.size() << '\n';
        for (std::size_t index = 0; index < manifest.diagnostics.size(); ++index) {
            PrintManifestDiagnostic(manifest.diagnostics[index], index, std::cerr);
        }
        return manifest.diagnostics.empty()
                   ? kExitInternal
                   : ExitCodeFor(manifest.diagnostics.front().code);
    }
    runtime::PlanResult result = runtime::build_resident_load_plan(
        *manifest.value, runtime::pinned_qwen36_27b_shards());
    if (!result) {
        const runtime::ResidentLoadDiagnostic& diagnostic = result.diagnostic;
        std::cerr << "kind=load-plan\nstatus=invalid\nerror.code="
                  << runtime::to_string(diagnostic.code)
                  << "\nerror.message=";
        WriteQuoted(std::cerr, diagnostic.message);
        std::cerr << "\nerror.context=";
        WriteQuoted(std::cerr, diagnostic.context);
        std::cerr << "\nerror.shard=";
        WriteQuoted(std::cerr, diagnostic.shard);
        std::cerr << "\nerror.offset=" << diagnostic.offset << '\n';
        return diagnostic.code == runtime::ResidentLoadErrorCode::kAllocationFailure
                   ? kExitInternal
                   : kExitInvalidInput;
    }

    const runtime::ResidentLoadPlan& plan = *result.value;
    std::cout << "kind=load-plan\nstatus=compatible\nshard_count="
              << plan.shards.size()
              << "\ntext_tensor_count=" << plan.tensors.size()
              << "\nsource_bytes=" << plan.source_bytes
              << "\ncopied_bytes=" << plan.copied_bytes
              << "\nskipped_bytes=" << plan.source_bytes - plan.copied_bytes
              << "\narena_alignment=" << runtime::kResidentTensorAlignment
              << "\narena_bytes=" << plan.arena_bytes << '\n';
    for (std::size_t index = 0; index < plan.shards.size(); ++index) {
        const runtime::PlannedShard& shard = plan.shards[index];
        const std::string prefix = "shard[" + std::to_string(index) + "].";
        std::cout << prefix << "filename=";
        WriteQuoted(std::cout, shard.identity.filename);
        std::cout << '\n' << prefix << "file_size=" << shard.identity.file_size
                  << '\n' << prefix << "sha256=";
        WriteQuoted(std::cout, shard.identity.sha256);
        std::cout << '\n' << prefix << "text_tensor_count="
                  << shard.tensor_indices.size()
                  << '\n' << prefix << "copied_bytes=" << shard.copied_bytes
                  << '\n';
    }
    if (!std::cout) {
        PrintInternalError("failed to write load plan summary");
        return kExitInternal;
    }
    return kExitSuccess;
}

int UsageError(std::string_view message, std::string_view context = {}) {
    std::cerr << "error.code=usage_error\nerror.message=";
    WriteQuoted(std::cerr, message);
    std::cerr << "\nerror.context=";
    WriteQuoted(std::cerr, context);
    std::cerr << "\nerror.offset=unknown\n\n";
    PrintUsage(std::cerr);
    return kExitUsage;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    std::cerr.imbue(std::locale::classic());

    try {
        if (argc < 2) {
            return UsageError("missing command");
        }

        const std::string_view command(argv[1]);
        if (command == "help" || command == "--help" || command == "-h") {
            if (argc != 2) {
                return UsageError("help takes no arguments", command);
            }
            PrintUsage(std::cout);
            return std::cout ? kExitSuccess : kExitInternal;
        }
        if (command == "header") {
            if (argc != 3) {
                return UsageError("header requires exactly one FILE argument",
                                  command);
            }
            return RunHeader(argv[2]);
        }
        if (command == "index") {
            if (argc != 3) {
                return UsageError("index requires exactly one FILE argument",
                                  command);
            }
            return RunIndex(argv[2]);
        }
        if (command == "checkpoint") {
            if (argc != 3 && argc != 4) {
                return UsageError(
                    "checkpoint requires DIR and optional --require-shards");
            }
            const bool require_shards = argc == 4;
            if (require_shards &&
                std::string_view(argv[3]) != "--require-shards") {
                return UsageError("unknown checkpoint option");
            }
            return RunCheckpoint(argv[2], require_shards);
        }
        if (command == "manifest") {
            if (argc != 3) {
                return UsageError("manifest requires exactly one DIR argument",
                                  command);
            }
            return RunManifest(argv[2]);
        }
        if (command == "load-plan") {
            if (argc != 3) {
                return UsageError("load-plan requires exactly one DIR argument",
                                  command);
            }
            return RunLoadPlan(argv[2]);
        }
        return UsageError("unknown command", command);
    } catch (const std::bad_alloc&) {
        PrintInternalError("memory allocation failed");
        return kExitInternal;
    } catch (const std::exception& error) {
        PrintInternalError(error.what());
        return kExitInternal;
    } catch (...) {
        PrintInternalError("unknown internal failure");
        return kExitInternal;
    }
}
