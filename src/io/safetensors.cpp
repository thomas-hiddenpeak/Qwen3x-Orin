#include "q3x/io/safetensors.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace q3x::io::safetensors {

std::optional<DType> parse_dtype(std::string_view name) noexcept {
    if (name == "BOOL") {
        return DType::kBool;
    }
    if (name == "F4") {
        return DType::kF4;
    }
    if (name == "F6_E2M3") {
        return DType::kF6E2M3;
    }
    if (name == "F6_E3M2") {
        return DType::kF6E3M2;
    }
    if (name == "U8") {
        return DType::kU8;
    }
    if (name == "I8") {
        return DType::kI8;
    }
    if (name == "F8_E5M2") {
        return DType::kF8E5M2;
    }
    if (name == "F8_E4M3") {
        return DType::kF8E4M3;
    }
    if (name == "F8_E8M0") {
        return DType::kF8E8M0;
    }
    if (name == "F8_E4M3FNUZ") {
        return DType::kF8E4M3Fnuz;
    }
    if (name == "F8_E5M2FNUZ") {
        return DType::kF8E5M2Fnuz;
    }
    if (name == "I16") {
        return DType::kI16;
    }
    if (name == "U16") {
        return DType::kU16;
    }
    if (name == "F16") {
        return DType::kF16;
    }
    if (name == "BF16") {
        return DType::kBf16;
    }
    if (name == "I32") {
        return DType::kI32;
    }
    if (name == "U32") {
        return DType::kU32;
    }
    if (name == "F32") {
        return DType::kF32;
    }
    if (name == "C64") {
        return DType::kC64;
    }
    if (name == "F64") {
        return DType::kF64;
    }
    if (name == "I64") {
        return DType::kI64;
    }
    if (name == "U64") {
        return DType::kU64;
    }
    return std::nullopt;
}

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
        case DType::kBool:
            return "BOOL";
        case DType::kF4:
            return "F4";
        case DType::kF6E2M3:
            return "F6_E2M3";
        case DType::kF6E3M2:
            return "F6_E3M2";
        case DType::kU8:
            return "U8";
        case DType::kI8:
            return "I8";
        case DType::kF8E5M2:
            return "F8_E5M2";
        case DType::kF8E4M3:
            return "F8_E4M3";
        case DType::kF8E8M0:
            return "F8_E8M0";
        case DType::kF8E4M3Fnuz:
            return "F8_E4M3FNUZ";
        case DType::kF8E5M2Fnuz:
            return "F8_E5M2FNUZ";
        case DType::kI16:
            return "I16";
        case DType::kU16:
            return "U16";
        case DType::kF16:
            return "F16";
        case DType::kBf16:
            return "BF16";
        case DType::kI32:
            return "I32";
        case DType::kU32:
            return "U32";
        case DType::kF32:
            return "F32";
        case DType::kC64:
            return "C64";
        case DType::kF64:
            return "F64";
        case DType::kI64:
            return "I64";
        case DType::kU64:
            return "U64";
    }
    return "unknown";
}

std::uint8_t bit_width(DType dtype) noexcept {
    switch (dtype) {
        case DType::kF4:
            return 4;
        case DType::kF6E2M3:
        case DType::kF6E3M2:
            return 6;
        case DType::kBool:
        case DType::kU8:
        case DType::kI8:
        case DType::kF8E5M2:
        case DType::kF8E4M3:
        case DType::kF8E8M0:
        case DType::kF8E4M3Fnuz:
        case DType::kF8E5M2Fnuz:
            return 8;
        case DType::kI16:
        case DType::kU16:
        case DType::kF16:
        case DType::kBf16:
            return 16;
        case DType::kI32:
        case DType::kU32:
        case DType::kF32:
            return 32;
        case DType::kC64:
        case DType::kF64:
        case DType::kI64:
        case DType::kU64:
            return 64;
    }
    return 0;
}

namespace {

template <typename T>
Result<T> failure(ErrorCode code,
                  std::uint64_t offset = kUnknownOffset,
                  std::string context = {}) {
    Result<T> result;
    result.error.code = code;
    result.error.offset = offset;
    result.error.context = std::move(context);
    return result;
}

template <typename T>
Result<T> success(T value) {
    Result<T> result;
    result.value.emplace(std::move(value));
    return result;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t& output) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool determine_file_size(std::ifstream& input, std::uint64_t& size) {
    input.seekg(0, std::ios::end);
    const auto position = input.tellg();
    if (position == std::streampos(-1)) {
        return false;
    }
    const auto offset = static_cast<std::streamoff>(position);
    if (offset < 0) {
        return false;
    }
    size = static_cast<std::uint64_t>(offset);
    input.seekg(0, std::ios::beg);
    return static_cast<bool>(input);
}

bool read_exact(std::ifstream& input, char* destination, std::size_t size) {
    if (size == 0) {
        return true;
    }
    if (size > static_cast<std::size_t>(
                   std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    const auto requested = static_cast<std::streamsize>(size);
    input.read(destination, requested);
    return input.gcount() == requested;
}

std::uint64_t decode_little_endian_u64(
    const std::array<unsigned char, 8>& bytes) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

const json::Value* find_member(const json::Value::Object& object,
                               std::string_view name) noexcept {
    const auto entry = object.find(name);
    return entry == object.end() ? nullptr : &entry->second;
}

bool is_known_tensor_field(std::string_view name) noexcept {
    return name == "dtype" || name == "shape" || name == "data_offsets";
}

struct DataRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    const std::string* name = nullptr;
};

Result<Header> parse_header_document(const json::Value& document,
                                     std::uint64_t file_size,
                                     std::uint64_t header_size,
                                     const ReadOptions& options) {
    const auto* root = document.as_object();
    if (root == nullptr) {
        return failure<Header>(ErrorCode::kHeaderNotObject, 8);
    }

    std::uint64_t data_offset = 0;
    if (!checked_add(8, header_size, data_offset) || data_offset > file_size) {
        return failure<Header>(ErrorCode::kInvalidHeaderLength, 0);
    }

    Header header;
    header.file_size = file_size;
    header.header_size = header_size;
    header.data_offset = data_offset;
    header.data_size = file_size - data_offset;

    std::size_t tensor_count = 0;
    for (const auto& entry : *root) {
        if (entry.first != "__metadata__") {
            ++tensor_count;
        }
    }
    if (tensor_count > options.max_tensors) {
        return failure<Header>(ErrorCode::kTooManyTensors, 8);
    }

    std::vector<DataRange> ranges;
    ranges.reserve(tensor_count);

    for (const auto& entry : *root) {
        const auto& name = entry.first;
        const auto& value = entry.second;
        if (name == "__metadata__") {
            const auto* metadata = value.as_object();
            if (metadata == nullptr) {
                return failure<Header>(ErrorCode::kInvalidMetadata,
                                       8,
                                       "__metadata__");
            }
            for (const auto& item : *metadata) {
                const auto* text = item.second.as_string();
                if (text == nullptr) {
                    return failure<Header>(ErrorCode::kInvalidMetadata,
                                           8,
                                           item.first);
                }
                header.metadata.emplace(item.first, *text);
            }
            continue;
        }

        const auto* descriptor = value.as_object();
        if (descriptor == nullptr) {
            return failure<Header>(ErrorCode::kTensorDescriptorNotObject,
                                   8,
                                   name);
        }
        for (const auto& field : *descriptor) {
            if (!is_known_tensor_field(field.first)) {
                return failure<Header>(ErrorCode::kUnknownTensorField,
                                       8,
                                       name + "." + field.first);
            }
        }

        const auto* dtype_value = find_member(*descriptor, "dtype");
        const auto* shape_value = find_member(*descriptor, "shape");
        const auto* offsets_value = find_member(*descriptor, "data_offsets");
        if (dtype_value == nullptr || shape_value == nullptr ||
            offsets_value == nullptr) {
            return failure<Header>(ErrorCode::kMissingTensorField, 8, name);
        }

        const auto* dtype_text = dtype_value->as_string();
        if (dtype_text == nullptr) {
            return failure<Header>(ErrorCode::kUnsupportedDType, 8, name);
        }
        const auto dtype = parse_dtype(*dtype_text);
        if (!dtype.has_value()) {
            return failure<Header>(ErrorCode::kUnsupportedDType,
                                   8,
                                   name + ":" + *dtype_text);
        }

        const auto* shape = shape_value->as_array();
        if (shape == nullptr) {
            return failure<Header>(ErrorCode::kInvalidShape, 8, name);
        }
        if (shape->size() > options.max_rank) {
            return failure<Header>(ErrorCode::kRankLimitExceeded, 8, name);
        }

        TensorInfo tensor;
        tensor.dtype = *dtype;
        tensor.shape.reserve(shape->size());
        tensor.element_count = 1;
        for (const auto& dimension_value : *shape) {
            const auto* number = dimension_value.as_number();
            std::uint64_t dimension = 0;
            if (number == nullptr || !number->to_uint64(dimension)) {
                return failure<Header>(ErrorCode::kInvalidShape, 8, name);
            }
            tensor.shape.push_back(dimension);
            if (!checked_multiply(tensor.element_count,
                                  dimension,
                                  tensor.element_count)) {
                return failure<Header>(ErrorCode::kArithmeticOverflow,
                                       8,
                                       name);
            }
        }

        std::uint64_t bit_count = 0;
        if (!checked_multiply(tensor.element_count,
                              static_cast<std::uint64_t>(bit_width(*dtype)),
                              bit_count)) {
            return failure<Header>(ErrorCode::kArithmeticOverflow, 8, name);
        }
        if ((bit_count % 8U) != 0) {
            return failure<Header>(ErrorCode::kMisalignedTensor, 8, name);
        }
        tensor.byte_size = bit_count / 8U;

        const auto* offsets = offsets_value->as_array();
        if (offsets == nullptr || offsets->size() != 2) {
            return failure<Header>(ErrorCode::kInvalidDataOffsets, 8, name);
        }
        const auto* begin_number = (*offsets)[0].as_number();
        const auto* end_number = (*offsets)[1].as_number();
        if (begin_number == nullptr || end_number == nullptr ||
            !begin_number->to_uint64(tensor.data_begin) ||
            !end_number->to_uint64(tensor.data_end) ||
            tensor.data_end < tensor.data_begin) {
            return failure<Header>(ErrorCode::kInvalidDataOffsets, 8, name);
        }
        if (tensor.data_end - tensor.data_begin != tensor.byte_size) {
            return failure<Header>(ErrorCode::kTensorSizeMismatch, 8, name);
        }
        if (tensor.data_begin > header.data_size ||
            tensor.data_end > header.data_size) {
            return failure<Header>(ErrorCode::kDataOutOfRange, 8, name);
        }
        if (!checked_add(header.data_offset,
                         tensor.data_begin,
                         tensor.file_begin) ||
            !checked_add(header.data_offset, tensor.data_end, tensor.file_end)) {
            return failure<Header>(ErrorCode::kArithmeticOverflow, 8, name);
        }

        auto inserted = header.tensors.emplace(name, std::move(tensor));
        const auto& stored = inserted.first->second;
        ranges.push_back(
            {stored.data_begin, stored.data_end, &inserted.first->first});
    }

    std::sort(ranges.begin(),
              ranges.end(),
              [](const DataRange& left, const DataRange& right) {
                  if (left.begin != right.begin) {
                      return left.begin < right.begin;
                  }
                  if (left.end != right.end) {
                      return left.end < right.end;
                  }
                  return *left.name < *right.name;
              });

    std::uint64_t cursor = 0;
    for (const auto& range : ranges) {
        if (range.begin < cursor) {
            return failure<Header>(ErrorCode::kOverlappingData,
                                   8,
                                   *range.name);
        }
        if (range.begin > cursor) {
            return failure<Header>(ErrorCode::kDataGap, 8, *range.name);
        }
        cursor = range.end;
    }
    if (cursor != header.data_size) {
        return failure<Header>(ErrorCode::kDataNotFullyCovered, 8);
    }

    return success(std::move(header));
}

Result<Index> parse_index_document(const json::Value& document,
                                   const IndexReadOptions& options) {
    const auto* root = document.as_object();
    if (root == nullptr) {
        return failure<Index>(ErrorCode::kIndexNotObject, 0);
    }
    const auto* weight_map_value = find_member(*root, "weight_map");
    if (weight_map_value == nullptr) {
        return failure<Index>(ErrorCode::kMissingWeightMap, 0, "weight_map");
    }
    const auto* weight_map = weight_map_value->as_object();
    if (weight_map == nullptr || weight_map->empty()) {
        return failure<Index>(ErrorCode::kInvalidWeightMap, 0, "weight_map");
    }
    if (weight_map->size() > options.max_tensors) {
        return failure<Index>(ErrorCode::kTooManyTensors, 0, "weight_map");
    }

    Index index;
    std::set<std::string, std::less<>> shards;
    for (const auto& item : *weight_map) {
        const auto* shard = item.second.as_string();
        if (item.first.empty() || shard == nullptr || shard->empty() ||
            shard->find('\0') != std::string::npos) {
            return failure<Index>(ErrorCode::kInvalidWeightMap, 0, item.first);
        }
        index.weight_map.emplace(item.first, *shard);
        shards.emplace(*shard);
        if (shards.size() > options.max_shards) {
            return failure<Index>(ErrorCode::kTooManyShards, 0, *shard);
        }
    }
    index.shards.assign(shards.begin(), shards.end());

    if (const auto* metadata_value = find_member(*root, "metadata")) {
        const auto* metadata = metadata_value->as_object();
        if (metadata == nullptr) {
            return failure<Index>(ErrorCode::kInvalidMetadata, 0, "metadata");
        }
        index.metadata = *metadata;
        if (const auto* total_size_value = find_member(*metadata, "total_size")) {
            const auto* number = total_size_value->as_number();
            std::uint64_t total_size = 0;
            if (number == nullptr || !number->to_uint64(total_size)) {
                return failure<Index>(ErrorCode::kInvalidMetadata,
                                      0,
                                      "metadata.total_size");
            }
            index.total_size = total_size;
        }
    }

    return success(std::move(index));
}

}  // namespace

const TensorInfo* Header::find_tensor(std::string_view name) const noexcept {
    const auto entry = tensors.find(name);
    return entry == tensors.end() ? nullptr : &entry->second;
}

Result<Header> read_header(const std::string& path, const ReadOptions& options) {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return failure<Header>(ErrorCode::kOpenFailed,
                                   kUnknownOffset,
                                   path);
        }

        std::uint64_t file_size = 0;
        if (!determine_file_size(input, file_size)) {
            return failure<Header>(ErrorCode::kIoFailure,
                                   kUnknownOffset,
                                   path);
        }
        if (file_size > options.max_file_bytes) {
            return failure<Header>(ErrorCode::kFileTooLarge,
                                   kUnknownOffset,
                                   path);
        }
        if (file_size < 8) {
            return failure<Header>(ErrorCode::kFileTooSmall, file_size, path);
        }

        std::array<unsigned char, 8> length_bytes{};
        if (!read_exact(input,
                        reinterpret_cast<char*>(length_bytes.data()),
                        length_bytes.size())) {
            return failure<Header>(ErrorCode::kIoFailure, 0, path);
        }
        const auto header_size = decode_little_endian_u64(length_bytes);
        if (header_size > options.max_header_bytes ||
            header_size > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max()) ||
            header_size > static_cast<std::uint64_t>(
                              std::numeric_limits<std::streamsize>::max())) {
            return failure<Header>(ErrorCode::kHeaderTooLarge, 0, path);
        }
        std::uint64_t data_offset = 0;
        if (!checked_add(8, header_size, data_offset) ||
            data_offset > file_size) {
            return failure<Header>(ErrorCode::kInvalidHeaderLength, 0, path);
        }

        std::string header_text(static_cast<std::size_t>(header_size), '\0');
        if (!read_exact(input, header_text.data(), header_text.size())) {
            return failure<Header>(ErrorCode::kIoFailure, 8, path);
        }
        if (header_text.empty() || header_text.front() != '{') {
            return failure<Header>(ErrorCode::kInvalidHeaderStart, 8, path);
        }

        const auto parsed = json::parse(header_text, options.json_options);
        if (!parsed) {
            std::uint64_t error_offset = 8;
            if (!checked_add(error_offset,
                             static_cast<std::uint64_t>(parsed.error.offset),
                             error_offset)) {
                error_offset = kUnknownOffset;
            }
            return failure<Header>(ErrorCode::kInvalidJson,
                                   error_offset,
                                   std::string(parsed.error.message()));
        }
        return parse_header_document(*parsed.value,
                                     file_size,
                                     header_size,
                                     options);
    } catch (const std::bad_alloc&) {
        return failure<Header>(ErrorCode::kAllocationFailure);
    } catch (const std::length_error&) {
        return failure<Header>(ErrorCode::kAllocationFailure);
    }
}

const std::string* Index::shard_for(std::string_view tensor_name) const noexcept {
    const auto entry = weight_map.find(tensor_name);
    return entry == weight_map.end() ? nullptr : &entry->second;
}

Result<Index> read_index(const std::string& path,
                         const IndexReadOptions& options) {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return failure<Index>(ErrorCode::kOpenFailed,
                                  kUnknownOffset,
                                  path);
        }

        std::uint64_t file_size = 0;
        if (!determine_file_size(input, file_size)) {
            return failure<Index>(ErrorCode::kIoFailure,
                                  kUnknownOffset,
                                  path);
        }
        if (file_size > options.max_file_bytes ||
            file_size > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()) ||
            file_size > static_cast<std::uint64_t>(
                            std::numeric_limits<std::streamsize>::max())) {
            return failure<Index>(ErrorCode::kIndexTooLarge,
                                  kUnknownOffset,
                                  path);
        }

        std::string text(static_cast<std::size_t>(file_size), '\0');
        if (!read_exact(input, text.data(), text.size())) {
            return failure<Index>(ErrorCode::kIoFailure, 0, path);
        }
        const auto parsed = json::parse(text, options.json_options);
        if (!parsed) {
            return failure<Index>(ErrorCode::kInvalidJson,
                                  static_cast<std::uint64_t>(parsed.error.offset),
                                  std::string(parsed.error.message()));
        }
        return parse_index_document(*parsed.value, options);
    } catch (const std::bad_alloc&) {
        return failure<Index>(ErrorCode::kAllocationFailure);
    } catch (const std::length_error&) {
        return failure<Index>(ErrorCode::kAllocationFailure);
    }
}

bool is_safe_relative_shard_path(const std::string_view path) noexcept {
    if (path.empty() || path.size() > 4096U || path.front() == '/' ||
        path.back() == '/' || path.find("//") != std::string_view::npos ||
        path.find('\\') != std::string_view::npos ||
        path.find(':') != std::string_view::npos) {
        return false;
    }
    for (const char character : path) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 0x20U || byte == 0x7FU) {
            return false;
        }
    }
    try {
        const std::filesystem::path candidate{std::string(path)};
        if (candidate.empty() || candidate.is_absolute() ||
            candidate.has_root_path() ||
            candidate.extension() != ".safetensors") {
            return false;
        }
        for (const auto& component : candidate) {
            if (component.empty() || component == "." || component == "..") {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

Result<ShardValidationSummary> validate_index_shards(
    const std::filesystem::path& directory,
    const Index& index,
    const ShardValidationOptions& options) {
    try {
        if (options.max_shards == 0 || options.max_tensors == 0 ||
            options.header_options.max_tensors == 0) {
            return failure<ShardValidationSummary>(ErrorCode::kInvalidOption);
        }
        if (index.weight_map.empty()) {
            return failure<ShardValidationSummary>(ErrorCode::kInvalidWeightMap,
                                                   kUnknownOffset,
                                                   "weight_map");
        }
        if (index.weight_map.size() > options.max_tensors) {
            return failure<ShardValidationSummary>(ErrorCode::kTooManyTensors,
                                                   kUnknownOffset,
                                                   "weight_map");
        }

        std::set<std::string_view, std::less<>> referenced_shards;
        for (const auto& mapping : index.weight_map) {
            if (mapping.first.empty() || mapping.second.empty()) {
                return failure<ShardValidationSummary>(
                    ErrorCode::kInvalidWeightMap,
                    kUnknownOffset,
                    mapping.first);
            }
            if (!is_safe_relative_shard_path(mapping.second)) {
                return failure<ShardValidationSummary>(
                    ErrorCode::kUnsafeShardPath,
                    kUnknownOffset,
                    mapping.second);
            }
            referenced_shards.emplace(mapping.second);
            if (referenced_shards.size() > options.max_shards) {
                return failure<ShardValidationSummary>(
                    ErrorCode::kTooManyShards,
                    kUnknownOffset,
                    mapping.second);
            }
        }

        if (index.shards.size() != referenced_shards.size()) {
            auto result = failure<ShardValidationSummary>(
                ErrorCode::kShardSetMismatch,
                kUnknownOffset,
                "shards");
            result.error.expected =
                std::to_string(referenced_shards.size()) + " referenced shards";
            result.error.actual =
                std::to_string(index.shards.size()) + " stored shards";
            return result;
        }
        auto stored_shard = index.shards.begin();
        for (const std::string_view referenced_shard : referenced_shards) {
            if (stored_shard == index.shards.end() ||
                *stored_shard != referenced_shard) {
                auto result = failure<ShardValidationSummary>(
                    ErrorCode::kShardSetMismatch,
                    kUnknownOffset,
                    "shards");
                result.error.expected = std::string(referenced_shard);
                result.error.actual =
                    stored_shard == index.shards.end() ? "<end>" : *stored_shard;
                return result;
            }
            ++stored_shard;
        }
        if (!index.total_size.has_value()) {
            return failure<ShardValidationSummary>(ErrorCode::kMissingTotalSize,
                                                   kUnknownOffset,
                                                   "metadata.total_size");
        }

        ReadOptions header_options = options.header_options;
        header_options.max_tensors =
            std::min(header_options.max_tensors, options.max_tensors);

        ShardValidationSummary summary;
        std::set<std::string_view, std::less<>> seen_tensors;
        for (const std::string_view shard : referenced_shards) {
            const std::filesystem::path relative_path{std::string(shard)};
            std::filesystem::path shard_path = directory;
            for (auto component = relative_path.begin();
                 component != relative_path.end(); ++component) {
                shard_path /= *component;
                std::error_code filesystem_error;
                const std::filesystem::file_status status =
                    std::filesystem::symlink_status(shard_path,
                                                    filesystem_error);
                if (filesystem_error) {
                    if (filesystem_error ==
                        std::errc::no_such_file_or_directory) {
                        return failure<ShardValidationSummary>(
                            ErrorCode::kShardMissing,
                            kUnknownOffset,
                            std::string(shard));
                    }
                    auto result = failure<ShardValidationSummary>(
                        ErrorCode::kIoFailure,
                        kUnknownOffset,
                        std::string(shard));
                    result.error.actual = filesystem_error.message();
                    return result;
                }
                if (!std::filesystem::exists(status)) {
                    return failure<ShardValidationSummary>(
                        ErrorCode::kShardMissing,
                        kUnknownOffset,
                        std::string(shard));
                }
                if (std::filesystem::is_symlink(status)) {
                    auto result = failure<ShardValidationSummary>(
                        ErrorCode::kUnsafeShardPath,
                        kUnknownOffset,
                        std::string(shard));
                    result.error.expected =
                        "path components contained within checkpoint root";
                    result.error.actual = shard_path.string();
                    return result;
                }
                const auto next = std::next(component);
                if (next != relative_path.end() &&
                    !std::filesystem::is_directory(status)) {
                    return failure<ShardValidationSummary>(
                        ErrorCode::kShardNotRegular,
                        kUnknownOffset,
                        std::string(shard));
                }
                if (next == relative_path.end() &&
                    !std::filesystem::is_regular_file(status)) {
                    return failure<ShardValidationSummary>(
                        ErrorCode::kShardNotRegular,
                        kUnknownOffset,
                        std::string(shard));
                }
            }

            Result<Header> header =
                read_header(shard_path.string(), header_options);
            if (!header) {
                Result<ShardValidationSummary> result;
                result.error = std::move(header.error);
                result.error.actual = std::move(result.error.context);
                result.error.context = std::string(shard);
                result.error.expected = "valid safetensors shard header";
                return result;
            }

            if (summary.validated_shards >= options.max_shards) {
                return failure<ShardValidationSummary>(
                    ErrorCode::kTooManyShards,
                    kUnknownOffset,
                    std::string(shard));
            }
            ++summary.validated_shards;
            if (!checked_add(summary.payload_bytes,
                             header.value->data_size,
                             summary.payload_bytes)) {
                return failure<ShardValidationSummary>(
                    ErrorCode::kArithmeticOverflow,
                    kUnknownOffset,
                    std::string(shard));
            }

            for (const auto& tensor : header.value->tensors) {
                const auto expected_mapping =
                    index.weight_map.find(tensor.first);
                if (expected_mapping == index.weight_map.end()) {
                    auto result = failure<ShardValidationSummary>(
                        ErrorCode::kUnexpectedTensor,
                        kUnknownOffset,
                        tensor.first);
                    result.error.expected = "tensor present in index weight_map";
                    result.error.actual = std::string(shard);
                    return result;
                }
                if (expected_mapping->second != shard) {
                    auto result = failure<ShardValidationSummary>(
                        ErrorCode::kTensorInWrongShard,
                        kUnknownOffset,
                        tensor.first);
                    result.error.expected = expected_mapping->second;
                    result.error.actual = std::string(shard);
                    return result;
                }
                if (!seen_tensors.emplace(expected_mapping->first).second) {
                    auto result = failure<ShardValidationSummary>(
                        ErrorCode::kUnexpectedTensor,
                        kUnknownOffset,
                        tensor.first);
                    result.error.expected = "tensor stored exactly once";
                    result.error.actual = "duplicate tensor across shards";
                    return result;
                }
                if (summary.validated_tensors >= options.max_tensors) {
                    return failure<ShardValidationSummary>(
                        ErrorCode::kTooManyTensors,
                        kUnknownOffset,
                        tensor.first);
                }
                ++summary.validated_tensors;
            }
        }

        for (const auto& mapping : index.weight_map) {
            if (seen_tensors.find(mapping.first) == seen_tensors.end()) {
                auto result = failure<ShardValidationSummary>(
                    ErrorCode::kMissingIndexedTensor,
                    kUnknownOffset,
                    mapping.first);
                result.error.expected = mapping.second;
                result.error.actual = "tensor absent from referenced shard";
                return result;
            }
        }
        if (summary.validated_tensors != index.weight_map.size()) {
            return failure<ShardValidationSummary>(
                ErrorCode::kArithmeticOverflow,
                kUnknownOffset,
                "validated_tensors");
        }
        if (summary.payload_bytes != *index.total_size) {
            auto result = failure<ShardValidationSummary>(
                ErrorCode::kPayloadSizeMismatch,
                kUnknownOffset,
                "metadata.total_size");
            result.error.expected = std::to_string(*index.total_size);
            result.error.actual = std::to_string(summary.payload_bytes);
            return result;
        }
        return success(std::move(summary));
    } catch (const std::bad_alloc&) {
        return failure<ShardValidationSummary>(ErrorCode::kAllocationFailure);
    } catch (const std::length_error&) {
        return failure<ShardValidationSummary>(ErrorCode::kAllocationFailure);
    } catch (const std::filesystem::filesystem_error& error) {
        auto result = failure<ShardValidationSummary>(ErrorCode::kIoFailure);
        result.error.actual = error.what();
        return result;
    }
}

std::string_view Error::message() const noexcept { return to_string(code); }

std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kNone:
            return "valid safetensors data";
        case ErrorCode::kInvalidOption:
            return "invalid safetensors read option";
        case ErrorCode::kOpenFailed:
            return "failed to open file";
        case ErrorCode::kIoFailure:
            return "file I/O failed";
        case ErrorCode::kFileTooSmall:
            return "file is too small for a safetensors prefix";
        case ErrorCode::kFileTooLarge:
            return "file exceeds the configured byte limit";
        case ErrorCode::kHeaderTooLarge:
            return "safetensors header exceeds the configured byte limit";
        case ErrorCode::kInvalidHeaderLength:
            return "safetensors header length is outside the file";
        case ErrorCode::kInvalidHeaderStart:
            return "safetensors header does not begin with an object";
        case ErrorCode::kInvalidJson:
            return "invalid JSON document";
        case ErrorCode::kHeaderNotObject:
            return "safetensors header root is not an object";
        case ErrorCode::kTooManyTensors:
            return "tensor count exceeds the configured limit";
        case ErrorCode::kTensorDescriptorNotObject:
            return "tensor descriptor is not an object";
        case ErrorCode::kMissingTensorField:
            return "tensor descriptor is missing a required field";
        case ErrorCode::kUnknownTensorField:
            return "tensor descriptor contains an unknown field";
        case ErrorCode::kUnsupportedDType:
            return "tensor dtype is missing, malformed, or unsupported";
        case ErrorCode::kInvalidShape:
            return "tensor shape must contain unsigned integer dimensions";
        case ErrorCode::kRankLimitExceeded:
            return "tensor rank exceeds the configured limit";
        case ErrorCode::kArithmeticOverflow:
            return "integer overflow while validating tensor metadata";
        case ErrorCode::kInvalidDataOffsets:
            return "tensor data_offsets must be two ordered unsigned integers";
        case ErrorCode::kMisalignedTensor:
            return "sub-byte tensor size is not byte aligned";
        case ErrorCode::kTensorSizeMismatch:
            return "tensor offsets do not match shape and dtype size";
        case ErrorCode::kDataOutOfRange:
            return "tensor data range is outside the file data buffer";
        case ErrorCode::kOverlappingData:
            return "tensor data ranges overlap";
        case ErrorCode::kDataGap:
            return "tensor data ranges leave an unindexed gap";
        case ErrorCode::kDataNotFullyCovered:
            return "tensor metadata does not cover the complete data buffer";
        case ErrorCode::kInvalidMetadata:
            return "checkpoint or index metadata is malformed";
        case ErrorCode::kIndexTooLarge:
            return "index JSON exceeds the configured byte limit";
        case ErrorCode::kIndexNotObject:
            return "index JSON root is not an object";
        case ErrorCode::kMissingWeightMap:
            return "index JSON has no weight_map";
        case ErrorCode::kInvalidWeightMap:
            return "index weight_map must map tensor names to shard names";
        case ErrorCode::kTooManyShards:
            return "shard count exceeds the configured limit";
        case ErrorCode::kUnsafeShardPath:
            return "index contains an unsafe relative shard path";
        case ErrorCode::kShardMissing:
            return "referenced shard is missing";
        case ErrorCode::kShardNotRegular:
            return "referenced shard is not a regular non-symlink file";
        case ErrorCode::kShardSetMismatch:
            return "index shard list disagrees with weight_map";
        case ErrorCode::kUnexpectedTensor:
            return "shard contains a tensor not mapped there by the index";
        case ErrorCode::kTensorInWrongShard:
            return "tensor is stored in a different shard than the index declares";
        case ErrorCode::kMissingIndexedTensor:
            return "index tensor is missing from its declared shard";
        case ErrorCode::kMissingTotalSize:
            return "index metadata.total_size is required for shard validation";
        case ErrorCode::kPayloadSizeMismatch:
            return "aggregate shard payload bytes differ from metadata.total_size";
        case ErrorCode::kAllocationFailure:
            return "memory allocation failed while reading checkpoint metadata";
    }
    return "unknown safetensors error";
}

}  // namespace q3x::io::safetensors
