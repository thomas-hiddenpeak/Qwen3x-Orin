#include "q3x/io/safetensors.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

class TestContext {
public:
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAILED: " << message << '\n';
        }
    }

    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_ = 0;
};

class TempFile {
public:
    explicit TempFile(std::string_view suffix) {
        static std::uint64_t counter = 0;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("q3x-io-test-" + std::to_string(tick) + "-" +
                 std::to_string(counter++) + std::string(suffix));
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    ~TempFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] bool write(const std::string& bytes) const {
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            return false;
        }
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return static_cast<bool>(output);
    }

    [[nodiscard]] std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

class TempDirectory {
public:
    TempDirectory() {
        static std::uint64_t counter = 0;
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("q3x-shard-contract-test-" + std::to_string(tick) + "-" +
                 std::to_string(counter++));
        std::error_code error;
        if (!std::filesystem::create_directory(path_, error) || error) {
            throw std::runtime_error("failed to create shard test directory");
        }
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

bool write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(output);
}

std::string prefix(std::uint64_t header_size) {
    std::string bytes(8, '\0');
    for (std::size_t index = 0; index < 8; ++index) {
        bytes[index] = static_cast<char>((header_size >> (index * 8U)) & 0xFFU);
    }
    return bytes;
}

std::string archive_bytes(std::string_view header, std::size_t data_size) {
    auto bytes = prefix(static_cast<std::uint64_t>(header.size()));
    bytes.append(header);
    bytes.append(data_size, static_cast<char>(0x5A));
    return bytes;
}

q3x::io::safetensors::Result<q3x::io::safetensors::Header> read_archive(
    std::string_view header,
    std::size_t data_size,
    const q3x::io::safetensors::ReadOptions& options = {}) {
    TempFile file(".safetensors");
    if (!file.write(archive_bytes(header, data_size))) {
        q3x::io::safetensors::Result<q3x::io::safetensors::Header> result;
        result.error.code = q3x::io::safetensors::ErrorCode::kIoFailure;
        return result;
    }
    return q3x::io::safetensors::read_header(file.string(), options);
}

q3x::io::safetensors::Result<q3x::io::safetensors::Index> read_index_text(
    std::string_view text,
    const q3x::io::safetensors::IndexReadOptions& options = {}) {
    TempFile file(".index.json");
    if (!file.write(std::string(text))) {
        q3x::io::safetensors::Result<q3x::io::safetensors::Index> result;
        result.error.code = q3x::io::safetensors::ErrorCode::kIoFailure;
        return result;
    }
    return q3x::io::safetensors::read_index(file.string(), options);
}

void test_dtype_catalog(TestContext& test) {
    using namespace q3x::io::safetensors;
    const std::vector<std::pair<std::string_view, std::uint8_t>> expected = {
        {"BOOL", 8},          {"F4", 4},          {"F6_E2M3", 6},
        {"F6_E3M2", 6},      {"U8", 8},          {"I8", 8},
        {"F8_E5M2", 8},      {"F8_E4M3", 8},     {"F8_E8M0", 8},
        {"F8_E4M3FNUZ", 8},  {"F8_E5M2FNUZ", 8}, {"I16", 16},
        {"U16", 16},         {"F16", 16},        {"BF16", 16},
        {"I32", 32},         {"U32", 32},        {"F32", 32},
        {"C64", 64},         {"F64", 64},        {"I64", 64},
        {"U64", 64},
    };
    for (const auto& item : expected) {
        const auto dtype = parse_dtype(item.first);
        test.expect(dtype.has_value(), "known safetensors dtype parses");
        if (dtype) {
            test.expect(to_string(*dtype) == item.first, "dtype round trips to its spelling");
            test.expect(bit_width(*dtype) == item.second, "dtype bit width is correct");
        }
    }
    test.expect(!parse_dtype("F4_E2M1").has_value(),
                "nonstandard dtype aliases are not silently accepted");
}

void test_valid_header(TestContext& test) {
    using namespace q3x::io::safetensors;
    const std::string header_text =
        R"({"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"__metadata__":{"format":"pt","producer":"q3x"},"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})";
    const auto result = read_archive(header_text, 12);
    test.expect(result.ok(), "valid safetensors header is accepted");
    if (!result) {
        return;
    }
    test.expect(result.value->file_size == 8 + header_text.size() + 12,
                "file size is recorded");
    test.expect(result.value->header_size == header_text.size(),
                "header size is recorded");
    test.expect(result.value->data_offset == 8 + header_text.size() &&
                    result.value->data_size == 12,
                "data section bounds are recorded");
    test.expect(result.value->tensors.size() == 2,
                "all tensor descriptors are retained");
    const auto* tensor = result.value->find_tensor("a");
    test.expect(tensor != nullptr && tensor->dtype == DType::kF16 &&
                    tensor->shape == std::vector<std::uint64_t>{2} &&
                    tensor->element_count == 2 && tensor->byte_size == 4 &&
                    tensor->data_begin == 0 && tensor->data_end == 4 &&
                    tensor->file_begin == result.value->data_offset,
                "tensor dimensions and relative/absolute ranges are derived");
    test.expect(result.value->metadata.at("format") == "pt" &&
                    result.value->metadata.at("producer") == "q3x",
                "string metadata is retained");
    test.expect(result.value->find_tensor("missing") == nullptr,
                "missing tensor lookup returns null");
}

void test_scalars_empty_and_subbyte(TestContext& test) {
    using namespace q3x::io::safetensors;
    auto result = read_archive(
        R"({"scalar":{"dtype":"I64","shape":[],"data_offsets":[0,8]}})",
        8);
    test.expect(result.ok() &&
                    result.value->find_tensor("scalar")->element_count == 1,
                "rank-zero scalar has one element");

    result = read_archive(
        R"({"empty":{"dtype":"F16","shape":[0,18446744073709551615],"data_offsets":[0,0]}})",
        0);
    test.expect(result.ok() &&
                    result.value->find_tensor("empty")->element_count == 0,
                "zero-sized dimensions produce an empty tensor without overflow");

    result = read_archive(
        R"({"packed":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}})",
        1);
    test.expect(result.ok() && result.value->find_tensor("packed")->byte_size == 1,
                "byte-aligned sub-byte tensor is accepted");

    result = read_archive(
        R"({"packed":{"dtype":"F4","shape":[1],"data_offsets":[0,0]}})",
        0);
    test.expect(!result && result.error.code == ErrorCode::kMisalignedTensor,
                "non-byte-aligned sub-byte tensor is rejected");
}

void expect_header_error(std::string_view header,
                         std::size_t data_size,
                         q3x::io::safetensors::ErrorCode code,
                         TestContext& test,
                         std::string_view message) {
    const auto result = read_archive(header, data_size);
    test.expect(!result && result.error.code == code, message);
}

void test_header_rejections(TestContext& test) {
    using ErrorCode = q3x::io::safetensors::ErrorCode;
    expect_header_error(
        R"({"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"F16","shape":[1],"data_offsets":[2,4]}})",
        4,
        ErrorCode::kInvalidJson,
        test,
        "duplicate tensor names are rejected by the JSON layer");
    expect_header_error(
        R"({"x":{"dtype":"NOT_REAL","shape":[1],"data_offsets":[0,1]}})",
        1,
        ErrorCode::kUnsupportedDType,
        test,
        "unsupported dtype is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}})",
        1,
        ErrorCode::kInvalidShape,
        test,
        "non-integer shape dimension is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}})",
        0,
        ErrorCode::kArithmeticOverflow,
        test,
        "shape product overflow is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[-1,0]}})",
        0,
        ErrorCode::kInvalidDataOffsets,
        test,
        "negative data offset is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}})",
        1,
        ErrorCode::kTensorSizeMismatch,
        test,
        "shape and data size mismatch is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
        1,
        ErrorCode::kDataOutOfRange,
        test,
        "out-of-range tensor bytes are rejected");
    expect_header_error(
        R"({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}})",
        6,
        ErrorCode::kOverlappingData,
        test,
        "overlapping tensor ranges are rejected");
    expect_header_error(
        R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[1],"data_offsets":[3,4]}})",
        4,
        ErrorCode::kDataGap,
        test,
        "interior data holes are rejected");
    expect_header_error(
        R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
        2,
        ErrorCode::kDataNotFullyCovered,
        test,
        "unindexed trailing data is rejected");
    expect_header_error(
        R"({"__metadata__":{"bad":1}})",
        0,
        ErrorCode::kInvalidMetadata,
        test,
        "non-string safetensors metadata is rejected");
    expect_header_error(
        R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"extra":0}})",
        1,
        ErrorCode::kUnknownTensorField,
        test,
        "unknown tensor descriptor fields are rejected");
}

void test_file_and_option_failures(TestContext& test) {
    using namespace q3x::io::safetensors;
    TempFile short_file(".safetensors");
    test.expect(short_file.write("short"), "short fixture is written");
    auto result = read_header(short_file.string());
    test.expect(!result && result.error.code == ErrorCode::kFileTooSmall,
                "file shorter than prefix is rejected");

    TempFile bad_length(".safetensors");
    test.expect(bad_length.write(prefix(100)), "bad-length fixture is written");
    result = read_header(bad_length.string());
    test.expect(!result && result.error.code == ErrorCode::kInvalidHeaderLength,
                "declared header outside file is rejected");

    TempFile bad_start(".safetensors");
    test.expect(bad_start.write(archive_bytes(" {}", 0)),
                "bad-start fixture is written");
    result = read_header(bad_start.string());
    test.expect(!result && result.error.code == ErrorCode::kInvalidHeaderStart,
                "header must begin with an object byte");

    ReadOptions options;
    options.max_header_bytes = 1;
    result = read_archive("{}", 0, options);
    test.expect(!result && result.error.code == ErrorCode::kHeaderTooLarge,
                "configured header byte limit is enforced");

    options = {};
    options.max_file_bytes = 9;
    result = read_archive("{}", 0, options);
    test.expect(!result && result.error.code == ErrorCode::kFileTooLarge,
                "configured file byte limit is enforced");

    options = {};
    options.max_tensors = 0;
    result = read_archive(
        R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
        1,
        options);
    test.expect(!result && result.error.code == ErrorCode::kTooManyTensors,
                "configured tensor count limit is enforced");

    options = {};
    options.max_rank = 1;
    result = read_archive(
        R"({"x":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]}})",
        1,
        options);
    test.expect(!result && result.error.code == ErrorCode::kRankLimitExceeded,
                "configured tensor rank limit is enforced");
}

void test_valid_index(TestContext& test) {
    using namespace q3x::io::safetensors;
    const auto result = read_index_text(
        R"({"metadata":{"total_size":9007199254740993,"format":"pt"},"weight_map":{"z":"model-00002-of-00002.safetensors","a":"model-00001-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}})");
    test.expect(result.ok(), "valid Hugging Face safetensors index is accepted");
    if (!result) {
        return;
    }
    test.expect(result.value->weight_map.size() == 3 &&
                    *result.value->shard_for("z") ==
                        "model-00002-of-00002.safetensors",
                "index weight map is retained");
    test.expect(result.value->shards ==
                    std::vector<std::string>{
                        "model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors"},
                "index shard list is sorted and deduplicated");
    test.expect(result.value->total_size == 9007199254740993ULL,
                "index total_size remains exact above 2^53");
    test.expect(result.value->metadata.at("format").as_string() != nullptr &&
                    *result.value->metadata.at("format").as_string() == "pt",
                "arbitrary index metadata is retained");
    test.expect(result.value->shard_for("missing") == nullptr,
                "missing index tensor lookup returns null");
}

void test_index_rejections(TestContext& test) {
    using namespace q3x::io::safetensors;
    auto result = read_index_text("[]");
    test.expect(!result && result.error.code == ErrorCode::kIndexNotObject,
                "index root must be an object");
    result = read_index_text(R"({"metadata":{}})");
    test.expect(!result && result.error.code == ErrorCode::kMissingWeightMap,
                "index requires weight_map");
    result = read_index_text(R"({"weight_map":{"x":1}})");
    test.expect(!result && result.error.code == ErrorCode::kInvalidWeightMap,
                "weight_map values must be shard strings");
    result = read_index_text(
        R"({"weight_map":{"x":"a","x":"b"}})");
    test.expect(!result && result.error.code == ErrorCode::kInvalidJson,
                "duplicate index tensor names are rejected by JSON layer");
    result = read_index_text(
        R"({"metadata":{"total_size":-1},"weight_map":{"x":"a"}})");
    test.expect(!result && result.error.code == ErrorCode::kInvalidMetadata,
                "negative index total_size is rejected");

    IndexReadOptions options;
    options.max_shards = 1;
    result = read_index_text(
        R"({"weight_map":{"x":"a","y":"b"}})", options);
    test.expect(!result && result.error.code == ErrorCode::kTooManyShards,
                "configured shard count limit is enforced");

    options = {};
    options.max_file_bytes = 2;
    result = read_index_text(R"({"weight_map":{"x":"a"}})", options);
    test.expect(!result && result.error.code == ErrorCode::kIndexTooLarge,
                "configured index byte limit is enforced");
}

q3x::io::safetensors::Index make_contract_index() {
    q3x::io::safetensors::Index index;
    index.weight_map.emplace("a", "model-00001-of-00002.safetensors");
    index.weight_map.emplace("b", "model-00002-of-00002.safetensors");
    index.weight_map.emplace("c", "model-00002-of-00002.safetensors");
    index.shards = {"model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors"};
    index.total_size = 5;
    return index;
}

void write_valid_contract_shards(const TempDirectory& directory,
                                 TestContext& test) {
    test.expect(
        write_file(
            directory.path() / "model-00001-of-00002.safetensors",
            archive_bytes(
                R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})",
                2)),
        "first contract shard is written");
    test.expect(
        write_file(
            directory.path() / "model-00002-of-00002.safetensors",
            archive_bytes(
                R"({"b":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"c":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}})",
                3)),
        "second contract shard is written");
}

void test_shard_contract_success_and_policy(TestContext& test) {
    using namespace q3x::io::safetensors;
    TempDirectory directory;
    write_valid_contract_shards(directory, test);
    const Index index = make_contract_index();
    const auto result = validate_index_shards(directory.path(), index);
    test.expect(result.ok(), "complete index/shard contract validates");
    if (result) {
        test.expect(result.value->validated_shards == 2 &&
                        result.value->validated_tensors == 3 &&
                        result.value->payload_bytes == 5,
                    "contract validator reports checked aggregate totals");
    }

    test.expect(is_safe_relative_shard_path("weights/model.safetensors"),
                "nested relative safetensors path is safe");
    for (const std::string_view unsafe : {
             "", "/tmp/model.safetensors", "../model.safetensors",
             "weights/../model.safetensors", "./model.safetensors",
             "weights\\model.safetensors", "C:model.safetensors",
             "model.bin", "weights//model.safetensors"}) {
        test.expect(!is_safe_relative_shard_path(unsafe),
                    "unsafe shard path is rejected by I/O policy");
    }

    Index unsafe;
    unsafe.weight_map.emplace("a", "../escape.safetensors");
    unsafe.shards.emplace_back("../escape.safetensors");
    unsafe.total_size = 1;
    const auto unsafe_result = validate_index_shards(directory.path(), unsafe);
    test.expect(!unsafe_result &&
                    unsafe_result.error.code == ErrorCode::kUnsafeShardPath,
                "validator rejects unsafe index paths before filesystem access");

    TempDirectory outside;
    test.expect(write_file(outside.path() / "escape.safetensors",
                           archive_bytes(
                               R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
                               1)),
                "symlink escape target is written");
    std::error_code symlink_error;
    std::filesystem::create_directory_symlink(
        outside.path(), directory.path() / "weights", symlink_error);
    test.expect(!symlink_error, "intermediate directory symlink is created");
    Index symlink_escape;
    symlink_escape.weight_map.emplace("a", "weights/escape.safetensors");
    symlink_escape.shards.emplace_back("weights/escape.safetensors");
    symlink_escape.total_size = 1;
    const auto symlink_result =
        validate_index_shards(directory.path(), symlink_escape);
    test.expect(!symlink_result &&
                    symlink_result.error.code == ErrorCode::kUnsafeShardPath,
                "intermediate symlink cannot escape the checkpoint root");
}

void test_shard_contract_header_failures(TestContext& test) {
    using namespace q3x::io::safetensors;
    TempDirectory directory;
    Index index;
    index.weight_map.emplace("a", "model.safetensors");
    index.shards.emplace_back("model.safetensors");
    index.total_size = 1;

    const std::string lfs_pointer =
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 1234\n";
    test.expect(write_file(directory.path() / "model.safetensors", lfs_pointer),
                "Git LFS pointer fixture is written");
    auto result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kHeaderTooLarge &&
                    result.error.context == "model.safetensors",
                "Git LFS pointer is rejected as an invalid shard header");

    test.expect(write_file(directory.path() / "model.safetensors",
                           prefix(2) + "[]"),
                "bad safetensors header fixture is written");
    result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kInvalidHeaderStart,
                "malformed shard header is propagated diagnostically");

    std::error_code error;
    std::filesystem::remove(directory.path() / "model.safetensors", error);
    result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kShardMissing,
                "missing referenced shard is rejected");

    std::filesystem::create_directory(directory.path() / "model.safetensors",
                                      error);
    result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kShardNotRegular,
                "non-regular referenced shard is rejected");
}

void test_shard_contract_membership_failures(TestContext& test) {
    using namespace q3x::io::safetensors;

    {
        TempDirectory directory;
        test.expect(write_file(
                        directory.path() / "one.safetensors",
                        archive_bytes(
                            R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
                            1)),
                    "wrong-shard first fixture is written");
        test.expect(write_file(
                        directory.path() / "two.safetensors",
                        archive_bytes(
                            R"({"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
                            1)),
                    "wrong-shard second fixture is written");
        Index index;
        index.weight_map.emplace("a", "two.safetensors");
        index.weight_map.emplace("b", "one.safetensors");
        index.shards = {"one.safetensors", "two.safetensors"};
        index.total_size = 2;
        const auto result = validate_index_shards(directory.path(), index);
        test.expect(!result &&
                        result.error.code == ErrorCode::kTensorInWrongShard &&
                        result.error.context == "a" &&
                        result.error.expected == "two.safetensors" &&
                        result.error.actual == "one.safetensors",
                    "tensor stored in the wrong shard is rejected");
    }

    {
        TempDirectory directory;
        test.expect(write_file(
                        directory.path() / "one.safetensors",
                        archive_bytes(
                            R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"extra":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
                            2)),
                    "extra tensor fixture is written");
        Index index;
        index.weight_map.emplace("a", "one.safetensors");
        index.shards.emplace_back("one.safetensors");
        index.total_size = 2;
        const auto result = validate_index_shards(directory.path(), index);
        test.expect(!result && result.error.code == ErrorCode::kUnexpectedTensor &&
                        result.error.context == "extra",
                    "tensor absent from the index is rejected");
    }

    {
        TempDirectory directory;
        test.expect(write_file(
                        directory.path() / "one.safetensors",
                        archive_bytes(
                            R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
                            1)),
                    "missing tensor fixture is written");
        Index index;
        index.weight_map.emplace("a", "one.safetensors");
        index.weight_map.emplace("b", "one.safetensors");
        index.shards.emplace_back("one.safetensors");
        index.total_size = 1;
        const auto result = validate_index_shards(directory.path(), index);
        test.expect(!result &&
                        result.error.code == ErrorCode::kMissingIndexedTensor &&
                        result.error.context == "b",
                    "index tensor absent from its shard is rejected");
    }
}

void test_shard_contract_total_and_limits(TestContext& test) {
    using namespace q3x::io::safetensors;
    TempDirectory directory;
    Index index;
    index.weight_map.emplace("a", "one.safetensors");
    index.shards.emplace_back("one.safetensors");
    test.expect(write_file(
                    directory.path() / "one.safetensors",
                    archive_bytes(
                        R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
                        1)),
                "payload size fixture is written");

    auto result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kMissingTotalSize,
                "strict contract requires metadata.total_size");

    index.total_size = 2;
    result = validate_index_shards(directory.path(), index);
    test.expect(!result && result.error.code == ErrorCode::kPayloadSizeMismatch &&
                    result.error.expected == "2" && result.error.actual == "1",
                "aggregate payload bytes must equal metadata.total_size");

    index.total_size = 1;
    ShardValidationOptions options;
    options.max_tensors = 0;
    result = validate_index_shards(directory.path(), index, options);
    test.expect(!result && result.error.code == ErrorCode::kInvalidOption,
                "zero shard validation limits are rejected");
}

}  // namespace

int main() {
    TestContext test;
    test_dtype_catalog(test);
    test_valid_header(test);
    test_scalars_empty_and_subbyte(test);
    test_header_rejections(test);
    test_file_and_option_failures(test);
    test_valid_index(test);
    test_index_rejections(test);
    test_shard_contract_success_and_policy(test);
    test_shard_contract_header_failures(test);
    test_shard_contract_membership_failures(test);
    test_shard_contract_total_and_limits(test);

    if (test.failures() != 0) {
        std::cerr << test.failures() << " safetensors test assertion(s) failed\n";
        return 1;
    }
    std::cout << "Safetensors tests passed\n";
    return 0;
}
