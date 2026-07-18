#include "q3x/text/tokenizer.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/utypes.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace q3x::text {
namespace {

using JsonObject = q3x::io::json::Value::Object;

constexpr std::size_t kAbsoluteTokenizerBytes = 100'000'000;
constexpr std::size_t kAbsoluteVocabSize = 1'000'000;
constexpr std::size_t kAbsoluteMerges = 2'000'000;
constexpr std::size_t kAbsoluteAddedTokens = 4'096;
constexpr std::size_t kAbsoluteInputBytes = 64'000'000;
constexpr std::size_t kAbsoluteTokens = 16'000'000;
constexpr std::size_t kAbsoluteChatMessages = 16'384;

constexpr std::string_view kPretokenizeRegex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?"
    "[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+"
    "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct AddedToken {
  std::uint32_t id = 0;
  std::string content;
  bool special = false;
};

struct MergeRule {
  std::uint32_t rank = 0;
  std::uint32_t result_id = 0;
};

struct TokenizerData {
  TokenizerLimits limits;
  std::unordered_map<std::string, std::uint32_t> vocabulary;
  std::vector<std::string> id_to_token;
  std::unordered_map<std::uint64_t, MergeRule> merges;
  std::vector<AddedToken> added_tokens;
  std::unordered_map<std::uint32_t, std::size_t> added_by_id;
  std::array<std::string, 256> byte_encoder;
  std::array<std::uint32_t, 256> byte_token_ids{};
  std::unordered_map<std::uint32_t, std::uint8_t> byte_decoder;
  std::unique_ptr<icu::RegexPattern> regex;
};

bool set_error(TokenizerError& error,
               const TokenizerErrorCode code,
               std::string message,
               const std::size_t offset = 0) {
  error.code = code;
  error.message = std::move(message);
  error.offset = offset;
  return false;
}

TokenizerError exception_error(const TokenizerErrorCode code,
                               const char* const message) noexcept {
  TokenizerError error;
  error.code = code;
  try {
    error.message = message;
  } catch (...) {
    // The code still carries a structured failure if even the diagnostic
    // string cannot be allocated.
  }
  return error;
}

bool validate_limits(const TokenizerLimits& limits, TokenizerError& error) {
  const bool nonzero = limits.max_tokenizer_bytes != 0 &&
                       limits.max_vocab_size != 0 && limits.max_merges != 0 &&
                       limits.max_added_tokens != 0 &&
                       limits.max_input_bytes != 0 && limits.max_tokens != 0 &&
                       limits.max_chat_messages != 0;
  const bool bounded =
      limits.max_tokenizer_bytes <= kAbsoluteTokenizerBytes &&
      limits.max_vocab_size <= kAbsoluteVocabSize &&
      limits.max_merges <= kAbsoluteMerges &&
      limits.max_added_tokens <= kAbsoluteAddedTokens &&
      limits.max_input_bytes <= kAbsoluteInputBytes &&
      limits.max_tokens <= kAbsoluteTokens &&
      limits.max_chat_messages <= kAbsoluteChatMessages;
  if (!nonzero || !bounded) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidArgument,
                     "tokenizer limits are zero or exceed absolute ceilings");
  }
  return true;
}

bool read_file(const std::string& path,
               const std::size_t max_bytes,
               std::string& output,
               TokenizerError& error) {
  if (path.empty()) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidArgument,
                     "tokenizer path is empty");
  }
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return set_error(error,
                     TokenizerErrorCode::kIoError,
                     "could not open tokenizer.json");
  }
  const std::streampos end = stream.tellg();
  if (end < 0) {
    return set_error(error,
                     TokenizerErrorCode::kIoError,
                     "could not determine tokenizer.json size");
  }
  const auto size = static_cast<std::uint64_t>(end);
  if (size > static_cast<std::uint64_t>(max_bytes) ||
      size > static_cast<std::uint64_t>(
                 std::numeric_limits<std::size_t>::max())) {
    return set_error(error,
                     TokenizerErrorCode::kTokenizerTooLarge,
                     "tokenizer.json exceeds max_tokenizer_bytes");
  }
  output.resize(static_cast<std::size_t>(size));
  stream.seekg(0, std::ios::beg);
  if (!output.empty()) {
    stream.read(output.data(), static_cast<std::streamsize>(output.size()));
  }
  if (!stream || stream.gcount() != static_cast<std::streamsize>(output.size())) {
    return set_error(error,
                     TokenizerErrorCode::kIoError,
                     "short read while loading tokenizer.json");
  }
  return true;
}

bool exact_keys(const q3x::io::json::Value& value,
                const std::initializer_list<std::string_view> keys,
                const std::string_view path,
                TokenizerError& error) {
  const JsonObject* const object = value.as_object();
  if (object == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     std::string(path) + " must be an object");
  }
  if (object->size() != keys.size()) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     std::string(path) + " has unexpected or missing fields");
  }
  for (const std::string_view key : keys) {
    if (object->find(key) == object->end()) {
      return set_error(error,
                       TokenizerErrorCode::kSchemaMismatch,
                       std::string(path) + " is missing field " +
                           std::string(key));
    }
  }
  return true;
}

const q3x::io::json::Value* member(const q3x::io::json::Value& object,
                                   const std::string_view key) noexcept {
  return object.find(key);
}

bool require_string(const q3x::io::json::Value& object,
                    const std::string_view key,
                    const std::string_view expected,
                    const std::string_view path,
                    TokenizerError& error) {
  const q3x::io::json::Value* const value = member(object, key);
  const std::string* const text = value == nullptr ? nullptr : value->as_string();
  if (text == nullptr || *text != expected) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     std::string(path) + "." + std::string(key) +
                         " does not match the pinned schema");
  }
  return true;
}

bool require_bool(const q3x::io::json::Value& object,
                  const std::string_view key,
                  const bool expected,
                  const std::string_view path,
                  TokenizerError& error) {
  const q3x::io::json::Value* const value = member(object, key);
  const bool* const boolean = value == nullptr ? nullptr : value->as_bool();
  if (boolean == nullptr || *boolean != expected) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     std::string(path) + "." + std::string(key) +
                         " does not match the pinned schema");
  }
  return true;
}

bool require_null(const q3x::io::json::Value& object,
                  const std::string_view key,
                  const std::string_view path,
                  TokenizerError& error) {
  const q3x::io::json::Value* const value = member(object, key);
  if (value == nullptr || !value->is_null()) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     std::string(path) + "." + std::string(key) +
                         " must be null");
  }
  return true;
}

bool validate_byte_level(const q3x::io::json::Value& value,
                         const std::string_view path,
                         TokenizerError& error) {
  return exact_keys(value,
                    {"type", "add_prefix_space", "trim_offsets", "use_regex"},
                    path,
                    error) &&
         require_string(value, "type", "ByteLevel", path, error) &&
         require_bool(value, "add_prefix_space", false, path, error) &&
         require_bool(value, "trim_offsets", false, path, error) &&
         require_bool(value, "use_regex", false, path, error);
}

bool validate_frontend_schema(const q3x::io::json::Value& root,
                              TokenizerError& error) {
  if (!exact_keys(root,
                  {"version",
                   "truncation",
                   "padding",
                   "added_tokens",
                   "normalizer",
                   "pre_tokenizer",
                   "post_processor",
                   "decoder",
                   "model"},
                  "$",
                  error) ||
      !require_string(root, "version", "1.0", "$", error) ||
      !require_null(root, "truncation", "$", error) ||
      !require_null(root, "padding", "$", error)) {
    return false;
  }

  const q3x::io::json::Value* const normalizer = member(root, "normalizer");
  if (normalizer == nullptr ||
      !exact_keys(*normalizer, {"type"}, "$.normalizer", error) ||
      !require_string(*normalizer, "type", "NFC", "$.normalizer", error)) {
    return false;
  }

  const q3x::io::json::Value* const pre = member(root, "pre_tokenizer");
  if (pre == nullptr ||
      !exact_keys(*pre, {"type", "pretokenizers"}, "$.pre_tokenizer", error) ||
      !require_string(*pre, "type", "Sequence", "$.pre_tokenizer", error)) {
    return false;
  }
  const q3x::io::json::Value* const sequence = member(*pre, "pretokenizers");
  const q3x::io::json::Value::Array* const pretokenizers =
      sequence == nullptr ? nullptr : sequence->as_array();
  if (pretokenizers == nullptr || pretokenizers->size() != 2) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "$.pre_tokenizer.pretokenizers must contain Split and "
                     "ByteLevel");
  }
  const q3x::io::json::Value& split = (*pretokenizers)[0];
  if (!exact_keys(split,
                  {"type", "pattern", "behavior", "invert"},
                  "$.pre_tokenizer.pretokenizers[0]",
                  error) ||
      !require_string(split,
                      "type",
                      "Split",
                      "$.pre_tokenizer.pretokenizers[0]",
                      error) ||
      !require_string(split,
                      "behavior",
                      "Isolated",
                      "$.pre_tokenizer.pretokenizers[0]",
                      error) ||
      !require_bool(split,
                    "invert",
                    false,
                    "$.pre_tokenizer.pretokenizers[0]",
                    error)) {
    return false;
  }
  const q3x::io::json::Value* const pattern = member(split, "pattern");
  if (pattern == nullptr ||
      !exact_keys(*pattern,
                  {"Regex"},
                  "$.pre_tokenizer.pretokenizers[0].pattern",
                  error) ||
      !require_string(*pattern,
                      "Regex",
                      kPretokenizeRegex,
                      "$.pre_tokenizer.pretokenizers[0].pattern",
                      error)) {
    return false;
  }
  if (!validate_byte_level((*pretokenizers)[1],
                           "$.pre_tokenizer.pretokenizers[1]",
                           error)) {
    return false;
  }

  const q3x::io::json::Value* const post = member(root, "post_processor");
  const q3x::io::json::Value* const decoder = member(root, "decoder");
  if (post == nullptr || decoder == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "tokenizer is missing ByteLevel post-processor or decoder");
  }
  return validate_byte_level(*post, "$.post_processor", error) &&
         validate_byte_level(*decoder, "$.decoder", error);
}

bool decode_utf8_at(const std::string_view input,
                    const std::size_t offset,
                    std::uint32_t& codepoint,
                    std::size_t& next) noexcept {
  if (offset >= input.size()) {
    return false;
  }
  const auto byte = static_cast<std::uint8_t>(input[offset]);
  if (byte <= 0x7FU) {
    codepoint = byte;
    next = offset + 1;
    return true;
  }
  auto continuation = [&](const std::size_t index, std::uint8_t& value) {
    if (index >= input.size()) {
      return false;
    }
    value = static_cast<std::uint8_t>(input[index]);
    return (value & 0xC0U) == 0x80U;
  };
  std::uint8_t b1 = 0;
  std::uint8_t b2 = 0;
  std::uint8_t b3 = 0;
  if (byte >= 0xC2U && byte <= 0xDFU && continuation(offset + 1, b1)) {
    codepoint = (static_cast<std::uint32_t>(byte & 0x1FU) << 6U) |
                static_cast<std::uint32_t>(b1 & 0x3FU);
    next = offset + 2;
    return true;
  }
  if (byte >= 0xE0U && byte <= 0xEFU && continuation(offset + 1, b1) &&
      continuation(offset + 2, b2)) {
    if ((byte == 0xE0U && b1 < 0xA0U) ||
        (byte == 0xEDU && b1 >= 0xA0U)) {
      return false;
    }
    codepoint = (static_cast<std::uint32_t>(byte & 0x0FU) << 12U) |
                (static_cast<std::uint32_t>(b1 & 0x3FU) << 6U) |
                static_cast<std::uint32_t>(b2 & 0x3FU);
    next = offset + 3;
    return true;
  }
  if (byte >= 0xF0U && byte <= 0xF4U && continuation(offset + 1, b1) &&
      continuation(offset + 2, b2) && continuation(offset + 3, b3)) {
    if ((byte == 0xF0U && b1 < 0x90U) ||
        (byte == 0xF4U && b1 > 0x8FU)) {
      return false;
    }
    codepoint = (static_cast<std::uint32_t>(byte & 0x07U) << 18U) |
                (static_cast<std::uint32_t>(b1 & 0x3FU) << 12U) |
                (static_cast<std::uint32_t>(b2 & 0x3FU) << 6U) |
                static_cast<std::uint32_t>(b3 & 0x3FU);
    next = offset + 4;
    return true;
  }
  return false;
}

void append_utf8(const std::uint32_t codepoint, std::string& output) {
  if (codepoint <= 0x7FU) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    output.push_back(
        static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
    output.push_back(
        static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
    output.push_back(
        static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
}

bool validate_utf8(const std::string_view input,
                   std::size_t& bad_offset) noexcept {
  std::size_t offset = 0;
  while (offset < input.size()) {
    std::uint32_t codepoint = 0;
    std::size_t next = 0;
    if (!decode_utf8_at(input, offset, codepoint, next)) {
      bad_offset = offset;
      return false;
    }
    offset = next;
  }
  return true;
}

bool initialize_byte_mapping(TokenizerData& data, TokenizerError& error) {
  std::array<bool, 256> direct{};
  for (std::uint32_t value = 33; value <= 126; ++value) {
    direct[value] = true;
  }
  for (std::uint32_t value = 161; value <= 172; ++value) {
    direct[value] = true;
  }
  for (std::uint32_t value = 174; value <= 255; ++value) {
    direct[value] = true;
  }
  std::uint32_t extension = 0;
  for (std::uint32_t value = 0; value <= 255; ++value) {
    const std::uint32_t codepoint =
        direct[value] ? value : 256U + extension++;
    append_utf8(codepoint, data.byte_encoder[value]);
    const auto inserted = data.byte_decoder.emplace(
        codepoint, static_cast<std::uint8_t>(value));
    if (!inserted.second) {
      return set_error(error,
                       TokenizerErrorCode::kInternalError,
                       "duplicate GPT-2 byte mapping");
    }
  }
  return true;
}

bool parse_model(const q3x::io::json::Value& root,
                 TokenizerData& data,
                 TokenizerError& error) {
  const q3x::io::json::Value* const model = member(root, "model");
  if (model == nullptr ||
      !exact_keys(*model,
                  {"type",
                   "dropout",
                   "unk_token",
                   "continuing_subword_prefix",
                   "end_of_word_suffix",
                   "fuse_unk",
                   "byte_fallback",
                   "ignore_merges",
                   "vocab",
                   "merges"},
                  "$.model",
                  error) ||
      !require_string(*model, "type", "BPE", "$.model", error) ||
      !require_null(*model, "dropout", "$.model", error) ||
      !require_null(*model, "unk_token", "$.model", error) ||
      !require_string(
          *model, "continuing_subword_prefix", "", "$.model", error) ||
      !require_string(*model, "end_of_word_suffix", "", "$.model", error) ||
      !require_bool(*model, "fuse_unk", false, "$.model", error) ||
      !require_bool(*model, "byte_fallback", false, "$.model", error) ||
      !require_bool(*model, "ignore_merges", false, "$.model", error)) {
    return false;
  }

  const q3x::io::json::Value* const vocab_value = member(*model, "vocab");
  const JsonObject* const vocab =
      vocab_value == nullptr ? nullptr : vocab_value->as_object();
  if (vocab == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "$.model.vocab must be an object");
  }
  if (vocab->size() > data.limits.max_vocab_size) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidVocabulary,
                     "vocabulary exceeds max_vocab_size");
  }
  if (vocab->size() != Tokenizer::kPinnedBaseVocabularySize) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "base vocabulary size does not match pinned tokenizer");
  }

  data.vocabulary.reserve(vocab->size());
  data.id_to_token.resize(vocab->size());
  std::vector<bool> seen(vocab->size(), false);
  for (const auto& entry : *vocab) {
    const q3x::io::json::Number* const number = entry.second.as_number();
    std::uint64_t id = 0;
    if (number == nullptr || !number->to_uint64(id) || id >= vocab->size()) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidVocabulary,
                       "vocabulary id is not a bounded unsigned integer");
    }
    const std::size_t index = static_cast<std::size_t>(id);
    if (seen[index]) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidVocabulary,
                       "vocabulary ids are not unique");
    }
    seen[index] = true;
    data.id_to_token[index] = entry.first;
    data.vocabulary.emplace(entry.first, static_cast<std::uint32_t>(id));
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidVocabulary,
                     "vocabulary ids are not contiguous");
  }

  for (std::size_t byte = 0; byte < data.byte_encoder.size(); ++byte) {
    const auto found = data.vocabulary.find(data.byte_encoder[byte]);
    if (found == data.vocabulary.end()) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidVocabulary,
                       "vocabulary is missing a GPT-2 byte token");
    }
    data.byte_token_ids[byte] = found->second;
  }
  for (const std::string& token : data.id_to_token) {
    std::size_t offset = 0;
    while (offset < token.size()) {
      std::uint32_t codepoint = 0;
      std::size_t next = 0;
      if (!decode_utf8_at(token, offset, codepoint, next) ||
          data.byte_decoder.find(codepoint) == data.byte_decoder.end()) {
        return set_error(error,
                         TokenizerErrorCode::kInvalidVocabulary,
                         "vocabulary token is not ByteLevel encoded");
      }
      offset = next;
    }
  }

  const q3x::io::json::Value* const merges_value = member(*model, "merges");
  const q3x::io::json::Value::Array* const merges =
      merges_value == nullptr ? nullptr : merges_value->as_array();
  if (merges == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "$.model.merges must be an array");
  }
  if (merges->size() > data.limits.max_merges) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidMerge,
                     "merges exceed max_merges");
  }
  if (merges->size() != Tokenizer::kPinnedMergeCount) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "merge count does not match pinned tokenizer");
  }
  data.merges.reserve(merges->size());
  for (std::size_t rank = 0; rank < merges->size(); ++rank) {
    const std::string* const text = (*merges)[rank].as_string();
    if (text == nullptr) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidMerge,
                       "merge entry must be a string");
    }
    const std::size_t separator = text->find(' ');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= text->size() ||
        text->find(' ', separator + 1) != std::string::npos) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidMerge,
                       "merge entry must contain exactly two tokens");
    }
    const std::string left = text->substr(0, separator);
    const std::string right = text->substr(separator + 1);
    const auto left_id = data.vocabulary.find(left);
    const auto right_id = data.vocabulary.find(right);
    const auto result_id = data.vocabulary.find(left + right);
    if (left_id == data.vocabulary.end() || right_id == data.vocabulary.end() ||
        result_id == data.vocabulary.end()) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidMerge,
                       "merge references a token absent from the vocabulary");
    }
    const std::uint64_t key =
        (static_cast<std::uint64_t>(left_id->second) << 32U) |
        static_cast<std::uint64_t>(right_id->second);
    const auto inserted = data.merges.emplace(
        key,
        MergeRule{static_cast<std::uint32_t>(rank), result_id->second});
    if (!inserted.second) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidMerge,
                       "duplicate merge pair");
    }
  }
  return true;
}

bool parse_added_tokens(const q3x::io::json::Value& root,
                        TokenizerData& data,
                        TokenizerError& error) {
  static constexpr std::array<std::string_view,
                              Tokenizer::kPinnedAddedTokenCount>
      expected_contents{{"<|endoftext|>",
                         "<|im_start|>",
                         "<|im_end|>",
                         "<|object_ref_start|>",
                         "<|object_ref_end|>",
                         "<|box_start|>",
                         "<|box_end|>",
                         "<|quad_start|>",
                         "<|quad_end|>",
                         "<|vision_start|>",
                         "<|vision_end|>",
                         "<|vision_pad|>",
                         "<|image_pad|>",
                         "<|video_pad|>",
                         "<tool_call>",
                         "</tool_call>",
                         "<|fim_prefix|>",
                         "<|fim_middle|>",
                         "<|fim_suffix|>",
                         "<|fim_pad|>",
                         "<|repo_name|>",
                         "<|file_sep|>",
                         "<tool_response>",
                         "</tool_response>",
                         "<think>",
                         "</think>"}};

  const q3x::io::json::Value* const value = member(root, "added_tokens");
  const q3x::io::json::Value::Array* const tokens =
      value == nullptr ? nullptr : value->as_array();
  if (tokens == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "$.added_tokens must be an array");
  }
  if (tokens->size() > data.limits.max_added_tokens) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "added tokens exceed max_added_tokens");
  }
  if (tokens->size() != expected_contents.size()) {
    return set_error(error,
                     TokenizerErrorCode::kSchemaMismatch,
                     "added token count does not match pinned tokenizer");
  }
  data.added_tokens.reserve(tokens->size());
  data.added_by_id.reserve(tokens->size());
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    const q3x::io::json::Value& token = (*tokens)[index];
    const std::string path = "$.added_tokens[" + std::to_string(index) + "]";
    if (!exact_keys(token,
                    {"id",
                     "content",
                     "single_word",
                     "lstrip",
                     "rstrip",
                     "normalized",
                     "special"},
                    path,
                    error) ||
        !require_string(token,
                        "content",
                        expected_contents[index],
                        path,
                        error) ||
        !require_bool(token, "single_word", false, path, error) ||
        !require_bool(token, "lstrip", false, path, error) ||
        !require_bool(token, "rstrip", false, path, error) ||
        !require_bool(token, "normalized", false, path, error)) {
      return false;
    }
    const q3x::io::json::Value* const id_value = member(token, "id");
    const q3x::io::json::Number* const number =
        id_value == nullptr ? nullptr : id_value->as_number();
    std::uint64_t id = 0;
    const std::uint64_t expected_id =
        static_cast<std::uint64_t>(Tokenizer::kPinnedBaseVocabularySize + index);
    if (number == nullptr || !number->to_uint64(id) || id != expected_id) {
      return set_error(error,
                       TokenizerErrorCode::kSchemaMismatch,
                       path + ".id does not match pinned tokenizer");
    }
    const q3x::io::json::Value* const special_value = member(token, "special");
    const bool* const special =
        special_value == nullptr ? nullptr : special_value->as_bool();
    const bool expected_special = index < 14;
    if (special == nullptr || *special != expected_special) {
      return set_error(error,
                       TokenizerErrorCode::kSchemaMismatch,
                       path + ".special does not match pinned tokenizer");
    }
    data.added_by_id.emplace(static_cast<std::uint32_t>(id), index);
    data.added_tokens.push_back(AddedToken{static_cast<std::uint32_t>(id),
                                           std::string(expected_contents[index]),
                                           *special});
  }
  return true;
}

bool compile_regex(TokenizerData& data, TokenizerError& error) {
  UErrorCode status = U_ZERO_ERROR;
  const icu::UnicodeString pattern = icu::UnicodeString::fromUTF8(
      icu::StringPiece(kPretokenizeRegex.data(),
                       static_cast<std::int32_t>(kPretokenizeRegex.size())));
  data.regex.reset(icu::RegexPattern::compile(pattern, 0, status));
  if (U_FAILURE(status) || data.regex == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kRegexError,
                     std::string("could not compile pinned tokenizer regex: ") +
                         u_errorName(status));
  }
  return true;
}

struct BpeNode {
  std::uint32_t token_id = 0;
  std::size_t previous = std::numeric_limits<std::size_t>::max();
  std::size_t next = std::numeric_limits<std::size_t>::max();
  std::uint32_t generation = 0;
  bool alive = true;
};

struct BpeCandidate {
  std::uint32_t rank = 0;
  std::size_t left = 0;
  std::size_t right = 0;
  std::uint32_t left_generation = 0;
  std::uint32_t right_generation = 0;
};

struct CandidateLater {
  bool operator()(const BpeCandidate& left,
                  const BpeCandidate& right) const noexcept {
    if (left.rank != right.rank) {
      return left.rank > right.rank;
    }
    return left.left > right.left;
  }
};

std::uint64_t pair_key(const std::uint32_t left,
                       const std::uint32_t right) noexcept {
  return (static_cast<std::uint64_t>(left) << 32U) |
         static_cast<std::uint64_t>(right);
}

bool push_token(std::vector<std::uint32_t>& output,
                const std::uint32_t id,
                const std::size_t limit,
                TokenizerError& error) {
  if (output.size() >= limit) {
    return set_error(error,
                     TokenizerErrorCode::kTooManyTokens,
                     "encoded output exceeds max_tokens");
  }
  output.push_back(id);
  return true;
}

bool encode_bpe_piece(const TokenizerData& data,
                      const std::string_view piece,
                      std::vector<std::uint32_t>& output,
                      TokenizerError& error) {
  if (piece.empty()) {
    return true;
  }
  std::vector<BpeNode> nodes(piece.size());
  for (std::size_t index = 0; index < piece.size(); ++index) {
    nodes[index].token_id =
        data.byte_token_ids[static_cast<std::uint8_t>(piece[index])];
    nodes[index].previous =
        index == 0 ? std::numeric_limits<std::size_t>::max() : index - 1;
    nodes[index].next = index + 1 < piece.size()
                            ? index + 1
                            : std::numeric_limits<std::size_t>::max();
  }

  std::priority_queue<BpeCandidate,
                      std::vector<BpeCandidate>,
                      CandidateLater>
      candidates;
  auto add_candidate = [&](const std::size_t left) {
    if (left == std::numeric_limits<std::size_t>::max() ||
        !nodes[left].alive ||
        nodes[left].next == std::numeric_limits<std::size_t>::max()) {
      return;
    }
    const std::size_t right = nodes[left].next;
    const auto rule = data.merges.find(
        pair_key(nodes[left].token_id, nodes[right].token_id));
    if (rule != data.merges.end()) {
      candidates.push(BpeCandidate{rule->second.rank,
                                   left,
                                   right,
                                   nodes[left].generation,
                                   nodes[right].generation});
    }
  };
  for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
    add_candidate(index);
  }

  while (!candidates.empty()) {
    const BpeCandidate candidate = candidates.top();
    candidates.pop();
    BpeNode& left = nodes[candidate.left];
    BpeNode& right = nodes[candidate.right];
    if (!left.alive || !right.alive || left.next != candidate.right ||
        left.generation != candidate.left_generation ||
        right.generation != candidate.right_generation) {
      continue;
    }
    const auto rule =
        data.merges.find(pair_key(left.token_id, right.token_id));
    if (rule == data.merges.end() || rule->second.rank != candidate.rank) {
      continue;
    }
    left.token_id = rule->second.result_id;
    ++left.generation;
    right.alive = false;
    ++right.generation;
    left.next = right.next;
    if (right.next != std::numeric_limits<std::size_t>::max()) {
      nodes[right.next].previous = candidate.left;
    }
    add_candidate(left.previous);
    add_candidate(candidate.left);
  }

  std::size_t current = 0;
  while (current != std::numeric_limits<std::size_t>::max()) {
    if (!push_token(output, nodes[current].token_id, data.limits.max_tokens, error)) {
      return false;
    }
    current = nodes[current].next;
  }
  return true;
}

bool unicode_piece_to_utf8(const icu::UnicodeString& value,
                           std::string& output,
                           TokenizerError& error) {
  output.clear();
  value.toUTF8String(output);
  if (output.size() > kAbsoluteInputBytes) {
    return set_error(error,
                     TokenizerErrorCode::kInputTooLarge,
                     "normalized tokenizer segment exceeds absolute limit");
  }
  return true;
}

bool encode_normal_segment(const TokenizerData& data,
                           const std::string_view segment,
                           std::vector<std::uint32_t>& output,
                           TokenizerError& error) {
  if (segment.empty()) {
    return true;
  }
  std::size_t bad_offset = 0;
  if (!validate_utf8(segment, bad_offset)) {
    return set_error(error,
                     TokenizerErrorCode::kInvalidUtf8,
                     "input is not valid UTF-8",
                     bad_offset);
  }
  if (segment.size() > static_cast<std::size_t>(
                           std::numeric_limits<std::int32_t>::max())) {
    return set_error(error,
                     TokenizerErrorCode::kInputTooLarge,
                     "input exceeds ICU string length");
  }

  UErrorCode status = U_ZERO_ERROR;
  const icu::Normalizer2* const normalizer =
      icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status) || normalizer == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kUnicodeError,
                     std::string("could not acquire ICU NFC normalizer: ") +
                         u_errorName(status));
  }
  const icu::UnicodeString unicode = icu::UnicodeString::fromUTF8(
      icu::StringPiece(segment.data(), static_cast<std::int32_t>(segment.size())));
  icu::UnicodeString normalized;
  normalizer->normalize(unicode, normalized, status);
  if (U_FAILURE(status)) {
    return set_error(error,
                     TokenizerErrorCode::kUnicodeError,
                     std::string("ICU NFC normalization failed: ") +
                         u_errorName(status));
  }

  std::unique_ptr<icu::RegexMatcher> matcher(data.regex->matcher(normalized, status));
  if (U_FAILURE(status) || matcher == nullptr) {
    return set_error(error,
                     TokenizerErrorCode::kRegexError,
                     std::string("could not create tokenizer regex matcher: ") +
                         u_errorName(status));
  }
  std::int32_t previous_end = 0;
  std::string piece;
  while (matcher->find(status)) {
    const std::int32_t start = matcher->start(status);
    const std::int32_t end = matcher->end(status);
    if (U_FAILURE(status) || start < previous_end || end <= start) {
      return set_error(error,
                       TokenizerErrorCode::kRegexError,
                       "tokenizer regex returned invalid match offsets");
    }
    if (start > previous_end) {
      if (!unicode_piece_to_utf8(
              normalized.tempSubStringBetween(previous_end, start), piece, error) ||
          piece.size() > data.limits.max_input_bytes ||
          !encode_bpe_piece(data, piece, output, error)) {
        if (error.ok()) {
          set_error(error,
                    TokenizerErrorCode::kInputTooLarge,
                    "normalized input exceeds max_input_bytes");
        }
        return false;
      }
    }
    if (!unicode_piece_to_utf8(
            normalized.tempSubStringBetween(start, end), piece, error) ||
        piece.size() > data.limits.max_input_bytes ||
        !encode_bpe_piece(data, piece, output, error)) {
      if (error.ok()) {
        set_error(error,
                  TokenizerErrorCode::kInputTooLarge,
                  "normalized input exceeds max_input_bytes");
      }
      return false;
    }
    previous_end = end;
  }
  if (U_FAILURE(status)) {
    return set_error(error,
                     TokenizerErrorCode::kRegexError,
                     std::string("tokenizer regex matching failed: ") +
                         u_errorName(status));
  }
  if (previous_end < normalized.length()) {
    if (!unicode_piece_to_utf8(
            normalized.tempSubString(previous_end), piece, error) ||
        piece.size() > data.limits.max_input_bytes ||
        !encode_bpe_piece(data, piece, output, error)) {
      if (error.ok()) {
        set_error(error,
                  TokenizerErrorCode::kInputTooLarge,
                  "normalized input exceeds max_input_bytes");
      }
      return false;
    }
  }
  return true;
}

EncodeResult encode_impl(const TokenizerData& data,
                         const std::string_view text) {
  EncodeResult result;
  if (text.size() > data.limits.max_input_bytes) {
    set_error(result.error,
              TokenizerErrorCode::kInputTooLarge,
              "input exceeds max_input_bytes");
    return result;
  }
  result.token_ids.reserve(std::min(text.size(), data.limits.max_tokens));
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t match_position = std::string_view::npos;
    const AddedToken* matched = nullptr;
    for (const AddedToken& token : data.added_tokens) {
      const std::size_t position = text.find(token.content, cursor);
      if (position < match_position ||
          (position == match_position && matched != nullptr &&
           token.content.size() > matched->content.size())) {
        match_position = position;
        matched = &token;
      }
    }
    if (matched == nullptr) {
      if (!encode_normal_segment(
              data, text.substr(cursor), result.token_ids, result.error)) {
        return result;
      }
      cursor = text.size();
      continue;
    }
    if (match_position > cursor &&
        !encode_normal_segment(data,
                               text.substr(cursor, match_position - cursor),
                               result.token_ids,
                               result.error)) {
      return result;
    }
    if (!push_token(result.token_ids,
                    matched->id,
                    data.limits.max_tokens,
                    result.error)) {
      return result;
    }
    cursor = match_position + matched->content.size();
  }
  return result;
}

bool append_bounded(std::string& output,
                    const std::string_view text,
                    const std::size_t limit,
                    TokenizerError& error) {
  if (text.size() > limit || output.size() > limit - text.size()) {
    return set_error(error,
                     TokenizerErrorCode::kInputTooLarge,
                     "rendered chat exceeds max_input_bytes");
  }
  output.append(text.data(), text.size());
  return true;
}

bool trim_unicode(const std::string& input,
                  std::string_view& trimmed,
                  TokenizerError& error) {
  std::size_t offset = 0;
  std::size_t start = 0;
  std::size_t end = 0;
  bool saw_nonspace = false;
  while (offset < input.size()) {
    const std::size_t current = offset;
    std::uint32_t codepoint = 0;
    std::size_t next = 0;
    if (!decode_utf8_at(input, offset, codepoint, next)) {
      return set_error(error,
                       TokenizerErrorCode::kInvalidUtf8,
                       "chat content is not valid UTF-8",
                       offset);
    }
    const bool whitespace =
        u_isUWhiteSpace(static_cast<UChar32>(codepoint)) != 0 ||
        codepoint <= 0x20U;
    if (!whitespace) {
      if (!saw_nonspace) {
        start = current;
        saw_nonspace = true;
      }
      end = next;
    }
    offset = next;
  }
  trimmed = saw_nonspace ? std::string_view(input).substr(start, end - start)
                         : std::string_view{};
  return true;
}

}  // namespace

struct Tokenizer::Impl {
  TokenizerData data;
};

TokenizerLoadResult Tokenizer::load_file(
    const std::string& tokenizer_json_path,
    const TokenizerLimits& limits) noexcept {
  try {
    TokenizerError error;
    if (!validate_limits(limits, error)) {
      return {nullptr, std::move(error)};
    }
    std::string contents;
    if (!read_file(
            tokenizer_json_path, limits.max_tokenizer_bytes, contents, error)) {
      return {nullptr, std::move(error)};
    }
    return load_json(contents, limits);
  } catch (const std::bad_alloc&) {
    return {nullptr,
            exception_error(TokenizerErrorCode::kAllocationFailure,
                            "allocation failure while loading tokenizer")};
  } catch (...) {
    return {nullptr,
            exception_error(TokenizerErrorCode::kInternalError,
                            "unexpected failure while loading tokenizer")};
  }
}

TokenizerLoadResult Tokenizer::load_json(
    const std::string_view tokenizer_json,
    const TokenizerLimits& limits) noexcept {
  try {
    TokenizerError error;
    if (!validate_limits(limits, error)) {
      return {nullptr, std::move(error)};
    }
    if (tokenizer_json.size() > limits.max_tokenizer_bytes) {
      set_error(error,
                TokenizerErrorCode::kTokenizerTooLarge,
                "tokenizer JSON exceeds max_tokenizer_bytes");
      return {nullptr, std::move(error)};
    }

    q3x::io::json::ParseOptions parse_options;
    parse_options.max_input_bytes = limits.max_tokenizer_bytes;
    parse_options.max_nesting_depth = 32;
    const std::size_t fixed_overhead = 4'096;
    const std::size_t dynamic_budget = limits.max_vocab_size +
                                       limits.max_merges +
                                       limits.max_added_tokens * 16U;
    parse_options.max_values = dynamic_budget + fixed_overhead;
    parse_options.max_container_items = dynamic_budget + fixed_overhead;
    q3x::io::json::ParseResult parsed =
        q3x::io::json::parse(tokenizer_json, parse_options);
    if (!parsed) {
      set_error(error,
                TokenizerErrorCode::kJsonError,
                std::string("invalid tokenizer JSON: ") +
                    std::string(parsed.error.message()),
                parsed.error.offset);
      return {nullptr, std::move(error)};
    }
    if (!validate_frontend_schema(*parsed.value, error)) {
      return {nullptr, std::move(error)};
    }

    auto implementation = std::make_unique<Impl>();
    implementation->data.limits = limits;
    if (!initialize_byte_mapping(implementation->data, error) ||
        !parse_model(*parsed.value, implementation->data, error) ||
        !parse_added_tokens(*parsed.value, implementation->data, error)) {
      return {nullptr, std::move(error)};
    }

    const std::string digest = q3x::core::sha256(tokenizer_json).hex();
    if (digest != kPinnedTokenizerSha256) {
      set_error(error,
                TokenizerErrorCode::kUnsupportedRevision,
                "tokenizer JSON schema is compatible but its SHA-256 is not "
                "the pinned revision");
      return {nullptr, std::move(error)};
    }
    if (!compile_regex(implementation->data, error)) {
      return {nullptr, std::move(error)};
    }
    return {std::unique_ptr<Tokenizer>(
                new Tokenizer(std::move(implementation))),
            {}};
  } catch (const std::bad_alloc&) {
    return {nullptr,
            exception_error(TokenizerErrorCode::kAllocationFailure,
                            "allocation failure while parsing tokenizer")};
  } catch (...) {
    return {nullptr,
            exception_error(TokenizerErrorCode::kInternalError,
                            "unexpected failure while parsing tokenizer")};
  }
}

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

EncodeResult Tokenizer::encode(const std::string_view text,
                               const EncodeOptions& options) const noexcept {
  (void)options;
  try {
    if (impl_ == nullptr) {
      EncodeResult result;
      result.error = exception_error(TokenizerErrorCode::kInternalError,
                                     "tokenizer is in a moved-from state");
      return result;
    }
    return encode_impl(impl_->data, text);
  } catch (const std::bad_alloc&) {
    EncodeResult result;
    result.error = exception_error(TokenizerErrorCode::kAllocationFailure,
                                   "allocation failure while encoding");
    return result;
  } catch (...) {
    EncodeResult result;
    result.error = exception_error(TokenizerErrorCode::kInternalError,
                                   "unexpected failure while encoding");
    return result;
  }
}

DecodeResult Tokenizer::decode(
    const std::vector<std::uint32_t>& token_ids,
    const DecodeOptions& options) const noexcept {
  try {
    DecodeResult result;
    if (impl_ == nullptr) {
      result.error = exception_error(TokenizerErrorCode::kInternalError,
                                     "tokenizer is in a moved-from state");
      return result;
    }
    const TokenizerData& data = impl_->data;
    if (token_ids.size() > data.limits.max_tokens) {
      set_error(result.error,
                TokenizerErrorCode::kTooManyTokens,
                "decode input exceeds max_tokens");
      return result;
    }
    std::string byte_symbols;
    auto append_output = [&](const std::string_view value) {
      if (value.size() > data.limits.max_input_bytes ||
          result.text.size() > data.limits.max_input_bytes - value.size()) {
        return set_error(result.error,
                         TokenizerErrorCode::kInputTooLarge,
                         "decoded output exceeds max_input_bytes");
      }
      result.text.append(value.data(), value.size());
      return true;
    };
    auto flush_symbols = [&]() {
      std::string bytes;
      bytes.reserve(byte_symbols.size());
      std::size_t offset = 0;
      while (offset < byte_symbols.size()) {
        std::uint32_t codepoint = 0;
        std::size_t next = 0;
        if (!decode_utf8_at(byte_symbols, offset, codepoint, next)) {
          return set_error(result.error,
                           TokenizerErrorCode::kInternalError,
                           "vocabulary contained invalid UTF-8 during decode");
        }
        const auto decoded = data.byte_decoder.find(codepoint);
        if (decoded == data.byte_decoder.end()) {
          return set_error(result.error,
                           TokenizerErrorCode::kInternalError,
                           "vocabulary contained a non-ByteLevel code point");
        }
        bytes.push_back(static_cast<char>(decoded->second));
        offset = next;
      }
      byte_symbols.clear();
      return append_output(bytes);
    };

    for (const std::uint32_t id : token_ids) {
      if (id < data.id_to_token.size()) {
        const std::string& token = data.id_to_token[id];
        if (token.size() > data.limits.max_input_bytes ||
            byte_symbols.size() > data.limits.max_input_bytes - token.size()) {
          set_error(result.error,
                    TokenizerErrorCode::kInputTooLarge,
                    "decoded output exceeds max_input_bytes");
          return result;
        }
        byte_symbols += token;
        continue;
      }
      const auto added = data.added_by_id.find(id);
      if (added == data.added_by_id.end()) {
        set_error(result.error,
                  TokenizerErrorCode::kInvalidTokenId,
                  "token id is outside the pinned tokenizer vocabulary");
        return result;
      }
      if (!flush_symbols()) {
        return result;
      }
      const AddedToken& token = data.added_tokens[added->second];
      if ((!options.skip_special_tokens || !token.special) &&
          !append_output(token.content)) {
        return result;
      }
    }
    if (!flush_symbols()) {
      return result;
    }
    return result;
  } catch (const std::bad_alloc&) {
    DecodeResult result;
    result.error = exception_error(TokenizerErrorCode::kAllocationFailure,
                                   "allocation failure while decoding");
    return result;
  } catch (...) {
    DecodeResult result;
    result.error = exception_error(TokenizerErrorCode::kInternalError,
                                   "unexpected failure while decoding");
    return result;
  }
}

ChatResult Tokenizer::format_qwen36_chat(
    const std::vector<ChatMessage>& messages,
    const Qwen36ChatOptions& options) const noexcept {
  try {
    ChatResult result;
    if (impl_ == nullptr) {
      result.error = exception_error(TokenizerErrorCode::kInternalError,
                                     "tokenizer is in a moved-from state");
      return result;
    }
    const TokenizerData& data = impl_->data;
    if (messages.empty()) {
      set_error(result.error,
                TokenizerErrorCode::kUnsupportedChat,
                "Qwen chat requires at least one user message");
      return result;
    }
    if (messages.size() > data.limits.max_chat_messages) {
      set_error(result.error,
                TokenizerErrorCode::kUnsupportedChat,
                "chat exceeds max_chat_messages");
      return result;
    }

    std::size_t index = 0;
    std::size_t last_user_index = std::numeric_limits<std::size_t>::max();
    for (std::size_t message_index = 0; message_index < messages.size();
         ++message_index) {
      if (messages[message_index].role == "user") {
        last_user_index = message_index;
      }
    }
    bool expect_user = true;
    bool saw_user = false;
    for (const ChatMessage& message : messages) {
      std::string_view content;
      if (!trim_unicode(message.content, content, result.error)) {
        return result;
      }
      if (message.role == "system") {
        if (index != 0) {
          set_error(result.error,
                    TokenizerErrorCode::kUnsupportedChat,
                    "system message is only supported in first position");
          return result;
        }
        if (!append_bounded(result.rendered,
                            "<|im_start|>system\n",
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            content,
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            "<|im_end|>\n",
                            data.limits.max_input_bytes,
                            result.error)) {
          return result;
        }
      } else if (message.role == "user") {
        if (!expect_user) {
          set_error(result.error,
                    TokenizerErrorCode::kUnsupportedChat,
                    "text-only chat messages must alternate user/assistant");
          return result;
        }
        saw_user = true;
        expect_user = false;
        if (!append_bounded(result.rendered,
                            "<|im_start|>user\n",
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            content,
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            "<|im_end|>\n",
                            data.limits.max_input_bytes,
                            result.error)) {
          return result;
        }
      } else if (message.role == "assistant") {
        if (expect_user || content.find("<think>") != std::string_view::npos ||
            content.find("</think>") != std::string_view::npos) {
          set_error(result.error,
                    TokenizerErrorCode::kUnsupportedChat,
                    "assistant messages must follow user and cannot contain "
                    "thinking-history markup in the fixed subset");
          return result;
        }
        expect_user = true;
        const std::string_view assistant_prefix =
            index > last_user_index
                ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                : "<|im_start|>assistant\n";
        if (!append_bounded(result.rendered,
                            assistant_prefix,
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            content,
                            data.limits.max_input_bytes,
                            result.error) ||
            !append_bounded(result.rendered,
                            "<|im_end|>\n",
                            data.limits.max_input_bytes,
                            result.error)) {
          return result;
        }
      } else {
        set_error(result.error,
                  TokenizerErrorCode::kUnsupportedChat,
                  "only system, user, and assistant text roles are supported");
        return result;
      }
      ++index;
    }
    if (!saw_user) {
      set_error(result.error,
                TokenizerErrorCode::kUnsupportedChat,
                "Qwen chat requires a user message");
      return result;
    }
    if (options.add_generation_prompt) {
      if (expect_user) {
        set_error(result.error,
                  TokenizerErrorCode::kUnsupportedChat,
                  "generation prompt requires the final message to be user");
        return result;
      }
      const std::string_view suffix =
          options.enable_thinking
              ? "<|im_start|>assistant\n<think>\n"
              : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
      if (!append_bounded(result.rendered,
                          suffix,
                          data.limits.max_input_bytes,
                          result.error)) {
        return result;
      }
    }
    EncodeResult encoded = encode_impl(data, result.rendered);
    if (!encoded) {
      result.error = std::move(encoded.error);
      return result;
    }
    result.token_ids = std::move(encoded.token_ids);
    return result;
  } catch (const std::bad_alloc&) {
    ChatResult result;
    result.error = exception_error(TokenizerErrorCode::kAllocationFailure,
                                   "allocation failure while formatting chat");
    return result;
  } catch (...) {
    ChatResult result;
    result.error = exception_error(TokenizerErrorCode::kInternalError,
                                   "unexpected failure while formatting chat");
    return result;
  }
}

std::size_t Tokenizer::base_vocabulary_size() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.id_to_token.size();
}

std::size_t Tokenizer::merge_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.merges.size();
}

std::size_t Tokenizer::added_token_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.added_tokens.size();
}

const TokenizerLimits& Tokenizer::limits() const noexcept {
  static const TokenizerLimits empty{};
  return impl_ == nullptr ? empty : impl_->data.limits;
}

std::string_view to_string(const TokenizerErrorCode code) noexcept {
  switch (code) {
    case TokenizerErrorCode::kNone:
      return "none";
    case TokenizerErrorCode::kInvalidArgument:
      return "invalid-argument";
    case TokenizerErrorCode::kIoError:
      return "io-error";
    case TokenizerErrorCode::kTokenizerTooLarge:
      return "tokenizer-too-large";
    case TokenizerErrorCode::kJsonError:
      return "json-error";
    case TokenizerErrorCode::kSchemaMismatch:
      return "schema-mismatch";
    case TokenizerErrorCode::kUnsupportedRevision:
      return "unsupported-revision";
    case TokenizerErrorCode::kInvalidUtf8:
      return "invalid-utf8";
    case TokenizerErrorCode::kUnicodeError:
      return "unicode-error";
    case TokenizerErrorCode::kRegexError:
      return "regex-error";
    case TokenizerErrorCode::kInvalidVocabulary:
      return "invalid-vocabulary";
    case TokenizerErrorCode::kInvalidMerge:
      return "invalid-merge";
    case TokenizerErrorCode::kInputTooLarge:
      return "input-too-large";
    case TokenizerErrorCode::kTooManyTokens:
      return "too-many-tokens";
    case TokenizerErrorCode::kInvalidTokenId:
      return "invalid-token-id";
    case TokenizerErrorCode::kUnsupportedChat:
      return "unsupported-chat";
    case TokenizerErrorCode::kAllocationFailure:
      return "allocation-failure";
    case TokenizerErrorCode::kInternalError:
      return "internal-error";
  }
  return "unknown";
}

}  // namespace q3x::text
