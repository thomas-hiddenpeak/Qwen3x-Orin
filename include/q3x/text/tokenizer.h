#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::text {

// Resource limits are enforced both while loading untrusted tokenizer JSON and
// while processing caller-controlled text. The implementation also applies
// fixed absolute ceilings, so an accidentally unbounded Limits value cannot
// disable its allocation guards.
struct TokenizerLimits {
  std::size_t max_tokenizer_bytes = 32'000'000;
  std::size_t max_vocab_size = 300'000;
  std::size_t max_merges = 300'000;
  std::size_t max_added_tokens = 64;
  std::size_t max_input_bytes = 4'000'000;
  std::size_t max_tokens = 4'000'000;
  std::size_t max_chat_messages = 1'024;
};

enum class TokenizerErrorCode : std::uint8_t {
  kNone,
  kInvalidArgument,
  kIoError,
  kTokenizerTooLarge,
  kJsonError,
  kSchemaMismatch,
  kUnsupportedRevision,
  kInvalidUtf8,
  kUnicodeError,
  kRegexError,
  kInvalidVocabulary,
  kInvalidMerge,
  kInputTooLarge,
  kTooManyTokens,
  kInvalidTokenId,
  kUnsupportedChat,
  kAllocationFailure,
  kInternalError,
};

struct TokenizerError {
  TokenizerErrorCode code = TokenizerErrorCode::kNone;
  std::string message;
  std::size_t offset = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == TokenizerErrorCode::kNone;
  }
};

struct EncodeOptions {
  // The pinned tokenizer has a ByteLevel post-processor and does not inject a
  // BOS/EOS token, so this flag is accepted for an explicit API but has no
  // effect on the resulting ids.
  bool add_special_tokens = false;
};

struct EncodeResult {
  std::vector<std::uint32_t> token_ids;
  TokenizerError error;

  [[nodiscard]] bool ok() const noexcept { return error.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct DecodeOptions {
  // Only tokenizer.json entries whose `special` field is true are skipped.
  // Added tokens such as <think>, which the pinned file marks non-special,
  // remain in the decoded text.
  bool skip_special_tokens = false;
};

struct DecodeResult {
  std::string text;
  TokenizerError error;

  [[nodiscard]] bool ok() const noexcept { return error.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ChatMessage {
  std::string role;
  std::string content;
};

struct Qwen36ChatOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
};

struct ChatResult {
  std::string rendered;
  std::vector<std::uint32_t> token_ids;
  TokenizerError error;

  [[nodiscard]] bool ok() const noexcept { return error.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

class Tokenizer;

struct TokenizerLoadResult {
  std::unique_ptr<Tokenizer> tokenizer;
  TokenizerError error;

  [[nodiscard]] bool ok() const noexcept {
    return tokenizer != nullptr && error.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure-C++17/ICU implementation of the exact tokenizer.json shipped by the
// pinned nvidia/Qwen3.6-27B-NVFP4 revision. Loading rejects schema or revision
// drift instead of guessing at a related tokenizer configuration.
class Tokenizer final {
 public:
  static constexpr std::string_view kPinnedTokenizerSha256 =
      "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";
  static constexpr std::size_t kPinnedBaseVocabularySize = 248'044;
  static constexpr std::size_t kPinnedMergeCount = 247'587;
  static constexpr std::size_t kPinnedAddedTokenCount = 26;

  [[nodiscard]] static TokenizerLoadResult load_file(
      const std::string& tokenizer_json_path,
      const TokenizerLimits& limits = {}) noexcept;

  [[nodiscard]] static TokenizerLoadResult load_json(
      std::string_view tokenizer_json,
      const TokenizerLimits& limits = {}) noexcept;

  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  [[nodiscard]] EncodeResult encode(
      std::string_view text,
      const EncodeOptions& options = {}) const noexcept;

  [[nodiscard]] DecodeResult decode(
      const std::vector<std::uint32_t>& token_ids,
      const DecodeOptions& options = {}) const noexcept;

  // This is a deliberately explicit, text-only subset of the pinned Qwen 3.6
  // Jinja template, not a general Jinja interpreter. It supports an optional
  // leading system message followed by alternating user/assistant messages.
  // Tools, multimodal content, tool messages, and embedded thinking histories
  // fail closed with kUnsupportedChat.
  [[nodiscard]] ChatResult format_qwen36_chat(
      const std::vector<ChatMessage>& messages,
      const Qwen36ChatOptions& options = {}) const noexcept;

  [[nodiscard]] std::size_t base_vocabulary_size() const noexcept;
  [[nodiscard]] std::size_t merge_count() const noexcept;
  [[nodiscard]] std::size_t added_token_count() const noexcept;
  [[nodiscard]] const TokenizerLimits& limits() const noexcept;

 private:
  struct Impl;
  explicit Tokenizer(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string_view to_string(TokenizerErrorCode code) noexcept;

}  // namespace q3x::text
