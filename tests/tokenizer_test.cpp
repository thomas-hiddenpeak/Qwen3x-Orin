#include "q3x/io/json.h"
#include "q3x/text/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using q3x::io::json::Value;
using q3x::text::ChatMessage;
using q3x::text::Tokenizer;
using q3x::text::TokenizerErrorCode;

int failures = 0;

void check(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

bool read_file(const std::string& path, std::string& output) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return false;
  }
  output.assign(std::istreambuf_iterator<char>(stream),
                std::istreambuf_iterator<char>());
  return stream.good() || stream.eof();
}

const Value* required(const Value& value, const std::string_view key) {
  const Value* const found = value.find(key);
  check(found != nullptr, std::string("fixture is missing ") + std::string(key));
  return found;
}

bool uint_value(const Value& value, std::uint64_t& output) {
  const auto* const number = value.as_number();
  return number != nullptr && number->to_uint64(output);
}

bool token_array(const Value& value, std::vector<std::uint32_t>& output) {
  const Value::Array* const values = value.as_array();
  if (values == nullptr) {
    return false;
  }
  output.clear();
  output.reserve(values->size());
  for (const Value& entry : *values) {
    std::uint64_t id = 0;
    if (!uint_value(entry, id) || id > UINT32_MAX) {
      return false;
    }
    output.push_back(static_cast<std::uint32_t>(id));
  }
  return true;
}

void check_encode(Tokenizer& tokenizer,
                  const std::string_view text,
                  const std::vector<std::uint32_t>& expected,
                  const std::string_view label) {
  const auto encoded = tokenizer.encode(text);
  check(encoded.ok(), std::string(label) + " encode succeeds");
  if (encoded.ok()) {
    check(encoded.token_ids == expected,
          std::string(label) + " ids match independent oracle");
    if (encoded.token_ids != expected) {
      std::cerr << "  expected=";
      for (const auto id : expected) {
        std::cerr << id << ',';
      }
      std::cerr << " actual=";
      for (const auto id : encoded.token_ids) {
        std::cerr << id << ',';
      }
      std::cerr << '\n';
    }
  } else {
    std::cerr << "  error=" << q3x::text::to_string(encoded.error.code)
              << " message=" << encoded.error.message << '\n';
  }
}

void fixture_cases(Tokenizer& tokenizer, const Value& fixture) {
  const Value* const source = required(fixture, "source");
  const Value* const hash = source == nullptr
                                ? nullptr
                                : required(*source, "tokenizer_json_sha256");
  check(hash != nullptr && hash->as_string() != nullptr &&
            *hash->as_string() == Tokenizer::kPinnedTokenizerSha256,
        "fixture pins the implementation tokenizer SHA-256");

  const Value* const vocabulary = required(fixture, "vocabulary");
  std::uint64_t base_size = 0;
  std::uint64_t configured_size = 0;
  check(vocabulary != nullptr &&
            uint_value(*required(*vocabulary, "base_size"), base_size) &&
            base_size == Tokenizer::kPinnedBaseVocabularySize,
        "fixture base vocabulary size is pinned");
  check(vocabulary != nullptr &&
            uint_value(*required(*vocabulary, "size_with_added_tokens"),
                       configured_size) &&
            configured_size == 248'077,
        "fixture records tokenizer_config's seven extra audio/TTS tokens");
  check(tokenizer.base_vocabulary_size() ==
            Tokenizer::kPinnedBaseVocabularySize,
        "loaded base vocabulary size");
  check(tokenizer.merge_count() == Tokenizer::kPinnedMergeCount,
        "loaded merge count");
  check(tokenizer.added_token_count() == Tokenizer::kPinnedAddedTokenCount,
        "loaded tokenizer.json added-token count");

  const Value* const cases_value = required(fixture, "encode_cases");
  const Value::Array* const cases =
      cases_value == nullptr ? nullptr : cases_value->as_array();
  check(cases != nullptr, "fixture encode_cases is an array");
  if (cases != nullptr) {
    for (std::size_t index = 0; index < cases->size(); ++index) {
      const Value* const text_value = required((*cases)[index], "text");
      const Value* const ids_value = required((*cases)[index], "token_ids");
      std::vector<std::uint32_t> expected;
      check(text_value != nullptr && text_value->as_string() != nullptr,
            "fixture encode text is a string");
      check(ids_value != nullptr && token_array(*ids_value, expected),
            "fixture token_ids are uint32");
      if (text_value != nullptr && text_value->as_string() != nullptr &&
          ids_value != nullptr && token_array(*ids_value, expected)) {
        check_encode(tokenizer,
                     *text_value->as_string(),
                     expected,
                     "fixture encode case " + std::to_string(index));
      }
    }
  }

  const Value* const chat = required(fixture, "chat_template_case");
  const Value* const messages_value =
      chat == nullptr ? nullptr : required(*chat, "messages");
  const Value::Array* const messages_array =
      messages_value == nullptr ? nullptr : messages_value->as_array();
  std::vector<ChatMessage> messages;
  if (messages_array != nullptr) {
    for (const Value& entry : *messages_array) {
      const Value* const role = required(entry, "role");
      const Value* const content = required(entry, "content");
      if (role != nullptr && content != nullptr && role->as_string() != nullptr &&
          content->as_string() != nullptr) {
        messages.push_back(ChatMessage{*role->as_string(), *content->as_string()});
      }
    }
  }
  q3x::text::Qwen36ChatOptions chat_options;
  chat_options.add_generation_prompt = true;
  chat_options.enable_thinking = false;
  const auto formatted = tokenizer.format_qwen36_chat(messages, chat_options);
  check(formatted.ok(), "fixture chat formatting succeeds");
  if (formatted.ok() && chat != nullptr) {
    const Value* const rendered = required(*chat, "rendered");
    std::vector<std::uint32_t> expected_ids;
    const Value* const ids = required(*chat, "token_ids");
    check(rendered != nullptr && rendered->as_string() != nullptr &&
              formatted.rendered == *rendered->as_string(),
          "enable_thinking=false rendered chat matches oracle exactly");
    check(ids != nullptr && token_array(*ids, expected_ids) &&
              formatted.token_ids == expected_ids,
          "enable_thinking=false chat token ids match oracle exactly");
  }
}

void differential_smoke(Tokenizer& tokenizer) {
  const std::vector<std::pair<std::string, std::vector<std::uint32_t>>> cases{
      {"", {}},
      {"hello", {14556}},
      {" Hello", {21251}},
      {"I'm testing Qwen's tokenizer.",
       {40, 2688, 7262, 1167, 16451, 579, 44424, 13}},
      {"e\xCC\x81 caf\xC3\xA9 \xC3\x85ngstr\xC3\xB6m",
       {933, 50203, 76533, 938, 485, 82628}},
      {"\xE4\xB8\xAD\xE6\x96\x87"
       "English\xE6\xB7\xB7\xE5\x90\x88 12345",
       {99986, 21874, 101742, 220, 16, 17, 18, 19, 20}},
      {"\xF0\x9F\x9A\x80 Jetson\xE2\x80\x94Orin\xE2\x80\xA6 \xE2\x9C\x85",
       {9008, 248, 222, 23597, 901, 2218, 2126, 258, 1873, 189551}},
      {"a\n\n b\r\nc\t ", {64, 271, 292, 317, 66, 3551}},
      {"   ", {262}},
      {"\n\n", {271}},
      {"<|im_start|>user\nhi<|im_end|>\n",
       {248045, 846, 198, 5834, 248046, 198}},
      {"<think>abc</think>", {248068, 13290, 248069}},
      {"foo_bar+baz=42;", {7724, 13975, 33932, 1322, 28, 19, 17, 26}},
      {"\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D"
       "\xE0\xA4\xA4\xE0\xA5\x87 \xE0\xA4\xA6\xE0\xA5\x81"
       "\xE0\xA4\xA8\xE0\xA4\xBF\xE0\xA4\xAF\xE0\xA4\xBE",
       {58069, 84237, 150104, 153348, 184642, 235886}},
      {"\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7 "
       "\xD8\xA8\xD8\xA7\xD9\x84\xD8\xB9\xD8\xA7\xD9\x84\xD9\x85",
       {148739, 28850, 150027, 182946, 149650}},
  };
  for (std::size_t index = 0; index < cases.size(); ++index) {
    check_encode(tokenizer,
                 cases[index].first,
                 cases[index].second,
                 "Python differential case " + std::to_string(index));
  }

  const auto normal = tokenizer.encode("Hello, world!");
  if (normal.ok()) {
    const auto decoded = tokenizer.decode(normal.token_ids);
    check(decoded.ok() && decoded.text == "Hello, world!",
          "ByteLevel decode round-trip");
  }
  const auto special = tokenizer.encode("<|im_start|>x<|im_end|><think>y</think>");
  if (special.ok()) {
    q3x::text::DecodeOptions options;
    options.skip_special_tokens = true;
    const auto decoded = tokenizer.decode(special.token_ids, options);
    check(decoded.ok() && decoded.text == "x<think>y</think>",
          "skip_special_tokens follows pinned special flags");
  }

  q3x::text::Qwen36ChatOptions no_generation;
  no_generation.add_generation_prompt = false;
  no_generation.enable_thinking = false;
  const auto completed = tokenizer.format_qwen36_chat(
      {{"user", "one"}, {"assistant", "two"}}, no_generation);
  check(completed.ok() &&
            completed.rendered ==
                "<|im_start|>user\none<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                "two<|im_end|>\n" &&
            completed.token_ids ==
                std::vector<std::uint32_t>{248045,
                                           846,
                                           198,
                                           588,
                                           248046,
                                           198,
                                           248045,
                                           74455,
                                           198,
                                           248068,
                                           271,
                                           248069,
                                           271,
                                           19186,
                                           248046,
                                           198},
        "completed plain assistant chat matches Transformers oracle");

  q3x::text::Qwen36ChatOptions thinking;
  thinking.add_generation_prompt = true;
  thinking.enable_thinking = true;
  const auto thinking_prompt =
      tokenizer.format_qwen36_chat({{"user", "one"}}, thinking);
  check(thinking_prompt.ok() &&
            thinking_prompt.rendered ==
                "<|im_start|>user\none<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n" &&
            thinking_prompt.token_ids ==
                std::vector<std::uint32_t>{248045,
                                           846,
                                           198,
                                           588,
                                           248046,
                                           198,
                                           248045,
                                           74455,
                                           198,
                                           248068,
                                           198},
        "enable_thinking=true generation prompt matches Transformers oracle");
}

void failure_cases(Tokenizer& tokenizer, const std::string& tokenizer_json) {
  const std::string invalid_utf8(1, static_cast<char>(0xC0));
  const auto invalid = tokenizer.encode(invalid_utf8);
  check(!invalid.ok() && invalid.error.code == TokenizerErrorCode::kInvalidUtf8,
        "invalid UTF-8 fails with a structured error");

  const auto invalid_id = tokenizer.decode({999'999});
  check(!invalid_id.ok() &&
            invalid_id.error.code == TokenizerErrorCode::kInvalidTokenId,
        "invalid token id fails with a structured error");

  std::string oversized(tokenizer.limits().max_input_bytes + 1, 'a');
  const auto too_large = tokenizer.encode(oversized);
  check(!too_large.ok() &&
            too_large.error.code == TokenizerErrorCode::kInputTooLarge,
        "input byte limit is enforced before tokenization");

  const auto unsupported_role = tokenizer.format_qwen36_chat({{"tool", "x"}});
  check(!unsupported_role.ok() &&
            unsupported_role.error.code == TokenizerErrorCode::kUnsupportedChat,
        "unsupported chat roles fail closed");

  q3x::text::TokenizerLimits tiny;
  tiny.max_tokenizer_bytes = 128;
  const auto bounded = Tokenizer::load_json(tokenizer_json, tiny);
  check(!bounded.ok() &&
            bounded.error.code == TokenizerErrorCode::kTokenizerTooLarge,
        "tokenizer byte limit is enforced before JSON parsing");

  std::string schema_drift = tokenizer_json;
  const std::string needle = "\"type\": \"NFC\"";
  const std::size_t position = schema_drift.find(needle);
  check(position != std::string::npos, "schema drift test finds normalizer");
  if (position != std::string::npos) {
    schema_drift.replace(position, needle.size(), "\"type\": \"NFD\"");
    const auto drift = Tokenizer::load_json(schema_drift);
    check(!drift.ok() &&
              drift.error.code == TokenizerErrorCode::kSchemaMismatch,
          "normalizer schema drift is rejected explicitly");
  }
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc != 3) {
    std::cerr << "usage: tokenizer_test FIXTURE TOKENIZER_JSON\n";
    return 2;
  }

  std::string fixture_json;
  check(read_file(argv[1], fixture_json), "read tokenizer oracle fixture");
  q3x::io::json::ParseOptions fixture_options;
  fixture_options.max_input_bytes = 100'000;
  fixture_options.max_values = 10'000;
  fixture_options.max_container_items = 10'000;
  const auto fixture = q3x::io::json::parse(fixture_json, fixture_options);
  check(fixture.ok(), "parse tokenizer oracle fixture");
  if (!fixture.ok()) {
    return 1;
  }

  if (std::string_view(argv[2]) == "-") {
    std::cout << "SKIP: set Q3X_TOKENIZER_TEST_JSON to the pinned tokenizer.json\n";
    return 77;
  }
  std::string tokenizer_json;
  if (!read_file(argv[2], tokenizer_json)) {
    std::cout << "SKIP: pinned tokenizer.json is not readable\n";
    return 77;
  }
  auto loaded = Tokenizer::load_json(tokenizer_json);
  check(loaded.ok(), "load pinned Qwen 3.6 tokenizer.json");
  if (!loaded.ok()) {
    std::cerr << "  error=" << q3x::text::to_string(loaded.error.code)
              << " message=" << loaded.error.message << '\n';
    return 1;
  }

  fixture_cases(*loaded.tokenizer, *fixture.value);
  differential_smoke(*loaded.tokenizer);
  failure_cases(*loaded.tokenizer, tokenizer_json);

  if (failures != 0) {
    std::cerr << failures << " tokenizer test(s) failed\n";
    return 1;
  }
  std::cout << "tokenizer tests passed\n";
  return 0;
}
