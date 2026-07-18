#include "q3x/io/json.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
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

void test_complete_document(TestContext& test) {
    using namespace q3x::io::json;
    const auto result = parse(
        R"({"null":null,"bool":true,"false":false,"number":-12.5e+2,"array":[1,"x"],"unicode":"A\u4E2D\uD83D\uDE80","escaped":"\"\\\/\b\f\n\r\t"})");
    test.expect(result.ok(), "complete JSON document parses");
    if (!result) {
        return;
    }

    const auto* root = result.value->as_object();
    test.expect(root != nullptr, "root is an object");
    test.expect(result.value->find("null") != nullptr &&
                    result.value->find("null")->is_null(),
                "null value is retained");
    const auto* truth = result.value->find("bool")->as_bool();
    const auto* falsehood = result.value->find("false")->as_bool();
    test.expect(truth != nullptr && *truth, "true value is retained");
    test.expect(falsehood != nullptr && !*falsehood, "false value is retained");

    const auto* number = result.value->find("number")->as_number();
    test.expect(number != nullptr && number->text() == "-12.5e+2",
                "number spelling is retained exactly");
    double converted = 0.0;
    test.expect(number != nullptr && number->to_double(converted) &&
                    converted == -1250.0,
                "floating number conversion works");

    const auto* array = result.value->find("array")->as_array();
    test.expect(array != nullptr && array->size() == 2,
                "array elements are retained");
    const auto* unicode = result.value->find("unicode")->as_string();
    test.expect(unicode != nullptr && *unicode == "A\xE4\xB8\xAD\xF0\x9F\x9A\x80",
                "BMP and surrogate-pair escapes become UTF-8");
    const auto* escaped = result.value->find("escaped")->as_string();
    test.expect(escaped != nullptr &&
                    *escaped == std::string("\"\\/\b\f\n\r\t", 8),
                "JSON string escapes are decoded");
    test.expect(result.value->find("missing") == nullptr,
                "missing object member is reported as null pointer");
}

void test_integer_conversions(TestContext& test) {
    using namespace q3x::io::json;
    const auto result = parse(
        "[18446744073709551615,9223372036854775807,-9223372036854775808,"
        "18446744073709551616,1.0]");
    test.expect(result.ok(), "large integer spellings parse without double conversion");
    if (!result) {
        return;
    }
    const auto* values = result.value->as_array();
    test.expect(values != nullptr && values->size() == 5,
                "integer conversion fixture has five elements");
    if (values == nullptr || values->size() != 5) {
        return;
    }

    std::uint64_t unsigned_value = 0;
    std::int64_t signed_value = 0;
    test.expect((*values)[0].as_number()->to_uint64(unsigned_value) &&
                    unsigned_value == std::numeric_limits<std::uint64_t>::max(),
                "uint64 maximum converts exactly");
    test.expect((*values)[1].as_number()->to_int64(signed_value) &&
                    signed_value == std::numeric_limits<std::int64_t>::max(),
                "int64 maximum converts exactly");
    test.expect((*values)[2].as_number()->to_int64(signed_value) &&
                    signed_value == std::numeric_limits<std::int64_t>::min(),
                "int64 minimum converts exactly");
    test.expect(!(*values)[3].as_number()->to_uint64(unsigned_value),
                "uint64 overflow is rejected");
    test.expect(!(*values)[4].as_number()->to_uint64(unsigned_value),
                "non-integral JSON spelling is not an unsigned integer");
}

void expect_error(std::string_view input,
                  q3x::io::json::ErrorCode expected,
                  TestContext& test,
                  std::string_view message) {
    const auto result = q3x::io::json::parse(input);
    test.expect(!result.ok() && result.error.code == expected, message);
    test.expect(!result.ok() && result.error.offset <= input.size(),
                "JSON error includes an in-range byte offset");
}

void test_rejections(TestContext& test) {
    using q3x::io::json::ErrorCode;
    expect_error(R"({"a":1,"a":2})",
                 ErrorCode::kDuplicateKey,
                 test,
                 "duplicate object keys are rejected");
    expect_error(R"({"a":1,"\u0061":2})",
                 ErrorCode::kDuplicateKey,
                 test,
                 "escaped duplicate object keys are rejected");
    expect_error("true false",
                 ErrorCode::kTrailingCharacters,
                 test,
                 "trailing non-whitespace is rejected");
    expect_error("01", ErrorCode::kInvalidNumber, test, "leading zero is rejected");
    expect_error("1.", ErrorCode::kInvalidNumber, test, "empty fraction is rejected");
    expect_error("1e+", ErrorCode::kInvalidNumber, test, "empty exponent is rejected");
    expect_error("-", ErrorCode::kInvalidNumber, test, "bare minus is rejected");
    expect_error(R"("\q")",
                 ErrorCode::kInvalidEscape,
                 test,
                 "unknown string escape is rejected");
    expect_error(R"("\u12XZ")",
                 ErrorCode::kInvalidUnicodeEscape,
                 test,
                 "malformed Unicode escape is rejected");
    expect_error(R"("\uD800")",
                 ErrorCode::kInvalidUnicodeScalar,
                 test,
                 "unpaired high surrogate is rejected");
    expect_error(R"("\uDC00")",
                 ErrorCode::kInvalidUnicodeScalar,
                 test,
                 "unpaired low surrogate is rejected");
    expect_error(std::string("\"line\nfeed\""),
                 ErrorCode::kInvalidString,
                 test,
                 "raw control characters are rejected in strings");

    std::string invalid_utf8 = "\"";
    invalid_utf8.push_back(static_cast<char>(0xC0));
    invalid_utf8.push_back(static_cast<char>(0x80));
    invalid_utf8.push_back('"');
    expect_error(invalid_utf8,
                 ErrorCode::kInvalidUtf8,
                 test,
                 "overlong raw UTF-8 is rejected");
}

void test_limits(TestContext& test) {
    using namespace q3x::io::json;
    ParseOptions size_options;
    size_options.max_input_bytes = 3;
    auto result = parse("null", size_options);
    test.expect(!result && result.error.code == ErrorCode::kInputTooLarge &&
                    result.error.offset == 3,
                "input byte limit is enforced with an offset");

    ParseOptions depth_options;
    depth_options.max_nesting_depth = 2;
    result = parse("[[0]]", depth_options);
    test.expect(result.ok(), "configured nesting depth is inclusive");
    result = parse("[[[0]]]", depth_options);
    test.expect(!result && result.error.code == ErrorCode::kNestingTooDeep &&
                    result.error.offset == 2,
                "nesting depth limit is enforced at the nested container");

    depth_options.max_nesting_depth = 0;
    test.expect(parse("1", depth_options).ok(),
                "zero nesting limit still permits scalar JSON");
    result = parse("{}", depth_options);
    test.expect(!result && result.error.code == ErrorCode::kNestingTooDeep,
                "zero nesting limit rejects a root container");

    std::string excessively_deep(kAbsoluteMaxNestingDepth + 1, '[');
    excessively_deep.push_back('0');
    excessively_deep.append(kAbsoluteMaxNestingDepth + 1, ']');
    depth_options.max_nesting_depth =
        std::numeric_limits<std::size_t>::max();
    result = parse(excessively_deep, depth_options);
    test.expect(!result && result.error.code == ErrorCode::kNestingTooDeep &&
                    result.error.offset == kAbsoluteMaxNestingDepth,
                "absolute recursion ceiling applies even to an unbounded option");

    ParseOptions value_options;
    value_options.max_values = 3;
    result = parse(R"([0,{"nested":1}])", value_options);
    test.expect(!result && result.error.code == ErrorCode::kTooManyValues &&
                    result.error.offset == 13,
                "value budget is global across nested containers");
    value_options.max_values = 0;
    result = parse("null", value_options);
    test.expect(!result && result.error.code == ErrorCode::kTooManyValues &&
                    result.error.offset == 0,
                "zero value budget rejects the root before DOM construction");

    ParseOptions item_options;
    item_options.max_container_items = 3;
    result = parse(R"({"a":[0,1],"b":2})", item_options);
    test.expect(!result &&
                    result.error.code == ErrorCode::kTooManyContainerItems &&
                    result.error.offset == 11,
                "container item budget is global across arrays and objects");
    item_options.max_container_items = 0;
    test.expect(parse("[]", item_options).ok() &&
                    parse("{}", item_options).ok(),
                "zero container item budget permits empty containers");
    result = parse("[0]", item_options);
    test.expect(!result &&
                    result.error.code == ErrorCode::kTooManyContainerItems &&
                    result.error.offset == 1,
                "container item budget is checked before retaining an element");
}

void test_deterministic_malformed_corpus(TestContext& test) {
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    bool offsets_are_bounded = true;
    for (std::size_t sample = 0; sample < 5000; ++sample) {
        state ^= state << 7U;
        state ^= state >> 9U;
        const auto length = static_cast<std::size_t>(state % 65U);
        std::string input(length, '\0');
        for (char& byte : input) {
            state ^= state << 7U;
            state ^= state >> 9U;
            byte = static_cast<char>(state & 0xFFU);
        }
        const auto result = q3x::io::json::parse(input);
        if (!result && result.error.offset > input.size()) {
            offsets_are_bounded = false;
            break;
        }
    }
    test.expect(offsets_are_bounded,
                "deterministic arbitrary-byte corpus returns bounded errors");
}

}  // namespace

int main() {
    TestContext test;
    test_complete_document(test);
    test_integer_conversions(test);
    test_rejections(test);
    test_limits(test);
    test_deterministic_malformed_corpus(test);

    if (test.failures() != 0) {
        std::cerr << test.failures() << " JSON test assertion(s) failed\n";
        return 1;
    }
    std::cout << "JSON tests passed\n";
    return 0;
}
