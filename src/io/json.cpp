#include "q3x/io/json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

namespace q3x::io::json {

Number::Number(std::string text) : text_(std::move(text)) {}

const std::string& Number::text() const noexcept { return text_; }

bool Number::is_integer() const noexcept {
    return text_.find_first_of(".eE") == std::string::npos;
}

bool Number::to_uint64(std::uint64_t& value) const noexcept {
    if (!is_integer() || text_.empty() || text_.front() == '-') {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto result =
        std::from_chars(text_.data(), text_.data() + text_.size(), parsed, 10);
    if (result.ec != std::errc{} || result.ptr != text_.data() + text_.size()) {
        return false;
    }
    value = parsed;
    return true;
}

bool Number::to_int64(std::int64_t& value) const noexcept {
    if (!is_integer() || text_.empty()) {
        return false;
    }
    std::int64_t parsed = 0;
    const auto result =
        std::from_chars(text_.data(), text_.data() + text_.size(), parsed, 10);
    if (result.ec != std::errc{} || result.ptr != text_.data() + text_.size()) {
        return false;
    }
    value = parsed;
    return true;
}

bool Number::to_double(double& value) const noexcept {
    if (text_.empty()) {
        return false;
    }
    double parsed = 0.0;
    const auto result = std::from_chars(text_.data(),
                                        text_.data() + text_.size(),
                                        parsed,
                                        std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != text_.data() + text_.size() ||
        !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

Value::Value() noexcept = default;

Value::Value(bool value) noexcept : type_(Type::kBool), bool_value_(value) {}

Value::Value(Number value)
    : type_(Type::kNumber), number_value_(std::move(value)) {}

Value::Value(std::string value)
    : type_(Type::kString), string_value_(std::move(value)) {}

Value::Value(Array value)
    : type_(Type::kArray), array_value_(std::move(value)) {}

Value::Value(Object value)
    : type_(Type::kObject), object_value_(std::move(value)) {}

Value Value::make_bool(bool value) { return Value(value); }

Value Value::make_number(std::string text) {
    return Value(Number(std::move(text)));
}

Value Value::make_string(std::string text) { return Value(std::move(text)); }

Value Value::make_array(Array values) { return Value(std::move(values)); }

Value Value::make_object(Object values) { return Value(std::move(values)); }

Value::Type Value::type() const noexcept { return type_; }

bool Value::is_null() const noexcept { return type_ == Type::kNull; }

const bool* Value::as_bool() const noexcept {
    return type_ == Type::kBool ? &bool_value_ : nullptr;
}

const Number* Value::as_number() const noexcept {
    return type_ == Type::kNumber ? &*number_value_ : nullptr;
}

const std::string* Value::as_string() const noexcept {
    return type_ == Type::kString ? &string_value_ : nullptr;
}

const Value::Array* Value::as_array() const noexcept {
    return type_ == Type::kArray ? &array_value_ : nullptr;
}

const Value::Object* Value::as_object() const noexcept {
    return type_ == Type::kObject ? &object_value_ : nullptr;
}

const Value* Value::find(std::string_view key) const noexcept {
    if (type_ != Type::kObject) {
        return nullptr;
    }
    const auto entry = object_value_.find(key);
    return entry == object_value_.end() ? nullptr : &entry->second;
}

namespace {

constexpr bool is_json_whitespace(char value) noexcept {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

constexpr bool is_decimal_digit(char value) noexcept {
    return value >= '0' && value <= '9';
}

constexpr int hex_value(char value) noexcept {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return 10 + value - 'a';
    }
    if (value >= 'A' && value <= 'F') {
        return 10 + value - 'A';
    }
    return -1;
}

class Parser {
public:
    Parser(std::string_view input,
           std::size_t max_depth,
           std::size_t max_values,
           std::size_t max_container_items) noexcept
        : input_(input),
          max_depth_(max_depth),
          max_values_(max_values),
          max_container_items_(max_container_items) {}

    ParseResult run() {
        skip_whitespace();
        Value root;
        if (!parse_value(0, root)) {
            return {std::nullopt, error_};
        }
        skip_whitespace();
        if (position_ != input_.size()) {
            fail(ErrorCode::kTrailingCharacters, position_);
            return {std::nullopt, error_};
        }
        ParseResult result;
        result.value.emplace(std::move(root));
        return result;
    }

private:
    void skip_whitespace() noexcept {
        while (position_ < input_.size() && is_json_whitespace(input_[position_])) {
            ++position_;
        }
    }

    bool fail(ErrorCode code, std::size_t offset) noexcept {
        if (error_.ok()) {
            error_.code = code;
            error_.offset = offset;
        }
        return false;
    }

    bool consume_value(std::size_t offset) noexcept {
        if (value_count_ >= max_values_) {
            return fail(ErrorCode::kTooManyValues, offset);
        }
        ++value_count_;
        return true;
    }

    bool consume_container_item(std::size_t offset) noexcept {
        if (container_item_count_ >= max_container_items_) {
            return fail(ErrorCode::kTooManyContainerItems, offset);
        }
        ++container_item_count_;
        return true;
    }

    bool parse_value(std::size_t depth, Value& output) {
        skip_whitespace();
        if (position_ == input_.size()) {
            return fail(ErrorCode::kUnexpectedEnd, position_);
        }
        if (!consume_value(position_)) {
            return false;
        }

        switch (input_[position_]) {
            case 'n':
                if (!parse_literal("null")) {
                    return false;
                }
                output = Value{};
                return true;
            case 't':
                if (!parse_literal("true")) {
                    return false;
                }
                output = Value::make_bool(true);
                return true;
            case 'f':
                if (!parse_literal("false")) {
                    return false;
                }
                output = Value::make_bool(false);
                return true;
            case '"': {
                std::string value;
                if (!parse_string(value)) {
                    return false;
                }
                output = Value::make_string(std::move(value));
                return true;
            }
            case '[':
                return parse_array(depth, output);
            case '{':
                return parse_object(depth, output);
            default:
                if (input_[position_] == '-' || is_decimal_digit(input_[position_])) {
                    return parse_number(output);
                }
                return fail(ErrorCode::kUnexpectedToken, position_);
        }
    }

    bool parse_literal(std::string_view literal) noexcept {
        const auto start = position_;
        if (input_.size() - position_ < literal.size() ||
            input_.substr(position_, literal.size()) != literal) {
            return fail(ErrorCode::kInvalidLiteral, start);
        }
        position_ += literal.size();
        return true;
    }

    bool parse_number(Value& output) {
        const auto start = position_;
        if (input_[position_] == '-') {
            ++position_;
            if (position_ == input_.size()) {
                return fail(ErrorCode::kInvalidNumber, position_);
            }
        }

        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && is_decimal_digit(input_[position_])) {
                return fail(ErrorCode::kInvalidNumber, position_);
            }
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            do {
                ++position_;
            } while (position_ < input_.size() &&
                     is_decimal_digit(input_[position_]));
        } else {
            return fail(ErrorCode::kInvalidNumber, position_);
        }

        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            if (position_ == input_.size() ||
                !is_decimal_digit(input_[position_])) {
                return fail(ErrorCode::kInvalidNumber, position_);
            }
            do {
                ++position_;
            } while (position_ < input_.size() &&
                     is_decimal_digit(input_[position_]));
        }

        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            if (position_ == input_.size() ||
                !is_decimal_digit(input_[position_])) {
                return fail(ErrorCode::kInvalidNumber, position_);
            }
            do {
                ++position_;
            } while (position_ < input_.size() &&
                     is_decimal_digit(input_[position_]));
        }

        output = Value::make_number(
            std::string(input_.substr(start, position_ - start)));
        return true;
    }

    bool parse_array(std::size_t depth, Value& output) {
        const auto start = position_;
        if (depth >= max_depth_) {
            return fail(ErrorCode::kNestingTooDeep, start);
        }
        ++position_;
        skip_whitespace();

        Value::Array values;
        if (position_ < input_.size() && input_[position_] == ']') {
            ++position_;
            output = Value::make_array(std::move(values));
            return true;
        }

        while (true) {
            if (!consume_container_item(position_)) {
                return false;
            }
            Value value;
            if (!parse_value(depth + 1, value)) {
                return false;
            }
            values.emplace_back(std::move(value));
            skip_whitespace();
            if (position_ == input_.size()) {
                return fail(ErrorCode::kUnexpectedEnd, position_);
            }
            if (input_[position_] == ']') {
                ++position_;
                output = Value::make_array(std::move(values));
                return true;
            }
            if (input_[position_] != ',') {
                return fail(ErrorCode::kUnexpectedToken, position_);
            }
            ++position_;
        }
    }

    bool parse_object(std::size_t depth, Value& output) {
        const auto start = position_;
        if (depth >= max_depth_) {
            return fail(ErrorCode::kNestingTooDeep, start);
        }
        ++position_;
        skip_whitespace();

        Value::Object values;
        if (position_ < input_.size() && input_[position_] == '}') {
            ++position_;
            output = Value::make_object(std::move(values));
            return true;
        }

        while (true) {
            skip_whitespace();
            if (position_ == input_.size()) {
                return fail(ErrorCode::kUnexpectedEnd, position_);
            }
            if (input_[position_] != '"') {
                return fail(ErrorCode::kUnexpectedToken, position_);
            }
            if (!consume_container_item(position_)) {
                return false;
            }
            const auto key_offset = position_;
            std::string key;
            if (!parse_string(key)) {
                return false;
            }
            if (values.find(key) != values.end()) {
                return fail(ErrorCode::kDuplicateKey, key_offset);
            }

            skip_whitespace();
            if (position_ == input_.size()) {
                return fail(ErrorCode::kUnexpectedEnd, position_);
            }
            if (input_[position_] != ':') {
                return fail(ErrorCode::kUnexpectedToken, position_);
            }
            ++position_;

            Value value;
            if (!parse_value(depth + 1, value)) {
                return false;
            }
            values.emplace(std::move(key), std::move(value));

            skip_whitespace();
            if (position_ == input_.size()) {
                return fail(ErrorCode::kUnexpectedEnd, position_);
            }
            if (input_[position_] == '}') {
                ++position_;
                output = Value::make_object(std::move(values));
                return true;
            }
            if (input_[position_] != ',') {
                return fail(ErrorCode::kUnexpectedToken, position_);
            }
            ++position_;
        }
    }

    bool parse_hex4(std::uint16_t& value) noexcept {
        value = 0;
        for (int index = 0; index < 4; ++index) {
            if (position_ == input_.size()) {
                return fail(ErrorCode::kInvalidUnicodeEscape, position_);
            }
            const auto digit = hex_value(input_[position_]);
            if (digit < 0) {
                return fail(ErrorCode::kInvalidUnicodeEscape, position_);
            }
            value = static_cast<std::uint16_t>((value << 4U) |
                                               static_cast<unsigned>(digit));
            ++position_;
        }
        return true;
    }

    static void append_utf8(std::uint32_t scalar, std::string& output) {
        if (scalar <= 0x7FU) {
            output.push_back(static_cast<char>(scalar));
        } else if (scalar <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (scalar >> 6U)));
            output.push_back(static_cast<char>(0x80U | (scalar & 0x3FU)));
        } else if (scalar <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (scalar >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((scalar >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (scalar & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (scalar >> 18U)));
            output.push_back(
                static_cast<char>(0x80U | ((scalar >> 12U) & 0x3FU)));
            output.push_back(
                static_cast<char>(0x80U | ((scalar >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (scalar & 0x3FU)));
        }
    }

    bool copy_valid_utf8(std::string& output) {
        const auto start = position_;
        const auto lead = static_cast<unsigned char>(input_[position_]);
        std::size_t length = 0;
        if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            length = 3;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            length = 4;
        } else {
            return fail(ErrorCode::kInvalidUtf8, start);
        }
        if (input_.size() - position_ < length) {
            return fail(ErrorCode::kInvalidUtf8, start);
        }

        const auto second = static_cast<unsigned char>(input_[position_ + 1]);
        if ((second & 0xC0U) != 0x80U ||
            (lead == 0xE0U && second < 0xA0U) ||
            (lead == 0xEDU && second > 0x9FU) ||
            (lead == 0xF0U && second < 0x90U) ||
            (lead == 0xF4U && second > 0x8FU)) {
            return fail(ErrorCode::kInvalidUtf8, start);
        }
        for (std::size_t index = 2; index < length; ++index) {
            const auto byte =
                static_cast<unsigned char>(input_[position_ + index]);
            if ((byte & 0xC0U) != 0x80U) {
                return fail(ErrorCode::kInvalidUtf8, start);
            }
        }

        output.append(input_.substr(position_, length));
        position_ += length;
        return true;
    }

    bool parse_string(std::string& output) {
        ++position_;  // Opening quote was checked by the caller.
        while (position_ < input_.size()) {
            const auto byte = static_cast<unsigned char>(input_[position_]);
            if (byte == static_cast<unsigned char>('"')) {
                ++position_;
                return true;
            }
            if (byte == static_cast<unsigned char>('\\')) {
                const auto escape_offset = position_;
                ++position_;
                if (position_ == input_.size()) {
                    return fail(ErrorCode::kUnexpectedEnd, position_);
                }
                const char escape = input_[position_++];
                switch (escape) {
                    case '"':
                    case '\\':
                    case '/':
                        output.push_back(escape);
                        break;
                    case 'b':
                        output.push_back('\b');
                        break;
                    case 'f':
                        output.push_back('\f');
                        break;
                    case 'n':
                        output.push_back('\n');
                        break;
                    case 'r':
                        output.push_back('\r');
                        break;
                    case 't':
                        output.push_back('\t');
                        break;
                    case 'u': {
                        std::uint16_t first = 0;
                        if (!parse_hex4(first)) {
                            return false;
                        }
                        std::uint32_t scalar = first;
                        if (first >= 0xD800U && first <= 0xDBFFU) {
                            if (input_.size() - position_ < 2 ||
                                input_[position_] != '\\' ||
                                input_[position_ + 1] != 'u') {
                                return fail(ErrorCode::kInvalidUnicodeScalar,
                                            escape_offset);
                            }
                            position_ += 2;
                            std::uint16_t second = 0;
                            if (!parse_hex4(second)) {
                                return false;
                            }
                            if (second < 0xDC00U || second > 0xDFFFU) {
                                return fail(ErrorCode::kInvalidUnicodeScalar,
                                            position_ - 4);
                            }
                            scalar =
                                0x10000U +
                                ((static_cast<std::uint32_t>(first) - 0xD800U)
                                 << 10U) +
                                (static_cast<std::uint32_t>(second) - 0xDC00U);
                        } else if (first >= 0xDC00U && first <= 0xDFFFU) {
                            return fail(ErrorCode::kInvalidUnicodeScalar,
                                        escape_offset);
                        }
                        append_utf8(scalar, output);
                        break;
                    }
                    default:
                        return fail(ErrorCode::kInvalidEscape, position_ - 1);
                }
                continue;
            }
            if (byte <= 0x1FU) {
                return fail(ErrorCode::kInvalidString, position_);
            }
            if (byte < 0x80U) {
                output.push_back(input_[position_++]);
                continue;
            }
            if (!copy_valid_utf8(output)) {
                return false;
            }
        }
        return fail(ErrorCode::kUnexpectedEnd, position_);
    }

    std::string_view input_;
    std::size_t max_depth_ = 0;
    std::size_t max_values_ = 0;
    std::size_t max_container_items_ = 0;
    std::size_t value_count_ = 0;
    std::size_t container_item_count_ = 0;
    std::size_t position_ = 0;
    Error error_;
};

}  // namespace

ParseResult parse(std::string_view input, const ParseOptions& options) {
    if (input.size() > options.max_input_bytes) {
        return {std::nullopt,
                {ErrorCode::kInputTooLarge, options.max_input_bytes}};
    }
    try {
        const auto max_depth =
            std::min(options.max_nesting_depth, kAbsoluteMaxNestingDepth);
        return Parser(input,
                      max_depth,
                      options.max_values,
                      options.max_container_items)
            .run();
    } catch (const std::bad_alloc&) {
        return {std::nullopt, {ErrorCode::kAllocationFailure, 0}};
    } catch (const std::length_error&) {
        return {std::nullopt, {ErrorCode::kAllocationFailure, 0}};
    }
}

std::string_view Error::message() const noexcept { return to_string(code); }

std::string_view to_string(Value::Type type) noexcept {
    switch (type) {
        case Value::Type::kNull:
            return "null";
        case Value::Type::kBool:
            return "bool";
        case Value::Type::kNumber:
            return "number";
        case Value::Type::kString:
            return "string";
        case Value::Type::kArray:
            return "array";
        case Value::Type::kObject:
            return "object";
    }
    return "unknown";
}

std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kNone:
            return "valid JSON";
        case ErrorCode::kInputTooLarge:
            return "JSON input exceeds the configured byte limit";
        case ErrorCode::kUnexpectedEnd:
            return "unexpected end of JSON input";
        case ErrorCode::kUnexpectedToken:
            return "unexpected JSON token";
        case ErrorCode::kInvalidLiteral:
            return "invalid JSON literal";
        case ErrorCode::kInvalidNumber:
            return "invalid JSON number";
        case ErrorCode::kInvalidString:
            return "invalid character in JSON string";
        case ErrorCode::kInvalidEscape:
            return "invalid JSON string escape";
        case ErrorCode::kInvalidUnicodeEscape:
            return "invalid JSON Unicode escape";
        case ErrorCode::kInvalidUnicodeScalar:
            return "invalid JSON Unicode surrogate pair";
        case ErrorCode::kInvalidUtf8:
            return "invalid UTF-8 in JSON string";
        case ErrorCode::kDuplicateKey:
            return "duplicate JSON object key";
        case ErrorCode::kNestingTooDeep:
            return "JSON nesting exceeds the configured depth limit";
        case ErrorCode::kTooManyValues:
            return "JSON value count exceeds the configured limit";
        case ErrorCode::kTooManyContainerItems:
            return "JSON container item count exceeds the configured limit";
        case ErrorCode::kTrailingCharacters:
            return "non-whitespace characters follow the JSON value";
        case ErrorCode::kAllocationFailure:
            return "memory allocation failed while parsing JSON";
    }
    return "unknown JSON error";
}

}  // namespace q3x::io::json
