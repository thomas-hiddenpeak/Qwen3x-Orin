#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::io::json {

// JSON numbers are retained as their validated source spelling. This avoids
// losing precision when checkpoint dimensions and offsets exceed 2^53.
class Number {
public:
    explicit Number(std::string text);

    [[nodiscard]] const std::string& text() const noexcept;
    [[nodiscard]] bool is_integer() const noexcept;
    [[nodiscard]] bool to_uint64(std::uint64_t& value) const noexcept;
    [[nodiscard]] bool to_int64(std::int64_t& value) const noexcept;
    [[nodiscard]] bool to_double(double& value) const noexcept;

private:
    std::string text_;
};

class Value {
public:
    enum class Type : std::uint8_t {
        kNull,
        kBool,
        kNumber,
        kString,
        kArray,
        kObject,
    };

    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value, std::less<>>;

    Value() noexcept;

    [[nodiscard]] static Value make_bool(bool value);
    [[nodiscard]] static Value make_number(std::string text);
    [[nodiscard]] static Value make_string(std::string text);
    [[nodiscard]] static Value make_array(Array values);
    [[nodiscard]] static Value make_object(Object values);

    [[nodiscard]] Type type() const noexcept;
    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] const bool* as_bool() const noexcept;
    [[nodiscard]] const Number* as_number() const noexcept;
    [[nodiscard]] const std::string* as_string() const noexcept;
    [[nodiscard]] const Array* as_array() const noexcept;
    [[nodiscard]] const Object* as_object() const noexcept;
    [[nodiscard]] const Value* find(std::string_view key) const noexcept;

private:
    explicit Value(bool value) noexcept;
    explicit Value(Number value);
    explicit Value(std::string value);
    explicit Value(Array value);
    explicit Value(Object value);

    Type type_ = Type::kNull;
    bool bool_value_ = false;
    std::optional<Number> number_value_;
    std::string string_value_;
    Array array_value_;
    Object object_value_;
};

struct ParseOptions {
    // 100 MB matches the safetensors format's defensive header ceiling.
    std::size_t max_input_bytes = 100'000'000;
    std::size_t max_nesting_depth = 128;
    // Global DOM budgets. The root and every array/object value count toward
    // max_values. Every array element and object member additionally counts
    // toward max_container_items. These limits bound allocation amplification
    // from compact inputs such as [0,0,...].
    std::size_t max_values = 1'000'000;
    std::size_t max_container_items = 1'000'000;
};

enum class ErrorCode : std::uint8_t {
    kNone,
    kInputTooLarge,
    kUnexpectedEnd,
    kUnexpectedToken,
    kInvalidLiteral,
    kInvalidNumber,
    kInvalidString,
    kInvalidEscape,
    kInvalidUnicodeEscape,
    kInvalidUnicodeScalar,
    kInvalidUtf8,
    kDuplicateKey,
    kNestingTooDeep,
    kTooManyValues,
    kTooManyContainerItems,
    kTrailingCharacters,
    kAllocationFailure,
};

struct Error {
    ErrorCode code = ErrorCode::kNone;
    std::size_t offset = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == ErrorCode::kNone;
    }

    [[nodiscard]] std::string_view message() const noexcept;
};

struct ParseResult {
    std::optional<Value> value;
    Error error;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error.ok();
    }

    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// The implementation additionally caps recursive descent at 1024 levels even
// if a caller supplies a larger limit, preventing an untrusted option from
// turning deeply nested input into a process stack overflow.
inline constexpr std::size_t kAbsoluteMaxNestingDepth = 1024;

[[nodiscard]] ParseResult parse(std::string_view input,
                                const ParseOptions& options = {});

[[nodiscard]] std::string_view to_string(Value::Type type) noexcept;
[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

}  // namespace q3x::io::json
