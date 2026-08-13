#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::core {

struct Sha256Digest {
  std::array<std::uint8_t, 32> bytes{};

  [[nodiscard]] std::string hex() const;

  [[nodiscard]] friend bool operator==(
      const Sha256Digest& left, const Sha256Digest& right) noexcept {
    return left.bytes == right.bytes;
  }
};

class Sha256 {
 public:
  Sha256() noexcept;

  // Returns false for a null non-empty input, input-length overflow, or an
  // update attempted after finalize().
  [[nodiscard]] bool update(const void* data, std::size_t size) noexcept;
  [[nodiscard]] Sha256Digest finalize() noexcept;

 private:
  void transform_blocks(const std::uint8_t* blocks,
                        std::size_t block_count) noexcept;

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t total_bytes_ = 0;
  bool finalized_ = false;
  Sha256Digest digest_{};
};

[[nodiscard]] Sha256Digest sha256(std::string_view input) noexcept;

struct Sha256FileResult {
  std::optional<Sha256Digest> digest;
  std::string error;

  [[nodiscard]] bool ok() const noexcept { return digest.has_value(); }
};

// Hashes a file using bounded streaming reads. Tensor payloads are not mapped
// or retained in memory.
[[nodiscard]] Sha256FileResult sha256_file(const std::string& path);

}  // namespace q3x::core
