#include "q3x/core/sha256.h"

#include "sha256_internal.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#if defined(Q3X_SHA256_ARM_SHA2_COMPILED)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace q3x::core {
namespace {

constexpr std::uint32_t RotateRight(const std::uint32_t value,
                                    const unsigned int bits) noexcept {
  return (value >> bits) | (value << (32U - bits));
}

std::uint32_t ReadBigEndian32(const std::uint8_t* data) noexcept {
  return (static_cast<std::uint32_t>(data[0]) << 24U) |
         (static_cast<std::uint32_t>(data[1]) << 16U) |
         (static_cast<std::uint32_t>(data[2]) << 8U) |
         static_cast<std::uint32_t>(data[3]);
}

void WriteBigEndian32(const std::uint32_t value, std::uint8_t* output) noexcept {
  output[0] = static_cast<std::uint8_t>(value >> 24U);
  output[1] = static_cast<std::uint8_t>(value >> 16U);
  output[2] = static_cast<std::uint8_t>(value >> 8U);
  output[3] = static_cast<std::uint8_t>(value);
}

}  // namespace

namespace internal {

void sha256_transform_blocks_portable(
    std::uint32_t* const state, const std::uint8_t* blocks,
    const std::size_t block_count) noexcept {
  for (std::size_t block_index = 0; block_index < block_count;
       ++block_index, blocks += 64U) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16U; ++index) {
      schedule[index] = ReadBigEndian32(blocks + index * 4U);
    }
    for (std::size_t index = 16; index < schedule.size(); ++index) {
      const std::uint32_t sigma0 = RotateRight(schedule[index - 15U], 7U) ^
                                   RotateRight(schedule[index - 15U], 18U) ^
                                   (schedule[index - 15U] >> 3U);
      const std::uint32_t sigma1 = RotateRight(schedule[index - 2U], 17U) ^
                                   RotateRight(schedule[index - 2U], 19U) ^
                                   (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + sigma0 +
                        schedule[index - 7U] + sigma1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sum1 =
          RotateRight(e, 6U) ^ RotateRight(e, 11U) ^ RotateRight(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 = h + sum1 + choose +
                                       kSha256RoundConstants[index] +
                                       schedule[index];
      const std::uint32_t sum0 =
          RotateRight(a, 2U) ^ RotateRight(a, 13U) ^ RotateRight(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
}

bool sha256_arm_sha2_compiled() noexcept {
#if defined(Q3X_SHA256_ARM_SHA2_COMPILED)
  return true;
#else
  return false;
#endif
}

bool sha256_arm_sha2_available() noexcept {
#if defined(Q3X_SHA256_ARM_SHA2_COMPILED)
  static const bool available =
      (::getauxval(AT_HWCAP) & static_cast<unsigned long>(HWCAP_SHA2)) != 0U;
  return available;
#else
  return false;
#endif
}

bool sha256_transform_blocks_arm_sha2(
    std::uint32_t* const state, const std::uint8_t* const blocks,
    const std::size_t block_count) noexcept {
#if defined(Q3X_SHA256_ARM_SHA2_COMPILED)
  if (!sha256_arm_sha2_available()) {
    return false;
  }
  sha256_transform_blocks_arm_sha2_unchecked(state, blocks, block_count);
  return true;
#else
  static_cast<void>(state);
  static_cast<void>(blocks);
  static_cast<void>(block_count);
  return false;
#endif
}

}  // namespace internal

std::string Sha256Digest::hex() const {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const std::uint8_t byte : bytes) {
    out << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return out.str();
}

Sha256::Sha256() noexcept : state_(internal::kSha256InitialState) {}

bool Sha256::update(const void* const data, const std::size_t size) noexcept {
  if (finalized_ || (data == nullptr && size != 0U) ||
      size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
    return false;
  }

  const auto* input = static_cast<const std::uint8_t*>(data);
  total_bytes_ += static_cast<std::uint64_t>(size);
  std::size_t remaining = size;

  if (buffer_size_ != 0U && remaining != 0U) {
    const std::size_t copied = std::min(remaining, buffer_.size() - buffer_size_);
    std::copy_n(input, copied, buffer_.data() + buffer_size_);
    buffer_size_ += copied;
    input += copied;
    remaining -= copied;
    if (buffer_size_ == buffer_.size()) {
      transform_blocks(buffer_.data(), 1U);
      buffer_size_ = 0;
    }
  }

  if (remaining >= buffer_.size()) {
    const std::size_t block_count = remaining / buffer_.size();
    const std::size_t transformed_bytes = block_count * buffer_.size();
    transform_blocks(input, block_count);
    input += transformed_bytes;
    remaining -= transformed_bytes;
  }
  if (remaining != 0U) {
    std::copy_n(input, remaining, buffer_.data());
    buffer_size_ = remaining;
  }
  return true;
}

Sha256Digest Sha256::finalize() noexcept {
  if (finalized_) {
    return digest_;
  }

  const std::uint64_t bit_length = total_bytes_ * 8U;
  buffer_[buffer_size_++] = 0x80U;
  if (buffer_size_ > 56U) {
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
              buffer_.end(), 0U);
    transform_blocks(buffer_.data(), 1U);
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
            buffer_.begin() + 56, 0U);
  for (std::size_t index = 0; index < 8U; ++index) {
    buffer_[63U - index] =
        static_cast<std::uint8_t>(bit_length >> (index * 8U));
  }
  transform_blocks(buffer_.data(), 1U);

  for (std::size_t index = 0; index < state_.size(); ++index) {
    WriteBigEndian32(state_[index], digest_.bytes.data() + index * 4U);
  }
  finalized_ = true;
  return digest_;
}

void Sha256::transform_blocks(const std::uint8_t* const blocks,
                              const std::size_t block_count) noexcept {
  if (internal::sha256_transform_blocks_arm_sha2(
          state_.data(), blocks, block_count)) {
    return;
  }
  internal::sha256_transform_blocks_portable(state_.data(), blocks,
                                              block_count);
}

Sha256Digest sha256(const std::string_view input) noexcept {
  Sha256 hasher;
  (void)hasher.update(input.data(), input.size());
  return hasher.finalize();
}

Sha256FileResult sha256_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {std::nullopt, "cannot open file"};
  }

  Sha256 hasher;
  std::array<char, 1024U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        !hasher.update(buffer.data(), static_cast<std::size_t>(count))) {
      return {std::nullopt, "file length exceeds SHA-256 input limit"};
    }
  }
  if (!input.eof()) {
    return {std::nullopt, "failed while reading file"};
  }
  return {hasher.finalize(), {}};
}

}  // namespace q3x::core
