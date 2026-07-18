#include "q3x/core/sha256.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace q3x::core {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

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

std::string Sha256Digest::hex() const {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const std::uint8_t byte : bytes) {
    out << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return out.str();
}

Sha256::Sha256() noexcept
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

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
      transform(buffer_.data());
      buffer_size_ = 0;
    }
  }

  while (remaining >= buffer_.size()) {
    transform(input);
    input += buffer_.size();
    remaining -= buffer_.size();
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
    transform(buffer_.data());
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
            buffer_.begin() + 56, 0U);
  for (std::size_t index = 0; index < 8U; ++index) {
    buffer_[63U - index] =
        static_cast<std::uint8_t>(bit_length >> (index * 8U));
  }
  transform(buffer_.data());

  for (std::size_t index = 0; index < state_.size(); ++index) {
    WriteBigEndian32(state_[index], digest_.bytes.data() + index * 4U);
  }
  finalized_ = true;
  return digest_;
}

void Sha256::transform(const std::uint8_t* const block) noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t index = 0; index < 16U; ++index) {
    schedule[index] = ReadBigEndian32(block + index * 4U);
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

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];

  for (std::size_t index = 0; index < schedule.size(); ++index) {
    const std::uint32_t sum1 =
        RotateRight(e, 6U) ^ RotateRight(e, 11U) ^ RotateRight(e, 25U);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 =
        h + sum1 + choose + kRoundConstants[index] + schedule[index];
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

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
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
