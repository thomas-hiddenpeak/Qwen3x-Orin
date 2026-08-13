#include "q3x/core/sha256.h"

#include "sha256_internal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& message) {
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

void expect(const bool condition, const std::string& message) {
  if (!condition) {
    fail(message);
  }
}

std::vector<std::uint8_t> make_input(const std::size_t size,
                                     const std::size_t prefix = 0U) {
  std::vector<std::uint8_t> input(prefix + size);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint8_t>(index * 131U + 17U);
  }
  return input;
}

q3x::core::Sha256Digest portable_hash(const std::uint8_t* const input,
                                      const std::size_t size) {
  std::array<std::uint32_t, 8> state =
      q3x::core::internal::kSha256InitialState;
  const std::size_t complete_blocks = size / 64U;
  q3x::core::internal::sha256_transform_blocks_portable(
      state.data(), input, complete_blocks);

  std::array<std::uint8_t, 128> tail{};
  const std::size_t remainder = size % 64U;
  if (remainder != 0U) {
    std::copy_n(input + complete_blocks * 64U, remainder, tail.data());
  }
  tail[remainder] = 0x80U;
  const std::size_t tail_blocks = remainder < 56U ? 1U : 2U;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(size) * 8U;
  for (std::size_t index = 0; index < 8U; ++index) {
    tail[tail_blocks * 64U - 1U - index] =
        static_cast<std::uint8_t>(bit_length >> (index * 8U));
  }
  q3x::core::internal::sha256_transform_blocks_portable(
      state.data(), tail.data(), tail_blocks);

  q3x::core::Sha256Digest digest;
  for (std::size_t word = 0; word < state.size(); ++word) {
    for (std::size_t byte = 0; byte < 4U; ++byte) {
      digest.bytes[word * 4U + byte] =
          static_cast<std::uint8_t>(state[word] >> ((3U - byte) * 8U));
    }
  }
  return digest;
}

void expect_hash(const std::string_view input, const std::string_view expected,
                 const std::string& label) {
  const std::string actual = q3x::core::sha256(input).hex();
  if (actual != expected) {
    fail(label + ": expected " + std::string(expected) + ", got " + actual);
  }
}

void test_known_vectors() {
  expect_hash("",
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "empty string");
  expect_hash("abc",
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "abc");
  expect_hash("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
              "NIST two-block vector");
  expect_hash(std::string(1'000'000U, 'a'),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
              "NIST million-a vector");
}

void test_boundary_lengths() {
  constexpr std::array<std::pair<std::size_t, std::string_view>, 15> vectors = {{
      {0U, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {1U, "4a64a107f0cb32536e5bce6c98c393db21cca7f4ea187ba8c4dca8b51d4ea80a"},
      {55U, "59aaae80b8e7958f5757b4ac9274f1fd57ec0dc7aa8349319102316781b92006"},
      {56U, "dca902d31487ffab357ce36cc5abc6947fbb69127ec04c468ba5687268c4bf7c"},
      {63U, "b8e655e9e7ad413b96f0a371eb1d6db71dcca27f4eee35892517cbe4bfaa6f9a"},
      {64U, "e5146be62accc56709594cb45c651b361df94f622cbb09b91ea3ca7a2060bd53"},
      {65U, "4b6c6774f0cbcd776a90e187b9494e04ca880c0afe3ea1c5cad51c8bbcb41f74"},
      {119U, "3244509f8746f6fb511ca8a72a292a4d7d12e456cce283c03f169dbbcb0d512e"},
      {120U, "c5236e50172da3b69c903dcd3b3bb8ce855182f79364d1b59ed36367aa25e468"},
      {127U, "b4f5398bf618d741f2ab0f6e2fcde06f4949c20d476b5f6e1d4bf8618bc8973a"},
      {128U, "ed045c7c319e0c9286e1dea21c80bfd4a08eca43e02e13ee8a7a2c8b4b2baec9"},
      {129U, "d6f2fca0407b3f98c9ce29ae92a6cdf489506450fdbb29172f6b18987bc685db"},
      {1023U, "fec7806f5596be60cc8d7f1694fec937b3d7feb14ccd9be4f6c79f218d44f941"},
      {1024U, "575383deff0775797e1ff4c82da069141937c843a81cbe50959a96bcfa88140e"},
      {1025U, "b49a266849885c837e4295d53d91dc873fbd77f76fd0c06b12ff4da04fe3e3c9"},
  }};

  for (const auto& [size, expected] : vectors) {
    const std::vector<std::uint8_t> input = make_input(size);
    expect_hash(std::string_view(
                    reinterpret_cast<const char*>(input.data()), input.size()),
                expected, "boundary length " + std::to_string(size));
  }
}

void test_incremental_updates() {
  const std::vector<std::uint8_t> input = make_input(65'537U);
  const q3x::core::Sha256Digest expected = portable_hash(input.data(), input.size());
  constexpr std::array<std::size_t, 15> chunk_sizes = {
      1U, 2U, 3U, 7U, 15U, 31U, 55U, 56U,
      63U, 64U, 65U, 127U, 1024U, 4093U, 65'537U,
  };

  for (const std::size_t chunk_size : chunk_sizes) {
    q3x::core::Sha256 hasher;
    std::size_t offset = 0U;
    while (offset < input.size()) {
      const std::size_t count =
          std::min(chunk_size, input.size() - offset);
      expect(hasher.update(input.data() + offset, count),
             "incremental update accepted for chunk " +
                 std::to_string(chunk_size));
      expect(hasher.update(nullptr, 0U), "zero-byte null update accepted");
      offset += count;
    }
    expect(hasher.finalize() == expected,
           "incremental digest matches portable for chunk " +
               std::to_string(chunk_size));
    expect(hasher.finalize() == expected,
           "repeated finalize is stable for chunk " +
               std::to_string(chunk_size));
    const char extra = 'x';
    expect(!hasher.update(&extra, 1U), "update after finalize is rejected");
  }

  q3x::core::Sha256 invalid;
  expect(!invalid.update(nullptr, 1U), "non-empty null update is rejected");
}

void test_portable_public_equivalence() {
  for (std::size_t size = 0U; size <= 2048U; ++size) {
    const std::vector<std::uint8_t> input = make_input(size);
    const q3x::core::Sha256Digest expected = portable_hash(input.data(), size);
    const q3x::core::Sha256Digest actual = q3x::core::sha256(std::string_view(
        reinterpret_cast<const char*>(input.data()), input.size()));
    if (!(actual == expected)) {
      fail("portable/public mismatch at length " + std::to_string(size));
      return;
    }
  }
}

void test_arm_sha2_compression_equivalence() {
  const bool compiled = q3x::core::internal::sha256_arm_sha2_compiled();
  const bool available = q3x::core::internal::sha256_arm_sha2_available();
  expect(!available || compiled,
         "runtime ARM SHA2 availability requires compiled implementation");

  constexpr std::array<std::size_t, 9> block_counts = {
      0U, 1U, 2U, 3U, 4U, 7U, 16U, 63U, 257U,
  };
  for (const std::size_t block_count : block_counts) {
    // The one-byte prefix deliberately verifies unaligned source loads.
    const std::vector<std::uint8_t> storage = make_input(block_count * 64U, 1U);
    const std::uint8_t* const input = storage.data() + 1U;
    std::array<std::uint32_t, 8> portable = {
        0x12345678U, 0x9abcdef0U, 0x0badc0deU, 0xfeedfaceU,
        0x89abcdefU, 0x76543210U, 0x13579bdfU, 0x2468ace0U,
    };
    std::array<std::uint32_t, 8> accelerated = portable;
    const std::array<std::uint32_t, 8> original = portable;

    q3x::core::internal::sha256_transform_blocks_portable(
        portable.data(), input, block_count);
    const bool used = q3x::core::internal::sha256_transform_blocks_arm_sha2(
        accelerated.data(), input, block_count);
    expect(used == available,
           "ARM SHA2 dispatch agrees with availability at block count " +
               std::to_string(block_count));
    if (available) {
      expect(accelerated == portable,
             "ARM SHA2 equals portable compression at block count " +
                 std::to_string(block_count));
    } else {
      expect(accelerated == original,
             "unavailable ARM SHA2 leaves state unchanged");
    }
  }

  std::cout << "SHA-256 ARM SHA2: compiled=" << (compiled ? "yes" : "no")
            << " runtime_available=" << (available ? "yes" : "no") << '\n';
}

}  // namespace

int main() {
  test_known_vectors();
  test_boundary_lengths();
  test_incremental_updates();
  test_portable_public_equivalence();
  test_arm_sha2_compression_equivalence();

  if (failures != 0) {
    std::cerr << failures << " SHA-256 test(s) failed\n";
    return 1;
  }
  std::cout << "SHA-256 tests passed\n";
  return 0;
}
