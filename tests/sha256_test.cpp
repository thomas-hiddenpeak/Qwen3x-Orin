#include "q3x/core/sha256.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect_hash(const std::string& input, const std::string& expected,
                 const char* label) {
  const std::string actual = q3x::core::sha256(input).hex();
  if (actual != expected) {
    ++failures;
    std::cerr << "FAIL: " << label << ": expected " << expected << ", got "
              << actual << '\n';
  }
}

void test_incremental() {
  q3x::core::Sha256 hasher;
  const std::string first = "abcdbcdecdefdefgefghfghighij";
  const std::string second = "hijkijkljklmklmnlmnomnopnopq";
  if (!hasher.update(first.data(), first.size()) ||
      !hasher.update(second.data(), second.size())) {
    ++failures;
    std::cerr << "FAIL: incremental update rejected valid input\n";
    return;
  }
  const std::string expected =
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
  if (hasher.finalize().hex() != expected || hasher.finalize().hex() != expected) {
    ++failures;
    std::cerr << "FAIL: incremental or repeated finalize\n";
  }
  const char extra = 'x';
  if (hasher.update(&extra, 1)) {
    ++failures;
    std::cerr << "FAIL: update after finalize was accepted\n";
  }
}

}  // namespace

int main() {
  expect_hash("",
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "empty string");
  expect_hash("abc",
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "abc");
  test_incremental();

  if (failures != 0) {
    std::cerr << failures << " SHA-256 test(s) failed\n";
    return 1;
  }
  std::cout << "SHA-256 tests passed\n";
  return 0;
}
