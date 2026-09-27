#pragma once

#include <cstddef>
#include <string>

namespace q3x::server {

// Decode byte-level model output without changing token identity/accounting.
// Retain at most an incomplete code point between tokens. At end-of-output,
// replace incomplete/ill-formed subsequences with U+FFFD, identically for SSE
// and nonstream responses (Unicode maximal-subpart replacement).
inline std::string render_utf8_output(std::string& pending, const bool final) {
  std::string output;
  std::size_t i = 0;
  while (i < pending.size()) {
    const auto first = static_cast<unsigned char>(pending[i]);
    std::size_t count = first < 0x80 ? 1 :
        (first >= 0xc2 && first <= 0xdf ? 2 :
         (first >= 0xe0 && first <= 0xef ? 3 :
          (first >= 0xf0 && first <= 0xf4 ? 4 : 0)));
    if (count == 0) {
      output += "\xef\xbf\xbd";
      ++i;
      continue;
    }
    std::size_t j = 1;
    for (; j < count && i + j < pending.size(); ++j) {
      const auto byte = static_cast<unsigned char>(pending[i + j]);
      if (byte < 0x80 || byte > 0xbf ||
          (j == 1 && ((first == 0xe0 && byte < 0xa0) ||
                      (first == 0xed && byte >= 0xa0) ||
                      (first == 0xf0 && byte < 0x90) ||
                      (first == 0xf4 && byte >= 0x90)))) {
        break;
      }
    }
    if (j == count) {
      output.append(pending, i, count);
      i += count;
    } else if (i + j == pending.size() && !final) {
      break;
    } else {
      output += "\xef\xbf\xbd";
      i += j;
    }
  }
  pending.erase(0, i);
  return output;
}

}  // namespace q3x::server
