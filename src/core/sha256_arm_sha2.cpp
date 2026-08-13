#include "sha256_internal.h"

#if !defined(Q3X_SHA256_ARM_SHA2_COMPILED)
#error "sha256_arm_sha2.cpp requires the isolated ARM SHA2 build definition"
#endif

#include <arm_neon.h>

#if !defined(__ARM_FEATURE_SHA2)
#error "sha256_arm_sha2.cpp must be compiled with ARM SHA2 enabled"
#endif

namespace q3x::core::internal {
namespace {

inline uint32x4_t LoadMessage(const std::uint8_t* const input) noexcept {
  return vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(input)));
}

inline void Schedule(uint32x4_t& next, const uint32x4_t prior1,
                     const uint32x4_t prior2,
                     const uint32x4_t prior3) noexcept {
  next = vsha256su1q_u32(vsha256su0q_u32(next, prior1), prior2, prior3);
}

inline void Rounds(uint32x4_t& state_abcd, uint32x4_t& state_efgh,
                   const uint32x4_t message,
                   const std::size_t round) noexcept {
  const uint32x4_t round_constants =
      vld1q_u32(kSha256RoundConstants.data() + round);
  const uint32x4_t message_and_constants =
      vaddq_u32(message, round_constants);
  const uint32x4_t previous_abcd = state_abcd;
  state_abcd = vsha256hq_u32(state_abcd, state_efgh,
                            message_and_constants);
  state_efgh = vsha256h2q_u32(state_efgh, previous_abcd,
                             message_and_constants);
}

}  // namespace

void sha256_transform_blocks_arm_sha2_unchecked(
    std::uint32_t* const state, const std::uint8_t* blocks,
    const std::size_t block_count) noexcept {
  if (block_count == 0U) {
    return;
  }
  uint32x4_t state_abcd = vld1q_u32(state);
  uint32x4_t state_efgh = vld1q_u32(state + 4U);

  for (std::size_t block_index = 0; block_index < block_count;
       ++block_index, blocks += 64U) {
    const uint32x4_t previous_abcd = state_abcd;
    const uint32x4_t previous_efgh = state_efgh;

    uint32x4_t message0 = LoadMessage(blocks);
    uint32x4_t message1 = LoadMessage(blocks + 16U);
    uint32x4_t message2 = LoadMessage(blocks + 32U);
    uint32x4_t message3 = LoadMessage(blocks + 48U);

    Rounds(state_abcd, state_efgh, message0, 0U);
    Rounds(state_abcd, state_efgh, message1, 4U);
    Rounds(state_abcd, state_efgh, message2, 8U);
    Rounds(state_abcd, state_efgh, message3, 12U);

    for (std::size_t round = 16U; round < 64U; round += 16U) {
      Schedule(message0, message1, message2, message3);
      Rounds(state_abcd, state_efgh, message0, round);
      Schedule(message1, message2, message3, message0);
      Rounds(state_abcd, state_efgh, message1, round + 4U);
      Schedule(message2, message3, message0, message1);
      Rounds(state_abcd, state_efgh, message2, round + 8U);
      Schedule(message3, message0, message1, message2);
      Rounds(state_abcd, state_efgh, message3, round + 12U);
    }

    state_abcd = vaddq_u32(state_abcd, previous_abcd);
    state_efgh = vaddq_u32(state_efgh, previous_efgh);
  }

  vst1q_u32(state, state_abcd);
  vst1q_u32(state + 4U, state_efgh);
}

}  // namespace q3x::core::internal
