#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"

#include <type_traits>

#ifndef Q3X_EXPECT_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION
#error "The contract target must state the configured Oracle admission"
#endif

#if Q3X_EXPECT_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#error "Oracle admission did not propagate through the q3x_core interface"
#endif
static_assert(std::is_class_v<
              q3x::runtime::reference_engine_test_detail::
                  Sm87TargetAotLayer0M192OracleAccess>);
#else
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#error "Default-OFF consumer unexpectedly received the Oracle admission"
#endif
#endif

int main() { return 0; }
