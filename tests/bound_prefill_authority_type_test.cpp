#include "reference_engine_prefill_authority.h"

#include <type_traits>

namespace detail = q3x::runtime::reference_engine_detail;

static_assert(!std::is_default_constructible_v<
              detail::BoundPrefillExecutionPlan>);
static_assert(!std::is_copy_constructible_v<
              detail::BoundPrefillExecutionPlan>);
static_assert(!std::is_move_constructible_v<
              detail::BoundPrefillExecutionPlan>);
static_assert(!std::is_aggregate_v<detail::BoundPrefillExecutionPlan>);

static_assert(!std::is_default_constructible_v<
              detail::BoundPrefillRequestReceipt>);
static_assert(!std::is_copy_constructible_v<
              detail::BoundPrefillRequestReceipt>);
static_assert(std::is_move_constructible_v<
              detail::BoundPrefillRequestReceipt>);
static_assert(!std::is_move_assignable_v<
              detail::BoundPrefillRequestReceipt>);
static_assert(!std::is_aggregate_v<detail::BoundPrefillRequestReceipt>);

using BindFunction = decltype(
    &detail::ReferenceEnginePrefillPlanFactory::bind);
static_assert(!std::is_invocable_v<
              BindFunction,
              const q3x::runtime::PrefillOperatorBindingSet&>);

int main() { return 0; }
