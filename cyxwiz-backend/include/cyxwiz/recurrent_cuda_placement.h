#pragma once

namespace cyxwiz {

// tofix67 slice 5: name of the recurrent staged ArrayFire execution plan.
// The LSTM / GRU ArrayFire recurrence materializes per timestep (input
// projection, recurrent projection, gate combination, state update), which
// bounds the lazy JIT graph and CUDA's generated-kernel parameter block.
// Placement explanations reference this name so evidence and support bundles
// can identify the exact plan; bump the version suffix when the staging
// boundaries change.
inline constexpr const char* RecurrentStagedArrayFirePlanName =
    "recurrent_timestep_materialization_v1";

} // namespace cyxwiz
