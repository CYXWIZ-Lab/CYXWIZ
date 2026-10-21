#pragma once

namespace cyxwiz::training_contract {

inline constexpr char kGradientAccumulationStepsKey[] = "grad_accum_steps";
inline constexpr char kGradientAccumulationStepsDefaultValue[] = "1";
inline constexpr char kGradientAccumulationStepsValidation[] = "1-100000";
inline constexpr int kGradientAccumulationStepsDefault = 1;
inline constexpr int kGradientAccumulationStepsMinimum = 1;
inline constexpr int kGradientAccumulationStepsMaximum = 100000;

constexpr int ClampGradientAccumulationSteps(int value) noexcept {
    return value < kGradientAccumulationStepsMinimum
        ? kGradientAccumulationStepsMinimum
        : (value > kGradientAccumulationStepsMaximum
               ? kGradientAccumulationStepsMaximum
               : value);
}

} // namespace cyxwiz::training_contract
