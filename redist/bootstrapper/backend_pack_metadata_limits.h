#pragma once

#include <cstdint>

namespace cyxwiz::runtime {

inline constexpr std::uintmax_t kMaximumBackendPackMetadataBytes =
    16U * 1024U * 1024U;
inline constexpr std::uintmax_t kMaximumBackendPackTrustBytes =
    4U * 1024U * 1024U;

}  // namespace cyxwiz::runtime
