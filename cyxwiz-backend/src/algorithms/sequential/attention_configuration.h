#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace cyxwiz::attention_configuration_detail {

// Module APIs take size_t; attention layers take int. Reject before narrowing
// or allocating a layer, including values wrapped from negative caller input.
inline int CheckedAttentionDimension(size_t value, const char* field) {
    if (value == 0 || value > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::invalid_argument(std::string(field) + " must be in [1, INT_MAX]");
    }
    return static_cast<int>(value);
}

} // namespace cyxwiz::attention_configuration_detail
