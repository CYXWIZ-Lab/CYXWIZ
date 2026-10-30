#pragma once

// A count as screens and compile messages show it: "25,000". Header only, so
// a presentation model's standalone test needs no other source.

#include <cstdint>
#include <string>

namespace cyxwiz {

inline std::string CountText(uint64_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(static_cast<size_t>(at), ",");
    return digits;
}

}  // namespace cyxwiz
