// Colours written as 0xAARRGGBB (how designers and theme files write them)
// to ImGui's ImU32 layout 0xAABBGGRR. TOFIX133 P0 item 5: the Monokai,
// Dracula, One Dark and GitHub editor palettes were written ARGB and drawn
// with red and blue swapped.
#pragma once

#include <cstdint>

namespace cyxwiz {

constexpr std::uint32_t ArgbToAbgr(std::uint32_t argb) {
    return (argb & 0xFF00FF00u) | ((argb & 0x00FF0000u) >> 16) | ((argb & 0x000000FFu) << 16);
}

}  // namespace cyxwiz
