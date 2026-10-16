// Editor palette byte order (TOFIX133 P0 item 5): ARGB hex as written in the
// palette sources becomes ImGui's ABGR, so Monokai pink stays pink.
#include "../src/core/argb_colour.h"

#include <cstdlib>
#include <iostream>

static_assert(cyxwiz::ArgbToAbgr(0xFFF92672u) == 0xFF7226F9u, "Monokai keyword pink");
static_assert(cyxwiz::ArgbToAbgr(0x80494440u) == 0x80404449u, "alpha kept");
static_assert(cyxwiz::ArgbToAbgr(cyxwiz::ArgbToAbgr(0x12345678u)) == 0x12345678u, "round trip");

int main() {
    // Red stays red: ImGui reads the low byte as red.
    const auto red = cyxwiz::ArgbToAbgr(0xFFFF0000u);
    if ((red & 0xFFu) != 0xFFu || ((red >> 16) & 0xFFu) != 0u) {
        std::cerr << "FAIL: red is not in the low byte\n";
        return 1;
    }
    std::cout << "argb colour: palette byte order. OK\n";
    return 0;
}
