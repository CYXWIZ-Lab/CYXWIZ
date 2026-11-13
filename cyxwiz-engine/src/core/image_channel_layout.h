#pragma once

// What an image's channels are (TOFIX209): one name each ("R", "G", "B",
// "grey", "edges", "channel 5"), and how many leading channels are still the
// colour picture (3 or 0). An image that is neither grey nor RGB is shown as
// that picture and one grey tile per further channel (image_sheet.h).

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct ImageChannelLayout {
    std::vector<std::string> names;
    size_t colour = 0;
    bool operator==(const ImageChannelLayout&) const = default;
};

// A decoded image: RGB, grey, or numbered channels.
inline ImageChannelLayout DecodedChannelLayout(size_t channels) {
    if (channels == 3) return {{"R", "G", "B"}, 3};
    if (channels == 1) return {{"grey"}, 0};
    ImageChannelLayout layout;
    for (size_t c = 0; c < channels; ++c) layout.names.push_back("channel " + std::to_string(c + 1));
    return layout;
}

// "R, G, B, edges".
inline std::string JoinedChannelNames(const std::vector<std::string>& names) {
    std::string text;
    for (const auto& name : names) text += (text.empty() ? "" : ", ") + name;
    return text;
}

}  // namespace cyxwiz
