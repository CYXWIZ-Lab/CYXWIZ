#include "image_split_presentation.h"

#include <array>
#include <cmath>
#include <cstdio>

namespace cyxwiz::image_split {

namespace {

std::string Thousands(size_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) digits.insert(static_cast<size_t>(at), ",");
    return digits;
}

std::string Percent(double share) {
    return std::to_string(static_cast<int>(std::lround(share * 100.0))) + " %";
}

std::string Ratio(float value) { return std::to_string(static_cast<int>(std::lround(value * 100.0f))); }

}  // namespace

View Build(const Input& in) {
    View v;
    const auto& split = in.split;
    v.subtitle = in.dataset + " \xC2\xB7 images \xC2\xB7 " + Ratio(in.train) + " / " + Ratio(in.val) + " / " +
                 Ratio(in.test) + ", " + (in.stratified ? "stratified" : "one shuffle") + ", seed " +
                 std::to_string(in.seed);

    // Each part: the training images' share, or a wired dataset whole.
    const char* names[3] = {"Train", "Validation", "Test"};
    const char* lower[3] = {"train", "validation", "test"};
    const ImageSplitCounts* wired[3] = {nullptr, in.wired_val.known ? &in.wired_val : nullptr,
                                        in.wired_test.known ? &in.wired_test : nullptr};
    const std::string wired_names[3] = {"", in.wired_val_name, in.wired_test_name};
    // The shares the split uses: the ratios over their sum (0.7 / 0.2 / 0.2
    // makes 64 / 18 / 18 %), as SplitSamples normalises them.
    const float sum = in.train + in.val + in.test;
    const float shares[3] = {sum > 0.0f ? in.train / sum : 0.0f, sum > 0.0f ? in.val / sum : 0.0f,
                             sum > 0.0f ? in.test / sum : 0.0f};
    const size_t classes = split.shown_names.size();
    std::vector<std::array<size_t, 3>> by_class(classes, {0, 0, 0});
    size_t totals[3] = {0, 0, 0};
    for (size_t part = 0; part < 3; ++part) {
        const ImageSplitCounts& from = wired[part] ? *wired[part] : split;
        const size_t column = wired[part] ? 2 : part;
        totals[part] = from.parts[column];
        for (size_t c = 0; c < classes && c < from.per_class.size(); ++c) by_class[c][part] = from.per_class[c][column];
        PartCard card{names[part], Thousands(totals[part]), {}};
        if (wired[part]) {
            card.detail = "all of '" + wired_names[part] + "'";
        } else {
            card.detail = Percent(shares[part]);
            for (size_t c = 0; c < classes && classes <= 6; ++c)
                // "cat: 10,000"; the colon keeps a numbered class ("0: 10,000") readable.
                card.detail += " \xC2\xB7 " + split.shown_names[c] + ": " + Thousands(by_class[c][part]);
        }
        v.cards.push_back(std::move(card));
    }

    // Each class in each part, as a share of that part, against its share of the set.
    v.balanced = true;
    std::string first_off;
    for (size_t c = 0; c < classes; ++c) {
        const size_t in_set = c < split.per_class.size()
            ? split.per_class[c][0] + split.per_class[c][1] + split.per_class[c][2] : 0;
        const double set_share = split.images ? static_cast<double>(in_set) / static_cast<double>(split.images) : 0.0;
        ClassRow row{split.shown_names[c], Thousands(in_set), {}};
        for (size_t part = 0; part < 3; ++part) {
            const double share = totals[part] ? static_cast<double>(by_class[c][part]) / static_cast<double>(totals[part]) : 0.0;
            row.parts.push_back(totals[part] ? Thousands(by_class[c][part]) + " \xC2\xB7 " + Percent(share) : "-");
            if (totals[part] && std::abs(share - set_share) > 0.02) {
                v.balanced = false;
                if (first_off.empty())
                    first_off = split.shown_names[c] + " is " + Percent(share) + " of " +
                                lower[part] +
                                " but " + Percent(set_share) + " of the set";
            }
        }
        v.classes.push_back(std::move(row));
    }
    v.balance = v.balanced ? "Every part has every class in the same share as the whole set."
                           : "The parts' class shares differ: " + first_off + "." +
                                 (in.stratified ? "" : " Stratified on the Data Split keeps them alike.");

    v.counted = "The images training reads: unreadable files (" + Thousands(split.unreadable) +
                ") and images a Quality node leaves out (" +
                (split.left_out ? Thousands(split.left_out) : std::string("none")) +
                ") are taken out first, then the split is made, the same way training makes it. Counts follow the "
                "Data Split's settings as you change them.";
    return v;
}

}  // namespace cyxwiz::image_split
