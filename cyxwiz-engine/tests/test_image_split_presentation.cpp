// Data Studio > ML readiness for an image dataset (TOFIX178 1.5): the
// three parts the compile counted, each class's share in them, and the
// balance verdict.
#include "../src/core/image_split_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

const std::string kDot = "\xC2\xB7";

cyxwiz::image_split::Input CatsAndDogs() {
    cyxwiz::image_split::Input in;
    in.dataset = "cat_dog";
    auto& split = in.split;
    split.known = true;
    split.class_names = split.shown_names = {"cat", "dog"};
    split.images = 25000;
    split.parts = {20000, 2500, 2500};
    split.per_class = {{10000, 1250, 1250}, {10000, 1250, 1250}};
    return in;
}

void CheckStratified() {
    const auto v = cyxwiz::image_split::Build(CatsAndDogs());
    Check(v.subtitle == "cat_dog " + kDot + " images " + kDot + " 80 / 10 / 10, stratified, seed 42",
          "subtitle: " + v.subtitle);
    Check(v.cards.size() == 3 && v.cards[0].name == "Train" && v.cards[0].count == "20,000" &&
              v.cards[0].detail == "80 % " + kDot + " cat: 10,000 " + kDot + " dog: 10,000",
          "train card: " + v.cards[0].detail);
    Check(v.cards[2].count == "2,500", "test card");
    Check(v.classes.size() == 2 && v.classes[0].images == "12,500" &&
              v.classes[0].parts == std::vector<std::string>({"10,000 " + kDot + " 50 %", "1,250 " + kDot + " 50 %",
                                                              "1,250 " + kDot + " 50 %"}),
          "each class in each part");
    Check(v.balanced && v.balance == "Every part has every class in the same share as the whole set.", v.balance);
    Check(v.counted.find("none") != std::string::npos, "no images left out: " + v.counted);
}

void CheckUnbalanced() {
    auto in = CatsAndDogs();
    in.stratified = false;
    in.split.per_class = {{10100, 1150, 1250}, {9900, 1350, 1250}};
    const auto v = cyxwiz::image_split::Build(in);
    Check(!v.balanced, "a part 4 points off is not balanced");
    Check(v.balance == "The parts' class shares differ: cat is 46 % of validation but 50 % of the set. "
                       "Stratified on the Data Split keeps them alike.",
          v.balance);
    Check(v.subtitle.find("one shuffle") != std::string::npos, "one shuffle named");
}

void CheckWired() {
    auto in = CatsAndDogs();
    in.wired_test = in.split;
    in.wired_test.parts = {0, 0, 400};
    in.wired_test.per_class = {{0, 0, 200}, {0, 0, 200}};
    in.wired_test_name = "holdout";
    const auto v = cyxwiz::image_split::Build(in);
    Check(v.cards[2].count == "400" && v.cards[2].detail == "all of 'holdout'", "wired test card");
    Check(v.classes[1].parts[2] == "200 " + kDot + " 50 %", "the wired part's classes");
}

}  // namespace

int main() {
    CheckStratified();
    CheckUnbalanced();
    CheckWired();
    std::cout << "Image split presentation passed\n";
    return 0;
}
