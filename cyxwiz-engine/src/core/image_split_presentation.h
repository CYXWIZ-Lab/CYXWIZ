#pragma once

// Data Studio > ML readiness for an image dataset (TOFIX178 1.5, mockup
// board 2): the three parts the compile counted (ImageSplitCounts) and each
// class's share in them. Pure data, no ImGui.

#include "image_split_counts.h"

#include <string>
#include <vector>

namespace cyxwiz::image_split {

struct Input {
    ImageSplitCounts split;               // the training images' parts
    ImageSplitCounts wired_val, wired_test;  // a dataset wired as that part (counted whole), or not known
    std::string wired_val_name, wired_test_name;
    std::string dataset;                  // the training images' dataset
    float train = 0.8f, val = 0.1f, test = 0.1f;
    bool stratified = true;
    int seed = 42;
};

struct PartCard {
    std::string name;       // "Train"
    std::string count;      // "20,000"
    std::string detail;     // "80 % · cat: 10,000 · dog: 10,000", or "all of 'holdout'"
};

struct ClassRow {
    std::string name;
    std::string images;                   // the class's images in the training set
    std::vector<std::string> parts;       // "10,000 · 50 %" for train, validation, test
};

struct View {
    std::string subtitle;                 // "cat_dog · images · 80 / 10 / 10, stratified, seed 42"
    std::vector<PartCard> cards;
    std::vector<ClassRow> classes;
    std::string balance;                  // the verdict on the class shares
    bool balanced = false;
    std::string counted;                  // what was counted
};

View Build(const Input& in);

}  // namespace cyxwiz::image_split
