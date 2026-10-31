// Data Studio > Prepare's image recipe (TOFIX178 1.6): training's image steps
// by node, in its order and groups, and which moves the compiler allows.
#include "../src/core/image_recipe_presentation.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace {

namespace ir = cyxwiz::image_recipe;
namespace img = cyxwiz::image;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

const std::string kTimes = " \xC3\x97 ";

// cnn_clahe's steps, plus a flip and a blur after it: Resize 72 exact (node 1),
// CLAHE 2 / 8 (2), Center Crop 64 (3), Horizontal Flip (4), Gaussian Blur 3,
// sigma 0.8 (5), Normalize 0.5 / 0.5 (6).
cyxwiz::TrainingConfiguration Graph() {
    cyxwiz::TrainingConfiguration c;
    c.image_preprocessing.resize_mode = cyxwiz::ResizeMode::Exact;
    c.image_preprocessing.target_width = 72;
    c.image_preprocessing.target_height = 72;
    cyxwiz::CvImageStep clahe;
    clahe.kind = cyxwiz::CvImageStepKind::Contrast;
    clahe.position = 0;
    clahe.node_id = 2;
    clahe.node_name = "Image Contrast (CLAHE)";
    c.image_cv_steps.push_back(clahe);
    img::ImageOp crop;
    crop.kind = img::ImageOpKind::CenterCrop;
    crop.height = crop.width = 64;
    img::ImageOp flip;
    flip.kind = img::ImageOpKind::HorizontalFlip;
    img::ImageOp blur;
    blur.kind = img::ImageOpKind::GaussianBlur;
    blur.kernel_size = 3;
    blur.sigma = 0.8f;
    c.image_augmentation.ops = {crop, flip, blur};
    c.image_op_node_ids = {3, 4, 5};
    c.preprocessing.has_normalization = true;
    c.preprocessing.norm_mean = 0.5f;
    c.preprocessing.norm_std = 0.5f;
    c.preprocessing.norm_node_id = 6;
    return c;
}

ir::View Build(const cyxwiz::TrainingConfiguration& c) {
    static const std::map<int, std::string> names = {{1, "Resize 64x64"}, {2, "Image Contrast (CLAHE)"},
                                                       {3, "Center Crop 64"}, {4, "Horizontal Flip"},
                                                       {5, "Blur"}, {6, "Normalize"}};
    ir::Input in;
    in.config = &c;
    in.resize_node_id = 1;
    in.dataset = "Cats_Dogs Data";
    in.name_of = [](int id) {
        const auto it = names.find(id);
        return it == names.end() ? std::string() : it->second;
    };
    return ir::Build(in);
}

void CheckRows() {
    const auto c = Graph();
    const auto v = Build(c);
    Check(v.known && v.rows.size() == 6, "six rows");
    const ir::Group groups[] = {ir::Group::Resize, ir::Group::Once,     ir::Group::Once,
                                ir::Group::PerBatch, ir::Group::PerBatch, ir::Group::Normalize};
    const int nodes[] = {1, 2, 3, 4, 5, 6};
    for (size_t i = 0; i < 6; ++i)
        Check(v.rows[i].group == groups[i] && v.rows[i].node_id == nodes[i], "row " + std::to_string(i) + ": " + v.rows[i].name);
    Check(v.rows[0].what == "Resize 72" + kTimes + "72, exact" && v.rows[0].output_text == "72" + kTimes + "72" + kTimes + "3",
          "resize: " + v.rows[0].what);
    Check(v.rows[1].opencv && !v.rows[1].random && v.rows[1].ops_through == 0, "CLAHE is an OpenCV step before the ops");
    Check(v.rows[2].output_text == "64" + kTimes + "64" + kTimes + "3" && v.rows[2].ops_through == 1, "the crop's size");
    Check(v.rows[3].random, "the flip is random");
    // A fixed step after a random one runs per batch, as training splits it.
    Check(!v.rows[4].random && v.rows[4].group == ir::Group::PerBatch, "blur after the flip: per batch");
    Check(v.rows[5].what == "mean 0.5 \xC2\xB7 std 0.5", "normalize: " + v.rows[5].what);
    Check(v.once == 2 && v.per_batch == 2, "two once, two per batch");
    Check(v.subtitle == "Cats_Dogs Data to model input [64, 64, 3] \xC2\xB7 2 steps once, 2 per batch",
          "subtitle: " + v.subtitle);
}

void CheckMoves() {
    const auto c = Graph();
    const auto v = Build(c);
    Check(!ir::MoveRefusal(v, 0, 3).empty(), "Resize stays first");
    Check(!ir::MoveRefusal(v, 5, 1).empty(), "Normalize stays last");
    Check(ir::MoveRefusal(v, 4, 2).empty(), "the blur may move before the flip");
    Check(ir::MoveRefusal(v, 4, -1).empty(), "the blur may move first (it is not random)");
    const std::string clahe = ir::MoveRefusal(v, 1, 3);
    Check(clahe.find("must come before the first random step") != std::string::npos, "CLAHE after the flip: " + clahe);
    // Moving the flip first puts CLAHE after a random step.
    Check(!ir::MoveRefusal(v, 3, 0).empty(), "the flip before CLAHE is refused");
    Check(!ir::MoveRefusal(v, 2, 5).empty(), "nothing after Normalize");
    // Dropping a step after itself or after the row before it changes nothing.
    Check(ir::MoveChangesNothing(v, 4, 3) && ir::MoveChangesNothing(v, 4, 4), "the blur after the flip or itself: no change");
    Check(ir::MoveChangesNothing(v, 1, 0) && ir::MoveChangesNothing(v, 1, -1), "CLAHE after the Resize: no change");
    Check(!ir::MoveChangesNothing(v, 4, 2) && !ir::MoveChangesNothing(v, 2, -1), "real moves change the order");
}

void CheckPresetAndMix() {
    auto c = Graph();
    // A preset's two ops from one node are one row; a batch mix is a per-batch row.
    img::ImageOp vflip;
    vflip.kind = img::ImageOpKind::VerticalFlip;
    c.image_augmentation.ops.push_back(vflip);
    c.image_op_node_ids.push_back(5);
    c.image_augmentation.mix = img::BatchMix::MixUp;
    c.image_augmentation.mix_alpha = 0.2f;
    c.image_mix_node_id = 7;
    const auto v = Build(c);
    Check(v.rows.size() == 7 && v.rows[4].node_id == 5 && v.rows[4].ops_through == 4 && v.rows[4].random,
          "the blur and the vertical flip share their node's row");
    Check(v.rows[5].node_id == 7 && v.rows[5].group == ir::Group::PerBatch && v.rows[5].mix &&
              v.rows[5].what.rfind("MixUp (alpha 0.2", 0) == 0,
          "the batch mix: " + v.rows[5].what);
    // Training mixes the batch after every augmentation: the mix row stays last.
    Check(!ir::CanMove(v.rows[5]) && ir::CanMove(v.rows[4]) && !ir::CanMove(v.rows[0]) && !ir::CanMove(v.rows[6]),
          "only image steps move, not the mix");
    Check(!ir::MoveRefusal(v, 5, 2).empty(), "the mix does not move");
    Check(ir::MoveRefusal(v, 3, 5).find("come before") != std::string::npos, "nothing after the mix");
    Check(ir::MoveRefusal(v, 3, 4).empty(), "a flip may still go last of the augmentations");
}

void CheckUnsizedStep() {
    // A crop larger than the resized images: the reason, not the rows.
    auto c = Graph();
    c.image_augmentation.ops[0].height = 100;
    const auto v = Build(c);
    Check(!v.known && v.rows.empty() && v.reason.find("larger") != std::string::npos, "unsized step: " + v.reason);
}

void CheckNotReady() {
    ir::Input in;
    Check(!ir::Build(in).known, "no compile yet");
    cyxwiz::TrainingConfiguration none;
    in.config = &none;
    Check(ir::Build(in).reason.find("Resize") != std::string::npos, "no Resize: says so");
}

}  // namespace

int main() {
    CheckRows();
    CheckMoves();
    CheckPresetAndMix();
    CheckUnsizedStep();
    CheckNotReady();
    std::cout << "Image recipe presentation passed\n";
    return 0;
}
