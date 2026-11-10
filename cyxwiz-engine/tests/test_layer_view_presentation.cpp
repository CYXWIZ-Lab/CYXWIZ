// The Studio Debugger's layer view model (TOFIX178 6.2): views picked by the
// tensors' roles and shapes. Main scenario: the catdog CNN's capture (Conv 32
// on a 64 x 64 RGB image, BatchNorm, ReLU, Flatten, Dense 2), as Local Debug
// saved it on 2026-11-06. The capture control's settings (6.2d) use the
// ResNet example's compiled sizes from the approved Capture mockup.

#include "core/layer_capture_settings.h"
#include "core/layer_view_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <string>
#include <vector>

using namespace cyxwiz;
using namespace cyxwiz::layer_view;

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void CheckText(const std::string& got, const std::string& want, const std::string& what) {
    if (got != want) {
        std::printf("FAIL: %s\n  got:  %s\n  want: %s\n", what.c_str(), got.c_str(), want.c_str());
        ++g_failures;
    }
}

using R = CapturedTensorRole;

LayerInfo Conv32() {
    return {0, 6, "Conv 32", "Conv2D",
            {{R::Input, "input", {64, 64, 3, 1}},
             {R::Output, "output", {64, 64, 32, 1}},
             {R::Parameter, "bias", {32}},
             {R::Parameter, "weights", {3, 3, 3, 32}},
             {R::Gradient, "bias", {32}},
             {R::Gradient, "weights", {3, 3, 3, 32}}}};
}

LayerInfo Conv64() {
    return {4, 10, "Conv 64", "Conv2D",
            {{R::Input, "input", {32, 32, 32, 1}},
             {R::Output, "output", {32, 32, 64, 1}},
             {R::Parameter, "bias", {64}},
             {R::Parameter, "weights", {3, 3, 32, 64}},
             {R::Gradient, "weights", {3, 3, 32, 64}}}};
}

LayerInfo BatchNorm1() {
    return {1, 7, "BatchNorm 1", "BatchNorm",
            {{R::Input, "input", {64, 64, 32, 1}},
             {R::Output, "output", {64, 64, 32, 1}},
             {R::Parameter, "beta", {32}},
             {R::Parameter, "gamma", {32}},
             {R::State, "running_mean", {32}},
             {R::State, "running_var", {32}},
             {R::Gradient, "beta", {32}},
             {R::Gradient, "gamma", {32}}}};
}

LayerInfo Flatten() {
    return {12, 18, "Flatten", "Flatten",
            {{R::Input, "input", {8, 8, 128, 1}}, {R::Output, "output", {1, 8192}}}};
}

LayerInfo Dense2() {
    return {14, 20, "Dense 2", "Dense",
            {{R::Input, "input", {1, 8192}},
             {R::Output, "output", {1, 2}},
             {R::Parameter, "bias", {2}},
             {R::Parameter, "weight", {2, 8192}},
             {R::Gradient, "weight", {2, 8192}}}};
}

std::vector<std::string> Labels(const LayerViews& v) {
    std::vector<std::string> out;
    for (const auto& t : v.tabs) out.push_back(t.label);
    return out;
}

std::vector<std::string> Later(const LayerViews& v) {
    std::vector<std::string> out;
    for (const auto& t : v.later) out.push_back(t.label);
    return out;
}

void TestRowsAndShapes() {
    CheckText(ShapeText({64, 64, 32}), "64 \xC3\x97 64 \xC3\x97 32", "shape text");
    CheckText(ShapeText({}), "-", "empty shape text");
    Check(PerSampleShape({64, 64, 32, 8}) == std::vector<size_t>({64, 64, 32}), "spatial per-sample shape");
    Check(PerSampleShape({8, 8192}) == std::vector<size_t>({8192}), "flat per-sample shape");
    const auto row = BuildLayerRow(Conv32());
    CheckText(row.name, "Conv 32", "row name");
    CheckText(row.kind, "Conv2D 3 \xC3\x97 3", "row kind with the kernel");
    CheckText(BuildLayerRow(Dense2()).kind, "Dense, 16,386 parameters", "Dense row with its parameters");
    CheckText(BuildLayerRow(BatchNorm1()).kind, "BatchNorm", "no detail without a kernel, stride or matrix");
    LayerInfo pool{3, 9, "Pool 1", "MaxPool2D",
                   {{R::Input, "input", {64, 64, 32, 1}}, {R::Output, "output", {32, 32, 32, 1}}}};
    CheckText(BuildLayerRow(pool).kind, "MaxPool2D, stride 2", "stride from the sizes");
    LayerInfo strided = Conv64();
    strided.tensors[0].shape = {64, 64, 32, 1};
    CheckText(BuildLayerRow(strided).kind, "Conv2D 3 \xC3\x97 3, stride 2", "kernel and stride");
    strided.kind.clear();
    CheckText(BuildLayerRow(strided).kind, "3 \xC3\x97 3, stride 2", "no kind: the detail alone");
    CheckText(row.output, "64 \xC3\x97 64 \xC3\x97 32", "row output per sample");
    CheckText(BuildLayerRow(Dense2()).output, "2", "Dense row output");
    LayerInfo unnamed;
    unnamed.module_index = 3;
    CheckText(BuildLayerRow(unnamed).name, "Layer 3", "unnamed layer row");
    CheckText(BuildLayerRow(unnamed).output, "-", "layer without output");
}

void TestViewsFollowShapes() {
    DebugTensorNumericSummary relu;
    relu.available = true;
    relu.element_count = 131072;
    relu.finite_count = 131072;
    relu.zero_count = 55837;  // 42.6 %
    const auto first = PlanLayerViews(Conv32(), &relu, true, true);
    Check(Labels(first) == std::vector<std::string>({"Feature maps", "Values", "Weights", "Gradients", "Filters",
                                                     "Where it looks", "Top images"}),
          "first conv tabs");
    Check(first.tabs[2].tensor == "weights" && first.tabs[3].tensor == "weights" && first.tabs[4].tensor == "weights",
          "weights, gradients and filters read the conv weights, not the bias");
    Check(Later(first).empty(), "first conv: nothing left for later");
    const auto dense_first = Labels(PlanLayerViews(Dense2(), nullptr, false, true, false));
    Check(dense_first[dense_first.size() - 3] == "Where it looks" && dense_first.back() == "Top samples",
          "a Dense layer taking the model's features explains them too; its top items are samples");
    const auto conv64 = Labels(PlanLayerViews(Conv64(), nullptr, false, false));
    Check(std::find(conv64.begin(), conv64.end(), "Where it looks") == conv64.end(),
          "only the layer the input enters explains it");
    Check(!TopImagesHint().empty(), "Top images has its Inspector note");
    // A sample as a picture: a 1 x 2 RGB image stretched to 0..255.
    const auto picture = SamplePicture({0.0f, 0.5f, 1.0f, 1.0f, 0.5f, 0.0f}, {1, 2, 3});
    Check(picture && picture->width == 2 && picture->height == 1 && picture->rgba[0] == 0 && picture->rgba[2] == 255 &&
              picture->rgba[3] == 255 && picture->rgba[4] == 255 && picture->rgba[6] == 0,
          "a sample as an RGBA picture");
    Check(!SamplePicture({1, 2, 3, 4}, {2, 2}), "a flat sample is no picture");
    CheckText(first.facts,
              "weights [3, 3, 3, 32] + bias [32], 896 parameters \xC2\xB7 output 64 \xC3\x97 64 \xC3\x97 32, "
              "42.6 % zeros",
              "first conv facts");

    DebugTensorNumericSummary conv_out = relu;
    conv_out.zero_count = 0;
    CheckText(PlanLayerViews(Conv32(), &conv_out, true).facts,
              "weights [3, 3, 3, 32] + bias [32], 896 parameters \xC2\xB7 output 64 \xC3\x97 64 \xC3\x97 32",
              "no zeros in the facts when nothing is zero");

    const auto deeper = PlanLayerViews(Conv64(), nullptr, false);
    Check(Later(deeper).empty() && Labels(deeper).size() == 5 && Labels(deeper).back() == "Top images",
          "a deeper conv has no Filters or occlusion, and has Top images");
    CheckText(deeper.facts,
              "weights [3, 3, 32, 64] + bias [64], 18,496 parameters \xC2\xB7 output 32 \xC3\x97 32 \xC3\x97 64",
              "deeper conv facts (no numerics, no zeros)");

    const auto bn = PlanLayerViews(BatchNorm1(), nullptr, false);
    Check(Labels(bn) == std::vector<std::string>({"Feature maps", "Values", "Weights", "Gradients", "Top images"}),
          "BatchNorm tabs");
    Check(bn.tabs[2].tensor == "beta" || bn.tabs[2].tensor == "gamma", "BatchNorm weights are a 1-D parameter");
    Check(bn.facts.find("running_mean") == std::string::npos, "kept state is not a parameter");

    const auto flatten = PlanLayerViews(Flatten(), nullptr, false);
    Check(Labels(flatten) == std::vector<std::string>({"Units", "Values", "Top images"}), "flat output tabs");
    Check(Later(flatten) == std::vector<std::string>({"Embedding"}), "flat output can be projected later");
    CheckText(flatten.facts, "output 8192", "Flatten facts");

    const auto dense = PlanLayerViews(Dense2(), nullptr, false);
    Check(Labels(dense) ==
              std::vector<std::string>({"Units", "Values", "Weights", "Gradients", "Contributions", "Top images"}),
          "Dense tabs");
    Check(Later(dense) == std::vector<std::string>({"Embedding"}), "Dense later tabs");
    for (const auto& t : dense.later) Check(!t.why.empty(), "a later tab says why: " + t.label);
    Check(!dense.hint.empty(), "a weights layer has an inspector hint");

    const LayerInfo empty;
    Check(PlanLayerViews(empty, nullptr, false).tabs.empty(), "nothing captured, no tabs");
}

// 6.3a: two 2 x 2 RGB filters from weights [2, 2, 3, 2] in row-major order.
void TestFilterPictures() {
    std::vector<float> w(2 * 2 * 3 * 2);
    // Filter 0: red rises with x (0 then 1), green and blue 0; filter 1: all 0.25.
    for (size_t y = 0; y < 2; ++y)
        for (size_t x = 0; x < 2; ++x)
            for (size_t c = 0; c < 3; ++c) {
                const size_t k = (y * 2 + x) * 3 + c;
                w[k * 2 + 0] = c == 0 ? static_cast<float>(x) : 0.0f;
                w[k * 2 + 1] = 0.25f;
            }
    const auto pictures = BuildFilterPictures(w, {2, 2, 3, 2});
    Check(pictures && pictures->height == 2 && pictures->width == 2 && pictures->channels == 3 &&
              pictures->filters.size() == 2,
          "two 2 x 2 RGB pictures");
    if (pictures) {
        const auto& red = pictures->filters[0].pixels;
        Check(red[0] == 0.0f && red[3] == 1.0f && red[1] == 0.0f && red[9] == 1.0f,
              "filter 0: red 0 on the left, 1 on the right; green 0");
        Check(std::fabs(pictures->filters[0].norm - std::sqrt(2.0f)) < 1e-6f, "filter 0 norm sqrt(2)");
        Check(pictures->filters[1].pixels[5] == 0.5f && std::fabs(pictures->filters[1].norm - 0.25f * std::sqrt(12.0f)) < 1e-6f,
              "a constant filter is mid grey, with its norm");
        CheckText(FiltersText(*pictures),
                  "The 2 learned 2 \xC3\x97 2 RGB filters of the first layer, each scaled to its own range. Colour edges "
                  "and blobs are what the first layer looks for.",
                  "filters text");
    }
    Check(!FiltersHint().empty(), "the Filters view has its own Inspector note");
    Check(!BuildFilterPictures(w, {2, 2, 2, 3}), "2 input channels are not a picture");
    Check(!BuildFilterPictures({1.0f, 2.0f}, {2, 2, 3, 2}), "values that do not fill the shape");
    const auto grey = BuildFilterPictures(std::vector<float>(9 * 4, 1.0f), {3, 3, 1, 4});
    Check(grey && grey->channels == 1 && grey->filters.size() == 4 &&
              FiltersText(*grey).find("4 learned 3 \xC3\x97 3 grey filters") != std::string::npos,
          "grey filters");
}

void TestMatrixLayout() {
    const TensorInfo conv{R::Parameter, "weights", {3, 3, 32, 64}};
    const auto layout = LayoutAsMatrix(conv);
    Check(layout.rows == 64 && layout.columns == 288, "conv weights as [cout, kh*kw*cin]");
    CheckText(layout.description,
              "64 output channels (rows) by 288 inputs (3 \xC3\x97 3 kernel \xC3\x97 32 input channels)",
              "conv layout words");
    // Element [ky=2, kx=1, ci=17, co=40]: row 40, column (2*3+1)*32+17 = 241.
    const size_t flat = ((2 * 3 + 1) * 32 + 17) * 64 + 40;
    Check(MatrixFlatIndex(conv, 40, 241) == flat, "conv cell to flat index");
    CheckText(ElementText(conv, flat, -0.0312f), "[2, 1, 17, 40] = -0.0312", "conv element text");

    const TensorInfo dense{R::Parameter, "weight", {2, 8192}};
    const auto d = LayoutAsMatrix(dense);
    Check(d.rows == 2 && d.columns == 8192, "Dense weights [out, in]");
    CheckText(d.description, "2 outputs (rows) by 8,192 inputs", "Dense layout words");
    Check(MatrixFlatIndex(dense, 1, 5) == 8197, "Dense cell to flat index");

    const TensorInfo bias{R::Parameter, "bias", {32}};
    Check(LayoutAsMatrix(bias).rows == 1 && LayoutAsMatrix(bias).columns == 32, "a vector is one row");
    const TensorInfo grad{R::Gradient, "weights", {3, 3, 3, 32}};
    Check(LayoutAsMatrix(grad).rows == 32 && LayoutAsMatrix(grad).columns == 27, "gradients lay out as weights");
    const TensorInfo act{R::Output, "output", {16, 16, 128, 8}};
    Check(LayoutAsMatrix(act).rows == 16 && LayoutAsMatrix(act).columns == 16 * 128 * 8, "activation rows");

    // One kernel position: output channels by input channels.
    Check(KernelPositions(conv) == 9 && KernelPositions(dense) == 0 &&
              KernelPositions({R::Parameter, "weights", {1, 1, 32, 64}}) == 0,
          "kernel positions only for a kernel above 1 x 1");
    const auto slice = LayoutAsMatrix(conv, size_t{7});
    Check(slice.rows == 64 && slice.columns == 32, "a slice is [cout, cin]");
    CheckText(slice.description,
              "64 output channels (rows) by 32 input channels at kernel position (2, 1) of the 3 \xC3\x97 3 kernel",
              "slice words");
    Check(MatrixFlatIndex(conv, 40, 17, size_t{7}) == flat, "slice cell to the same flat index");
    Check(LayoutAsMatrix(dense, size_t{0}).columns == 8192, "no slice for Dense weights");
}

void TestSections() {
    std::vector<LayerInfo> layers = {Conv32(), BatchNorm1(), Conv64(), Flatten(), Dense2()};
    const std::map<int, std::string> groups = {{6, "Stem"}, {7, "Stem"}, {10, "Stage 1"}, {20, "Head"}};
    const auto sections = SectionLayers(layers, groups);
    Check(sections.size() == 4, "Stem, Stage 1, ungrouped, Head");
    if (sections.size() == 4) {
        CheckText(sections[0].name, "Stem", "first section");
        Check(sections[0].layers == std::vector<size_t>({0, 1}), "Stem layers");
        Check(sections[2].name.empty() && sections[2].layers == std::vector<size_t>({3}), "ungrouped Flatten");
        CheckText(sections[3].name, "Head", "last section");
    }
    const auto none = SectionLayers(layers, {});
    Check(none.size() == 1 && none[0].name.empty() && none[0].layers.size() == 5, "no groups: one section");
}

void TestPrediction() {
    const std::vector<std::string> catdog = {"cat", "dog"};
    // BCE with logits: logit -4.556 -> sigmoid 0.0104 -> cat, 99.0 %.
    const auto p = Predict({-4.556f}, OutputRule::Sigmoid, catdog);
    Check(p && p->class_index == 0, "sigmoid below 0.5 is class 0");
    if (p) {
        CheckText(p->text, "cat, 99.0 %", "sigmoid prediction text");
        CheckText(p->detail, "logit -4.556 \xE2\x86\x92 sigmoid 0.0104 (dog is 1)", "sigmoid detail");
    }
    const auto soft = Predict({0.1f, 2.3f, -1.0f}, OutputRule::Softmax, {"a", "b"});
    Check(soft && soft->class_index == 1, "softmax argmax");
    if (soft) CheckText(soft->text, "b, 87.1 %", "softmax share");
    const auto probs = Predict({0.2f, 0.7f, 0.1f}, OutputRule::Probabilities, {});
    Check(probs && probs->class_index == 1, "probabilities argmax");
    if (probs) CheckText(probs->text, "class 1, 70.0 %", "unknown class names");
    Check(Predict({0.83f}, OutputRule::Probability, catdog)->text == "dog, 83.0 %", "a sigmoid layer's probability");
    Check(Predict({3.25f}, OutputRule::Value, {})->text == "3.25", "regression value");
    Check(!Predict({1.0f, 2.0f}, OutputRule::Sigmoid, catdog), "sigmoid needs one output");
    Check(!Predict({NAN}, OutputRule::Sigmoid, catdog), "no prediction from NaN");
    Check(!Predict({1.0f}, OutputRule::Unknown, catdog), "unknown rule");
    CheckText(ClassName(1, {"0", "1"}), "class 1", "a numeric class name");
    CheckText(ClassName(0, catdog), "cat", "a class name");
    Check(OutputRuleFromKey(OutputRuleKey(OutputRule::Softmax)) == OutputRule::Softmax &&
              OutputRuleFromKey("") == OutputRule::Unknown,
          "rule keys round-trip");

    Check(ItemLabels({0, 1, 1, 0, 0, 1}, {3, 2}) == std::vector<int>({1, 0, 1}), "one-hot labels");
    Check(ItemLabels({1, 0, 3}, {3}) == std::vector<int>({1, 0, 3}), "scalar labels");
    Check(ItemLabels({0.5f}, {1, 1}) == std::vector<int>({-1}), "a value is not a class");

    CheckText(SampleHeadline("cat.10599.jpg", "cat", p), "cat.10599.jpg \xC2\xB7 label cat \xC2\xB7 predicted cat, 99.0 %",
              "headline");
    CheckText(SampleHeadline("", "", std::nullopt), "", "nothing known");
    CheckText(SampleHeadline("", "", p), "predicted cat, 99.0 %", "prediction alone");
}

void TestChannelsAndScales() {
    const std::vector<float> means = {0.2f, 0.9f, NAN, 0.9f, 0.5f};
    using core::ChannelSort;
    const std::vector<float> maxima = {5.0f, 1.0f, 2.0f, 3.0f, 5.0f};
    Check(OrderChannels(means, maxima, ChannelSort::Strongest) == std::vector<size_t>({1, 3, 4, 0, 2}),
          "strongest first, ties by number, NaN last");
    Check(OrderChannels(means, maxima, ChannelSort::Peak) == std::vector<size_t>({0, 4, 3, 2, 1}), "highest peak first");
    Check(OrderChannels(means, maxima, ChannelSort::Number) == std::vector<size_t>({0, 1, 2, 3, 4}), "by number");
    CheckText(ChannelsShownText(16, 32, false, ChannelSort::Strongest),
              "The 16 strongest of 32 channels for this sample, strongest first.", "16 strongest");
    CheckText(ChannelsShownText(16, 64, false, ChannelSort::Peak),
              "The first 16 of 64 channels for this sample, highest peak first.", "first 16 by peak");
    CheckText(ChannelsShownText(32, 32, true, ChannelSort::Number),
              "All 32 of 32 channels for this sample, in channel order.", "all by number");
    Check(CountDeadChannels({0.0f, 1.2f, 0.0f, 3.0f}) == 2, "dead channels");
    Check(AsSpatial({64, 64, 32, 1}).has_value() && AsSpatial({64, 64, 32, 1})->channels == 32, "spatial shape");
    Check(!AsSpatial({1, 1, 256, 8}).has_value(), "a 1 x 1 map is not a picture");
    Check(!AsSpatial({8, 8192}).has_value(), "a flat activation is not spatial");

    const auto seq = ToSequential({-1.0f, 0.0f, 3.405f, 6.81f, 9.0f, NAN}, 6.81f);
    Check(seq[0] == 0.0f && seq[1] == 0.0f && std::abs(seq[2] - 0.5f) < 1e-6f && seq[3] == 1.0f && seq[4] == 1.0f &&
              seq[5] == 0.0f,
          "sequential scale clamps and drops NaN");
    Check(ToSequential({1.0f}, 0.0f)[0] == 0.0f, "an all-zero map stays dark");
    const auto div = ToDiverging({-0.5f, 0.0f, 0.25f, 2.0f}, 0.5f);
    Check(div[0] == -1.0f && div[1] == 0.0f && div[2] == 0.5f && div[3] == 1.0f, "diverging scale");
}

void TestHistogramAndNumbers() {
    // 1000 values over [0, 10] in 10 bins: 995 in the first bins, 5 in the last.
    std::vector<unsigned> counts = {600, 200, 100, 50, 30, 10, 5, 0, 0, 5};
    Check(std::abs(HistogramQuantile(counts, 0.0f, 10.0f, 0.995) - 7.0f) < 1e-5f, "99.5th percentile bin top");
    Check(std::abs(HistogramQuantile(counts, 0.0f, 10.0f, 1.0) - 10.0f) < 1e-5f, "the maximum");
    Check(HistogramQuantile({}, 0.0f, 4.0f, 0.5) == 4.0f, "empty histogram gives the top");
    const auto log_bars = HistogramBars({1000, 10, 0}, true);
    Check(log_bars[0] == 1.0f && log_bars[1] > 0.3f && log_bars[1] < 0.4f && log_bars[2] == 0.0f, "log bars");
    const auto bars = HistogramBars({4, 2, 0}, false);
    Check(bars[0] == 1.0f && bars[1] == 0.5f && bars[2] == 0.0f, "linear bars");

    DebugTensorNumericSummary n;
    n.available = true;
    n.element_count = 131072;
    n.finite_count = 131072;
    n.zero_count = 55837;
    n.minimum = 0.0;
    n.maximum = 6.81;
    n.mean = 0.2861;
    const auto lines = NumbersLines(n, true);
    Check(lines.size() == 3, "three number lines");
    CheckText(lines[0], "min 0 \xC2\xB7 max 6.81 \xC2\xB7 mean 0.2861", "range line");
    CheckText(lines[1], "42.6 % zeros", "zeros line");
    CheckText(lines[2], "131,072 values, no NaN or Inf", "count line");
    n.nan_count = 3;
    n.inf_count = 1;
    CheckText(NumbersLines(n, false).back(), "131,072 values, 3 NaN, 1 Inf", "NaN and Inf counted");
    n.zero_count = 0;
    Check(NumbersLines(n, true).size() == 2, "no zeros line when nothing is zero");
    n.zero_count = 55837;
    CheckText(ElementText({R::Parameter, "w", {2}}, 1, -0.0f), "[1] = 0", "negative zero reads 0");
    Check(NumbersLines(n, false).size() == 2, "weights skip the zeros line");
    DebugTensorNumericSummary none;
    CheckText(NumbersLines(none, true).front(), "No summary: the tensor is not Float32.", "no summary");
}

void TestCaptureSettings() {
    namespace lc = cyxwiz::layer_capture;
    using core::LayerCaptureWhen;
    core::StudioDebuggerConfig s;
    Check(!lc::CapturesAnything(s, false) && !lc::CapturesAnything(s, true), "off captures nothing");
    CheckText(lc::Summary(s), "off", "summary off");
    s.capture_when = LayerCaptureWhen::LocalDebug;
    Check(lc::CapturesAnything(s, false) && !lc::CapturesAnything(s, true), "Local Debug only");
    CheckText(lc::Summary(s), "Local Debug runs", "summary Local Debug");
    s.capture_when = LayerCaptureWhen::Training;
    Check(lc::CapturesAnything(s, false) && lc::CapturesAnything(s, true), "training covers Local Debug too");
    CheckText(lc::Summary(s), "every 5 epochs", "summary training");
    s.capture_every_epochs = 1;
    CheckText(lc::Summary(s), "every epoch", "summary every epoch");
    s.capture_activations = s.capture_parameters = s.capture_gradients = false;
    Check(!lc::CapturesAnything(s, false) && !lc::CapturesAnything(s, true), "nothing ticked captures nothing");
    CheckText(lc::Summary(s), "off", "summary nothing ticked");

    Check(lc::SnapshotEpochs(5, 20) == std::vector<int>{5, 10, 15, 20}, "every 5 of 20");
    Check(lc::SnapshotEpochs(3, 20) == std::vector<int>{3, 6, 9, 12, 15, 18, 20}, "every 3 of 20 and the last");
    Check(lc::SnapshotEpochs(30, 20) == std::vector<int>{20}, "longer than the run: the last");
    Check(lc::SnapshotEpochs(0, 2) == std::vector<int>{1, 2}, "every 0 reads as 1");
    Check(lc::SnapshotEpochs(5, 0).empty(), "no epochs");

    ModelMemoryEstimate model;
    Check(!lc::EstimateSize(model, core::StudioDebuggerConfig{}, 1u << 28).available, "no model, no size");
    model.available = true;
    model.output_elements_per_sample = 2355713;  // the input and all 41 layers' outputs
    model.activation_elements_per_sample = 9999999;  // with the caches: not what a capture keeps
    model.trained_elements = 1228097;
    model.state_elements = 2880;
    core::StudioDebuggerConfig all;
    all.capture_when = LayerCaptureWhen::Training;
    auto size = lc::EstimateSize(model, all, uint64_t{256} << 20);
    Check(size.activation_values == 2355713 && size.activation_bytes == 9422852, "one sample's activations");
    Check(size.parameter_bytes == 4923908 && size.gradient_bytes == 4912388, "parameters with kept state, gradients");
    Check(size.snapshot_bytes == 19259148 && !size.over_budget, "one snapshot");
    all.capture_batch = true;
    size = lc::EstimateSize(model, all, uint64_t{256} << 20);
    Check(size.activation_values == 2355713ull * 8 && size.snapshot_bytes == 85219112, "a batch of 8");
    Check(lc::CaptureSamples(all) == 8, "batch samples");
    all.capture_activations = false;
    size = lc::EstimateSize(model, all, uint64_t{8} << 20);
    Check(size.snapshot_bytes == 9836296 && size.over_budget, "unticked activations; over a small budget");
}

}  // namespace

int main() {
    TestRowsAndShapes();
    TestViewsFollowShapes();
    TestMatrixLayout();
    TestFilterPictures();
    TestChannelsAndScales();
    TestHistogramAndNumbers();
    TestCaptureSettings();
    TestSections();
    TestPrediction();
    if (g_failures) {
        std::printf("%d layer view checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("layer view presentation passed\n");
    return EXIT_SUCCESS;
}
