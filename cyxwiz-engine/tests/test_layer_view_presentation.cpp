// The Studio Debugger's layer view model (TOFIX178 6.2): views picked by the
// tensors' roles and shapes. Main scenario: the catdog CNN's capture (Conv 32
// on a 64 x 64 RGB image, BatchNorm, ReLU, Flatten, Dense 2), as Local Debug
// saved it on 2026-11-06.

#include "core/layer_view_presentation.h"

#include <cmath>
#include <cstdio>
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
    CheckText(row.kind, "Conv2D", "row kind");
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
    const auto first = PlanLayerViews(Conv32(), &relu, true);
    Check(Labels(first) == std::vector<std::string>({"Feature maps", "Values", "Weights", "Gradients"}),
          "first conv tabs");
    Check(first.tabs[2].tensor == "weights" && first.tabs[3].tensor == "weights",
          "weights and gradients read the conv weights, not the bias");
    Check(Later(first) == std::vector<std::string>({"Where it looks", "Filters"}), "first conv later tabs");
    CheckText(first.facts,
              "weights [3, 3, 3, 32] + bias [32], 896 parameters \xC2\xB7 output 64 \xC3\x97 64 \xC3\x97 32, "
              "42.6 % zeros",
              "first conv facts");

    const auto deeper = PlanLayerViews(Conv64(), nullptr, false);
    Check(Later(deeper).empty(), "a deeper conv has no Filters or occlusion");
    CheckText(deeper.facts,
              "weights [3, 3, 32, 64] + bias [64], 18,496 parameters \xC2\xB7 output 32 \xC3\x97 32 \xC3\x97 64",
              "deeper conv facts (no numerics, no zeros)");

    const auto bn = PlanLayerViews(BatchNorm1(), nullptr, false);
    Check(Labels(bn) == std::vector<std::string>({"Feature maps", "Values", "Weights", "Gradients"}),
          "BatchNorm tabs");
    Check(bn.tabs[2].tensor == "beta" || bn.tabs[2].tensor == "gamma", "BatchNorm weights are a 1-D parameter");
    Check(bn.facts.find("running_mean") == std::string::npos, "kept state is not a parameter");

    const auto flatten = PlanLayerViews(Flatten(), nullptr, false);
    Check(Labels(flatten) == std::vector<std::string>({"Units", "Values"}), "flat output tabs");
    Check(Later(flatten) == std::vector<std::string>({"Embedding"}), "flat output can be projected later");
    CheckText(flatten.facts, "output 8192", "Flatten facts");

    const auto dense = PlanLayerViews(Dense2(), nullptr, false);
    Check(Labels(dense) == std::vector<std::string>({"Units", "Values", "Weights", "Gradients"}), "Dense tabs");
    Check(Later(dense) == std::vector<std::string>({"Contributions", "Embedding"}), "Dense later tabs");
    for (const auto& t : dense.later) Check(!t.why.empty(), "a later tab says why: " + t.label);
    Check(!dense.hint.empty(), "a weights layer has an inspector hint");

    const LayerInfo empty;
    Check(PlanLayerViews(empty, nullptr, false).tabs.empty(), "nothing captured, no tabs");
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
}

void TestChannelsAndScales() {
    const std::vector<float> means = {0.2f, 0.9f, NAN, 0.9f, 0.5f};
    Check(StrongestFirst(means) == std::vector<size_t>({1, 3, 4, 0, 2}), "strongest first, ties by number, NaN last");
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
    Check(NumbersLines(n, false).size() == 2, "weights skip the zeros line");
    DebugTensorNumericSummary none;
    CheckText(NumbersLines(none, true).front(), "No summary: the tensor is not Float32.", "no summary");
}

}  // namespace

int main() {
    TestRowsAndShapes();
    TestViewsFollowShapes();
    TestMatrixLayout();
    TestChannelsAndScales();
    TestHistogramAndNumbers();
    if (g_failures) {
        std::printf("%d layer view checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("layer view presentation passed\n");
    return EXIT_SUCCESS;
}
