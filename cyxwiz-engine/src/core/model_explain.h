#pragma once

// Explain one prediction of the trained model (the Model window's Explain
// tab, TOFIX129 D11): Grad-CAM, saliency and Integrated Gradients through the
// SDK's ModelInterpretability, on any model the Engine trains (a layer stack
// or a graph with branches; Grad-CAM needs a layer stack with an image-shaped
// layer). The sample is one row of model input as Test keeps it
// (TestingMetrics::kept_inputs): already prepared, [H, W, C] rows for an
// image model.

#include "executable_model.h"
#include "graph_compiler.h"
#include "test_executor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct ExplainRequest {
    std::vector<float> sample;  // one row of model input
    int target_class = -1;      // -1: the predicted class
    int ig_steps = 64;
    bool grad_cam = true;
};

struct ExplainOutcome {
    bool ok = false;
    std::string error;

    int predicted = -1;
    int target = -1;
    float target_probability = 0.0f;  // softmax of the scores
    float predicted_probability = 0.0f;
    std::vector<float> probabilities;

    // Maps over the input's height x width (a table row: 1 x features);
    // saliency |d score / d x| and Integrated Gradients summed over channels.
    size_t height = 1, width = 0, channels = 1;
    std::vector<float> input;       // the sample, [height, width, channels]
    std::vector<float> saliency;
    std::vector<float> integrated;  // signed
    // Grad-CAM, resized to height x width; empty with a note when it cannot run.
    std::vector<float> grad_cam;
    std::string grad_cam_layer;
    std::string grad_cam_note;

    // Completeness of Integrated Gradients.
    int ig_steps = 0;
    float ig_sum = 0.0f;
    float score = 0.0f, baseline_score = 0.0f;
};

// The baseline of Integrated Gradients: the value 0 as the model sees it
// after the graph's Normalize (a black image; a table row of zeros), one
// value per input feature.
std::vector<float> ExplainBaseline(const TrainingConfiguration& config, size_t features);

// Runs forward and backward passes on `model` (set to eval mode; its
// parameter gradients change). Blocking: run it on a task, on the model's
// device.
ExplainOutcome ExplainSample(IExecutableModel& model, const TrainingConfiguration& config,
                             const ExplainRequest& request);

// The words over the four maps: "Row 9009 is a 7, but the model says 2 with
// 98.7 %" and the completeness line.
struct ExplainWords {
    std::string headline;
    bool wrong = false;
    std::string completeness;  // "They add up to 9.579; the score moved 9.515 from the baseline (0.7 % off)."
};
ExplainWords DescribeExplanation(const ExplainOutcome& outcome, long long row, int truth,
                                 const std::vector<std::string>& class_names);

// The rows Explain can show: the test samples whose inputs the last test
// kept, most confident mistakes first, then the others in test order.
struct ExplainRowChoice {
    size_t kept = 0;        // index into TestingMetrics::kept_inputs
    int sample = 0;
    long long row = -1;     // the dataset row, when the test knows it
    int truth = -1, predicted = -1;
    float confidence = 0.0f;
    bool wrong = false;
    std::string label;      // "row 9009 · a 7, read as 2 (98.7 %)"
};
std::vector<ExplainRowChoice> ExplainRowChoices(const TestingMetrics& test, bool wrong_only,
                                                const std::vector<std::string>& class_names);

}  // namespace cyxwiz
