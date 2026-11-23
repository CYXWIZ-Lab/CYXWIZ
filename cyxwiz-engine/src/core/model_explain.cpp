#include "model_explain.h"

#include "spatial_batch_layout.h"
#include "spatial_sequential_head.h"
#include "class_words.h"
#include "model_decision.h"

#include <cyxwiz/model_interpretability.h>
#include <cyxwiz/sequential.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace cyxwiz {

namespace {

size_t Product(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return shape.empty() ? 0 : n;
}

std::string ClassName(int index, const std::vector<std::string>& names) {
    if (index >= 0 && static_cast<size_t>(index) < names.size() && !names[index].empty()) return names[index];
    return std::to_string(index);
}

std::string Percent(float p) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.1f %%", p * 100.0f);
    return text;
}

// Per pixel over the channels of an [H, W, C] row: sum of |v| or of v.
std::vector<float> PerPixel(const std::vector<float>& values, size_t channels, bool absolute) {
    std::vector<float> map(values.size() / std::max<size_t>(1, channels), 0.0f);
    for (size_t i = 0; i < values.size(); ++i) map[i / channels] += absolute ? std::fabs(values[i]) : values[i];
    return map;
}

}  // namespace

std::vector<float> ExplainBaseline(const TrainingConfiguration& config, size_t features) {
    std::vector<float> baseline(features, 0.0f);
    const auto& pre = config.preprocessing;
    if (!pre.has_normalization) return baseline;
    const size_t channels = config.input_shape.size() == 3 ? config.input_shape[2] : 1;
    for (size_t i = 0; i < features; ++i) {
        const size_t c = i % channels;
        const bool per_channel = c < pre.norm_channel_mean.size() && c < pre.norm_channel_std.size();
        const float mean = per_channel ? pre.norm_channel_mean[c] : pre.norm_mean;
        const float sd = per_channel ? pre.norm_channel_std[c] : pre.norm_std;
        baseline[i] = sd != 0.0f ? (0.0f - mean) / sd : 0.0f;
    }
    return baseline;
}

ExplainOutcome ExplainSample(IExecutableModel& model, const TrainingConfiguration& config,
                             const ExplainRequest& request) {
    ExplainOutcome out;
    try {
        if (!config.layers.empty() && (config.layers.front().type == gui::NodeType::Embedding ||
                                       config.layers.front().type == gui::NodeType::Concatenate))
            throw std::invalid_argument(
                "Explain needs number inputs: this model starts with an Embedding, which reads token ids.");
        const size_t features = request.sample.size();
        if (features == 0) throw std::invalid_argument("No sample to explain.");
        const bool spatial = UsesSpatialSequentialInput(config);
        std::vector<size_t> sample_shape = config.input_shape;
        if (spatial && Product(sample_shape) != features)
            throw std::invalid_argument("The kept sample does not match the model's input shape.");
        if (sample_shape.size() == 3) {
            out.height = sample_shape[0];
            out.width = sample_shape[1];
            out.channels = sample_shape[2];
        } else {
            out.height = 1;
            out.width = features;
            out.channels = 1;
        }
        out.input = request.sample;

        model.SetTraining(false);
        ExplainFunctions functions;
        functions.forward = [&](const Tensor& rows) {
            return spatial ? model.Forward(SpatialBatchFromRows(rows, sample_shape)) : model.Forward(rows);
        };
        functions.backward = [&](const Tensor& grad) {
            const Tensor g = model.Backward(grad);
            return spatial ? SpatialBatchToRows(g) : g;
        };
        const Tensor row({1, features}, request.sample.data());
        const std::vector<float> baseline_values = ExplainBaseline(config, features);
        const Tensor baseline({1, features}, baseline_values.data());

        // Scores and probabilities, as Test reads them: by the model's loss
        // (softmax of logits, a sigmoid of one logit, probabilities as they
        // are), then the kept threshold and calibration.
        const Tensor scores = functions.forward(row);
        const float* s = scores.ReadData<float>();
        const size_t outputs = scores.NumElements();
        if (outputs == 0) throw std::runtime_error("The model gave no scores.");
        ModelDecision decision = config.decision;
        decision.scores = DecisionScoresForLoss(config.loss_type);
        out.probabilities = DecisionProbabilities(decision, s, outputs);
        const size_t classes = out.probabilities.size();
        out.predicted = ApplyModelDecision(decision, out.probabilities);
        out.target = request.target_class >= 0 ? request.target_class : out.predicted;
        if (out.target < 0 || out.target >= static_cast<int>(classes))
            throw std::invalid_argument("Class " + std::to_string(out.target) + " is outside the model's " +
                                        std::to_string(classes) + " classes.");
        out.predicted_probability = out.probabilities[out.predicted];
        out.target_probability = out.probabilities[out.target];
        // The score explained: the target's output. A yes / no model with one
        // output scores the yes class; its no class is explained by the same
        // score (a higher yes score is a lower no probability).
        const int score_index = outputs == 1 ? 0 : out.target;

        const SaliencyMap saliency = ModelInterpretability::ComputeSaliencyMap(functions, row, score_index);
        if (!saliency.success) throw std::runtime_error("Saliency: " + saliency.error_message);
        out.saliency = PerPixel(saliency.gradients, out.channels, true);

        const auto ig = ModelInterpretability::ComputeIntegratedGradients(functions, row, baseline, score_index,
                                                                          std::max(1, request.ig_steps));
        if (!ig.success) throw std::runtime_error("Integrated Gradients: " + ig.error_message);
        out.integrated = PerPixel(ig.attributions, out.channels, false);
        out.ig_steps = ig.steps;
        out.ig_sum = ig.attribution_sum;
        out.score = ig.score;
        out.baseline_score = ig.baseline_score;

        if (!request.grad_cam) {
            out.grad_cam_note = "Grad-CAM was not asked for.";
        } else if (!spatial) {
            out.grad_cam_note = "Grad-CAM needs a layer with an image-shaped output; this model has none.";
        } else if (!model.AsSequentialModel()) {
            out.grad_cam_note = "Grad-CAM runs on layer stacks; this graph has branches. Saliency and Integrated "
                                "Gradients cover it.";
        } else {
            SequentialModel& stack = *model.AsSequentialModel();
            // The last convolution: the module with an [H, W, C] output deepest in the stack.
            const Tensor spatial_row = SpatialBatchFromRows(row, sample_shape);
            Tensor current = spatial_row;
            int layer = -1;
            for (size_t i = 0; i < stack.Size(); ++i) {
                current = stack.GetModule(i)->Forward(current);
                if (current.Shape().size() == 4 && stack.GetModule(i)->GetName().find("Conv") != std::string::npos)
                    layer = static_cast<int>(i);
            }
            if (layer < 0) {
                out.grad_cam_note = "Grad-CAM needs a convolution; this model has none.";
            } else {
                const auto cam = ModelInterpretability::ComputeGradCAM(stack, spatial_row, layer, score_index);
                if (!cam.success) {
                    out.grad_cam_note = "Grad-CAM: " + cam.error_message;
                } else {
                    out.grad_cam = cam.raw_heatmap;
                    for (auto it = config.layers.rbegin(); it != config.layers.rend(); ++it) {
                        if (it->type == gui::NodeType::Conv2D || it->type == gui::NodeType::DepthwiseConv2D) {
                            out.grad_cam_layer = it->name;
                            break;
                        }
                    }
                    if (out.grad_cam_layer.empty()) out.grad_cam_layer = cam.layer_name;
                }
            }
        }
        out.ok = true;
    } catch (const std::exception& error) {
        out.ok = false;
        out.error = error.what();
    }
    return out;
}

ExplainWords DescribeExplanation(const ExplainOutcome& outcome, long long row, int truth,
                                 const std::vector<std::string>& class_names) {
    ExplainWords words;
    if (!outcome.ok) {
        words.headline = outcome.error;
        return words;
    }
    const std::string said = ClassName(outcome.predicted, class_names);
    const std::string subject = row >= 0 ? "Row " + std::to_string(row) : std::string("This sample");
    if (truth >= 0 && truth != outcome.predicted) {
        words.wrong = true;
        words.headline = subject + " is " + WithArticle(ClassName(truth, class_names)) + ", but the model says " + said + " with " +
                         Percent(outcome.predicted_probability) + ".";
    } else if (truth >= 0) {
        words.headline = subject + " is " + WithArticle(ClassName(truth, class_names)) + ", and the model says " + said + " (" +
                         Percent(outcome.predicted_probability) + ").";
    } else {
        words.headline = "The model says " + said + " (" + Percent(outcome.predicted_probability) + ").";
    }
    if (outcome.target != outcome.predicted)
        words.headline += " Explaining the score of " + ClassName(outcome.target, class_names) + " (" +
                          Percent(outcome.target_probability) + ").";
    const float moved = outcome.score - outcome.baseline_score;
    char text[200];
    const double off = moved != 0.0f ? std::fabs((outcome.ig_sum - moved) / moved) * 100.0 : 0.0;
    std::snprintf(text, sizeof(text), "They add up to %.3f; the score moved %.3f from the baseline (%.1f %% off).",
                  outcome.ig_sum, moved, off);
    words.completeness = text;
    return words;
}

std::vector<ExplainRowChoice> ExplainRowChoices(const TestingMetrics& test, bool wrong_only,
                                                const std::vector<std::string>& class_names) {
    std::vector<ExplainRowChoice> choices;
    for (size_t k = 0; k < test.kept_inputs.size(); ++k) {
        const int sample = test.kept_inputs[k].sample;
        if (sample < 0 || static_cast<size_t>(sample) >= test.predictions.size()) continue;
        ExplainRowChoice choice;
        choice.kept = k;
        choice.sample = sample;
        choice.row = static_cast<size_t>(sample) < test.row_ids.size() ? test.row_ids[sample] : sample;
        choice.predicted = test.predictions[sample];
        choice.truth = static_cast<size_t>(sample) < test.ground_truth.size() ? test.ground_truth[sample] : -1;
        choice.confidence = static_cast<size_t>(sample) < test.confidences.size() ? test.confidences[sample] : 0.0f;
        choice.wrong = choice.truth >= 0 && choice.truth != choice.predicted;
        if (wrong_only && !choice.wrong) continue;
        choice.label = "row " + std::to_string(choice.row) + " \xC2\xB7 " +
                       (choice.truth >= 0 ? WithArticle(ClassName(choice.truth, class_names)) + ", " : std::string()) +
                       "read as " + ClassName(choice.predicted, class_names) + " (" + Percent(choice.confidence) + ")";
        choices.push_back(std::move(choice));
    }
    std::stable_sort(choices.begin(), choices.end(), [](const ExplainRowChoice& a, const ExplainRowChoice& b) {
        if (a.wrong != b.wrong) return a.wrong;
        return a.wrong && a.confidence > b.confidence;
    });
    return choices;
}

}  // namespace cyxwiz
