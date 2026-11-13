#include "nas_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::nas_presentation {

namespace {

const char* const kDot = " \xC2\xB7 ";  // " · "

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
    return it == nodes.end() ? nullptr : &*it;
}

// A node's name, with its number when another node has the same name
// ("Dense #9").
std::string NameOf(const std::vector<gui::MLNode>& nodes, const gui::MLNode& node) {
    const bool repeated = std::any_of(nodes.begin(), nodes.end(), [&node](const gui::MLNode& other) {
        return other.id != node.id && other.name == node.name;
    });
    return repeated ? node.name + " #" + std::to_string(node.id) : node.name;
}

std::string Plural(int count, const char* one, const char* many) {
    return std::to_string(count) + " " + (count == 1 ? one : many);
}

}  // namespace

GraphShape ReadShape(const TrainingConfiguration& compiled) {
    GraphShape shape;
    const auto& in = compiled.input_shape;
    if (in.empty()) {
        shape.why = "The compile does not know the input size yet: load the graph's data (its Data Input). The "
                    "panel reads the graph again when the compile changes.";
        return shape;
    }
    if (compiled.output_size == 0) {
        shape.why = "The compile found no output size: link the model's last layer to the loss.";
        return shape;
    }
    const bool regression = UsesContinuousTargetMetrics(compiled);
    shape.outputs = static_cast<int>(compiled.output_size);
    if (in.size() == 3) {
        shape.inputs_text = std::to_string(in[0]) + " x " + std::to_string(in[1]) + " x " + std::to_string(in[2]) +
                            " images";
        shape.task = regression ? "regression" : "image_classification";
    } else {
        size_t features = 1;
        for (size_t d : in) features *= d;
        shape.inputs_text = std::to_string(features) + " features";
        shape.task = regression ? "regression" : "classification";
    }
    shape.outputs_text = regression ? Plural(shape.outputs, "value", "values") : Plural(shape.outputs, "class", "classes");
    shape.text = shape.inputs_text + " in, " + shape.outputs_text + " out";
    shape.ok = true;
    return shape;
}

std::string HeadNote(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                     const GraphShape& shape) {
    const std::vector<int> trainable = NASEvaluator::FindTrainableLayers(nodes);
    const std::vector<int> body = NASEvaluator::FindBodyLayers(nodes, links);
    std::string heads;
    int count = 0;
    for (int index : trainable) {
        if (std::find(body.begin(), body.end(), index) != body.end()) continue;
        heads += (heads.empty() ? "" : ", ") + NameOf(nodes, nodes[static_cast<size_t>(index)]);
        ++count;
    }
    if (count == 0) return "The output layer is never removed, resized or swapped.";
    std::string note = std::string("The ") + (count == 1 ? "head" : "heads") + " (" + heads;
    if (shape.ok) note += ", the " + std::to_string(shape.outputs) + " outputs";
    note += count == 1 ? ") is" : ") are";
    return note + " never removed, resized or swapped.";
}

std::vector<ScorePart> ScoreParts(const ArchitectureScore& score) {
    return {
        {"Size", score.complexity_score, "Fewer parameters score higher (1 million: 0.25)."},
        {"Efficiency", score.efficiency_score, "Computation per parameter (50 operations per parameter: 1)."},
        {"Depth", score.depth_score, "Trainable layers near 10 score highest."},
        {"Variety", score.diversity_score, "Different kinds of trainable layers (5 kinds: 1)."},
        {"Skip links", score.connectivity_score, "Skip links and merges of two paths (3: 1)."},
    };
}

std::string ScoreSummary(const ArchitectureScore& score, const std::vector<gui::MLNode>& nodes,
                         const std::vector<gui::NodeLink>& links) {
    std::vector<std::pair<std::string, int>> counts;
    for (int index : NASEvaluator::ModelChain(nodes, links)) {
        const std::string kind = NASEvaluator::LayerKind(nodes[static_cast<size_t>(index)]);
        auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == kind; });
        if (it == counts.end()) counts.emplace_back(kind, 1);
        else ++it->second;
    }
    std::string layers;
    for (const auto& [kind, count] : counts) layers += (layers.empty() ? "" : ", ") + std::to_string(count) + "x " + kind;
    std::string text = "Parameters " + Grouped(score.total_params);
    text += kDot + std::string("model layers: ") + (layers.empty() ? "none" : layers);
    text += kDot + (score.skip_connections == 0 ? std::string("no skip connections")
                                                : Plural(score.skip_connections, "skip connection", "skip connections"));
    return text;
}

std::string Grouped(int64_t value) {
    std::string digits = std::to_string(value < 0 ? -value : value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return value < 0 ? "-" + digits : digits;
}

std::string ScoreText(double value) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.2f", value);
    return text;
}

const std::vector<Mutation>& Mutations() {
    static const std::vector<Mutation> mutations = {
        {MutationType::AddLayer, "Add layer", "A Dense, BatchNorm or Dropout in front of a layer, linked in"},
        {MutationType::RemoveLayer, "Remove layer", "A layer other than the head; what fed it feeds what it fed"},
        {MutationType::SwapLayer, "Swap layer", "Another type that fits there; its links stay"},
        {MutationType::ChangeUnits, "Change units", "Units or filters x 0.5 to 2, not the head"},
        {MutationType::ChangeActivation, "Change activation", "ReLU, LeakyReLU, GELU, Swish or Mish"},
        {MutationType::Random, "Random", "One of the above"},
    };
    return mutations;
}

std::string DescribeChange(const std::vector<gui::MLNode>& before, const std::vector<gui::NodeLink>& before_links,
                           const std::vector<gui::MLNode>& after, const std::vector<gui::NodeLink>& after_links) {
    std::vector<std::string> parts;
    for (const auto& node : after) {
        const gui::MLNode* old = FindNode(before, node.id);
        if (!old) {
            std::string text = "Added " + NASEvaluator::LayerText(node);
            for (const auto& link : after_links) {
                if (link.from_node != node.id) continue;
                if (const gui::MLNode* next = FindNode(after, link.to_node)) {
                    text += " before " + NameOf(after, *next);
                    break;
                }
            }
            parts.push_back(text);
        } else if (old->type != node.type) {
            parts.push_back(NameOf(before, *old) + " became " + NASEvaluator::LayerText(node));
        } else {
            for (const auto& [key, value] : node.parameters) {
                const auto was = old->parameters.find(key);
                if (was == old->parameters.end() || was->second == value) continue;
                std::string label = key;
                std::replace(label.begin(), label.end(), '_', ' ');
                parts.push_back(NameOf(after, node) + " " + label + " " + was->second + " to " + value);
            }
        }
    }
    for (const auto& node : before) {
        if (FindNode(after, node.id)) continue;
        std::string text = "Removed " + NameOf(before, node);
        const gui::MLNode* feeder = nullptr;
        const gui::MLNode* fed = nullptr;
        for (const auto& link : before_links) {
            if (!feeder && link.to_node == node.id) feeder = FindNode(before, link.from_node);
            if (!fed && link.from_node == node.id) fed = FindNode(before, link.to_node);
        }
        if (feeder && fed) text += " (" + NameOf(before, *feeder) + " feeds " + NameOf(before, *fed) + ")";
        parts.push_back(text);
    }
    std::string text;
    for (const auto& part : parts) text += (text.empty() ? "" : "; ") + part;
    return text;
}

std::string LayersText(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    std::string text;
    for (int index : NASEvaluator::ModelChain(nodes, links))
        text += (text.empty() ? "" : ", ") + NASEvaluator::LayerText(nodes[static_cast<size_t>(index)]);
    return text.empty() ? "no layers" : text;
}

std::string LossText(const TrialScore& score) {
    if (!score.has_validation) return "-";
    char text[32];
    std::snprintf(text, sizeof(text), "%.4g", score.val_loss);
    return text;
}

std::string AccuracyText(const TrialScore& score) {
    if (!score.has_validation) return "-";
    char text[32];
    if (score.regression) std::snprintf(text, sizeof(text), "MAE %.4g", score.val_mae);
    else std::snprintf(text, sizeof(text), "%.1f %%", score.val_accuracy * 100.0f);
    return text;
}

void ApplyDefaults(NASSearchConfig& config, int& train_best, const core::ArchitectureSearchDefaults& defaults) {
    config.population_size = defaults.versions;
    config.generations = defaults.rounds;
    config.elite_count = defaults.kept;
    config.mutation_rate = defaults.changed_percent / 100.0;
    config.crossover_rate = defaults.mixed_percent / 100.0;
    train_best = defaults.train_best;
}

core::ArchitectureSearchDefaults DefaultsOf(const NASSearchConfig& config, int train_best) {
    core::ArchitectureSearchDefaults defaults;
    defaults.versions = config.population_size;
    defaults.rounds = config.generations;
    defaults.kept = config.elite_count;
    defaults.changed_percent = static_cast<int>(std::lround(config.mutation_rate * 100.0));
    defaults.mixed_percent = static_cast<int>(std::lround(config.crossover_rate * 100.0));
    defaults.train_best = train_best;
    return defaults;
}

}  // namespace cyxwiz::nas_presentation
