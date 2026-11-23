#include "training_health_presentation.h"

#include "compiled_node_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>

namespace cyxwiz {

namespace {

std::string Number(double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", v);
    return text;
}

std::string Arrows(const std::vector<double>& values) {
    std::string text;
    for (size_t i = 0; i < values.size(); ++i) text += (i ? " \xE2\x86\x92 " : "") + Number(values[i]);
    return text;
}

std::string Percent(double share) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.0f %%", share * 100.0);
    return text;
}

// The ReLU right after weight layer `layer` (no other weight layer between).
std::optional<ReluHealth> ReluAfter(const EpochHealth& epoch, size_t layer) {
    std::optional<ReluHealth> best;
    for (const auto& relu : epoch.relus) {
        if (relu.layer <= layer) continue;
        const bool weight_between = std::any_of(epoch.weights.begin(), epoch.weights.end(), [&](const WeightHealth& w) {
            return w.layer > layer && w.layer < relu.layer;
        });
        if (weight_between) continue;
        if (!best || relu.layer < best->layer) best = relu;
    }
    return best;
}

}  // namespace

HealthView BuildHealthView(const TrainingHealth& health, const std::vector<std::string>& layer_names) {
    HealthView view;
    if (health.epochs.empty()) {
        view.empty_text = "Train the graph: health is recorded once an epoch (weights, gradients, dead units).";
        return view;
    }
    view.has_health = true;
    view.epochs = std::to_string(health.epochs.size()) + (health.epochs.size() == 1 ? " epoch" : " epochs");
    const EpochHealth& last = health.epochs.back();
    const EpochHealth* checked = nullptr;  // the last epoch with a validation pass
    for (const auto& epoch : health.epochs)
        if (!epoch.relus.empty()) checked = &epoch;

    struct Dead {
        std::string name;
        long long dead = 0, units = 0;
    };
    std::vector<Dead> dead_layers;
    bool exploding = false, vanishing = false;

    for (const auto& w : last.weights) {
        HealthRow row;
        row.layer = w.layer;
        row.name = w.layer < layer_names.size() && !layer_names[w.layer].empty() ? layer_names[w.layer]
                                                                                 : "layer " + std::to_string(w.layer);
        std::vector<double> norms, gradients;
        std::vector<std::string> dead_by_epoch;
        for (const auto& epoch : health.epochs) {
            for (const auto& e : epoch.weights) {
                if (e.parameter != w.parameter) continue;
                norms.push_back(e.norm);
                gradients.push_back(e.mean_gradient_norm);
            }
            if (const auto relu = ReluAfter(epoch, w.layer)) dead_by_epoch.push_back(std::to_string(relu->dead));
        }
        row.weight_norms = Arrows(norms);
        row.gradient_norms = Arrows(gradients);
        row.histogram = w.histogram;
        row.low = Number(w.low);
        row.high = Number(w.high);
        row.count = w.count;
        if (!gradients.empty()) {
            const double g = gradients.back();
            if (!std::isfinite(g) || g > 1e3) exploding = true;
            if (g < 1e-7) vanishing = true;
        }
        row.words = FormatCompiledCount(static_cast<long long>(w.count)) + " weights from " + row.low + " to " + row.high + ".";
        const auto relu = checked ? ReluAfter(*checked, w.layer) : std::nullopt;
        if (!checked) {
            row.dead = "not checked";
        } else if (!relu) {
            row.dead = "no ReLU";
        } else {
            const double share = relu->units > 0 ? static_cast<double>(relu->dead) / static_cast<double>(relu->units) : 0.0;
            row.dead = std::to_string(relu->dead) + " of " + std::to_string(relu->units) +
                       (relu->dead > 0 ? " (" + Percent(share) + ")" : "");
            row.dead_warning = share >= 0.2;
            if (relu->dead > 0) {
                dead_layers.push_back({row.name, relu->dead, relu->units});
                std::string history;
                for (size_t i = 0; i < dead_by_epoch.size(); ++i) history += (i ? " \xE2\x86\x92 " : "") + dead_by_epoch[i];
                row.words += " Dead after its ReLU by epoch: " + history + " of " + std::to_string(relu->units) + ".";
            } else {
                row.words += " Every unit after its ReLU fires for some validation row.";
            }
        }
        view.rows.push_back(std::move(row));
    }
    std::sort(view.rows.begin(), view.rows.end(), [](const HealthRow& a, const HealthRow& b) { return a.layer < b.layer; });

    const std::string gradients = exploding ? "Gradients grow very large: try a lower learning rate or gradient clipping."
                                  : vanishing ? "Gradients are vanishing (below 1e-7) in some layer."
                                              : "Gradients stay in range (no exploding or vanishing).";
    std::sort(dead_layers.begin(), dead_layers.end(), [](const Dead& a, const Dead& b) {
        return static_cast<double>(a.dead) / std::max<long long>(1, a.units) > static_cast<double>(b.dead) / std::max<long long>(1, b.units);
    });
    if (!dead_layers.empty() && static_cast<double>(dead_layers[0].dead) / std::max<long long>(1, dead_layers[0].units) >= 0.2) {
        const Dead& worst = dead_layers[0];
        view.warning = true;
        view.headline = std::to_string(worst.dead) + " of " + worst.name + "'s " + std::to_string(worst.units) +
                        " units are dead: they give 0 for every validation row.";
        view.detail = "A unit that never fires gets no gradient, so it stops learning.";
        for (size_t i = 1; i < dead_layers.size(); ++i)
            view.detail += " " + std::to_string(dead_layers[i].dead) + " of " + dead_layers[i].name + "'s " +
                           std::to_string(dead_layers[i].units) + " too.";
        view.detail += " " + gradients;
        view.advice = "LeakyReLU in place of ReLU after " + worst.name + ", a lower learning rate, or a narrower layer (" +
                      std::to_string(worst.units - worst.dead) + " live units do the work).";
    } else if (checked) {
        view.warning = exploding || vanishing;
        view.headline = dead_layers.empty() ? "No dead units: every ReLU fires for some validation row."
                                            : "A few dead units, under a fifth of any layer.";
        view.detail = gradients;
        view.advice = exploding ? "A lower learning rate, or Gradient clip on the optimizer." : "Nothing stands out.";
    } else {
        view.warning = exploding || vanishing;
        view.headline = "Weights and gradients by epoch";
        view.detail = gradients + " Dead units are counted on the validation pass of a layer stack.";
        view.advice = exploding ? "A lower learning rate, or Gradient clip on the optimizer." : "Nothing stands out.";
    }
    return view;
}

}  // namespace cyxwiz
