#include "live_graph_compile.h"

#include "model_builder.h"

#include <cyxwiz/device.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>
#include <string>

namespace cyxwiz {

namespace {

void Mix(uint64_t& h, const std::string& text) {
    for (unsigned char c : text) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    h ^= 0xff;
    h *= 0x100000001b3ULL;
}

void Mix(uint64_t& h, long long value) { Mix(h, std::to_string(value)); }

}  // namespace

LiveGraphCompile::~LiveGraphCompile() {
    // A running compile holds only its own graph copy; wait so it does not
    // outlive the objects it logs through.
    if (pending_.valid()) pending_.wait();
}

uint64_t LiveGraphCompile::HashGraph(const std::vector<gui::MLNode>& nodes,
                                     const std::vector<gui::NodeLink>& links) {
    uint64_t h = 0xcbf29ce484222325ULL;
    std::vector<const gui::MLNode*> sorted;
    sorted.reserve(nodes.size());
    for (const auto& node : nodes) sorted.push_back(&node);
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) { return a->id < b->id; });
    for (const auto* node : sorted) {
        Mix(h, node->id);
        Mix(h, static_cast<long long>(node->type));
        Mix(h, node->name);
        for (const auto& [key, value] : node->parameters) {
            Mix(h, key);
            Mix(h, value);
        }
        for (const auto& pin : node->inputs) Mix(h, pin.id);
        for (const auto& pin : node->outputs) Mix(h, pin.id);
    }
    std::vector<const gui::NodeLink*> sorted_links;
    for (const auto& link : links) sorted_links.push_back(&link);
    std::sort(sorted_links.begin(), sorted_links.end(), [](const auto* a, const auto* b) {
        return std::tie(a->from_node, a->from_pin, a->to_node, a->to_pin) <
               std::tie(b->from_node, b->from_pin, b->to_node, b->to_pin);
    });
    for (const auto* link : sorted_links) {
        Mix(h, link->from_node);
        Mix(h, link->from_pin);
        Mix(h, link->to_node);
        Mix(h, link->to_pin);
    }
    return h;
}

void LiveGraphCompile::Collect() {
    if (!pending_.valid() ||
        pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }
    try {
        result_ = pending_.get();
        ++result_serial_;
    } catch (const std::exception& error) {
        spdlog::warn("Live graph compile failed: {}", error.what());
        pending_ = {};
    }
}

void LiveGraphCompile::Update(const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links,
                              bool count_parameters,
                              Clock::time_point now) {
    Collect();
    const uint64_t revision = HashGraph(nodes, links);
    if (!seen_any_ || revision != seen_revision_) {
        seen_any_ = true;
        seen_revision_ = revision;
        last_change_ = now;
    }
    const bool up_to_date = result_ && result_->revision == revision;
    const bool running = pending_.valid();
    if (up_to_date || running || now - last_change_ < debounce) return;
    if (nodes.empty()) {
        result_.reset();
        return;
    }

    requested_revision_ = revision;
    pending_ = std::async(std::launch::async,
        [nodes, links, revision, count_parameters]() {
            auto result = std::make_shared<Result>();
            result->revision = revision;
            // allow_unloaded_data: the canvas view compiles from the saved
            // data contract; data readiness is reported as issues.
            result->config = GraphCompiler{}.Compile(nodes, links, true);
            if (!count_parameters || result->config.layers.empty()) return result;
            try {
                // Build where training would: the worker otherwise starts on
                // ArrayFire's default backend, not the selected route.
                if (const auto selected = Device::GetProcessDevice()) {
                    Device(selected->type, selected->device_id).ActivateExact(false);
                }
                auto built = BuildExecutableFromConfig(result->config);
                if (built.ok()) {
                    std::vector<std::pair<std::string, long long>> named;
                    for (const auto& [name, tensor] : built.model->GetParameters()) {
                        named.emplace_back(name, static_cast<long long>(tensor.NumElements()));
                    }
                    result->layer_parameters = CountParametersPerLayer(named);
                    result->parameters_counted = true;
                }
            } catch (const std::exception& error) {
                spdlog::debug("Live graph compile: parameter count skipped: {}", error.what());
            }
            return result;
        });
}

LiveCompileState LiveGraphCompile::State() const {
    if (pending_.valid()) return LiveCompileState::Compiling;
    if (!result_) return LiveCompileState::NotCompiled;
    return result_->config.is_valid ? LiveCompileState::Compiled : LiveCompileState::Failed;
}

const std::map<size_t, long long>& LiveGraphCompile::LayerParameters() const {
    static const std::map<size_t, long long> empty;
    return result_ ? result_->layer_parameters : empty;
}

bool LiveGraphCompile::NodeHasError(int node_id) const {
    return !NodeErrorMessage(node_id).empty();
}

std::string LiveGraphCompile::NodeErrorMessage(int node_id) const {
    if (!result_) return {};
    for (const auto& issue : result_->config.issues) {
        if (issue.node_id == node_id && issue.level == IssueLevel::Error) return issue.message;
    }
    return {};
}

}  // namespace cyxwiz
