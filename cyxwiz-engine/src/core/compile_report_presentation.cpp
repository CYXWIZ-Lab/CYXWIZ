#include "compile_report_presentation.h"

#include "compiled_node_presentation.h"

#include <cyxwiz/error_codes.h>

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace cyxwiz::compile_report {

namespace {

std::string Number(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

// 0.8 -> "0.8", 0.001 -> "0.001", 1e-05 -> "1e-05".
std::string Short(double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%g", value);
    return text;
}

std::string Shape(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? " x " : "") + std::to_string(shape[i]);
    return text;
}

Tone LevelTone(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error: return Tone::Error;
        case IssueLevel::Warning: return Tone::Warning;
        case IssueLevel::Info: return Tone::Info;
    }
    return Tone::Info;
}

int LevelRank(IssueLevel level) {
    return level == IssueLevel::Error ? 0 : level == IssueLevel::Warning ? 1 : 2;
}

const char* LevelWord(IssueLevel level) {
    return level == IssueLevel::Error ? "Error" : level == IssueLevel::Warning ? "Warning" : "Note";
}

bool StartsWith(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

std::string RunsOn(const std::string& status) {
    if (status == BackendPlacementStatus::Gpu) return "GPU";
    if (status == BackendPlacementStatus::Cpu) return "CPU";
    if (status == BackendPlacementStatus::Mixed) return "GPU and CPU";
    if (status == BackendPlacementStatus::Risk) return "At risk";
    if (status == BackendPlacementStatus::Unsupported) return "Unsupported";
    return "Unknown";
}

Tone RunsOnTone(const std::string& status) {
    if (status == BackendPlacementStatus::Gpu) return Tone::Ok;
    if (status == BackendPlacementStatus::Unsupported) return Tone::Error;
    if (status == BackendPlacementStatus::Unknown) return Tone::Neutral;
    return Tone::Warning;  // CPU, mixed, at risk: worth a look
}

// The engine a layer runs on, in words: "ArrayFire", "native provider X".
std::string Engine(const BackendPlacementEntry& p) {
    const std::string& expected = p.expected_backend;
    if (expected.rfind("native provider ", 0) == 0) return expected;
    if (expected.find("ArrayFire") != std::string::npos) return "ArrayFire";
    if (expected == BackendPlacementStatus::Unknown || expected.empty()) return "";
    return expected;
}

std::string Why(const BackendPlacementEntry& p) {
    using namespace BackendPlacementReason;
    std::string why;
    if (p.reason_code == ArrayFireTensorOpCapable) why = "ArrayFire tensor operations on the selected device.";
    else if (p.reason_code == GraphRuntimeArrayFire) why = "An ArrayFire graph operation on the selected device.";
    else if (p.reason_code == NativeProviderSelected)
        why = "A native provider is installed and selected; ArrayFire runs it if the provider fails.";
    else if (p.reason_code == BackendCapabilityUnclassified) why = "No placement rule exists for this layer yet.";
    else if (p.reason_code == TimeDistributedSequenceWrapper)
        why = "Runs inside a time-distributed wrapper on the sequence path.";
    else if (p.reason_code == UnsupportedSequentialModelLayer) why = "The sequential model does not support this layer.";
    else why = p.explanation;
    if (p.NeedsUserAttention() && !p.suggested_action.empty() && p.suggested_action != "No action needed.")
        why += (why.empty() ? "" : " ") + p.suggested_action;
    return why;
}

std::vector<std::pair<std::string, std::string>> PlacementDetails(const BackendPlacementEntry& p) {
    std::vector<std::pair<std::string, std::string>> d;
    const auto add = [&](const char* key, const std::string& value) {
        if (!value.empty()) d.emplace_back(key, value);
    };
    add("Node type", p.node_type);
    add("Expected backend", p.expected_backend);
    add("Fallback", p.fallback_backend);
    add("Status", p.status);
    add("Reason code", p.reason_code);
    add("Explanation", p.explanation);
    add("Suggested action", p.suggested_action);
    add("Execution mode", p.declared_execution_mode);
    add("Observed device", p.observation_device);
    add("Observed dtype", p.observation_dtype);
    add("Probe outcome", p.observation_probe_outcome);
    add("Probe scope", p.observation_probe_scope);
    return d;
}

std::string Title(Mode mode, const std::string& graph) {
    const std::string suffix = graph.empty() ? std::string() : ": " + graph;
    switch (mode) {
        case Mode::Compile: return "Compile" + suffix;
        case Mode::TrainBlocked: return "Cannot start training" + suffix;
        case Mode::DebugBlocked: return "Cannot run Local Debug" + suffix;
        case Mode::CheckpointBlocked: return "Cannot load the checkpoint" + suffix;
    }
    return "Compile" + suffix;
}

std::string BackingText(const ConfigFacts& c) {
    switch (c.backing) {
        case Backing::InMemoryArrow:
            return "in memory (Arrow), " + Number(c.backing_megabytes, 1) + " MB, " + FormatCompiledCount(c.rows) +
                   " rows";
        case Backing::DiskParquet:
            return "on disk (Parquet), " + Number(c.backing_megabytes, 1) + " MB, " + FormatCompiledCount(c.rows) +
                   " rows in " + FormatCompiledCount(c.row_groups) + " row groups";
        case Backing::OtherLoaded:
            return c.backing_label + (c.rows >= 0 ? ", " + FormatCompiledCount(c.rows) + " samples" : std::string());
        case Backing::NotLoaded: return "not loaded yet (Apply the Data Input, or Train loads it)";
        case Backing::Unknown: break;
    }
    return "";
}

void AddConfig(Report& r, const ConfigFacts& c) {
    r.tiles.push_back({std::to_string(c.layers), c.layers == 1 ? "layer" : "layers"});
    r.tiles.push_back({c.input_shape.empty() ? std::to_string(c.input_size) : Shape(c.input_shape), "input per sample"});
    r.tiles.push_back({std::to_string(c.output_size), "outputs"});
    if (c.parameters >= 0)
        r.tiles.push_back({FormatCompiledCount(c.parameters),
                           "parameters, " + FormatCompiledBytes(static_cast<unsigned long long>(c.parameters) * 4ull)});

    ConfigGroup data{"Data", {}};
    data.rows.emplace_back("Dataset", c.dataset.empty() ? "not set in the Data Input" : c.dataset);
    if (const std::string backing = BackingText(c); !backing.empty()) data.rows.emplace_back("Where", backing);
    data.rows.emplace_back("Split", "train " + Short(c.train_ratio) + ", validation " + Short(c.val_ratio) + ", test " +
                                        Short(c.test_ratio) +
                                        (c.has_split ? (c.split_node.empty() ? std::string(" (from the Data Split)")
                                                                             : " (from " + c.split_node + ")")
                                                     : std::string(" (default: the graph has no Data Split)")));
    r.config.push_back(std::move(data));

    ConfigGroup loop{c.has_loader ? (c.loader_node.empty() ? std::string("Training loop (from the Data Loader)")
                                                           : "Training loop (from " + c.loader_node + ")")
                                  : std::string("Training loop (defaults: the graph has no Data Loader)"),
                     {}};
    loop.rows.emplace_back("Batches", "batch size " + std::to_string(c.batch_size) + ", " +
                                          (c.shuffle ? "shuffled" : "in order") + ", " +
                                          (c.drop_last ? "drop the last short batch" : "keep the last short batch"));
    loop.rows.emplace_back("Epochs", std::to_string(c.epochs) + ", validation every " +
                                         Count(static_cast<size_t>(std::max(1, c.validation_freq)), "epoch", "epochs") +
                                         " (the last epoch always validates)");
    loop.rows.emplace_back("Metrics", "every " + Count(static_cast<size_t>(std::max(0, c.log_interval)), "batch", "batches"));
    if (c.num_workers > 0)
        loop.rows.emplace_back("Workers", std::to_string(c.num_workers) + " (synchronous per-batch conversion where supported)");
    if (c.prefetch_factor > 0)
        loop.rows.emplace_back("Prefetch", std::to_string(c.prefetch_factor) +
                                               " (a bounded queue for Arrow and Parquet batchers)");
    loop.rows.emplace_back("Seed", std::to_string(c.seed) + " (split and shuffle order)");
    loop.rows.emplace_back("Gradient accumulation", std::to_string(c.grad_accum_steps) +
                                                        (c.grad_accum_steps == 1 ? " (an optimizer step every backward pass)"
                                                                                 : " backward passes per optimizer step"));
    r.config.push_back(std::move(loop));

    ConfigGroup learning{"Learning", {}};
    learning.rows.emplace_back("Loss", c.loss);
    learning.rows.emplace_back("Optimizer", c.optimizer + ", learning rate " + Short(c.learning_rate));
    if (c.normalize)
        learning.rows.emplace_back("Normalize", "mean " + Short(c.norm_mean) + ", std " + Short(c.norm_std));
    if (c.one_hot) learning.rows.emplace_back("Labels", "one-hot, " + Count(static_cast<size_t>(c.num_classes), "class", "classes"));
    r.config.push_back(std::move(learning));
}

std::string CopyText(const Report& r) {
    std::ostringstream out;
    out << r.title << "\n" << r.verdict << "\n";
    if (!r.subtitle.empty()) out << r.subtitle << "\n";
    out << "\nFindings\n";
    if (r.findings.empty()) out << "  none\n";
    for (const auto& f : r.findings) {
        out << "  " << LevelWord(f.level) << ": " << (f.node_name.empty() ? std::string() : f.node_name + ": ")
            << f.message << (f.code.empty() ? std::string() : " [" + f.code + "]") << "\n";
    }
    if (!r.placements.empty()) {
        out << "\nWhere each layer runs\n";
        if (!r.placement_sentence.empty()) out << "  " << r.placement_sentence << "\n";
        for (const auto& p : r.placements)
            out << "  " << p.layer << ": " << p.runs_on << (p.engine.empty() ? std::string() : " (" + p.engine + ")")
                << ". " << p.why << "\n";
    }
    if (!r.config.empty()) {
        out << "\nConfiguration\n";
        for (const auto& t : r.tiles) out << "  " << t.value << " " << t.label << "\n";
        for (const auto& g : r.config) {
            out << "  " << g.title << "\n";
            for (const auto& [key, value] : g.rows) out << "    " << key << ": " << value << "\n";
        }
    }
    return out.str();
}

}  // namespace

std::string Count(size_t n, const char* one, const char* many) {
    return FormatCompiledCount(static_cast<long long>(n)) + " " + (n == 1 ? one : many);
}

Report BuildReport(const Input& in) {
    Report r;
    r.title = Title(in.mode, in.graph_name);

    for (const auto& issue : in.issues) {
        if (issue.level == IssueLevel::Error) ++r.errors;
        else if (issue.level == IssueLevel::Warning) ++r.warnings;
        else ++r.notes;
        Finding f;
        f.level = issue.level;
        f.tone = LevelTone(issue.level);
        f.node_id = issue.node_id;
        f.node_name = issue.node_name;
        f.message = issue.message;
        f.code = issue.error_code == errors::Compiler::GenericIssue ? std::string() : issue.error_code;
        f.open_dialog = issue.node_id >= 0 && (issue.error_code == errors::Compiler::DataNotLoaded ||
                                               StartsWith(issue.message, "Could not load the data before training"));
        f.open_data_studio = issue.error_code == errors::Compiler::DataCheckSuggested;
        if (issue.node_id >= 0) {
            if (issue.error_code == errors::Compiler::TextNotTokenized) f.add_node = AddNodeFix::TextTokenizer;
            else if (issue.error_code == errors::Compiler::ImageNotResized) f.add_node = AddNodeFix::Resize;
            else if (issue.error_code == errors::Compiler::AudioFeaturesMissing) f.add_node = AddNodeFix::AudioFeatures;
        }
        r.findings.push_back(std::move(f));
    }
    std::stable_sort(r.findings.begin(), r.findings.end(),
                     [](const Finding& a, const Finding& b) { return LevelRank(a.level) < LevelRank(b.level); });
    r.ready = in.compiled_ok && r.errors == 0 && in.stop_reason.empty();

    switch (in.mode) {
        case Mode::Compile:
            if (r.ready) {
                r.verdict = r.warnings == 0 ? "Ready to train"
                                            : "Ready to train, with " + Count(r.warnings, "warning", "warnings");
                r.verdict_tone = r.warnings == 0 ? Tone::Ok : Tone::Warning;
            } else {
                r.verdict = "Not ready to train: " + Count(std::max<size_t>(r.errors, 1), "error", "errors");
                r.verdict_tone = Tone::Error;
            }
            break;
        case Mode::TrainBlocked:
            r.verdict = "Training did not start: " +
                        (in.stop_reason.empty() ? std::string("fix the errors below") : in.stop_reason);
            r.verdict_tone = Tone::Error;
            r.subtitle = "Nothing was trained and nothing changed in the graph.";
            break;
        case Mode::DebugBlocked:
            r.verdict = "Local Debug did not run: " +
                        (in.stop_reason.empty() ? std::string("fix the errors below") : in.stop_reason);
            r.verdict_tone = Tone::Error;
            break;
        case Mode::CheckpointBlocked:
            r.verdict = "The checkpoint was not loaded: " +
                        (in.stop_reason.empty() ? std::string("the open graph must compile first") : in.stop_reason);
            r.verdict_tone = Tone::Error;
            r.subtitle = "Open the graph the checkpoint was trained with, fix what is listed below, then load it again.";
            break;
    }

    r.chips.push_back({Count(r.errors, "error", "errors"), r.errors ? Tone::Error : Tone::Neutral});
    r.chips.push_back({Count(r.warnings, "warning", "warnings"), r.warnings ? Tone::Warning : Tone::Neutral});
    r.chips.push_back({Count(r.notes, "note", "notes"), r.notes ? Tone::Info : Tone::Neutral});

    // Where each layer runs.
    size_t gpu = 0, cpu = 0, mixed = 0, risk = 0, unsupported = 0, unknown = 0;
    std::string device;
    for (const auto& p : in.placements) {
        if (p.status == BackendPlacementStatus::Gpu) ++gpu;
        else if (p.status == BackendPlacementStatus::Cpu) ++cpu;
        else if (p.status == BackendPlacementStatus::Mixed) ++mixed;
        else if (p.status == BackendPlacementStatus::Risk) ++risk;
        else if (p.status == BackendPlacementStatus::Unsupported) ++unsupported;
        else ++unknown;
        if (device.empty()) device = p.observation_device;
        PlacementRow row;
        row.node_id = p.node_id;
        row.layer = p.node_name.empty() ? p.node_type : p.node_name;
        row.runs_on = RunsOn(p.status);
        row.tone = RunsOnTone(p.status);
        row.engine = Engine(p);
        row.why = Why(p);
        row.attention = p.NeedsUserAttention();
        row.details = PlacementDetails(p);
        r.placements.push_back(std::move(row));
    }
    std::stable_sort(r.placements.begin(), r.placements.end(),
                     [](const PlacementRow& a, const PlacementRow& b) { return a.attention && !b.attention; });
    if (!in.placements.empty()) {
        const size_t total = in.placements.size();
        const std::string on_device = device.empty() ? std::string() : " (" + device + ")";
        if (gpu == total) {
            r.placement_sentence = (total == 1 ? std::string("The layer runs") : "All " + std::to_string(total) + " layers run") +
                                   " on the GPU" + on_device + ". None falls back to the CPU.";
        } else {
            r.placement_sentence = std::to_string(gpu) + " of " + Count(total, "layer", "layers") + " run on the GPU" +
                                   on_device + "; the others are listed first with what to do.";
        }
        r.placement_chips = {{"GPU " + std::to_string(gpu), gpu ? Tone::Ok : Tone::Neutral},
                             {"CPU " + std::to_string(cpu), cpu ? Tone::Warning : Tone::Neutral},
                             {"Mixed " + std::to_string(mixed), mixed ? Tone::Warning : Tone::Neutral},
                             {"At risk " + std::to_string(risk), risk ? Tone::Warning : Tone::Neutral},
                             {"Unsupported " + std::to_string(unsupported), unsupported ? Tone::Error : Tone::Neutral},
                             {"Unknown " + std::to_string(unknown), Tone::Neutral}};
        r.chips.push_back({gpu == total ? Count(total, "layer on the GPU", "layers on the GPU")
                                        : std::to_string(gpu) + " of " + Count(total, "layer", "layers") + " on the GPU",
                           gpu == total ? Tone::Ok : Tone::Warning});
    }

    if (in.has_config) AddConfig(r, in.config);
    r.offer_train = in.mode == Mode::Compile && r.ready;
    r.copy_text = CopyText(r);
    return r;
}

MemoryRiskView BuildMemoryRiskView(const MemoryRiskInput& in) {
    MemoryRiskView v;
    const bool risky = in.risk == "risky";
    const std::string step = in.node_name.empty() ? std::string("this step") : "\"" + in.node_name + "\"";
    v.title = "Memory risk before training";
    v.verdict = risky ? "Preparing " + step + " may use most of the free memory"
                      : "Preparing " + step + " needs a large share of the free memory";
    v.tone = risky ? Tone::Error : Tone::Warning;
    v.subtitle = "Nothing has started yet.";
    if (!in.node_name.empty())
        v.rows.emplace_back("Step", in.node_name + (in.node_id >= 0 ? " (node " + std::to_string(in.node_id) + ")" : ""));
    v.rows.emplace_back("Estimate", "about " + FormatCompiledBytes(in.estimated_bytes));
    if (in.available_bytes) v.rows.emplace_back("Free memory now", FormatCompiledBytes(in.available_bytes));
    if (in.budget_bytes) v.rows.emplace_back("Safe budget", FormatCompiledBytes(in.budget_bytes));
    v.rows.emplace_back("Risk", risky ? "Risky: close to the free memory" : "Warning: above the warning line");
    v.estimate_share = in.available_bytes
                           ? std::min(1.0, static_cast<double>(in.estimated_bytes) / static_cast<double>(in.available_bytes))
                           : 0.0;
    v.advice = "Continuing may make Windows page to disk and slow everything down. A disk-backed Data Input (Memory tab) "
               "avoids it.";
    v.detail = in.detail;
    return v;
}

}  // namespace cyxwiz::compile_report
