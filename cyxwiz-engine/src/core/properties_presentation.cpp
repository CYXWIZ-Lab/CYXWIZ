#include "properties_presentation.h"

#include "../gui/node_editor.h"
#include "../gui/properties_parameter_rules.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>

namespace cyxwiz::properties_view {

namespace {

using gui::properties_truth::TruthStatus;

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

std::vector<std::string> Words(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream in(text);
    std::string word;
    while (in >> word) words.push_back(word);
    return words;
}

std::string HumanizeParameterName(const std::string& name) {
    std::string label;
    label.reserve(name.size());
    bool capitalize_next = true;
    for (char ch : name) {
        if (ch == '_' || ch == '-') {
            label.push_back(' ');
            capitalize_next = true;
            continue;
        }
        const auto c = static_cast<unsigned char>(ch);
        label.push_back(capitalize_next ? static_cast<char>(std::toupper(c)) : ch);
        capitalize_next = false;
    }
    return label.empty() ? name : label;
}

std::string ParameterLabel(const ParameterDefinition& param) {
    return param.display_name.empty() ? HumanizeParameterName(param.name) : param.display_name;
}

std::string ParamOr(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

std::string WithThousands(size_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count && count % 3 == 0) out.insert(out.begin(), ',');
        out.insert(out.begin(), *it);
        ++count;
    }
    return out;
}

const char* ImplementationName(NodeImplementationStatus status) {
    switch (status) {
        case NodeImplementationStatus::Implemented: return "Implemented";
        case NodeImplementationStatus::Template: return "Planned";
        case NodeImplementationStatus::Deprecated: return "Deprecated";
        case NodeImplementationStatus::External: return "External";
    }
    return "Unknown";
}

ChipKind ImplementationKind(NodeImplementationStatus status) {
    switch (status) {
        case NodeImplementationStatus::Implemented: return ChipKind::Ok;
        case NodeImplementationStatus::Template: return ChipKind::Planned;
        case NodeImplementationStatus::Deprecated: return ChipKind::Deprecated;
        case NodeImplementationStatus::External: return ChipKind::External;
    }
    return ChipKind::Info;
}

Truth TruthFrom(const gui::properties_truth::PropertyTruth& property) {
    Truth t;
    t.canonical_key = property.canonical_key;
    if (!property.canonical_key.empty()) t.keys.push_back(property.canonical_key);
    if (!property.source_key.empty() && property.source_key != property.canonical_key)
        t.keys.push_back(property.source_key);
    for (const auto& alias : property.aliases_present) {
        if (std::find(t.keys.begin(), t.keys.end(), alias.key) == t.keys.end()) t.keys.push_back(alias.key);
        t.aliases.emplace_back(alias.key, alias.value);
    }
    t.label = property.label;
    t.value = property.effective_value;
    t.default_value = property.default_value;
    t.source_key = property.source_key;
    t.owner = gui::properties_truth::TruthOwnerName(property.owner);
    t.message = property.message;
    t.quick_editable = property.quick_editable;
    t.requires_dialog = property.requires_dialog;
    for (const auto status : property.statuses) t.chips.push_back(ChipFor(status));
    t.provenance = "Source: " + (property.source_key.empty() ? std::string("(none)") : property.source_key) +
                   " \xC2\xB7 Owner: " + t.owner;
    return t;
}

}  // namespace

const char* ChipKindName(ChipKind kind) {
    switch (kind) {
        case ChipKind::Ok: return "ok";
        case ChipKind::Dialog: return "dialog";
        case ChipKind::Default: return "default";
        case ChipKind::Alias: return "alias";
        case ChipKind::Missing: return "missing";
        case ChipKind::Conflict: return "conflict";
        case ChipKind::Stale: return "stale";
        case ChipKind::Unsupported: return "unsupported";
        case ChipKind::Info: return "info";
        case ChipKind::Planned: return "planned";
        case ChipKind::Deprecated: return "deprecated";
        case ChipKind::External: return "external";
    }
    return "info";
}

Chip ChipFor(TruthStatus status) {
    switch (status) {
        case TruthStatus::OK: return {ChipKind::Ok, "OK"};
        case TruthStatus::Missing: return {ChipKind::Missing, "missing"};
        case TruthStatus::Defaulted: return {ChipKind::Default, "default"};
        case TruthStatus::AliasUsed: return {ChipKind::Alias, "alias"};
        case TruthStatus::Stale: return {ChipKind::Stale, "stale"};
        case TruthStatus::Conflicting: return {ChipKind::Conflict, "conflict"};
        case TruthStatus::RuntimeOnly: return {ChipKind::Info, "runtime"};
        case TruthStatus::CompilerOnly: return {ChipKind::Info, "compiler"};
        case TruthStatus::Unsupported: return {ChipKind::Unsupported, "unsupported"};
        case TruthStatus::RequiresDialog: return {ChipKind::Dialog, "dialog"};
    }
    return {ChipKind::Info, gui::properties_truth::TruthStatusName(status)};
}

const Truth* FindTruth(const std::vector<Truth>& truths, const std::string& key) {
    if (key.empty()) return nullptr;
    for (const auto& truth : truths) {
        if (std::find(truth.keys.begin(), truth.keys.end(), key) != truth.keys.end()) return &truth;
    }
    return nullptr;
}

std::string ShortLabel(const std::string& label, const std::string& group) {
    if (group.empty()) return label;
    const auto label_words = Words(label);
    const auto group_words = Words(group);
    size_t strip = 0;
    while (strip < group_words.size() && strip + 1 < label_words.size()) {
        const std::string a = Lower(label_words[strip]);
        const std::string b = Lower(group_words[strip]);
        // "Preview" matches "previews": one is a prefix of the other.
        if (a.rfind(b, 0) != 0 && b.rfind(a, 0) != 0) break;
        ++strip;
    }
    if (strip == 0) return label;
    std::string out;
    for (size_t i = strip; i < label_words.size(); ++i) {
        if (!out.empty()) out.push_back(' ');
        out += label_words[i];
    }
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

View Build(const Inputs& in) {
    View v;
    if (!in.node) return v;
    const gui::MLNode& node = *in.node;

    // Header
    v.header.name = node.name;
    v.header.id = node.id;
    if (in.metadata) {
        v.header.icon = in.metadata->icon;
        v.header.type_name = in.metadata->name;
        v.header.category = GetCategoryDisplayName(in.metadata->category);
        v.header.implementation = ImplementationName(in.metadata->status);
        v.header.implementation_kind = ImplementationKind(in.metadata->status);
        v.header.badge = in.metadata->badge;
    }
    if (node.type == gui::NodeType::Plot || node.type == gui::NodeType::Dashboard) {
        v.header.primary_label = node.type == gui::NodeType::Plot ? "Open Plot window" : "Open Dashboard";
        v.header.primary_tip = "Pick the plot type, columns, rows and colours";
        v.header.primary_tip_detail = "(reads the data at this node)";
    } else if (in.has_dialog) {
        v.header.primary_label = "Open Dialog...";
        v.header.primary_tip = "Open detailed configuration dialog";
        v.header.primary_tip_detail = "(Configure all settings with preview)";
    }

    // Truths
    for (const auto& property : in.truth.properties) v.truths.push_back(TruthFrom(property));

    // Route and rows
    if (in.dialog_only) {
        v.route = SettingsRoute::DialogOnly;
    } else if (in.custom_editor) {
        v.route = SettingsRoute::Custom;
    } else if (node.type == gui::NodeType::Plot || node.type == gui::NodeType::Dashboard) {
        v.route = SettingsRoute::ViewSettings;
    } else if (!in.metadata || in.metadata->parameters.empty()) {
        v.route = SettingsRoute::Fallback;
    } else {
        v.route = SettingsRoute::Metadata;
        auto group_for = [&v](const std::string& name, bool advanced) -> SettingGroup& {
            for (auto& g : v.groups) {
                if (g.name == name && g.advanced == advanced) return g;
            }
            v.groups.push_back(SettingGroup{name, advanced, {}});
            return v.groups.back();
        };
        for (const auto& param : in.metadata->parameters) {
            if (gui::properties_rules::ShouldHideGenericParameter(node, param)) continue;
            Setting row;
            row.param = &param;
            row.key = param.name;
            const std::string group_name = param.advanced ? "Advanced settings" : param.group;
            row.full_label = ParameterLabel(param);
            row.label = ShortLabel(row.full_label, group_name);
            const auto it = node.parameters.find(param.name);
            row.stored = it != node.parameters.end() && !it->second.empty();
            row.value = row.stored ? it->second : param.default_value;
            row.required = param.required;
            row.differs_from_default = !param.default_value.empty() && row.value != param.default_value;
            std::string error;
            if (!gui::properties_rules::ValidateParameter(row.value, param, error)) row.validation_error = error;
            group_for(group_name, param.advanced).rows.push_back(std::move(row));
        }
        // Advanced after the named groups, named groups after the flat one.
        std::stable_sort(v.groups.begin(), v.groups.end(), [](const SettingGroup& a, const SettingGroup& b) {
            if (a.advanced != b.advanced) return !a.advanced;
            return a.name.empty() && !b.name.empty();
        });
        if (v.groups.empty()) v.settings_note = "Configure this node from its dialog.";
    }

    // Attach truths to rows; the rest is listed read-only.
    std::vector<const Truth*> matched;
    for (auto& group : v.groups) {
        for (auto& row : group.rows) {
            row.truth = FindTruth(v.truths, row.key);
            if (row.truth) matched.push_back(row.truth);
        }
    }
    for (const auto& truth : v.truths) {
        if (std::find(matched.begin(), matched.end(), &truth) == matched.end()) v.loose_truths.push_back(&truth);
    }
    // Data Input: the data fact (A7-4)
    if (node.type == gui::NodeType::DataInput) {
        DataFact fact;
        const std::string file = ParamOr(node, "file_path");
        if (!file.empty()) {
            fact.label = std::filesystem::path(file).filename().string();
        } else if (in.dataset && !in.dataset->dataset_name.empty()) {
            fact.label = in.dataset->dataset_name;
        } else {
            fact.label = "No file chosen";
        }
        if (in.dataset && in.dataset->found) {
            fact.loaded = true;
            fact.store = in.dataset->backing_store;
            std::string detail;
            if (in.dataset->rows > 0) {
                detail = WithThousands(in.dataset->rows) +
                         (in.dataset->backing_store == "ImageDataset" ? " images" : " rows");
            }
            if (!in.dataset->columns.empty()) {
                if (!detail.empty()) detail += " \xC3\x97 ";
                detail += std::to_string(in.dataset->columns.size()) + " columns";
            }
            if (in.dataset->has_class_count && in.dataset->class_count > 0) {
                if (!detail.empty()) detail += ", ";
                detail += std::to_string(in.dataset->class_count) + " classes";
            }
            fact.detail = detail;
        } else {
            fact.loaded = false;
            fact.note = file.empty() ? "Open the dialog to choose the data."
                                     : "Not loaded: press Apply in the dialog to read it.";
        }
        v.data = std::move(fact);
    }

    if (v.route == SettingsRoute::DialogOnly) {
        v.settings_note = v.loose_truths.empty() && !v.data
                              ? "This node is set up in its dialog."
                              : "Everything else about this node is set in its dialog.";
    }

    // Cards
    if (in.compiled && in.nodes && in.links) {
        v.has_compiled = true;
        v.compiled = BuildCompiledNodeCard(*in.nodes, *in.links, node.id, *in.compiled);
    }
    if (node.type == gui::NodeType::Concatenate && in.nodes && in.links) {
        v.fusion = BuildSequenceFusionCard(*in.nodes, *in.links, node.id);
    }

    // Advanced
    v.has_position = node.has_initial_position;
    v.position_x = node.initial_pos_x;
    v.position_y = node.initial_pos_y;
    if (in.links) {
        for (const auto& link : *in.links) {
            if (link.to_node == node.id) ++v.links_in;
            if (link.from_node == node.id) ++v.links_out;
        }
    }
    for (const auto& raw : in.truth.raw_parameters) {
        RawRow row;
        row.key = raw.key;
        row.value = raw.value;
        row.maps_to = raw.maps_to;
        row.cleanup_allowed = raw.cleanup_allowed;
        row.cleanup_reason = raw.cleanup_reason;
        for (const auto status : raw.statuses) row.chips.push_back(ChipFor(status));
        v.raw.push_back(std::move(row));
    }
    for (const auto& [key, value] : node.parameters) {
        const bool listed = std::any_of(v.raw.begin(), v.raw.end(), [&key](const RawRow& r) { return r.key == key; });
        if (!listed) v.raw.push_back(RawRow{key, value, {}, false, {}, {}});
    }
    v.has_executor = in.has_executor;
    return v;
}

EmptyView BuildEmpty(const std::string& file_path,
                     size_t node_count,
                     size_t link_count,
                     LiveCompileState state,
                     const TrainingConfiguration* config,
                     const std::vector<gui::MLNode>* nodes) {
    EmptyView e;
    e.graph_name = file_path.empty() ? "Untitled graph" : std::filesystem::path(file_path).stem().string();
    e.nodes = node_count;
    e.links = link_count;
    switch (state) {
        case LiveCompileState::NotCompiled:
            e.status = node_count ? "Not compiled yet" : "";
            e.kind = CompiledStatusKind::None;
            break;
        case LiveCompileState::Compiling:
            e.status = "Compiling...";
            e.kind = CompiledStatusKind::Pending;
            break;
        case LiveCompileState::Compiled:
            e.status = "Compiled";
            e.kind = CompiledStatusKind::Ok;
            break;
        case LiveCompileState::Failed:
            e.status = "Compile failed";
            e.kind = CompiledStatusKind::Failed;
            break;
    }
    if (config) {
        size_t errors = 0;
        std::string first_node;
        for (const auto& issue : config->issues) {
            if (issue.level != IssueLevel::Error) continue;
            ++errors;
            if (first_node.empty() && nodes) {
                for (const auto& n : *nodes) {
                    if (n.id == issue.node_id) {
                        first_node = n.name;
                        break;
                    }
                }
            }
        }
        if (errors > 0) {
            e.first_error = std::to_string(errors) + (errors == 1 ? " error" : " errors");
            if (!first_node.empty()) e.first_error += " on " + first_node;
        }
    }
    return e;
}

}  // namespace cyxwiz::properties_view
