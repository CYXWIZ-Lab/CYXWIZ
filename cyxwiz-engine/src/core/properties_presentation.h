#pragma once

// The Properties panel as data (TOFIX129 piece A7): the header, the settings
// rows with their truth, the AS COMPILED and fusion cards, the Advanced table
// and the empty state. Built once per (node, graph revision, compile serial,
// edit) by the panel and drawn from the result every frame. Pure data in and
// out: no ImGui, no registry probes, no backend calls; the panel gathers the
// facts (truth report, dataset facts, compile result) and hands them in.

#include "compiled_node_presentation.h"
#include "node_metadata.h"
#include "sequence_fusion_presentation.h"
#include "../gui/properties_truth.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::properties_view {

// The shared status vocabulary of the panel: one chip per status.
enum class ChipKind { Ok, Dialog, Default, Alias, Missing, Conflict, Stale, Unsupported, Info, Planned, Deprecated, External };

struct Chip {
    ChipKind kind = ChipKind::Info;
    std::string text;  // "OK", "dialog", "default", "alias", "missing", ...
};

const char* ChipKindName(ChipKind kind);
Chip ChipFor(gui::properties_truth::TruthStatus status);

// One truth entry (an effective setting the rules know), attached to the
// setting row whose key it matches, or listed read-only when no row has it.
struct Truth {
    std::string canonical_key;
    std::vector<std::string> keys;  // canonical, source and alias keys that match a row
    std::string label;
    std::string value;
    std::string default_value;
    std::string source_key;
    std::string owner;              // "Compiler", "Runtime", "Loader", ...
    std::string message;
    bool quick_editable = false;
    bool requires_dialog = false;
    std::vector<Chip> chips;
    std::vector<std::pair<std::string, std::string>> aliases;  // key, value present
    // "Source: units · Owner: Compiler" for the hover text and Details.
    std::string provenance;
};

// Matches a setting key against a truth's keys.
const Truth* FindTruth(const std::vector<Truth>& truths, const std::string& key);

struct Header {
    std::string icon;
    std::string name;
    int id = -1;
    std::string type_name;
    std::string category;
    std::string implementation;  // "Implemented", "Planned", "Deprecated", "External"
    ChipKind implementation_kind = ChipKind::Ok;
    std::string badge;
    // The one primary action: "Open Dialog...", "Open Plot window", "Open Dashboard"; empty when none.
    std::string primary_label;
    std::string primary_tip;
    std::string primary_tip_detail;
};

enum class SettingsRoute {
    Metadata,       // rows below, from the node metadata
    Custom,         // a per-node editor draws the rows (properties_node_editors)
    ViewSettings,   // Plot / Dashboard: the saved settings in words, read-only
    DialogOnly,     // everything is set in the node's dialog
    Fallback        // no metadata parameters: the generic editor
};

// A metadata-route row. The editor widget comes from `param`; the value is
// what the node stores, or the default when the node has no value yet
// (displaying a node never writes into it).
struct Setting {
    const ParameterDefinition* param = nullptr;
    std::string key;
    std::string label;
    std::string value;
    bool stored = false;            // the node has a value for this key
    bool required = false;
    bool differs_from_default = false;
    std::string validation_error;   // empty when valid
    const Truth* truth = nullptr;   // the matching truth, if any
};

struct SettingGroup {
    std::string name;   // "" for the flat group
    bool advanced = false;
    std::vector<Setting> rows;
};

// Data Input: what the panel knows about the data (A7-4).
struct DataFact {
    std::string label;        // file name or dataset name
    std::string store;        // "Arrow", "Parquet", "ImageDataset", ...
    bool loaded = false;
    std::string detail;       // "9,000 rows × 23 columns", "1,200 images, 2 classes"
    std::string note;         // "not loaded: ..." when not
};

struct RawRow {
    std::string key;
    std::string value;
    std::string maps_to;        // "" when unmapped
    bool cleanup_allowed = false;
    std::string cleanup_reason;
    std::vector<Chip> chips;
};

struct View {
    Header header;
    SettingsRoute route = SettingsRoute::Metadata;
    std::vector<SettingGroup> groups;        // Metadata route
    std::vector<Truth> truths;               // every truth of the node
    std::vector<const Truth*> loose_truths;  // truths no metadata row matched (listed read-only)
    std::string settings_note;               // under the rows; "" when none
    std::optional<DataFact> data;

    bool has_compiled = false;
    CompiledNodeCard compiled;
    SequenceFusionCard fusion;               // fusion.applies when shown

    bool has_position = false;
    float position_x = 0.0f, position_y = 0.0f;
    int links_in = 0, links_out = 0;
    std::vector<RawRow> raw;

    bool has_executor = false;               // K-Means: the executor section follows
};

struct Inputs {
    const gui::MLNode* node = nullptr;
    const NodeMetadata* metadata = nullptr;
    const std::vector<gui::MLNode>* nodes = nullptr;
    const std::vector<gui::NodeLink>* links = nullptr;
    gui::properties_truth::NodeTruthReport truth;
    const gui::properties_truth::DatasetTruthFact* dataset = nullptr;  // this node's, Data Input only
    bool has_dialog = false;        // NodeConfigDialogFactory has one for the type
    bool has_executor = false;
    bool custom_editor = false;     // metadata says Custom
    bool dialog_only = false;       // metadata says Dialog
    const CompiledNodeInputs* compiled = nullptr;  // nullptr: no compile facts
};

View Build(const Inputs& inputs);

// Nothing selected: the graph in one line.
struct EmptyView {
    std::string graph_name;     // "Untitled graph" or the file's stem
    size_t nodes = 0, links = 0;
    std::string status;         // "Compiled", "Compile failed", ... ("" when no compile yet)
    CompiledStatusKind kind = CompiledStatusKind::None;
    std::string first_error;    // "1 error on MSE" ("" when none)
};

EmptyView BuildEmpty(const std::string& file_path,
                     size_t node_count,
                     size_t link_count,
                     LiveCompileState state,
                     const TrainingConfiguration* config,
                     const std::vector<gui::MLNode>* nodes);

// "Generation Preview Every Epochs" under "Generation previews" -> "Every epochs":
// drops the group's words from the front of a label; the full label stays for hover.
std::string ShortLabel(const std::string& label, const std::string& group);

// What changed between two views that an editor would need to redraw: the
// key of a cache entry.
struct Key {
    int node_id = -1;
    uint64_t graph_revision = 0;
    uint64_t compile_serial = 0;
    uint64_t edit_serial = 0;
    uint64_t facts_serial = 0;   // dataset facts refresh
    bool operator==(const Key& other) const {
        return node_id == other.node_id && graph_revision == other.graph_revision &&
               compile_serial == other.compile_serial && edit_serial == other.edit_serial &&
               facts_serial == other.facts_serial;
    }
};

}  // namespace cyxwiz::properties_view
