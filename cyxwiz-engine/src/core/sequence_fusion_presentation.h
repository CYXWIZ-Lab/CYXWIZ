#pragma once

// Word + POS sequence tagging (TOFIX112 / TOFIX58 phase 2): the one place
// that decides whether a Concatenate over Embeddings compiles to the fused
// word + POS layer, and why not. The graph compiler turns rejections into
// issues; the Properties panel shows the same verdict as a card. Pure data:
// no ImGui, no backend calls.

#include "graph_model.h"

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cyxwiz {

// The token-tagging contract a sequence Data Input carries.
struct SequenceTaggingContract {
    bool is_sequence = false;  // file_category is a sequence/token-tagging category
    std::string token_column;
    std::string pos_column;
    std::string tag_column;
    std::string sentence_id_column;
    int max_sequence_length = 0;  // 0 = the longest in the data
    bool create_attention_mask = false;
};

// file_category values the compiler treats as token tagging.
bool IsSequenceTaggingCategory(const std::string& file_category);

// Reads canonical keys first, then the older aliases.
SequenceTaggingContract ReadSequenceTaggingContract(const gui::MLNode& data_input);

enum class SequenceFusionRejection {
    None,
    SecondConcatenate,      // only one Concatenate may join Embedding outputs
    InputsNotConnected,     // Input 1 and Input 2 must both be connected, nothing else
    InputNotEmbedding,      // each input from its own Embedding node
    DifferentBatchSources,  // both Embeddings read the same batch output
    EmbeddingSharedOutput,  // an Embedding feeds something besides the Concatenate
    WrongAxis,              // dim is not -1 / 2
    NoPosColumn,            // the sequence data declares no POS column
};

// The single fix an action can apply safely; the rest need rewiring.
enum class SequenceFusionAction { None, SetFeatureAxis, OpenDataInput };

struct SequenceFusionVerdict {
    int concat_node_id = -1;
    int word_node_id = -1;  // set when the input exists and is an Embedding
    int pos_node_id = -1;
    SequenceFusionRejection rejection = SequenceFusionRejection::None;
    std::string reason;  // plain words; empty when it compiles
    std::string fix;     // what to do next
    SequenceFusionAction action = SequenceFusionAction::None;
    std::string embedding_name;  // the Embedding named by EmbeddingSharedOutput
    std::string dim;             // the Concatenate's dim as written

    bool Compiles() const { return rejection == SequenceFusionRejection::None; }
};

// One verdict per Concatenate that an Embedding feeds (others are not word +
// POS fusion and are left to the generic graph runtime). `path_ids` limits
// the check to the selected training path (empty = every node).
// `pos_column_declared` is whether the sequence data declares a POS column.
std::vector<SequenceFusionVerdict> AnalyzeSequenceFusion(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& path_ids,
    bool pos_column_declared);

// The compiler's issue text for a rejected verdict.
std::string SequenceFusionIssueMessage(const SequenceFusionVerdict& verdict);

// --- Properties panel ------------------------------------------------------

// The Data Input feeding `node_id` (walking links upstream), if any.
const gui::MLNode* FindUpstreamDataInput(const std::vector<gui::MLNode>& nodes,
                                         const std::vector<gui::NodeLink>& links,
                                         int node_id);

struct SequenceFusionCard {
    bool applies = false;  // the Concatenate is fed by an Embedding
    bool compiles = false;
    std::string status;  // "Compiles" / "Will not compile"
    std::string reason;
    std::string fix;
    SequenceFusionAction action = SequenceFusionAction::None;
    std::string action_label;         // "Set dim to -1" / "Open Data Input"
    int data_input_node_id = -1;      // target of OpenDataInput
    // Input 1 / Input 2 / Output rows, then Details (key, value), empty skipped.
    std::vector<std::pair<std::string, std::string>> rows;
    std::vector<std::pair<std::string, std::string>> details;
};

SequenceFusionCard BuildSequenceFusionCard(const std::vector<gui::MLNode>& nodes,
                                           const std::vector<gui::NodeLink>& links,
                                           int concat_node_id);

// "word ids, fused with POS (Concatenate Input 1)" for an Embedding that
// feeds a word + POS Concatenate; nullopt otherwise.
std::optional<std::string> SequenceFusionEmbeddingRole(const std::vector<gui::MLNode>& nodes,
                                                       const std::vector<gui::NodeLink>& links,
                                                       int embedding_node_id);

}  // namespace cyxwiz
