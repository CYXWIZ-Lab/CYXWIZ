#include "sequence_fusion_presentation.h"

#include <algorithm>
#include <cctype>
#include <deque>

namespace cyxwiz {

namespace {

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string Param(const gui::MLNode& node, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        const auto it = node.parameters.find(key);
        if (it != node.parameters.end() && !it->second.empty()) return it->second;
    }
    return {};
}

const gui::MLNode* NodeById(const std::vector<gui::MLNode>& nodes, int id) {
    for (const auto& node : nodes) {
        if (node.id == id) return &node;
    }
    return nullptr;
}

const gui::NodeLink* LinkInto(const std::vector<gui::NodeLink>& links, int node_id, int pin_id) {
    for (const auto& link : links) {
        if (link.to_node == node_id && link.to_pin == pin_id) return &link;
    }
    return nullptr;
}

bool InPath(const std::unordered_set<int>& path_ids, int id) {
    return path_ids.empty() || path_ids.count(id) > 0;
}

// "30,000"
std::string Grouped(const std::string& digits) {
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(),
                                       [](unsigned char c) { return std::isdigit(c) != 0; })) {
        return digits;
    }
    std::string out;
    const size_t lead = digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && i >= lead && (i - lead) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}

int ParseInt(const std::string& value, int fallback) {
    try {
        return value.empty() ? fallback : std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

void Reject(SequenceFusionVerdict& verdict, SequenceFusionRejection why) {
    verdict.rejection = why;
    switch (why) {
        case SequenceFusionRejection::None:
            break;
        case SequenceFusionRejection::SecondConcatenate:
            verdict.reason = "Only one Concatenate may join Embedding outputs.";
            verdict.fix = "Keep one word + POS Concatenate; remove or rewire the others.";
            break;
        case SequenceFusionRejection::InputsNotConnected:
            verdict.reason = "Connect exactly Input 1 and Input 2.";
            verdict.fix = "Word Embedding goes to Input 1, POS Embedding to Input 2.";
            break;
        case SequenceFusionRejection::InputNotEmbedding:
            verdict.reason = "Both inputs must come from their own Embedding node.";
            verdict.fix = "Connect a Word Embedding to Input 1 and a POS Embedding to Input 2.";
            break;
        case SequenceFusionRejection::DifferentBatchSources:
            verdict.reason = "Both Embeddings must read the same batch output.";
            verdict.fix = "Connect both Embeddings to the DataLoader's Data output.";
            break;
        case SequenceFusionRejection::EmbeddingSharedOutput:
            verdict.reason = verdict.embedding_name + " may feed only the Concatenate.";
            verdict.fix = "Remove its other output links.";
            break;
        case SequenceFusionRejection::WrongAxis:
            verdict.reason = "Concatenate dim is " + verdict.dim + ", the sequence axis.";
            verdict.fix = "Word and POS features join on the feature axis: dim = -1.";
            verdict.action = SequenceFusionAction::SetFeatureAxis;
            break;
        case SequenceFusionRejection::NoPosColumn:
            verdict.reason = "The sequence data declares no POS column.";
            verdict.fix = "Set the POS column in the Data Input's Sequence columns.";
            verdict.action = SequenceFusionAction::OpenDataInput;
            break;
    }
}

}  // namespace

bool IsSequenceTaggingCategory(const std::string& file_category) {
    const std::string value = Lower(file_category);
    return value == "sequence" || value == "sequence_text" || value == "sequence_tagging" ||
           value == "token_tagging" || value == "ner";
}

SequenceTaggingContract ReadSequenceTaggingContract(const gui::MLNode& data_input) {
    SequenceTaggingContract contract;
    contract.is_sequence = IsSequenceTaggingCategory(Param(data_input, {"file_category"}));
    contract.token_column = Param(data_input, {"token_column", "tokens_column", "token_sequence_column"});
    contract.pos_column = Param(data_input, {"pos_column", "pos_sequence_column"});
    contract.tag_column = Param(data_input, {"tag_column", "tags_column", "tag_sequence_column"});
    contract.sentence_id_column = Param(data_input, {"sentence_id_column", "sequence_id_column"});
    contract.max_sequence_length = std::max(0, ParseInt(Param(data_input, {"max_sequence_length"}), 0));
    const std::string mask = Lower(Param(data_input, {"create_attention_mask"}));
    contract.create_attention_mask = !mask.empty() && mask != "false" && mask != "0" && mask != "off";
    return contract;
}

std::vector<SequenceFusionVerdict> AnalyzeSequenceFusion(
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::unordered_set<int>& path_ids,
    bool pos_column_declared) {
    std::vector<SequenceFusionVerdict> verdicts;
    bool have_fusion = false;
    for (const auto& concat : nodes) {
        if (concat.type != gui::NodeType::Concatenate || !InPath(path_ids, concat.id)) continue;
        std::vector<const gui::NodeLink*> inputs;
        bool fed_by_embedding = false;
        for (const auto& pin : concat.inputs) {
            if (const auto* link = LinkInto(links, concat.id, pin.id)) {
                inputs.push_back(link);
                const auto* source = NodeById(nodes, link->from_node);
                fed_by_embedding |= source && source->type == gui::NodeType::Embedding;
            }
        }
        if (!fed_by_embedding) continue;

        SequenceFusionVerdict verdict;
        verdict.concat_node_id = concat.id;
        const auto dim_it = concat.parameters.find("dim");
        verdict.dim = dim_it == concat.parameters.end() ? "1" : dim_it->second;
        const auto* input1 = concat.inputs.size() > 0 ? LinkInto(links, concat.id, concat.inputs[0].id) : nullptr;
        const auto* input2 = concat.inputs.size() > 1 ? LinkInto(links, concat.id, concat.inputs[1].id) : nullptr;
        const auto* word = input1 ? NodeById(nodes, input1->from_node) : nullptr;
        const auto* pos = input2 ? NodeById(nodes, input2->from_node) : nullptr;
        if (word && word->type == gui::NodeType::Embedding) verdict.word_node_id = word->id;
        if (pos && pos->type == gui::NodeType::Embedding) verdict.pos_node_id = pos->id;

        if (have_fusion) {
            Reject(verdict, SequenceFusionRejection::SecondConcatenate);
        } else if (!input1 || !input2 || inputs.size() != 2) {
            Reject(verdict, SequenceFusionRejection::InputsNotConnected);
        } else if (verdict.word_node_id < 0 || verdict.pos_node_id < 0 || word->id == pos->id) {
            Reject(verdict, SequenceFusionRejection::InputNotEmbedding);
        } else {
            const auto* word_source = word->inputs.empty() ? nullptr : LinkInto(links, word->id, word->inputs[0].id);
            const auto* pos_source = pos->inputs.empty() ? nullptr : LinkInto(links, pos->id, pos->inputs[0].id);
            if (!word_source || !pos_source || word_source->from_node != pos_source->from_node ||
                word_source->from_pin != pos_source->from_pin) {
                Reject(verdict, SequenceFusionRejection::DifferentBatchSources);
            } else {
                for (const auto* embedding : {word, pos}) {
                    const bool shared = std::any_of(links.begin(), links.end(), [&](const gui::NodeLink& link) {
                        return link.from_node == embedding->id && link.to_node != concat.id;
                    });
                    if (shared && verdict.Compiles()) {
                        verdict.embedding_name = embedding->name;
                        Reject(verdict, SequenceFusionRejection::EmbeddingSharedOutput);
                    }
                }
                if (verdict.Compiles() && verdict.dim != "-1" && verdict.dim != "2") {
                    Reject(verdict, SequenceFusionRejection::WrongAxis);
                } else if (verdict.Compiles() && !pos_column_declared) {
                    Reject(verdict, SequenceFusionRejection::NoPosColumn);
                }
            }
        }
        have_fusion |= verdict.Compiles();
        verdicts.push_back(std::move(verdict));
    }
    return verdicts;
}

std::string SequenceFusionIssueMessage(const SequenceFusionVerdict& verdict) {
    return "Word + POS fusion: " + verdict.reason + " " + verdict.fix +
           " Supported shape: Word Embedding -> Concatenate Input 1, POS Embedding -> "
           "Concatenate Input 2, both fed by the same DataLoader output, Concatenate dim=-1.";
}

const gui::MLNode* FindUpstreamDataInput(const std::vector<gui::MLNode>& nodes,
                                         const std::vector<gui::NodeLink>& links,
                                         int node_id) {
    std::deque<int> queue{node_id};
    std::unordered_set<int> seen{node_id};
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop_front();
        for (const auto& link : links) {
            if (link.to_node != current || !seen.insert(link.from_node).second) continue;
            const auto* source = NodeById(nodes, link.from_node);
            if (!source) continue;
            if (source->type == gui::NodeType::DataInput || source->type == gui::NodeType::DatasetInput) {
                return source;
            }
            queue.push_back(source->id);
        }
    }
    return nullptr;
}

SequenceFusionCard BuildSequenceFusionCard(const std::vector<gui::MLNode>& nodes,
                                           const std::vector<gui::NodeLink>& links,
                                           int concat_node_id) {
    SequenceFusionCard card;
    const auto* data_input = FindUpstreamDataInput(nodes, links, concat_node_id);
    const SequenceTaggingContract contract =
        data_input ? ReadSequenceTaggingContract(*data_input) : SequenceTaggingContract{};
    const auto verdicts = AnalyzeSequenceFusion(nodes, links, {}, !contract.pos_column.empty());
    const auto it = std::find_if(verdicts.begin(), verdicts.end(), [&](const SequenceFusionVerdict& v) {
        return v.concat_node_id == concat_node_id;
    });
    if (it == verdicts.end()) return card;

    card.applies = true;
    card.compiles = it->Compiles();
    card.status = card.compiles ? "Compiles" : "Will not compile";
    card.reason = it->reason;
    card.fix = it->fix;
    card.data_input_node_id = data_input ? data_input->id : -1;
    card.action = it->action;
    if (card.action == SequenceFusionAction::SetFeatureAxis) card.action_label = "Set dim to -1";
    if (card.action == SequenceFusionAction::OpenDataInput) {
        card.action_label = "Open Data Input";
        if (!data_input) card.action = SequenceFusionAction::None, card.action_label.clear();
    }

    int width = 0;
    const auto embedding_row = [&](int id) -> std::string {
        const auto* node = NodeById(nodes, id);
        if (!node) return {};
        const std::string vocab = Param(*node, {"num_embeddings"});
        const std::string dim = Param(*node, {"embedding_dim"});
        width += ParseInt(dim, 0);
        return node->name + ", " + Grouped(vocab.empty() ? "?" : vocab) + " x " + (dim.empty() ? "?" : dim);
    };
    const std::string word_row = embedding_row(it->word_node_id);
    const std::string pos_row = embedding_row(it->pos_node_id);
    if (!word_row.empty()) card.rows.emplace_back("Input 1 (word ids)", word_row);
    if (!pos_row.empty()) card.rows.emplace_back("Input 2 (POS ids)", pos_row);
    if (card.compiles) card.rows.emplace_back("Output", std::to_string(width) + " features per token");

    if (card.compiles) {
        card.details.emplace_back("Compiled layer", "SequenceFeatureFusion");
        card.details.emplace_back("Model input", "[batch, seq, 2] word/POS ids");
        card.details.emplace_back("Position", "first model layer");
        const auto* word = NodeById(nodes, it->word_node_id);
        if (word && !word->inputs.empty()) {
            if (const auto* source_link = LinkInto(links, word->id, word->inputs[0].id)) {
                if (const auto* source = NodeById(nodes, source_link->from_node)) {
                    std::string pin_name;
                    for (const auto& pin : source->outputs) {
                        if (pin.id == source_link->from_pin) pin_name = pin.name;
                    }
                    card.details.emplace_back("Batch source",
                                              pin_name.empty() ? source->name : source->name + ", " + pin_name);
                }
            }
        }
        if (data_input && !contract.pos_column.empty()) {
            card.details.emplace_back("POS column", contract.pos_column + " (" + data_input->name + ")");
        }
        card.details.emplace_back("PyTorch export", "torch.cat([word, pos], -1)");
    }
    return card;
}

std::optional<std::string> SequenceFusionEmbeddingRole(const std::vector<gui::MLNode>& nodes,
                                                       const std::vector<gui::NodeLink>& links,
                                                       int embedding_node_id) {
    for (const auto& verdict : AnalyzeSequenceFusion(nodes, links, {}, true)) {
        if (verdict.word_node_id == embedding_node_id && verdict.pos_node_id >= 0) {
            return std::string("word ids, fused with POS (Concatenate Input 1)");
        }
        if (verdict.pos_node_id == embedding_node_id && verdict.word_node_id >= 0) {
            return std::string("POS ids, fused with words (Concatenate Input 2)");
        }
    }
    return std::nullopt;
}

}  // namespace cyxwiz
