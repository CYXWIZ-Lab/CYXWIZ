#pragma once

#include "node_config_dialog.h"

#include <cstddef>
#include <string>

namespace gui {

// The Embedding layer's dialog: tabs Shape, Weights, Advanced. Weights
// (TOFIX152, mockup approved 2026-10-14): the table starts Random, from a
// Word Embeddings node's saved vectors (which also sets the shape), or from a
// file; the starter-weights builder stays, folded.
class EmbeddingDialog : public NodeConfigDialog {
public:
    explicit EmbeddingDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(820, 600); }

protected:
    void RenderContent() override;
    float FooterHeight() const override;
    void RenderFooter(bool& should_close) override;

private:
    enum class Source { Random = 0, Node = 1, File = 2 };

    void LoadFromNode();
    void ResolveSource();
    void RenderShapeTab();
    void RenderWeightsTab();
    void RenderNodeSource();
    void RenderFileSource();
    void RenderStarterBuilder();
    void RenderAdvancedTab();
    void InspectFile();
    void UseNode(const MLNode& vectors_node);
    bool BuildAndSaveWeights();
    std::string Resolve(const std::string& path) const;
    // The file a Word Embeddings node saves its vectors to, as the run sets it.
    std::string VectorsFileOf(const MLNode& vectors_node) const;

    int num_embeddings_ = 10000;
    int embedding_dim_ = 256;
    int padding_idx_ = -1;
    float max_norm_ = 0.0f;
    bool freeze_ = false;
    int init_mode_ = 0;  // 0=random normal, 1=random uniform, 2=one-hot/truncated
    std::string weights_file_;
    std::string output_file_;

    Source source_ = Source::Random;
    bool source_resolved_ = false;  // needs the graph (set after construction)
    int source_node_id_ = -1;

    // The weights file's shape, read when it changes.
    std::string inspected_path_;
    size_t file_rows_ = 0;
    size_t file_cols_ = 0;
    std::string file_error_;

    std::string status_message_;
    bool status_is_error_ = false;
};

}  // namespace gui
