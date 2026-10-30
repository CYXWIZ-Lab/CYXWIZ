#pragma once

// Data Studio > Prepare for an image graph (TOFIX178 1.6, mockup
// 6A5BVo57245gev3cU466SR): the graph's image steps as training runs them,
// in two groups - done once when the images are prepared, drawn per batch on
// training images - with Resize first and Normalize last. Each row is the
// graph's own node: its picture on a sample image, its settings (Properties'
// editor), Show on canvas, Open dialog, Remove; rows drag to reorder their
// nodes; "+ Add" puts an image step after a group's last row. Draws
// core/image_recipe_presentation.

#include "../../core/image_pipeline_preview.h"
#include "../../core/image_recipe_presentation.h"
#include "../properties.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class ImageRecipeView {
public:
    ImageRecipeView();
    ~ImageRecipeView();
    ImageRecipeView(const ImageRecipeView&) = delete;
    ImageRecipeView& operator=(const ImageRecipeView&) = delete;

    // True when the canvas graph is an image graph (its compile has a Resize
    // or image steps): Prepare shows this view instead of the table steps.
    static bool IsImageGraph(const ::gui::NodeEditor* editor);
    void Render(::gui::NodeEditor* editor);

    // Show these nodes on the canvas (Data Studio's Show on canvas).
    std::function<void(const std::vector<int>&)> show_on_canvas;

private:
    struct Previews;
    void UpdatePreviews(const TrainingConfiguration& config, const image_recipe::View& view);
    void ReleaseTextures();
    std::string SamplePath(const std::string& dataset, size_t* count) const;
    void RenderRow(::gui::NodeEditor* editor, const image_recipe::View& view, size_t index);
    void RenderAddMenu(::gui::NodeEditor* editor, int after_node, bool per_batch);
    void DropTarget(::gui::NodeEditor* editor, const image_recipe::View& view, long long after_index);

    size_t sample_ = 0;
    std::string preview_key_;
    uint64_t task_ = 0;
    std::unique_ptr<Previews> previews_;
    std::vector<uint32_t> textures_;
    std::map<int, uint32_t> row_texture_;                 // node -> its picture after the step
    std::map<int, std::vector<uint32_t>> row_draws_;      // per-batch node -> its draws
    uint32_t sample_texture_ = 0;
    int open_settings_ = -1;                               // the node whose settings show
    ::gui::Properties::SettingsRows settings_;
    std::string settings_key_;
    std::string message_;
    bool message_error_ = false;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
