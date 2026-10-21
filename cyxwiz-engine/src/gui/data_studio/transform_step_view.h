#pragma once

// Data Studio > Prepare: the view of a scale, power or log step (TOFIX129
// D2, approved mockup d2_transform_mockup.html). Its settings as typed fields
// and switches written to the node (with an undo point); the Method switch
// replaces a scaler node with another, keeping its links and columns; the
// last run's before / after per column with two histograms of the picked
// column; a power step's lambda, likelihood curve, skewness and normality
// test; Compare methods (one column through every scaling method, preview
// only, "Use this method" switches). Figures are computed off the UI thread
// (transform_step_presentation, the backend's DataTransform).

#include "../../core/data_studio_presentation.h"
#include "../../core/transform_step_presentation.h"
#include "explore_tab.h"

#include <functional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class TransformStepView {
public:
    // `input`: the node feeding the step (null when none).
    void Render(::gui::NodeEditor* editor, const studio::GraphNode& node, const studio::GraphNode* input);

private:
    void RenderHeader(const studio::GraphNode& node, const studio::GraphNode* input,
                      const std::map<std::string, std::string>& params);
    void RenderSettings(const studio::GraphNode& node, const std::map<std::string, std::string>& params);
    void RenderColumns(const std::map<std::string, std::string>& params);
    void RenderFit(const std::map<std::string, std::string>& params);
    void RenderFigures(const studio::GraphNode& node, const transform_step::StepFigures& figures);
    void RenderPowerFigures(const transform_step::StepFigures& figures);
    void RenderCompare(const studio::GraphNode& node, const studio::GraphNode* input);
    void RenderAllSettings(const std::map<std::string, std::string>& params);
    void Set(const std::string& key, const std::string& value);
    void AskFigures(const studio::GraphNode& node, const studio::GraphNode* input,
                    const std::map<std::string, std::string>& params);
    const transform_step::ColumnFigures* Picked(const transform_step::StepFigures& figures);

    ::gui::NodeEditor* editor_ = nullptr;
    int step_ = -1;
    ExploreJob<transform_step::StepFigures> figures_;
    std::string figures_key_;
    ExploreJob<transform_step::CompareFigures> compare_;
    bool comparing_ = false;
    std::string compare_column_;
    std::string column_;  // the picked column (histograms, curve)
    // The input's number columns (Pick columns), by dataset.
    std::string schema_of_;
    std::vector<std::string> number_columns_;
    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
