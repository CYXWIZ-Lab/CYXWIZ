#include "properties_executor.h"
#include "node_editor.h"
#include "../core/node_executors/node_executor_factory.h"
#include <imgui.h>
#include <string>

namespace gui::properties_executor {

namespace {

bool HasNodeExecutor(NodeType type) {
    return cyxwiz::NodeExecutorFactory::Instance().HasExecutor(type);
}

void SetupExecutorInputData(NodeEditor* node_editor, cyxwiz::INodeExecutor* executor, MLNode& node) {
    if (!node_editor) return;

    // TODO: Get input data from connected upstream nodes.
    // Placeholder until executor inputs are sourced from graph outputs.
    (void)executor;
    (void)node;
}

} // namespace

void RenderExecutorSection(NodeEditor* node_editor, MLNode& node) {
    if (!HasNodeExecutor(node.type)) {
        return;
    }

    auto* executor = cyxwiz::NodeExecutorFactory::Instance().GetExecutor(node.type);
    if (!executor) return;

    ImGui::Separator();

    bool executor_open = ImGui::CollapsingHeader(
        (std::string(executor->GetName()) + " Configuration").c_str(),
        ImGuiTreeNodeFlags_DefaultOpen
    );

    if (executor_open) {
        ImGui::Indent(10.0f);

        SetupExecutorInputData(node_editor, executor, node);

        ImGui::PushID("ExecutorConfig");
        executor->RenderConfigUI();
        ImGui::PopID();

        ImGui::Unindent(10.0f);
    }

    if (executor->GetState() == cyxwiz::ExecutorState::Completed ||
        executor->GetState() == cyxwiz::ExecutorState::Executing ||
        executor->GetState() == cyxwiz::ExecutorState::Error) {

        bool results_open = ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen);

        if (results_open) {
            ImGui::Indent(10.0f);

            ImGui::PushID("ExecutorResults");
            executor->RenderResultsUI();
            ImGui::PopID();

            ImGui::Unindent(10.0f);
        }
    }
}

} // namespace gui::properties_executor
