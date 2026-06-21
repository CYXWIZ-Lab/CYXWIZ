#!/usr/bin/env python3
with open('cyxwiz-engine/src/gui/node_editor.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''        ImGui::EndPopup();
    }
}

CanvasAnnotation* NodeEditor::FindAnnotationById(int annotation_id) {'''

new = '''        ImGui::EndPopup();
    }
}

void NodeEditor::ShowNodeDescriptionEditPopup() {
    if (editing_node_description_) {
        ImGui::OpenPopup("Edit Node Description");
    }

    if (ImGui::BeginPopupModal("Edit Node Description", &editing_node_description_, ImGuiWindowFlags_AlwaysAutoResize)) {
        MLNode* node = FindNodeById(editing_node_id_);
        if (node) {
            ImGui::Text("Node: %s", node->name.c_str());
            ImGui::Separator();

            ImGui::Text("Description (shown below node on canvas):");
            ImGui::InputTextMultiline("##node_desc", node_description_buffer_, sizeof(node_description_buffer_),
                                      ImVec2(350.0f, 100.0f));

            ImGui::Separator();

            if (ImGui::Button("Save", ImVec2(100.0f, 0.0f))) {
                node->description = node_description_buffer_;
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear", ImVec2(100.0f, 0.0f))) {
                node_description_buffer_[0] = '\\0';
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f))) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        } else {
            ImGui::Text("Node not found");
            if (ImGui::Button("Close")) {
                editing_node_description_ = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
}

CanvasAnnotation* NodeEditor::FindAnnotationById(int annotation_id) {'''

if old in content:
    content = content.replace(old, new)
    with open('cyxwiz-engine/src/gui/node_editor.cpp', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Added ShowNodeDescriptionEditPopup!')
else:
    print('Pattern not found!')
