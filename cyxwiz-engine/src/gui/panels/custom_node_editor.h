#pragma once

#include "../panel.h"
#include "../node_editor.h"
#include "../patterns/pattern_library.h"
#include <string>
#include <vector>
#include <memory>

namespace gui {

/**
 * Custom Node Editor Panel
 * Allows users to create custom node types with configurable pins, parameters, and code templates.
 * Custom nodes are saved as patterns and registered with the PatternLibrary.
 */
class CustomNodeEditorPanel : public cyxwiz::Panel {
public:
    // Pin definition for custom nodes
    struct PinDefinition {
        std::string name;
        PinType type = PinType::Tensor;
        std::string description;
        std::string default_shape;  // e.g., "[batch, features]"
        bool is_optional = false;
    };

    // Parameter definition for custom nodes
    struct ParameterDefinition {
        std::string name;
        std::string type;  // "int", "float", "string", "bool", "choice"
        std::string default_value;
        std::string description;
        std::string min_value;
        std::string max_value;
        std::vector<std::string> choices;  // For choice type
    };

    // Complete custom node definition
    struct CustomNodeDefinition {
        std::string id;
        std::string name;
        std::string category;
        std::string description;
        std::string icon;
        ImVec4 color = ImVec4(0.4f, 0.4f, 0.8f, 1.0f);
        std::vector<PinDefinition> inputs;
        std::vector<PinDefinition> outputs;
        std::vector<ParameterDefinition> parameters;
        std::string pytorch_template;
        std::string tensorflow_template;
        std::string cyxwiz_template;
        std::string validation_code;  // Optional Python validation
    };

    CustomNodeEditorPanel();
    ~CustomNodeEditorPanel() override = default;

    void Render() override;

    // Create a new empty definition
    void CreateNewDefinition();

    // Load an existing definition for editing
    bool LoadDefinition(const std::string& pattern_id);

    // Save the current definition
    bool SaveDefinition();
    bool SaveDefinitionAs(const std::string& path);

    // Delete a custom node definition
    bool DeleteDefinition(const std::string& pattern_id);

    // Get/set current definition
    CustomNodeDefinition& GetCurrentDefinition() { return current_def_; }
    const CustomNodeDefinition& GetCurrentDefinition() const { return current_def_; }

    // Check if there are unsaved changes
    bool HasUnsavedChanges() const { return has_unsaved_changes_; }

private:
    // Tab rendering functions
    void RenderGeneralTab();
    void RenderPinsTab();
    void RenderParametersTab();
    void RenderCodeTab();
    void RenderPreviewTab();

    // Pin editor helpers
    void RenderPinEditor(std::vector<PinDefinition>& pins, const char* label, bool is_input);
    void RenderPinTypeCombo(PinType& type);

    // Parameter editor helpers
    void RenderParameterEditor();
    void RenderParameterTypeCombo(std::string& type);

    // Code template helpers
    void RenderCodeTemplateEditor(const char* label, std::string& code, const char* language);
    void InsertPlaceholder(std::string& code, const char* placeholder);

    // Preview helpers
    void RenderNodePreview();
    void RenderCodePreview();

    // Validation
    bool ValidateDefinition(std::string& error_message);
    std::string GenerateUniqueId(const std::string& name);

    // Convert to/from pattern format
    patterns::Pattern ConvertToPattern() const;
    void LoadFromPattern(const patterns::Pattern& pattern);

    // Current editing state
    CustomNodeDefinition current_def_;
    bool has_unsaved_changes_ = false;
    std::string loaded_pattern_id_;  // Empty if new

    // UI state
    int current_tab_ = 0;
    int selected_input_pin_ = -1;
    int selected_output_pin_ = -1;
    int selected_parameter_ = -1;
    int code_tab_ = 0;  // 0=PyTorch, 1=TensorFlow, 2=CyxWiz

    // New pin/parameter dialog state
    bool show_new_pin_dialog_ = false;
    bool adding_input_pin_ = true;
    PinDefinition new_pin_;

    bool show_new_param_dialog_ = false;
    ParameterDefinition new_param_;

    // Delete confirmation dialog
    bool show_delete_confirm_ = false;
    std::string delete_target_;  // "pin:input:0", "pin:output:1", "param:2"

    // Save dialog state
    bool show_save_dialog_ = false;
    char save_path_buffer_[512] = "";

    // Pattern categories for dropdown
    static const std::vector<std::pair<std::string, std::string>> kCategoryOptions;

    // Available placeholder documentation
    static const char* kPlaceholderHelp;
};

} // namespace gui
