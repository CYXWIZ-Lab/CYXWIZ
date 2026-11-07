#include "plugin_ui_kit.h"

#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/file_dialogs.h"

#include <imgui.h>

namespace cyxwiz::ui {
namespace {

plugin::UiColor ToUi(const ImVec4& c) { return {c.x, c.y, c.z, c.w}; }
ImVec4 ToImGui(const plugin::UiColor& c) { return {c.r, c.g, c.b, c.a}; }
ButtonSize ToSize(plugin::UiButtonSize size) {
    return size == plugin::UiButtonSize::Small ? ButtonSize::Small : ButtonSize::Regular;
}

class EnginePluginUi final : public plugin::PluginUi {
public:
    plugin::UiColors Colors() const override {
        const Tokens& t = CurrentTokens();
        plugin::UiColors c;
        c.text = ToUi(t.text);
        c.text_dim = ToUi(t.text_dim);
        c.text_bright = ToUi(t.text_bright);
        c.bg_window = ToUi(t.bg_window);
        c.bg_panel = ToUi(t.bg_panel);
        c.bg_raised = ToUi(t.bg_raised);
        c.bg_input = ToUi(t.bg_input);
        c.accent = ToUi(t.accent);
        c.accent_text = ToUi(t.accent_text);
        c.success = ToUi(t.success);
        c.warning = ToUi(t.warning);
        c.caution = ToUi(t.caution);
        c.error = ToUi(t.error);
        c.pending = ToUi(t.pending);
        c.running = ToUi(t.running);
        return c;
    }

    bool PrimaryButton(const char* label, bool enabled, const char* disabled_reason, plugin::UiButtonSize size,
                       float width) override {
        return ui::PrimaryButton(label, enabled, disabled_reason, ToSize(size), width);
    }
    bool SecondaryButton(const char* label, bool enabled, const char* disabled_reason, plugin::UiButtonSize size,
                         float width) override {
        return ui::SecondaryButton(label, enabled, disabled_reason, ToSize(size), width);
    }
    bool LinkButton(const char* label, bool enabled) override { return ui::LinkButton(label, enabled); }
    bool DangerButton(const char* label, bool enabled, const char* disabled_reason,
                      plugin::UiButtonSize size) override {
        return ui::DangerButton(label, enabled, disabled_reason, ToSize(size));
    }
    float ButtonWidth(const char* label, plugin::UiButtonSize size) const override {
        return ui::ButtonWidth(label, ToSize(size));
    }
    float ButtonHeight(plugin::UiButtonSize size) const override { return ui::ButtonHeight(ToSize(size)); }

    bool SegmentedControl(const char* id, const char* const* labels, int count, int* selected) override {
        return ui::SegmentedControl(id, labels, count, selected);
    }
    float SegmentedControlWidth(const char* const* labels, int count) const override {
        return ui::SegmentedControlWidth(labels, count);
    }

    bool StatusPill(const char* id, const char* text, const plugin::UiColor& dot) override {
        return ui::StatusPill(id, text, ToImGui(dot));
    }
    float StatusPillWidth(const char* text) const override { return ui::StatusPillWidth(text); }

    bool NumberField(const char* id, double* value, double min, double max, double step, const char* format,
                     float width, const char* hint) override {
        NumberFieldOptions options;
        options.min = min;
        options.max = max;
        options.step = step;
        options.format = format;
        options.width = width;
        options.hint = hint;
        return ui::NumberField(id, value, options);
    }

    bool SearchField(const char* id, char* buffer, std::size_t size, const char* hint, float width) override {
        return ui::SearchField(id, buffer, size, hint, width);
    }

    void EmptyState(const char* title, const char* hint) override { ui::EmptyState("", title, hint); }

    std::string OpenFileDialog(const std::string& title, const std::string& filter_name,
                               const std::string& extensions) override {
        FileDialogs::FilterList filters;
        if (!extensions.empty()) filters.emplace_back(filter_name, extensions);
        filters.emplace_back("All Files", "*");
        return FileDialogs::OpenFile(title.c_str(), filters).value_or(std::string());
    }
};

}  // namespace

plugin::PluginUi& PluginUiKit() {
    static EnginePluginUi kit;
    return kit;
}

}  // namespace cyxwiz::ui
