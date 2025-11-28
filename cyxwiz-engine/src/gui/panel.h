#pragma once

#include <string>

namespace cyxwiz {

/**
 * Base class for all GUI panels in CyxWiz Engine
 * Provides common interface for dockable panels
 */
class Panel {
public:
    Panel(const std::string& name, bool visible = true)
        : name_(name), visible_(visible), focused_(false) {}

    virtual ~Panel() = default;

    // Main render function - called every frame if visible
    virtual void Render() = 0;

    // Panel metadata
    virtual const char* GetName() const { return name_.c_str(); }
    virtual const char* GetIcon() const { return ""; }
    void SetName(const std::string& name) { name_ = name; }

    // Visibility control
    bool IsVisible() const { return visible_; }
    void SetVisible(bool visible) { visible_ = visible; }
    void Hide() { visible_ = false; }
    void Show() { visible_ = true; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }  // For sidebar toggle binding

    // Focus tracking
    bool IsFocused() const { return focused_; }
    void SetFocused(bool focused) { focused_ = focused; }

    // Keyboard shortcuts (override in derived classes)
    virtual void HandleKeyboardShortcuts() {}

protected:
    std::string name_;
    bool visible_;
    bool focused_;
};

} // namespace cyxwiz
