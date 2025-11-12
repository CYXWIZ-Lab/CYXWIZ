#pragma once

namespace gui {

class Properties {
public:
    Properties();
    ~Properties();

    void Render();

private:
    bool show_window_;

    // Example properties (TODO: replace with actual node/layer properties)
    int units_ = 64;
    float learning_rate_ = 0.001f;
};

} // namespace gui
