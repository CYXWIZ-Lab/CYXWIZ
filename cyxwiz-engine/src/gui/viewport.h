#pragma once

namespace gui {

class Viewport {
public:
    Viewport();
    ~Viewport();

    void Render();

private:
    bool show_window_;
};

} // namespace gui
