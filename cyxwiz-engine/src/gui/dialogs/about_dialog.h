#pragma once

// Help > About CyxWiz (TOFIX129 A2-7): version, build, commit, compute
// devices, Python, licence and links, and "Copy details" for bug reports.
// The details are gathered once when the dialog opens.

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

class AboutDialog {
public:
    void Open();
    void Render();

private:
    void Gather();

    bool request_open_ = false;
    std::vector<std::pair<std::string, std::string>> rows_;
    std::string copied_note_;
};

}  // namespace cyxwiz
