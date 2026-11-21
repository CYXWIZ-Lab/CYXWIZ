#include "python_package_list.h"

#include <algorithm>

namespace cyxwiz::core {

std::vector<std::string> ParsePackageList(const std::string& text, std::vector<std::string>* ignored) {
    std::vector<std::string> packages;
    std::string word;
    const auto take = [&]() {
        if (word.empty()) return;
        if (word.front() == '-') {
            if (ignored) ignored->push_back(word);
        } else if (std::find(packages.begin(), packages.end(), word) == packages.end()) {
            packages.push_back(word);
        }
        word.clear();
    };
    for (const char c : text) {
        if (c == ' ' || c == ',' || c == '\t' || c == '\n' || c == '\r')
            take();
        else
            word += c;
    }
    take();
    return packages;
}

std::string JoinPackageList(const std::vector<std::string>& packages) {
    std::string text;
    for (const auto& p : packages) {
        if (!text.empty()) text += ' ';
        text += p;
    }
    return text;
}

std::vector<std::string> PipInstallCommand(const std::string& python, const std::vector<std::string>& packages) {
    std::vector<std::string> args = {python, "-m", "pip", "install", "--disable-pip-version-check"};
    args.insert(args.end(), packages.begin(), packages.end());
    return args;
}

}  // namespace cyxwiz::core
