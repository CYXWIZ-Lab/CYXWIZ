// Packages for new project environments (Preferences > Python, TOFIX129 B9-9)
#include "../src/core/python_package_list.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cyxwiz::core;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}  // namespace

int main() {
    using List = std::vector<std::string>;
    Check(ParsePackageList("numpy pandas") == List{"numpy", "pandas"}, "spaces");
    Check(ParsePackageList("numpy, torch,scikit-learn") == List{"numpy", "torch", "scikit-learn"},
          "commas, as the setting was typed before");
    Check(ParsePackageList("  numpy \n numpy  ") == List{"numpy"}, "blanks and repeats");
    Check(ParsePackageList("").empty() && ParsePackageList(" , ").empty(), "empty: none");
    Check(ParsePackageList("numpy==2.1 \"pandas>=2\"") == List{"numpy==2.1", "\"pandas>=2\""}, "versions stay with the name");
    List ignored;
    Check(ParsePackageList("numpy --index-url http://x -e .", &ignored) == List{"numpy", "http://x", "."} &&
              ignored == List{"--index-url", "-e"},
          "pip options never reach pip");
    Check(JoinPackageList({"numpy", "pandas"}) == "numpy pandas" && JoinPackageList({}).empty(), "join");
    const List cmd = PipInstallCommand("C:/p/python/Scripts/python.exe", {"numpy", "pandas"});
    Check(cmd == List{"C:/p/python/Scripts/python.exe", "-m", "pip", "install", "--disable-pip-version-check", "numpy",
                      "pandas"},
          "pip command");
    std::cout << "python package list: parse, options kept out, join, pip command. OK\n";
    return 0;
}
