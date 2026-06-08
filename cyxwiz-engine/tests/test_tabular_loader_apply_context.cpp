#include "../src/gui/loaders/tabular_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("csv"),
          "CSV should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("parquet"),
          "Parquet should remain an accepted tabular loader type");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("json"),
          "JSON should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("json").find(
              "JSON loading is not supported") != std::string::npos,
          "JSON failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("excel"),
          "Excel should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("excel").find(
              "Excel loading is not supported") != std::string::npos,
          "Excel failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("JSON"),
          "JSON validation should be case-insensitive");

    return 0;
}
