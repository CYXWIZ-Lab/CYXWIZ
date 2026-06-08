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
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("tsv"),
          "TSV should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("parquet"),
          "Parquet should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("feather"),
          "Feather should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("arrow"),
          "Arrow should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("ipc"),
          "IPC should remain an accepted tabular loader type");

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

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("hdf5"),
          "HDF5 should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("hdf5").find(
              "HDF5 loading is not supported") != std::string::npos,
          "HDF5 failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("txt"),
          "TXT should fail closed on tabular path");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("txt").find(
              "use Text source") != std::string::npos,
          "TXT failure should point to Text source");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("arff"),
          "ARFF should fail closed before async tabular load");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("JSON"),
          "JSON validation should be case-insensitive");

    return 0;
}
