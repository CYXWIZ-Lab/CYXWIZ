#include "core/data_convert_formats.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>

namespace dc = cyxwiz::data_convert;
namespace {
int checks = 0;
void Check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
}

int main() try {
    std::set<std::string> aliases;
    for (const auto& info : dc::kFormats) {
        Check(dc::Find(info.format) == &info, "format identity must be unique");
        Check(dc::Find(info.name) == &info, "canonical name round trip");
        Check(dc::ExtensionMatches(info.default_extension, info.format), "default extension must be valid");
        for (const char* alias : info.aliases) {
            if (!alias) continue;
            Check(aliases.insert(alias).second, "alias must have exactly one owner");
            Check(dc::FromName(std::string("  ") + alias + "\t") == info.format, "whitespace normalization");
        }
        for (const char* ext : info.extensions)
            if (ext) Check(dc::ExtensionMatches(std::string(".") + ext, info.format), "extension round trip");
    }
    Check(dc::FromName("XLSX") == dc::Format::Excel, "case normalization");
    Check(dc::Resolve(" auto ", ".XLSX") == dc::Format::Excel, "auto path resolution");
    Check(dc::Resolve("csv", ".txt") == dc::Format::Csv, "explicit format takes precedence");
    Check(dc::Resolve("unsupported", ".csv") == dc::Format::Unknown, "unsupported is not silently auto");
    Check(dc::FromName("xls") == dc::Format::Unknown, "legacy Excel remains unsupported");
    Check(!dc::ExtensionMatches(".arrowipc", dc::Format::ArrowIpc), "parameter alias is not an export extension");
    for (bool xlsx : {false, true}) for (bool hdf5 : {false, true}) {
        const dc::Features features{xlsx, hdf5};
        for (auto direction : {dc::Direction::Input, dc::Direction::Output}) {
            const auto allowed = dc::AllowedNames(direction, features);
            const auto properties = dc::PropertyChoices(direction, features);
            Check(properties == std::vector<std::string>(allowed.begin(), allowed.end()), "property/runtime choices agree");
            const auto filter = dc::ExtensionFilter(direction, features);
            Check(properties.front() == "auto", "auto remains available");
            for (const auto& info : dc::kFormats) {
                const bool available = dc::Available(info.format, direction, features);
                for (const char* alias : info.aliases) if (alias)
                    Check((std::find(properties.begin(), properties.end(), alias) != properties.end()) == available,
                          "every alias must obey adapter availability and direction");
            }
            Check((filter.find("xlsx") != std::string::npos) == (xlsx && direction == dc::Direction::Input),
                  "XLSX is input-only and build-dependent");
            Check((filter.find("hdf5") != std::string::npos) == hdf5, "HDF5 picker obeys build availability");
            Check(!dc::Available(dc::Format::Unknown, direction, features), "unknown format fails closed");
        }
    }
    const auto built = dc::PropertyChoices(dc::Direction::Input, dc::kBuildFeatures);
    Check((std::find(built.begin(), built.end(), "xlsx") != built.end()) == dc::kBuildFeatures.xlsx,
          "compiled feature selection");
    std::cout << "DataConvert format contract: " << checks << " checks passed; XLSX "
              << dc::kBuildFeatures.xlsx << "; HDF5 " << dc::kBuildFeatures.hdf5 << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
