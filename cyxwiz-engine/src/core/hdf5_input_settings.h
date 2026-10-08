#pragma once

#include "hdf5_table_adapter.h"

#include <cstdint>
#include <map>
#include <string>

namespace cyxwiz {

struct Hdf5InputSettings {
    Hdf5TableSelection selection;
    Hdf5NumericPolicy numeric_policy = Hdf5NumericPolicy::Preserve;
    uint64_t max_materialized_bytes = 256ULL * 1024 * 1024;
};

struct Hdf5InputSettingsResult {
    bool ok = false;
    Hdf5InputSettings settings;
    std::string error;
    bool migrated_legacy = false;
};

bool ValidateHdf5InputSettings(const Hdf5InputSettings& settings,
                               std::string& error);

Hdf5InputSettingsResult ReadHdf5InputSettings(
    const std::map<std::string, std::string>& parameters);

bool WriteHdf5InputSettings(const Hdf5InputSettings& settings,
                            std::map<std::string, std::string>& parameters,
                            std::string& error);

} // namespace cyxwiz
