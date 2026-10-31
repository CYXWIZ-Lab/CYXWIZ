// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
//
// A plugin's dataset file formats (plugin API 1.3). Each loader names a
// format and its file extensions; Data Input lists it beside CSV and Parquet,
// and opens a file of that format through OpenDataset, which hands the rows
// over as an Arrow C stream of record batches. From there the dataset is like
// any other: Data Studio, SQL, training and remote training use it.
#pragma once

#include "arrow_c_stream.h"

#include <string>
#include <vector>

namespace cyxwiz::plugin {

struct PluginDataLoaderInfo {
    std::string loader_id;               // Unique across plugins, e.g. "com.acme.formats.dicom"
    std::string format_name;             // Shown in Data Input, e.g. "DICOM series"
    std::vector<std::string> extensions; // Without the dot, e.g. {"dcm", "dicom"}
    std::string description;
};

class IDataProvider {
public:
    virtual ~IDataProvider() = default;

    virtual std::vector<PluginDataLoaderInfo> GetLoaders() = 0;

    // Whether loader `loader_id` reads this file (it may look inside it).
    // Called before OpenDataset when the extension matches.
    virtual bool CanLoad(const std::string& loader_id, const std::string& file_path) = 0;

    // Opens the file. On success fill *out and return true: the Engine reads
    // the stream on a worker thread, maybe after OpenDataset returned, and
    // calls out->release once. On failure return false with `error` set (shown
    // to the user) and leave *out untouched. Streams larger than memory are
    // fine: the Engine writes big datasets to the project's Parquet cache.
    virtual bool OpenDataset(const std::string& loader_id, const std::string& file_path,
                             ArrowArrayStream* out, std::string& error) = 0;
};

} // namespace cyxwiz::plugin
