#pragma once

#include <string>
#include <functional>

namespace cyxwiz {

/**
 * Model Converter - Convert between .cyxmodel formats
 *
 * Binary format (single file):
 *   Magic: "CYXW" (0x43595857)
 *   Version: 2
 *   JSON metadata + serialized tensors
 *   Created by: Tools > Save Trained Model
 *
 * Directory format:
 *   model.cyxmodel/
 *   ├── manifest.json
 *   ├── graph.cyxgraph
 *   ├── config.json
 *   ├── weights/manifest.json
 *   └── weights/*.bin
 *   Created by: Deploy > Export Model
 */
class ModelConverter {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& message)>;

    /**
     * Check if a path is binary format (.cyxmodel file with CYXW magic)
     */
    static bool IsBinaryFormat(const std::string& path);

    /**
     * Check if a path is directory format (.cyxmodel folder with manifest.json)
     */
    static bool IsDirectoryFormat(const std::string& path);

    /**
     * Convert binary format to directory format
     * @param input_path Path to binary .cyxmodel file
     * @param output_path Path for output directory (will be created)
     * @param progress_cb Optional progress callback
     * @return true on success
     */
    static bool BinaryToDirectory(
        const std::string& input_path,
        const std::string& output_path,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Convert directory format to binary format
     * @param input_path Path to .cyxmodel directory
     * @param output_path Path for output binary file
     * @param progress_cb Optional progress callback
     * @return true on success
     */
    static bool DirectoryToBinary(
        const std::string& input_path,
        const std::string& output_path,
        ProgressCallback progress_cb = nullptr
    );

    /**
     * Get last error message
     */
    static std::string GetLastError() { return last_error_; }

private:
    static std::string last_error_;
};

} // namespace cyxwiz
