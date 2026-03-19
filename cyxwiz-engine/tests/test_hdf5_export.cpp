#include <iostream>
#include <filesystem>
#include "../src/core/data_registry.h"

namespace fs = std::filesystem;

int main() {
    std::cout << "=== HDF5 Export Test ===" << std::endl;

    auto& registry = cyxwiz::DataRegistry::Instance();

    // Test file paths
    std::string input_file = "D:/demo/mrcj/data/test_small_hdf5.h5";
    std::string output_file = "D:/demo/mrcj/data/test_export_output.h5";

    // Check input exists
    if (!fs::exists(input_file)) {
        std::cerr << "Input file not found: " << input_file << std::endl;
        return 1;
    }

    // Load the dataset
    std::cout << "\n1. Loading HDF5 dataset..." << std::endl;
    cyxwiz::HDF5Config load_config;
    load_config.lazy_loading = false;  // Load all data for export test

    auto handle = registry.LoadHDF5(input_file, "test_dataset", load_config);
    if (!handle.IsValid()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    auto info = handle.GetInfo();
    std::cout << "   Loaded: " << info.num_samples << " samples" << std::endl;
    std::cout << "   Shape: " << info.GetShapeString() << std::endl;
    std::cout << "   Classes: " << info.num_classes << std::endl;

    // Export with default settings
    std::cout << "\n2. Exporting with default settings..." << std::endl;
    cyxwiz::HDF5ExportConfig export_config;
    export_config.compress = true;
    export_config.compression_level = 4;
    export_config.include_metadata = true;

    bool success = registry.ExportHDF5("test_dataset", output_file, export_config);
    if (!success) {
        std::cerr << "Export failed!" << std::endl;
        return 1;
    }

    // Verify output file exists
    if (!fs::exists(output_file)) {
        std::cerr << "Output file was not created!" << std::endl;
        return 1;
    }

    auto output_size = fs::file_size(output_file);
    auto input_size = fs::file_size(input_file);
    std::cout << "   Input size:  " << input_size << " bytes" << std::endl;
    std::cout << "   Output size: " << output_size << " bytes" << std::endl;
    std::cout << "   Compression ratio: " << (100.0 * output_size / input_size) << "%" << std::endl;

    // Load the exported file to verify
    std::cout << "\n3. Verifying exported file..." << std::endl;
    auto verify_handle = registry.LoadHDF5(output_file, "verify_dataset");
    if (!verify_handle.IsValid()) {
        std::cerr << "Failed to load exported file!" << std::endl;
        return 1;
    }

    auto verify_info = verify_handle.GetInfo();
    std::cout << "   Samples: " << verify_info.num_samples << std::endl;
    std::cout << "   Shape: " << verify_info.GetShapeString() << std::endl;
    std::cout << "   Classes: " << verify_info.num_classes << std::endl;

    // Compare sample counts
    if (verify_info.num_samples != info.num_samples) {
        std::cerr << "Sample count mismatch!" << std::endl;
        return 1;
    }

    // Test with NCHW export
    std::cout << "\n4. Testing NCHW export..." << std::endl;
    std::string nchw_output = "D:/demo/mrcj/data/test_export_nchw.h5";

    cyxwiz::HDF5ExportConfig nchw_config;
    nchw_config.store_as_nchw = true;
    nchw_config.compress = true;
    nchw_config.include_metadata = true;

    success = registry.ExportHDF5("test_dataset", nchw_output, nchw_config);
    if (!success) {
        std::cerr << "NCHW export failed!" << std::endl;
        return 1;
    }
    std::cout << "   NCHW export successful" << std::endl;

    // Test with uint8 export
    std::cout << "\n5. Testing uint8 export..." << std::endl;
    std::string uint8_output = "D:/demo/mrcj/data/test_export_uint8.h5";

    cyxwiz::HDF5ExportConfig uint8_config;
    uint8_config.store_as_uint8 = true;
    uint8_config.compress = true;
    uint8_config.include_metadata = true;

    success = registry.ExportHDF5("test_dataset", uint8_output, uint8_config);
    if (!success) {
        std::cerr << "uint8 export failed!" << std::endl;
        return 1;
    }

    auto uint8_size = fs::file_size(uint8_output);
    std::cout << "   uint8 export size: " << uint8_size << " bytes" << std::endl;
    std::cout << "   Savings vs float32: " << (100.0 - 100.0 * uint8_size / output_size) << "%" << std::endl;

    // Cleanup
    registry.UnloadAll();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
