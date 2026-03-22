/**
 * Test HDF5 Dataset Loading
 *
 * Tests the HDF5Dataset class with various data formats
 */

#include "../src/core/data_registry.h"
#include <iostream>
#include <cassert>

using namespace cyxwiz;

void test_image_hdf5() {
    std::cout << "\n=== Testing Image HDF5 (test_hdf5.h5) ===" << std::endl;

    auto& registry = DataRegistry::Instance();

    // Load with auto-detection
    auto handle = registry.LoadHDF5("test_data/test_hdf5.h5", "test_images");

    if (!handle.IsValid()) {
        std::cerr << "FAILED: Could not load test_hdf5.h5" << std::endl;
        return;
    }

    auto info = handle.GetInfo();
    std::cout << "  Name: " << info.name << std::endl;
    std::cout << "  Samples: " << info.num_samples << std::endl;
    std::cout << "  Shape: " << info.GetShapeString() << std::endl;
    std::cout << "  Classes: " << info.num_classes << std::endl;
    std::cout << "  Memory: " << info.GetMemoryString() << std::endl;

    // Verify expected values
    assert(info.num_samples == 100);
    assert(info.shape.size() == 3);
    assert(info.shape[0] == 32);  // H
    assert(info.shape[1] == 32);  // W
    assert(info.shape[2] == 3);   // C

    // Test GetItem
    auto [sample, label] = handle.GetSample(0);
    assert(sample.size() == 32 * 32 * 3);
    assert(label >= 0 && label < 10);

    // Check normalization (uint8 -> [0,1])
    bool all_normalized = true;
    for (float val : sample) {
        if (val < 0.0f || val > 1.0f) {
            all_normalized = false;
            break;
        }
    }
    assert(all_normalized);

    std::cout << "  First sample label: " << label << std::endl;
    std::cout << "  Sample size: " << sample.size() << std::endl;
    std::cout << "  Values normalized: " << (all_normalized ? "YES" : "NO") << std::endl;

    std::cout << "PASSED: Image HDF5 test" << std::endl;

    registry.UnloadDataset("test_images");
}

void test_tabular_hdf5() {
    std::cout << "\n=== Testing Tabular HDF5 (test_tabular.h5) ===" << std::endl;

    auto& registry = DataRegistry::Instance();

    // Load with auto-detection
    auto handle = registry.LoadHDF5("test_data/test_tabular.h5", "test_tabular");

    if (!handle.IsValid()) {
        std::cerr << "FAILED: Could not load test_tabular.h5" << std::endl;
        return;
    }

    auto info = handle.GetInfo();
    std::cout << "  Name: " << info.name << std::endl;
    std::cout << "  Samples: " << info.num_samples << std::endl;
    std::cout << "  Shape: " << info.GetShapeString() << std::endl;
    std::cout << "  Classes: " << info.num_classes << std::endl;

    // Verify expected values
    assert(info.num_samples == 500);
    assert(info.shape.size() == 1);
    assert(info.shape[0] == 10);  // 10 features

    // Test GetItem
    auto [sample, label] = handle.GetSample(0);
    assert(sample.size() == 10);
    assert(label >= 0 && label < 3);

    std::cout << "  First sample label: " << label << std::endl;
    std::cout << "  Sample values: [";
    for (size_t i = 0; i < std::min(sample.size(), size_t(5)); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << sample[i];
    }
    std::cout << ", ...]" << std::endl;

    std::cout << "PASSED: Tabular HDF5 test" << std::endl;

    registry.UnloadDataset("test_tabular");
}

void test_batch_loading() {
    std::cout << "\n=== Testing Batch Loading ===" << std::endl;

    auto& registry = DataRegistry::Instance();
    auto handle = registry.LoadHDF5("test_data/test_hdf5.h5", "batch_test");

    if (!handle.IsValid()) {
        std::cerr << "FAILED: Could not load for batch test" << std::endl;
        return;
    }

    // Get batch of 10 samples
    std::vector<size_t> indices = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45};
    auto [samples, labels] = handle.GetBatch(indices);

    assert(samples.size() == 10);
    assert(labels.size() == 10);

    std::cout << "  Batch size: " << samples.size() << std::endl;
    std::cout << "  Each sample size: " << samples[0].size() << std::endl;

    std::cout << "PASSED: Batch loading test" << std::endl;

    registry.UnloadDataset("batch_test");
}

void test_split_config() {
    std::cout << "\n=== Testing Split Configuration ===" << std::endl;

    auto& registry = DataRegistry::Instance();
    auto handle = registry.LoadHDF5("test_data/test_hdf5.h5", "split_test");

    if (!handle.IsValid()) {
        std::cerr << "FAILED: Could not load for split test" << std::endl;
        return;
    }

    // Apply 70/15/15 split
    SplitConfig config;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.shuffle = true;
    config.seed = 42;

    handle.ApplySplit(config);

    auto& train_idx = handle.GetTrainIndices();
    auto& val_idx = handle.GetValIndices();
    auto& test_idx = handle.GetTestIndices();

    std::cout << "  Train samples: " << train_idx.size() << std::endl;
    std::cout << "  Val samples: " << val_idx.size() << std::endl;
    std::cout << "  Test samples: " << test_idx.size() << std::endl;

    assert(train_idx.size() == 70);
    assert(val_idx.size() == 15);
    assert(test_idx.size() == 15);

    std::cout << "PASSED: Split configuration test" << std::endl;

    registry.UnloadDataset("split_test");
}

void test_detect_type() {
    std::cout << "\n=== Testing DetectType ===" << std::endl;

    auto type = DataRegistry::DetectType("test_data/test_hdf5.h5");
    std::cout << "  test_hdf5.h5 -> " << DataRegistry::TypeToString(type) << std::endl;
    assert(type == DatasetType::HDF5);

    type = DataRegistry::DetectType("test.hdf5");
    std::cout << "  test.hdf5 -> " << DataRegistry::TypeToString(type) << std::endl;
    // Note: file doesn't exist, so returns None

    std::cout << "PASSED: DetectType test" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  HDF5 Dataset Loading Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_detect_type();
        test_image_hdf5();
        test_tabular_hdf5();
        test_batch_loading();
        test_split_config();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
