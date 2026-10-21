#pragma once

#include "graph_compiler.h"

#include <string>

namespace cyxwiz {

enum class TestDatasetScope {
    ConfiguredTestSplit,
    EntireProvidedDataset,
};

struct GraphTestDatasetSelection {
    std::string dataset_name;
    std::string label_column;
    int source_node_id = -1;
    TestDatasetScope scope = TestDatasetScope::ConfiguredTestSplit;
};

inline GraphTestDatasetSelection ResolveGraphTestDataset(
    const TrainingConfiguration& config) {
    if (config.dataset_roles.test.IsSupplied()) {
        return {
            config.dataset_roles.test.dataset_name,
            config.dataset_roles.test.label_column,
            config.dataset_roles.test.source_node_id,
            TestDatasetScope::EntireProvidedDataset,
        };
    }

    if (config.dataset_roles.train.IsSupplied()) {
        return {
            config.dataset_roles.train.dataset_name,
            config.dataset_roles.train.label_column,
            config.dataset_roles.train.source_node_id,
            TestDatasetScope::ConfiguredTestSplit,
        };
    }

    return {
        config.dataset_name,
        {},
        config.data_source_node_id,
        TestDatasetScope::ConfiguredTestSplit,
    };
}

inline TrainingConfiguration ConfigureTestDatasetScope(
    TrainingConfiguration config,
    TestDatasetScope scope) {
    config.prefetch_factor = 0;
    if (scope != TestDatasetScope::EntireProvidedDataset) {
        return config;
    }

    // A dataset explicitly assigned to Test is already a semantic role, not a
    // source that should be partitioned again. Route every row through the
    // train phase internally because that phase has whole-dataset semantics.
    config.has_data_split = true;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.stratified = false;
    config.shuffle = false;
    config.drop_last = false;
    config.balance_classes = false;
    return config;
}

} // namespace cyxwiz
