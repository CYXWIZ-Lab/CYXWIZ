#include "../src/core/arrow_dataset.h"
#include "../src/core/training_batcher_setup.h"

#include <arrow/api.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishIntArray(const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    for (int32_t value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::int32()),
    });
    auto table = arrow::Table::Make(schema, {
        FinishFloatArray({1.0f, 2.0f, 3.0f, 4.0f}),
        FinishFloatArray({10.0f, 20.0f, 30.0f, 40.0f}),
        FinishIntArray({0, 1, 0, 1}),
    }, 4);
    return std::make_shared<cyxwiz::ArrowDataset>(table, "batcher_setup");
}

cyxwiz::TrainingConfiguration MakeConfig() {
    cyxwiz::TrainingConfiguration config;
    config.output_size = 2;
    config.train_ratio = 0.75f;
    config.shuffle = false;
    config.num_workers = 0;
    return config;
}

} // namespace

int main() {
    auto batchers = cyxwiz::BuildArrowTrainingBatchers(
        MakeConfig(),
        MakeDataset(),
        "label",
        /*batch_size=*/2);

    Check(batchers.arrow_train != nullptr, "train Arrow batcher should be owned");
    Check(batchers.arrow_val != nullptr, "val Arrow batcher should be owned");
    Check(batchers.train == batchers.arrow_train.get(), "train pointer should target train owner");
    Check(batchers.val == batchers.arrow_val.get(), "val pointer should target val owner");
    Check(batchers.num_train_samples == 3, "train split should contain 3 samples");

    auto batch = batchers.train->GetNextBatch();
    Check(batch.IsValid(), "train batch should be valid");
    Check(batch.data.Shape().size() == 2, "feature tensor should be 2D");
    Check(batch.data.Shape()[0] == 2, "batch dimension should be 2");
    Check(batch.data.Shape()[1] == 2, "feature width should be 2");
    Check(batch.labels.Shape().size() == 2, "label tensor should be 2D");
    Check(batch.labels.Shape()[1] == 2, "labels should be one-hot by output_size");

    std::cout << "Training batcher setup passed\n";
    return 0;
}
