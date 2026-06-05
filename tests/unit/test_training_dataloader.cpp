#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/data_loader.h>
#include <cyxwiz/dataloader.h>
#include <memory>
#include <type_traits>

TEST_CASE("Tabular and training data loaders have distinct public types", "[training_dataloader]") {
    static_assert(!std::is_same_v<cyxwiz::DataLoader, cyxwiz::TrainingDataLoader>);

    REQUIRE(cyxwiz::DataLoader::IsAvailable() == cyxwiz::DataLoader::IsAvailable());
}

TEST_CASE("TrainingDataLoader batches synthetic dataset", "[training_dataloader]") {
    auto dataset = std::make_shared<cyxwiz::SyntheticDataset>(
        5, std::vector<size_t>{2}, 3, 7);

    cyxwiz::TrainingDataLoader loader(dataset, 2, false);

    REQUIRE(loader.NumSamples() == 5);
    REQUIRE(loader.NumBatches() == 3);
    REQUIRE(loader.GetBatchSize() == 2);
    REQUIRE(loader.GetShape() == std::vector<size_t>{2});
    REQUIRE(loader.NumClasses() == 3);

    cyxwiz::DataBatch first = loader.GetNextBatch();
    REQUIRE(first.IsValid());
    REQUIRE(first.size == 2);
    REQUIRE(first.data.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(first.labels.Shape() == std::vector<size_t>{2});
    REQUIRE(first.labels.GetDataType() == cyxwiz::DataType::Int32);

    cyxwiz::DataBatch second = loader.GetNextBatch();
    REQUIRE(second.IsValid());
    REQUIRE(second.size == 2);

    cyxwiz::DataBatch third = loader.GetNextBatch();
    REQUIRE(third.IsValid());
    REQUIRE(third.size == 1);
    REQUIRE(loader.IsEpochComplete());

    cyxwiz::DataBatch empty = loader.GetNextBatch();
    REQUIRE(!empty.IsValid());
}

TEST_CASE("CreateTrainingDataLoader batches tensor-backed data", "[training_dataloader]") {
    const float data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    const int32_t labels[] = {0, 1, 2};

    cyxwiz::Tensor data_tensor({3, 2}, data, cyxwiz::DataType::Float32);
    cyxwiz::Tensor label_tensor({3}, labels, cyxwiz::DataType::Int32);

    cyxwiz::TrainingDataLoader loader =
        cyxwiz::CreateTrainingDataLoader(data_tensor, label_tensor, 2, false);

    cyxwiz::DataBatch first = loader.GetNextBatch();
    REQUIRE(first.IsValid());
    REQUIRE(first.size == 2);
    REQUIRE(first.data.Shape() == std::vector<size_t>{2, 2});
    REQUIRE(first.labels.Shape() == std::vector<size_t>{2});

    const float* first_data = first.data.Data<float>();
    REQUIRE(first_data[0] == 1.0f);
    REQUIRE(first_data[3] == 4.0f);

    const int32_t* first_labels = first.labels.Data<int32_t>();
    REQUIRE(first_labels[0] == 0);
    REQUIRE(first_labels[1] == 1);
}
