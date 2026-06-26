#include "core/metric_learning_dataset_builder.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor FloatTensor(const std::vector<size_t>& shape,
                           const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data());
}

cyxwiz::Tensor IntTensor(const std::vector<size_t>& shape,
                         const std::vector<int64_t>& values) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Int64);
}

void TestPairBatchContract() {
    const std::vector<float> left = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const std::vector<float> right = {
        1.1f, 2.1f,
        3.1f, 4.1f,
        5.1f, 6.1f,
    };
    const std::vector<float> labels = {0.0f, 1.0f, 0.0f};
    const std::vector<int64_t> ids = {10, 20, 30};

    cyxwiz::PairBatch batch;
    batch.input_a = FloatTensor({3, 2}, left);
    batch.input_b = FloatTensor({3, 2}, right);
    batch.pair_label = FloatTensor({3}, labels);
    batch.sample_id_a = IntTensor({3}, ids);
    batch.sample_id_b = IntTensor({3}, ids);
    batch.size = 3;

    std::string error;
    Check(batch.IsValid(), "valid PairBatch should pass shape checks");
    Check(cyxwiz::ValidatePairBatchShape(batch, &error),
          "valid PairBatch should validate with explicit error sink");
    Check(batch.HasSampleIds(), "PairBatch should report paired sample IDs");

    auto missing_label = batch;
    missing_label.pair_label = cyxwiz::Tensor();
    Check(!missing_label.IsValid(),
          "PairBatch without labels should be invalid");

    auto mismatched_inputs = batch;
    mismatched_inputs.input_b = FloatTensor({3, 3}, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    });
    Check(!mismatched_inputs.IsValid(),
          "PairBatch should reject mismatched branch shapes");

    auto bad_label_shape = batch;
    bad_label_shape.pair_label = FloatTensor({3, 2}, {
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    });
    Check(!bad_label_shape.IsValid(),
          "PairBatch labels must be [batch] or [batch, 1]");

    auto bad_optional_ids = batch;
    bad_optional_ids.sample_id_b = IntTensor({2}, {1, 2});
    Check(!bad_optional_ids.IsValid(),
          "PairBatch optional metadata must match batch size");
}

void TestTripletBatchContract() {
    const std::vector<float> anchor = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const std::vector<float> positive = {
        0.9f, 0.1f, 0.0f,
        0.1f, 0.9f, 0.0f,
    };
    const std::vector<float> negative = {
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
    };
    const std::vector<int64_t> ids = {100, 200};

    cyxwiz::TripletBatch batch;
    batch.anchor = FloatTensor({2, 3}, anchor);
    batch.positive = FloatTensor({2, 3}, positive);
    batch.negative = FloatTensor({2, 3}, negative);
    batch.anchor_class_id = IntTensor({2}, ids);
    batch.positive_class_id = IntTensor({2}, ids);
    batch.negative_class_id = IntTensor({2}, ids);
    batch.size = 2;

    Check(batch.IsValid(), "valid TripletBatch should pass shape checks");
    Check(batch.HasClassIds(), "TripletBatch should report class IDs");

    auto missing_negative = batch;
    missing_negative.negative = cyxwiz::Tensor();
    Check(!missing_negative.IsValid(),
          "TripletBatch should require negative inputs");

    auto mismatched_branch = batch;
    mismatched_branch.positive = FloatTensor({2, 2}, {
        0.9f, 0.1f,
        0.1f, 0.9f,
    });
    Check(!mismatched_branch.IsValid(),
          "TripletBatch should reject mismatched branch shapes");

    auto bad_optional_ids = batch;
    bad_optional_ids.negative_class_id = IntTensor({2, 1}, {1, 2});
    Check(bad_optional_ids.IsValid(),
          "TripletBatch should accept [batch, 1] metadata");

    bad_optional_ids.negative_class_id = IntTensor({2, 2}, {
        1, 2,
        3, 4,
    });
    Check(!bad_optional_ids.IsValid(),
          "TripletBatch should reject wide metadata tensors");
}

void TestMetricLearningLabelConventions() {
    using cyxwiz::MetricLearningLabelConvention;

    Check(cyxwiz::MetricLearningConventionRequiresLabels(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar),
          "contrastive loss should require pair labels");
    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              0.0),
          "contrastive should accept 0 = similar");
    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              1.0),
          "contrastive should accept 1 = dissimilar");
    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              -1.0),
          "contrastive should reject cosine label convention");

    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              1.0),
          "cosine embedding should accept 1 = similar");
    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              -1.0),
          "cosine embedding should accept -1 = dissimilar");
    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              0.0),
          "cosine embedding should reject contrastive label convention");

    Check(!cyxwiz::MetricLearningConventionRequiresLabels(
              MetricLearningLabelConvention::TripletNoLabels),
          "triplet loss should not require pair labels");
    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::TripletNoLabels,
              1.0),
          "triplet convention should not validate scalar pair labels");
}

void TestPairBatcherBuildsAlignedBatches() {
    std::vector<cyxwiz::PairSample> samples = {
        {{1.0f, 2.0f}, {1.1f, 2.1f}, 0.0f, 10, 11, 1, 1, true, true},
        {{3.0f, 4.0f}, {4.0f, 3.0f}, 1.0f, 20, 21, 2, 3, true, true},
        {{5.0f, 6.0f}, {5.2f, 6.2f}, 0.0f, 30, 31, 4, 4, true, true},
    };

    cyxwiz::MetricLearningBatcherConfig config;
    config.batch_size = 2;
    config.shuffle = false;
    config.label_convention =
        cyxwiz::MetricLearningLabelConvention::
            ContrastiveZeroSimilarOneDissimilar;

    cyxwiz::PairBatcher batcher(samples, config);
    Check(batcher.GetNumSamples() == 3,
          "PairBatcher should report sample count");
    Check(batcher.GetNumBatches() == 2,
          "PairBatcher should keep partial final batch by default");

    const auto first = batcher.GetNextPairBatch();
    Check(first.IsValid(), "first PairBatch should be valid");
    Check(first.input_a.Shape() == std::vector<size_t>({2, 2}),
          "PairBatcher should infer flat feature shape");
    Check(first.pair_label.Shape() == std::vector<size_t>({2}),
          "PairBatcher labels should be [batch]");
    Check(first.HasSampleIds(), "PairBatcher should preserve sample IDs");
    Check(first.HasClassIds(), "PairBatcher should preserve class IDs");

    const float* input_a = first.input_a.Data<float>();
    const float* input_b = first.input_b.Data<float>();
    const float* labels = first.pair_label.Data<float>();
    const int64_t* sample_ids = first.sample_id_a.Data<int64_t>();
    Check(input_a[0] == 1.0f && input_a[1] == 2.0f &&
              input_a[2] == 3.0f && input_a[3] == 4.0f,
          "PairBatcher should copy left branch rows in order");
    Check(input_b[0] == 1.1f && input_b[1] == 2.1f &&
              input_b[2] == 4.0f && input_b[3] == 3.0f,
          "PairBatcher should copy right branch rows in order");
    Check(labels[0] == 0.0f && labels[1] == 1.0f,
          "PairBatcher should copy labels in order");
    Check(sample_ids[0] == 10 && sample_ids[1] == 20,
          "PairBatcher should copy metadata in order");

    const auto final_batch = batcher.GetNextPairBatch();
    Check(final_batch.IsValid() && final_batch.size == 1,
          "PairBatcher should emit partial final batch");
    Check(batcher.IsEpochComplete(),
          "PairBatcher should complete after final batch");

    bool rejected_bad_label = false;
    try {
        auto bad_samples = samples;
        bad_samples[0].label = -1.0f;
        cyxwiz::PairBatcher bad_batcher(bad_samples, config);
        (void)bad_batcher;
    } catch (const std::invalid_argument&) {
        rejected_bad_label = true;
    }
    Check(rejected_bad_label,
          "PairBatcher should reject labels outside convention");
}

void TestPairBatcherDropLastAndShapeValidation() {
    std::vector<cyxwiz::PairSample> samples = {
        {{1.0f, 2.0f}, {1.0f, 2.0f}, 0.0f},
        {{3.0f, 4.0f}, {4.0f, 3.0f}, 1.0f},
        {{5.0f, 6.0f}, {5.0f, 6.0f}, 0.0f},
    };

    cyxwiz::MetricLearningBatcherConfig config;
    config.batch_size = 2;
    config.drop_last = true;

    cyxwiz::PairBatcher batcher(samples, config);
    Check(batcher.GetNumBatches() == 1,
          "PairBatcher drop_last should floor batch count");
    Check(batcher.GetNextPairBatch().size == 2,
          "PairBatcher drop_last should emit one full batch");
    Check(!batcher.GetNextPairBatch().IsValid(),
          "PairBatcher drop_last should suppress final partial batch");

    bool rejected_bad_shape = false;
    try {
        samples[1].input_b = {1.0f, 2.0f, 3.0f};
        cyxwiz::PairBatcher bad_batcher(samples, config);
        (void)bad_batcher;
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    Check(rejected_bad_shape,
          "PairBatcher should reject mismatched sample shapes");
}

void TestTripletBatcherBuildsAlignedBatches() {
    std::vector<cyxwiz::TripletSample> samples = {
        {{1.0f, 0.0f}, {0.9f, 0.1f}, {0.0f, 1.0f},
         10, 11, 12, 1, 1, 2, true, true},
        {{0.0f, 1.0f}, {0.1f, 0.9f}, {1.0f, 0.0f},
         20, 21, 22, 3, 3, 4, true, true},
        {{2.0f, 2.0f}, {2.1f, 2.0f}, {-2.0f, -2.0f},
         30, 31, 32, 5, 5, 6, true, true},
    };

    cyxwiz::MetricLearningBatcherConfig config;
    config.batch_size = 2;
    config.shuffle = false;
    config.label_convention =
        cyxwiz::MetricLearningLabelConvention::TripletNoLabels;

    cyxwiz::TripletBatcher batcher(samples, config);
    Check(batcher.GetNumSamples() == 3,
          "TripletBatcher should report sample count");
    Check(batcher.GetNumBatches() == 2,
          "TripletBatcher should keep partial final batch by default");

    const auto first = batcher.GetNextTripletBatch();
    Check(first.IsValid(), "first TripletBatch should be valid");
    Check(first.anchor.Shape() == std::vector<size_t>({2, 2}),
          "TripletBatcher should infer flat feature shape");
    Check(first.HasSampleIds(), "TripletBatcher should preserve sample IDs");
    Check(first.HasClassIds(), "TripletBatcher should preserve class IDs");

    const float* anchor = first.anchor.Data<float>();
    const float* positive = first.positive.Data<float>();
    const float* negative = first.negative.Data<float>();
    const int64_t* class_ids = first.negative_class_id.Data<int64_t>();
    Check(anchor[0] == 1.0f && anchor[1] == 0.0f &&
              anchor[2] == 0.0f && anchor[3] == 1.0f,
          "TripletBatcher should copy anchor rows in order");
    Check(positive[0] == 0.9f && positive[1] == 0.1f &&
              positive[2] == 0.1f && positive[3] == 0.9f,
          "TripletBatcher should copy positive rows in order");
    Check(negative[0] == 0.0f && negative[1] == 1.0f &&
              negative[2] == 1.0f && negative[3] == 0.0f,
          "TripletBatcher should copy negative rows in order");
    Check(class_ids[0] == 2 && class_ids[1] == 4,
          "TripletBatcher should copy metadata in order");

    const auto final_batch = batcher.GetNextTripletBatch();
    Check(final_batch.IsValid() && final_batch.size == 1,
          "TripletBatcher should emit partial final batch");
    Check(batcher.IsEpochComplete(),
          "TripletBatcher should complete after final batch");

    bool rejected_bad_shape = false;
    try {
        samples[0].negative = {1.0f, 2.0f, 3.0f};
        cyxwiz::TripletBatcher bad_batcher(samples, config);
        (void)bad_batcher;
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    Check(rejected_bad_shape,
          "TripletBatcher should reject mismatched sample shapes");
}

void TestPairDatasetBuilderWithExplicitLabels() {
    std::vector<cyxwiz::PairDatasetRow> rows = {
        {{1.0f, 2.0f}, {1.1f, 2.1f}, 0.0f, 10, 11, 1, 1,
         true, true, true},
        {{3.0f, 4.0f}, {4.0f, 3.0f}, 1.0f, 20, 21, 2, 3,
         true, true, true},
    };

    cyxwiz::PairDatasetBuilderConfig config;
    config.batcher.batch_size = 2;
    const auto built = cyxwiz::BuildPairDataset(rows, config);
    Check(built.samples.size() == 2,
          "PairDatasetBuilder should emit pair samples");
    Check(built.has_sample_ids && built.has_class_ids,
          "PairDatasetBuilder should report metadata presence");
    Check(built.samples[0].label == 0.0f &&
              built.samples[1].label == 1.0f,
          "PairDatasetBuilder should preserve explicit labels");

    auto batcher = built.CreateBatcher();
    const auto batch = batcher.GetNextPairBatch();
    Check(batch.IsValid() && batch.size == 2,
          "PairDatasetBuildResult should create a working PairBatcher");

    bool rejected_bad_label = false;
    try {
        rows[0].label = -1.0f;
        (void)cyxwiz::BuildPairDataset(rows, config);
    } catch (const std::runtime_error&) {
        rejected_bad_label = true;
    }
    Check(rejected_bad_label,
          "PairDatasetBuilder should reject labels outside convention");
}

void TestPairDatasetBuilderDerivesLabelsFromClassIds() {
    std::vector<cyxwiz::PairDatasetRow> rows = {
        {{1.0f, 2.0f}, {1.2f, 2.2f}, 0.0f, 0, 0, 7, 7,
         false, false, true},
        {{3.0f, 4.0f}, {4.0f, 3.0f}, 0.0f, 0, 0, 7, 9,
         false, false, true},
    };

    cyxwiz::PairDatasetBuilderConfig contrastive;
    contrastive.require_labels = false;
    contrastive.derive_labels_from_class_ids = true;
    auto built = cyxwiz::BuildPairDataset(rows, contrastive);
    Check(built.samples[0].label == 0.0f &&
              built.samples[1].label == 1.0f,
          "PairDatasetBuilder should derive contrastive labels");

    cyxwiz::PairDatasetBuilderConfig cosine;
    cosine.require_labels = false;
    cosine.derive_labels_from_class_ids = true;
    cosine.batcher.label_convention =
        cyxwiz::MetricLearningLabelConvention::
            CosineOneSimilarNegativeOneDissimilar;
    built = cyxwiz::BuildPairDataset(rows, cosine);
    Check(built.samples[0].label == 1.0f &&
              built.samples[1].label == -1.0f,
          "PairDatasetBuilder should derive cosine labels");

    bool rejected_missing_label = false;
    try {
        cyxwiz::PairDatasetBuilderConfig missing;
        missing.require_labels = true;
        missing.derive_labels_from_class_ids = false;
        (void)cyxwiz::BuildPairDataset(rows, missing);
    } catch (const std::runtime_error&) {
        rejected_missing_label = true;
    }
    Check(rejected_missing_label,
          "PairDatasetBuilder should reject missing labels by default");
}

void TestTripletDatasetBuilderValidatesClassSemantics() {
    std::vector<cyxwiz::TripletDatasetRow> rows = {
        {{1.0f, 0.0f}, {0.9f, 0.1f}, {0.0f, 1.0f},
         10, 11, 12, 4, 4, 5, true, true},
        {{0.0f, 1.0f}, {0.1f, 0.9f}, {1.0f, 0.0f},
         20, 21, 22, 8, 8, 9, true, true},
    };

    cyxwiz::TripletDatasetBuilderConfig config;
    config.batcher.batch_size = 2;
    const auto built = cyxwiz::BuildTripletDataset(rows, config);
    Check(built.samples.size() == 2,
          "TripletDatasetBuilder should emit triplet samples");
    Check(built.batcher_config.label_convention ==
              cyxwiz::MetricLearningLabelConvention::TripletNoLabels,
          "TripletDatasetBuilder should force triplet label convention");

    auto batcher = built.CreateBatcher();
    const auto batch = batcher.GetNextTripletBatch();
    Check(batch.IsValid() && batch.size == 2,
          "TripletDatasetBuildResult should create a working TripletBatcher");

    bool rejected_bad_classes = false;
    try {
        rows[0].negative_class_id = rows[0].anchor_class_id;
        (void)cyxwiz::BuildTripletDataset(rows, config);
    } catch (const std::runtime_error&) {
        rejected_bad_classes = true;
    }
    Check(rejected_bad_classes,
          "TripletDatasetBuilder should reject invalid class semantics");

    config.validate_class_ids = false;
    const auto unchecked = cyxwiz::BuildTripletDataset(rows, config);
    Check(unchecked.samples.size() == 2,
          "TripletDatasetBuilder should allow disabling class validation");
}

}  // namespace

int main() {
    TestPairBatchContract();
    TestTripletBatchContract();
    TestMetricLearningLabelConventions();
    TestPairBatcherBuildsAlignedBatches();
    TestPairBatcherDropLastAndShapeValidation();
    TestTripletBatcherBuildsAlignedBatches();
    TestPairDatasetBuilderWithExplicitLabels();
    TestPairDatasetBuilderDerivesLabelsFromClassIds();
    TestTripletDatasetBuilderValidatesClassSemantics();
    std::cout << "Metric-learning batch contracts passed\n";
    return 0;
}
