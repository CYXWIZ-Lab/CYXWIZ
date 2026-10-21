// Focused seams for optional services that are not part of the legacy
// DatasetHandle lifecycle contract. Any accidental preprocessing use fails
// loudly instead of turning this target into a second full engine link.

#include "../src/core/annotation_manager.h"
#include "../src/core/data_registry.h"
#include "../src/preprocessing/image_transform.h"
#include "../src/preprocessing/normalization_transform.h"
#include "../src/preprocessing/scaling_transform.h"
#include "../src/preprocessing/statistics_calculator.h"
#include "../src/transforms/transform.h"

#include <stdexcept>

namespace cyxwiz {

DataRegistry& DataRegistry::Instance() {
    static DataRegistry instance;
    return instance;
}

bool DataRegistry::HasPreprocessingConfig(const std::string&) const {
    return false;
}

PreprocessingConfig DataRegistry::GetPreprocessingConfig(
    const std::string& dataset_id) const {
    PreprocessingConfig config;
    config.dataset_id = dataset_id;
    return config;
}

bool DataRegistry::HasAugmentationPipeline(const std::string&) const {
    return false;
}

std::shared_ptr<transforms::Compose> DataRegistry::GetAugmentationPipeline(
    const std::string&) const {
    return nullptr;
}

AnnotationManager& DataRegistry::GetAnnotationManager() {
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

const AnnotationManager& DataRegistry::GetAnnotationManager() const {
    if (!annotation_manager_) {
        annotation_manager_ = std::make_unique<AnnotationManager>();
    }
    return *annotation_manager_;
}

bool AnnotationManager::HasAnnotationSet(const std::string&) const {
    return false;
}

std::vector<float> AnnotationManager::GetSegmentationMask(
    const std::string&,
    size_t,
    int,
    int) const {
    return {};
}

DatasetStatistics StatisticsCalculator::Compute(
    const std::string&,
    DataRegistry*,
    std::function<void(float)>) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable preprocessing statistics");
}

NormalizationTransform::NormalizationTransform(
    const NormalizationConfig& config)
    : config_(config) {}

void NormalizationTransform::Initialize(const DatasetStatistics&) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable normalization");
}

Tensor NormalizationTransform::Apply(const Tensor&) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable normalization");
}

ScalingTransform::ScalingTransform(const ScalingConfig& config)
    : config_(config) {}

void ScalingTransform::Initialize(const DatasetStatistics&) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable scaling");
}

Tensor ScalingTransform::Apply(const Tensor&) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable scaling");
}

ImageTransform::ImageTransform(const ImagePreprocessingConfig& config)
    : config_(config) {}

Tensor ImageTransform::Apply(const Tensor&) {
    throw std::logic_error(
        "legacy lifecycle harness does not enable image preprocessing");
}

} // namespace cyxwiz
