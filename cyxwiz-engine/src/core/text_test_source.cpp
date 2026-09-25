// Text-dataset test batchers for the shared TestExecutor (declared in
// legacy_dataset_batchers.h; separate so DatasetHandle users do not link the
// text tokenizer stack).
#include "legacy_dataset_batchers.h"

#include "classification_decision.h"
#include "text_dataset_batcher.h"

#include <algorithm>

namespace cyxwiz {

std::unique_ptr<IBatcher> BuildTextTestBatcher(const TrainingConfiguration& config,
                                               const DataRegistry::TextDatasetEntry& entry, int batch_size) {
    auto test = std::make_unique<TextDatasetBatcher>(
        entry, config.text_preprocessing, batch_size, config.train_ratio, config.val_ratio, config.test_ratio,
        false, config.num_workers, static_cast<uint32_t>(config.dataloader_seed), config.stratified,
        static_cast<uint32_t>(std::max(0, config.split_seed)), false, "none", "max",
        static_cast<uint32_t>(std::max(0, config.balance_seed)));
    test->SetPhase(BatcherPhase::Test);
    test->Reset();
    if (UsesScalarBinaryTargets(config.loss_type)) {
        test->SetScalarLabelMode(true);
    } else if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
        test->SetOneHotEncoding(config.preprocessing.num_classes);
    } else if (config.output_size > 0) {
        test->SetOneHotEncoding(config.output_size);
    }
    return test;
}

ExternalTestSource TextTestSource(DataRegistry::TextDatasetEntry entry) {
    ExternalTestSource source;
    source.class_names = entry.class_names;
    source.make_batcher = [entry = std::move(entry)](const TrainingConfiguration& config, int batch_size) {
        return BuildTextTestBatcher(config, entry, batch_size);
    };
    return source;
}

}  // namespace cyxwiz
