#pragma once

#include "data_loader.h"

namespace cyxwiz::loaders {

// Handles FileCategory::Text. Supports both single-file corpora
// (CSV/TSV/JSON/TXT with an explicit text column + optional label
// column) and class-subdirectory corpora (folder/<class>/*.txt).
// TextDataset's constructor auto-detects file vs directory from the
// source path itself, so the loader just forwards ApplyContext::
// source_path — the dialog puts either the file_path or folder_path
// there based on the user's layout choice.
//
// The worker builds a TextDataset probe (tokenizer + vocab + sample
// enumeration), then calls DataRegistry::RegisterTextDataset with
// the resulting metadata. Feature tensors are materialized lazily at
// training time by TextDatasetBatcher, not here.
class TextLoader : public DataLoader {
public:
    FileCategory Category() const override { return FileCategory::Text; }
    const char* CategoryName() const override { return "Text"; }
    int BackendTag() const override { return 5; }
    bool IsLazyLoaded() const override { return true; }

    bool ValidateApplyContext(const ApplyContext& ctx,
                              std::string& err) const override;
    uint64_t LaunchAsyncLoad(const ApplyContext& ctx,
                             std::shared_ptr<AsyncLoadState> state) override;

    bool IsRegistered(const std::string& name) const override;
    void Unregister(const std::string& name) override;

    bool RestoreFromRegistry(const std::string& name,
                             const gui::MLNode& node,
                             RestoreState& out) const override;
    CompletedLoadDescription DescribeCompletedLoad(
        const AsyncLoadState& state) const override;
};

}  // namespace cyxwiz::loaders
