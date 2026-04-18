#pragma once

#include "data_loader.h"

namespace cyxwiz::loaders {

// Handles FileCategory::Tabular (and FileCategory::TimeSeries, which
// shares the same load path). Storage backend (Arrow in-memory vs
// Parquet disk-backed) is chosen inside LaunchAsyncLoad based on file
// size and ApplyContext::force_disk_backed — the worker overrides
// BackendTag() from 1 to 2 when it ends up on the disk-backed path.
class TabularLoader : public DataLoader {
public:
    FileCategory Category() const override { return FileCategory::Tabular; }
    const char* CategoryName() const override { return "Tabular"; }
    int BackendTag() const override { return 1; }

    bool ValidateApplyContext(const ApplyContext& ctx,
                              std::string& err) const override;
    uint64_t LaunchAsyncLoad(const ApplyContext& ctx,
                             std::shared_ptr<AsyncLoadState> state) override;

    bool IsRegistered(const std::string& name) const override;
    void Unregister(const std::string& name) override;
};

}  // namespace cyxwiz::loaders
