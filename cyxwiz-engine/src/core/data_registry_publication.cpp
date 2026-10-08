#include "data_registry.h"
#include "arrow_dataset.h"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cyxwiz {

struct DataRegistry::PreparedArrowPublication::Impl {
    TabularPublicationToken token;
    DatasetInfo info;
    DatasetLoadedCallback callback;
    bool committed = false;
    bool notified = false;
    decltype(arrow_datasets_)::node_type arrow;
    decltype(tabular_source_paths_by_name_)::node_type source;
    decltype(last_access_times_)::node_type access;
    decltype(arrow_datasets_)::node_type displaced_arrow;
    decltype(parquet_backed_datasets_)::node_type displaced_parquet;
    decltype(materialization_provenance_)::node_type displaced_provenance;
    decltype(last_access_times_)::node_type displaced_access;
};

DataRegistry::PreparedArrowPublication::PreparedArrowPublication() = default;
DataRegistry::PreparedArrowPublication::~PreparedArrowPublication() = default;
DataRegistry::PreparedArrowPublication::PreparedArrowPublication(PreparedArrowPublication&&) noexcept = default;
DataRegistry::PreparedArrowPublication& DataRegistry::PreparedArrowPublication::operator=(PreparedArrowPublication&&) noexcept = default;

void DataRegistry::PreparedArrowPublication::Notify() noexcept {
    if (!impl_ || !impl_->committed || impl_->notified) return;
    static_assert(std::is_nothrow_move_constructible_v<DatasetInfo>);
    auto callback = std::move(impl_->callback);
    auto info = std::move(impl_->info);
    impl_->notified = true;
    // No access to this after delivery: a reentrant callback may destroy its owner.
    try {
        if (callback) callback(info.name, info);
    } catch (const std::exception& e) {
        try {
            spdlog::error("Tabular publication callback for '{}' failed: {}", info.name, e.what());
        } catch (...) {
            // Notification/logging failure cannot undo a committed publication.
        }
    } catch (...) {
        try {
            spdlog::error("Tabular publication callback for '{}' failed", info.name);
        } catch (...) {
        }
    }
}

DataRegistry::TabularPublicationToken DataRegistry::CaptureTabularPublication(
    const std::string& name) {
    if (name.empty()) throw std::invalid_argument("Tabular publication requires a non-empty name");
    TabularPublicationToken result;
    result.name_ = name;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = tabular_publication_tokens_.begin(); it != tabular_publication_tokens_.end();) {
        if (it->second.expired()) it = tabular_publication_tokens_.erase(it);
        else ++it;
    }
    const auto existing = tabular_publication_tokens_.find(name);
    if (existing != tabular_publication_tokens_.end()) result.token_ = existing->second.lock();
    if (!result.token_) {
        result.token_ = std::make_shared<const unsigned char>(0);
        tabular_publication_tokens_.insert_or_assign(name, result.token_);
    }
    return result;
}

std::unique_ptr<DataRegistry::PreparedArrowPublication> DataRegistry::PrepareArrowTablePublication(
    const TabularPublicationToken& token, std::shared_ptr<ArrowDataset> candidate,
    const std::string& source_path, std::string& error) {
    error.clear();
    if (token.name_.empty() || !token.token_) {
        error = "Invalid tabular publication token";
        return nullptr;
    }
    if (!candidate || candidate->GetName() != token.name_) {
        error = "Tabular publication candidate name does not match its token";
        return nullptr;
    }
    try {
        const auto table = candidate->GetArrowTable();
        if (!table || table->num_rows() <= 0 || table->num_columns() <= 0) {
            error = "Tabular publication candidate has no nonempty Arrow table";
            return nullptr;
        }
        const auto status = table->ValidateFull();
        if (!status.ok()) {
            error = "Invalid tabular publication table: " + status.ToString();
            return nullptr;
        }
        auto result = std::make_unique<PreparedArrowPublication>();
        result->impl_ = std::make_unique<PreparedArrowPublication::Impl>();
        auto& prepared = *result->impl_;
        prepared.token = token;
        const auto& name = token.name_;
        const auto normalized = NormalizeTabularSourcePath(source_path);
        prepared.info.name = name;
        prepared.info.path = normalized;
        prepared.info.num_samples = candidate->GetNumRows();
        prepared.info.memory_usage = candidate->GetMemoryUsage();
        prepared.info.is_loaded = true;

        decltype(arrow_datasets_) arrow;
        arrow.emplace(name, std::move(candidate));
        prepared.arrow = arrow.extract(name);
        decltype(tabular_source_paths_by_name_) source;
        if (!normalized.empty()) source.emplace(name, normalized);
        prepared.source = source.extract(name);
        decltype(last_access_times_) access;
        access.emplace(name, std::chrono::steady_clock::time_point{});
        prepared.access = access.extract(name);
        return result;
    } catch (const std::exception& e) {
        error = std::string("Tabular publication preparation failed: ") + e.what();
    } catch (...) {
        error = "Tabular publication preparation failed with an unknown exception";
    }
    return nullptr;
}

bool DataRegistry::TryPublishPreparedArrowTable(PreparedArrowPublication& prepared,
                                               std::string& error) {
    error.clear();
    if (!prepared.impl_ || prepared.impl_->committed) {
        error = "Invalid or already committed tabular publication";
        return false;
    }
    auto& value = *prepared.impl_;
    const auto& name = value.token.name_;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto current = tabular_publication_tokens_.find(name);
        if (current == tabular_publication_tokens_.end() ||
            current->second.lock() != value.token.token_) {
            error = "Stale tabular publication token: registration changed";
            return false;
        }
        if (datasets_.count(name) || image_dataset_entries_.count(name) ||
            audio_dataset_entries_.count(name) || text_dataset_entries_.count(name) ||
            sparse_feature_datasets_.count(name)) {
            error = "Tabular publication name is occupied by another dataset category";
            return false;
        }
        // Callback copying can throw. Nothing after it allocates, scans a
        // table, or calls external code. Retired datasets stay in the handoff.
        value.callback = on_loaded_;
        value.access.mapped() = std::chrono::steady_clock::now();
        value.displaced_arrow = arrow_datasets_.extract(name);
        value.displaced_parquet = parquet_backed_datasets_.extract(name);
        value.displaced_provenance = materialization_provenance_.extract(name);
        value.displaced_access = last_access_times_.extract(name);
        ForgetTabularSourcePathUnlocked(name);
        arrow_datasets_.insert(std::move(value.arrow));
        if (!value.source.empty()) tabular_source_paths_by_name_.insert(std::move(value.source));
        last_access_times_.insert(std::move(value.access));
        InvalidateTabularPublicationUnlocked(name);
        value.committed = true;
        return true;
    } catch (const std::exception& e) {
        error = std::string("Tabular publication failed: ") + e.what();
    } catch (...) {
        error = "Tabular publication failed with an unknown exception";
    }
    return false;
}

bool DataRegistry::TryPublishArrowTable(
    const TabularPublicationToken& token, std::shared_ptr<ArrowDataset> candidate,
    const std::string& source_path, std::string& error) {
    auto prepared = PrepareArrowTablePublication(token, std::move(candidate), source_path, error);
    if (!prepared || !TryPublishPreparedArrowTable(*prepared, error)) return false;
    prepared->Notify();
    return true;
}

} // namespace cyxwiz
