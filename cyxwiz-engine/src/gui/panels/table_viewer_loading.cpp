#include "table_viewer.h"
#include <spdlog/spdlog.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz {

bool TableViewerPanel::LoadCSV(const std::string& filepath) {
    // Check if already open
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "csv");
    return true;
}

bool TableViewerPanel::LoadTXT(const std::string& filepath, char delimiter) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "txt", delimiter);
    return true;
}

bool TableViewerPanel::LoadHDF5(const std::string& filepath, const std::string& dataset_name) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    // For HDF5, we pass dataset name as part of type
    LoadFileAsync(filepath, "hdf5:" + dataset_name);
    return true;
}

bool TableViewerPanel::LoadExcel(const std::string& filepath, const std::string& sheet_name) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "excel:" + sheet_name);
    return true;
}

void TableViewerPanel::LoadFileAsync(const std::string& filepath, const std::string& type, char delimiter) {
    // Create new tab
    auto tab = std::make_unique<TableTab>();
    tab->filepath = filepath;
    tab->filename = fs::path(filepath).filename().string();
    tab->is_loading = true;
    tab->load_progress = 0.0f;
    tab->load_status = "Loading...";

    int tab_index = static_cast<int>(tabs_.size());
    tabs_.push_back(std::move(tab));
    active_tab_index_ = tab_index;

    // Check if lazy loading should be used (only for CSV/TXT, large files)
    bool use_lazy = false;
    std::string main_type = type;
    size_t colon_pos = type.find(':');
    if (colon_pos != std::string::npos) {
        main_type = type.substr(0, colon_pos);
    }

    if (prefer_lazy_loading_ && (main_type == "csv" || main_type == "txt")) {
        use_lazy = LazyDataTable::ShouldUseLazyLoading(filepath, LAZY_LOAD_THRESHOLD_MB);
    }

    if (use_lazy) {
        spdlog::info("Using lazy loading for large file: {}", filepath);
    } else {
        spdlog::info("Starting async load of: {}", filepath);
    }

    // Capture values for lambda
    std::string path = filepath;
    std::string file_type = type;

    AsyncTaskManager::Instance().RunAsync(
        "Loading: " + fs::path(filepath).filename().string(),
        [this, tab_index, path, file_type, delimiter, use_lazy](LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            // Parse type and optional parameter
            std::string main_type = file_type;
            std::string type_param;
            size_t colon_pos = file_type.find(':');
            if (colon_pos != std::string::npos) {
                main_type = file_type.substr(0, colon_pos);
                type_param = file_type.substr(colon_pos + 1);
            }

            bool success = false;

            if (use_lazy) {
                // Lazy loading for large CSV/TXT files
                auto lazy_table = std::make_shared<LazyDataTable>();
                char delim = (main_type == "txt") ? delimiter : ',';

                success = lazy_table->IndexFile(path, delim,
                    [&task](float progress, const std::string& status) {
                        task.ReportProgress(0.1f + progress * 0.8f, status);
                    });

                if (success) {
                    lazy_table->SetName(fs::path(path).stem().string());
                    task.MarkCompleted();

                    // Update tab with lazy table
                    if (tab_index < static_cast<int>(tabs_.size())) {
                        auto& tab = tabs_[tab_index];
                        tab->lazy_table = lazy_table;
                        tab->use_lazy_loading = true;
                        tab->is_loading = false;
                        tab->load_progress = 1.0f;
                        tab->load_status = "Complete (lazy)";
                    }
                } else {
                    task.MarkFailed("Failed to index file");
                    if (tab_index < static_cast<int>(tabs_.size())) {
                        auto& tab = tabs_[tab_index];
                        tab->is_loading = false;
                        tab->load_progress = 1.0f;
                        tab->load_status = "Failed";
                    }
                }
            } else {
                // Standard in-memory loading
                auto table = std::make_shared<DataTable>();
                task.ReportProgress(0.3f, "Parsing data...");

                if (main_type == "csv") {
                    success = table->LoadFromCSV(path);
                } else if (main_type == "txt") {
                    success = table->LoadFromTXT(path, delimiter);
                } else if (main_type == "hdf5") {
                    success = table->LoadFromHDF5(path, type_param.empty() ? "data" : type_param);
                } else if (main_type == "excel") {
                    success = table->LoadFromExcel(path, type_param);
                } else {
                    // Default to CSV
                    success = table->LoadFromCSV(path);
                }

                task.ReportProgress(0.9f, "Finalizing...");

                if (success) {
                    table->SetName(fs::path(path).stem().string());
                    task.MarkCompleted();
                } else {
                    task.MarkFailed("Failed to parse file");
                }

                // Update tab with result
                if (tab_index < static_cast<int>(tabs_.size())) {
                    auto& tab = tabs_[tab_index];
                    tab->table = success ? table : nullptr;
                    tab->use_lazy_loading = false;
                    tab->is_loading = false;
                    tab->load_progress = 1.0f;
                    tab->load_status = success ? "Complete" : "Failed";
                }
            }
        },
        [this, tab_index](float progress, const std::string& status) {
            // Progress callback - update tab
            if (tab_index < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[tab_index];
                tab->load_progress = progress;
                tab->load_status = status;
            }
        },
        [this, tab_index, path, use_lazy](bool success, const std::string& error) {
            if (success) {
                spdlog::info("Async load completed{}: {}", use_lazy ? " (lazy)" : "", path);
            } else {
                spdlog::error("Async load failed: {} - {}", path, error);
            }
        }
    );
}


}  // namespace cyxwiz

