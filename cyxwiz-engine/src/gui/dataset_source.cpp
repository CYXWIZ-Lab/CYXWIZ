#include "dataset_source.h"

#include "../core/arrow_data_table.h"
#include "../core/dataset_catalog.h"
#include "../core/session_query_engine.h"
#include "../core/session_query_service.h"
#include "../data/data_table.h"
#include "dataset_picker.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <utility>

namespace cyxwiz::ui {

void DatasetSource::Render(const char* id, Loaded on_loaded) {
    const Tokens& t = CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Dataset");
    ImGui::SameLine();
    if (DatasetPicker(id, name_)) Read(std::move(on_loaded));
    if (loading_) {
        ImGui::TextColored(t.text_dim, "Reading the dataset...");
    } else if (!error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void DatasetSource::Reload(Loaded on_loaded) {
    if (!name_.empty()) Read(std::move(on_loaded));
}

void DatasetSource::Read(Loaded on_loaded) {
    loading_ = true;
    error_.clear();
    QueryRequest request;
    request.sql = "SELECT * FROM " + SessionQueryEngine::QuoteIdentifier(name_);
    const auto entry = DatasetCatalog::Instance().Resolve(name_);
    request.label = "Read " + (entry ? entry->Shown() : name_);
    const std::string name = name_;
    SessionQueryService::Instance().Submit(
        std::move(request),
        [this, name, on_loaded = std::move(on_loaded)](const QueryResult& result) {
            if (name != name_) return;  // a newer pick is being read
            loading_ = false;
            if (!result.ok || !result.table) {
                error_ = result.cancelled ? "Reading the dataset was cancelled" : "Cannot read the dataset: " + result.error;
                return;
            }
            on_loaded(DataTableFromArrow(*result.table, name));
        },
        alive_);
}

}  // namespace cyxwiz::ui
