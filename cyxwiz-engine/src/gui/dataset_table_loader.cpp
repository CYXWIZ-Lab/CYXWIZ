#include "dataset_table_loader.h"

#include "../core/arrow_data_table.h"
#include "../core/dataset_catalog.h"
#include "../core/session_query_engine.h"
#include "../core/session_query_service.h"
#include "../data/data_table.h"

#include <utility>

namespace cyxwiz {

void LoadDatasetTable(const std::string& dataset_name, std::weak_ptr<const void> owner, DatasetTableDone done) {
    QueryRequest request;
    request.sql = "SELECT * FROM " + SessionQueryEngine::QuoteIdentifier(dataset_name);
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_name);
    request.label = "Read " + (entry ? entry->Shown() : dataset_name);
    SessionQueryService::Instance().Submit(
        std::move(request),
        [dataset_name, done = std::move(done)](const QueryResult& result) {
            if (!result.ok || !result.table) {
                done(nullptr, result.cancelled ? "Reading the dataset was cancelled"
                                               : "Cannot read the dataset: " + result.error);
                return;
            }
            done(DataTableFromArrow(*result.table, dataset_name), {});
        },
        std::move(owner));
}

}  // namespace cyxwiz
