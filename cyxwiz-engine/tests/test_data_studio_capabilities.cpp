#include "core/data_studio_capabilities.h"
#include "core/data_studio_execution_plan.h"
#include "core/node_metadata_registry.h"
#include <iostream>
#include <stdexcept>

using namespace cyxwiz;
using N = gui::NodeType;
using S = PipelineStorageBackend;
using A = DataStudioActionState;
int checks = 0;
void Check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        auto& registry = NodeMetadataRegistry::Instance();
        registry.Initialize();
        auto resolve = [&](N type, std::map<std::string, std::string> parameters,
                           std::vector<S> storage) {
            return ResolveDataStudioCapability({type, parameters, storage}, registry.GetMetadata(type));
        };
        auto filter = resolve(N::FilterRows, {{"condition", "id > 0"}}, {S::ArrowTable});
        Check(filter.state == A::RuntimeBacked, "Arrow row filter should pass structural preflight");
        Check(!filter.display_name.empty(), "Display name must come from metadata");
        Check(resolve(N::SortRows, {{"columns", "id"}, {"order", "random"}}, {S::ArrowTable}).state == A::Unavailable, "Unsupported enum value must be rejected");
        auto invalid_length = resolve(N::TextTokenizer, {{"text_col", "body_text"}, {"max_length", "0"}}, {S::ArrowTable});
        if (invalid_length.reason.find("max_length") == std::string::npos) std::cerr << "Length capability: " << invalid_length.reason << '\n';
        Check(invalid_length.state == A::Unavailable && invalid_length.reason.find("max_length") != std::string::npos, "Numeric validation must come from runtime contract");
        Check(resolve(N::FilterRows, {}, {S::ArrowTable}).state == A::Unavailable, "Missing condition");
        Check(resolve(N::FilterRows, {{"condition", " \t"}}, {S::ArrowTable}).state == A::Unavailable, "Blank condition");
        Check(resolve(N::FilterRows, {{"condition", "id > 0"}}, {}).state == A::Unavailable, "Missing input");
        Check(resolve(N::FilterRows, {{"condition", "id > 0"}}, {S::ArrowTable, S::ArrowTable}).state == A::Unavailable, "Excess input");
        for (auto storage : {S::Unknown, S::ParquetBacked, S::ImageDataset, S::AudioDataset, S::TextDataset, S::SparseFeatureDataset}) {
            auto value = resolve(N::FilterRows, {{"condition", "id > 0"}}, {storage});
            Check(value.state == A::Unavailable && value.reason.find("Input 1") != std::string::npos, "Storage rejection must identify the input");
        }
        const std::map<std::string, std::string> keys{{"left_on", "id"}, {"right_on", "id"}};
        Check(resolve(N::JoinTables, keys, {S::ArrowTable, S::ArrowTable}).state == A::RuntimeBacked, "Two-input join");
        Check(resolve(N::JoinTables, keys, {S::ArrowTable}).state == A::Unavailable, "Join requires both inputs");
        auto mixed = resolve(N::JoinTables, keys, {S::ArrowTable, S::ParquetBacked});
        Check(mixed.state == A::Unavailable && mixed.reason.find("Input 2") != std::string::npos, "Mixed join storage");
        for (const auto& [type, parameters] : std::vector<std::pair<N, std::map<std::string, std::string>>>{
            {N::SelectColumns, {{"columns", "id"}}}, {N::SortRows, {{"columns", "id"}}},
            {N::GroupByAggregate, {{"group_columns", "id"}, {"aggregations", "COUNT(*)"}}},
            {N::TextCleanNode, {{"text_column", "body_text"}}}}) {
            Check(resolve(type, parameters, {S::ArrowTable}).state == A::RuntimeBacked, "Audited Arrow operation");
        }
        auto sql = resolve(N::SQLQuery, {{"query", "SELECT * FROM dataset"}}, {S::ArrowTable});
        if (sql.state != A::ExploreOnly) std::cerr << "SQL capability: " << sql.reason << '\n';
        Check(sql.state == A::ExploreOnly, "SQL must not advertise pipeline support");
        Check(resolve(N::SQLQuery, {{"query", "SELECT 1"}}, {}).state == A::Unavailable, "SQL missing source");
        Check(resolve(N::SQLQuery, {{"query", "SELECT 1"}}, {S::ParquetBacked}).state == A::Unavailable, "SQL unsupported source");
        Check(resolve(N::SQLQuery, {{"query", " \n\t"}}, {S::ArrowTable}).state == A::Unavailable, "SQL empty query");
        Check(resolve(N::Subgraph, {}, {S::ArrowTable}).state == A::Unavailable, "Subgraph must remain unavailable");
        NodeMetadata advertised;
        advertised.type = N::Subgraph;
        advertised.status = NodeImplementationStatus::Implemented;
        Check(ResolveDataStudioCapability({N::Subgraph, {}, {}}, &advertised).state == A::Unavailable, "Implemented badge cannot enable unknown runtime");
        Check(ResolveDataStudioCapability({N::FilterRows, {}, {}}, nullptr).state == A::Unavailable, "Missing metadata");
        Check(ResolveDataStudioCapability({N::FilterRows, {}, {}}, &advertised).state == A::Unavailable, "Mismatched metadata");
        auto blocked = *registry.GetMetadata(N::FilterRows);
        blocked.badge = "Blocked";
        Check(ResolveDataStudioCapability({N::FilterRows, {{"condition", "id > 0"}}, {S::ArrowTable}}, &blocked).state == A::Unavailable, "Product-wide block must be respected");
        std::string error;
        Check(!ValidateDataStudioOperationConfiguration({0, "FilterRows", "", {}}, 1, error) && error.find("condition") != std::string::npos, "Shared required parameter validation");
        Check(!ValidateDataStudioOperationConfiguration({0, "FilterRows", "", {{"condition", "x"}}}, -1, error), "Negative input count");
        Check(!BuildDataStudioExecutionPlan({{0, "SQLQuery", "", {{"query", "SELECT 1"}}}}, {}).valid, "an unversioned SQL node stays fail closed in a plan");
        std::cout << "Data Studio capability checks passed: " << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
