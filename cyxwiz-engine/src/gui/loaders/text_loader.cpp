#include "text_loader.h"
#include "text_csv_preflight.h"

#include "../../core/async_task_manager.h"
#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/formats/text_dataset.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/text_arrow_adapter.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {
namespace {

std::string LowerExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return ext;
}

bool IsArrowNativeTextFile(const fs::path& path) {
    const std::string ext = LowerExtension(path);
    return fs::is_regular_file(path) &&
           (ext == ".parquet" || ext == ".pq" ||
            ext == ".feather" || ext == ".fea" ||
            ext == ".arrow" || ext == ".ipc");
}

void RequireColumn(const std::shared_ptr<arrow::Table>& table,
                   const std::string& column,
                   const std::string& role) {
    if (column.empty()) {
        throw std::runtime_error(role + " column is not configured");
    }
    if (!table || !table->GetColumnByName(column)) {
        throw std::runtime_error(role + " column '" + column + "' was not found");
    }
}

// Rows with text and the classes of those rows: an empty text is not a
// sample (the same rule as the folder / JSON reader).
struct TextTableStats {
    size_t samples = 0;
    std::vector<std::string> class_names;
};

std::string CellText(const arrow::Array& array, int64_t row) {
    if (array.type_id() == arrow::Type::STRING)
        return std::string(static_cast<const arrow::StringArray&>(array).GetView(row));
    if (array.type_id() == arrow::Type::LARGE_STRING)
        return std::string(static_cast<const arrow::LargeStringArray&>(array).GetView(row));
    auto scalar = array.GetScalar(row);
    return scalar.ok() && *scalar && (*scalar)->is_valid ? (*scalar)->ToString() : std::string();
}

TextTableStats CountTextRows(const std::shared_ptr<arrow::Table>& table,
                             const std::string& text_column,
                             const std::string& label_column) {
    TextTableStats stats;
    const auto text = table->GetColumnByName(text_column);
    const auto label = label_column.empty() ? nullptr : table->GetColumnByName(label_column);
    std::set<std::string> classes;
    for (int c = 0; c < text->num_chunks(); ++c) {
        const auto& text_chunk = *text->chunk(c);
        const auto label_chunk = label ? label->chunk(c) : nullptr;
        for (int64_t row = 0; row < text_chunk.length(); ++row) {
            if (text_chunk.IsNull(row) || CellText(text_chunk, row).empty()) continue;
            ++stats.samples;
            if (label_chunk && !label_chunk->IsNull(row)) classes.insert(CellText(*label_chunk, row));
        }
    }
    stats.class_names.assign(classes.begin(), classes.end());
    return stats;
}

}  // namespace

bool TextLoader::ValidateApplyContext(const ApplyContext& ctx,
                                      std::string& err) const {
    if (ctx.source_path.empty()) {
        err = "Text load needs a file or folder path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
        return false;
    }
    return true;
}

uint64_t TextLoader::LaunchAsyncLoad(const ApplyContext& ctx,
                                     std::shared_ptr<AsyncLoadState> state) {
    if (!state) return 0;

    auto& registry = cyxwiz::DataRegistry::Instance();

    // Cross-category cleanup: if the user toggled from Tabular to
    // Text on the same file, clear the stale Tabular entry so
    // IsArrowDataset / IsParquetBackedDataset on this name don't
    // mislead downstream code. Safe on the UI thread — different
    // registry map than the one the async worker will register into.
    registry.UnregisterTabularDataset(ctx.dataset_name);

    // Drop the previous text entry if the user re-Applied with a
    // different file (different dataset_name). Matches the pre-
    // refactor behavior.
    if (!ctx.previous_dataset_name.empty() &&
        ctx.previous_dataset_name != ctx.dataset_name) {
        registry.UnregisterTextDataset(ctx.previous_dataset_name);
        registry.UnregisterTabularDataset(ctx.previous_dataset_name);
    }
    // NOTE: we intentionally do NOT pre-clear a same-name text entry
    // here. RegisterTextDataset uses map[name]=entry which atomically
    // replaces any prior entry, so a premature unregister just opens a
    // window where the registry is briefly empty — during which a
    // stray Compile click would correctly fail with "Data not loaded".
    // Let the worker do the atomic replace.

    // Snapshot everything by value for the worker.
    const std::string path       = ctx.source_path;
    const std::string name       = ctx.dataset_name;
    const std::string text_col   = ctx.text_column;
    const std::string label_col  = ctx.text_label_column;
    const bool has_labels        = ctx.text_has_labels;

    state->dataset_name = name;
    state->source_path  = path;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading text " + name,
        [path, name, text_col, label_col, has_labels, state]
        (cyxwiz::LambdaTask& task) {
            try {
                // Data Input only reads the text and its label (TOFIX129 B-8):
                // the Text Tokenizer node tokenizes.
                task.ReportProgress(0.1f, "Reading text");

                auto& reg = cyxwiz::DataRegistry::Instance();

                fs::path source(path);
                const auto source_check = ValidateTextSource(path);
                if (!source_check.ok) {
                    throw std::runtime_error(source_check.message);
                }
                const std::string ext = LowerExtension(source);
                const bool delimited = fs::is_regular_file(source) && (ext == ".csv" || ext == ".tsv");
                size_t num_samples = 0;
                std::vector<std::string> class_names;
                std::shared_ptr<arrow::Table> raw_table;

                if (delimited || IsArrowNativeTextFile(source)) {
                    std::shared_ptr<cyxwiz::ArrowDataset> raw_arrow;
                    if (delimited) {
                        const char delimiter = (ext == ".tsv") ? '\t' : ',';
                        const auto csv_preflight = ValidateTextCsvRowWidths(path, delimiter);
                        if (!csv_preflight.ok) {
                            throw std::runtime_error(csv_preflight.message);
                        }
                        auto parse_options = arrow::csv::ParseOptions::Defaults();
                        parse_options.delimiter = delimiter;
                        parse_options.newlines_in_values = true;
                        raw_arrow = cyxwiz::ArrowDataset::FromCSV(
                            path, name, arrow::csv::ReadOptions::Defaults(), parse_options,
                            arrow::csv::ConvertOptions::Defaults());
                    } else {
                        raw_arrow = cyxwiz::ArrowDataset::FromFile(path, name);
                    }
                    if (!raw_arrow || !raw_arrow->GetArrowTable()) {
                        throw std::runtime_error("failed to read the text table");
                    }
                    raw_table = raw_arrow->GetArrowTable();
                    RequireColumn(raw_table, text_col, "Text");
                    if (has_labels) {
                        RequireColumn(raw_table, label_col, "Label");
                    }
                    task.ReportProgress(0.6f, "Counting samples");
                    auto combined = raw_table->CombineChunks();
                    const auto stats = CountTextRows(combined.ok() ? *combined : raw_table, text_col,
                                                     has_labels ? label_col : std::string());
                    num_samples = stats.samples;
                    class_names = stats.class_names;
                } else {
                    cyxwiz::TextDatasetConfig raw_cfg;
                    raw_cfg.text_column  = text_col;
                    raw_cfg.label_column = label_col;
                    raw_cfg.has_labels   = has_labels;
                    cyxwiz::TextDataset raw(path, raw_cfg);
                    auto raw_table_result = cyxwiz::BuildRawTextArrowTable(raw, text_col, label_col);
                    if (!raw_table_result.ok() || !raw_table_result.ValueOrDie()) {
                        throw std::runtime_error(
                            "failed to build raw text Arrow table" +
                            (raw_table_result.ok() ? std::string() : ": " + raw_table_result.status().ToString()));
                    }
                    raw_table = raw_table_result.ValueOrDie();
                    const auto info = raw.GetInfo();
                    num_samples = info.num_samples;
                    class_names = info.class_names;
                }
                const auto samples_check = ValidateTextSamples(num_samples, path);
                if (!samples_check.ok) {
                    throw std::runtime_error(samples_check.message);
                }
                if (!reg.RegisterArrowTable(raw_table, name)) {
                    throw std::runtime_error("failed to store the text table");
                }
                const size_t num_classes = class_names.size();
                const auto stored = reg.GetArrowDataset(name);
                const size_t table_bytes = stored ? stored->GetMemoryUsage() : 0;

                task.ReportProgress(0.9f, "Registering text metadata");

                cyxwiz::DataRegistry::TextDatasetEntry text_entry;
                text_entry.source_path    = path;
                text_entry.text_column    = text_col;
                text_entry.label_column   = label_col;
                text_entry.has_labels     = has_labels;
                text_entry.num_samples    = num_samples;
                text_entry.num_classes    = num_classes;
                text_entry.class_names    = class_names;

                reg.RegisterTextDataset(name, text_entry);

                state->success      = true;
                state->backend      = 5;
                state->rows         = static_cast<int64_t>(num_samples);
                state->cols         = 1;
                state->bytes        = table_bytes;
                state->num_classes  = num_classes;
                state->message      = "Loaded " + std::to_string(num_samples) +
                                      " text samples (" + std::to_string(num_classes) + " classes)";
                auto audit = cyxwiz::DatasetAudit::AuditText(name, text_entry);
                state->audit_errors = audit.ErrorCount();
                state->audit_warnings = audit.WarningCount();
                state->audit_message = cyxwiz::FormatAuditSummary(audit);
                state->audit_issue_lines = cyxwiz::FormatAuditIssueLines(audit);
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error loading text: ") + e.what();
                spdlog::error("TextLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST.
            state->done.store(true);
            // The dialog reads the published state; the task itself must also
            // end Failed (not Completed) so the Tasks panel tells the truth.
            if (!state->success && !task.ShouldStop()) {
                throw std::runtime_error(state->message.empty() ? std::string("Load failed") : state->message);
            }
        });
}

bool TextLoader::IsRegistered(const std::string& name) const {
    return cyxwiz::DataRegistry::Instance().IsTextDataset(name);
}

void TextLoader::Unregister(const std::string& name) {
    auto& reg = cyxwiz::DataRegistry::Instance();
    reg.UnregisterTextDataset(name);
    reg.UnregisterTabularDataset(name);
}

bool TextLoader::RestoreFromRegistry(const std::string& name,
                                     const gui::MLNode& /*node*/,
                                     RestoreState& out) const {
    auto* entry = cyxwiz::DataRegistry::Instance().GetTextDatasetEntry(name);
    if (!entry) return false;

    out.found   = true;
    out.rows    = static_cast<int64_t>(entry->num_samples);
    out.cols    = 1;
    const auto stored = cyxwiz::DataRegistry::Instance().GetArrowDataset(name);
    out.bytes   = stored ? stored->GetMemoryUsage() : 0;
    out.backend = 5;
    out.memory_is_estimate = false;
    out.status_message = "Loaded " + name + " (" +
        std::to_string(entry->num_samples) + " text samples, " +
        std::to_string(entry->num_classes) + " classes)";
    return true;
}

CompletedLoadDescription TextLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = false;

    d.node_description_suffix =
        std::to_string(state.rows) + " samples, " +
        std::to_string(state.num_classes) + " classes";

    fs::path p(state.source_path);
    d.default_status_message = std::string("Loaded text from ") +
        p.filename().string();
    return d;
}

cyxwiz::PreprocessingDomain TextLoader::Domain(
    const std::string& /*file_category*/) const {
    return cyxwiz::PreprocessingDomain::Text;
}

bool TextLoader::LaunchTraining(
    cyxwiz::TrainingConfiguration /*config*/,
    const std::string& dataset_name,
    const std::string& /*label_column*/,
    int /*epochs*/,
    int /*batch_size*/,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> /*plot_panel*/,
    std::function<void(bool)> /*node_editor_callback*/) {
    spdlog::error("TextLoader: text dataset '{}' cannot train as raw text. Add a Text Tokenizer "
                  "(or TF-IDF / Count Vectorizer) node; text trains through its tokenized table.",
                  dataset_name);
    return false;
}

std::vector<ParamSchema> TextLoader::NodeParams() const {
    return {
        {"text_layout",         "0",    "0=SingleFile, 1=CorpusSubdirs"},
        {"text_column",         "text", "Column with the text"},
        {"text_label_column",   "label","Label column (single-file mode)"},
    };
}

SyntheticBatch TextLoader::MakeSynthetic(
    const cyxwiz::TrainingConfiguration& config, uint32_t seed) const {
    return MakeSyntheticForDomain(
        config, cyxwiz::PreprocessingDomain::Text, seed, "Text");
}

}  // namespace cyxwiz::loaders
