#include "text_loader.h"
#include "text_csv_preflight.h"

#include "../../core/async_task_manager.h"
#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/formats/text_dataset.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/text_arrow_adapter.h"
#include "../../core/training_manager.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

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

std::vector<std::string> CollectClassNames(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& label_column) {
    std::set<std::string> classes;
    if (!table || label_column.empty()) {
        return {};
    }

    auto column = table->GetColumnByName(label_column);
    if (!column) {
        return {};
    }

    for (const auto& chunk : column->chunks()) {
        if (!chunk) continue;
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) continue;
            auto scalar = chunk->GetScalar(row);
            if (!scalar.ok() || !*scalar || !(*scalar)->is_valid) continue;
            classes.insert((*scalar)->ToString());
        }
    }
    return {classes.begin(), classes.end()};
}

size_t EstimateVocabularySize(const std::shared_ptr<arrow::Table>& table,
                              const std::string& text_column,
                              bool lowercase,
                              int max_vocab) {
    if (!table || text_column.empty()) {
        return 0;
    }
    auto column = table->GetColumnByName(text_column);
    if (!column) {
        return 0;
    }

    const size_t cap = max_vocab > 0 ? static_cast<size_t>(max_vocab) : 0;
    std::unordered_set<std::string> vocab;
    for (const auto& chunk : column->chunks()) {
        if (!chunk) continue;
        for (int64_t row = 0; row < chunk->length(); ++row) {
            if (chunk->IsNull(row)) continue;
            auto scalar = chunk->GetScalar(row);
            if (!scalar.ok() || !*scalar || !(*scalar)->is_valid) continue;
            std::string text = (*scalar)->ToString();
            if (lowercase) {
                std::transform(text.begin(), text.end(), text.begin(),
                               [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                               });
            }
            std::istringstream stream(text);
            std::string token;
            while (stream >> token) {
                vocab.insert(std::move(token));
                if (cap > 0 && vocab.size() >= cap) {
                    return vocab.size();
                }
            }
        }
    }
    return vocab.size();
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
    const int tok_type           = ctx.text_tokenizer_type;
    const int max_length         = ctx.text_max_length;
    const bool lowercase         = ctx.text_lowercase;
    const int min_freq           = ctx.text_min_freq;
    const int max_vocab          = ctx.text_max_vocab_size;

    state->dataset_name = name;
    state->source_path  = path;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading text " + name,
        [path, name, text_col, label_col, has_labels, tok_type, max_length,
         lowercase, min_freq, max_vocab, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.1f, "Building tokenizer");

                cyxwiz::TextDatasetConfig probe_cfg;
                probe_cfg.text_column  = text_col;
                probe_cfg.label_column = label_col;
                probe_cfg.has_labels   = has_labels;
                switch (tok_type) {
                    case 0: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Whitespace; break;
                    case 2: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Character; break;
                    default: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Word; break;
                }
                probe_cfg.max_length     = max_length;
                probe_cfg.lowercase      = lowercase;
                probe_cfg.do_padding     = true;
                probe_cfg.do_truncation  = true;
                probe_cfg.min_word_freq  = min_freq;
                probe_cfg.max_vocab_size = max_vocab;

                task.ReportProgress(0.75f, "Registering raw text table");

                auto& reg = cyxwiz::DataRegistry::Instance();

                fs::path source(path);
                const std::string ext = LowerExtension(source);
                const bool native_arrow_text_file =
                    fs::is_regular_file(source) &&
                    (ext == ".csv" || ext == ".tsv");
                size_t num_samples = 0;
                size_t num_classes = 0;
                std::vector<std::string> class_names;
                size_t vocab_size = 0;

                if (native_arrow_text_file) {
                    cyxwiz::TextDataset probe(path, probe_cfg);
                    auto info = probe.GetInfo();

                    const char delimiter = (ext == ".tsv") ? '\t' : ',';
                    const auto csv_preflight =
                        ValidateTextCsvRowWidths(path, delimiter);
                    if (!csv_preflight.ok) {
                        throw std::runtime_error(csv_preflight.message);
                    }

                    auto read_options = arrow::csv::ReadOptions::Defaults();
                    auto parse_options = arrow::csv::ParseOptions::Defaults();
                    auto convert_options = arrow::csv::ConvertOptions::Defaults();
                    parse_options.delimiter = delimiter;
                    parse_options.newlines_in_values = true;

                    auto raw_arrow = cyxwiz::ArrowDataset::FromCSV(
                        path, name, read_options, parse_options, convert_options);
                    if (!raw_arrow || !raw_arrow->GetArrowTable()) {
                        throw std::runtime_error(
                            "failed to register text CSV as Arrow table");
                    }
                    if (!reg.RegisterArrowTable(raw_arrow->GetArrowTable(), name)) {
                        throw std::runtime_error(
                            "failed to store text CSV Arrow table");
                    }
                    num_samples = info.num_samples;
                    num_classes = info.num_classes;
                    class_names = info.class_names;
                    vocab_size = probe.GetVocabSize();
                } else if (IsArrowNativeTextFile(source)) {
                    auto raw_arrow = cyxwiz::ArrowDataset::FromFile(path, name);
                    if (!raw_arrow || !raw_arrow->GetArrowTable()) {
                        throw std::runtime_error(
                            "failed to load Arrow-native text table");
                    }
                    auto raw_table = raw_arrow->GetArrowTable();
                    RequireColumn(raw_table, text_col, "Text");
                    if (has_labels) {
                        RequireColumn(raw_table, label_col, "Label");
                    }
                    if (!reg.RegisterArrowTable(raw_table, name)) {
                        throw std::runtime_error(
                            "failed to store Arrow-native text table");
                    }

                    class_names = has_labels
                        ? CollectClassNames(raw_table, label_col)
                        : std::vector<std::string>{};
                    num_samples = static_cast<size_t>(raw_table->num_rows());
                    num_classes = class_names.size();
                    vocab_size = EstimateVocabularySize(raw_table, text_col,
                                                        lowercase, max_vocab);
                    spdlog::info("TextLoader: '{}' registered Arrow-native text table "
                                 "from {}",
                                 name, path);
                } else {
                    cyxwiz::TextDataset probe(path, probe_cfg);
                    auto info = probe.GetInfo();

                    auto raw_table_result = cyxwiz::BuildRawTextArrowTable(
                        probe, text_col, label_col);
                    if (!raw_table_result.ok()) {
                        throw std::runtime_error(
                            "failed to build raw text Arrow table: " +
                            raw_table_result.status().ToString());
                    }
                    auto raw_table = raw_table_result.ValueOrDie();
                    if (!raw_table) {
                        throw std::runtime_error(
                            "failed to build raw text Arrow table");
                    }
                    if (!reg.RegisterArrowTable(raw_table, name)) {
                        throw std::runtime_error(
                            "failed to store raw text Arrow table");
                    }
                    spdlog::info("TextLoader: '{}' registered raw text Arrow table "
                                 "from TextDataset adapter",
                                 name);
                    num_samples = info.num_samples;
                    num_classes = info.num_classes;
                    class_names = info.class_names;
                    vocab_size = probe.GetVocabSize();
                }

                task.ReportProgress(0.9f, "Registering text metadata");

                cyxwiz::DataRegistry::TextDatasetEntry text_entry;
                text_entry.source_path    = path;
                text_entry.text_column    = text_col;
                text_entry.label_column   = label_col;
                text_entry.has_labels     = has_labels;
                text_entry.tokenizer_type = tok_type;
                text_entry.max_length     = max_length;
                text_entry.lowercase      = lowercase;
                text_entry.do_padding     = true;
                text_entry.do_truncation  = true;
                text_entry.min_word_freq  = min_freq;
                text_entry.max_vocab_size = max_vocab;
                text_entry.num_samples    = num_samples;
                text_entry.num_classes    = num_classes;
                text_entry.class_names    = class_names;
                text_entry.vocab_size     = vocab_size;

                reg.RegisterTextDataset(name, text_entry);

                state->success      = true;
                state->backend      = 5;
                state->rows         = static_cast<int64_t>(num_samples);
                state->cols         = 1;
                // Rough bytes estimate: max_length tokens * sizeof(float)
                // per sample. TextDatasetBatcher materializes the real
                // tensors lazily so this is a ceiling, not an actual
                // allocation.
                size_t per_sample   = static_cast<size_t>(max_length) * sizeof(float);
                state->bytes        = num_samples * per_sample;
                state->num_classes  = num_classes;
                state->vocab_size   = vocab_size;
                state->message      = "Loaded " + std::to_string(num_samples) +
                                      " text samples (" + std::to_string(num_classes) +
                                      " classes, vocab " + std::to_string(vocab_size) + ")";
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
    // Estimate: max_length * sizeof(float) per sample. Actual memory
    // during training is dominated by the tokenized in-RAM corpus,
    // not the training tensor batches.
    const size_t per_sample = static_cast<size_t>(entry->max_length) * sizeof(float);
    out.bytes   = entry->num_samples * per_sample;
    out.backend = 5;
    out.memory_is_estimate = true;
    out.status_message = "Loaded " + name + " (" +
        std::to_string(entry->num_samples) + " text samples, " +
        std::to_string(entry->num_classes) + " classes, vocab " +
        std::to_string(entry->vocab_size) + ")";
    return true;
}

CompletedLoadDescription TextLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = true;

    d.node_description_suffix =
        std::to_string(state.rows) + " samples, " +
        std::to_string(state.num_classes) + " classes, vocab " +
        std::to_string(state.vocab_size);

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
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& /*label_column*/,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback) {
    auto* entry = cyxwiz::DataRegistry::Instance().GetTextDatasetEntry(dataset_name);
    if (!entry) {
        spdlog::error("TextLoader: text dataset '{}' is registered but entry "
                      "could not be retrieved", dataset_name);
        return false;
    }
    if (cyxwiz::DataRegistry::Instance().IsArrowDataset(dataset_name)) {
        spdlog::info("TextLoader: '{}' also has raw Arrow backing. Using legacy "
                     "TextDatasetBatcher because no materialized Cat-1 text table "
                     "was selected; graphs with TextTokenizer materialize to "
                     "'{}__materialized' and route through Arrow training.",
                     dataset_name, dataset_name);
    }
    spdlog::info("TextLoader: Starting text training: dataset={}, epochs={}, "
                 "batch_size={}, num_workers={}, {} samples, {} classes, "
                 "max_length={}, vocab_size={}",
                 dataset_name, epochs, batch_size, config.num_workers,
                 entry->num_samples, entry->num_classes,
                 entry->max_length, entry->vocab_size);
    return cyxwiz::TrainingManager::Instance().StartTrainingText(
        std::move(config), *entry, epochs, batch_size, plot_panel,
        std::move(node_editor_callback));
}

std::vector<ParamSchema> TextLoader::NodeParams() const {
    return {
        {"text_layout",         "0",    "0=SingleFile, 1=CorpusSubdirs"},
        {"text_column",         "text", "Column with text for tokenization"},
        {"text_label_column",   "label","Label column (single-file mode)"},
        {"text_tokenizer_type", "1",    "0=Whitespace, 1=Word, 2=Character"},
        {"text_max_length",     "512",  "Max tokens per sample"},
        {"text_lowercase",      "true", "Lowercase before tokenization"},
        {"text_min_freq",       "1",    "Min word frequency for vocab"},
        {"text_max_vocab_size", "-1",   "Vocab cap (-1 = unlimited)"},
    };
}

SyntheticBatch TextLoader::MakeSynthetic(
    const cyxwiz::TrainingConfiguration& config, uint32_t seed) const {
    return MakeSyntheticForDomain(
        config, cyxwiz::PreprocessingDomain::Text, seed, "Text");
}

}  // namespace cyxwiz::loaders
