#include "text_loader.h"

#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/formats/text_dataset.h"

#include <spdlog/spdlog.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {

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

                cyxwiz::TextDataset probe(path, probe_cfg);
                auto info = probe.GetInfo();

                task.ReportProgress(0.9f, "Registering dataset");

                auto& reg = cyxwiz::DataRegistry::Instance();
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
                text_entry.num_samples    = info.num_samples;
                text_entry.num_classes    = info.num_classes;
                text_entry.class_names    = info.class_names;
                text_entry.vocab_size     = probe.GetVocabSize();

                reg.RegisterTextDataset(name, text_entry);

                state->success      = true;
                state->backend      = 5;
                state->rows         = static_cast<int64_t>(info.num_samples);
                state->cols         = 1;
                // Rough bytes estimate: max_length tokens * sizeof(float)
                // per sample. TextDatasetBatcher materializes the real
                // tensors lazily so this is a ceiling, not an actual
                // allocation.
                size_t per_sample   = static_cast<size_t>(max_length) * sizeof(float);
                state->bytes        = info.num_samples * per_sample;
                state->num_classes  = info.num_classes;
                state->vocab_size   = probe.GetVocabSize();
                state->message      = "Loaded " + std::to_string(info.num_samples) +
                                      " text samples (" + std::to_string(info.num_classes) +
                                      " classes, vocab " + std::to_string(probe.GetVocabSize()) + ")";
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
    cyxwiz::DataRegistry::Instance().UnregisterTextDataset(name);
}

}  // namespace cyxwiz::loaders
