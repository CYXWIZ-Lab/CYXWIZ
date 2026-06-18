#pragma once

// Undefine Windows macros that conflict with our method names
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif

#include <string>
#include <map>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <chrono>
#include <vector>
#include <imgui.h>
#include "node_editor.h"
#include "data_input_capabilities.h"
#include "loaders/data_loader.h"
#include "../core/data_convert_service.h"

namespace gui {

/**
 * NodeConfigDialog - Base class for KNIME-style node configuration dialogs
 */
class NodeConfigDialog {
public:
    NodeConfigDialog(const std::string& title, MLNode* node);
    virtual ~NodeConfigDialog() = default;

    bool Render();
    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    virtual void Apply() = 0;
    virtual void Reset() = 0;
    virtual ImVec2 GetDefaultSize() const { return ImVec2(600, 500); }

    // Subclasses return true while a long-running async operation is in
    // flight (e.g. an async data load). The base Render() greys out OK
    // and Apply while busy so the user can't fire two concurrent loads.
    // Cancel stays enabled regardless.
    virtual bool IsBusy() const { return false; }

    // Optional graph context. Properties.cpp sets this right after
    // constructing a dialog so visualization / inspection dialogs can
    // walk upstream pins to auto-populate dataset hints. Most dialogs
    // ignore it. Stored as a raw pointer — the NodeEditor outlives the
    // Properties panel.
    void SetNodeEditor(NodeEditor* editor) { node_editor_ = editor; }

protected:
    virtual void RenderContent() = 0;
    void RenderSettingsTab();
    void RenderAdvancedTab();
    void RenderPreviewTab();
    bool FileSelector(const char* label, std::string& path, const char* filter = "All Files\0*.*\0");
    void ValidationMessage(const std::string& message, bool is_error = true);

    std::string title_;
    MLNode* node_;
    NodeEditor* node_editor_ = nullptr;  // Optional, set by Properties.
    bool is_open_ = false;
    bool first_open_ = true;
    int current_tab_ = 0;
    bool has_changes_ = false;
    std::map<std::string, std::string> original_params_;
};

/**
 * CSVReaderDialog - Configuration dialog for CSV/File Input nodes
 */
class CSVReaderDialog : public NodeConfigDialog {
public:
    CSVReaderDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(800, 600); }

protected:
    void RenderContent() override;

private:
    void RenderFileSettingsTab();
    void RenderTransformationTab();
    void RenderAdvancedSettingsTab();
    void RenderLimitRowsTab();
    void RenderPreviewTab();
    void LoadPreview();

    char file_path_[512] = {};
    int delimiter_idx_ = 0;
    char custom_delimiter_[8] = {};
    bool has_header_ = true;
    bool skip_empty_lines_ = true;
    int skip_first_n_ = 0;
    int limit_rows_ = 1000;
    std::vector<std::string> preview_headers_;
    std::vector<std::vector<std::string>> preview_rows_;
    bool preview_loaded_ = false;
    std::string preview_error_;
};

/**
 * TokenizerDialog - Configuration dialog for text preprocessing nodes
 */
class TokenizerDialog : public NodeConfigDialog {
public:
    TokenizerDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(780, 620); }

protected:
    void RenderContent() override;

private:
    void RenderTokenizerTab();
    void RenderVocabularyTab();
    void RenderPaddingTab();
    void RenderPreviewTab();
    void LoadFromNode();
    bool BuildVocabularyFile();
    bool InspectVocabularyFile();
    bool IsTokenizerNode() const;
    bool IsVocabularyNode() const;
    bool IsPaddingNode() const;

    int tokenizer_type_ = 1;
    int max_length_ = 256;
    int max_vocab_size_ = 10000;
    int min_word_freq_ = 2;
    int pad_value_ = 0;
    bool lowercase_ = true;
    bool padding_ = true;
    bool truncation_ = true;
    bool vocab_build_if_missing_ = false;
    char text_col_[128] = "text";
    char label_col_[128] = "";
    char vocab_file_[260] = "";
    char source_csv_[512] = "";
    char sample_text_[1024] = "Hello world! This is a sample text for tokenization preview.";
    std::vector<std::string> preview_tokens_;
    std::string status_message_;
    bool status_is_error_ = false;
};

/**
 * EmbeddingDialog - Configuration dialog for token embedding lookup layers.
 *
 * The Embedding node is trainable by default. This dialog can also attach
 * a prebuilt text matrix file or create a deterministic starter matrix
 * from the configured vocabulary size/dimension.
 */
class EmbeddingDialog : public NodeConfigDialog {
public:
    EmbeddingDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(760, 560); }

protected:
    void RenderContent() override;

private:
    void LoadFromNode();
    void RenderShapeTab();
    void RenderWeightsTab();
    void RenderAdvancedTab();
    bool BuildAndSaveWeights();
    bool InspectWeightFile();

    int num_embeddings_ = 10000;
    int embedding_dim_ = 256;
    int padding_idx_ = -1;
    float max_norm_ = 0.0f;
    bool freeze_ = false;
    int init_mode_ = 0;  // 0=random normal, 1=random uniform, 2=one-hot/truncated
    char weights_file_[512] = "";
    char output_file_[512] = "";
    std::string status_message_;
    bool status_is_error_ = false;
};

/**
 * FilterDialog - Configuration dialog for Filter/Query nodes
 */
class FilterDialog : public NodeConfigDialog {
public:
    FilterDialog(MLNode* node);
    void Apply() override;
    void Reset() override;

protected:
    void RenderContent() override;

private:
    struct FilterCondition {
        int column_idx = 0;
        int operator_idx = 0;
        char value[256] = {};
        int join_type = 0;
    };
    std::vector<FilterCondition> conditions_;
    std::vector<std::string> available_columns_;
};

/**
 * DataInputDialog - Universal smart data input configuration (KNIME-style)
 *
 * Comprehensive data source support:
 * - File: Tabular (CSV, Excel, Parquet), Image, Audio, Video
 * - ML Dataset: MNIST, CIFAR, ImageNet, HuggingFace, Kaggle
 * - Database: SQLite, PostgreSQL, MySQL
 * - Cloud: S3, GCS, Azure Blob
 *
 * KNIME-style tabbed interface with live preview
 */
class DataInputDialog : public NodeConfigDialog {
public:
    DataInputDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(900, 700); }
    bool IsBusy() const override { return is_loading_async_; }

protected:
    void RenderContent() override;

private:
    // Data source types
    using SourceType = gui::data_input::SourceType;
    using MLDatasetType = gui::data_input::MLDatasetType;
    using DatabaseType = gui::data_input::DatabaseType;
    using ImageLayout = gui::data_input::ImageLayout;
    using AudioLayout = gui::data_input::AudioLayout;
    using TextLayout = gui::data_input::TextLayout;

    // FileCategory lives in cyxwiz::loaders now so the dialog and the
    // loader module share a single definition. Aliased here so existing
    // unqualified `FileCategory::Tabular` usages throughout the dialog
    // .cpp keep compiling unchanged.
    using FileCategory = cyxwiz::loaders::FileCategory;

    // Main tab renderers
    void RenderSourceSelector();
    void RenderFileSource();
    void RenderMLDatasetSource();
    void RenderDatabaseSource();
    void RenderCloudSource();
    void RenderOptionsPanel();
    void RenderPreviewPanel();
    void RenderDatasetSummaryPanel();

    // File source sub-renderers
    void RenderTabularOptions();
    void RenderImageOptions();
    void RenderAudioOptions();
    void RenderVideoOptions();
    void RenderTextOptions();

    // KNIME-style tabs
    void RenderTransformationTab();
    void RenderLimitRowsTab();
    void RenderEncodingTab();
    void RenderMemoryTab();
    void RenderAuditTab();
    void RenderMLDatasetOptions();

    // ML Dataset sub-renderers
    void RenderBuiltinDatasets();
    void RenderImageFolderPicker();
    void RenderHuggingFaceConfig();
    void RenderKaggleConfig();

    // Database sub-renderers
    void RenderDatabaseConnection();
    void RenderSQLQuery();

    // Preview renderers
    void RenderTabularPreview();
    void RenderImagePreview();
    void RenderAudioPreview();
    void RenderTextPreview();

    // Core functionality
    void DetectFileType();
    void DetectFileCategory();
    void LoadPreview();
    void LoadColumnList();
    void UpdateRAMEstimate();
    void BrowseFile();
    void BrowseFolder();
    bool IsApplySupported() const;
    bool IsPreviewSupported() const;
    const char* UnsupportedApplyMessage() const;
    const char* PreviewUnavailableMessage() const;
    void MarkApplyUnsupported(const char* message);
    std::string CurrentSourcePath() const;
    std::string CurrentSourceLabel() const;
    std::string CurrentApplySummary() const;
    const char* BackendSummary() const;
    const char* GetFileTypeName() const;

    // STATE: Source selection
    SourceType source_type_ = SourceType::File;
    FileCategory file_category_ = FileCategory::Tabular;
    MLDatasetType ml_dataset_type_ = MLDatasetType::MNIST;
    DatabaseType database_type_ = DatabaseType::SQLite;

    // STATE: File source
    char file_path_[512] = {};
    char folder_path_[512] = {};
    // 0=Auto, 1=CSV, 2=TSV, 3=JSON, 4=Parquet, 5=Excel, 6=HDF5, 7=Feather, 8=Arrow, 9=TXT, 10=ARFF
    int detected_type_ = 0;
    int encoding_idx_ = 0;
    size_t file_size_ = 0;

    // STATE: Tabular format options
    int delimiter_idx_ = 0;
    char custom_delimiter_[8] = ",";
    bool has_header_ = true;
    char quote_char_[4] = "\"";
    int sheet_idx_ = 0;
    char sheet_range_[64] = {};
    bool json_lines_ = false;
    char json_path_[128] = {};
    char hdf5_dataset_[128] = "data";

    // STATE: Image options
    ImageLayout image_layout_ = ImageLayout::ClassSubdirs;
    int target_width_ = 224;
    int target_height_ = 224;
    bool normalize_images_ = true;
    bool rgb_mode_ = true;
    char labels_csv_[512] = {};

    // STATE: Audio options
    int sample_rate_ = 16000;
    float duration_sec_ = 5.0f;     // pad/truncate to this length
    bool mono_ = true;

    // STATE: Text options (Phase 3). Populated by RenderTextOptions
    // and consumed by the text branch of Apply(). The dialog's text
    // category targets CSV / JSON / TXT files with an explicit text
    // column and optional label column — the tokenizer then builds
    // its vocabulary from the text corpus.
    char text_column_[128] = "text";
    char text_label_column_[128] = "label";
    int text_tokenizer_type_ = 1;   // 0=Whitespace, 1=Word, 2=Character
    int text_max_length_ = 512;
    bool text_lowercase_ = true;
    int text_min_freq_ = 1;
    int text_max_vocab_size_ = -1;  // -1 = unlimited

    AudioLayout audio_layout_ = AudioLayout::ClassSubdirs;

    // Text layout — analogous to ImageLayout/AudioLayout, but with
    // "SingleFile" meaning CSV/JSON/TXT picked via the shared file
    // picker (the default — this is how most Kaggle text datasets
    // ship) and "CorpusSubdirs" meaning folder/<class>/*.txt with a
    // folder picker exposed inline in RenderTextOptions. The underlying
    // TextDataset class auto-detects file vs directory at construction
    // time, so the dispatch is purely a dialog-level choice about
    // which picker to show.
    TextLayout text_layout_ = TextLayout::SingleFile;
    char audio_labels_csv_[512] = {};
    char audio_filename_col_[64] = {};   // empty = auto-detect
    char audio_label_col_[64] = {};      // empty = auto-detect

    // STATE: ML Dataset options
    char dataset_name_[128] = "mnist";
    char dataset_subset_[64] = "";
    char cache_dir_[512] = "";
    char hf_token_[256] = "";
    char kaggle_slug_[256] = "";

    // STATE: Database options
    char db_host_[128] = "localhost";
    int db_port_ = 5432;
    char db_name_[128] = "";
    char db_user_[64] = "";
    char db_password_[128] = "";
    char db_file_[512] = "";
    char sql_query_[2048] = "SELECT * FROM table_name LIMIT 1000";

    // STATE: Cloud options
    char cloud_bucket_[256] = "";
    char cloud_path_[512] = "";
    char cloud_credentials_[512] = "";

    // STATE: Column selection
    bool select_all_columns_ = true;
    std::vector<std::string> available_columns_;
    std::vector<bool> selected_columns_;
    int label_column_idx_ = -1;  // Index of label/target column (-1 = none selected)

    // STATE: Row filter
    int skip_rows_ = 0;
    int max_rows_ = 0;
    char where_clause_[256] = {};

    // (Removed: memory_policy_, auto_chunk_size_, chunk_size_kb_, lru_chunks_,
    //  prefetch_enabled_. These were remnants of the pre-Arrow streaming/LRU
    //  path that never shipped. v0.2.0 uses Arrow as the sole tabular data
    //  path with automatic block_size detection in LoadCSVToArrow.)

    // STATE: Advanced-tab escape hatch for the Parquet disk-backed path.
    // Default off — LoadTabularCSV picks in-memory vs disk-backed automatically
    // based on file size. Setting this to true forces the Parquet cache path
    // even for small datasets, so power users can test/verify that path on
    // their own machine. Tracked separately from the auto-detect so the
    // default UX stays fully automatic.
    bool force_disk_backed_ = false;

    // STATE: Which backing store the most recent Apply() actually used.
    // 0 = not loaded, 1 = in-memory Arrow, 2 = disk-backed Parquet.
    // Used by the Memory tab's Current Status section to tell the user
    // which path the engine picked for their data.
    int loaded_backend_ = 0;

    // STATE: Async load state for every category path. The CSV /
    // Parquet-convert / image-scan / text-tokenize / audio-scan steps
    // can each take multiple seconds on real datasets, so they run on
    // AsyncTaskManager workers. The worker writes into AsyncLoadState
    // and sets done.store(true) LAST; PollAsyncLoadResult drains it on
    // the UI thread on the next frame.
    //
    // The struct lives in cyxwiz::loaders so the loader module and the
    // dialog share one definition. Aliased locally to keep existing
    // unqualified `AsyncLoadState` usages in the dialog .cpp working.
    //
    // shared_ptr ownership means the worker can outlive the dialog —
    // its capture keeps the memory alive until the task finishes.
    using AsyncLoadState = cyxwiz::loaders::AsyncLoadState;
    std::shared_ptr<AsyncLoadState> async_load_state_;
    bool is_loading_async_ = false;
    uint64_t loading_task_id_ = 0;
    // Animated dot count for the "Loading..." UI placeholder.
    float loading_anim_phase_ = 0.0f;

    // Drains the async load result if the worker has finished. Called from
    // RenderContent() every frame; cheap when no load is in flight. Writes
    // back to loaded_*, node_->parameters, and apply_status_message_.
    void PollAsyncLoadResult();

    // STATE: Preview
    std::vector<std::string> preview_columns_;
    std::vector<std::vector<std::string>> preview_data_;
    std::vector<ImTextureID> preview_image_textures_;
    std::vector<std::string> preview_image_labels_;
    std::vector<std::pair<std::string, size_t>> label_distribution_;
    std::string label_distribution_column_;
    size_t label_distribution_total_ = 0;
    bool preview_loaded_ = false;
    std::string preview_error_;
    float estimated_ram_mb_ = 0.0f;

    // STATE: Apply feedback
    bool apply_in_progress_ = false;
    bool apply_success_ = false;
    std::string apply_status_message_;
    float apply_status_timer_ = 0.0f;
    std::chrono::steady_clock::time_point apply_started_at_{};
    float last_load_elapsed_ms_ = -1.0f;
    int64_t loaded_rows_ = 0;
    int64_t loaded_cols_ = 0;
    size_t loaded_memory_bytes_ = 0;
    // True for image/audio datasets where loaded_memory_bytes_ is the
    // *if-fully-cached* size estimate, not actual RAM use. The Memory tab
    // displays "(estimated, lazy)" so users don't think the dataset is
    // sitting in RAM when it isn't.
    bool loaded_memory_is_estimate_ = false;
    std::string loaded_dataset_name_;
    std::vector<std::string> audit_issue_lines_;

    // STATE: Data profiling
    struct ColumnStats {
        std::string name;
        std::string dtype;      // "integer", "numeric", "string", "boolean"
        size_t count = 0;
        size_t unique_count = 0;
        size_t null_count = 0;
        float null_percentage = 0.0f;
        double min_val = 0.0;
        double max_val = 0.0;
        double mean = 0.0;
        double std_dev = 0.0;
    };
    bool profile_computed_ = false;
    bool profile_in_progress_ = false;
    std::vector<ColumnStats> column_stats_;
    float data_quality_score_ = 0.0f;

    // STATE: Memory management
    enum class DataLoadState { NotLoaded, OnDisk, InMemory };
    DataLoadState data_load_state_ = DataLoadState::NotLoaded;

    // Helper methods
    void UpdateTextLabelDistribution();
    void RenderTextLabelDistribution();
    void RenderDataProfilingTab();
    void ComputeDataProfile();
    void UnloadDataset();
    std::string GenerateDatasetName() const;
    static std::string FormatBytes(size_t bytes);
};

/**
 * DataOutputDialog - Universal smart data output configuration (KNIME-style)
 */
class DataOutputDialog : public NodeConfigDialog {
public:
    DataOutputDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(700, 500); }

protected:
    void RenderContent() override;

private:
    void RenderSettingsTab();
    void RenderAdvancedTab();

    char file_path_[512] = {};
    int output_type_ = 0;
    bool overwrite_ = false;
    bool include_header_ = true;
    int compression_ = 0;
};

/**
 * DataConvertDialog - Data utility dialog for one-time dataset format conversion.
 */
class DataConvertDialog : public NodeConfigDialog {
public:
    DataConvertDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(820, 620); }

protected:
    void RenderContent() override;

private:
    void LoadFromNode();
    void RenderSourceTab();
    void RenderOutputTab();
    void RenderOptionsTab();
    void RenderPreviewTab();
    void RenderRunTab();
    void RenderLogsTab();
    cyxwiz::DataConvertOptions BuildOptions() const;
    void PreviewInput();
    void RunConversion();
    void SetStatus(std::string message, bool is_error);
    void AddLogLine(const std::string& message);

    char input_path_[512] = {};
    char output_path_[512] = {};
    int input_format_ = 0;
    int output_format_ = 0;
    char delimiter_[8] = ",";
    bool auto_detect_delimiter_ = true;
    bool has_header_ = true;
    bool allow_newlines_in_values_ = true;
    int skip_rows_ = 0;
    int compression_ = 1;
    int row_group_size_ = 1048576;
    bool overwrite_ = false;
    bool create_parent_dirs_ = true;
    bool write_manifest_ = true;
    cyxwiz::DataConvertPreview preview_;
    cyxwiz::DataConvertResult last_result_;
    std::string status_message_ = "Not run";
    bool status_is_error_ = false;
    std::vector<std::string> log_lines_;
};

/**
 * DataLoaderDialog - Configures batch iteration: batch size, shuffle, drop_last, num_workers
 * Single source of truth for batching behavior. Optimizer nodes should NOT carry batch_size.
 */
class DataLoaderDialog : public NodeConfigDialog {
public:
    DataLoaderDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(560, 560); }

protected:
    void RenderContent() override;

private:
    int epochs_ = 10;
    int batch_size_ = 32;
    bool shuffle_ = true;
    bool drop_last_ = false;
    int num_workers_ = 0;
    int prefetch_factor_ = 2;
    bool save_best_checkpoint_ = true;
    int early_stopping_patience_ = 5;
    char checkpoint_dir_[512] = "";
};

/**
 * DataSplitDialog - Configures train/val/test ratios and split seed
 * Single source of truth for dataset partitioning.
 */
class DataSplitDialog : public NodeConfigDialog {
public:
    DataSplitDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(520, 420); }

protected:
    void RenderContent() override;

private:
    float train_ratio_ = 0.8f;
    float val_ratio_ = 0.1f;
    float test_ratio_ = 0.1f;
    int seed_ = 42;
    bool stratified_ = true;
};

/**
 * NodeConfigDialogFactory - Creates appropriate dialog for a node type
 */
class NodeConfigDialogFactory {
public:
    static NodeConfigDialogFactory& Instance();
    using DialogCreator = std::function<std::unique_ptr<NodeConfigDialog>(MLNode*)>;
    void RegisterDialog(NodeType type, DialogCreator creator);
    bool HasDialog(NodeType type) const;
    std::unique_ptr<NodeConfigDialog> CreateDialog(MLNode* node);

private:
    NodeConfigDialogFactory();
    std::map<NodeType, DialogCreator> creators_;
};

bool ShouldShowOpenDialogButton(NodeType type);

} // namespace gui
