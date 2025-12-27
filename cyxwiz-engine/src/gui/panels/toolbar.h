#pragma once

#include "../panel.h"
#include "plot_window.h"
#include "auth/auth_client.h"
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <future>

// Forward declarations to avoid including device.h (which pulls in CUDA/OpenCL headers)
namespace cyxwiz {
    enum class DeviceType;
    struct DeviceInfo;
}

namespace cyxwiz {

// Tool entry for command palette search
struct ToolEntry {
    std::string name;           // Display name (e.g., "K-Means Clustering")
    std::string category;       // Category (e.g., "Clustering", "Statistics")
    std::string keywords;       // Search keywords (e.g., "cluster kmeans machine learning")
    std::string icon;           // FontAwesome icon code
    std::string shortcut;       // Keyboard shortcut if any (e.g., "Ctrl+P")
    std::function<void()> callback;  // Action to execute

    // Fuzzy match score (used during search)
    mutable int match_score = 0;
};

/**
 * Top Toolbar Panel
 * Renders main menu bar with File, Edit, View, Nodes, Train, Dataset, Script, Deploy, Plots, Help
 */
class ToolbarPanel : public Panel {
public:
    ToolbarPanel();
    ~ToolbarPanel() override = default;

    void Render() override;

    // Command Palette (Ctrl+P)
    void OpenCommandPalette();
    void HandleGlobalShortcuts();  // Call this from main window to handle Ctrl+P
    bool IsCommandPaletteOpen() const { return show_command_palette_; }

    // User profile popup (for custom title bar integration)
    void ToggleUserProfilePopup() { show_user_profile_popup_ = !show_user_profile_popup_; }

    // Callbacks
    void SetResetLayoutCallback(std::function<void()> callback) { reset_layout_callback_ = callback; }
    void SetSaveLayoutCallback(std::function<void()> callback) { save_layout_callback_ = callback; }
    void SetSaveProjectSettingsCallback(std::function<void()> callback) { save_project_settings_callback_ = callback; }
    void SetTogglePlotTestControlCallback(std::function<void()> callback) { toggle_plot_test_control_callback_ = callback; }
    void SetConnectToServerCallback(std::function<void()> callback) { connect_to_server_callback_ = callback; }
    void SetDeployToServerCallback(std::function<void()> callback) { deploy_to_server_callback_ = callback; }
    void SetImportDatasetCallback(std::function<void()> callback) { import_dataset_callback_ = callback; }
    void SetOpenCustomNodeEditorCallback(std::function<void()> callback) { open_custom_node_editor_callback_ = callback; }
    void SetOpenThemeEditorCallback(std::function<void()> callback) { open_theme_editor_callback_ = callback; }
    void SetOpenProfilerCallback(std::function<void()> callback) { open_profiler_callback_ = callback; }
    void SetOpenMemoryMonitorCallback(std::function<void()> callback) { open_memory_monitor_callback_ = callback; }
    void SetNewScriptCallback(std::function<void()> callback) { new_script_callback_ = callback; }
    void SetOpenScriptCallback(std::function<void()> callback) { open_script_callback_ = callback; }
    void SetOpenScriptInEditorCallback(std::function<void(const std::string&)> callback) { open_script_in_editor_callback_ = callback; }
    void SetSaveAllCallback(std::function<void()> callback) { save_all_callback_ = callback; }
    void SetAccountSettingsCallback(std::function<void()> callback) { account_settings_callback_ = callback; }
    void SetExitCallback(std::function<void()> callback) { exit_callback_ = callback; }
    void SetHasUnsavedChangesCallback(std::function<bool()> callback) { has_unsaved_changes_callback_ = callback; }
    void SetOnLoginSuccessCallback(std::function<void(const std::string&)> callback) { on_login_success_callback_ = callback; }
    void SetOnLogoutCallback(std::function<void()> callback) { on_logout_callback_ = callback; }

    // Edit menu callbacks
    void SetUndoCallback(std::function<void()> callback) { undo_callback_ = callback; }
    void SetRedoCallback(std::function<void()> callback) { redo_callback_ = callback; }
    void SetCutCallback(std::function<void()> callback) { cut_callback_ = callback; }
    void SetCopyCallback(std::function<void()> callback) { copy_callback_ = callback; }
    void SetPasteCallback(std::function<void()> callback) { paste_callback_ = callback; }
    void SetDeleteCallback(std::function<void()> callback) { delete_callback_ = callback; }
    void SetSelectAllCallback(std::function<void()> callback) { select_all_callback_ = callback; }
    void SetToggleLineCommentCallback(std::function<void()> callback) { toggle_line_comment_callback_ = callback; }
    void SetToggleBlockCommentCallback(std::function<void()> callback) { toggle_block_comment_callback_ = callback; }
    void SetFindCallback(std::function<void(const std::string&, bool, bool, bool)> callback) { find_callback_ = callback; }
    void SetFindNextCallback(std::function<void()> callback) { find_next_callback_ = callback; }
    void SetReplaceCallback(std::function<void(const std::string&, const std::string&, bool, bool, bool)> callback) { replace_callback_ = callback; }
    void SetReplaceAllCallback(std::function<void(const std::string&, const std::string&, bool, bool, bool)> callback) { replace_all_callback_ = callback; }

    // New Edit menu callbacks
    void SetGoToLineCallback(std::function<void(int)> callback) { go_to_line_callback_ = callback; }
    void SetDuplicateLineCallback(std::function<void()> callback) { duplicate_line_callback_ = callback; }
    void SetMoveLineUpCallback(std::function<void()> callback) { move_line_up_callback_ = callback; }
    void SetMoveLineDownCallback(std::function<void()> callback) { move_line_down_callback_ = callback; }
    void SetIndentCallback(std::function<void()> callback) { indent_callback_ = callback; }
    void SetOutdentCallback(std::function<void()> callback) { outdent_callback_ = callback; }
    void SetTransformUppercaseCallback(std::function<void()> callback) { transform_uppercase_callback_ = callback; }
    void SetTransformLowercaseCallback(std::function<void()> callback) { transform_lowercase_callback_ = callback; }
    void SetTransformTitleCaseCallback(std::function<void()> callback) { transform_titlecase_callback_ = callback; }
    void SetSortLinesAscCallback(std::function<void()> callback) { sort_lines_asc_callback_ = callback; }
    void SetSortLinesDescCallback(std::function<void()> callback) { sort_lines_desc_callback_ = callback; }
    void SetJoinLinesCallback(std::function<void()> callback) { join_lines_callback_ = callback; }

    // Find dialog visibility
    bool IsFindDialogOpen() const { return show_find_dialog_; }
    bool IsReplaceDialogOpen() const { return show_replace_dialog_; }
    bool IsFindInFilesDialogOpen() const { return show_find_in_files_dialog_; }
    bool IsReplaceInFilesDialogOpen() const { return show_replace_in_files_dialog_; }

    void OpenFindDialog() { show_find_dialog_ = true; }
    void OpenReplaceDialog() { show_replace_dialog_ = true; }
    void OpenFindInFilesDialog() { show_find_in_files_dialog_ = true; }
    void OpenReplaceInFilesDialog() { show_replace_in_files_dialog_ = true; }

    // Auto-save state
    bool IsAutoSaveEnabled() const { return auto_save_enabled_; }
    void SetAutoSaveEnabled(bool enabled) { auto_save_enabled_ = enabled; }

    // Editor settings callbacks (for Preferences -> Script Editor synchronization)
    void SetEditorThemeCallback(std::function<void(int)> callback) { editor_theme_callback_ = callback; }
    void SetEditorTabSizeCallback(std::function<void(int)> callback) { editor_tab_size_callback_ = callback; }
    void SetEditorFontScaleCallback(std::function<void(float)> callback) { editor_font_scale_callback_ = callback; }
    void SetEditorShowWhitespaceCallback(std::function<void(bool)> callback) { editor_show_whitespace_callback_ = callback; }
    void SetEditorWordWrapCallback(std::function<void(bool)> callback) { editor_word_wrap_callback_ = callback; }
    void SetEditorAutoIndentCallback(std::function<void(bool)> callback) { editor_auto_indent_callback_ = callback; }

    // Application theme callback (called when View -> Theme changes)
    void SetAppThemeChangedCallback(std::function<void(int)> callback) { app_theme_changed_callback_ = callback; }

    // Testing callbacks
    void SetRunTestCallback(std::function<void()> callback) { run_test_callback_ = callback; }
    void SetViewTestResultsCallback(std::function<void()> callback) { view_test_results_callback_ = callback; }

    // Tools menu callbacks
    void SetResumeCheckpointCallback(std::function<void()> cb) { resume_checkpoint_callback_ = cb; }
    void SetSaveCheckpointCallback(std::function<void()> cb) { save_checkpoint_callback_ = cb; }
    void SetSaveModelCallback(std::function<void()> cb) { save_model_callback_ = cb; }
    void SetRunQuickTestCallback(std::function<void()> cb) { run_quick_test_callback_ = cb; }
    void SetCompareTestResultsCallback(std::function<void()> cb) { compare_test_results_callback_ = cb; }
    void SetExportTestReportCallback(std::function<void()> cb) { export_test_report_callback_ = cb; }
    void SetClearCacheCallback(std::function<void()> cb) { clear_cache_callback_ = cb; }
    void SetRunGCCallback(std::function<void()> cb) { run_gc_callback_ = cb; }

    // Model conversion callbacks
    void SetConvertBinaryToDirCallback(std::function<void()> cb) { convert_binary_to_dir_callback_ = cb; }
    void SetConvertDirToBinaryCallback(std::function<void()> cb) { convert_dir_to_binary_callback_ = cb; }

    // Model Analysis callbacks (Phase 2)
    void SetOpenModelSummaryCallback(std::function<void()> cb) { open_model_summary_callback_ = cb; }
    void SetOpenArchitectureDiagramCallback(std::function<void()> cb) { open_architecture_diagram_callback_ = cb; }
    void SetOpenLRFinderCallback(std::function<void()> cb) { open_lr_finder_callback_ = cb; }

    // Data Science callbacks (Phase 3)
    void SetOpenDataProfilerCallback(std::function<void()> cb) { open_data_profiler_callback_ = cb; }
    void SetOpenCorrelationMatrixCallback(std::function<void()> cb) { open_correlation_matrix_callback_ = cb; }
    void SetOpenMissingValuePanelCallback(std::function<void()> cb) { open_missing_value_callback_ = cb; }
    void SetOpenOutlierDetectionCallback(std::function<void()> cb) { open_outlier_detection_callback_ = cb; }

    // Statistics callbacks (Phase 4)
    void SetOpenDescriptiveStatsCallback(std::function<void()> cb) { open_descriptive_stats_callback_ = cb; }
    void SetOpenHypothesisTestCallback(std::function<void()> cb) { open_hypothesis_test_callback_ = cb; }
    void SetOpenDistributionFitterCallback(std::function<void()> cb) { open_distribution_fitter_callback_ = cb; }
    void SetOpenRegressionCallback(std::function<void()> cb) { open_regression_callback_ = cb; }

    // Advanced Tools callbacks (Phase 5)
    void SetOpenDimReductionCallback(std::function<void()> cb) { open_dim_reduction_callback_ = cb; }
    void SetOpenGradCAMCallback(std::function<void()> cb) { open_gradcam_callback_ = cb; }
    void SetOpenFeatureImportanceCallback(std::function<void()> cb) { open_feature_importance_callback_ = cb; }
    void SetOpenNASCallback(std::function<void()> cb) { open_nas_callback_ = cb; }

    // Clustering callbacks (Phase 6A)
    void SetOpenKMeansCallback(std::function<void()> cb) { open_kmeans_callback_ = cb; }
    void SetOpenDBSCANCallback(std::function<void()> cb) { open_dbscan_callback_ = cb; }
    void SetOpenHierarchicalCallback(std::function<void()> cb) { open_hierarchical_callback_ = cb; }
    void SetOpenGMMCallback(std::function<void()> cb) { open_gmm_callback_ = cb; }
    void SetOpenClusterEvalCallback(std::function<void()> cb) { open_cluster_eval_callback_ = cb; }

    // Model Evaluation callbacks (Phase 6B)
    void SetOpenConfusionMatrixCallback(std::function<void()> cb) { open_confusion_matrix_callback_ = cb; }
    void SetOpenROCAUCCallback(std::function<void()> cb) { open_roc_auc_callback_ = cb; }
    void SetOpenPRCurveCallback(std::function<void()> cb) { open_pr_curve_callback_ = cb; }
    void SetOpenCrossValidationCallback(std::function<void()> cb) { open_cross_validation_callback_ = cb; }
    void SetOpenLearningCurvesCallback(std::function<void()> cb) { open_learning_curves_callback_ = cb; }

    // Data Transformation callbacks (Phase 6C)
    void SetOpenNormalizationCallback(std::function<void()> cb) { open_normalization_callback_ = cb; }
    void SetOpenStandardizationCallback(std::function<void()> cb) { open_standardization_callback_ = cb; }
    void SetOpenLogTransformCallback(std::function<void()> cb) { open_log_transform_callback_ = cb; }
    void SetOpenBoxCoxCallback(std::function<void()> cb) { open_boxcox_callback_ = cb; }
    void SetOpenFeatureScalingCallback(std::function<void()> cb) { open_feature_scaling_callback_ = cb; }

    // Linear Algebra callbacks (Phase 7)
    void SetOpenMatrixCalculatorCallback(std::function<void()> cb) { open_matrix_calculator_callback_ = cb; }
    void SetOpenEigenDecompCallback(std::function<void()> cb) { open_eigen_decomp_callback_ = cb; }
    void SetOpenSVDCallback(std::function<void()> cb) { open_svd_callback_ = cb; }
    void SetOpenQRCallback(std::function<void()> cb) { open_qr_callback_ = cb; }
    void SetOpenCholeskyCallback(std::function<void()> cb) { open_cholesky_callback_ = cb; }

    // Signal Processing callbacks (Phase 8)
    void SetOpenFFTCallback(std::function<void()> cb) { open_fft_callback_ = cb; }
    void SetOpenSpectrogramCallback(std::function<void()> cb) { open_spectrogram_callback_ = cb; }
    void SetOpenFilterDesignerCallback(std::function<void()> cb) { open_filter_designer_callback_ = cb; }
    void SetOpenConvolutionCallback(std::function<void()> cb) { open_convolution_callback_ = cb; }
    void SetOpenWaveletCallback(std::function<void()> cb) { open_wavelet_callback_ = cb; }

    // Optimization & Calculus callbacks (Phase 9)
    void SetOpenGradientDescentCallback(std::function<void()> cb) { open_gradient_descent_callback_ = cb; }
    void SetOpenConvexityCallback(std::function<void()> cb) { open_convexity_callback_ = cb; }
    void SetOpenLPCallback(std::function<void()> cb) { open_lp_callback_ = cb; }
    void SetOpenQPCallback(std::function<void()> cb) { open_qp_callback_ = cb; }
    void SetOpenDifferentiationCallback(std::function<void()> cb) { open_differentiation_callback_ = cb; }
    void SetOpenIntegrationCallback(std::function<void()> cb) { open_integration_callback_ = cb; }

    // Time Series Analysis callbacks (Phase 10)
    void SetOpenDecompositionCallback(std::function<void()> cb) { open_decomposition_callback_ = cb; }
    void SetOpenACFPACFCallback(std::function<void()> cb) { open_acf_pacf_callback_ = cb; }
    void SetOpenStationarityCallback(std::function<void()> cb) { open_stationarity_callback_ = cb; }
    void SetOpenSeasonalityCallback(std::function<void()> cb) { open_seasonality_callback_ = cb; }
    void SetOpenForecastingCallback(std::function<void()> cb) { open_forecasting_callback_ = cb; }

    // Text Processing callbacks (Phase 11)
    void SetOpenTokenizationCallback(std::function<void()> cb) { open_tokenization_callback_ = cb; }
    void SetOpenWordFrequencyCallback(std::function<void()> cb) { open_word_frequency_callback_ = cb; }
    void SetOpenTFIDFCallback(std::function<void()> cb) { open_tfidf_callback_ = cb; }
    void SetOpenEmbeddingsCallback(std::function<void()> cb) { open_embeddings_callback_ = cb; }
    void SetOpenSentimentCallback(std::function<void()> cb) { open_sentiment_callback_ = cb; }

    // Utilities callbacks (Phase 12)
    void SetOpenCalculatorCallback(std::function<void()> cb) { open_calculator_callback_ = cb; }
    void SetOpenUnitConverterCallback(std::function<void()> cb) { open_unit_converter_callback_ = cb; }
    void SetOpenRandomGeneratorCallback(std::function<void()> cb) { open_random_generator_callback_ = cb; }
    void SetOpenHashGeneratorCallback(std::function<void()> cb) { open_hash_generator_callback_ = cb; }
    void SetOpenJSONViewerCallback(std::function<void()> cb) { open_json_viewer_callback_ = cb; }
    void SetOpenRegexTesterCallback(std::function<void()> cb) { open_regex_tester_callback_ = cb; }

    // Export/Import callbacks
    void SetExportModelCallback(std::function<void(int)> callback) { export_model_callback_ = callback; }  // int = format index
    void SetImportModelCallback(std::function<void()> callback) { import_model_callback_ = callback; }

    // Minimap visibility pointers (for View -> Minimaps menu)
    void SetNodeEditorMinimapPtr(bool* ptr) { node_editor_minimap_ptr_ = ptr; }
    void SetScriptEditorMinimapPtr(bool* ptr) { script_editor_minimap_ptr_ = ptr; }

    // Debug logging pointers (for View -> Developer Tools)
    void SetIdleLogPtr(bool* ptr) { idle_log_ptr_ = ptr; }
    void SetVerbosePythonLogPtr(bool* ptr) { verbose_python_log_ptr_ = ptr; }

    // Initialize editor settings from script editor's current values
    void SetEditorTheme(int theme) { editor_theme_ = theme; }
    void SetEditorTabSize(int size) { editor_tab_size_ = size; }
    void SetEditorFontScale(float scale);  // Converts scale to font size
    void SetEditorShowWhitespace(bool show) { editor_show_whitespace_ = show; }
    void SetEditorWordWrap(bool wrap) { editor_word_wrap_ = wrap; }
    void SetEditorAutoIndent(bool indent) { editor_auto_indent_ = indent; }

    // Compute device selection callback
    void SetComputeDeviceChangedCallback(std::function<void(DeviceType, int)> cb) { compute_device_changed_callback_ = cb; }

    // Access to created plot windows
    const std::vector<std::shared_ptr<PlotWindow>>& GetPlotWindows() const { return plot_windows_; }

private:
    void RenderFileMenu();
    void RenderEditMenu();
    void RenderViewMenu();
    void RenderNodesMenu();
    void RenderTrainMenu();
    void RenderDatasetMenu();
    void RenderScriptMenu();
    void RenderPlotsMenu();
    void RenderDeployMenu();
    void RenderToolsMenu();

    // File search functionality
    void SearchInFiles(const std::string& search_text, const std::string& search_path,
                       const std::string& file_patterns, bool case_sensitive,
                       bool whole_word, bool use_regex);
    void RenderHelpMenu();
    void RenderUserAvatar();
    void RenderUserProfilePopup();

    // Command Palette functionality
    void InitializeToolEntries();
    void RenderCommandPalette();
    void UpdateSearchResults(const std::string& query);
    int FuzzyMatch(const std::string& pattern, const std::string& text) const;
    std::string ToLowerCase(const std::string& str) const;

    // Helper functions
    std::string OpenFolderDialog();
    std::string OpenFileDialog(const char* filter, const char* title);
    void CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type);

    bool show_new_project_dialog_;
    bool show_about_dialog_;
    bool show_account_settings_dialog_ = false;
    bool show_exit_confirmation_dialog_ = false;
    bool auto_save_enabled_ = false;
    float auto_save_interval_ = 60.0f;  // Auto save every 60 seconds
    float auto_save_timer_ = 0.0f;      // Current timer countdown

    // Account/Auth state
    bool is_logged_in_ = false;
    bool is_logging_in_ = false;
    bool session_restore_pending_ = true;
    bool show_user_profile_popup_ = false;
    bool show_login_required_popup_ = false;
    std::string login_required_action_;  // What action requires login (for popup message)
    int popup_open_frames_ = 0;  // Track frames since popup opened (for click-away delay)
    float avatar_popup_x_ = 0.0f;  // X position for profile popup
    char login_identifier_[256] = "";  // Email or phone (auto-detected)
    char login_password_[256] = "";
    bool show_password_ = false;  // Toggle password visibility
    std::string logged_in_user_;
    std::string login_error_message_;
    std::string login_success_message_;
    std::future<auth::AuthResult> login_future_;

    // Wallet connection state
    bool show_wallet_connect_dialog_ = false;
    int wallet_connect_step_ = 0;  // 0 = enter address, 1 = sign message, 2 = verifying
    char wallet_address_buffer_[128] = "";
    char wallet_signature_buffer_[512] = "";  // Base58 signatures can be long
    std::string wallet_nonce_;
    std::string wallet_sign_message_;  // Full message to sign from server
    std::string wallet_error_message_;
    std::future<auth::WalletNonceResult> wallet_nonce_future_;
    std::future<auth::WalletVerifyResult> wallet_link_future_;

    std::function<void()> reset_layout_callback_;
    std::function<void()> save_layout_callback_;
    std::function<void()> save_project_settings_callback_;
    std::function<void()> toggle_plot_test_control_callback_;
    std::function<void()> connect_to_server_callback_;
    std::function<void()> deploy_to_server_callback_;
    std::function<void()> import_dataset_callback_;
    std::function<void()> open_custom_node_editor_callback_;
    std::function<void()> open_theme_editor_callback_;
    std::function<void()> open_profiler_callback_;
    std::function<void()> open_memory_monitor_callback_;
    std::function<void()> new_script_callback_;
    std::function<void()> open_script_callback_;
    std::function<void(const std::string&)> open_script_in_editor_callback_;
    std::function<void()> save_all_callback_;
    std::function<void()> account_settings_callback_;
    std::function<void()> exit_callback_;
    std::function<bool()> has_unsaved_changes_callback_;
    std::function<void(const std::string&)> on_login_success_callback_;
    std::function<void()> on_logout_callback_;

    // Project creation state
    char project_name_buffer_[256];
    char project_path_buffer_[512];

    // Save As dialog state
    bool show_save_as_dialog_ = false;
    char save_as_name_buffer_[256] = "";
    char save_as_path_buffer_[512] = "";

    // New script dialog state
    bool show_new_script_dialog_ = false;
    char new_script_name_[256] = "";
    int new_script_type_ = 0;  // 0 = .cyx, 1 = .py

    // Plot windows management
    std::vector<std::shared_ptr<PlotWindow>> plot_windows_;

    // Edit menu callbacks
    std::function<void()> undo_callback_;
    std::function<void()> redo_callback_;
    std::function<void()> cut_callback_;
    std::function<void()> copy_callback_;
    std::function<void()> paste_callback_;
    std::function<void()> delete_callback_;
    std::function<void()> select_all_callback_;
    std::function<void()> toggle_line_comment_callback_;
    std::function<void()> toggle_block_comment_callback_;
    std::function<void(const std::string&, bool, bool, bool)> find_callback_;  // text, case_sensitive, whole_word, regex
    std::function<void()> find_next_callback_;  // Find next occurrence
    std::function<void(const std::string&, const std::string&, bool, bool, bool)> replace_callback_;  // find, replace, case_sensitive, whole_word, regex
    std::function<void(const std::string&, const std::string&, bool, bool, bool)> replace_all_callback_;

    // New Edit menu callbacks
    std::function<void(int)> go_to_line_callback_;
    std::function<void()> duplicate_line_callback_;
    std::function<void()> move_line_up_callback_;
    std::function<void()> move_line_down_callback_;
    std::function<void()> indent_callback_;
    std::function<void()> outdent_callback_;
    std::function<void()> transform_uppercase_callback_;
    std::function<void()> transform_lowercase_callback_;
    std::function<void()> transform_titlecase_callback_;
    std::function<void()> sort_lines_asc_callback_;
    std::function<void()> sort_lines_desc_callback_;
    std::function<void()> join_lines_callback_;

    // Find/Replace dialog state
    bool show_find_dialog_ = false;
    bool show_replace_dialog_ = false;
    bool show_find_in_files_dialog_ = false;
    bool show_replace_in_files_dialog_ = false;

    // Find/Replace buffers
    char find_text_buffer_[512] = "";
    char replace_text_buffer_[512] = "";
    char find_in_files_pattern_[256] = "*.py;*.cyx";  // File filter pattern
    char find_in_files_path_[512] = "";  // Search path

    // Find/Replace options
    bool find_case_sensitive_ = false;
    bool find_whole_word_ = false;
    bool find_use_regex_ = false;

    // Find in Files results
    struct SearchResult {
        std::string file_path;
        int line_number;
        std::string line_content;
        int match_start;
        int match_length;
    };
    std::vector<SearchResult> search_results_;
    bool search_in_progress_ = false;

    // Preferences dialog state
    bool show_preferences_dialog_ = false;
    int preferences_tab_ = 0;  // 0 = Python/Scripting, 1 = Keyboard Shortcuts

    // Python/Scripting preferences
    char python_interpreter_path_[512] = "";
    char python_startup_script_[512] = "";
    bool python_auto_import_numpy_ = true;
    bool python_auto_import_cyxwiz_ = true;
    int python_output_limit_ = 1000;  // Max lines in output

    // Keyboard shortcuts (action name -> shortcut string)
    struct ShortcutEntry {
        std::string category;     // Category name (e.g., "General", "Script Editor", "Node Editor")
        std::string action;
        std::string shortcut;
        std::string description;
        bool editable;
    };
    std::vector<ShortcutEntry> shortcuts_;
    int editing_shortcut_index_ = -1;
    char shortcut_edit_buffer_[64] = "";

    // Go to Line dialog state
    bool show_go_to_line_dialog_ = false;
    int go_to_line_number_ = 1;

    // Editor preferences
    int editor_theme_ = 3;  // Default to Monokai (index 3)
    int editor_font_size_ = 16;  // Maps to font_scale: 8=1.0, 12=1.3, 16=1.6, 20=2.0
    int editor_tab_size_ = 4;
    bool editor_word_wrap_ = false;
    bool editor_show_line_numbers_ = true;
    bool editor_show_whitespace_ = true;
    bool editor_auto_indent_ = true;
    bool editor_highlight_current_line_ = true;
    bool editor_show_minimap_ = false;

    // Editor settings callbacks
    std::function<void(int)> editor_theme_callback_;
    std::function<void(int)> editor_tab_size_callback_;
    std::function<void(float)> editor_font_scale_callback_;
    std::function<void(bool)> editor_show_whitespace_callback_;
    std::function<void(bool)> editor_word_wrap_callback_;
    std::function<void(bool)> editor_auto_indent_callback_;

    // Application theme callback
    std::function<void(int)> app_theme_changed_callback_;

    // Testing callbacks
    std::function<void()> run_test_callback_;
    std::function<void()> view_test_results_callback_;

    // Tools menu callbacks
    std::function<void()> resume_checkpoint_callback_;
    std::function<void()> save_checkpoint_callback_;
    std::function<void()> save_model_callback_;
    std::function<void()> run_quick_test_callback_;
    std::function<void()> compare_test_results_callback_;
    std::function<void()> export_test_report_callback_;
    std::function<void()> clear_cache_callback_;
    std::function<void()> run_gc_callback_;

    // Model conversion callbacks
    std::function<void()> convert_binary_to_dir_callback_;
    std::function<void()> convert_dir_to_binary_callback_;

    // Model Analysis callbacks (Phase 2)
    std::function<void()> open_model_summary_callback_;
    std::function<void()> open_architecture_diagram_callback_;
    std::function<void()> open_lr_finder_callback_;

    // Data Science callbacks (Phase 3)
    std::function<void()> open_data_profiler_callback_;
    std::function<void()> open_correlation_matrix_callback_;
    std::function<void()> open_missing_value_callback_;
    std::function<void()> open_outlier_detection_callback_;

    // Statistics callbacks (Phase 4)
    std::function<void()> open_descriptive_stats_callback_;
    std::function<void()> open_hypothesis_test_callback_;
    std::function<void()> open_distribution_fitter_callback_;
    std::function<void()> open_regression_callback_;

    // Advanced Tools callbacks (Phase 5)
    std::function<void()> open_dim_reduction_callback_;
    std::function<void()> open_gradcam_callback_;
    std::function<void()> open_feature_importance_callback_;
    std::function<void()> open_nas_callback_;

    // Clustering callbacks (Phase 6A)
    std::function<void()> open_kmeans_callback_;
    std::function<void()> open_dbscan_callback_;
    std::function<void()> open_hierarchical_callback_;
    std::function<void()> open_gmm_callback_;
    std::function<void()> open_cluster_eval_callback_;

    // Model Evaluation callbacks (Phase 6B)
    std::function<void()> open_confusion_matrix_callback_;
    std::function<void()> open_roc_auc_callback_;
    std::function<void()> open_pr_curve_callback_;
    std::function<void()> open_cross_validation_callback_;
    std::function<void()> open_learning_curves_callback_;

    // Data Transformation callbacks (Phase 6C)
    std::function<void()> open_normalization_callback_;
    std::function<void()> open_standardization_callback_;
    std::function<void()> open_log_transform_callback_;
    std::function<void()> open_boxcox_callback_;
    std::function<void()> open_feature_scaling_callback_;

    // Linear Algebra callbacks (Phase 7)
    std::function<void()> open_matrix_calculator_callback_;
    std::function<void()> open_eigen_decomp_callback_;
    std::function<void()> open_svd_callback_;
    std::function<void()> open_qr_callback_;
    std::function<void()> open_cholesky_callback_;

    // Signal Processing callbacks (Phase 8)
    std::function<void()> open_fft_callback_;
    std::function<void()> open_spectrogram_callback_;
    std::function<void()> open_filter_designer_callback_;
    std::function<void()> open_convolution_callback_;
    std::function<void()> open_wavelet_callback_;

    // Optimization & Calculus callbacks (Phase 9)
    std::function<void()> open_gradient_descent_callback_;
    std::function<void()> open_convexity_callback_;
    std::function<void()> open_lp_callback_;
    std::function<void()> open_qp_callback_;
    std::function<void()> open_differentiation_callback_;
    std::function<void()> open_integration_callback_;

    // Time Series Analysis callbacks (Phase 10)
    std::function<void()> open_decomposition_callback_;
    std::function<void()> open_acf_pacf_callback_;
    std::function<void()> open_stationarity_callback_;
    std::function<void()> open_seasonality_callback_;
    std::function<void()> open_forecasting_callback_;

    // Text Processing callbacks (Phase 11)
    std::function<void()> open_tokenization_callback_;
    std::function<void()> open_word_frequency_callback_;
    std::function<void()> open_tfidf_callback_;
    std::function<void()> open_embeddings_callback_;
    std::function<void()> open_sentiment_callback_;

    // Utilities callbacks (Phase 12)
    std::function<void()> open_calculator_callback_;
    std::function<void()> open_unit_converter_callback_;
    std::function<void()> open_random_generator_callback_;
    std::function<void()> open_hash_generator_callback_;
    std::function<void()> open_json_viewer_callback_;
    std::function<void()> open_regex_tester_callback_;

    // Export/Import callbacks
    std::function<void(int)> export_model_callback_;  // int = format index (0=CyxModel, 1=Safetensors, 2=ONNX, 3=GGUF)
    std::function<void()> import_model_callback_;

    // General preferences
    bool general_restore_last_session_ = true;
    bool general_check_updates_ = true;
    int general_recent_files_limit_ = 10;
    bool general_confirm_on_exit_ = true;

    // Appearance preferences
    float appearance_ui_scale_ = 1.0f;
    bool appearance_smooth_scrolling_ = true;
    int appearance_sidebar_position_ = 0;  // 0 = Left, 1 = Right

    // Minimap visibility pointers
    bool* node_editor_minimap_ptr_ = nullptr;
    bool* script_editor_minimap_ptr_ = nullptr;

    // Debug logging pointers
    bool* idle_log_ptr_ = nullptr;
    bool* verbose_python_log_ptr_ = nullptr;

    // Files preferences
    int files_default_encoding_ = 0;  // 0 = UTF-8, 1 = UTF-16, 2 = ASCII
    int files_line_ending_ = 0;  // 0 = Auto, 1 = LF, 2 = CRLF
    bool files_trim_trailing_whitespace_ = false;
    bool files_insert_final_newline_ = true;
// Performance/Compute preferences
    int compute_device_index_ = 0;
    std::vector<cyxwiz::DeviceInfo> compute_devices_;
    bool compute_devices_initialized_ = false;
    std::function<void(cyxwiz::DeviceType, int)> compute_device_changed_callback_;

    // Command Palette state
    bool show_command_palette_ = false;
    char search_buffer_[256] = "";
    std::vector<ToolEntry> all_tools_;           // All available tools
    std::vector<const ToolEntry*> filtered_tools_;  // Filtered results (pointers to all_tools_)
    int selected_index_ = 0;                     // Currently selected item in list
    bool focus_search_input_ = false;            // Flag to focus input on open
};

} // namespace cyxwiz
