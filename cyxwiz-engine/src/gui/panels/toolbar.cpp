#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include "../../auth/auth_client.h"
#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/cyxwiz.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <chrono>
#include <initializer_list>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../icons.h"

namespace cyxwiz {

namespace {

bool IsCategory(const ToolEntry& tool, const char* category) {
    return tool.category == category;
}

bool IsAnyCategory(const ToolEntry& tool, std::initializer_list<const char*> categories) {
    for (const char* category : categories) {
        if (IsCategory(tool, category)) {
            return true;
        }
    }
    return false;
}

bool IsAnyToolName(const ToolEntry& tool, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

void AnnotateCommandSurface(std::vector<ToolEntry>& tools) {
    for (auto& tool : tools) {
        if (IsCategory(tool, "Utilities")) {
            tool.surface = ToolSurface::Utility;
            tool.status_detail = "Transient utility";
            tool.keywords += " utility transient standalone";
            continue;
        }

        if (IsAnyCategory(tool, {
                "Clustering",
                "Transform",
                "Signal Processing",
                "Time Series",
                "Text",
                "Linear Algebra"
            }) ||
            IsAnyToolName(tool, {
                "Correlation Matrix",
                "Outlier Detection",
                "Regression Analysis",
                "Dimensionality Reduction",
                "GradCAM",
                "Feature Importance"
            })) {
            tool.surface = ToolSurface::GraphBackedPanel;
            tool.status_detail = "Also available as graph workflow";
            tool.keywords += " graph node pipeline durable";
            continue;
        }

        if (IsAnyCategory(tool, {
                "Model Analysis",
                "Data Science",
                "Statistics",
                "Advanced",
                "Evaluation",
                "Profile",
                "Developer"
            })) {
            tool.surface = ToolSurface::StandalonePanel;
            tool.status_detail = "Standalone panel";
            tool.keywords += " panel inspector";
        }
    }
}

} // namespace

ToolbarPanel::ToolbarPanel()
    : Panel("Toolbar", true)
    , show_new_project_dialog_(false)
    , show_about_dialog_(false)
{
    memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
    memset(project_path_buffer_, 0, sizeof(project_path_buffer_));

    // Initialize tool entries for command palette
    InitializeToolEntries();
}

void ToolbarPanel::SetEditorFontScale(float scale) {
    // Convert editor scale to the native atlas font size shown in Preferences
    // 1.0 -> 14px, 1.3 -> 16px, 1.6 -> 20px, 2.0 -> 24px
    if (scale <= 1.15f) editor_font_size_ = 14;
    else if (scale <= 1.45f) editor_font_size_ = 16;
    else if (scale <= 1.8f) editor_font_size_ = 20;
    else editor_font_size_ = 24;
}

void ToolbarPanel::Render() {
    if (!visible_) return;

    // Check for session restore (runs once on first render)
    if (session_restore_pending_) {
        session_restore_pending_ = false;
        auto& auth = auth::AuthClient::Instance();
        if (auth.LoadSavedSession()) {
            is_logged_in_ = true;
            auto user = auth.GetUserInfo();
            logged_in_user_ = user.email.empty() ? user.username : user.email;
            spdlog::info("Restored saved session for: {}", logged_in_user_);
            // Notify application of restored session with JWT token
            if (on_login_success_callback_) {
                on_login_success_callback_(auth.GetJwtToken());
            }
        }
    }

    // Check if async login completed
    if (login_future_.valid()) {
        auto status = login_future_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            auto result = login_future_.get();
            is_logging_in_ = false;

            if (result.success) {
                is_logged_in_ = true;
                login_error_message_.clear();
                auto user = result.user_info;
                logged_in_user_ = user.email.empty() ? user.username : user.email;
                login_success_message_ = "Login successful!";
                spdlog::info("Login successful: {}", logged_in_user_);
                memset(login_password_, 0, sizeof(login_password_));
                // Close login dialog on success
                // Notify application of successful login with JWT token
                show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    auto& auth = auth::AuthClient::Instance();
                    on_login_success_callback_(auth.GetJwtToken());
                }
            } else {
                login_error_message_ = result.error;
                login_success_message_.clear();
                spdlog::error("Login failed: {}", result.error);
            }
        }
    }

    // Use standard ImGui main menu bar (positioned right below the OS title bar)
    if (ImGui::BeginMainMenuBar()) {
        RenderFileMenu();
        RenderEditMenu();
        RenderViewMenu();
        RenderProfileMenu();
        RenderNodesMenu();
        RenderTrainMenu();
        RenderSimulationMenu();
        RenderToolsMenu();
        RenderDatasetMenu();
        RenderScriptMenu();
        RenderDeployMenu();
        RenderAppsMenu();
        RenderHelpMenu();

        // Show current project name in menu bar if active
        auto& pm = ProjectManager::Instance();
        if (pm.HasActiveProject()) {
            ImGui::Separator();
            ImGui::TextDisabled("| Project: %s", pm.GetProjectName().c_str());
        }

        // User avatar on the right side of menu bar
        RenderUserAvatar();

        ImGui::EndMainMenuBar();
    }

    // Render user profile popup outside menu bar
    if (show_user_profile_popup_ && is_logged_in_) {
        RenderUserProfilePopup();
    }

    // Render all plot windows
    for (auto& plot_window : plot_windows_) {
        if (plot_window) {
            plot_window->Render();
        }
    }

    RenderProjectDialogs();

    HandleAutoSaveTimer();

    RenderEditorDialogs();

    // Render command palette overlay
    RenderCommandPalette();
}

// ============================================================================
// Command Palette Implementation
// ============================================================================

void ToolbarPanel::InitializeToolEntries() {
    // Clear any existing entries
    all_tools_.clear();

    // ==================== File Commands ====================
    all_tools_.push_back({"New Project", "File", "new project create", ICON_FA_FOLDER_PLUS, "Ctrl+Shift+N", [this]() { show_new_project_dialog_ = true; }});
    all_tools_.push_back({"Save All", "File", "save all files", ICON_FA_COPY, "Ctrl+Shift+S", [this]() { if (save_all_callback_) save_all_callback_(); }});

    // ==================== Edit Commands ====================
    all_tools_.push_back({"Undo", "Edit", "undo revert back", ICON_FA_ROTATE_LEFT, "Ctrl+Z", [this]() { if (undo_callback_) undo_callback_(); }});
    all_tools_.push_back({"Redo", "Edit", "redo forward", ICON_FA_ROTATE_RIGHT, "Ctrl+Y", [this]() { if (redo_callback_) redo_callback_(); }});
    all_tools_.push_back({"Cut", "Edit", "cut selection", ICON_FA_SCISSORS, "Ctrl+X", [this]() { if (cut_callback_) cut_callback_(); }});
    all_tools_.push_back({"Copy", "Edit", "copy selection", ICON_FA_COPY, "Ctrl+C", [this]() { if (copy_callback_) copy_callback_(); }});
    all_tools_.push_back({"Paste", "Edit", "paste clipboard", ICON_FA_PASTE, "Ctrl+V", [this]() { if (paste_callback_) paste_callback_(); }});
    all_tools_.push_back({"Delete", "Edit", "delete selection", ICON_FA_TRASH, "Del", [this]() { if (delete_callback_) delete_callback_(); }});
    all_tools_.push_back({"Select All", "Edit", "select all", ICON_FA_OBJECT_GROUP, "Ctrl+A", [this]() { if (select_all_callback_) select_all_callback_(); }});
    all_tools_.push_back({"Find", "Edit", "find search text", ICON_FA_MAGNIFYING_GLASS, "Ctrl+F", [this]() { OpenFindDialog(); }});
    all_tools_.push_back({"Replace", "Edit", "replace substitute text", ICON_FA_RIGHT_LEFT, "Ctrl+H", [this]() { OpenReplaceDialog(); }});
    all_tools_.push_back({"Find in Files", "Edit", "find search files grep", ICON_FA_FOLDER_TREE, "Ctrl+Shift+F", [this]() { OpenFindInFilesDialog(); }});
    all_tools_.push_back({"Go to Line", "Edit", "go to line jump", ICON_FA_HASHTAG, "Ctrl+G", [this]() { show_go_to_line_dialog_ = true; }});

    // ==================== Script Commands ====================
    all_tools_.push_back({"Python Console", "Script", "python console repl terminal command", ICON_FA_TERMINAL, "F12", [this]() { if (open_python_console_callback_) open_python_console_callback_(); }});
    all_tools_.push_back({"New Script", "Script", "new script python file", ICON_FA_FILE_CODE, "Ctrl+N", [this]() { if (new_script_callback_) new_script_callback_(); }});
    all_tools_.push_back({"Open Script", "Script", "open script python file", ICON_FA_FILE_IMPORT, "Ctrl+O", [this]() { if (open_script_callback_) open_script_callback_(); }});

    // ==================== Training Commands ====================
    all_tools_.push_back({"Connect to Server", "Training", "connect server network cloud", ICON_FA_PLUG, "", [this]() { if (connect_to_server_callback_) connect_to_server_callback_(); }});
    all_tools_.push_back({"Export Model", "Training", "export model save onnx safetensors", ICON_FA_FILE_EXPORT, "", [this]() { if (export_model_callback_) export_model_callback_(0); }});
    all_tools_.push_back({"Import Model", "Training", "import model load onnx pytorch", ICON_FA_FILE_IMPORT, "", [this]() { if (import_model_callback_) import_model_callback_(); }});

    // ==================== View Commands ====================
    all_tools_.push_back({"Reset Layout", "View", "reset layout default dock", ICON_FA_WINDOW_RESTORE, "", [this]() { if (reset_layout_callback_) reset_layout_callback_(); }});
    all_tools_.push_back({"Save Layout", "View", "save layout dock", ICON_FA_FLOPPY_DISK, "", [this]() { if (save_layout_callback_) save_layout_callback_(); }});
    all_tools_.push_back({"Preferences", "View", "preferences settings options", ICON_FA_GEAR, "", [this]() { show_preferences_dialog_ = true; }});
    all_tools_.push_back({"Theme Editor", "View", "theme color customize", ICON_FA_PALETTE, "", [this]() { if (open_theme_editor_callback_) open_theme_editor_callback_(); }});
    // Profiling tools moved to "Profile" category

    // Model Analysis (Phase 2)
    all_tools_.push_back({"Model Summary", "Model Analysis", "model summary architecture layers parameters", ICON_FA_CUBES, "", [this]() { if (open_model_summary_callback_) open_model_summary_callback_(); }});
    all_tools_.push_back({"Architecture Diagram", "Model Analysis", "architecture diagram visual graph", ICON_FA_DIAGRAM_PROJECT, "", [this]() { if (open_architecture_diagram_callback_) open_architecture_diagram_callback_(); }});
    all_tools_.push_back({"LR Finder", "Model Analysis", "learning rate finder lr range test", ICON_FA_CHART_LINE, "", [this]() { if (open_lr_finder_callback_) open_lr_finder_callback_(); }});

    // Data Science (Phase 3)
    all_tools_.push_back({"Data Profiler", "Data Science", "data profile statistics overview", ICON_FA_MAGNIFYING_GLASS_CHART, "", [this]() { if (open_data_profiler_callback_) open_data_profiler_callback_(); }});
    all_tools_.push_back({"Correlation Matrix", "Data Science", "correlation heatmap features", ICON_FA_TABLE_CELLS, "", [this]() { if (open_correlation_matrix_callback_) open_correlation_matrix_callback_(); }});
    all_tools_.push_back({"Missing Values", "Data Science", "missing null nan values imputation", ICON_FA_QUESTION, "", [this]() { if (open_missing_value_callback_) open_missing_value_callback_(); }});
    all_tools_.push_back({"Outlier Detection", "Data Science", "outlier anomaly detection zscore iqr", ICON_FA_TRIANGLE_EXCLAMATION, "", [this]() { if (open_outlier_detection_callback_) open_outlier_detection_callback_(); }});

    // Statistics (Phase 4)
    all_tools_.push_back({"Descriptive Statistics", "Statistics", "mean median std variance descriptive stats", ICON_FA_CALCULATOR, "", [this]() { if (open_descriptive_stats_callback_) open_descriptive_stats_callback_(); }});
    all_tools_.push_back({"Hypothesis Test", "Statistics", "hypothesis test ttest anova chi square", ICON_FA_SCALE_BALANCED, "", [this]() { if (open_hypothesis_test_callback_) open_hypothesis_test_callback_(); }});
    all_tools_.push_back({"Distribution Fitter", "Statistics", "distribution fit normal gaussian poisson", ICON_FA_CHART_AREA, "", [this]() { if (open_distribution_fitter_callback_) open_distribution_fitter_callback_(); }});
    all_tools_.push_back({"Regression Analysis", "Statistics", "regression linear polynomial fit", ICON_FA_ARROW_TREND_UP, "", [this]() { if (open_regression_callback_) open_regression_callback_(); }});

    // Advanced Tools (Phase 5)
    all_tools_.push_back({"Dimensionality Reduction", "Advanced", "pca tsne umap dimensionality reduction", ICON_FA_COMPRESS, "", [this]() { if (open_dim_reduction_callback_) open_dim_reduction_callback_(); }});
    all_tools_.push_back({"GradCAM", "Advanced", "gradcam visualization explainability heatmap", ICON_FA_EYE, "", [this]() { if (open_gradcam_callback_) open_gradcam_callback_(); }});
    all_tools_.push_back({"Feature Importance", "Advanced", "feature importance shap permutation", ICON_FA_RANKING_STAR, "", [this]() { if (open_feature_importance_callback_) open_feature_importance_callback_(); }});
    all_tools_.push_back({"Neural Architecture Search", "Advanced", "nas automl neural architecture search", ICON_FA_MICROCHIP, "", [this]() { if (open_nas_callback_) open_nas_callback_(); }});
    all_tools_.push_back({"Hyperparameter Search", "Advanced", "hyperparameter tuning grid random bayesian optimization", ICON_FA_MAGNIFYING_GLASS_CHART, "", [this]() { if (open_hyperparam_search_callback_) open_hyperparam_search_callback_(); }});
    all_tools_.push_back({"Model Serving", "Advanced", "deploy serving api rest inference endpoint", ICON_FA_SERVER, "", [this]() { if (open_serving_callback_) open_serving_callback_(); }});
    all_tools_.push_back({"DNN Inference", "Panels", "dnn inference yolo detection face pose classification object detection neural network", ICON_FA_BRAIN, "", [this]() { if (open_dnn_inference_callback_) open_dnn_inference_callback_(); }});

    // Clustering (Phase 6A)
    all_tools_.push_back({"K-Means Clustering", "Clustering", "kmeans clustering centroid", ICON_FA_OBJECT_GROUP, "", [this]() { if (open_kmeans_callback_) open_kmeans_callback_(); }});
    all_tools_.push_back({"DBSCAN", "Clustering", "dbscan density clustering", ICON_FA_CIRCLE_NODES, "", [this]() { if (open_dbscan_callback_) open_dbscan_callback_(); }});
    all_tools_.push_back({"Hierarchical Clustering", "Clustering", "hierarchical dendrogram agglomerative", ICON_FA_SITEMAP, "", [this]() { if (open_hierarchical_callback_) open_hierarchical_callback_(); }});
    all_tools_.push_back({"GMM", "Clustering", "gmm gaussian mixture model", ICON_FA_CHART_PIE, "", [this]() { if (open_gmm_callback_) open_gmm_callback_(); }});
    all_tools_.push_back({"Cluster Evaluation", "Clustering", "silhouette elbow clustering evaluation", ICON_FA_CHART_SIMPLE, "", [this]() { if (open_cluster_eval_callback_) open_cluster_eval_callback_(); }});

    // Model Evaluation (Phase 6B)
    all_tools_.push_back({"Confusion Matrix", "Evaluation", "confusion matrix classification accuracy", ICON_FA_TABLE, "", [this]() { if (open_confusion_matrix_callback_) open_confusion_matrix_callback_(); }});
    all_tools_.push_back({"ROC AUC", "Evaluation", "roc auc curve receiver operating", ICON_FA_CHART_LINE, "", [this]() { if (open_roc_auc_callback_) open_roc_auc_callback_(); }});
    all_tools_.push_back({"PR Curve", "Evaluation", "precision recall curve pr", ICON_FA_CHART_AREA, "", [this]() { if (open_pr_curve_callback_) open_pr_curve_callback_(); }});
    all_tools_.push_back({"Cross Validation", "Evaluation", "cross validation kfold cv", ICON_FA_REPEAT, "", [this]() { if (open_cross_validation_callback_) open_cross_validation_callback_(); }});
    all_tools_.push_back({"Learning Curves", "Evaluation", "learning curve bias variance", ICON_FA_GRADUATION_CAP, "", [this]() { if (open_learning_curves_callback_) open_learning_curves_callback_(); }});

    // Data Transformation (Phase 6C)
    all_tools_.push_back({"Normalization", "Transform", "normalize min max scaling", ICON_FA_CROSSHAIRS, "", [this]() { if (open_normalization_callback_) open_normalization_callback_(); }});
    all_tools_.push_back({"Standardization", "Transform", "standardize zscore standard", ICON_FA_ARROWS_LEFT_RIGHT, "", [this]() { if (open_standardization_callback_) open_standardization_callback_(); }});
    all_tools_.push_back({"Log Transform", "Transform", "log logarithm transform", ICON_FA_SUPERSCRIPT, "", [this]() { if (open_log_transform_callback_) open_log_transform_callback_(); }});
    all_tools_.push_back({"Box-Cox Transform", "Transform", "boxcox transform power", ICON_FA_WAND_MAGIC_SPARKLES, "", [this]() { if (open_boxcox_callback_) open_boxcox_callback_(); }});
    all_tools_.push_back({"Feature Scaling", "Transform", "feature scaling robust scaler", ICON_FA_MAXIMIZE, "", [this]() { if (open_feature_scaling_callback_) open_feature_scaling_callback_(); }});

    // Linear Algebra (Phase 7)
    all_tools_.push_back({"Matrix Calculator", "Linear Algebra", "matrix calculator multiply inverse transpose", ICON_FA_TABLE, "", [this]() { if (open_matrix_calculator_callback_) open_matrix_calculator_callback_(); }});
    all_tools_.push_back({"Eigendecomposition", "Linear Algebra", "eigen eigenvalue eigenvector decomposition", ICON_FA_SQUARE, "", [this]() { if (open_eigen_decomp_callback_) open_eigen_decomp_callback_(); }});
    all_tools_.push_back({"SVD", "Linear Algebra", "svd singular value decomposition", ICON_FA_LAYER_GROUP, "", [this]() { if (open_svd_callback_) open_svd_callback_(); }});
    all_tools_.push_back({"QR Decomposition", "Linear Algebra", "qr decomposition orthogonal", ICON_FA_SQUARE_ROOT_VARIABLE, "", [this]() { if (open_qr_callback_) open_qr_callback_(); }});
    all_tools_.push_back({"Cholesky Decomposition", "Linear Algebra", "cholesky decomposition positive definite", ICON_FA_BORDER_ALL, "", [this]() { if (open_cholesky_callback_) open_cholesky_callback_(); }});

    // Signal Processing (Phase 8)
    all_tools_.push_back({"FFT", "Signal Processing", "fft fourier transform frequency", ICON_FA_WAVE_SQUARE, "", [this]() { if (open_fft_callback_) open_fft_callback_(); }});
    all_tools_.push_back({"Spectrogram", "Signal Processing", "spectrogram time frequency stft", ICON_FA_CHART_COLUMN, "", [this]() { if (open_spectrogram_callback_) open_spectrogram_callback_(); }});
    all_tools_.push_back({"Filter Designer", "Signal Processing", "filter design lowpass highpass bandpass", ICON_FA_FILTER, "", [this]() { if (open_filter_designer_callback_) open_filter_designer_callback_(); }});
    all_tools_.push_back({"Convolution", "Signal Processing", "convolution convolve signal", ICON_FA_ARROWS_LEFT_RIGHT, "", [this]() { if (open_convolution_callback_) open_convolution_callback_(); }});
    all_tools_.push_back({"Wavelet Transform", "Signal Processing", "wavelet transform dwt cwt", ICON_FA_WATER, "", [this]() { if (open_wavelet_callback_) open_wavelet_callback_(); }});

    // Optimization & Calculus (Phase 9)
    all_tools_.push_back({"Gradient Descent", "Optimization", "gradient descent optimizer sgd adam", ICON_FA_ARROW_DOWN_LONG, "", [this]() { if (open_gradient_descent_callback_) open_gradient_descent_callback_(); }});
    all_tools_.push_back({"Convexity Analysis", "Optimization", "convex convexity optimization", ICON_FA_ROUTE, "", [this]() { if (open_convexity_callback_) open_convexity_callback_(); }});
    all_tools_.push_back({"Linear Programming", "Optimization", "linear programming lp simplex", ICON_FA_MAXIMIZE, "", [this]() { if (open_lp_callback_) open_lp_callback_(); }});
    all_tools_.push_back({"Quadratic Programming", "Optimization", "quadratic programming qp", ICON_FA_SQUARE, "", [this]() { if (open_qp_callback_) open_qp_callback_(); }});
    all_tools_.push_back({"Differentiation", "Calculus", "derivative differentiation gradient jacobian", ICON_FA_INFINITY, "", [this]() { if (open_differentiation_callback_) open_differentiation_callback_(); }});
    all_tools_.push_back({"Integration", "Calculus", "integral integration numerical", ICON_FA_INTEGRAL, "", [this]() { if (open_integration_callback_) open_integration_callback_(); }});

    // Time Series (Phase 10)
    all_tools_.push_back({"Decomposition", "Time Series", "decomposition trend seasonality residual", ICON_FA_CHART_LINE, "", [this]() { if (open_decomposition_callback_) open_decomposition_callback_(); }});
    all_tools_.push_back({"ACF/PACF", "Time Series", "acf pacf autocorrelation", ICON_FA_CHART_BAR, "", [this]() { if (open_acf_pacf_callback_) open_acf_pacf_callback_(); }});
    all_tools_.push_back({"Stationarity Test", "Time Series", "stationarity adf kpss test", ICON_FA_FLASK, "", [this]() { if (open_stationarity_callback_) open_stationarity_callback_(); }});
    all_tools_.push_back({"Seasonality Detection", "Time Series", "seasonality periodic pattern", ICON_FA_CALENDAR, "", [this]() { if (open_seasonality_callback_) open_seasonality_callback_(); }});
    all_tools_.push_back({"Forecasting", "Time Series", "forecast prediction arima lstm", ICON_FA_FORWARD, "", [this]() { if (open_forecasting_callback_) open_forecasting_callback_(); }});

    // Text Processing (Phase 11)
    all_tools_.push_back({"Tokenization", "Text", "tokenize tokenization nlp words", ICON_FA_SCISSORS, "", [this]() { if (open_tokenization_callback_) open_tokenization_callback_(); }});
    all_tools_.push_back({"Word Frequency", "Text", "word frequency count terms", ICON_FA_HASHTAG, "", [this]() { if (open_word_frequency_callback_) open_word_frequency_callback_(); }});
    all_tools_.push_back({"TF-IDF", "Text", "tfidf term frequency inverse document", ICON_FA_FILE_LINES, "", [this]() { if (open_tfidf_callback_) open_tfidf_callback_(); }});
    all_tools_.push_back({"Embeddings", "Text", "embeddings word2vec bert transformer", ICON_FA_CUBE, "", [this]() { if (open_embeddings_callback_) open_embeddings_callback_(); }});
    all_tools_.push_back({"Sentiment Analysis", "Text", "sentiment analysis positive negative", ICON_FA_FACE_SMILE, "", [this]() { if (open_sentiment_callback_) open_sentiment_callback_(); }});

    // Utilities (Phase 12)
    all_tools_.push_back({"Calculator", "Utilities", "calculator math compute", ICON_FA_CALCULATOR, "", [this]() { if (open_calculator_callback_) open_calculator_callback_(); }});
    all_tools_.push_back({"Unit Converter", "Utilities", "unit convert conversion", ICON_FA_RIGHT_LEFT, "", [this]() { if (open_unit_converter_callback_) open_unit_converter_callback_(); }});
    all_tools_.push_back({"Random Generator", "Utilities", "random number generator", ICON_FA_DICE, "", [this]() { if (open_random_generator_callback_) open_random_generator_callback_(); }});
    all_tools_.push_back({"Hash Generator", "Utilities", "hash md5 sha256 checksum", ICON_FA_FINGERPRINT, "", [this]() { if (open_hash_generator_callback_) open_hash_generator_callback_(); }});
    all_tools_.push_back({"JSON Viewer", "Utilities", "json viewer formatter", ICON_FA_CODE, "", [this]() { if (open_json_viewer_callback_) open_json_viewer_callback_(); }});
    all_tools_.push_back({"Regex Tester", "Utilities", "regex regular expression test", ICON_FA_ASTERISK, "", [this]() { if (open_regex_tester_callback_) open_regex_tester_callback_(); }});

    // ==================== Profile Commands ====================
    all_tools_.push_back({"Performance Profiler", "Profile", "profiler performance timing cpu gpu layer", ICON_FA_GAUGE_HIGH, "", [this]() { if (open_profiler_callback_) open_profiler_callback_(); }});
    all_tools_.push_back({"Memory Visualization", "Profile", "memory visualization cpu gpu tensor heap", ICON_FA_CHART_PIE, "", [this]() { if (open_memory_panel_callback_) open_memory_panel_callback_(); }});
    all_tools_.push_back({"System Monitor", "Profile", "system monitor cpu ram gpu usage", ICON_FA_MICROCHIP, "", [this]() { if (open_memory_monitor_callback_) open_memory_monitor_callback_(); }});

    // Development tools
    all_tools_.push_back({"Custom Node Editor", "Developer", "custom node create define", ICON_FA_GEARS, "", [this]() { if (open_custom_node_editor_callback_) open_custom_node_editor_callback_(); }});

    auto require_tool_callback = [this](const char* tool_name, std::function<bool()> is_enabled) {
        for (auto& tool : all_tools_) {
            if (tool.name == tool_name) {
                tool.is_enabled = std::move(is_enabled);
                return;
            }
        }
    };

    require_tool_callback("Save All", [this]() { return static_cast<bool>(save_all_callback_); });
    require_tool_callback("Undo", [this]() { return static_cast<bool>(undo_callback_); });
    require_tool_callback("Redo", [this]() { return static_cast<bool>(redo_callback_); });
    require_tool_callback("Cut", [this]() { return static_cast<bool>(cut_callback_); });
    require_tool_callback("Copy", [this]() { return static_cast<bool>(copy_callback_); });
    require_tool_callback("Paste", [this]() { return static_cast<bool>(paste_callback_); });
    require_tool_callback("Delete", [this]() { return static_cast<bool>(delete_callback_); });
    require_tool_callback("Select All", [this]() { return static_cast<bool>(select_all_callback_); });
    require_tool_callback("Python Console", [this]() { return static_cast<bool>(open_python_console_callback_); });
    require_tool_callback("New Script", [this]() { return static_cast<bool>(new_script_callback_); });
    require_tool_callback("Open Script", [this]() { return static_cast<bool>(open_script_callback_); });
    require_tool_callback("Connect to Server", [this]() { return static_cast<bool>(connect_to_server_callback_); });
    require_tool_callback("Export Model", [this]() { return static_cast<bool>(export_model_callback_); });
    require_tool_callback("Import Model", [this]() { return static_cast<bool>(import_model_callback_); });
    require_tool_callback("Reset Layout", [this]() { return static_cast<bool>(reset_layout_callback_); });
    require_tool_callback("Save Layout", [this]() { return static_cast<bool>(save_layout_callback_); });
    require_tool_callback("Theme Editor", [this]() { return static_cast<bool>(open_theme_editor_callback_); });
    require_tool_callback("Model Summary", [this]() { return static_cast<bool>(open_model_summary_callback_); });
    require_tool_callback("Architecture Diagram", [this]() { return static_cast<bool>(open_architecture_diagram_callback_); });
    require_tool_callback("LR Finder", [this]() { return static_cast<bool>(open_lr_finder_callback_); });
    require_tool_callback("Data Profiler", [this]() { return static_cast<bool>(open_data_profiler_callback_); });
    require_tool_callback("Correlation Matrix", [this]() { return static_cast<bool>(open_correlation_matrix_callback_); });
    require_tool_callback("Missing Values", [this]() { return static_cast<bool>(open_missing_value_callback_); });
    require_tool_callback("Outlier Detection", [this]() { return static_cast<bool>(open_outlier_detection_callback_); });
    require_tool_callback("Descriptive Statistics", [this]() { return static_cast<bool>(open_descriptive_stats_callback_); });
    require_tool_callback("Hypothesis Test", [this]() { return static_cast<bool>(open_hypothesis_test_callback_); });
    require_tool_callback("Distribution Fitter", [this]() { return static_cast<bool>(open_distribution_fitter_callback_); });
    require_tool_callback("Regression Analysis", [this]() { return static_cast<bool>(open_regression_callback_); });
    require_tool_callback("Dimensionality Reduction", [this]() { return static_cast<bool>(open_dim_reduction_callback_); });
    require_tool_callback("GradCAM", [this]() { return static_cast<bool>(open_gradcam_callback_); });
    require_tool_callback("Feature Importance", [this]() { return static_cast<bool>(open_feature_importance_callback_); });
    require_tool_callback("Neural Architecture Search", [this]() { return static_cast<bool>(open_nas_callback_); });
    require_tool_callback("Hyperparameter Search", [this]() { return static_cast<bool>(open_hyperparam_search_callback_); });
    require_tool_callback("Model Serving", [this]() { return static_cast<bool>(open_serving_callback_); });
    require_tool_callback("DNN Inference", [this]() { return static_cast<bool>(open_dnn_inference_callback_); });
    require_tool_callback("K-Means Clustering", [this]() { return static_cast<bool>(open_kmeans_callback_); });
    require_tool_callback("DBSCAN", [this]() { return static_cast<bool>(open_dbscan_callback_); });
    require_tool_callback("Hierarchical Clustering", [this]() { return static_cast<bool>(open_hierarchical_callback_); });
    require_tool_callback("GMM", [this]() { return static_cast<bool>(open_gmm_callback_); });
    require_tool_callback("Cluster Evaluation", [this]() { return static_cast<bool>(open_cluster_eval_callback_); });
    require_tool_callback("Confusion Matrix", [this]() { return static_cast<bool>(open_confusion_matrix_callback_); });
    require_tool_callback("ROC AUC", [this]() { return static_cast<bool>(open_roc_auc_callback_); });
    require_tool_callback("PR Curve", [this]() { return static_cast<bool>(open_pr_curve_callback_); });
    require_tool_callback("Cross Validation", [this]() { return static_cast<bool>(open_cross_validation_callback_); });
    require_tool_callback("Learning Curves", [this]() { return static_cast<bool>(open_learning_curves_callback_); });
    require_tool_callback("Normalization", [this]() { return static_cast<bool>(open_normalization_callback_); });
    require_tool_callback("Standardization", [this]() { return static_cast<bool>(open_standardization_callback_); });
    require_tool_callback("Log Transform", [this]() { return static_cast<bool>(open_log_transform_callback_); });
    require_tool_callback("Box-Cox Transform", [this]() { return static_cast<bool>(open_boxcox_callback_); });
    require_tool_callback("Feature Scaling", [this]() { return static_cast<bool>(open_feature_scaling_callback_); });
    require_tool_callback("Matrix Calculator", [this]() { return static_cast<bool>(open_matrix_calculator_callback_); });
    require_tool_callback("Eigendecomposition", [this]() { return static_cast<bool>(open_eigen_decomp_callback_); });
    require_tool_callback("SVD", [this]() { return static_cast<bool>(open_svd_callback_); });
    require_tool_callback("QR Decomposition", [this]() { return static_cast<bool>(open_qr_callback_); });
    require_tool_callback("Cholesky Decomposition", [this]() { return static_cast<bool>(open_cholesky_callback_); });
    require_tool_callback("FFT", [this]() { return static_cast<bool>(open_fft_callback_); });
    require_tool_callback("Spectrogram", [this]() { return static_cast<bool>(open_spectrogram_callback_); });
    require_tool_callback("Filter Designer", [this]() { return static_cast<bool>(open_filter_designer_callback_); });
    require_tool_callback("Convolution", [this]() { return static_cast<bool>(open_convolution_callback_); });
    require_tool_callback("Wavelet Transform", [this]() { return static_cast<bool>(open_wavelet_callback_); });
    require_tool_callback("Gradient Descent", [this]() { return static_cast<bool>(open_gradient_descent_callback_); });
    require_tool_callback("Convexity Analysis", [this]() { return static_cast<bool>(open_convexity_callback_); });
    require_tool_callback("Linear Programming", [this]() { return static_cast<bool>(open_lp_callback_); });
    require_tool_callback("Quadratic Programming", [this]() { return static_cast<bool>(open_qp_callback_); });
    require_tool_callback("Differentiation", [this]() { return static_cast<bool>(open_differentiation_callback_); });
    require_tool_callback("Integration", [this]() { return static_cast<bool>(open_integration_callback_); });
    require_tool_callback("Decomposition", [this]() { return static_cast<bool>(open_decomposition_callback_); });
    require_tool_callback("ACF/PACF", [this]() { return static_cast<bool>(open_acf_pacf_callback_); });
    require_tool_callback("Stationarity Test", [this]() { return static_cast<bool>(open_stationarity_callback_); });
    require_tool_callback("Seasonality Detection", [this]() { return static_cast<bool>(open_seasonality_callback_); });
    require_tool_callback("Forecasting", [this]() { return static_cast<bool>(open_forecasting_callback_); });
    require_tool_callback("Tokenization", [this]() { return static_cast<bool>(open_tokenization_callback_); });
    require_tool_callback("Word Frequency", [this]() { return static_cast<bool>(open_word_frequency_callback_); });
    require_tool_callback("TF-IDF", [this]() { return static_cast<bool>(open_tfidf_callback_); });
    require_tool_callback("Embeddings", [this]() { return static_cast<bool>(open_embeddings_callback_); });
    require_tool_callback("Sentiment Analysis", [this]() { return static_cast<bool>(open_sentiment_callback_); });
    require_tool_callback("Calculator", [this]() { return static_cast<bool>(open_calculator_callback_); });
    require_tool_callback("Unit Converter", [this]() { return static_cast<bool>(open_unit_converter_callback_); });
    require_tool_callback("Random Generator", [this]() { return static_cast<bool>(open_random_generator_callback_); });
    require_tool_callback("Hash Generator", [this]() { return static_cast<bool>(open_hash_generator_callback_); });
    require_tool_callback("JSON Viewer", [this]() { return static_cast<bool>(open_json_viewer_callback_); });
    require_tool_callback("Regex Tester", [this]() { return static_cast<bool>(open_regex_tester_callback_); });
    require_tool_callback("Performance Profiler", [this]() { return static_cast<bool>(open_profiler_callback_); });
    require_tool_callback("Memory Visualization", [this]() { return static_cast<bool>(open_memory_panel_callback_); });
    require_tool_callback("System Monitor", [this]() { return static_cast<bool>(open_memory_monitor_callback_); });
    require_tool_callback("Custom Node Editor", [this]() { return static_cast<bool>(open_custom_node_editor_callback_); });

    AnnotateCommandSurface(all_tools_);

    spdlog::debug("Initialized {} tool entries for command palette", all_tools_.size());
}

} // namespace cyxwiz
