#include "toolbar.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

void ToolbarPanel::RenderToolsMenu() {
    if (ImGui::BeginMenu("Tools")) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // ==================== Advanced ====================
        if (ImGui::BeginMenu(ICON_FA_WAND_MAGIC_SPARKLES " Advanced")) {
            if (ImGui::MenuItem(ICON_FA_MAGNIFYING_GLASS_CHART " Hyperparameter Search")) {
                if (open_hyperparam_search_callback_) open_hyperparam_search_callback_();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Grid, Random, or Bayesian hyperparameter optimization");
            }

            if (ImGui::MenuItem(ICON_FA_SERVER " Model Serving")) {
                if (open_serving_callback_) open_serving_callback_();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Deploy and serve models as REST API endpoints");
            }

            ImGui::EndMenu();
        }

        ImGui::Separator();

        // ==================== Model Export ====================
        if (ImGui::BeginMenu(ICON_FA_FILE_EXPORT " Model Export")) {
            if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Trained Model...", "Ctrl+Shift+S")) {
                if (save_model_callback_) save_model_callback_();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Save the trained model weights to a binary .cyxmodel file");
            }

            ImGui::Separator();
            ImGui::TextDisabled("Format Conversion:");

            if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Binary to Directory...")) {
                if (convert_binary_to_dir_callback_) convert_binary_to_dir_callback_();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Convert binary .cyxmodel file to directory format\n(for use with Deploy > Export Model)");
            }

            if (ImGui::MenuItem(ICON_FA_FILE " Directory to Binary...")) {
                if (convert_dir_to_binary_callback_) convert_dir_to_binary_callback_();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Convert .cyxmodel directory to binary file format\n(for smaller, single-file storage)");
            }

            ImGui::EndMenu();
        }

        // ==================== Checkpoints ====================
        if (ImGui::BeginMenu(ICON_FA_CLOCK_ROTATE_LEFT " Checkpoints")) {
            if (ImGui::MenuItem("Resume from Checkpoint...")) {
                if (resume_checkpoint_callback_) resume_checkpoint_callback_();
            }
            if (ImGui::MenuItem("Save Checkpoint...")) {
                if (save_checkpoint_callback_) save_checkpoint_callback_();
            }
            ImGui::EndMenu();
        }

        // ==================== Testing ====================
        if (ImGui::BeginMenu(ICON_FA_FLASK " Testing")) {
            if (ImGui::MenuItem(ICON_FA_GAUGE " Run Test", "F7")) {
                if (run_test_callback_) run_test_callback_();
            }
            if (ImGui::MenuItem("Run Quick Test", "Shift+F7")) {
                if (run_quick_test_callback_) run_quick_test_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_CHART_BAR " View Test Results")) {
                if (view_test_results_callback_) view_test_results_callback_();
            }
            if (ImGui::MenuItem("Compare Test Results...")) {
                if (compare_test_results_callback_) compare_test_results_callback_();
            }
            if (ImGui::MenuItem("Export Test Report...")) {
                if (export_test_report_callback_) export_test_report_callback_();
            }
            ImGui::EndMenu();
        }

        // ==================== Memory & Performance ====================
        if (ImGui::BeginMenu(ICON_FA_MICROCHIP " Memory & Performance")) {
            if (ImGui::MenuItem(ICON_FA_MEMORY " Memory Monitor")) {
                if (open_memory_monitor_callback_) open_memory_monitor_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Cache")) {
                if (clear_cache_callback_) clear_cache_callback_();
                spdlog::info("Cache cleared");
            }
            if (ImGui::MenuItem("Garbage Collection")) {
                if (run_gc_callback_) run_gc_callback_();
                spdlog::info("Garbage collection triggered");
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // ==================== Model Analysis ====================
        if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Model Analysis")) {
            if (ImGui::MenuItem(ICON_FA_LIST " Model Summary", "Ctrl+I")) {
                if (open_model_summary_callback_) open_model_summary_callback_();
            }
            if (ImGui::MenuItem(ICON_FA_CALCULATOR " FLOPs Calculator")) {
                // Opens Model Summary with FLOPs column visible
                if (open_model_summary_callback_) open_model_summary_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_DIAGRAM_PROJECT " Architecture Diagram")) {
                if (open_architecture_diagram_callback_) open_architecture_diagram_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem(ICON_FA_MAGNIFYING_GLASS_CHART " Learning Rate Finder")) {
                if (open_lr_finder_callback_) open_lr_finder_callback_();
            }
            ImGui::EndMenu();
        }

        // ==================== Data Science (Phase 3 + 6C) ====================
        if (ImGui::BeginMenu(ICON_FA_CHART_BAR " Data Science")) {
            // Data Exploration submenu
            if (ImGui::BeginMenu(ICON_FA_MAGNIFYING_GLASS " Data Exploration")) {
                if (ImGui::MenuItem(ICON_FA_CHART_BAR " Data Profiler", "Ctrl+D")) {
                    if (open_data_profiler_callback_) open_data_profiler_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Comprehensive dataset analysis with statistics and distributions");
                }

                if (ImGui::MenuItem(ICON_FA_CIRCLE_EXCLAMATION " Missing Value Analysis")) {
                    if (open_missing_value_callback_) open_missing_value_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Analyze and visualize missing values in your data");
                }

                if (ImGui::MenuItem(ICON_FA_TRIANGLE_EXCLAMATION " Outlier Detection")) {
                    if (open_outlier_detection_callback_) open_outlier_detection_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Detect outliers using IQR or Z-score methods");
                }

                if (ImGui::MenuItem(ICON_FA_TABLE " Correlation Matrix")) {
                    if (open_correlation_matrix_callback_) open_correlation_matrix_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Visualize correlations between numeric columns");
                }

                ImGui::EndMenu();
            }

            // Data Transformation submenu (Phase 6C)
            if (ImGui::BeginMenu(ICON_FA_SLIDERS " Data Transformation")) {
                if (ImGui::MenuItem(ICON_FA_ARROWS_LEFT_RIGHT " Normalization (Min-Max)")) {
                    if (open_normalization_callback_) open_normalization_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Scale data to a specified range [0,1] or custom");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_SIMPLE " Standardization (Z-Score)")) {
                    if (open_standardization_callback_) open_standardization_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Transform to mean=0 and std=1, with robust option");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_LINE " Log Transform")) {
                    if (open_log_transform_callback_) open_log_transform_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Apply log, log10, log2, or log1p transform");
                }

                if (ImGui::MenuItem(ICON_FA_WAND_MAGIC_SPARKLES " Box-Cox Transform")) {
                    if (open_boxcox_callback_) open_boxcox_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Power transform for normality with automatic lambda selection");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_LAYER_GROUP " Feature Scaling (All Methods)")) {
                    if (open_feature_scaling_callback_) open_feature_scaling_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compare and apply all scaling methods: Min-Max, Z-Score, Robust, MaxAbs, Quantile");
                }

                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Statistics (Phase 4) ====================
        if (ImGui::BeginMenu(ICON_FA_CALCULATOR " Statistics")) {
            // Descriptive Statistics submenu
            if (ImGui::BeginMenu(ICON_FA_LIST " Descriptive Statistics")) {
                if (ImGui::MenuItem(ICON_FA_TABLE " Summary Statistics")) {
                    if (open_descriptive_stats_callback_) open_descriptive_stats_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Calculate mean, median, std dev, quartiles, skewness, kurtosis");
                }
                ImGui::EndMenu();
            }

            // Inferential Statistics submenu
            if (ImGui::BeginMenu(ICON_FA_WEIGHT_SCALE " Inferential Statistics")) {
                if (ImGui::MenuItem(ICON_FA_FLASK " Hypothesis Testing")) {
                    if (open_hypothesis_test_callback_) open_hypothesis_test_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("t-Test, ANOVA, Chi-Square, Mann-Whitney U tests");
                }
                ImGui::EndMenu();
            }

            // Regression Analysis submenu
            if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Regression Analysis")) {
                if (ImGui::MenuItem(ICON_FA_SITEMAP " Linear/Polynomial Regression")) {
                    if (open_regression_callback_) open_regression_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Linear, Polynomial, and Multiple regression with diagnostics");
                }
                ImGui::EndMenu();
            }

            // Probability Distributions submenu
            if (ImGui::BeginMenu(ICON_FA_CHART_AREA " Probability Distributions")) {
                if (ImGui::MenuItem(ICON_FA_CHART_PIE " Distribution Fitter")) {
                    if (open_distribution_fitter_callback_) open_distribution_fitter_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Fit Normal, Uniform, Exponential, LogNormal distributions");
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Machine Learning (Phase 5 + 6A + 6B) ====================
        if (ImGui::BeginMenu(ICON_FA_BRAIN " Machine Learning")) {
            // Model Evaluation (Phase 6B)
            if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Model Evaluation")) {
                if (ImGui::MenuItem(ICON_FA_REPEAT " Cross-Validation")) {
                    if (open_cross_validation_callback_) open_cross_validation_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("K-Fold and Stratified K-Fold cross-validation");
                }

                if (ImGui::MenuItem(ICON_FA_TABLE_CELLS " Confusion Matrix")) {
                    if (open_confusion_matrix_callback_) open_confusion_matrix_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Visualize classification results with precision, recall, F1");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_AREA " ROC Curve / AUC")) {
                    if (open_roc_auc_callback_) open_roc_auc_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("ROC curve with AUC metric and threshold analysis");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_LINE " Precision-Recall Curve")) {
                    if (open_pr_curve_callback_) open_pr_curve_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("PR curve with average precision and F1 optimization");
                }

                if (ImGui::MenuItem(ICON_FA_GRADUATION_CAP " Learning Curves")) {
                    if (open_learning_curves_callback_) open_learning_curves_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Diagnose bias-variance tradeoff with learning curves");
                }

                ImGui::EndMenu();
            }

            // Clustering (Phase 6A)
            if (ImGui::BeginMenu(ICON_FA_OBJECT_GROUP " Clustering")) {
                if (ImGui::MenuItem(ICON_FA_BULLSEYE " K-Means Clustering")) {
                    if (open_kmeans_callback_) open_kmeans_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Partition data into K clusters with centroid-based optimization");
                }

                if (ImGui::MenuItem(ICON_FA_CIRCLE_NODES " DBSCAN")) {
                    if (open_dbscan_callback_) open_dbscan_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Density-based clustering with automatic noise detection");
                }

                if (ImGui::MenuItem(ICON_FA_SITEMAP " Hierarchical Clustering")) {
                    if (open_hierarchical_callback_) open_hierarchical_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Agglomerative clustering with dendrogram visualization");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_PIE " Gaussian Mixture Models")) {
                    if (open_gmm_callback_) open_gmm_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Soft clustering with probabilistic Gaussian components");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_CHART_COLUMN " Cluster Evaluation")) {
                    if (open_cluster_eval_callback_) open_cluster_eval_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Evaluate clustering quality: Silhouette, Davies-Bouldin, Calinski-Harabasz");
                }

                ImGui::EndMenu();
            }

            // Dimensionality Reduction
            if (ImGui::BeginMenu(ICON_FA_COMPRESS " Dimensionality Reduction")) {
                if (ImGui::MenuItem(ICON_FA_CHART_SIMPLE " PCA / t-SNE / UMAP")) {
                    if (open_dim_reduction_callback_) open_dim_reduction_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Reduce dimensions and visualize data with PCA, t-SNE, or UMAP");
                }
                ImGui::EndMenu();
            }

            // Feature Engineering
            if (ImGui::BeginMenu(ICON_FA_STAR " Feature Engineering")) {
                if (ImGui::MenuItem(ICON_FA_RANKING_STAR " Feature Importance")) {
                    if (open_feature_importance_callback_) open_feature_importance_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute permutation, drop-column, or weight-based feature importance");
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Deep Learning (Phase 5) ====================
        if (ImGui::BeginMenu(ICON_FA_NETWORK_WIRED " Deep Learning")) {
            // Interpretability
            if (ImGui::BeginMenu(ICON_FA_EYE " Interpretability")) {
                if (ImGui::MenuItem(ICON_FA_FIRE " Grad-CAM Visualization")) {
                    if (open_gradcam_callback_) open_gradcam_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Visualize CNN decision regions with Grad-CAM heatmaps");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_AREA " Saliency Maps")) {
                    if (open_gradcam_callback_) open_gradcam_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute gradient-based saliency maps and SmoothGrad");
                }
                ImGui::EndMenu();
            }

            // Architecture Tools
            if (ImGui::BeginMenu(ICON_FA_WAND_MAGIC_SPARKLES " Architecture Tools")) {
                if (ImGui::MenuItem(ICON_FA_DNA " Neural Architecture Search")) {
                    if (open_nas_callback_) open_nas_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Score, mutate, and evolve neural network architectures");
                }

                if (ImGui::MenuItem(ICON_FA_LIGHTBULB " Architecture Suggestions")) {
                    if (open_nas_callback_) open_nas_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Generate architecture suggestions for your task");
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        ImGui::Separator();

        // ==================== Numerical Computations (Phase 7) ====================
        if (ImGui::BeginMenu(ICON_FA_SQUARE_ROOT_VARIABLE " Numerical Computations")) {
            // Linear Algebra submenu
            if (ImGui::BeginMenu(ICON_FA_TABLE_CELLS " Linear Algebra")) {
                if (ImGui::MenuItem(ICON_FA_CALCULATOR " Matrix Calculator")) {
                    if (open_matrix_calculator_callback_) open_matrix_calculator_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Add, Subtract, Multiply, Transpose, Inverse, Determinant, Trace, Rank");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_CHART_LINE " Eigenvalue Decomposition")) {
                    if (open_eigen_decomp_callback_) open_eigen_decomp_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute eigenvalues and eigenvectors of square matrices");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_PIE " SVD (Singular Value Decomposition)")) {
                    if (open_svd_callback_) open_svd_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Decompose A = U * S * V^T with low-rank approximation");
                }

                if (ImGui::MenuItem(ICON_FA_TABLE_COLUMNS " QR Decomposition")) {
                    if (open_qr_callback_) open_qr_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Decompose A = Q * R (orthogonal x upper triangular)");
                }

                if (ImGui::MenuItem(ICON_FA_SQUARE_ROOT_VARIABLE " Cholesky Decomposition")) {
                    if (open_cholesky_callback_) open_cholesky_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Decompose A = L * L^T for positive definite matrices");
                }

                ImGui::EndMenu();
            }

            // Signal Processing submenu (Phase 8)
            if (ImGui::BeginMenu(ICON_FA_WAVE_SQUARE " Signal Processing")) {
                if (ImGui::MenuItem(ICON_FA_WAVE_SQUARE " FFT (Fourier Transform)")) {
                    if (open_fft_callback_) open_fft_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute FFT, visualize frequency spectrum and phase");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_AREA " Spectrogram (STFT)")) {
                    if (open_spectrogram_callback_) open_spectrogram_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Time-frequency analysis with Short-Time Fourier Transform");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_FILTER " Filter Designer")) {
                    if (open_filter_designer_callback_) open_filter_designer_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Design FIR filters: Lowpass, Highpass, Bandpass, Bandstop");
                }

                if (ImGui::MenuItem(ICON_FA_ASTERISK " Convolution Calculator")) {
                    if (open_convolution_callback_) open_convolution_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute 1D convolution with various modes and kernel presets");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_WATER " Wavelet Transform")) {
                    if (open_wavelet_callback_) open_wavelet_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Discrete Wavelet Transform with Haar, Daubechies wavelets");
                }

                ImGui::EndMenu();
            }

            // Optimization & Calculus submenu (Phase 9)
            if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Optimization & Calculus")) {
                if (ImGui::MenuItem(ICON_FA_CHART_LINE " Gradient Descent Visualizer")) {
                    if (open_gradient_descent_callback_) open_gradient_descent_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Visualize GD, Momentum, Adam, RMSprop on test functions");
                }

                if (ImGui::MenuItem(ICON_FA_MOUNTAIN " Convexity Analyzer")) {
                    if (open_convexity_callback_) open_convexity_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Analyze convexity via Hessian eigenvalues");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_CHART_PIE " Linear Programming (LP)")) {
                    if (open_lp_callback_) open_lp_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Solve linear programs with Simplex method");
                }

                if (ImGui::MenuItem(ICON_FA_SQUARE_POLL_VERTICAL " Quadratic Programming (QP)")) {
                    if (open_qp_callback_) open_qp_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Solve quadratic programs with active set method");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_SUPERSCRIPT " Numerical Differentiation")) {
                    if (open_differentiation_callback_) open_differentiation_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Forward, backward, central differences with comparison");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_AREA " Numerical Integration")) {
                    if (open_integration_callback_) open_integration_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Trapezoid, Simpson, Romberg, Adaptive, Gaussian methods");
                }

                ImGui::EndMenu();
            }

            // Time Series Analysis submenu (Phase 10)
            if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Time Series Analysis")) {
                if (ImGui::MenuItem(ICON_FA_LAYER_GROUP " Time Series Decomposition")) {
                    if (open_decomposition_callback_) open_decomposition_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Classical and STL decomposition into trend, seasonal, residual");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_BAR " ACF/PACF (Correlogram)")) {
                    if (open_acf_pacf_callback_) open_acf_pacf_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Autocorrelation and Partial Autocorrelation for model identification");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_SCALE_BALANCED " Stationarity Testing")) {
                    if (open_stationarity_callback_) open_stationarity_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("ADF and KPSS tests with differencing suggestions");
                }

                if (ImGui::MenuItem(ICON_FA_CALENDAR " Seasonality Detection")) {
                    if (open_seasonality_callback_) open_seasonality_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Detect seasonal periods using periodogram and ACF peaks");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_CHART_LINE " Forecasting")) {
                    if (open_forecasting_callback_) open_forecasting_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Simple ES, Holt, Holt-Winters, and ARIMA forecasting");
                }

                ImGui::EndMenu();
            }

            // Text Processing submenu (Phase 11)
            if (ImGui::BeginMenu(ICON_FA_FONT " Text Processing")) {
                if (ImGui::MenuItem(ICON_FA_SCISSORS " Tokenization")) {
                    if (open_tokenization_callback_) open_tokenization_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Word, sentence, and n-gram tokenization with statistics");
                }

                if (ImGui::MenuItem(ICON_FA_CHART_BAR " Word Frequency")) {
                    if (open_word_frequency_callback_) open_word_frequency_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Analyze word frequencies, length distribution, and type-token ratio");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_TABLE " TF-IDF Analysis")) {
                    if (open_tfidf_callback_) open_tfidf_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Term frequency-inverse document frequency for document analysis");
                }

                if (ImGui::MenuItem(ICON_FA_CUBE " Word Embeddings")) {
                    if (open_embeddings_callback_) open_embeddings_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Create word embeddings and find similar words");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_FACE_SMILE " Sentiment Analysis")) {
                    if (open_sentiment_callback_) open_sentiment_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Analyze text sentiment with polarity and subjectivity scores");
                }

                ImGui::EndMenu();
            }

            // Utilities submenu (Phase 12)
            if (ImGui::BeginMenu(ICON_FA_TOOLBOX " Utilities")) {
                if (ImGui::MenuItem(ICON_FA_CALCULATOR " Calculator")) {
                    if (open_calculator_callback_) open_calculator_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Scientific calculator with expression evaluation and variables");
                }

                if (ImGui::MenuItem(ICON_FA_SCALE_BALANCED " Unit Converter")) {
                    if (open_unit_converter_callback_) open_unit_converter_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Convert between units: length, mass, temperature, time, data");
                }

                if (ImGui::MenuItem(ICON_FA_DICE " Random Generator")) {
                    if (open_random_generator_callback_) open_random_generator_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Generate random numbers with various distributions");
                }

                ImGui::Separator();

                if (ImGui::MenuItem(ICON_FA_FINGERPRINT " Hash Generator")) {
                    if (open_hash_generator_callback_) open_hash_generator_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Compute MD5, SHA-1, SHA-256, SHA-512 hashes");
                }

                if (ImGui::MenuItem(ICON_FA_CODE " JSON Viewer")) {
                    if (open_json_viewer_callback_) open_json_viewer_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Validate, format, and query JSON data");
                }

                if (ImGui::MenuItem(ICON_FA_ASTERISK " Regex Tester")) {
                    if (open_regex_tester_callback_) open_regex_tester_callback_();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Test regular expressions with pattern matching and replace");
                }

                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
