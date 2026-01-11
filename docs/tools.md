# CyxWiz Tools Menu - Design & Architecture

## Overview

This document outlines the design and architecture for the **Tools** menu in CyxWiz Engine. The Tools menu consolidates utility functions, diagnostic tools, and computational utilities for machine learning, deep learning, data science, numerical computations, and statistics.

---

## Menu Structure

### Current Menu Bar
```
File | Edit | View | Nodes | Train | Dataset | Script | Plots | Deploy | Help
```

### Proposed Menu Bar
```
File | Edit | View | Nodes | Train | Tools | Dataset | Script | Plots | Deploy | Help
```

The **Tools** menu will be positioned between **Train** and **Dataset** menus.

---

## Items Moving from Train Menu to Tools Menu

The following items will be relocated from the Train menu to the new Tools menu:

| Item | Current Location | New Location | Shortcut |
|------|-----------------|--------------|----------|
| Resume from Checkpoint... | Train menu | Tools > Checkpoints | - |
| Run Test | Train menu | Tools > Testing | F7 |
| View Test Results | Train menu | Tools > Testing | - |

### Updated Train Menu (After Changes)
```
Train
├── Start Training          F5
├── Pause                   F6
├── Stop                    Shift+F5
├── ─────────────────────
├── Training Settings...
└── Optimizer Settings...
```

---

## Tools Menu Architecture

### Complete Menu Structure

```
Tools
├── Checkpoints
│   ├── Resume from Checkpoint...
│   ├── Save Checkpoint...
│   ├── Manage Checkpoints...
│   └── Auto-checkpoint Settings...
│
├── Testing
│   ├── Run Test                           F7
│   ├── Run Quick Test                     Shift+F7
│   ├── View Test Results
│   ├── Compare Test Results...
│   └── Export Test Report...
│
├── Memory & Performance
│   ├── Memory Monitor
│   ├── GPU Memory Usage
│   ├── Clear Cache
│   ├── Garbage Collection
│   └── Performance Profiler...
│
├── ─────────────────────────────────────────
│
├── Machine Learning
│   ├── Model Analysis
│   │   ├── Model Summary
│   │   ├── Layer-wise Statistics
│   │   ├── Parameter Count
│   │   ├── FLOPs Calculator
│   │   └── Memory Footprint
│   │
│   ├── Feature Engineering
│   │   ├── Feature Importance
│   │   ├── Feature Selection
│   │   ├── Feature Scaling
│   │   ├── One-Hot Encoder
│   │   ├── Label Encoder
│   │   └── Polynomial Features
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA (Principal Component Analysis)
│   │   ├── t-SNE Visualization
│   │   ├── UMAP Visualization
│   │   ├── LDA (Linear Discriminant Analysis)
│   │   └── Autoencoder Compression
│   │
│   ├── Clustering
│   │   ├── K-Means Clustering
│   │   ├── DBSCAN
│   │   ├── Hierarchical Clustering
│   │   ├── Gaussian Mixture Models
│   │   └── Cluster Evaluation (Silhouette, Davies-Bouldin)
│   │
│   └── Model Evaluation
│       ├── Cross-Validation
│       ├── Confusion Matrix
│       ├── ROC Curve / AUC
│       ├── Precision-Recall Curve
│       ├── Learning Curves
│       └── Calibration Curve
│
├── Deep Learning
│   ├── Network Visualization
│   │   ├── Architecture Diagram
│   │   ├── Computation Graph
│   │   ├── Gradient Flow Visualization
│   │   └── Activation Maps
│   │
│   ├── Weight Analysis
│   │   ├── Weight Distribution
│   │   ├── Weight Histogram
│   │   ├── Weight Initialization Checker
│   │   ├── Dead Neuron Detector
│   │   └── Gradient Statistics
│   │
│   ├── Interpretability
│   │   ├── Saliency Maps
│   │   ├── Grad-CAM
│   │   ├── Integrated Gradients
│   │   ├── SHAP Values
│   │   └── LIME Explanations
│   │
│   ├── Hyperparameter Tools
│   │   ├── Learning Rate Finder
│   │   ├── Batch Size Analyzer
│   │   ├── Hyperparameter Search (Grid/Random/Bayesian)
│   │   └── Neural Architecture Search (NAS)
│   │
│   └── Regularization Analysis
│       ├── Dropout Visualization
│       ├── Weight Decay Analysis
│       ├── Batch Norm Statistics
│       └── Overfitting Detector
│
├── Data Science
│   ├── Data Exploration
│   │   ├── Dataset Summary
│   │   ├── Data Profiler
│   │   ├── Missing Value Analysis
│   │   ├── Outlier Detection
│   │   ├── Data Distribution
│   │   └── Correlation Matrix
│   │
│   ├── Data Quality
│   │   ├── Data Validation
│   │   ├── Schema Validator
│   │   ├── Data Type Checker
│   │   ├── Duplicate Finder
│   │   └── Anomaly Detection
│   │
│   ├── Data Transformation
│   │   ├── Normalization
│   │   ├── Standardization
│   │   ├── Log Transform
│   │   ├── Box-Cox Transform
│   │   └── Custom Transform...
│   │
│   ├── Time Series
│   │   ├── Time Series Decomposition
│   │   ├── Autocorrelation Plot
│   │   ├── Stationarity Test (ADF)
│   │   ├── Seasonality Detection
│   │   └── Forecasting Tools
│   │
│   └── Text Processing
│       ├── Tokenization
│       ├── Word Frequency Analysis
│       ├── TF-IDF Calculator
│       ├── Word Embeddings Viewer
│       └── Sentiment Analysis
│
├── Numerical Computations
│   ├── Linear Algebra
│   │   ├── Matrix Calculator
│   │   ├── Eigenvalue Decomposition
│   │   ├── SVD (Singular Value Decomposition)
│   │   ├── QR Decomposition
│   │   ├── Cholesky Decomposition
│   │   └── Matrix Rank/Determinant
│   │
│   ├── Optimization
│   │   ├── Gradient Descent Visualizer
│   │   ├── Convexity Checker
│   │   ├── Constraint Optimizer
│   │   ├── Linear Programming
│   │   └── Quadratic Programming
│   │
│   ├── Calculus
│   │   ├── Numerical Differentiation
│   │   ├── Numerical Integration
│   │   ├── Taylor Series Approximation
│   │   └── Fourier Transform
│   │
│   ├── Signal Processing
│   │   ├── FFT (Fast Fourier Transform)
│   │   ├── Spectrogram
│   │   ├── Filter Designer
│   │   ├── Convolution Calculator
│   │   └── Wavelet Transform
│   │
│   └── Tensor Operations
│       ├── Tensor Reshape Tool
│       ├── Broadcasting Visualizer
│       ├── Einsum Builder
│       └── Tensor Contraction
│
├── Statistics
│   ├── Descriptive Statistics
│   │   ├── Summary Statistics
│   │   ├── Central Tendency (Mean, Median, Mode)
│   │   ├── Dispersion (Variance, Std Dev, IQR)
│   │   ├── Skewness & Kurtosis
│   │   └── Percentile Calculator
│   │
│   ├── Inferential Statistics
│   │   ├── Hypothesis Testing
│   │   │   ├── t-Test (One/Two Sample)
│   │   │   ├── ANOVA
│   │   │   ├── Chi-Square Test
│   │   │   ├── Mann-Whitney U Test
│   │   │   └── Kolmogorov-Smirnov Test
│   │   │
│   │   ├── Confidence Intervals
│   │   ├── Effect Size Calculator
│   │   └── Power Analysis
│   │
│   ├── Regression Analysis
│   │   ├── Linear Regression
│   │   ├── Logistic Regression
│   │   ├── Polynomial Regression
│   │   ├── Ridge/Lasso Regression
│   │   └── Residual Analysis
│   │
│   ├── Probability Distributions
│   │   ├── Distribution Fitter
│   │   ├── Normal Distribution
│   │   ├── Binomial Distribution
│   │   ├── Poisson Distribution
│   │   ├── Uniform Distribution
│   │   └── Custom Distribution
│   │
│   └── Bayesian Statistics
│       ├── Prior/Posterior Visualization
│       ├── Bayesian Inference
│       ├── MCMC Diagnostics
│       └── Credible Intervals
│
├── ─────────────────────────────────────────
│
├── Utilities
│   ├── Calculator
│   ├── Unit Converter
│   ├── Random Number Generator
│   ├── Hash Generator (MD5, SHA256)
│   ├── JSON Validator
│   └── Regex Tester
│
└── Custom Tools...
```

---

## Implementation Architecture

### File Structure

```
cyxwiz-engine/
└── src/
    └── gui/
        └── panels/
            ├── toolbar.h              # Add RenderToolsMenu() declaration
            ├── toolbar.cpp            # Add Tools menu call in Render()
            └── toolbar_tools_menu.cpp # NEW: Tools menu implementation
```

### Header Additions (toolbar.h)

```cpp
// In toolbar.h, add to public methods:
void RenderToolsMenu();

// Add callback declarations for Tools functionality:
std::function<void()> resume_checkpoint_callback_;
std::function<void()> save_checkpoint_callback_;
std::function<void()> manage_checkpoints_callback_;
std::function<void()> run_test_callback_;          // Already exists
std::function<void()> run_quick_test_callback_;
std::function<void()> view_test_results_callback_; // Already exists
std::function<void()> compare_test_results_callback_;
std::function<void()> export_test_report_callback_;
std::function<void()> open_memory_monitor_callback_;
std::function<void()> open_gpu_monitor_callback_;
std::function<void()> clear_cache_callback_;
std::function<void()> run_gc_callback_;
std::function<void()> open_profiler_callback_;

// ML Tools callbacks
std::function<void()> show_model_summary_callback_;
std::function<void()> calculate_flops_callback_;
std::function<void()> open_pca_tool_callback_;
std::function<void()> open_tsne_tool_callback_;
std::function<void()> open_clustering_tool_callback_;
std::function<void()> open_confusion_matrix_callback_;

// Deep Learning Tools callbacks
std::function<void()> show_architecture_diagram_callback_;
std::function<void()> show_weight_distribution_callback_;
std::function<void()> open_gradcam_tool_callback_;
std::function<void()> open_lr_finder_callback_;

// Data Science Tools callbacks
std::function<void()> open_data_profiler_callback_;
std::function<void()> open_correlation_matrix_callback_;
std::function<void()> open_outlier_detection_callback_;

// Numerical computation callbacks
std::function<void()> open_matrix_calculator_callback_;
std::function<void()> open_fft_tool_callback_;
std::function<void()> open_gradient_visualizer_callback_;

// Statistics callbacks
std::function<void()> open_hypothesis_test_callback_;
std::function<void()> open_distribution_fitter_callback_;
std::function<void()> open_regression_tool_callback_;

// Add show flags for new dialogs/panels
bool show_memory_monitor_ = false;
bool show_model_summary_ = false;
bool show_lr_finder_ = false;
bool show_data_profiler_ = false;
bool show_statistics_panel_ = false;
```

### Main Menu Render Update (toolbar.cpp)

```cpp
// In Render() method, add RenderToolsMenu() call:
if (ImGui::BeginMainMenuBar()) {
    RenderFileMenu();
    RenderEditMenu();
    RenderViewMenu();
    RenderNodesMenu();
    RenderTrainMenu();
    RenderToolsMenu();     // NEW: Add Tools menu
    RenderDatasetMenu();
    RenderScriptMenu();
    RenderPlotsMenu();
    RenderDeployMenu();
    RenderHelpMenu();
    // ...
}
```

### Tools Menu Implementation (toolbar_tools_menu.cpp)

```cpp
#include "toolbar.h"
#include "../icons.h"
#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderToolsMenu() {
    if (ImGui::BeginMenu("Tools")) {
        // ==================== Checkpoints ====================
        if (ImGui::BeginMenu(ICON_FA_CLOCK_ROTATE_LEFT " Checkpoints")) {
            if (ImGui::MenuItem("Resume from Checkpoint...")) {
                if (resume_checkpoint_callback_) resume_checkpoint_callback_();
            }
            if (ImGui::MenuItem("Save Checkpoint...")) {
                if (save_checkpoint_callback_) save_checkpoint_callback_();
            }
            if (ImGui::MenuItem("Manage Checkpoints...")) {
                if (manage_checkpoints_callback_) manage_checkpoints_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Auto-checkpoint Settings...")) {
                // TODO: Open auto-checkpoint settings dialog
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
                show_memory_monitor_ = true;
            }
            if (ImGui::MenuItem("GPU Memory Usage")) {
                if (open_gpu_monitor_callback_) open_gpu_monitor_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Clear Cache")) {
                if (clear_cache_callback_) clear_cache_callback_();
            }
            if (ImGui::MenuItem("Garbage Collection")) {
                if (run_gc_callback_) run_gc_callback_();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Performance Profiler...")) {
                if (open_profiler_callback_) open_profiler_callback_();
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // ==================== Machine Learning ====================
        if (ImGui::BeginMenu(ICON_FA_ROBOT " Machine Learning")) {
            // Model Analysis submenu
            if (ImGui::BeginMenu("Model Analysis")) {
                if (ImGui::MenuItem("Model Summary")) {
                    show_model_summary_ = true;
                }
                if (ImGui::MenuItem("Layer-wise Statistics")) {
                    // TODO
                }
                if (ImGui::MenuItem("Parameter Count")) {
                    // TODO
                }
                if (ImGui::MenuItem("FLOPs Calculator")) {
                    if (calculate_flops_callback_) calculate_flops_callback_();
                }
                if (ImGui::MenuItem("Memory Footprint")) {
                    // TODO
                }
                ImGui::EndMenu();
            }

            // Feature Engineering submenu
            if (ImGui::BeginMenu("Feature Engineering")) {
                if (ImGui::MenuItem("Feature Importance")) {}
                if (ImGui::MenuItem("Feature Selection")) {}
                if (ImGui::MenuItem("Feature Scaling")) {}
                ImGui::Separator();
                if (ImGui::MenuItem("One-Hot Encoder")) {}
                if (ImGui::MenuItem("Label Encoder")) {}
                if (ImGui::MenuItem("Polynomial Features")) {}
                ImGui::EndMenu();
            }

            // Dimensionality Reduction submenu
            if (ImGui::BeginMenu("Dimensionality Reduction")) {
                if (ImGui::MenuItem("PCA (Principal Component Analysis)")) {
                    if (open_pca_tool_callback_) open_pca_tool_callback_();
                }
                if (ImGui::MenuItem("t-SNE Visualization")) {
                    if (open_tsne_tool_callback_) open_tsne_tool_callback_();
                }
                if (ImGui::MenuItem("UMAP Visualization")) {}
                if (ImGui::MenuItem("LDA (Linear Discriminant Analysis)")) {}
                if (ImGui::MenuItem("Autoencoder Compression")) {}
                ImGui::EndMenu();
            }

            // Clustering submenu
            if (ImGui::BeginMenu("Clustering")) {
                if (ImGui::MenuItem("K-Means Clustering")) {
                    if (open_clustering_tool_callback_) open_clustering_tool_callback_();
                }
                if (ImGui::MenuItem("DBSCAN")) {}
                if (ImGui::MenuItem("Hierarchical Clustering")) {}
                if (ImGui::MenuItem("Gaussian Mixture Models")) {}
                ImGui::Separator();
                if (ImGui::MenuItem("Cluster Evaluation")) {}
                ImGui::EndMenu();
            }

            // Model Evaluation submenu
            if (ImGui::BeginMenu("Model Evaluation")) {
                if (ImGui::MenuItem("Cross-Validation")) {}
                if (ImGui::MenuItem("Confusion Matrix")) {
                    if (open_confusion_matrix_callback_) open_confusion_matrix_callback_();
                }
                if (ImGui::MenuItem("ROC Curve / AUC")) {}
                if (ImGui::MenuItem("Precision-Recall Curve")) {}
                if (ImGui::MenuItem("Learning Curves")) {}
                if (ImGui::MenuItem("Calibration Curve")) {}
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Deep Learning ====================
        if (ImGui::BeginMenu(ICON_FA_BRAIN " Deep Learning")) {
            // Network Visualization submenu
            if (ImGui::BeginMenu("Network Visualization")) {
                if (ImGui::MenuItem("Architecture Diagram")) {
                    if (show_architecture_diagram_callback_) show_architecture_diagram_callback_();
                }
                if (ImGui::MenuItem("Computation Graph")) {}
                if (ImGui::MenuItem("Gradient Flow Visualization")) {}
                if (ImGui::MenuItem("Activation Maps")) {}
                ImGui::EndMenu();
            }

            // Weight Analysis submenu
            if (ImGui::BeginMenu("Weight Analysis")) {
                if (ImGui::MenuItem("Weight Distribution")) {
                    if (show_weight_distribution_callback_) show_weight_distribution_callback_();
                }
                if (ImGui::MenuItem("Weight Histogram")) {}
                if (ImGui::MenuItem("Weight Initialization Checker")) {}
                if (ImGui::MenuItem("Dead Neuron Detector")) {}
                if (ImGui::MenuItem("Gradient Statistics")) {}
                ImGui::EndMenu();
            }

            // Interpretability submenu
            if (ImGui::BeginMenu("Interpretability")) {
                if (ImGui::MenuItem("Saliency Maps")) {}
                if (ImGui::MenuItem("Grad-CAM")) {
                    if (open_gradcam_tool_callback_) open_gradcam_tool_callback_();
                }
                if (ImGui::MenuItem("Integrated Gradients")) {}
                if (ImGui::MenuItem("SHAP Values")) {}
                if (ImGui::MenuItem("LIME Explanations")) {}
                ImGui::EndMenu();
            }

            // Hyperparameter Tools submenu
            if (ImGui::BeginMenu("Hyperparameter Tools")) {
                if (ImGui::MenuItem("Learning Rate Finder")) {
                    show_lr_finder_ = true;
                }
                if (ImGui::MenuItem("Batch Size Analyzer")) {}
                if (ImGui::MenuItem("Hyperparameter Search...")) {}
                if (ImGui::MenuItem("Neural Architecture Search (NAS)")) {}
                ImGui::EndMenu();
            }

            // Regularization Analysis submenu
            if (ImGui::BeginMenu("Regularization Analysis")) {
                if (ImGui::MenuItem("Dropout Visualization")) {}
                if (ImGui::MenuItem("Weight Decay Analysis")) {}
                if (ImGui::MenuItem("Batch Norm Statistics")) {}
                if (ImGui::MenuItem("Overfitting Detector")) {}
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Data Science ====================
        if (ImGui::BeginMenu(ICON_FA_CHART_LINE " Data Science")) {
            // Data Exploration submenu
            if (ImGui::BeginMenu("Data Exploration")) {
                if (ImGui::MenuItem("Dataset Summary")) {}
                if (ImGui::MenuItem("Data Profiler")) {
                    if (open_data_profiler_callback_) open_data_profiler_callback_();
                }
                if (ImGui::MenuItem("Missing Value Analysis")) {}
                if (ImGui::MenuItem("Outlier Detection")) {
                    if (open_outlier_detection_callback_) open_outlier_detection_callback_();
                }
                if (ImGui::MenuItem("Data Distribution")) {}
                if (ImGui::MenuItem("Correlation Matrix")) {
                    if (open_correlation_matrix_callback_) open_correlation_matrix_callback_();
                }
                ImGui::EndMenu();
            }

            // Data Quality submenu
            if (ImGui::BeginMenu("Data Quality")) {
                if (ImGui::MenuItem("Data Validation")) {}
                if (ImGui::MenuItem("Schema Validator")) {}
                if (ImGui::MenuItem("Data Type Checker")) {}
                if (ImGui::MenuItem("Duplicate Finder")) {}
                if (ImGui::MenuItem("Anomaly Detection")) {}
                ImGui::EndMenu();
            }

            // Data Transformation submenu
            if (ImGui::BeginMenu("Data Transformation")) {
                if (ImGui::MenuItem("Normalization")) {}
                if (ImGui::MenuItem("Standardization")) {}
                if (ImGui::MenuItem("Log Transform")) {}
                if (ImGui::MenuItem("Box-Cox Transform")) {}
                if (ImGui::MenuItem("Custom Transform...")) {}
                ImGui::EndMenu();
            }

            // Time Series submenu
            if (ImGui::BeginMenu("Time Series")) {
                if (ImGui::MenuItem("Time Series Decomposition")) {}
                if (ImGui::MenuItem("Autocorrelation Plot")) {}
                if (ImGui::MenuItem("Stationarity Test (ADF)")) {}
                if (ImGui::MenuItem("Seasonality Detection")) {}
                if (ImGui::MenuItem("Forecasting Tools")) {}
                ImGui::EndMenu();
            }

            // Text Processing submenu
            if (ImGui::BeginMenu("Text Processing")) {
                if (ImGui::MenuItem("Tokenization")) {}
                if (ImGui::MenuItem("Word Frequency Analysis")) {}
                if (ImGui::MenuItem("TF-IDF Calculator")) {}
                if (ImGui::MenuItem("Word Embeddings Viewer")) {}
                if (ImGui::MenuItem("Sentiment Analysis")) {}
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Numerical Computations ====================
        if (ImGui::BeginMenu(ICON_FA_CALCULATOR " Numerical")) {
            // Linear Algebra submenu
            if (ImGui::BeginMenu("Linear Algebra")) {
                if (ImGui::MenuItem("Matrix Calculator")) {
                    if (open_matrix_calculator_callback_) open_matrix_calculator_callback_();
                }
                if (ImGui::MenuItem("Eigenvalue Decomposition")) {}
                if (ImGui::MenuItem("SVD (Singular Value Decomposition)")) {}
                if (ImGui::MenuItem("QR Decomposition")) {}
                if (ImGui::MenuItem("Cholesky Decomposition")) {}
                if (ImGui::MenuItem("Matrix Rank/Determinant")) {}
                ImGui::EndMenu();
            }

            // Optimization submenu
            if (ImGui::BeginMenu("Optimization")) {
                if (ImGui::MenuItem("Gradient Descent Visualizer")) {
                    if (open_gradient_visualizer_callback_) open_gradient_visualizer_callback_();
                }
                if (ImGui::MenuItem("Convexity Checker")) {}
                if (ImGui::MenuItem("Constraint Optimizer")) {}
                if (ImGui::MenuItem("Linear Programming")) {}
                if (ImGui::MenuItem("Quadratic Programming")) {}
                ImGui::EndMenu();
            }

            // Calculus submenu
            if (ImGui::BeginMenu("Calculus")) {
                if (ImGui::MenuItem("Numerical Differentiation")) {}
                if (ImGui::MenuItem("Numerical Integration")) {}
                if (ImGui::MenuItem("Taylor Series Approximation")) {}
                if (ImGui::MenuItem("Fourier Transform")) {}
                ImGui::EndMenu();
            }

            // Signal Processing submenu
            if (ImGui::BeginMenu("Signal Processing")) {
                if (ImGui::MenuItem("FFT (Fast Fourier Transform)")) {
                    if (open_fft_tool_callback_) open_fft_tool_callback_();
                }
                if (ImGui::MenuItem("Spectrogram")) {}
                if (ImGui::MenuItem("Filter Designer")) {}
                if (ImGui::MenuItem("Convolution Calculator")) {}
                if (ImGui::MenuItem("Wavelet Transform")) {}
                ImGui::EndMenu();
            }

            // Tensor Operations submenu
            if (ImGui::BeginMenu("Tensor Operations")) {
                if (ImGui::MenuItem("Tensor Reshape Tool")) {}
                if (ImGui::MenuItem("Broadcasting Visualizer")) {}
                if (ImGui::MenuItem("Einsum Builder")) {}
                if (ImGui::MenuItem("Tensor Contraction")) {}
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        // ==================== Statistics ====================
        if (ImGui::BeginMenu(ICON_FA_SQUARE_ROOT_VARIABLE " Statistics")) {
            // Descriptive Statistics submenu
            if (ImGui::BeginMenu("Descriptive Statistics")) {
                if (ImGui::MenuItem("Summary Statistics")) {}
                if (ImGui::MenuItem("Central Tendency (Mean, Median, Mode)")) {}
                if (ImGui::MenuItem("Dispersion (Variance, Std Dev, IQR)")) {}
                if (ImGui::MenuItem("Skewness & Kurtosis")) {}
                if (ImGui::MenuItem("Percentile Calculator")) {}
                ImGui::EndMenu();
            }

            // Inferential Statistics submenu
            if (ImGui::BeginMenu("Inferential Statistics")) {
                if (ImGui::BeginMenu("Hypothesis Testing")) {
                    if (ImGui::MenuItem("t-Test (One/Two Sample)")) {
                        if (open_hypothesis_test_callback_) open_hypothesis_test_callback_();
                    }
                    if (ImGui::MenuItem("ANOVA")) {}
                    if (ImGui::MenuItem("Chi-Square Test")) {}
                    if (ImGui::MenuItem("Mann-Whitney U Test")) {}
                    if (ImGui::MenuItem("Kolmogorov-Smirnov Test")) {}
                    ImGui::EndMenu();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Confidence Intervals")) {}
                if (ImGui::MenuItem("Effect Size Calculator")) {}
                if (ImGui::MenuItem("Power Analysis")) {}
                ImGui::EndMenu();
            }

            // Regression Analysis submenu
            if (ImGui::BeginMenu("Regression Analysis")) {
                if (ImGui::MenuItem("Linear Regression")) {
                    if (open_regression_tool_callback_) open_regression_tool_callback_();
                }
                if (ImGui::MenuItem("Logistic Regression")) {}
                if (ImGui::MenuItem("Polynomial Regression")) {}
                if (ImGui::MenuItem("Ridge/Lasso Regression")) {}
                if (ImGui::MenuItem("Residual Analysis")) {}
                ImGui::EndMenu();
            }

            // Probability Distributions submenu
            if (ImGui::BeginMenu("Probability Distributions")) {
                if (ImGui::MenuItem("Distribution Fitter")) {
                    if (open_distribution_fitter_callback_) open_distribution_fitter_callback_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Normal Distribution")) {}
                if (ImGui::MenuItem("Binomial Distribution")) {}
                if (ImGui::MenuItem("Poisson Distribution")) {}
                if (ImGui::MenuItem("Uniform Distribution")) {}
                if (ImGui::MenuItem("Custom Distribution")) {}
                ImGui::EndMenu();
            }

            // Bayesian Statistics submenu
            if (ImGui::BeginMenu("Bayesian Statistics")) {
                if (ImGui::MenuItem("Prior/Posterior Visualization")) {}
                if (ImGui::MenuItem("Bayesian Inference")) {}
                if (ImGui::MenuItem("MCMC Diagnostics")) {}
                if (ImGui::MenuItem("Credible Intervals")) {}
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        ImGui::Separator();

        // ==================== Utilities ====================
        if (ImGui::BeginMenu(ICON_FA_WRENCH " Utilities")) {
            if (ImGui::MenuItem(ICON_FA_CALCULATOR " Calculator")) {}
            if (ImGui::MenuItem("Unit Converter")) {}
            if (ImGui::MenuItem("Random Number Generator")) {}
            if (ImGui::MenuItem("Hash Generator (MD5, SHA256)")) {}
            if (ImGui::MenuItem("JSON Validator")) {}
            if (ImGui::MenuItem("Regex Tester")) {}
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Custom Tools...")) {
            // TODO: Open custom tools manager
        }

        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
```

---

## Integration with Existing System

### Current App State Integration

The Tools menu integrates with the following existing systems:

| System | Integration Point |
|--------|-------------------|
| **Training System** | TrainingExecutor for checkpoints, test results |
| **Data Registry** | DataRegistry for dataset operations |
| **Node Editor** | NodeEditor for model analysis |
| **Python Engine** | PythonEngine for numerical computations |
| **Backend** | cyxwiz-backend for tensor operations |

### Callback Connections (main_window.cpp)

```cpp
// In MainWindow constructor, connect callbacks:
toolbar_->resume_checkpoint_callback_ = [this]() {
    // Open checkpoint browser dialog
    // Load selected checkpoint into TrainingExecutor
};

toolbar_->show_model_summary_callback_ = [this]() {
    // Get model from node_editor_
    // Display summary in new panel
};

toolbar_->calculate_flops_callback_ = [this]() {
    // Calculate FLOPs from current model graph
    // Display in dialog
};

toolbar_->open_data_profiler_callback_ = [this]() {
    // Get loaded dataset from data_registry_
    // Open DataProfiler panel
};

toolbar_->open_correlation_matrix_callback_ = [this]() {
    // Compute correlation from data_registry_
    // Display as heatmap
};
```

---

## CMakeLists.txt Update

Add the new source file to the build:

```cmake
# In cyxwiz-engine/CMakeLists.txt, add to SOURCES:
set(GUI_SOURCES
    # ... existing sources ...
    src/gui/panels/toolbar_tools_menu.cpp  # NEW
)
```

---

## Priority Implementation Order

### Phase 1: Core Infrastructure (High Priority)
1. Create toolbar_tools_menu.cpp with basic menu structure
2. Move Checkpoint/Testing items from Train menu
3. Implement Memory Monitor panel
4. Connect existing callbacks (run_test, view_test_results)

### Phase 2: Model Analysis Tools (High Priority)
1. Model Summary dialog
2. FLOPs Calculator
3. Architecture Diagram viewer
4. Learning Rate Finder

### Phase 3: Data Science Tools (Medium Priority)
1. Data Profiler panel
2. Correlation Matrix visualization
3. Missing Value Analysis
4. Outlier Detection

### Phase 4: Statistical Tools (Medium Priority)
1. Descriptive Statistics calculator
2. Hypothesis Testing wizard
3. Distribution Fitter
4. Regression Analysis

### Phase 5: Advanced Tools (Lower Priority)
1. t-SNE/UMAP Visualization
2. Grad-CAM implementation
3. SHAP/LIME integrations
4. Neural Architecture Search

---

## Dialog/Panel Specifications

### Memory Monitor Panel

```cpp
// Displays real-time memory usage
struct MemoryMonitorPanel {
    float cpu_memory_used_mb;
    float cpu_memory_total_mb;
    float gpu_memory_used_mb;
    float gpu_memory_total_mb;
    std::vector<float> memory_history;  // Ring buffer for graph
    bool auto_refresh = true;
    float refresh_interval_ms = 500.0f;
};
```

### Model Summary Dialog

```cpp
// Shows model architecture details
struct ModelSummaryDialog {
    std::string model_name;
    size_t total_params;
    size_t trainable_params;
    size_t non_trainable_params;
    float estimated_size_mb;
    std::vector<LayerInfo> layers;

    struct LayerInfo {
        std::string name;
        std::string type;
        std::vector<int> input_shape;
        std::vector<int> output_shape;
        size_t param_count;
    };
};
```

### Data Profiler Panel

```cpp
// Comprehensive data analysis
struct DataProfilerPanel {
    std::string dataset_name;
    size_t num_samples;
    size_t num_features;

    struct ColumnProfile {
        std::string name;
        std::string dtype;
        size_t non_null_count;
        size_t null_count;
        double null_percentage;
        double mean, median, std;
        double min, max;
        std::vector<std::pair<std::string, int>> top_values;  // For categorical
    };

    std::vector<ColumnProfile> columns;
};
```

---

## Keyboard Shortcuts

| Action | Shortcut | Description |
|--------|----------|-------------|
| Run Test | F7 | Execute model test |
| Run Quick Test | Shift+F7 | Fast test with reduced samples |
| Memory Monitor | Ctrl+M | Toggle memory monitor panel |
| Model Summary | Ctrl+I | Show model information |
| Data Profiler | Ctrl+D | Open data profiler |

---

## Future Considerations

1. **Plugin System**: Allow third-party tools to be added to Tools menu
2. **Tool Presets**: Save frequently used tool configurations
3. **Batch Processing**: Run multiple tools in sequence
4. **Export Results**: Export tool outputs to various formats
5. **Command Palette**: Quick access to all tools via Ctrl+Shift+P

 Tools
  ├── Checkpoints
  │   ├── Resume from Checkpoint...
  │   └── Save Checkpoint...
  ├── Testing
  │   ├── Run Test (F7)
  │   ├── Run Quick Test (Shift+F7)
  │   ├── View Test Results
  │   ├── Compare Test Results...
  │   └── Export Test Report...
  └── Memory & Performance
      ├── Memory Monitor
      ├── Clear Cache
      └── Garbage Collection

  Memory Monitor Features:

  - Process memory usage tracking (Windows PSAPI)
  - Real-time CPU memory history graph (ImPlot)
  - GPU memory placeholder (for future ArrayFire integration)
  - Clear Cache and Force GC buttons

  - FLOPs calculation for Dense, Conv2D, LSTM, Attention, BatchNorm, Pooling
  - Parameter counting per layer type (trainable/non-trainable)
  - Shape inference through the network
  - Export to text/JSON/clipboard
  - Visual diagram with zoom, pan, color-coded layer types
  - LR Finder with exponential/linear schedules, smoothing, suggested LR

  To test the new Model Analysis tools:
  1. Create a node graph in the Node Editor (add Dense, Conv2D layers, etc.)
  2. Go to Tools > Model Analysis menu:
    - Model Summary (Ctrl+I) - See layer table, params, FLOPs
    - Architecture Diagram - Visual block diagram of your network
    - Learning Rate Finder - Run LR range test

      1. Tools → Statistics → Descriptive Statistics → Summary Statistics
    - Select a loaded dataset and column to see mean, median, mode, quartiles, etc.
    - Includes box plot and histogram visualizations
  2. Tools → Statistics → Inferential Statistics → Hypothesis Testing
    - Choose test type (One-sample t-test, Two-sample t-test, ANOVA, Chi-Square)
    - Configure parameters and see p-values, effect sizes, confidence intervals
  3. Tools → Statistics → Regression Analysis → Linear/Polynomial Regression
    - Select X and Y columns
    - View coefficients, R², scatter plot with fitted line, residuals
  4. Tools → Statistics → Probability Distributions → Distribution Fitter
    - Auto-fits multiple distributions (Normal, Uniform, Exponential, LogNormal)
    - Ranked by AIC/BIC with QQ-plot visualization


 Phase 6A - Clustering Panels (Tools → Machine Learning → Clustering)

  | Panel        | Menu Location                   | Test Steps                                               |
  |--------------|---------------------------------|----------------------------------------------------------|
  | K-Means      | Clustering → K-Means            | Load data table, select columns, set k=3, run clustering |
  | DBSCAN       | Clustering → DBSCAN             | Load data, set eps/min_samples, run clustering           |
  | Hierarchical | Clustering → Hierarchical       | Load data, select linkage method, visualize dendrogram   |
  | GMM          | Clustering → GMM                | Load data, set components, run EM algorithm              |
  | Cluster Eval | Clustering → Cluster Evaluation | Run after clustering, check silhouette/Davies-Bouldin    |

  Phase 6B - Model Evaluation (Tools → Machine Learning → Model Evaluation)

  | Panel            | Menu Location                       | Test Steps                                   |
  |------------------|-------------------------------------|----------------------------------------------|
  | Cross-Validation | Model Evaluation → Cross-Validation | Select model, set k-folds, run CV            |
  | Confusion Matrix | Model Evaluation → Confusion Matrix | Load predictions/labels, view matrix heatmap |
  | ROC/AUC          | Model Evaluation → ROC Curve        | Load binary predictions, view ROC curve      |
  | PR Curve         | Model Evaluation → PR Curve         | Load predictions, view precision-recall      |
  | Learning Curves  | Model Evaluation → Learning Curves  | Run training, view learning progression      |

  Phase 6C - Data Transformation (Tools → Data Science → Data Transformation)

  | Panel           | Menu Location                         | Test Steps                               |
  |-----------------|---------------------------------------|------------------------------------------|
  | Normalization   | Data Transformation → Normalize       | Load data, apply min-max normalization   |
  | Standardization | Data Transformation → Standardize     | Load data, apply z-score standardization |
  | Log Transform   | Data Transformation → Log Transform   | Load positive data, apply log transform  |
  | Box-Cox         | Data Transformation → Box-Cox         | Load positive data, apply Box-Cox        |
  | Feature Scaling | Data Transformation → Feature Scaling | Load data, apply scaling methods         |

  To test: Open the Tools menu and navigate to each panel to verify they load without errors. The panels should
  display their UI components correctly since they now link to the backend algorithms via <cyxwiz/clustering.h>,
  <cyxwiz/model_evaluation.h>, and <cyxwiz/data_transform.h>.


 🔲 REMAINING (Phases 7-12)

     ---
     Phase 7: Numerical Computations - Linear Algebra

     Backend: cyxwiz-backend/include/cyxwiz/linear_algebra.h

     7.1 Matrix Calculator Panel

     - Matrix input (manual entry or from dataset)
     - Operations: Add, Subtract, Multiply, Transpose, Inverse
     - Determinant, Trace, Rank calculation
     - GPU-accelerated via ArrayFire

     7.2 Eigenvalue Decomposition Panel

     - Input square matrix
     - Compute eigenvalues and eigenvectors
     - Visualization of eigenspectrum
     - af::eigen() for GPU acceleration

     7.3 SVD Panel (Singular Value Decomposition)

     - Input: Any matrix
     - Output: U, S, V^T matrices
     - Singular value plot
     - Rank approximation
     - af::svd() for GPU

     7.4 QR Decomposition Panel

     - Input matrix
     - Output: Q (orthogonal), R (upper triangular)
     - Visualization
     - af::qr() for GPU

     7.5 Cholesky Decomposition Panel

     - Input: Positive definite matrix
     - Output: Lower triangular L where A = LL^T
     - Validation of positive definiteness
     - af::cholesky() for GPU

     Files to Create:
     cyxwiz-backend/include/cyxwiz/linear_algebra.h
     cyxwiz-backend/src/algorithms/linear_algebra.cpp
     cyxwiz-engine/src/gui/panels/matrix_calculator_panel.h/cpp
     cyxwiz-engine/src/gui/panels/eigen_decomp_panel.h/cpp
     cyxwiz-engine/src/gui/panels/svd_panel.h/cpp
     cyxwiz-engine/src/gui/panels/qr_panel.h/cpp
     cyxwiz-engine/src/gui/panels/cholesky_panel.h/cpp

     ---
     Phase 8: Numerical Computations - Signal Processing

     Backend: cyxwiz-backend/include/cyxwiz/signal_processing.h

     8.1 FFT Panel (Fast Fourier Transform)

     - Input: 1D signal or 2D image
     - Output: Frequency domain representation
     - Magnitude and Phase plots
     - af::fft(), af::fft2() for GPU

     8.2 Spectrogram Panel

     - Input: Audio/time series data
     - STFT (Short-Time Fourier Transform)
     - Time-frequency heatmap visualization
     - Window size, overlap parameters

     8.3 Filter Designer Panel

     - Filter types: Low-pass, High-pass, Band-pass, Band-stop
     - Filter designs: Butterworth, Chebyshev, FIR
     - Frequency response plot
     - Apply to signal

     8.4 Convolution Calculator Panel

     - Input: Signal and kernel
     - 1D/2D convolution
     - Visualization of input, kernel, output
     - af::convolve() for GPU

     8.5 Wavelet Transform Panel

     - Discrete Wavelet Transform (DWT)
     - Wavelet families: Haar, Daubechies
     - Multi-level decomposition
     - Coefficient visualization

     Files to Create:
     cyxwiz-backend/include/cyxwiz/signal_processing.h
     cyxwiz-backend/src/algorithms/signal_processing.cpp
     cyxwiz-engine/src/gui/panels/fft_panel.h/cpp
     cyxwiz-engine/src/gui/panels/spectrogram_panel.h/cpp
     cyxwiz-engine/src/gui/panels/filter_designer_panel.h/cpp
     cyxwiz-engine/src/gui/panels/convolution_panel.h/cpp
     cyxwiz-engine/src/gui/panels/wavelet_panel.h/cpp

     ---
     Phase 9: Numerical Computations - Optimization & Calculus

     Backend: cyxwiz-backend/include/cyxwiz/optimization.h

     9.1 Gradient Descent Visualizer Panel

     - 2D/3D loss landscape visualization
     - Step-by-step gradient descent animation
     - Multiple optimizers: SGD, Momentum, Adam
     - Learning rate effects

     9.2 Convexity Checker Panel

     - Hessian computation
     - Eigenvalue analysis for convexity
     - Saddle point detection
     - Visualization

     9.3 Linear Programming Panel

     - Objective function input
     - Constraint editor
     - Simplex algorithm
     - Solution visualization (feasible region)

     9.4 Quadratic Programming Panel

     - QP problem formulation
     - Equality/inequality constraints
     - KKT conditions
     - Solution output

     9.5 Numerical Differentiation Panel

     - Function input (symbolic or tabular)
     - Forward, backward, central differences
     - Gradient visualization

     9.6 Numerical Integration Panel

     - Function input
     - Methods: Trapezoidal, Simpson's, Gaussian quadrature
     - Area visualization
     - Error estimation

     Files to Create:
     cyxwiz-backend/include/cyxwiz/optimization.h
     cyxwiz-backend/src/algorithms/optimization.cpp
     cyxwiz-engine/src/gui/panels/gradient_visualizer_panel.h/cpp
     cyxwiz-engine/src/gui/panels/convexity_panel.h/cpp
     cyxwiz-engine/src/gui/panels/linear_programming_panel.h/cpp
     cyxwiz-engine/src/gui/panels/quadratic_programming_panel.h/cpp
     cyxwiz-engine/src/gui/panels/numerical_diff_panel.h/cpp
     cyxwiz-engine/src/gui/panels/numerical_integration_panel.h/cpp

     ---
     Phase 10: Time Series Analysis

     Backend: cyxwiz-backend/include/cyxwiz/time_series.h

     10.1 Time Series Decomposition Panel

     - Additive/Multiplicative decomposition
     - Trend, Seasonal, Residual components
     - STL decomposition
     - Component plots

     10.2 Autocorrelation Panel

     - ACF (Autocorrelation Function) plot
     - PACF (Partial Autocorrelation) plot
     - Confidence intervals
     - Lag selection guidance

     10.3 Stationarity Test Panel (ADF)

     - Augmented Dickey-Fuller test
     - KPSS test
     - Test statistics and p-values
     - Differencing suggestions

     10.4 Seasonality Detection Panel

     - Periodogram analysis
     - Seasonal period detection
     - Multiple seasonality support
     - Visualization

     10.5 Forecasting Panel

     - Moving Average
     - Exponential Smoothing (Simple, Holt, Holt-Winters)
     - ARIMA parameter selection
     - Forecast visualization with confidence intervals

     Files to Create:
     cyxwiz-backend/include/cyxwiz/time_series.h
     cyxwiz-backend/src/algorithms/time_series.cpp
     cyxwiz-engine/src/gui/panels/ts_decomposition_panel.h/cpp
     cyxwiz-engine/src/gui/panels/autocorrelation_panel.h/cpp
     cyxwiz-engine/src/gui/panels/stationarity_panel.h/cpp
     cyxwiz-engine/src/gui/panels/seasonality_panel.h/cpp
     cyxwiz-engine/src/gui/panels/forecasting_panel.h/cpp

     ---
     Phase 11: Text Processing & NLP

     Backend: cyxwiz-backend/include/cyxwiz/text_processing.h

     11.1 Tokenization Panel

     - Word tokenization
     - Sentence tokenization
     - Custom delimiter support
     - Token frequency display

     11.2 Word Frequency Panel

     - Word count analysis
     - N-gram frequency
     - Word cloud visualization
     - Stop word filtering

     11.3 TF-IDF Calculator Panel

     - Document-term matrix
     - TF-IDF computation
     - Feature extraction for ML
     - Export to dataset

     11.4 Word Embeddings Viewer Panel

     - Load pre-trained embeddings (Word2Vec, GloVe format)
     - 2D/3D visualization (via t-SNE/UMAP)
     - Word similarity search
     - Analogy computation

     11.5 Sentiment Analysis Panel

     - Basic lexicon-based sentiment
     - Positive/Negative/Neutral classification
     - Sentence-level and document-level
     - Visualization

     Files to Create:
     cyxwiz-backend/include/cyxwiz/text_processing.h
     cyxwiz-backend/src/algorithms/text_processing.cpp
     cyxwiz-engine/src/gui/panels/tokenization_panel.h/cpp
     cyxwiz-engine/src/gui/panels/word_frequency_panel.h/cpp
     cyxwiz-engine/src/gui/panels/tfidf_panel.h/cpp
     cyxwiz-engine/src/gui/panels/embeddings_viewer_panel.h/cpp
     cyxwiz-engine/src/gui/panels/sentiment_panel.h/cpp

     ---
     Phase 12: Utilities (Engine-Only, No Backend)

     12.1 Calculator Panel

     - Scientific calculator UI
     - Expression parser
     - History
     - Variables/constants

     12.2 Unit Converter Panel

     - Length, Mass, Temperature, Time, etc.
     - Custom unit definitions
     - Conversion history

     12.3 Random Number Generator Panel

     - Distributions: Uniform, Normal, Exponential, Poisson
     - Seed control
     - Histogram visualization
     - Export to dataset

     12.4 Hash Generator Panel

     - MD5, SHA-1, SHA-256, SHA-512
     - File hash computation
     - Text hash computation
     - Comparison tool

     12.5 JSON Validator Panel

     - JSON syntax validation
     - Pretty print / minify
     - JSON path query
     - Schema validation

     12.6 Regex Tester Panel

     - Pattern input
     - Test string input
     - Match highlighting
     - Capture groups display
     - Common patterns library

     Files to Create:
     cyxwiz-engine/src/gui/panels/calculator_panel.h/cpp
     cyxwiz-engine/src/gui/panels/unit_converter_panel.h/cpp
     cyxwiz-engine/src/gui/panels/random_generator_panel.h/cpp
     cyxwiz-engine/src/gui/panels/hash_generator_panel.h/cpp
     cyxwiz-engine/src/gui/panels/json_validator_panel.h/cpp
     cyxwiz-engine/src/gui/panels/regex_tester_panel.h/cpp

phase 7
 Backend (cyxwiz-backend/include/cyxwiz/linear_algebra.h):
  - Matrix operations: Add, Subtract, Multiply, ScalarMultiply, Transpose, Inverse
  - Scalar properties: Determinant, Trace, Rank, FrobeniusNorm, ConditionNumber
  - Decompositions: Eigen, SVD, QR, Cholesky, LU
  - Utility functions: IsSymmetric, IsPositiveDefinite, IsOrthogonal, LowRankApproximation

  5 New GUI Panels:
  1. Matrix Calculator Panel - Interactive matrix operations with A and B matrices
  2. Eigenvalue Decomposition Panel - Complex eigenvalues/eigenvectors with spectrum visualization
  3. SVD Panel - Singular value decomposition with variance analysis and low-rank approximation
  4. QR Decomposition Panel - Q (orthogonal) and R (upper triangular) with verification
  5. Cholesky Decomposition Panel - L * L^T decomposition with positive definiteness checking

  Toolbar Integration:
  - Added under: Tools → Numerical Computations → Linear Algebra

  All panels feature:
  - Async computation with loading indicators
  - Matrix input grids with editable cells
  - Result visualization with ImPlot
  - Verification/validation of decompositions
  - Error handling with user-friendly 
  
    Phase 9 Summary - All panels implemented:
  - Gradient Descent Visualizer (with Vanilla GD, Momentum, Adam, RMSprop)
  - Convexity Analyzer (Hessian eigenvalue analysis)
  - Linear Programming (Simplex method with graphical solution)
  - Quadratic Programming (Active Set method)
  - Numerical Differentiation (Forward, Backward, Central differences)
  - Numerical Integration (Trapezoid, Simpson, Romberg, Gaussian quadrature)

  To test the new panels:
  Navigate to Tools > Numerical Computations > Optimization & Calculus in the menu bar.

  Each panel includes:
  - Test function presets (Rosenbrock, Sphere, Rastrigin, etc.)
  - Interactive parameter controls
  - Real-time visualization with ImPlot
  - Async computation for smooth UI

● Phase 9: Optimization & Calculus - Complete!

 Phase 10: Time Series Analysis is now complete!

  Here's a summary of what was implemented:

  Backend (cyxwiz-backend)

  - time_series.h/cpp: Full time series analysis backend with:
    - DecompositionResult - Classical and STL decomposition
    - AutocorrelationResult - ACF/PACF computation with confidence intervals
    - StationarityResult - ADF and KPSS tests
    - SeasonalityResult - Periodogram and peak detection
    - ForecastResult - Point forecasts with prediction intervals

  GUI Panels (cyxwiz-engine)

  1. Decomposition Panel - Classical/STL decomposition with trend, seasonal, residual plots
  2. ACF/PACF Panel - Correlogram with model order suggestions and Ljung-Box test
  3. Stationarity Panel - ADF/KPSS tests, rolling statistics, differencing
  4. Seasonality Panel - Periodogram, ACF peaks, strength measurement
  5. Forecasting Panel - Simple ES, Holt Linear, Holt-Winters, ARIMA methods

  Menu Integration

  - Added "Time Series Analysis" submenu under Tools > Numerical Computations with all 5 panels

  Build Fixes Applied

    Summary of Phase 11 Implementation:

  | Component                       | Status     |
  |---------------------------------|------------|
  | Backend (text_processing.h/cpp) | ✅ Complete |
  | Tokenization Panel              | ✅ Complete |
  | Word Frequency Panel            | ✅ Complete |
  | TF-IDF Panel                    | ✅ Complete |
  | Word Embeddings Panel           | ✅ Complete |
  | Sentiment Analysis Panel        | ✅ Complete |
  | Menu Integration                | ✅ Complete |

  New Menu Structure (Tools > Numerical Computations > Text Processing):
  - Tokenization
  - Word Frequency
  - TF-IDF Analysis
  - Word Embeddings
  - Sentiment Analysis

  Files Modified:
  - cyxwiz-backend/include/cyxwiz/text_processing.h - Result structs and class declaration
  - cyxwiz-backend/src/algorithms/text_processing.cpp - Full implementations
  - cyxwiz-engine/src/gui/panels/ - 5 new panel files
  - cyxwiz-engine/src/gui/panels/toolbar.h - Callback setters
  - cyxwiz-engine/src/gui/panels/toolbar_tools_menu.cpp - Text Processing submenu
  - cyxwiz-engine/src/gui/main_window.h/cpp - Panel integration
  - cyxwiz-engine/src/gui/icons.h - Added face icons for sentiment
  - CMakeLists.txt files - Source file additions

  Current: Phase 12 - Utilities (6 tools)

 ---
 Utilities to Implement

 | #   | Utility                 | Description                                      | Dependencies               |
 |-----|-------------------------|--------------------------------------------------|----------------------------|
 | 1   | Calculator              | Scientific calculator with expression evaluation | None (pure C++)            |
 | 2   | Unit Converter          | Length, mass, temperature, time, data units      | None (pure C++)            |
 | 3   | Random Number Generator | Various distributions, seed control              | std::random                |
 | 4   | Hash Generator          | MD5, SHA-1, SHA-256, SHA-512                     | OpenSSL (vcpkg)            |
 | 5   | JSON Viewer             | Pretty-print, validate, edit JSON                | nlohmann::json (available) |
 | 6   | Regex Tester            | Pattern matching, capture groups, replace        | std::regex                 |

 ---
 Backend: Result Structures

 File: cyxwiz-backend/include/cyxwiz/utilities.h

 // Calculator Result
 struct CYXWIZ_API CalculatorResult {
     double result = 0.0;
     std::string formatted_result;
     std::string expression;
     std::string parsed_expression;
     std::vector<std::pair<std::string, double>> variables;
     bool success = false;
     std::string error_message;
 };

 // Unit Conversion Result
 struct CYXWIZ_API UnitConversionResult {
     double input_value = 0.0;
     double output_value = 0.0;
     std::string input_unit;
     std::string output_unit;
     std::string category;
     std::string formula;
     std::vector<std::pair<std::string, double>> all_conversions;
     bool success = false;
     std::string error_message;
 };

 // Random Number Result
 struct CYXWIZ_API RandomNumberResult {
     std::vector<double> values;
     double min_value = 0.0;
     double max_value = 0.0;
     double mean = 0.0;
     double std_dev = 0.0;
     std::string distribution;
     uint64_t seed_used = 0;
     int count = 0;
     std::map<int, int> histogram;
     bool success = false;
     std::string error_message;
 };

 // Hash Result
 struct CYXWIZ_API HashResult {
     std::string input_text;
     std::string input_file;
     size_t input_size = 0;
     std::string md5_hash;
     std::string sha1_hash;
     std::string sha256_hash;
     std::string sha512_hash;
     std::string algorithm;
     double compute_time_ms = 0.0;
     bool success = false;
     std::string error_message;
 };

 // JSON Result
 struct CYXWIZ_API JSONResult {
     std::string input_json;
     std::string formatted_json;
     std::string minified_json;
     bool is_valid = false;
     int error_line = -1;
     int error_column = -1;
     std::string error_detail;
     int depth = 0;
     int object_count = 0;
     int array_count = 0;
     int string_count = 0;
     int number_count = 0;
     bool success = false;
     std::string error_message;
 };

 // Regex Result
 struct CYXWIZ_API RegexMatch {
     std::string match_text;
     int start_pos = 0;
     int end_pos = 0;
     std::vector<std::string> groups;
 };

 struct CYXWIZ_API RegexResult {
     std::string pattern;
     std::string input_text;
     std::string flags;
     std::vector<RegexMatch> matches;
     int match_count = 0;
     std::string replaced_text;
     bool is_valid_pattern = false;
     std::string pattern_error;
     bool success = false;
     std::string error_message;
 };

 ---
 Backend: Utilities Class API

 class CYXWIZ_API Utilities {
 public:
     // Calculator
     static CalculatorResult Evaluate(const std::string& expression,
         const std::map<std::string, double>& variables = {},
         const std::string& angle_mode = "radians");
     static std::map<std::string, std::string> GetSupportedFunctions();
     static std::map<std::string, double> GetSupportedConstants();

     // Unit Converter
     static UnitConversionResult ConvertUnit(double value,
         const std::string& from_unit, const std::string& to_unit);
     static std::vector<std::string> GetUnitCategories();
     static std::vector<std::string> GetUnitsForCategory(const std::string& category);
     static UnitConversionResult ConvertToAllUnits(double value, const std::string& from_unit);

     // Random Number Generator
     static RandomNumberResult GenerateUniform(int count, double min, double max, int64_t seed = -1);
     static RandomNumberResult GenerateNormal(int count, double mean, double std_dev, int64_t seed = -1);
     static RandomNumberResult GenerateExponential(int count, double lambda, int64_t seed = -1);
     static RandomNumberResult GeneratePoisson(int count, double lambda, int64_t seed = -1);
     static RandomNumberResult GenerateIntegers(int count, int min, int max, int64_t seed = -1);
     static std::vector<std::string> GenerateUUIDs(int count = 1);

     // Hash Generator
     static HashResult HashText(const std::string& text, const std::string& algorithm = "sha256");
     static HashResult HashFile(const std::string& file_path, const std::string& algorithm = "sha256");
     static bool VerifyHash(const std::string& text, const std::string& expected, const std::string& algorithm);

     // JSON Viewer
     static JSONResult ValidateJSON(const std::string& json_text);
     static JSONResult FormatJSON(const std::string& json_text, int indent_size = 2);
     static JSONResult MinifyJSON(const std::string& json_text);
     static std::string GetJSONValue(const std::string& json_text, const std::string& path);

     // Regex Tester
     static RegexResult TestRegex(const std::string& pattern, const std::string& text, const std::string& flags = "");
     static RegexResult ReplaceRegex(const std::string& pattern, const std::string& text,
         const std::string& replacement, const std::string& flags = "");
     static bool IsValidRegex(const std::string& pattern);
     static std::map<std::string, std::string> GetCommonPatterns();
 };

 ---
 Files to Create

 Backend (2 files)

 cyxwiz-backend/include/cyxwiz/utilities.h
 cyxwiz-backend/src/algorithms/utilities.cpp

 GUI Panels (12 files)

 cyxwiz-engine/src/gui/panels/calculator_panel.h
 cyxwiz-engine/src/gui/panels/calculator_panel.cpp
 cyxwiz-engine/src/gui/panels/unit_converter_panel.h
 cyxwiz-engine/src/gui/panels/unit_converter_panel.cpp
 cyxwiz-engine/src/gui/panels/random_generator_panel.h
 cyxwiz-engine/src/gui/panels/random_generator_panel.cpp
 cyxwiz-engine/src/gui/panels/hash_generator_panel.h
 cyxwiz-engine/src/gui/panels/hash_generator_panel.cpp
 cyxwiz-engine/src/gui/panels/json_viewer_panel.h
 cyxwiz-engine/src/gui/panels/json_viewer_panel.cpp
 cyxwiz-engine/src/gui/panels/regex_tester_panel.h
 cyxwiz-engine/src/gui/panels/regex_tester_panel.cpp

 ---
 Files to Modify

 | File                                                | Changes                                               |
 |-----------------------------------------------------|-------------------------------------------------------|
 | cyxwiz-backend/CMakeLists.txt                       | Add utilities.h to HEADERS, utilities.cpp to SOURCES  |
 | cyxwiz-backend/include/cyxwiz/cyxwiz.h              | Add #include "utilities.h"                            |
 | cyxwiz-engine/CMakeLists.txt                        | Add 12 panel files to ENGINE_SOURCES/HEADERS          |
 | cyxwiz-engine/src/gui/panels/toolbar.h              | Add 6 callback setters + 6 callback members           |
 | cyxwiz-engine/src/gui/panels/toolbar_tools_menu.cpp | Add Utilities submenu with 6 items                    |
 | cyxwiz-engine/src/gui/main_window.h                 | Add 6 panel forward declarations + unique_ptr members |
 | cyxwiz-engine/src/gui/main_window.cpp               | Instantiate panels, wire callbacks, render            |
 | cyxwiz-engine/src/gui/icons.h                       | Add ICON_FA_FINGERPRINT, ICON_FA_TOOLBOX              |

 ---
 Icons to Add to icons.h

 #define ICON_FA_FINGERPRINT     "\xef\x95\xb7"  // U+F577 - hash generator
 #define ICON_FA_TOOLBOX         "\xef\x95\x92"  // U+F552 - utilities menu

 ---
 Menu Structure

 Under Tools > Numerical Computations:
 Utilities (ICON_FA_TOOLBOX)
 ├── Calculator (ICON_FA_CALCULATOR)
 ├── Unit Converter (ICON_FA_SCALE_BALANCED)
 ├── Random Generator (ICON_FA_DICE)
 ├── ─────────────────────
 ├── Hash Generator (ICON_FA_FINGERPRINT)
 ├── JSON Viewer (ICON_FA_CODE)
 └── Regex Tester (ICON_FA_ASTERISK)

 ---
 Implementation Order

 Step 1: Backend Foundation

 1. Create utilities.h with result structs and class declaration
 2. Create utilities.cpp with implementations
 3. Update CMakeLists.txt files

 Step 2: Calculator Panel

 - Expression parser (recursive descent: +, -, *, /, ^, functions)
 - Supported functions: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, sqrt, cbrt, exp, log, log10, log2, abs,
 floor, ceil, round, pow, mod
 - Constants: pi, e, phi (golden ratio)
 - Variable support for custom values

 Step 3: Unit Converter Panel

 - Categories: Length, Mass, Temperature, Time, Data, Area, Volume, Speed
 - Show all conversions for selected category
 - Formula display

 Step 4: Random Number Generator Panel

 - Distributions: Uniform, Normal, Exponential, Poisson, Integer
 - Histogram visualization with ImPlot
 - Seed control for reproducibility
 - UUID generation

 Step 5: Hash Generator Panel

 - Algorithms: MD5, SHA-1, SHA-256, SHA-512
 - Text and file input modes
 - Hash verification
 - Uses OpenSSL via vcpkg

 Step 6: JSON Viewer Panel

 - Uses nlohmann::json (already in vcpkg)
 - Tabs: Formatted, Tree View, Statistics
 - JSON path query support

 Step 7: Regex Tester Panel

 - Uses std::regex (ECMAScript flavor)
 - Flags: ignore case, global, multiline
 - Match highlighting
 - Replacement preview
 - Common patterns presets (Email, Phone, URL, IP, Date)

 Step 8: Integration

 - Add Utilities submenu to toolbar_tools_menu.cpp
 - Wire callbacks in toolbar.h
 - Add panel members to main_window.h/cpp
 - Build and test

 ---
 Unit Conversion Tables

 Length

 meter (base), kilometer, centimeter, millimeter, mile, yard, foot, inch, nautical mile

 Mass

 kilogram (base), gram, milligram, pound, ounce, ton, stone

 Temperature

 Celsius (base), Fahrenheit, Kelvin
 - C to F: (C * 9/5) + 32
 - C to K: C + 273.15

 Time

 second (base), minute, hour, day, week, year

 Data

 byte (base), kilobyte, megabyte, gigabyte, terabyte, petabyte, bit, kilobit, megabit, gigabit

 ---
 Calculator Functions

 | Category      | Functions                       |
 |---------------|---------------------------------|
 | Trigonometric | sin, cos, tan, asin, acos, atan |
 | Hyperbolic    | sinh, cosh, tanh                |
 | Exponential   | exp, log, log10, log2, pow      |
 | Root          | sqrt, cbrt                      |
 | Rounding      | floor, ceil, round, abs         |
 | Other         | mod, min, max, factorial        |

 ---
 Common Regex Patterns

 | Name       | Pattern                                      |
 |------------|----------------------------------------------|
 | Email      | [\w.+-]+@[\w-]+\.[\w.-]+                     |
 | Phone (US) | \(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4} |
 | URL        | https?://[\w\-._~:/?#\[\]@!$&'()*+,;=]+      |
 | IPv4       | \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}           |
 | Date (ISO) | \d{4}-\d{2}-\d{2}                            |
 | Time (24h) | `([01]?[0-9]                                 |

 ---
 Key Patterns from Previous Phases

 Async Computation

 std::unique_ptr<std::thread> compute_thread_;
 std::atomic<bool> is_computing_{false};
 std::mutex result_mutex_;

 void ComputeAsync() {
     if (is_computing_.load()) return;
     is_computing_ = true;
     compute_thread_ = std::make_unique<std::thread>([this]() {
         std::lock_guard<std::mutex> lock(result_mutex_);
         try {
             result_ = Utilities::Method(...);
             has_result_ = true;
         } catch (...) {
             error_message_ = "Error";
         }
         is_computing_ = false;
     });
 }

 Panel Integration

 // toolbar.h - Add callback setters
 void SetOpenCalculatorCallback(std::function<void()> cb) { open_calculator_callback_ = cb; }
 // ... repeat for other 5 panels

 // main_window.h - Add panel members
 std::unique_ptr<cyxwiz::CalculatorPanel> calculator_panel_;
 // ... repeat for other 5 panels

 // main_window.cpp - Wire callbacks and render
 calculator_panel_ = std::make_unique<cyxwiz::CalculatorPanel>();
 toolbar_->SetOpenCalculatorCallback([this]() {
     if (calculator_panel_) calculator_panel_->SetVisible(true);
 });
 // In Render():
 if (calculator_panel_) calculator_panel_->Render();

 ---
 Estimated Effort

 | Step             | Files      | Complexity                                 |
 |------------------|------------|--------------------------------------------|
 | Backend          | 2          | Medium (expression parser is most complex) |
 | Calculator       | 2          | Medium                                     |
 | Unit Converter   | 2          | Low                                        |
 | Random Generator | 2          | Low                                        |
 | Hash Generator   | 2          | Low (if OpenSSL available)                 |
 | JSON Viewer      | 2          | Medium (tree view)                         |
 | Regex Tester     | 2          | Medium (highlighting)                      |
 | Integration      | 4 modified | Low                                        |

 Total: 14 new files, 8 modified 
 
   Successfully implemented 6 utility tools under Tools > Numerical Computations > Utilities:

  | Tool             | Description                     | Key Features
                                |
  |------------------|---------------------------------|------------------------------------------------------------
  ------------------------------|
  | Calculator       | Scientific expression evaluator | Math functions (sin, cos, sqrt, etc.), constants (pi, e),
  variables, history             |
  | Unit Converter   | Multi-category unit conversion  | Length, Mass, Temperature, Time, Data, Area, Volume, Speed
                                |
  | Random Generator | Random number distributions     | Uniform, Normal, Exponential, Poisson, Integer, UUID
  generation, histogram visualization |
  | Hash Generator   | Cryptographic hashing           | MD5, SHA-1, SHA-256, SHA-512, file hashing, verification
                                |
  | JSON Viewer      | JSON validation & formatting    | Pretty-print, minify, syntax highlighting, statistics, path
   queries                      |
  | Regex Tester     | Regular expression testing      | Pattern matching, capture groups, replacement, common
  patterns library                   |

  Files Created/Modified:
  - Backend: utilities.h, utilities.cpp (result structs and implementations)
  - GUI Panels: 12 new files (6 pairs of .h/.cpp)
  - Integration: toolbar.h, toolbar_tools_menu.cpp, main_window.h, main_window.cpp
  - Build Files: Both CMakeLists.txt updated, icons.h updated with new icons

  The engine (cyxwiz-engine.exe) built successfully. The build errors shown were in an unrelated test project
  (test_job_execution_service), not in the Phase 12 utilities code.


  The plan is ready. Here's a summary:

  MATLAB-Style Command Window Plan Summary

  Goal: Add 45 MATLAB-style functions to the Command Window

  Usage After Implementation:
  # Flat namespace (MATLAB-style)
  A = [[1, 2], [3, 4]]
  U, S, V = svd(A)

  # Or organized namespace
  result = cyx.linalg.svd(A)
  spectrum = cyx.signal.fft(x)
  labels = cyx.stats.kmeans(data, k=3)
  forecast = cyx.timeseries.arima(data, 12)

  Files to Modify:
  | File                                             | Changes                                         |
  |--------------------------------------------------|-------------------------------------------------|
  | cyxwiz-backend/python/bindings.cpp               | Add 4 submodules with 45 functions (~400 lines) |
  | cyxwiz-engine/src/scripting/scripting_engine.cpp | Add flat namespace imports (~30 lines)          |

  Function Categories:
  - Linear Algebra (17): svd, eig, qr, chol, lu, det, rank, inv, solve...
  - Signal Processing (12): fft, ifft, conv, spectrogram, filter...
  - Statistics/Clustering (8): kmeans, dbscan, gmm, pca, tsne...
  - Time Series (8): acf, pacf, arima, decompose, stationarity...

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 MATLAB-Style Command Window Functions

 Overview

 Add MATLAB-style interactive functions to the Command Window so users can type:
 A = [[1, 2], [3, 4]]
 U, S, V = svd(A)          # Flat namespace (simple)
 result = cyx.linalg.svd(A) # Grouped namespace (organized)

 Architecture

 Command Window (Python REPL)
     │
     ├─ pycyxwiz module (existing: Tensor, Device, Layer, Optimizer)
     │
     └─ NEW: pycyxwiz.linalg, pycyxwiz.signal, pycyxwiz.stats, pycyxwiz.timeseries
            │
            └─ Wraps C++ backend: LinearAlgebra, SignalProcessing, Clustering, TimeSeries

 User Requirements

 1. Both namespaces - Flat (svd(A)) + grouped (cyx.linalg.svd(A))
 2. All categories - Linear Algebra, Signal Processing, Statistics/Clustering, Time Series
 3. Auto-refresh Variable Explorer - Already implemented

 ---
 Files to Modify

 Backend (pybind11 bindings)

 | File                               | Changes                                           |
 |------------------------------------|---------------------------------------------------|
 | cyxwiz-backend/python/bindings.cpp | Add submodules: linalg, signal, stats, timeseries |

 Engine (Python initialization)

 | File                                             | Changes                             |
 |--------------------------------------------------|-------------------------------------|
 | cyxwiz-engine/src/scripting/scripting_engine.cpp | Auto-import flat aliases on startup |

 ---
 Implementation Plan

 Step 1: Add pybind11 Submodules to bindings.cpp

 Add after existing bindings (~line 369):

 // ============ LINEAR ALGEBRA SUBMODULE ============
 auto linalg = m.def_submodule("linalg", "Linear algebra functions");

 // Matrix creation
 linalg.def("eye", [](int n) {
     auto result = cyxwiz::LinearAlgebra::Identity(n);
     return result.matrix;
 }, "Create identity matrix", py::arg("n"));

 linalg.def("zeros", [](int rows, int cols) {
     auto result = cyxwiz::LinearAlgebra::Zeros(rows, cols);
     return result.matrix;
 }, "Create zero matrix");

 linalg.def("ones", [](int rows, int cols) {
     auto result = cyxwiz::LinearAlgebra::Ones(rows, cols);
     return result.matrix;
 }, "Create ones matrix");

 // Decompositions
 linalg.def("svd", [](const std::vector<std::vector<double>>& A, bool full_matrices) {
     auto result = cyxwiz::LinearAlgebra::SVD(A, full_matrices);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::make_tuple(result.U, result.S, result.Vt);
 }, "Singular Value Decomposition", py::arg("A"), py::arg("full_matrices") = false);

 linalg.def("eig", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Eigen(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::make_tuple(result.eigenvalues_real, result.eigenvalues_imag, result.eigenvectors);
 }, "Eigenvalue decomposition");

 linalg.def("qr", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::QR(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::make_tuple(result.Q, result.R);
 }, "QR decomposition");

 linalg.def("chol", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Cholesky(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.L;
 }, "Cholesky decomposition");

 linalg.def("lu", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::LU(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::make_tuple(result.L, result.U, result.P);
 }, "LU decomposition");

 // Matrix properties
 linalg.def("det", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Determinant(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.value;
 }, "Compute determinant");

 linalg.def("rank", [](const std::vector<std::vector<double>>& A, double tol) {
     auto result = cyxwiz::LinearAlgebra::Rank(A, tol);
     return static_cast<int>(result.value);
 }, "Compute matrix rank", py::arg("A"), py::arg("tol") = 1e-10);

 linalg.def("trace", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Trace(A);
     return result.value;
 }, "Compute trace");

 linalg.def("norm", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::FrobeniusNorm(A);
     return result.value;
 }, "Frobenius norm");

 linalg.def("cond", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::ConditionNumber(A);
     return result.value;
 }, "Condition number");

 // Matrix operations
 linalg.def("inv", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Inverse(A);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.matrix;
 }, "Matrix inverse");

 linalg.def("transpose", [](const std::vector<std::vector<double>>& A) {
     auto result = cyxwiz::LinearAlgebra::Transpose(A);
     return result.matrix;
 }, "Matrix transpose");

 linalg.def("solve", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
     auto result = cyxwiz::LinearAlgebra::Solve(A, b);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.matrix;
 }, "Solve Ax = b");

 linalg.def("lstsq", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& b) {
     auto result = cyxwiz::LinearAlgebra::LeastSquares(A, b);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.matrix;
 }, "Least squares solution");

 // Matrix multiply
 linalg.def("matmul", [](const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
     auto result = cyxwiz::LinearAlgebra::Multiply(A, B);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.matrix;
 }, "Matrix multiplication");

 // ============ SIGNAL PROCESSING SUBMODULE ============
 auto signal = m.def_submodule("signal", "Signal processing functions");

 signal.def("fft", [](const std::vector<double>& x, double sample_rate) {
     auto result = cyxwiz::SignalProcessing::FFT(x, sample_rate);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "magnitude"_a = result.magnitude,
         "phase"_a = result.phase,
         "frequencies"_a = result.frequencies
     );
 }, "Fast Fourier Transform", py::arg("x"), py::arg("sample_rate") = 1.0);

 signal.def("ifft", [](const std::vector<std::complex<double>>& X) {
     return cyxwiz::SignalProcessing::IFFT(X);
 }, "Inverse FFT");

 signal.def("conv", [](const std::vector<double>& x, const std::vector<double>& h, const std::string& mode) {
     auto result = cyxwiz::SignalProcessing::Convolve1D(x, h, mode);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.output;
 }, "1D Convolution", py::arg("x"), py::arg("h"), py::arg("mode") = "same");

 signal.def("conv2", [](const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& h, const
 std::string& mode) {
     auto result = cyxwiz::SignalProcessing::Convolve2D(x, h, mode);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.output;
 }, "2D Convolution");

 signal.def("spectrogram", [](const std::vector<double>& x, int window_size, int hop_size, double sample_rate, const
 std::string& window) {
     auto result = cyxwiz::SignalProcessing::ComputeSpectrogram(x, window_size, hop_size, sample_rate, window);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "S"_a = result.spectrogram,
         "frequencies"_a = result.frequencies,
         "times"_a = result.times
     );
 }, "Compute spectrogram", py::arg("x"), py::arg("window_size") = 256, py::arg("hop_size") = 128,
 py::arg("sample_rate") = 1.0, py::arg("window") = "hann");

 signal.def("lowpass", [](double cutoff, double fs, int order) {
     auto result = cyxwiz::SignalProcessing::DesignLowpass(cutoff, fs, order);
     return py::dict("b"_a = result.b, "a"_a = result.a);
 }, "Design lowpass filter", py::arg("cutoff"), py::arg("fs"), py::arg("order") = 4);

 signal.def("highpass", [](double cutoff, double fs, int order) {
     auto result = cyxwiz::SignalProcessing::DesignHighpass(cutoff, fs, order);
     return py::dict("b"_a = result.b, "a"_a = result.a);
 }, "Design highpass filter");

 signal.def("bandpass", [](double low, double high, double fs, int order) {
     auto result = cyxwiz::SignalProcessing::DesignBandpass(low, high, fs, order);
     return py::dict("b"_a = result.b, "a"_a = result.a);
 }, "Design bandpass filter");

 signal.def("filter", [](const std::vector<double>& x, const std::vector<double>& b, const std::vector<double>& a) {
     cyxwiz::FilterCoefficients coeffs;
     coeffs.b = b;
     coeffs.a = a;
     return cyxwiz::SignalProcessing::ApplyFilter(x, coeffs);
 }, "Apply filter to signal");

 signal.def("findpeaks", [](const std::vector<double>& x, double min_height, int min_distance) {
     auto peaks = cyxwiz::SignalProcessing::FindPeaks(x, min_height, min_distance);
     std::vector<int> indices;
     std::vector<double> values;
     for (const auto& p : peaks) {
         indices.push_back(p.index);
         values.push_back(p.value);
     }
     return py::dict("indices"_a = indices, "values"_a = values);
 }, "Find peaks in signal", py::arg("x"), py::arg("min_height") = 0.0, py::arg("min_distance") = 1);

 // Signal generation
 signal.def("sine", [](double freq, double fs, int n, double amp, double phase) {
     return cyxwiz::SignalProcessing::GenerateSineWave(freq, fs, n, amp, phase);
 }, "Generate sine wave", py::arg("freq"), py::arg("fs"), py::arg("n"), py::arg("amp") = 1.0, py::arg("phase") = 0.0);

 signal.def("square", [](double freq, double fs, int n, double amp) {
     return cyxwiz::SignalProcessing::GenerateSquareWave(freq, fs, n, amp);
 }, "Generate square wave");

 signal.def("noise", [](int n, double amp) {
     return cyxwiz::SignalProcessing::GenerateWhiteNoise(n, amp);
 }, "Generate white noise");

 // ============ STATISTICS/CLUSTERING SUBMODULE ============
 auto stats = m.def_submodule("stats", "Statistics and clustering functions");

 // Clustering
 stats.def("kmeans", [](const std::vector<std::vector<double>>& data, int k, int max_iter, const std::string& init) {
     auto result = cyxwiz::Clustering::KMeans(data, k, max_iter, init);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "labels"_a = result.labels,
         "centroids"_a = result.centroids,
         "inertia"_a = result.inertia
     );
 }, "K-Means clustering", py::arg("data"), py::arg("k"), py::arg("max_iter") = 300, py::arg("init") = "kmeans++");

 stats.def("dbscan", [](const std::vector<std::vector<double>>& data, double eps, int min_samples) {
     auto result = cyxwiz::Clustering::DBSCAN(data, eps, min_samples);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "labels"_a = result.labels,
         "n_clusters"_a = result.n_clusters
     );
 }, "DBSCAN clustering", py::arg("data"), py::arg("eps"), py::arg("min_samples") = 5);

 stats.def("gmm", [](const std::vector<std::vector<double>>& data, int n_components, const std::string& cov_type) {
     auto result = cyxwiz::Clustering::GMM(data, n_components, cov_type);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "labels"_a = result.labels,
         "means"_a = result.means,
         "weights"_a = result.weights,
         "aic"_a = result.aic,
         "bic"_a = result.bic
     );
 }, "Gaussian Mixture Model", py::arg("data"), py::arg("n_components"), py::arg("cov_type") = "full");

 // Dimensionality reduction
 stats.def("pca", [](const std::vector<std::vector<double>>& data, int n_components) {
     auto result = cyxwiz::DimensionalityReduction::ComputePCA(data, n_components);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "transformed"_a = result.transformed_data,
         "components"_a = result.components,
         "explained_variance"_a = result.explained_variance_ratio
     );
 }, "Principal Component Analysis", py::arg("data"), py::arg("n_components") = 2);

 stats.def("tsne", [](const std::vector<std::vector<double>>& data, int n_dims, int perplexity) {
     auto result = cyxwiz::DimensionalityReduction::ComputetSNE(data, n_dims, perplexity);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.embeddings;
 }, "t-SNE embedding", py::arg("data"), py::arg("n_dims") = 2, py::arg("perplexity") = 30);

 // Evaluation metrics
 stats.def("silhouette", [](const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
     return cyxwiz::Clustering::ComputeSilhouetteScore(data, labels);
 }, "Silhouette score");

 stats.def("confusion_matrix", [](const std::vector<int>& y_true, const std::vector<int>& y_pred) {
     auto result = cyxwiz::ModelEvaluation::ComputeConfusionMatrix(y_true, y_pred);
     return py::dict(
         "matrix"_a = result.matrix,
         "accuracy"_a = result.accuracy,
         "precision"_a = result.precision_per_class,
         "recall"_a = result.recall_per_class,
         "f1"_a = result.f1_per_class
     );
 }, "Compute confusion matrix");

 stats.def("roc", [](const std::vector<int>& y_true, const std::vector<double>& y_scores) {
     auto result = cyxwiz::ModelEvaluation::ComputeROC(y_true, y_scores);
     return py::dict(
         "fpr"_a = result.fpr,
         "tpr"_a = result.tpr,
         "auc"_a = result.auc
     );
 }, "ROC curve and AUC");

 // ============ TIME SERIES SUBMODULE ============
 auto timeseries = m.def_submodule("timeseries", "Time series analysis functions");

 timeseries.def("acf", [](const std::vector<double>& data, int max_lag) {
     auto result = cyxwiz::TimeSeries::ComputeACF(data, max_lag);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "acf"_a = result.acf,
         "confidence_interval"_a = result.confidence_interval
     );
 }, "Autocorrelation function", py::arg("data"), py::arg("max_lag") = -1);

 timeseries.def("pacf", [](const std::vector<double>& data, int max_lag) {
     auto result = cyxwiz::TimeSeries::ComputePACF(data, max_lag);
     if (!result.success) throw std::runtime_error(result.error_message);
     return result.pacf;
 }, "Partial autocorrelation function");

 timeseries.def("decompose", [](const std::vector<double>& data, int period, const std::string& method) {
     auto result = cyxwiz::TimeSeries::Decompose(data, period, method);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "trend"_a = result.trend,
         "seasonal"_a = result.seasonal,
         "residual"_a = result.residual
     );
 }, "Time series decomposition", py::arg("data"), py::arg("period"), py::arg("method") = "additive");

 timeseries.def("stationarity", [](const std::vector<double>& data) {
     auto result = cyxwiz::TimeSeries::TestStationarity(data);
     return py::dict(
         "is_stationary"_a = result.is_stationary,
         "adf_statistic"_a = result.adf_statistic,
         "adf_pvalue"_a = result.adf_pvalue,
         "kpss_statistic"_a = result.kpss_statistic,
         "kpss_pvalue"_a = result.kpss_pvalue
     );
 }, "Test for stationarity (ADF + KPSS)");

 timeseries.def("arima", [](const std::vector<double>& data, int horizon, int p, int d, int q) {
     auto result = cyxwiz::TimeSeries::ARIMA(data, horizon, p, d, q);
     if (!result.success) throw std::runtime_error(result.error_message);
     return py::dict(
         "forecast"_a = result.forecast,
         "lower"_a = result.lower_bound,
         "upper"_a = result.upper_bound,
         "mse"_a = result.mse,
         "aic"_a = result.aic
     );
 }, "ARIMA forecasting", py::arg("data"), py::arg("horizon"), py::arg("p") = -1, py::arg("d") = -1, py::arg("q") = -1);

 timeseries.def("diff", [](const std::vector<double>& data, int order) {
     return cyxwiz::TimeSeries::Difference(data, order);
 }, "Difference series", py::arg("data"), py::arg("order") = 1);

 timeseries.def("rolling_mean", [](const std::vector<double>& data, int window) {
     return cyxwiz::TimeSeries::RollingMean(data, window);
 }, "Rolling mean");

 timeseries.def("rolling_std", [](const std::vector<double>& data, int window) {
     return cyxwiz::TimeSeries::RollingStd(data, window);
 }, "Rolling standard deviation");

 Step 2: Add Flat Namespace Aliases in scripting_engine.cpp

 In ScriptingEngine::Initialize() or startup, inject Python code:

 # Auto-import pycyxwiz submodules
 import pycyxwiz
 cyx = pycyxwiz  # Alias

 # Flat namespace aliases for common functions
 # Linear Algebra
 svd = pycyxwiz.linalg.svd
 eig = pycyxwiz.linalg.eig
 qr = pycyxwiz.linalg.qr
 chol = pycyxwiz.linalg.chol
 lu = pycyxwiz.linalg.lu
 det = pycyxwiz.linalg.det
 rank = pycyxwiz.linalg.rank
 inv = pycyxwiz.linalg.inv
 solve = pycyxwiz.linalg.solve
 eye = pycyxwiz.linalg.eye
 zeros = pycyxwiz.linalg.zeros
 ones = pycyxwiz.linalg.ones

 # Signal Processing
 fft = pycyxwiz.signal.fft
 ifft = pycyxwiz.signal.ifft
 conv = pycyxwiz.signal.conv
 spectrogram = pycyxwiz.signal.spectrogram
 findpeaks = pycyxwiz.signal.findpeaks

 # Statistics/Clustering
 kmeans = pycyxwiz.stats.kmeans
 dbscan = pycyxwiz.stats.dbscan
 pca = pycyxwiz.stats.pca
 tsne = pycyxwiz.stats.tsne

 # Time Series
 acf = pycyxwiz.timeseries.acf
 pacf = pycyxwiz.timeseries.pacf
 arima = pycyxwiz.timeseries.arima
 decompose = pycyxwiz.timeseries.decompose

 Step 3: Add Required Includes to bindings.cpp

 #include <cyxwiz/linear_algebra.h>
 #include <cyxwiz/signal_processing.h>
 #include <cyxwiz/clustering.h>
 #include <cyxwiz/dimensionality_reduction.h>
 #include <cyxwiz/time_series.h>
 #include <cyxwiz/model_evaluation.h>

 ---
 Usage Examples

 After implementation, users can:

 # MATLAB-style flat namespace
 A = [[1, 2], [3, 4]]
 U, S, V = svd(A)
 print(S)  # [5.46, 0.37]

 # Or organized namespace
 result = cyx.linalg.svd(A)

 # Signal processing
 x = cyx.signal.sine(freq=10, fs=100, n=1000)
 spectrum = fft(x)

 # Clustering
 labels = kmeans(data, k=3)['labels']

 # Time series
 forecast = arima(sales_data, horizon=12)

 ---
 Files to Modify Summary

 | File                                             | Changes                                          |
 |--------------------------------------------------|--------------------------------------------------|
 | cyxwiz-backend/python/bindings.cpp               | Add ~400 lines: 4 submodules with all functions  |
 | cyxwiz-engine/src/scripting/scripting_engine.cpp | Add ~30 lines: flat namespace imports on startup |

 ---
 Implementation Steps

 1. Add includes to bindings.cpp for all backend headers
 2. Create submodules (linalg, signal, stats, timeseries)
 3. Bind functions - wrap each C++ function with error handling
 4. Add startup aliases in scripting_engine.cpp
 5. Build and test - verify functions work in Command Window

 ---
 Function Count by Category

 | Category              | Functions
                       |
 |-----------------------|---------------------------------------------------------------------------------------------
 ----------------------|
 | Linear Algebra        | 17 (svd, eig, qr, chol, lu, det, rank, trace, norm, cond, inv, transpose, solve, lstsq,
 matmul, eye, zeros, ones) |
 | Signal Processing     | 12 (fft, ifft, conv, conv2, spectrogram, lowpass, highpass, bandpass, filter, findpeaks,
 sine, square, noise)     |
 | Statistics/Clustering | 8 (kmeans, dbscan, gmm, pca, tsne, silhouette, confusion_matrix, roc)
                       |
 | Time Series           | 8 (acf, pacf, decompose, stationarity, arima, diff, rolling_mean, rolling_std)
                       |
 | Total                 | 45 functions
                       |