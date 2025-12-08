#pragma once

// Main header file for CyxWiz Backend
// Include this to get access to all backend functionality

#define CYXWIZ_VERSION_MAJOR 0
#define CYXWIZ_VERSION_MINOR 1
#define CYXWIZ_VERSION_PATCH 0

// API export macros
#include "api_export.h"

// Core components (order matters - no dependencies first)
#include "memory_manager.h"
#include "device.h"
#include "tensor.h"
#include "engine.h"

// Algorithms (depend on tensor.h)
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "layer.h"
#include "model.h"

// Concrete layer implementations
#include "layers/linear.h"

// Concrete activation implementations
#include "activations/relu.h"
#include "activations/sigmoid.h"
#include "activations/tanh.h"

// Loss functions are in loss.h (MSELoss, CrossEntropyLoss, BCELoss, etc.)

// Sequential model for dynamic layer management
#include "sequential.h"

// Learning rate schedulers
#include "scheduler.h"

// Clustering algorithms (GPU-accelerated)
#include "clustering.h"

// Data transformation (GPU-accelerated)
#include "data_transform.h"

// Model evaluation metrics (GPU-accelerated)
#include "model_evaluation.h"

// Dimensionality reduction (PCA, t-SNE, UMAP)
#include "dimensionality_reduction.h"

// Feature importance analysis
#include "feature_importance.h"

// Model interpretability (Grad-CAM, Saliency Maps)
#include "model_interpretability.h"

// Linear algebra operations (GPU-accelerated)
#include "linear_algebra.h"

// Signal processing (FFT, Convolution, Filters, Wavelets)
#include "signal_processing.h"

// Optimization & Calculus (Gradient Descent, LP, QP, Numerical Methods)
#include "optimization.h"

// Time Series Analysis (Decomposition, ACF, Stationarity, Forecasting)
#include "time_series.h"

// Text Processing (Tokenization, Word Frequency, TF-IDF, Embeddings, Sentiment)
#include "text_processing.h"

// Utilities (Calculator, Unit Converter, Random Generator, Hash, JSON, Regex)
#include "utilities.h"

namespace cyxwiz {

// Initialize the backend
CYXWIZ_API bool Initialize();

// Shutdown the backend
CYXWIZ_API void Shutdown();

// Get version string
CYXWIZ_API const char* GetVersionString();

} // namespace cyxwiz
