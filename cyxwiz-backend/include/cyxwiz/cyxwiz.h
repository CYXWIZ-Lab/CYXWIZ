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

namespace cyxwiz {

// Initialize the backend
CYXWIZ_API bool Initialize();

// Shutdown the backend
CYXWIZ_API void Shutdown();

// Get version string
CYXWIZ_API const char* GetVersionString();

} // namespace cyxwiz
