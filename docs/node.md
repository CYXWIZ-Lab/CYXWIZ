# CyxWiz Node Editor Architecture

> **Vision**: Build the most powerful visual ML/DL graph editor in the market, inspired by Neo4j's graph database concepts, with intent-based generation, pattern templates, and comprehensive neural network support.

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Architecture Overview](#3-architecture-overview)
4. [Graph Data Model](#4-graph-data-model)
5. [Node Type Taxonomy](#5-node-type-taxonomy)
6. [Intent-Based Generation System](#6-intent-based-generation-system)
7. [Pattern Library & Auto-Generation](#7-pattern-library--auto-generation)
8. [Multi-Connector System](#8-multi-connector-system)
9. [Query Language (CyxQL)](#9-query-language-cyxql)
10. [Visual Editor Enhancements](#10-visual-editor-enhancements)
11. [Backend Integration](#11-backend-integration)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Executive Summary

### 1.1 Goals

- **Neo4j-Inspired**: Treat neural networks as property graphs with nodes (layers) and relationships (connections)
- **Intent-Based Generation**: Natural language to graph: "4-layer MLP with ReLU" → auto-generated network
- **Pattern Templates**: Pre-built architectures (ResNet, LSTM, Transformer, etc.) as reusable patterns
- **Multi-Connector**: Support complex topologies (residual, attention, multi-input/output)
- **Best-in-Market**: Comprehensive ML/DL/RNN/CNN/Transformer support

### 1.2 Key Differentiators

| Feature | CyxWiz | Competitors |
|---------|--------|-------------|
| Intent-based generation | Natural language parsing | Manual drag-drop only |
| Pattern templates | 50+ pre-built architectures | Limited templates |
| Multi-connector | N-to-M connections | 1-to-1 mostly |
| Graph query language | CyxQL (Cypher-inspired) | None |
| Distributed execution | P2P compute network | Local only |
| Framework export | 5 targets (PyTorch, TF, JAX, ONNX, PyCyxWiz) | 1-2 targets |

---

## 2. Current State Analysis

### 2.1 What We Have

```
cyxwiz-engine/src/gui/node_editor.{h,cpp}  [2,116 lines]
├── ImNodes integration (fully working)
├── 16 node types (Input, Output, Dense, Conv2D, Activations, Loss, Optimizer)
├── 4-type pin system (Tensor, Parameters, Loss, Optimizer)
├── Graph operations (topological sort, cycle detection, validation)
├── Code generation (PyTorch, TensorFlow, PyCyxWiz, generic)
├── JSON serialization (.cyxwiz format)
└── Properties panel integration

cyxwiz-backend/
├── Tensor abstraction (ArrayFire backend)
├── LinearLayer (Dense) - full forward/backward
├── Activations: ReLU, Sigmoid, Tanh
├── Losses: MSE, CrossEntropy
├── Optimizers: SGD, Adam, AdamW
└── Device management (GPU/CPU)
```

### 2.2 What's Missing

| Category | Missing Items |
|----------|---------------|
| **Layers** | Conv2D, Conv1D, Conv3D, MaxPool, AvgPool, BatchNorm, LayerNorm, Dropout, Flatten, Reshape, Embedding, LSTM, GRU, RNN, Attention, MultiHeadAttention, Transformer |
| **Connections** | Residual (skip), Concatenate, Add, Multiply, Split, Merge |
| **Features** | Intent parser, pattern library, undo/redo, copy/paste, node groups, subgraphs, zoom controls, minimap |
| **Execution** | Graph compiler, runtime executor, gradient computation, training loop |
| **Export** | ONNX, JAX, SavedModel, checkpoint format |

---

## 3. Architecture Overview

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CyxWiz Node Editor                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Intent Parser│  │Pattern Engine│  │ Graph Editor │  │ Code Gen     │    │
│  │              │  │              │  │              │  │              │    │
│  │ NLP → Graph  │  │ Templates    │  │ ImNodes UI   │  │ Multi-target │    │
│  │ "4 layer..." │  │ ResNet, LSTM │  │ Drag & Drop  │  │ PyTorch, TF  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴────────┬────────┴─────────────────┘            │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                      Graph Data Model (GDM)                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Nodes       │  │ Edges       │  │ Properties  │  │ Metadata   │  │   │
│  │  │ (Vertices)  │  │ (Relations) │  │ (Attributes)│  │ (Schema)   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                      Graph Operations Engine                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Validation  │  │ Topology    │  │ Query (CyxQL)│ │ Transform  │  │   │
│  │  │ Type check  │  │ Sort, BFS   │  │ Pattern match│ │ Optimize   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                         Execution Layer                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Graph        │  │ Runtime      │  │ Distributed  │  │ Export       │     │
│  │ Compiler     │  │ Executor     │  │ Scheduler    │  │ Serializer   │     │
│  │              │  │              │  │              │  │              │     │
│  │ GDM → IR     │  │ ArrayFire    │  │ P2P Network  │  │ ONNX, PTH    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Intent Parser** | Parse natural language descriptions into graph operations |
| **Pattern Engine** | Store, retrieve, and instantiate pre-built network patterns |
| **Graph Editor** | Visual ImNodes-based editing with enhanced features |
| **Code Generator** | Convert graph to PyTorch, TensorFlow, JAX, ONNX, PyCyxWiz |
| **Graph Data Model** | Core data structures for nodes, edges, properties |
| **Query Engine** | CyxQL for programmatic graph manipulation |
| **Graph Compiler** | Convert GDM to intermediate representation for execution |
| **Runtime Executor** | Execute compiled graph on local/remote devices |

---

## 4. Graph Data Model

### 4.1 Neo4j-Inspired Node Structure

```cpp
// Core graph types inspired by Neo4j property graph model
namespace cyxwiz::graph {

// Unique identifier for graph elements
using NodeId = uint64_t;
using EdgeId = uint64_t;
using PropertyKey = std::string;
using PropertyValue = std::variant<
    int64_t, double, bool, std::string,
    std::vector<int64_t>, std::vector<double>,
    std::vector<std::string>, TensorShape
>;

// Labels define node categories (like Neo4j labels)
enum class NodeLabel {
    // Data Flow
    Input, Output, Constant,

    // Core Layers
    Dense, Linear,
    Conv1D, Conv2D, Conv3D,
    ConvTranspose1D, ConvTranspose2D, ConvTranspose3D,

    // Pooling
    MaxPool1D, MaxPool2D, MaxPool3D,
    AvgPool1D, AvgPool2D, AvgPool3D,
    GlobalMaxPool, GlobalAvgPool,
    AdaptiveMaxPool, AdaptiveAvgPool,

    // Normalization
    BatchNorm1D, BatchNorm2D, BatchNorm3D,
    LayerNorm, GroupNorm, InstanceNorm,

    // Recurrent
    RNN, LSTM, GRU, BiLSTM, BiGRU,

    // Attention & Transformers
    Attention, MultiHeadAttention,
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer,
    PositionalEncoding,

    // Activation Functions
    ReLU, LeakyReLU, PReLU, ELU, SELU, GELU,
    Sigmoid, Tanh, Softmax, LogSoftmax,
    Swish, Mish, Hardswish,

    // Regularization
    Dropout, Dropout2D, Dropout3D,
    AlphaDropout, SpatialDropout,

    // Shape Operations
    Flatten, Reshape, Squeeze, Unsqueeze,
    Permute, Transpose, View,

    // Merge Operations (Multi-Input)
    Add, Subtract, Multiply, Divide,
    Concatenate, Stack, Split,

    // Skip Connections
    Residual, DenseConnection, // DenseNet-style

    // Embedding & Text
    Embedding, EmbeddingBag,

    // Loss Functions
    MSELoss, L1Loss, SmoothL1Loss,
    CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss,
    CTCLoss, TripletMarginLoss, CosineEmbeddingLoss,

    // Optimizers
    SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta,
    LAMB, LARS, RAdam, Lookahead,

    // Learning Rate Schedulers
    StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau,

    // Utility
    Identity, Lambda, Custom,

    // Subgraph (for pattern encapsulation)
    Subgraph, Pattern
};

// Edge types define relationship semantics
enum class EdgeType {
    // Data flow edges
    TensorFlow,          // Main data path (tensor from A to B)
    GradientFlow,        // Backward gradient path

    // Parameter edges
    ParameterSource,     // Layer → Optimizer (for parameter updates)
    LossSource,          // Loss → Optimizer

    // Skip connections
    ResidualConnection,  // Skip connection (add)
    DenseConnection,     // Dense connection (concat)

    // Control flow
    SequenceNext,        // Execution order hint
    Conditional,         // Conditional execution

    // Multi-input aggregation
    ConcatInput,         // Multiple inputs to concat node
    AddInput,            // Multiple inputs to add node
    AttentionQuery,      // Q in attention
    AttentionKey,        // K in attention
    AttentionValue       // V in attention
};

// Pin types for type-safe connections
enum class PinType {
    // Tensor types (with shape inference)
    Tensor,              // Generic tensor
    Tensor1D,            // [N, C]
    Tensor2D,            // [N, C, H, W]
    Tensor3D,            // [N, C, D, H, W]
    TensorSequence,      // [N, T, C] for RNN

    // Special types
    Parameters,          // Trainable parameters
    Gradient,            // Gradient tensor
    Loss,                // Scalar loss value
    LearningRate,        // Scalar or scheduled

    // Attention-specific
    Query, Key, Value,   // QKV for attention
    AttentionMask,       // Mask tensor

    // Metadata
    Shape,               // Shape information
    Dtype                // Data type information
};

// Pin definition with validation rules
struct Pin {
    int id;
    std::string name;
    PinType type;
    bool is_input;
    bool is_optional = false;
    bool is_variadic = false;  // Can accept multiple connections
    std::optional<TensorShape> expected_shape;

    // Validation
    bool CanConnectTo(const Pin& other) const;
};

// Node in the property graph
struct Node {
    NodeId id;
    NodeLabel label;
    std::string name;                          // User-visible name
    std::string display_name;                  // Formatted display
    ImVec2 position;                           // Canvas position

    std::vector<Pin> inputs;
    std::vector<Pin> outputs;

    // Properties (Neo4j-style key-value pairs)
    std::map<PropertyKey, PropertyValue> properties;

    // Visual state
    bool is_selected = false;
    bool is_collapsed = false;
    ImVec4 color;

    // Metadata
    std::string description;
    std::string documentation_url;

    // For subgraphs
    std::optional<std::vector<NodeId>> subgraph_nodes;
};

// Edge (relationship) in the property graph
struct Edge {
    EdgeId id;
    NodeId source_node;
    int source_pin;
    NodeId target_node;
    int target_pin;
    EdgeType type;

    // Properties
    std::map<PropertyKey, PropertyValue> properties;

    // Visual state
    bool is_highlighted = false;
    ImVec4 color;
};

// The complete graph
class ComputeGraph {
public:
    // Node operations
    NodeId AddNode(NodeLabel label, const std::string& name = "");
    void RemoveNode(NodeId id);
    Node& GetNode(NodeId id);
    const Node& GetNode(NodeId id) const;

    // Edge operations
    EdgeId AddEdge(NodeId src, int src_pin, NodeId dst, int dst_pin, EdgeType type = EdgeType::TensorFlow);
    void RemoveEdge(EdgeId id);
    bool CanConnect(NodeId src, int src_pin, NodeId dst, int dst_pin) const;

    // Query operations (CyxQL backend)
    std::vector<NodeId> FindNodes(NodeLabel label) const;
    std::vector<NodeId> FindNodes(const std::function<bool(const Node&)>& predicate) const;
    std::vector<EdgeId> FindEdges(EdgeType type) const;
    std::vector<NodeId> GetPredecessors(NodeId id) const;
    std::vector<NodeId> GetSuccessors(NodeId id) const;
    std::vector<std::vector<NodeId>> GetPaths(NodeId from, NodeId to) const;

    // Graph analysis
    std::vector<NodeId> TopologicalSort() const;
    bool HasCycle() const;
    bool IsValid() const;
    std::vector<std::string> Validate() const;  // Returns list of errors

    // Shape inference
    void InferShapes(const std::map<NodeId, TensorShape>& input_shapes);
    TensorShape GetOutputShape(NodeId id, int output_pin = 0) const;

    // Subgraph operations
    NodeId CreateSubgraph(const std::vector<NodeId>& nodes, const std::string& name);
    void ExpandSubgraph(NodeId subgraph_id);

    // Serialization
    std::string ToJSON() const;
    static ComputeGraph FromJSON(const std::string& json);
    std::string ToONNX() const;
    std::string ToPyTorch() const;
    std::string ToTensorFlow() const;

private:
    std::map<NodeId, Node> nodes_;
    std::map<EdgeId, Edge> edges_;
    NodeId next_node_id_ = 1;
    EdgeId next_edge_id_ = 1;

    // Indexes for fast lookup
    std::multimap<NodeLabel, NodeId> label_index_;
    std::multimap<NodeId, EdgeId> outgoing_edges_;
    std::multimap<NodeId, EdgeId> incoming_edges_;
};

} // namespace cyxwiz::graph
```

### 4.2 Property Schema

Each node type has a defined property schema:

```cpp
// Property schemas for each node type
inline std::map<NodeLabel, std::vector<PropertySchema>> NODE_SCHEMAS = {
    {NodeLabel::Dense, {
        {"in_features", PropertyType::Int, true, "Input dimension"},
        {"out_features", PropertyType::Int, true, "Output dimension"},
        {"bias", PropertyType::Bool, false, true, "Use bias"},
        {"activation", PropertyType::String, false, "none", "Built-in activation"}
    }},
    {NodeLabel::Conv2D, {
        {"in_channels", PropertyType::Int, true, "Input channels"},
        {"out_channels", PropertyType::Int, true, "Output channels"},
        {"kernel_size", PropertyType::IntArray, true, "Kernel size [H, W]"},
        {"stride", PropertyType::IntArray, false, {1, 1}, "Stride"},
        {"padding", PropertyType::IntArray, false, {0, 0}, "Padding"},
        {"dilation", PropertyType::IntArray, false, {1, 1}, "Dilation"},
        {"groups", PropertyType::Int, false, 1, "Groups for grouped conv"},
        {"bias", PropertyType::Bool, false, true, "Use bias"}
    }},
    {NodeLabel::LSTM, {
        {"input_size", PropertyType::Int, true, "Input dimension"},
        {"hidden_size", PropertyType::Int, true, "Hidden state dimension"},
        {"num_layers", PropertyType::Int, false, 1, "Number of stacked layers"},
        {"batch_first", PropertyType::Bool, false, true, "Batch dimension first"},
        {"dropout", PropertyType::Float, false, 0.0, "Dropout between layers"},
        {"bidirectional", PropertyType::Bool, false, false, "Bidirectional LSTM"}
    }},
    {NodeLabel::MultiHeadAttention, {
        {"embed_dim", PropertyType::Int, true, "Embedding dimension"},
        {"num_heads", PropertyType::Int, true, "Number of attention heads"},
        {"dropout", PropertyType::Float, false, 0.0, "Attention dropout"},
        {"batch_first", PropertyType::Bool, false, true, "Batch dimension first"},
        {"add_bias_kv", PropertyType::Bool, false, false, "Add bias to K,V"},
        {"kdim", PropertyType::Int, false, -1, "Key dimension (-1 = embed_dim)"},
        {"vdim", PropertyType::Int, false, -1, "Value dimension (-1 = embed_dim)"}
    }},
    // ... more schemas
};
```

---

## 5. Node Type Taxonomy

### 5.1 Complete Node Catalog

```
ROOT
├── DATA_FLOW
│   ├── Input                 # Network input
│   ├── Output                # Network output
│   └── Constant              # Constant tensor
│
├── CORE_LAYERS
│   ├── Linear
│   │   ├── Dense             # Fully connected
│   │   └── Bilinear          # Bilinear transform
│   │
│   ├── Convolutional
│   │   ├── Conv1D            # 1D convolution
│   │   ├── Conv2D            # 2D convolution (images)
│   │   ├── Conv3D            # 3D convolution (video/3D)
│   │   ├── ConvTranspose1D   # Transposed conv 1D
│   │   ├── ConvTranspose2D   # Transposed conv 2D
│   │   ├── ConvTranspose3D   # Transposed conv 3D
│   │   ├── DepthwiseConv2D   # Depthwise separable
│   │   └── PointwiseConv2D   # 1x1 convolution
│   │
│   └── Recurrent
│       ├── RNN               # Simple RNN
│       ├── LSTM              # Long Short-Term Memory
│       ├── GRU               # Gated Recurrent Unit
│       ├── BiLSTM            # Bidirectional LSTM
│       └── BiGRU             # Bidirectional GRU
│
├── POOLING
│   ├── MaxPool1D/2D/3D       # Max pooling
│   ├── AvgPool1D/2D/3D       # Average pooling
│   ├── GlobalMaxPool         # Global max
│   ├── GlobalAvgPool         # Global average
│   └── AdaptivePool          # Adaptive output size
│
├── NORMALIZATION
│   ├── BatchNorm1D/2D/3D     # Batch normalization
│   ├── LayerNorm             # Layer normalization
│   ├── GroupNorm             # Group normalization
│   ├── InstanceNorm          # Instance normalization
│   └── RMSNorm               # Root Mean Square norm
│
├── ATTENTION
│   ├── Attention             # Single-head attention
│   ├── MultiHeadAttention    # Multi-head attention
│   ├── SelfAttention         # Self-attention
│   ├── CrossAttention        # Cross-attention
│   ├── PositionalEncoding    # Sinusoidal encoding
│   └── RelativePositionalEncoding
│
├── TRANSFORMER
│   ├── TransformerEncoder    # Full encoder stack
│   ├── TransformerDecoder    # Full decoder stack
│   ├── TransformerEncoderLayer  # Single encoder layer
│   ├── TransformerDecoderLayer  # Single decoder layer
│   └── Transformer           # Complete transformer
│
├── ACTIVATION
│   ├── ReLU                  # Rectified Linear
│   ├── LeakyReLU             # Leaky ReLU
│   ├── PReLU                 # Parametric ReLU
│   ├── ELU                   # Exponential Linear
│   ├── SELU                  # Scaled ELU
│   ├── GELU                  # Gaussian Error Linear
│   ├── Sigmoid               # Sigmoid
│   ├── Tanh                  # Hyperbolic tangent
│   ├── Softmax               # Softmax
│   ├── LogSoftmax            # Log Softmax
│   ├── Swish                 # x * sigmoid(x)
│   ├── Mish                  # x * tanh(softplus(x))
│   └── Hardswish             # Hard approximation
│
├── REGULARIZATION
│   ├── Dropout               # Standard dropout
│   ├── Dropout2D/3D          # Spatial dropout
│   ├── AlphaDropout          # For SELU
│   └── DropPath              # Stochastic depth
│
├── SHAPE_OPS
│   ├── Flatten               # Flatten to 1D
│   ├── Reshape               # Arbitrary reshape
│   ├── Squeeze               # Remove dim
│   ├── Unsqueeze             # Add dim
│   ├── Permute               # Reorder dims
│   ├── Transpose             # Swap dims
│   └── View                  # View as shape
│
├── MERGE_OPS (Multi-Input)
│   ├── Add                   # Element-wise add
│   ├── Multiply              # Element-wise multiply
│   ├── Concatenate           # Concat along axis
│   ├── Stack                 # Stack on new axis
│   └── Split                 # Split along axis
│
├── SKIP_CONNECTIONS
│   ├── Residual              # Add skip connection
│   ├── DenseConnection       # Concat skip (DenseNet)
│   └── GatedSkip             # Learnable gate
│
├── EMBEDDING
│   ├── Embedding             # Lookup table
│   ├── EmbeddingBag          # Sum/mean of embeddings
│   └── PositionalEmbedding   # Learnable positions
│
├── LOSS_FUNCTIONS
│   ├── Regression
│   │   ├── MSELoss           # Mean Squared Error
│   │   ├── L1Loss            # Mean Absolute Error
│   │   ├── SmoothL1Loss      # Huber loss
│   │   └── HuberLoss         # Huber loss (configurable)
│   │
│   ├── Classification
│   │   ├── CrossEntropyLoss  # Cross entropy
│   │   ├── NLLLoss           # Negative log likelihood
│   │   ├── BCELoss           # Binary cross entropy
│   │   ├── BCEWithLogitsLoss # BCE + sigmoid
│   │   └── FocalLoss         # Focal loss (class imbalance)
│   │
│   └── Specialized
│       ├── CTCLoss           # Connectionist Temporal
│       ├── TripletMarginLoss # Triplet loss
│       ├── ContrastiveLoss   # Contrastive learning
│       └── CosineEmbeddingLoss
│
├── OPTIMIZERS
│   ├── SGD                   # Stochastic Gradient Descent
│   ├── Adam                  # Adaptive Moment
│   ├── AdamW                 # Adam + weight decay
│   ├── RMSprop               # Root Mean Square prop
│   ├── Adagrad               # Adaptive gradient
│   ├── Adadelta              # Extension of Adagrad
│   ├── RAdam                 # Rectified Adam
│   ├── LAMB                  # Large Batch optimization
│   └── LARS                  # Layer-wise Adaptive Rate
│
├── LR_SCHEDULERS
│   ├── StepLR                # Step decay
│   ├── MultiStepLR           # Multi-step decay
│   ├── ExponentialLR         # Exponential decay
│   ├── CosineAnnealingLR     # Cosine annealing
│   ├── OneCycleLR            # 1cycle policy
│   ├── ReduceLROnPlateau     # Adaptive reduction
│   └── WarmupLR              # Linear warmup
│
└── UTILITY
    ├── Identity              # Pass-through
    ├── Lambda                # Custom function
    ├── Custom                # User-defined
    ├── Subgraph              # Encapsulated subgraph
    └── Pattern               # Instantiated pattern
```

---

## 6. Intent-Based Generation System

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Parser Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "4 layer MLP with 128 hidden units, ReLU, and dropout 0.5"     │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Tokenizer & NLP                                        │   │
│  │    - Named Entity Recognition (layer types, numbers)      │   │
│  │    - Dependency Parsing (relationships)                   │   │
│  │    - Intent Classification                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. Semantic Analyzer                                      │   │
│  │    - Extract: architecture type (MLP)                     │   │
│  │    - Extract: num_layers = 4                              │   │
│  │    - Extract: hidden_units = 128                          │   │
│  │    - Extract: activation = ReLU                           │   │
│  │    - Extract: dropout = 0.5                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. Graph Generator                                        │   │
│  │    - Select pattern template: MLP                         │   │
│  │    - Instantiate with parameters                          │   │
│  │    - Add regularization (dropout)                         │   │
│  │    - Layout nodes visually                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. Output: ComputeGraph                                   │   │
│  │    Input(784) → Dense(128) → ReLU → Dropout(0.5) →       │   │
│  │    Dense(128) → ReLU → Dropout(0.5) →                    │   │
│  │    Dense(128) → ReLU → Dropout(0.5) →                    │   │
│  │    Dense(10) → Softmax → Output                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Intent Grammar

```
// Supported intent patterns
INTENT := ARCHITECTURE_DESCRIPTION | MODIFICATION | QUERY

ARCHITECTURE_DESCRIPTION :=
    [NUM] "layer" ARCH_TYPE [WITH_CLAUSE]*

ARCH_TYPE :=
    "MLP" | "CNN" | "RNN" | "LSTM" | "GRU" | "Transformer" |
    "ResNet" | "VGG" | "DenseNet" | "UNet" | "Autoencoder"

WITH_CLAUSE :=
    "with" PROPERTY_LIST

PROPERTY_LIST :=
    PROPERTY ("," PROPERTY)* | PROPERTY ("and" PROPERTY)*

PROPERTY :=
    NUM "hidden units" |
    NUM "filters" |
    NUM "heads" |
    ACTIVATION "activation" |
    "dropout" NUM |
    "batch norm" |
    "skip connections" |
    "layer" NUM "having" NUM "nodes"

MODIFICATION :=
    "add" NODE_TYPE ["to"|"after"|"before"] REFERENCE |
    "remove" REFERENCE |
    "connect" REFERENCE "to" REFERENCE |
    "replace" REFERENCE "with" NODE_TYPE

QUERY :=
    "show" ["all"] NODE_TYPE |
    "find" "path" "from" REFERENCE "to" REFERENCE |
    "count" NODE_TYPE
```

### 6.3 Example Intents

```python
# Simple MLP
"4 layer neural network with 256 hidden units and ReLU"
→ Input → Dense(256) → ReLU → Dense(256) → ReLU → Dense(256) → ReLU → Dense(out) → Output

# CNN for images
"CNN with 3 conv layers, 32-64-128 filters, maxpool, and 2 dense layers"
→ Input → Conv2D(32) → ReLU → MaxPool → Conv2D(64) → ReLU → MaxPool →
  Conv2D(128) → ReLU → MaxPool → Flatten → Dense(256) → ReLU → Dense(out) → Output

# Custom layer configuration
"4 layer network with layer 2 having 64 nodes, layer 3 having 128 nodes, connected with ReLU"
→ Input → Dense(64) → ReLU → Dense(128) → ReLU → Dense(out) → Output

# Transformer
"Transformer encoder with 6 layers, 8 heads, 512 dimensions"
→ Input → PositionalEncoding → TransformerEncoder(6, 8, 512) → Output

# LSTM for sequences
"Bidirectional LSTM with 2 layers, 256 hidden size, dropout 0.3"
→ Input → BiLSTM(256, 2, dropout=0.3) → Dense(out) → Output

# ResNet-style
"ResNet block with 64 filters and skip connection"
→ Input → Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm →
  [Residual: Add input] → ReLU → Output

# Modifications
"add dropout 0.5 after every dense layer"
"replace all ReLU with GELU"
"add batch normalization before every activation"
```

### 6.4 Implementation

```cpp
namespace cyxwiz::intent {

// Parsed intent representation
struct ParsedIntent {
    enum class Type { Create, Modify, Query };
    Type type;

    // For Create
    std::string architecture;
    int num_layers = 0;
    std::vector<int> layer_sizes;
    std::string activation = "ReLU";
    float dropout = 0.0f;
    bool batch_norm = false;
    bool skip_connections = false;
    std::map<std::string, PropertyValue> extra_params;

    // For Modify
    std::string modification_action;
    std::string target_reference;
    std::string new_value;

    // For Query
    std::string query_type;
    std::string query_target;
};

class IntentParser {
public:
    // Parse natural language to structured intent
    ParsedIntent Parse(const std::string& input);

    // Generate graph from parsed intent
    ComputeGraph GenerateGraph(const ParsedIntent& intent);

    // Modify existing graph based on intent
    void ModifyGraph(ComputeGraph& graph, const ParsedIntent& intent);

    // Query graph based on intent
    std::string QueryGraph(const ComputeGraph& graph, const ParsedIntent& intent);

private:
    // NLP components
    Tokenizer tokenizer_;
    EntityExtractor entity_extractor_;
    IntentClassifier classifier_;

    // Pattern templates
    PatternLibrary patterns_;
};

// Entity types recognized by the parser
enum class EntityType {
    Number,           // "4", "128", "0.5"
    LayerType,        // "dense", "conv", "lstm"
    ActivationType,   // "relu", "sigmoid", "gelu"
    ArchitectureType, // "mlp", "cnn", "transformer"
    Keyword,          // "layer", "with", "and"
    Property,         // "hidden units", "filters", "dropout"
    Reference         // "layer 2", "all dense layers"
};

struct Entity {
    EntityType type;
    std::string text;
    std::variant<int, float, std::string> value;
    int start_pos;
    int end_pos;
};

} // namespace cyxwiz::intent
```

---

## 7. Pattern Library & Auto-Generation

### 7.1 Pattern Definition Format

```cpp
namespace cyxwiz::patterns {

// Pattern definition structure
struct Pattern {
    std::string name;
    std::string category;
    std::string description;

    // Parameters that can be customized
    struct Parameter {
        std::string name;
        PropertyType type;
        PropertyValue default_value;
        std::optional<PropertyValue> min_value;
        std::optional<PropertyValue> max_value;
        std::string description;
    };
    std::vector<Parameter> parameters;

    // Graph template (nodes and edges with parameter references)
    std::string template_json;

    // Visual thumbnail
    std::vector<uint8_t> thumbnail_png;

    // Tags for search
    std::vector<std::string> tags;

    // Usage examples
    std::vector<std::string> examples;
};

// Pattern categories
enum class PatternCategory {
    // Basic architectures
    MLP,              // Multi-layer perceptron
    CNN,              // Convolutional networks
    RNN,              // Recurrent networks
    Transformer,      // Attention-based

    // Computer Vision
    ImageClassification,
    ObjectDetection,
    Segmentation,
    StyleTransfer,

    // NLP
    TextClassification,
    SequenceToSequence,
    LanguageModel,

    // Generative
    Autoencoder,
    VAE,
    GAN,
    Diffusion,

    // Specialized
    RecommendationSystem,
    TimeSeriesForecasting,
    Reinforcement,

    // Building blocks
    AttentionBlock,
    ResidualBlock,
    InceptionBlock,
    SEBlock,          // Squeeze-and-Excitation

    // Custom
    UserDefined
};

} // namespace cyxwiz::patterns
```

### 7.2 Built-in Pattern Library (50+ Patterns)

```yaml
# Pattern Library Index

## Basic Architectures
- MLP_Basic: "Simple feedforward network"
- MLP_Deep: "Deep MLP with batch norm"
- CNN_Basic: "Basic convolutional network"
- CNN_Deep: "Deep CNN with residuals"
- RNN_Basic: "Simple RNN"
- LSTM_Basic: "Single LSTM layer"
- LSTM_Stacked: "Stacked LSTM"
- BiLSTM: "Bidirectional LSTM"
- GRU_Basic: "Single GRU layer"

## Classic CNN Architectures
- LeNet5: "Classic LeNet-5 (1998)"
- AlexNet: "AlexNet architecture"
- VGG16: "VGG-16 architecture"
- VGG19: "VGG-19 architecture"
- ResNet18: "ResNet-18"
- ResNet34: "ResNet-34"
- ResNet50: "ResNet-50"
- ResNet101: "ResNet-101"
- DenseNet121: "DenseNet-121"
- InceptionV3: "Inception v3"
- MobileNetV2: "MobileNet v2"
- EfficientNetB0: "EfficientNet B0"

## Transformer Architectures
- TransformerEncoder: "Transformer encoder stack"
- TransformerDecoder: "Transformer decoder stack"
- TransformerFull: "Complete transformer"
- BERT_Base: "BERT base configuration"
- GPT2_Small: "GPT-2 small configuration"
- ViT_Base: "Vision Transformer base"

## Generative Models
- Autoencoder_Conv: "Convolutional autoencoder"
- VAE_Conv: "Variational autoencoder"
- GAN_Basic: "Basic GAN"
- DCGAN: "Deep Convolutional GAN"
- WGAN: "Wasserstein GAN"
- CycleGAN: "CycleGAN architecture"

## Segmentation
- UNet: "U-Net for segmentation"
- UNet_Plus: "U-Net++"
- FCN: "Fully Convolutional Network"
- DeepLabV3: "DeepLab v3"

## Object Detection
- YOLO_Backbone: "YOLO backbone"
- SSD_Backbone: "SSD backbone"
- FPN: "Feature Pyramid Network"

## Building Blocks
- ResidualBlock: "Basic residual block"
- BottleneckBlock: "Bottleneck residual"
- InceptionModule: "Inception module"
- SEBlock: "Squeeze-and-Excitation"
- AttentionBlock: "Self-attention block"
- TransformerLayer: "Single transformer layer"
- ConvBNReLU: "Conv + BatchNorm + ReLU"
- DepthwiseSeparable: "Depthwise separable conv"

## Specialized
- TimeSeriesLSTM: "LSTM for time series"
- Seq2Seq: "Sequence to sequence"
- AttentionSeq2Seq: "Seq2Seq with attention"
- NCF: "Neural Collaborative Filtering"
- WideDeep: "Wide & Deep network"
```

### 7.3 Pattern Template Example

```json
{
  "name": "ResNet_Block",
  "category": "BuildingBlocks",
  "description": "Standard residual block with skip connection",
  "parameters": [
    {"name": "in_channels", "type": "int", "default": 64},
    {"name": "out_channels", "type": "int", "default": 64},
    {"name": "stride", "type": "int", "default": 1},
    {"name": "downsample", "type": "bool", "default": false}
  ],
  "template": {
    "nodes": [
      {"id": "input", "type": "Input", "position": [0, 200]},
      {"id": "conv1", "type": "Conv2D", "position": [150, 100],
       "properties": {"in_channels": "$in_channels", "out_channels": "$out_channels",
                      "kernel_size": [3, 3], "stride": "$stride", "padding": [1, 1]}},
      {"id": "bn1", "type": "BatchNorm2D", "position": [300, 100],
       "properties": {"num_features": "$out_channels"}},
      {"id": "relu1", "type": "ReLU", "position": [450, 100]},
      {"id": "conv2", "type": "Conv2D", "position": [600, 100],
       "properties": {"in_channels": "$out_channels", "out_channels": "$out_channels",
                      "kernel_size": [3, 3], "stride": 1, "padding": [1, 1]}},
      {"id": "bn2", "type": "BatchNorm2D", "position": [750, 100],
       "properties": {"num_features": "$out_channels"}},
      {"id": "add", "type": "Add", "position": [900, 200]},
      {"id": "relu2", "type": "ReLU", "position": [1050, 200]},
      {"id": "output", "type": "Output", "position": [1200, 200]}
    ],
    "edges": [
      {"from": "input", "to": "conv1"},
      {"from": "conv1", "to": "bn1"},
      {"from": "bn1", "to": "relu1"},
      {"from": "relu1", "to": "conv2"},
      {"from": "conv2", "to": "bn2"},
      {"from": "bn2", "to": "add", "pin": 0},
      {"from": "input", "to": "add", "pin": 1, "type": "ResidualConnection"},
      {"from": "add", "to": "relu2"},
      {"from": "relu2", "to": "output"}
    ],
    "conditional": {
      "if": "$downsample",
      "then": {
        "add_nodes": [
          {"id": "downsample_conv", "type": "Conv2D", "position": [450, 300],
           "properties": {"in_channels": "$in_channels", "out_channels": "$out_channels",
                          "kernel_size": [1, 1], "stride": "$stride"}},
          {"id": "downsample_bn", "type": "BatchNorm2D", "position": [600, 300],
           "properties": {"num_features": "$out_channels"}}
        ],
        "modify_edges": [
          {"remove": {"from": "input", "to": "add", "pin": 1}},
          {"add": {"from": "input", "to": "downsample_conv"}},
          {"add": {"from": "downsample_conv", "to": "downsample_bn"}},
          {"add": {"from": "downsample_bn", "to": "add", "pin": 1}}
        ]
      }
    }
  },
  "tags": ["resnet", "residual", "skip", "cnn", "block"]
}
```

### 7.4 Pattern Engine API

```cpp
class PatternLibrary {
public:
    // Load built-in patterns
    void LoadBuiltinPatterns();

    // Load user patterns from directory
    void LoadUserPatterns(const std::filesystem::path& dir);

    // Search patterns
    std::vector<Pattern> Search(const std::string& query);
    std::vector<Pattern> GetByCategory(PatternCategory category);
    std::vector<Pattern> GetByTags(const std::vector<std::string>& tags);

    // Instantiate pattern
    ComputeGraph Instantiate(
        const std::string& pattern_name,
        const std::map<std::string, PropertyValue>& params = {}
    );

    // Save custom pattern
    void SavePattern(const Pattern& pattern);
    void SavePatternFromSelection(
        const ComputeGraph& graph,
        const std::vector<NodeId>& selected_nodes,
        const std::string& name,
        const std::vector<std::string>& param_names
    );

    // Get pattern info
    Pattern GetPattern(const std::string& name);
    std::vector<std::string> GetAllPatternNames();

private:
    std::map<std::string, Pattern> patterns_;
    std::filesystem::path user_patterns_dir_;
};
```

---

## 8. Multi-Connector System

### 8.1 Connection Types

```cpp
// Multi-connector edge semantics
namespace cyxwiz::graph {

// Connection cardinality
enum class Cardinality {
    One,          // Exactly one connection
    ZeroOrOne,    // Optional single connection
    OneOrMore,    // At least one
    ZeroOrMore    // Any number
};

// Pin configuration for multi-input/output
struct PinConfig {
    PinType type;
    Cardinality cardinality;
    bool ordered = false;  // Order matters for concat, etc.

    // For variadic pins
    int min_connections = 0;
    int max_connections = INT_MAX;
};

// Multi-connector nodes
struct ConcatenateNode : public Node {
    // Variadic input pin
    Pin inputs;  // cardinality = ZeroOrMore, ordered = true

    // Properties
    int axis = 1;  // Concatenation axis
};

struct AddNode : public Node {
    // Variadic input pin
    Pin inputs;  // cardinality = OneOrMore, ordered = false (commutative)
};

struct SplitNode : public Node {
    // Single input
    Pin input;

    // Variadic output (created dynamically)
    std::vector<Pin> outputs;

    // Properties
    int num_splits = 2;
    int axis = 1;
};

struct MultiHeadAttentionNode : public Node {
    // Named inputs
    Pin query;     // Required
    Pin key;       // Required
    Pin value;     // Required
    Pin attn_mask; // Optional
    Pin key_padding_mask; // Optional

    // Outputs
    Pin output;
    Pin attention_weights; // Optional output
};

} // namespace cyxwiz::graph
```

### 8.2 Visual Multi-Connector Rendering

```cpp
// Custom ImNodes rendering for multi-input pins
void NodeEditor::RenderMultiInputPin(const Pin& pin, int node_id) {
    // Get connected edges count
    int connection_count = GetConnectionCount(node_id, pin.id);

    // Render expandable pin area
    float pin_height = std::max(20.0f, connection_count * 15.0f);

    ImNodes::BeginInputAttribute(pin.id);

    // Draw pin with connection count indicator
    ImGui::Text("%s", pin.name.c_str());
    if (connection_count > 1) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%d)", connection_count);
    }

    // Visual indicator for "add more" on hover
    if (pin.config.cardinality == Cardinality::ZeroOrMore ||
        pin.config.cardinality == Cardinality::OneOrMore) {
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Drag to add more connections");
        }
    }

    ImNodes::EndInputAttribute();
}

// Handle multiple connections to same pin
void NodeEditor::HandleMultiConnection(int start_pin, int end_pin) {
    auto& end_node = GetNodeByPin(end_pin);
    auto& end_pin_config = end_node.GetPinConfig(end_pin);

    if (end_pin_config.cardinality == Cardinality::ZeroOrMore ||
        end_pin_config.cardinality == Cardinality::OneOrMore) {
        // Allow multiple connections
        AddEdge(start_pin, end_pin);
    } else if (end_pin_config.cardinality == Cardinality::One ||
               end_pin_config.cardinality == Cardinality::ZeroOrOne) {
        // Replace existing connection
        RemoveEdgesTo(end_pin);
        AddEdge(start_pin, end_pin);
    }
}
```

### 8.3 Skip Connection Patterns

```cpp
// Skip connection helpers
class SkipConnectionManager {
public:
    // Add residual (additive) skip connection
    EdgeId AddResidualSkip(
        ComputeGraph& graph,
        NodeId from_node,
        NodeId to_node,  // Must be an Add node or will insert one
        bool project_if_needed = true  // Add 1x1 conv if shapes mismatch
    );

    // Add dense (concatenative) skip connection
    EdgeId AddDenseSkip(
        ComputeGraph& graph,
        NodeId from_node,
        NodeId to_node  // Must be a Concat node or will insert one
    );

    // Create residual block around selection
    void WrapWithResidual(
        ComputeGraph& graph,
        const std::vector<NodeId>& nodes
    );

    // Detect skip connections in graph
    std::vector<std::pair<NodeId, NodeId>> FindSkipConnections(
        const ComputeGraph& graph
    );
};
```

---

## 9. Query Language (CyxQL)

### 9.1 Language Design (Cypher-Inspired)

```cypher
// CyxQL - CyxWiz Query Language
// Inspired by Neo4j's Cypher query language

// Find all Dense layers
MATCH (n:Dense)
RETURN n

// Find path from input to output
MATCH path = (input:Input)-[*]->(output:Output)
RETURN path

// Find Dense layers with more than 256 units
MATCH (n:Dense)
WHERE n.out_features > 256
RETURN n.name, n.out_features

// Find all nodes connected to a specific node
MATCH (n)-[:TensorFlow]->(target {name: "dense_1"})
RETURN n

// Count layers by type
MATCH (n)
RETURN labels(n), count(n) as layer_count
ORDER BY layer_count DESC

// Find residual connections
MATCH (a)-[:ResidualConnection]->(b)
RETURN a.name, b.name

// Find nodes between two points
MATCH path = (start {name: "input"})-[*1..5]->(end {name: "output"})
RETURN nodes(path)

// Create a new Dense layer
CREATE (n:Dense {name: "new_dense", in_features: 128, out_features: 64})

// Connect two nodes
MATCH (a {name: "relu_1"}), (b {name: "dense_2"})
CREATE (a)-[:TensorFlow]->(b)

// Delete a node and its connections
MATCH (n {name: "dropout_1"})
DETACH DELETE n

// Replace activation function
MATCH (old:ReLU)
CREATE (new:GELU)
WITH old, new
MATCH (pred)-[:TensorFlow]->(old)-[:TensorFlow]->(succ)
CREATE (pred)-[:TensorFlow]->(new)-[:TensorFlow]->(succ)
DETACH DELETE old

// Add dropout after every Dense layer
MATCH (d:Dense)-[:TensorFlow]->(next)
WHERE NOT next:Dropout
CREATE (dropout:Dropout {rate: 0.5})
CREATE (d)-[:TensorFlow]->(dropout)-[:TensorFlow]->(next)
DELETE (d)-[:TensorFlow]->(next)

// Find all paths of length 3
MATCH path = ()-[*3]->()
RETURN path

// Aggregate: get average hidden size
MATCH (n:Dense)
RETURN avg(n.out_features) as avg_hidden_size

// Pattern matching: find Conv-BN-ReLU sequences
MATCH (conv:Conv2D)-[:TensorFlow]->(bn:BatchNorm2D)-[:TensorFlow]->(relu:ReLU)
RETURN conv.name, bn.name, relu.name
```

### 9.2 CyxQL Parser Implementation

```cpp
namespace cyxwiz::query {

// AST Node types
enum class ASTNodeType {
    Match, Where, Create, Delete, Return, With, OrderBy, Limit,
    NodePattern, RelationshipPattern, PropertyAccess,
    Comparison, LogicalOp, Function, Literal, Identifier
};

struct ASTNode {
    ASTNodeType type;
    std::vector<std::shared_ptr<ASTNode>> children;
    std::variant<std::string, int64_t, double, bool> value;
};

// Query execution result
struct QueryResult {
    std::vector<std::string> columns;
    std::vector<std::vector<PropertyValue>> rows;

    // For graph modifications
    std::vector<NodeId> created_nodes;
    std::vector<EdgeId> created_edges;
    std::vector<NodeId> deleted_nodes;
    std::vector<EdgeId> deleted_edges;

    bool success;
    std::string error_message;
};

class CyxQLEngine {
public:
    // Parse and execute query
    QueryResult Execute(ComputeGraph& graph, const std::string& query);

    // Parse only (for syntax validation)
    std::shared_ptr<ASTNode> Parse(const std::string& query);

    // Execute parsed AST
    QueryResult Execute(ComputeGraph& graph, std::shared_ptr<ASTNode> ast);

    // Register custom functions
    void RegisterFunction(
        const std::string& name,
        std::function<PropertyValue(const std::vector<PropertyValue>&)> func
    );

private:
    Lexer lexer_;
    Parser parser_;
    Executor executor_;
    std::map<std::string, std::function<PropertyValue(const std::vector<PropertyValue>&)>> functions_;
};

// Built-in functions
// count(nodes), sum(values), avg(values), min(values), max(values)
// labels(node), properties(node), keys(node)
// nodes(path), relationships(path), length(path)
// startNode(rel), endNode(rel), type(rel)

} // namespace cyxwiz::query
```

### 9.3 Query Console Integration

```cpp
// Add CyxQL console to the editor
class QueryConsole {
public:
    void Render() {
        ImGui::Begin("CyxQL Console");

        // Query input
        static char query_buffer[4096] = "";
        ImGui::InputTextMultiline("##query", query_buffer, sizeof(query_buffer),
            ImVec2(-1, 100), ImGuiInputTextFlags_AllowTabInput);

        if (ImGui::Button("Execute") ||
            (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
            ExecuteQuery(query_buffer);
        }

        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            query_buffer[0] = '\0';
        }

        // Results table
        ImGui::Separator();
        ImGui::Text("Results:");

        if (last_result_.success) {
            if (ImGui::BeginTable("results", last_result_.columns.size())) {
                // Headers
                for (const auto& col : last_result_.columns) {
                    ImGui::TableSetupColumn(col.c_str());
                }
                ImGui::TableHeadersRow();

                // Rows
                for (const auto& row : last_result_.rows) {
                    ImGui::TableNextRow();
                    for (const auto& cell : row) {
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", ToString(cell).c_str());
                    }
                }
                ImGui::EndTable();
            }
        } else {
            ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s",
                last_result_.error_message.c_str());
        }

        ImGui::End();
    }

private:
    QueryResult last_result_;
    CyxQLEngine engine_;
};
```

---

## 10. Visual Editor Enhancements

### 10.1 Enhanced UI Features

```cpp
// New features for the node editor UI
class EnhancedNodeEditor {
public:
    // Zoom and Pan
    void SetZoom(float zoom);
    float GetZoom() const;
    void FitToScreen();
    void CenterOnSelection();

    // Selection
    void SelectAll();
    void SelectNone();
    void SelectByType(NodeLabel type);
    void SelectConnected(NodeId start);
    void InvertSelection();

    // Clipboard
    void Copy();
    void Cut();
    void Paste();
    void Duplicate();

    // Undo/Redo
    void Undo();
    void Redo();
    bool CanUndo() const;
    bool CanRedo() const;

    // Alignment
    void AlignHorizontal();
    void AlignVertical();
    void DistributeHorizontal();
    void DistributeVertical();
    void SnapToGrid();

    // Grouping
    NodeId GroupSelection(const std::string& name);
    void UngroupNode(NodeId group_id);
    void ExpandGroup(NodeId group_id);
    void CollapseGroup(NodeId group_id);

    // Search
    void OpenSearchDialog();
    std::vector<NodeId> Search(const std::string& query);
    void NavigateToNode(NodeId id);

    // Minimap
    void RenderMinimap();
    bool show_minimap_ = true;

    // Grid
    void SetGridSize(float size);
    void SetGridVisible(bool visible);
    void SetSnapToGrid(bool snap);

private:
    // Undo system
    std::vector<GraphSnapshot> undo_stack_;
    std::vector<GraphSnapshot> redo_stack_;

    // Clipboard
    std::string clipboard_json_;

    // View state
    float zoom_ = 1.0f;
    ImVec2 pan_offset_ = {0, 0};

    // Grid
    float grid_size_ = 20.0f;
    bool grid_visible_ = true;
    bool snap_to_grid_ = false;
};
```

### 10.2 Context Menu System

```cpp
void NodeEditor::RenderContextMenu() {
    if (ImGui::BeginPopup("CanvasContextMenu")) {
        // Quick add by category
        if (ImGui::BeginMenu("Add Layer")) {
            if (ImGui::BeginMenu("Core")) {
                if (ImGui::MenuItem("Dense")) AddNode(NodeLabel::Dense);
                if (ImGui::MenuItem("Conv2D")) AddNode(NodeLabel::Conv2D);
                if (ImGui::MenuItem("LSTM")) AddNode(NodeLabel::LSTM);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Activation")) {
                for (auto label : {NodeLabel::ReLU, NodeLabel::GELU,
                                   NodeLabel::Sigmoid, NodeLabel::Tanh}) {
                    if (ImGui::MenuItem(NodeLabelToString(label).c_str())) {
                        AddNode(label);
                    }
                }
                ImGui::EndMenu();
            }
            // ... more categories
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // Patterns submenu
        if (ImGui::BeginMenu("Insert Pattern")) {
            for (const auto& category : pattern_categories_) {
                if (ImGui::BeginMenu(category.name.c_str())) {
                    for (const auto& pattern : category.patterns) {
                        if (ImGui::MenuItem(pattern.name.c_str())) {
                            InsertPattern(pattern.name);
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        // Intent input
        if (ImGui::MenuItem("Generate from Description...")) {
            show_intent_dialog_ = true;
        }

        ImGui::Separator();

        // Edit operations
        if (ImGui::MenuItem("Paste", "Ctrl+V", false, !clipboard_json_.empty())) {
            Paste();
        }
        if (ImGui::MenuItem("Select All", "Ctrl+A")) {
            SelectAll();
        }

        ImGui::EndPopup();
    }

    // Node context menu
    if (ImGui::BeginPopup("NodeContextMenu")) {
        if (ImGui::MenuItem("Delete", "Delete")) {
            DeleteSelected();
        }
        if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
            Duplicate();
        }
        if (ImGui::MenuItem("Copy", "Ctrl+C")) {
            Copy();
        }
        if (ImGui::MenuItem("Cut", "Ctrl+X")) {
            Cut();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Insert After")) {
            // Common operations
            if (ImGui::MenuItem("ReLU")) InsertAfterSelection(NodeLabel::ReLU);
            if (ImGui::MenuItem("Dropout")) InsertAfterSelection(NodeLabel::Dropout);
            if (ImGui::MenuItem("BatchNorm")) InsertAfterSelection(NodeLabel::BatchNorm2D);
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Wrap with Residual")) {
            WrapWithResidual();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Group Selection", "Ctrl+G")) {
            GroupSelection("Group");
        }

        ImGui::EndPopup();
    }
}
```

### 10.3 Intent Dialog

```cpp
void NodeEditor::RenderIntentDialog() {
    if (!show_intent_dialog_) return;

    ImGui::OpenPopup("Generate Network");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(600, 400));

    if (ImGui::BeginPopupModal("Generate Network", &show_intent_dialog_)) {
        ImGui::Text("Describe your neural network:");
        ImGui::Separator();

        // Input area
        static char intent_buffer[2048] = "";
        ImGui::InputTextMultiline("##intent", intent_buffer, sizeof(intent_buffer),
            ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);

        // Examples
        ImGui::TextDisabled("Examples:");
        const char* examples[] = {
            "4 layer MLP with 256 hidden units and ReLU activation",
            "CNN with 3 conv layers (32, 64, 128 filters), maxpool, dropout 0.5",
            "LSTM with 2 layers, 512 hidden size, bidirectional",
            "Transformer encoder with 6 layers, 8 heads, 512 dim",
            "ResNet block with 64 filters and skip connection"
        };

        for (const char* example : examples) {
            if (ImGui::Selectable(example)) {
                strcpy(intent_buffer, example);
            }
        }

        ImGui::Separator();

        // Preview
        if (strlen(intent_buffer) > 0) {
            ImGui::Text("Preview:");
            auto preview = intent_parser_.Parse(intent_buffer);
            ImGui::TextWrapped("Architecture: %s", preview.architecture.c_str());
            ImGui::TextWrapped("Layers: %d", preview.num_layers);
            ImGui::TextWrapped("Activation: %s", preview.activation.c_str());
            // Show generated node list
        }

        ImGui::Separator();

        if (ImGui::Button("Generate", ImVec2(120, 0))) {
            auto intent = intent_parser_.Parse(intent_buffer);
            auto graph = intent_parser_.GenerateGraph(intent);
            MergeGraph(graph);
            show_intent_dialog_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_intent_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}
```

### 10.4 Keyboard Shortcuts

```cpp
// Comprehensive keyboard shortcuts
std::map<std::string, ShortcutAction> SHORTCUTS = {
    // File
    {"Ctrl+N", Action::NewGraph},
    {"Ctrl+O", Action::OpenGraph},
    {"Ctrl+S", Action::SaveGraph},
    {"Ctrl+Shift+S", Action::SaveGraphAs},

    // Edit
    {"Ctrl+Z", Action::Undo},
    {"Ctrl+Y", Action::Redo},
    {"Ctrl+Shift+Z", Action::Redo},
    {"Ctrl+C", Action::Copy},
    {"Ctrl+X", Action::Cut},
    {"Ctrl+V", Action::Paste},
    {"Ctrl+D", Action::Duplicate},
    {"Delete", Action::Delete},
    {"Ctrl+A", Action::SelectAll},
    {"Escape", Action::Deselect},

    // View
    {"Ctrl+0", Action::FitToScreen},
    {"Ctrl+=", Action::ZoomIn},
    {"Ctrl+-", Action::ZoomOut},
    {"Ctrl+1", Action::Zoom100},
    {"M", Action::ToggleMinimap},
    {"G", Action::ToggleGrid},

    // Node operations
    {"Ctrl+G", Action::GroupSelection},
    {"Ctrl+Shift+G", Action::UngroupSelection},
    {"R", Action::AddResidual},
    {"Space", Action::OpenQuickAdd},
    {"Tab", Action::CycleSelection},

    // Alignment
    {"Ctrl+Shift+H", Action::AlignHorizontal},
    {"Ctrl+Shift+V", Action::AlignVertical},

    // Search
    {"Ctrl+F", Action::OpenSearch},
    {"Ctrl+Shift+F", Action::OpenQuery},

    // Generation
    {"Ctrl+I", Action::OpenIntentDialog},
    {"Ctrl+P", Action::OpenPatternBrowser},

    // Code generation
    {"Ctrl+E", Action::GenerateCode},
    {"F5", Action::RunGraph}
};
```

---

## 11. Backend Integration

### 11.1 Graph Compiler

```cpp
namespace cyxwiz::compiler {

// Intermediate representation for execution
struct IRNode {
    uint64_t id;
    std::string op_type;
    std::vector<uint64_t> inputs;
    std::vector<uint64_t> outputs;
    std::map<std::string, PropertyValue> attributes;
};

struct IRGraph {
    std::vector<IRNode> nodes;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    std::map<uint64_t, TensorShape> shapes;
    std::map<uint64_t, DataType> dtypes;
};

class GraphCompiler {
public:
    // Compile ComputeGraph to IR
    IRGraph Compile(const ComputeGraph& graph);

    // Optimize IR
    IRGraph Optimize(const IRGraph& ir);

    // Lower to backend-specific representation
    std::unique_ptr<ExecutableGraph> Lower(
        const IRGraph& ir,
        const Device& target_device
    );

private:
    // Optimization passes
    void FuseConvBatchNorm(IRGraph& ir);
    void FoldConstants(IRGraph& ir);
    void EliminateDeadCode(IRGraph& ir);
    void OptimizeMemory(IRGraph& ir);
};

// Executable graph that can run on a device
class ExecutableGraph {
public:
    // Execute forward pass
    std::map<std::string, Tensor> Forward(
        const std::map<std::string, Tensor>& inputs
    );

    // Execute backward pass
    std::map<std::string, Tensor> Backward(
        const std::map<std::string, Tensor>& grad_outputs
    );

    // Get trainable parameters
    std::vector<Tensor*> GetParameters();

    // Save/Load state
    void SaveState(const std::string& path);
    void LoadState(const std::string& path);

private:
    Device* device_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::map<uint64_t, Tensor> tensor_cache_;
};

} // namespace cyxwiz::compiler
```

### 11.2 Training Pipeline

```cpp
namespace cyxwiz::training {

struct TrainingConfig {
    int epochs = 100;
    int batch_size = 32;
    float learning_rate = 0.001f;
    std::string optimizer = "Adam";
    std::string lr_scheduler = "none";

    // Data
    std::string train_data_path;
    std::string val_data_path;
    float validation_split = 0.2f;

    // Checkpointing
    std::string checkpoint_dir;
    int checkpoint_every = 10;  // epochs
    bool save_best_only = true;

    // Early stopping
    bool early_stopping = false;
    int patience = 10;
    float min_delta = 0.001f;

    // Distributed
    bool distributed = false;
    std::vector<std::string> node_addresses;
};

struct TrainingMetrics {
    float train_loss;
    float val_loss;
    float train_accuracy;
    float val_accuracy;
    std::map<std::string, float> custom_metrics;
    float epoch_time;
    int current_epoch;
    int total_epochs;
};

class TrainingRunner {
public:
    // Start training
    void Start(
        const ComputeGraph& model,
        const TrainingConfig& config
    );

    // Control
    void Pause();
    void Resume();
    void Stop();

    // Status
    bool IsRunning() const;
    TrainingMetrics GetCurrentMetrics() const;
    std::vector<TrainingMetrics> GetHistory() const;

    // Callbacks
    void OnEpochEnd(std::function<void(const TrainingMetrics&)> callback);
    void OnBatchEnd(std::function<void(int, float)> callback);
    void OnTrainingEnd(std::function<void(const std::vector<TrainingMetrics>&)> callback);

private:
    std::unique_ptr<ExecutableGraph> model_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<DataLoader> train_loader_;
    std::unique_ptr<DataLoader> val_loader_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::thread training_thread_;
};

} // namespace cyxwiz::training
```

### 11.3 Export Formats

```cpp
namespace cyxwiz::export_ {

// Export to ONNX
std::vector<uint8_t> ToONNX(
    const ComputeGraph& graph,
    const std::map<std::string, TensorShape>& input_shapes,
    int opset_version = 13
);

// Export to PyTorch
std::string ToPyTorch(
    const ComputeGraph& graph,
    bool include_weights = false,
    const std::string& class_name = "Model"
);

// Export to TensorFlow/Keras
std::string ToTensorFlow(
    const ComputeGraph& graph,
    bool functional_api = true
);

// Export to JAX/Flax
std::string ToJAX(
    const ComputeGraph& graph,
    const std::string& module_name = "Model"
);

// Export to SavedModel (TF)
void ToSavedModel(
    const ComputeGraph& graph,
    const ExecutableGraph& executable,
    const std::filesystem::path& output_dir
);

// Export checkpoints
void SaveCheckpoint(
    const ExecutableGraph& executable,
    const Optimizer& optimizer,
    int epoch,
    const std::filesystem::path& path
);

void LoadCheckpoint(
    ExecutableGraph& executable,
    Optimizer& optimizer,
    int& epoch,
    const std::filesystem::path& path
);

} // namespace cyxwiz::export_
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Current + Enhancements)

**Duration: 2-4 weeks**

```
Tasks:
├── [DONE] ImNodes integration
├── [DONE] Basic node types (16)
├── [DONE] Code generation (4 frameworks)
├── [DONE] JSON serialization
├── [ ] Refactor to new Graph Data Model
├── [ ] Implement all 100+ node types
├── [ ] Add undo/redo system
├── [ ] Add copy/paste support
├── [ ] Implement zoom/pan controls
├── [ ] Add minimap
└── [ ] Keyboard shortcuts
```

### Phase 2: Intent & Patterns

**Duration: 3-4 weeks**

```
Tasks:
├── [ ] Intent parser (NLP tokenizer)
├── [ ] Entity extraction (numbers, types)
├── [ ] Intent classification
├── [ ] Graph generator from intent
├── [ ] Pattern template format (JSON)
├── [ ] 50+ built-in patterns
├── [ ] Pattern browser UI
├── [ ] Pattern instantiation engine
└── [ ] Save selection as pattern
```

### Phase 3: Multi-Connector & Skip Connections

**Duration: 2-3 weeks**

```
Tasks:
├── [ ] Multi-input pin rendering
├── [ ] Variadic connections
├── [ ] Residual connection helpers
├── [ ] Dense connection (DenseNet)
├── [ ] Attention Q/K/V connections
├── [ ] Split/Merge nodes
├── [ ] Connection validation
└── [ ] Visual skip connection indicators
```

### Phase 4: Query Language (CyxQL)

**Duration: 3-4 weeks**

```
Tasks:
├── [ ] CyxQL lexer
├── [ ] CyxQL parser
├── [ ] AST representation
├── [ ] Query executor
├── [ ] MATCH clause
├── [ ] WHERE clause
├── [ ] CREATE/DELETE clauses
├── [ ] RETURN clause
├── [ ] Built-in functions
└── [ ] Query console UI
```

### Phase 5: Backend Integration

**Duration: 4-6 weeks**

```
Tasks:
├── [ ] Graph compiler (GDM → IR)
├── [ ] IR optimizer
├── [ ] ExecutableGraph implementation
├── [ ] Layer implementations in backend
│   ├── [ ] Conv2D/3D
│   ├── [ ] Pooling layers
│   ├── [ ] Normalization layers
│   ├── [ ] RNN/LSTM/GRU
│   ├── [ ] Attention layers
│   └── [ ] Transformer layers
├── [ ] Training runner
├── [ ] Checkpoint system
├── [ ] ONNX export
└── [ ] Distributed training integration
```

### Phase 6: Polish & Advanced Features

**Duration: 2-3 weeks**

```
Tasks:
├── [ ] Node search/filter
├── [ ] Alignment tools
├── [ ] Node grouping
├── [ ] Subgraph encapsulation
├── [ ] Custom node creation UI
├── [ ] Theme customization
├── [ ] Performance profiling view
├── [ ] Memory visualization
├── [ ] Documentation tooltips
└── [ ] Tutorial walkthrough
```

---

## Appendix A: File Structure

```
cyxwiz-engine/src/
├── gui/
│   ├── node_editor/
│   │   ├── node_editor.h              # Main editor class
│   │   ├── node_editor.cpp            # Implementation
│   │   ├── graph_data_model.h         # ComputeGraph, Node, Edge
│   │   ├── graph_data_model.cpp
│   │   ├── node_factory.h             # Node creation by type
│   │   ├── node_factory.cpp
│   │   ├── node_renderer.h            # Custom node rendering
│   │   ├── node_renderer.cpp
│   │   ├── connection_validator.h     # Connection rules
│   │   ├── connection_validator.cpp
│   │   ├── undo_manager.h             # Undo/redo
│   │   ├── undo_manager.cpp
│   │   ├── clipboard_manager.h        # Copy/paste
│   │   └── clipboard_manager.cpp
│   │
│   ├── intent/
│   │   ├── intent_parser.h            # NLP parsing
│   │   ├── intent_parser.cpp
│   │   ├── entity_extractor.h         # Named entity extraction
│   │   ├── entity_extractor.cpp
│   │   ├── graph_generator.h          # Intent to graph
│   │   └── graph_generator.cpp
│   │
│   ├── patterns/
│   │   ├── pattern_library.h          # Pattern management
│   │   ├── pattern_library.cpp
│   │   ├── pattern_templates/         # JSON templates
│   │   │   ├── mlp.json
│   │   │   ├── cnn.json
│   │   │   ├── resnet.json
│   │   │   └── ...
│   │   └── pattern_browser.h          # UI for browsing
│   │
│   └── query/
│       ├── cyxql_lexer.h              # Tokenization
│       ├── cyxql_parser.h             # Parsing
│       ├── cyxql_executor.h           # Execution
│       └── query_console.h            # UI

cyxwiz-backend/
├── include/cyxwiz/
│   └── layers/
│       ├── conv.h                     # Conv1D/2D/3D
│       ├── pooling.h                  # MaxPool, AvgPool
│       ├── normalization.h            # BatchNorm, LayerNorm
│       ├── recurrent.h                # RNN, LSTM, GRU
│       ├── attention.h                # Attention, MHA
│       └── transformer.h              # Transformer layers
│
└── src/
    └── layers/
        ├── conv.cpp
        ├── pooling.cpp
        ├── normalization.cpp
        ├── recurrent.cpp
        ├── attention.cpp
        └── transformer.cpp
```

---

## Appendix B: Dependencies

```cmake
# New dependencies for node editor enhancements

# Intent parsing (optional, can use regex fallback)
find_package(ICU COMPONENTS uc data)  # Unicode support

# Pattern templates
find_package(nlohmann_json REQUIRED)  # Already have this

# ONNX export
find_package(ONNX)
find_package(Protobuf REQUIRED)  # Already have via gRPC

# Optional: ML framework integration for import
find_package(Python3 COMPONENTS Interpreter Development)
# Can use Python to import PyTorch/TF models
```

---

## Appendix C: Performance Considerations

### Graph Operations

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Add Node | O(1) | Hash map insert |
| Remove Node | O(E) | Must remove edges |
| Add Edge | O(1) | With validation: O(1) amortized |
| Find Node by ID | O(1) | Hash map lookup |
| Find Nodes by Label | O(n) | Use index for O(k) where k = matches |
| Topological Sort | O(V + E) | Kahn's algorithm |
| Cycle Detection | O(V + E) | DFS |
| Path Finding | O(V + E) | BFS |
| CyxQL Query | O(V * E) worst | Depends on query complexity |

### Memory Layout

```cpp
// Optimize for cache locality
struct alignas(64) Node {
    // Frequently accessed together
    NodeId id;
    NodeLabel label;
    uint8_t num_inputs;
    uint8_t num_outputs;
    ImVec2 position;

    // Less frequently accessed
    std::string name;
    std::map<std::string, PropertyValue> properties;
};
```

### Rendering

- Use frustum culling for nodes outside viewport
- Batch ImNodes calls where possible
- Cache node colors and text
- Use LOD for minimap (simplified edges at distance)

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: CyxWiz Team*
