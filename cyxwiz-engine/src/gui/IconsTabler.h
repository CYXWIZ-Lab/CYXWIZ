// Tabler Icons - Icon Font Definitions
// https://tabler.io/icons
// MIT License
//
// Node Editor Icon Pack - Clean 2px outline style

#pragma once

#define FONT_ICON_FILE_NAME_TI "tabler-icons.ttf"

// Icon range for Tabler Icons (0xea00 - 0xf9ff covers most icons)
#define ICON_MIN_TI 0xea00
#define ICON_MAX_TI 0xf9ff

// UTF-8 encoding: For U+XXXX where XXXX is 0x0800-0xFFFF
// Byte1 = 0xE0 | ((cp >> 12) & 0x0F)
// Byte2 = 0x80 | ((cp >> 6) & 0x3F)
// Byte3 = 0x80 | (cp & 0x3F)

// ============================================
// Data Pipeline Nodes
// ============================================
#define ICON_TI_DATABASE        "\xee\xaa\x88"  // U+EA88 - Data sources/Dataset
#define ICON_TI_LOADER          "\xee\xac\x83"  // U+EB03 - DataLoader (loader-2)
#define ICON_TI_WAND            "\xef\x81\xb4"  // U+F074 - Augmentation (wand)
#define ICON_TI_SITEMAP         "\xef\x86\x9d"  // U+F19D - DataSplit (sitemap)
#define ICON_TI_CHART_BAR       "\xee\xa9\x9a"  // U+EA5A - Normalize/Stats
#define ICON_TI_GRID_DOTS       "\xee\xab\xaa"  // U+EAEA - OneHotEncode

// ============================================
// Core Neural Network Layers
// ============================================
#define ICON_TI_BRAIN           "\xef\x96\x9f"  // U+F59F - Dense/Neural layers
#define ICON_TI_LAYERS_SUBTRACT "\xee\xae\x93"  // U+EB93 - Flatten (layers-subtract)
#define ICON_TI_TABLE           "\xee\xaf\xb7"  // U+EBF7 - Convolutional
#define ICON_TI_ARROWS_MINIMIZE "\xee\xa8\xa0"  // U+EA20 - Pooling (compress)
#define ICON_TI_DICE            "\xee\xaa\xb1"  // U+EAB1 - Dropout
#define ICON_TI_WAVE_SINE       "\xee\xbf\x8c"  // U+EFC6 - Activation (wave-sine)

// ============================================
// Recurrent & Sequence Layers
// ============================================
#define ICON_TI_ROTATE          "\xee\xaf\x13"  // U+EB13 - RNN/LSTM/GRU (rotate)
#define ICON_TI_TYPOGRAPHY      "\xee\xbf\x95"  // U+EF95 - Embedding (typography)

// ============================================
// Attention & Transformer
// ============================================
#define ICON_TI_FOCUS           "\xee\xab\x87"  // U+EAC7 - Attention (focus-2)
#define ICON_TI_CPU             "\xee\xaa\x86"  // U+EA86 - Transformer (cpu)
#define ICON_TI_HASH            "\xee\xab\x8d"  // U+EACD - PositionalEncoding

// ============================================
// Activation Functions
// ============================================
#define ICON_TI_BOLT            "\xee\xa8\x8f"  // U+EA0F - ReLU/Activations
#define ICON_TI_ACTIVITY        "\xee\xa7\x8a"  // U+EA0A - Sigmoid (activity)
#define ICON_TI_WAVE_SQUARE     "\xee\xbf\x8d"  // U+EFC7 - Tanh (wave-square)
#define ICON_TI_CHART_PIE       "\xee\xa9\x9f"  // U+EA5F - Softmax

// ============================================
// Shape Operations
// ============================================
#define ICON_TI_ARROWS_MAXIMIZE "\xee\xa8\x9e"  // U+EA1E - Reshape/Expand
#define ICON_TI_ARROWS_SHUFFLE  "\xee\xa8\xa2"  // U+EA22 - Permute/Shuffle
#define ICON_TI_ARROWS_DIAGONAL_MINIMIZE "\xee\xa8\x9a"  // U+EA1A - Squeeze

// ============================================
// Merge Operations
// ============================================
#define ICON_TI_LINK            "\xee\xae\x82"  // U+EB82 - Concatenate
#define ICON_TI_PLUS            "\xee\xaf\x8b"  // U+EB0B - Add
#define ICON_TI_X               "\xee\xbf\x9c"  // U+EFDC - Multiply (x)
#define ICON_TI_CHART_LINE      "\xee\xa9\x9c"  // U+EA5C - Average/Line chart

// ============================================
// Output & Loss
// ============================================
#define ICON_TI_CIRCLE_CHECK    "\xee\xa9\x87"  // U+EA47 - Output (circle-check)
#define ICON_TI_CALCULATOR      "\xee\xa9\x86"  // U+EA46 - Loss functions
#define ICON_TI_SCALE           "\xee\xaf\x16"  // U+EB16 - CrossEntropy (scale)

// ============================================
// Optimizers & Schedulers
// ============================================
#define ICON_TI_ADJUSTMENTS     "\xee\xa8\x85"  // U+EA05 - Optimizer settings

// ============================================
// Utility Nodes
// ============================================
#define ICON_TI_CODE            "\xee\xa9\xb7"  // U+EA77 - Lambda/Code
#define ICON_TI_EQUAL           "\xee\xaa\xb9"  // U+EAB9 - Identity
#define ICON_TI_CIRCLE          "\xee\xa9\x8a"  // U+EA4A - Constant
#define ICON_TI_SETTINGS        "\xee\xaf\x1e"  // U+EB1E - Parameter/Gear
#define ICON_TI_PLUG            "\xee\xaf\x87"  // U+EB07 - Plugin (plug)

// ============================================
// Signal / Control
// ============================================
#define ICON_TI_ADJUSTMENTS_ALT "\xef\x83\x9e"  // U+F09E - Slider (adjustments-alt)
#define ICON_TI_ARROW_UP_RIGHT  "\xee\xa8\xb6"  // U+EA36 - Ramp signal

// ============================================
// Subgraph & Grouping
// ============================================
#define ICON_TI_RECTANGLE_GROUP "\xef\x82\x8a"  // U+F08A - Subgraph (square-plus)

// ============================================
// DNN Inference
// ============================================
#define ICON_TI_SCAN            "\xee\xaf\x1c"  // U+EB1C - Detection (scan)
#define ICON_TI_TAG             "\xee\xaf\x24"  // U+EB24 - Classification
#define ICON_TI_USER            "\xee\xbf\x9a"  // U+EF9A - Pose estimation
#define ICON_TI_FILTER          "\xee\xaa\xa3"  // U+EAA3 - Filter/NMS

// ============================================
// Text & Audio
// ============================================
#define ICON_TI_BOOK            "\xee\xa9\x8f"  // U+EA4F - Vocabulary
#define ICON_TI_ALIGN_LEFT      "\xee\xa8\x8b"  // U+EA0B - Padding
#define ICON_TI_MUSIC           "\xee\xab\xa0"  // U+EAE0 - Audio

// ============================================
// I/O Nodes
// ============================================
#define ICON_TI_FILE_IMPORT     "\xee\xaa\xbd"  // U+EABD - Data input
#define ICON_TI_FILE_EXPORT     "\xee\xaa\xbb"  // U+EABB - Data output

// ============================================
// File Format Icons
// ============================================
#define ICON_TI_FILE_SPREADSHEET "\xee\xab\x8a" // U+EACA - CSV/Excel (file-spreadsheet)
#define ICON_TI_FILE_DATABASE   "\xee\xaa\xb3"  // U+EAB3 - Database files
#define ICON_TI_FILE_CODE       "\xee\xaa\xaf"  // U+EAAF - JSON/Code files
#define ICON_TI_FILE_ANALYTICS  "\xee\xaa\xad"  // U+EAAD - Analytics files
#define ICON_TI_FILE_TEXT       "\xee\xab\x8c"  // U+EACC - Text files
#define ICON_TI_PHOTO           "\xee\xae\xa8"  // U+EBA8 - Image files
#define ICON_TI_API             "\xee\xa8\x89"  // U+EA09 - API/REST

// ============================================
// KNIME-Style Table Manipulation
// ============================================
#define ICON_TI_TABLE_EXPORT    "\xee\xaf\xb9"  // U+EBF9 - Table operations
#define ICON_TI_TABLE_IMPORT    "\xee\xaf\xba"  // U+EBFA - Import table
#define ICON_TI_COLUMNS         "\xee\xa9\xb1"  // U+EA71 - Column operations
#define ICON_TI_ROWS            "\xee\xaf\x11"  // U+EB11 - Row operations
#define ICON_TI_LAYOUT_ROWS     "\xee\xae\x80"  // U+EB80 - Row layout
#define ICON_TI_LAYOUT_COLUMNS  "\xee\xad\xbf"  // U+EB7F - Column layout
#define ICON_TI_CROP            "\xee\xaa\x83"  // U+EA83 - Crop (table cropper)
#define ICON_TI_ARROW_MERGE     "\xee\xa8\xaa"  // U+EA2A - Merge/Join
#define ICON_TI_TRANSFORM       "\xef\x82\x97"  // U+F097 - Transform
#define ICON_TI_SORT_ASCENDING  "\xee\xaf\x1a"  // U+EB1A - Sort ascending
#define ICON_TI_SORT_DESCENDING "\xee\xaf\x1b"  // U+EB1B - Sort descending
#define ICON_TI_COPY            "\xee\xaa\x85"  // U+EA85 - Duplicate/Copy
#define ICON_TI_REPLACE         "\xef\x82\x86"  // U+F086 - Replace/Rename
#define ICON_TI_ARROWS_SPLIT    "\xee\xa8\xa3"  // U+EA23 - Split
#define ICON_TI_ARROWS_JOIN     "\xee\xa8\x9f"  // U+EA1F - Join/Append
#define ICON_TI_CELL            "\xef\x96\x9e"  // U+F59E - Cell operations

// ============================================
// Data Transform Operations
// ============================================
#define ICON_TI_FILTER_EDIT     "\xef\x82\xa9"  // U+F0A9 - Filter rows
#define ICON_TI_SELECT          "\xef\x82\x94"  // U+F094 - Select columns
#define ICON_TI_STACK           "\xee\xaf\x1f"  // U+EB1F - Stack/Union
#define ICON_TI_HIERARCHY       "\xee\xab\x8e"  // U+EACE - Hierarchy/GroupBy
#define ICON_TI_SWITCH          "\xee\xaf\x23"  // U+EB23 - Pivot/Unpivot
#define ICON_TI_TEXT_WRAP       "\xee\xaf\x33"  // U+EB33 - String manipulation
#define ICON_TI_MATH_FUNCTION   "\xee\xab\x97"  // U+EAD7 - Math formula
#define ICON_TI_RULE            "\xee\xab\xa1"  // U+EAE1 - Rule engine
#define ICON_TI_TRASH           "\xee\xaf\x2f"  // U+EB2F - Remove/Delete

// ============================================
// Analytics & Statistics
// ============================================
#define ICON_TI_REPORT_ANALYTICS "\xef\x82\x89" // U+F089 - Statistics
#define ICON_TI_CHART_DOTS      "\xee\xa9\x9b"  // U+EA5B - Scatter plot
#define ICON_TI_CHART_HISTOGRAM "\xee\xa9\x9d"  // U+EA5D - Histogram
#define ICON_TI_CHART_RADAR     "\xee\xa9\xa0"  // U+EA60 - Correlation
#define ICON_TI_PERCENTAGE      "\xee\xae\xa5"  // U+EBA5 - Value counts
#define ICON_TI_GRID_3X3        "\xee\xab\xa9"  // U+EAE9 - Cross tabulation

// ============================================
// Machine Learning Algorithms
// ============================================
#define ICON_TI_BINARY_TREE     "\xee\xa9\x8d"  // U+EA4D - Decision tree
#define ICON_TI_PLANT           "\xee\xae\xa3"  // U+EBA3 - Random forest
#define ICON_TI_VECTOR          "\xee\xbf\x97"  // U+EF97 - SVM
#define ICON_TI_CIRCLES         "\xee\xa9\x8b"  // U+EA4B - K-Means clustering
#define ICON_TI_AFFILIATE       "\xee\xa8\x84"  // U+EA04 - DBSCAN
#define ICON_TI_TIMELINE        "\xee\xaf\x2a"  // U+EB2A - Hierarchical
#define ICON_TI_3D_CUBE_SPHERE  "\xee\xa8\x82"  // U+EA02 - PCA/UMAP/t-SNE
#define ICON_TI_CHART_CANDLE    "\xee\xa9\x98"  // U+EA58 - Gradient boosting
#define ICON_TI_TARGET          "\xee\xaf\x25"  // U+EB25 - KNN

// ============================================
// Model Evaluation
// ============================================
#define ICON_TI_CHART_AREA      "\xee\xa9\x97"  // U+EA57 - ROC/PR curves
#define ICON_TI_CHECKBOX        "\xee\xa9\x85"  // U+EA45 - Confusion matrix
#define ICON_TI_LIST_NUMBERS    "\xee\xae\x85"  // U+EB85 - Metrics
#define ICON_TI_TRENDING_UP     "\xee\xaf\x31"  // U+EB31 - Learning curves

// ============================================
// Data Preprocessing
// ============================================
#define ICON_TI_RULER           "\xee\xaf\x14"  // U+EB14 - Scalers
#define ICON_TI_ABC             "\xee\xa8\x81"  // U+EA01 - Label encoder
#define ICON_TI_CUT             "\xee\xaa\x87"  // U+EA87 - Train/test split
#define ICON_TI_BUG             "\xee\xa9\x94"  // U+EA54 - Outlier detection

// ============================================
// Default/Misc
// ============================================
#define ICON_TI_CIRCLE_NODES    "\xef\x94\xa7"  // U+F527 - Default node icon (topology-star-3)
#define ICON_TI_NETWORK         "\xef\x82\x9f"  // U+F09F - Network/Connections
