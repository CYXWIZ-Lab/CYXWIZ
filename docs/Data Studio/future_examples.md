# CyxWiz Studio - Future Workflow Examples

This document contains example workflows showing how tasks will be accomplished in CyxWiz Studio after the node-based upgrade is complete.

---

## Example 1: Adult Dataset Income Prediction

**Task:** Predict whether income >$50K based on demographic attributes (age, education, occupation, etc.)

**Dataset:** UCI Adult Census (32,561 rows, 15 columns)

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CYXWIZ STUDIO CANVAS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ CSV       │───>│ Missing   │───>│ Column    │───>│ One-Hot   │          │
│  │ Reader    │    │ Value     │    │ Filter    │    │ Encoder   │          │
│  │           │    │           │    │           │    │           │          │
│  │ adult.csv │    │ mode/mean │    │ drop fnl  │    │ workclass │          │
│  └───────────┘    └───────────┘    └───────────┘    │ education │          │
│                                                      │ marital   │          │
│                                                      └─────┬─────┘          │
│                                                            │                 │
│                                                            ▼                 │
│                                    ┌───────────┐    ┌───────────┐          │
│                                    │ Normalizer│<───│ Category  │          │
│                                    │           │    │ to Number │          │
│                                    │ age,hours │    │           │          │
│                                    │ [0-1]     │    │ income    │          │
│                                    └─────┬─────┘    └───────────┘          │
│                                          │                                   │
│                         ┌────────────────┴────────────────┐                 │
│                         │                                 │                 │
│                         ▼                                 ▼                 │
│                   ┌───────────┐                     ┌───────────┐          │
│                   │Partitioner│                     │Partitioner│          │
│                   │ 80% Train │                     │ 80% Train │          │
│                   └─────┬─────┘                     └─────┬─────┘          │
│                         │                                 │                 │
│         ┌───────────────┤                 ┌───────────────┤                 │
│         │               │                 │               │                 │
│         ▼               ▼                 ▼               ▼                 │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│   │  Random   │   │  Random   │   │ Decision  │   │ Decision  │           │
│   │  Forest   │   │  Forest   │   │   Tree    │   │   Tree    │           │
│   │  Learner  │   │ Predictor │   │  Learner  │   │ Predictor │           │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘           │
│         │               │               │               │                   │
│         └───────┬───────┘               └───────┬───────┘                   │
│                 │                               │                           │
│                 ▼                               ▼                           │
│           ┌───────────┐                   ┌───────────┐                    │
│           │  Scorer   │                   │  Scorer   │                    │
│           │ Acc: 86%  │                   │ Acc: 82%  │                    │
│           └─────┬─────┘                   └─────┬─────┘                    │
│                 │                               │                           │
│                 └───────────────┬───────────────┘                           │
│                                 ▼                                           │
│                           ┌───────────┐    ┌───────────┐                   │
│                           │ ROC Curve │    │ Confusion │                   │
│                           │ AUC: 0.91 │    │  Matrix   │                   │
│                           └───────────┘    └───────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Instructions

#### Step 1: Load Data

| Action | Node | Properties |
|--------|------|------------|
| Drag from **I/O** category | **CSV Reader** | `file_path: adult.csv`<br>`delimiter: ,`<br>`has_header: true` |

**Info Panel shows:**
> "Reads CSV files into Arrow table. 32,561 rows detected."

#### Step 2: Handle Missing Values

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Manipulation** | **Missing Value** | `strategy: mode` (categorical)<br>`numeric_strategy: mean`<br>`columns: all` |

**Handles "?" values in workclass, occupation, native-country**

#### Step 3: Remove Irrelevant Columns

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Manipulation** | **Column Filter** | `mode: exclude`<br>`columns: [fnlwgt, education-num]` |

#### Step 4: Encode Categorical Features

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Manipulation** | **One-Hot Encoder** | `columns: [workclass, education, marital-status, occupation, relationship, race, sex, native-country]`<br>`drop_first: true` |

#### Step 5: Encode Target Variable

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Manipulation** | **Category to Number** | `column: income`<br>`mapping: {<=50K: 0, >50K: 1}` |

#### Step 6: Normalize Numeric Features

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Manipulation** | **Normalizer** | `columns: [age, hours-per-week, capital-gain, capital-loss]`<br>`method: min-max`<br>`range: [0, 1]` |

#### Step 7: Split Train/Test

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Analytics** | **Partitioner** | `ratio: 0.8`<br>`stratify_by: income`<br>`random_seed: 42` |

**Outputs:** Train port (26,048 rows) + Test port (6,513 rows)

#### Step 8: Train Random Forest

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Analytics** | **Random Forest Learner** | `n_estimators: 100`<br>`max_depth: 10`<br>`target: income` |
| Drag from **Analytics** | **Random Forest Predictor** | Connect test data + model |

#### Step 9: Train Decision Tree (Compare)

| Action | Node | Properties |
|--------|------|------------|
| Drag from **Analytics** | **Decision Tree Learner** | `max_depth: 8`<br>`criterion: gini`<br>`target: income` |
| Drag from **Analytics** | **Decision Tree Predictor** | Connect test data + model |

#### Step 10: Evaluate Models

| Node | Properties | Expected Output |
|------|------------|-----------------|
| **Scorer** | `target: income`<br>`prediction: prediction` | Accuracy, Precision, Recall, F1 |
| **ROC Curve** | `positive_class: 1` | AUC = 0.91 |
| **Confusion Matrix** | `labels: [0, 1]` | TP, FP, TN, FN grid |

#### Step 11: Export Best Model

| Action | Node | Properties |
|--------|------|------------|
| Drag from **I/O** | **Model Writer** | `path: adult_rf_classifier.onnx`<br>`format: ONNX` |

### Execution Flow

```
User clicks [▶ Execute] in toolbar
         │
         ▼
┌─────────────────────────────────────────────────┐
│ PipelineExecutor (DuckDB/Arrow backend)         │
│ 1. CSV Reader      → 32,561 rows loaded         │
│ 2. Missing Value   → "?" replaced               │
│ 3. Column Filter   → 13 columns remain          │
│ 4. One-Hot Encoder → 104 columns (expanded)     │
│ 5. Category→Number → income: 0/1                │
│ 6. Normalizer      → Values in [0,1]            │
│ 7. Partitioner     → 80/20 split                │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ ML Executor                                     │
│ 8. RF Learner      → Model trained (2.3s)       │
│ 9. RF Predictor    → 6,513 predictions          │
│ 10. Scorer         → Accuracy: 86.2%            │
│ 11. ROC Curve      → AUC: 0.912                 │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Results displayed in:                           │
│ - Console: "Execution complete. Accuracy: 86.2%"│
│ - Viewport: ROC curve visualization             │
│ - Properties: Confusion matrix table            │
└─────────────────────────────────────────────────┘
```

### Nodes Used

| Category | Nodes |
|----------|-------|
| **I/O** | CSV Reader, Model Writer |
| **Manipulation** | Missing Value, Column Filter, One-Hot Encoder, Category to Number, Normalizer |
| **Analytics** | Partitioner, Random Forest Learner/Predictor, Decision Tree Learner/Predictor, Scorer, ROC Curve, Confusion Matrix |

---

## Example 2: Time Series Forecasting (Sales Prediction)

**Task:** Forecast next 30 days of sales based on historical data

**Dataset:** Daily sales data (2 years, 730 rows)

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ CSV       │───>│ Date/Time │───>│ Time      │───>│ Lag       │          │
│  │ Reader    │    │ Parser    │    │ Series    │    │ Features  │          │
│  │           │    │           │    │ Sort      │    │           │          │
│  │ sales.csv │    │ YYYY-MM-DD│    │ by date   │    │ lag 1,7,30│          │
│  └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘          │
│                                                            │                 │
│                                                            ▼                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ Rolling   │<───│ Seasonal  │<───│Stationarity│<───│ Missing   │          │
│  │ Features  │    │ Decompose │    │ Test (ADF)│    │ Value     │          │
│  │           │    │           │    │           │    │           │          │
│  │ mean,std  │    │ trend,    │    │ p < 0.05  │    │ forward   │          │
│  │ 7,30 days │    │ seasonal  │    │           │    │ fill      │          │
│  └─────┬─────┘    └───────────┘    └───────────┘    └───────────┘          │
│        │                                                                     │
│        ▼                                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                           │
│  │ Train/Test│───>│   LSTM    │───>│ Forecast  │                           │
│  │ Split     │    │   Model   │    │ 30 days   │                           │
│  │           │    │           │    │           │                           │
│  │ 80/20     │    │ 64 units  │    │ horizon   │                           │
│  │ temporal  │    │ 2 layers  │    │           │                           │
│  └───────────┘    └─────┬─────┘    └─────┬─────┘                           │
│                         │                │                                   │
│                         ▼                ▼                                   │
│                   ┌───────────┐    ┌───────────┐                           │
│                   │   MAE     │    │ Line Plot │                           │
│                   │   RMSE    │    │ Actual vs │                           │
│                   │   MAPE    │    │ Predicted │                           │
│                   └───────────┘    └───────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Nodes Used

| Category | Nodes |
|----------|-------|
| **I/O** | CSV Reader |
| **Manipulation** | Date/Time Parser, Sorter, Missing Value |
| **Time Series** | Lag Features, Rolling Features, Seasonal Decompose, Stationarity Test |
| **Analytics** | Train/Test Split (temporal) |
| **ML Layers** | LSTM (64 units, 2 layers) |
| **Training** | MSE Loss, Adam Optimizer |
| **Views** | Line Plot |
| **Analytics** | MAE, RMSE, MAPE Scorer |

---

## Example 3: Image Classification (CIFAR-10)

**Task:** Classify images into 10 categories (airplane, car, bird, cat, etc.)

**Dataset:** CIFAR-10 (60,000 32x32 color images)

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                           │
│  │ Dataset   │───>│   Data    │───>│ Train/Val │                           │
│  │ Loader    │    │Augmentation│   │  Split    │                           │
│  │           │    │           │    │           │                           │
│  │ CIFAR-10  │    │ flip,rot  │    │ 90/10     │                           │
│  └───────────┘    │ normalize │    └─────┬─────┘                           │
│                   └───────────┘          │                                   │
│                                          │                                   │
│                         ┌────────────────┴────────────────┐                 │
│                         │                                 │                 │
│                         ▼                                 ▼                 │
│                   ┌───────────┐                     ┌───────────┐          │
│                   │  Conv2D   │                     │ Validation│          │
│                   │ 32 filters│                     │   Data    │          │
│                   │ 3x3, ReLU │                     └─────┬─────┘          │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │ MaxPool2D │                           │                 │
│                   │ 2x2       │                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │  Conv2D   │                           │                 │
│                   │ 64 filters│                           │                 │
│                   │ 3x3, ReLU │                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │  Flatten  │                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │  Dense    │                           │                 │
│                   │ 128, ReLU │                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │  Dropout  │                           │                 │
│                   │ 0.5       │                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│                         ▼                                 │                 │
│                   ┌───────────┐                           │                 │
│                   │  Dense    │                           │                 │
│                   │ 10,Softmax│                           │                 │
│                   └─────┬─────┘                           │                 │
│                         │                                 │                 │
│         ┌───────────────┼───────────────┬─────────────────┤                 │
│         │               │               │                 │                 │
│         ▼               ▼               ▼                 ▼                 │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│   │CrossEntropy│  │   Adam    │   │ Training  │   │ Evaluate  │           │
│   │   Loss    │   │ Optimizer │   │  Loop     │   │ on Val    │           │
│   │           │   │ lr=0.001  │   │ 50 epochs │   │           │           │
│   └───────────┘   └───────────┘   └─────┬─────┘   └─────┬─────┘           │
│                                         │               │                   │
│                                         ▼               ▼                   │
│                                   ┌───────────┐   ┌───────────┐           │
│                                   │ Learning  │   │ Confusion │           │
│                                   │  Curves   │   │  Matrix   │           │
│                                   │ loss/acc  │   │ 10 classes│           │
│                                   └───────────┘   └───────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Nodes Used

| Category | Nodes |
|----------|-------|
| **I/O** | Dataset Loader (CIFAR-10) |
| **Data Pipeline** | Data Augmentation, Train/Val Split |
| **ML Layers** | Conv2D, MaxPool2D, Flatten, Dense, Dropout |
| **Activation** | ReLU, Softmax |
| **Training** | CrossEntropy Loss, Adam Optimizer |
| **Views** | Learning Curves, Confusion Matrix |

---

## Example 4: Text Sentiment Analysis

**Task:** Classify movie reviews as positive or negative

**Dataset:** IMDB Reviews (50,000 reviews)

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ CSV       │───>│ Text      │───>│ Tokenizer │───>│ Vocabulary│          │
│  │ Reader    │    │ Cleaner   │    │           │    │ Builder   │          │
│  │           │    │           │    │           │    │           │          │
│  │ imdb.csv  │    │ lowercase │    │ word-level│    │ max 10000 │          │
│  │           │    │ remove    │    │           │    │ words     │          │
│  │           │    │ punctuation│   │           │    │           │          │
│  └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘          │
│                                                            │                 │
│                                                            ▼                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ Train/Test│<───│  Padding  │<───│ Text to   │<───│ Sequence  │          │
│  │ Split     │    │           │    │ Sequence  │    │ Encoder   │          │
│  │           │    │           │    │           │    │           │          │
│  │ 80/20     │    │ maxlen=200│    │ word→int  │    │           │          │
│  └─────┬─────┘    └───────────┘    └───────────┘    └───────────┘          │
│        │                                                                     │
│        ▼                                                                     │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ Embedding │───>│   LSTM    │───>│  Dense    │───>│  Sigmoid  │          │
│  │           │    │           │    │           │    │           │          │
│  │ dim=128   │    │ 64 units  │    │ 1 unit    │    │ output    │          │
│  └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘          │
│                                                            │                 │
│                         ┌──────────────────────────────────┤                 │
│                         │                                  │                 │
│                         ▼                                  ▼                 │
│                   ┌───────────┐                      ┌───────────┐          │
│                   │  Binary   │                      │ ROC Curve │          │
│                   │CrossEntropy│                     │ AUC       │          │
│                   │   Loss    │                      │           │          │
│                   └───────────┘                      └───────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Nodes Used

| Category | Nodes |
|----------|-------|
| **I/O** | CSV Reader |
| **Text Processing** | Text Cleaner, Tokenizer, Vocabulary Builder, Text to Sequence, Padding |
| **Analytics** | Train/Test Split |
| **ML Layers** | Embedding, LSTM, Dense |
| **Activation** | Sigmoid |
| **Training** | Binary CrossEntropy Loss |
| **Views** | ROC Curve |

---

## Example 5: Clustering Customer Segments

**Task:** Segment customers based on purchasing behavior

**Dataset:** Customer transactions (10,000 customers, RFM features)

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │ CSV       │───>│ GroupBy   │───>│ Feature   │───>│Standardizer│         │
│  │ Reader    │    │           │    │ Engineer  │    │           │          │
│  │           │    │           │    │           │    │           │          │
│  │transactions│   │ customer  │    │ Recency   │    │ z-score   │          │
│  │ .csv      │    │ _id       │    │ Frequency │    │           │          │
│  │           │    │           │    │ Monetary  │    │           │          │
│  └───────────┘    └───────────┘    └───────────┘    └─────┬─────┘          │
│                                                            │                 │
│                         ┌──────────────────────────────────┤                 │
│                         │                                  │                 │
│                         ▼                                  ▼                 │
│                   ┌───────────┐                      ┌───────────┐          │
│                   │  K-Means  │                      │  Elbow    │          │
│                   │           │                      │  Method   │          │
│                   │ k=5       │                      │           │          │
│                   │           │                      │ k=2..10   │          │
│                   └─────┬─────┘                      └───────────┘          │
│                         │                                                    │
│         ┌───────────────┼───────────────┐                                   │
│         │               │               │                                   │
│         ▼               ▼               ▼                                   │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐                           │
│   │ Silhouette│   │  Scatter  │   │  Cluster  │                           │
│   │  Score    │   │  Plot     │   │  Profile  │                           │
│   │           │   │           │   │           │                           │
│   │ 0.68      │   │ PCA 2D    │   │ stats per │                           │
│   │           │   │           │   │ cluster   │                           │
│   └───────────┘   └───────────┘   └───────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Nodes Used

| Category | Nodes |
|----------|-------|
| **I/O** | CSV Reader |
| **Manipulation** | GroupBy, Feature Engineer |
| **Preprocessing** | Standardizer |
| **Analytics** | K-Means, Elbow Method, Silhouette Score |
| **Views** | Scatter Plot (PCA), Cluster Profile |

---

## Comparison: KNIME vs CyxWiz Studio

| Aspect | KNIME | CyxWiz Studio |
|--------|-------|---------------|
| **Data Pipeline** | Yellow nodes | Data nodes (DuckDB/Arrow) |
| **ML Models** | Green nodes | Analytics nodes (traditional ML) |
| **Deep Learning** | Purple nodes (Keras) | ML Layer nodes (native GPU) |
| **Execution** | Right-click → Execute | Toolbar [▶] button |
| **Results** | Popup views | Viewport + Console panels |
| **Export** | PMML, ONNX | ONNX, CyxModel, Safetensors, GGUF |
| **GPU Training** | External (Python) | Native (ArrayFire CUDA/OpenCL) |
| **Distributed** | KNIME Server (paid) | P2P Network (free) |

---

## Notes for Implementation

1. **Execution Modes**:
   - `DuckDBPipeline`: For data transformation nodes
   - `LocalTraining`: For ML/DL nodes
   - `CodeGeneration`: Export to PyTorch/TensorFlow

2. **The (+) Bridge Node**:
   - Connects data pipeline output to ML model input
   - Converts Arrow table → Tensor
   - Handles train/test split context

3. **Progress Visualization**:
   - Each node shows execution state (Idle, Running, Complete, Error)
   - Viewport shows live training curves
   - Console shows detailed logs

4. **Model Persistence**:
   - After training, model is cached in memory
   - Model Writer node exports to file
   - Checkpoint nodes for long training
