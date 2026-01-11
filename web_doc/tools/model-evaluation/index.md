# Model Evaluation Tools

Comprehensive tools for assessing machine learning model performance, diagnosing issues, and comparing models.

## Overview

The Model Evaluation tools provide:
- **Classification Metrics** - Accuracy, precision, recall, F1, ROC-AUC
- **Regression Metrics** - MSE, MAE, R-squared, MAPE
- **Cross-Validation** - K-fold, stratified, time series splits
- **Model Comparison** - Side-by-side performance analysis
- **Error Analysis** - Confusion matrices, residual plots

## Tools Reference

### Classification Metrics Panel

```
+------------------------------------------------------------------+
|  Classification Metrics                                    [x]    |
+------------------------------------------------------------------+
|  Predictions: [model_predictions v]                               |
|  True Labels: [y_test           v]                                |
|  Class Type: (o) Binary  ( ) Multiclass  ( ) Multilabel           |
|                                                                   |
|  [ Calculate Metrics ]                                            |
|                                                                   |
|  BASIC METRICS                                                    |
|  +----------------------------+-------------------------------+   |
|  | Accuracy                   | 0.9234 (92.34%)               |   |
|  | Balanced Accuracy          | 0.9156                        |   |
|  | Log Loss                   | 0.2345                        |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  PER-CLASS METRICS                                                |
|  +-----------------------------------------------------------+   |
|  | Class     | Precision | Recall  | F1-Score | Support      |   |
|  +-----------------------------------------------------------+   |
|  | Negative  | 0.94      | 0.92    | 0.93     | 1,234        |   |
|  | Positive  | 0.91      | 0.93    | 0.92     | 1,089        |   |
|  +-----------------------------------------------------------+   |
|  | Macro Avg | 0.925     | 0.925   | 0.925    | 2,323        |   |
|  | Weighted  | 0.923     | 0.923   | 0.923    | 2,323        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  CONFUSION MATRIX                                                 |
|                    Predicted                                      |
|                 Neg      Pos                                      |
|  Actual  Neg   1,135     99                                       |
|          Pos     76    1,013                                      |
|                                                                   |
|  [x] Show percentages  [x] Normalize by row                       |
|                                                                   |
|  [Export Report]  [Show ROC Curve]  [Show PR Curve]               |
+------------------------------------------------------------------+
```

### ROC Curve Analysis

```
+------------------------------------------------------------------+
|  ROC Curve Analysis                                        [x]    |
+------------------------------------------------------------------+
|  Predictions (probabilities): [model_probs v]                     |
|  True Labels: [y_test      v]                                     |
|                                                                   |
|  [ Plot ROC Curve ]                                               |
|                                                                   |
|  ROC CURVE                                                        |
|  True Positive Rate                                               |
|  1.0 |                     ....****                               |
|      |                .****                                       |
|      |            .***                                            |
|      |         .***                                               |
|      |      .***                                                  |
|      |    .**.                                                    |
|      |  .**.                                                      |
|      |.**                                                         |
|  0.0 |*............................................                |
|      0                                           1.0              |
|                  False Positive Rate                              |
|                                                                   |
|      --- Model (AUC = 0.9567)                                     |
|      ... Random (AUC = 0.5000)                                    |
|                                                                   |
|  THRESHOLD ANALYSIS                                               |
|  +-----------------------------------------------------------+   |
|  | Threshold | TPR    | FPR    | Precision | F1-Score        |   |
|  +-----------------------------------------------------------+   |
|  | 0.30      | 0.98   | 0.15   | 0.87      | 0.92            |   |
|  | 0.50      | 0.93   | 0.08   | 0.92      | 0.93 <- Best    |   |
|  | 0.70      | 0.85   | 0.04   | 0.95      | 0.90            |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Optimal Threshold: 0.52 (Youden's J)                             |
|                                                                   |
|  [Export ROC Data]  [Find Optimal Threshold]  [Compare Models]    |
+------------------------------------------------------------------+
```

### Regression Metrics Panel

```
+------------------------------------------------------------------+
|  Regression Metrics                                        [x]    |
+------------------------------------------------------------------+
|  Predictions: [model_predictions v]                               |
|  True Values: [y_test           v]                                |
|                                                                   |
|  [ Calculate Metrics ]                                            |
|                                                                   |
|  ERROR METRICS                                                    |
|  +----------------------------+-------------------------------+   |
|  | Mean Squared Error (MSE)   | 234.567                       |   |
|  | Root MSE (RMSE)            | 15.316                        |   |
|  | Mean Absolute Error (MAE)  | 12.456                        |   |
|  | Median Absolute Error      | 10.234                        |   |
|  | Max Error                  | 45.678                        |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  RELATIVE METRICS                                                 |
|  +----------------------------+-------------------------------+   |
|  | Mean Absolute % Error      | 8.34%                         |   |
|  | Symmetric MAPE             | 7.89%                         |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  VARIANCE EXPLAINED                                               |
|  +----------------------------+-------------------------------+   |
|  | R-squared (R2)             | 0.8745                        |   |
|  | Adjusted R2                | 0.8712                        |   |
|  | Explained Variance         | 0.8756                        |   |
|  +----------------------------+-------------------------------+   |
|                                                                   |
|  RESIDUAL ANALYSIS                                                |
|  [x] Residual Plot  [x] Q-Q Plot  [x] Histogram                   |
|                                                                   |
|  [Export Report]  [Show Plots]  [Outlier Analysis]                |
+------------------------------------------------------------------+
```

### Cross-Validation Panel

```
+------------------------------------------------------------------+
|  Cross-Validation                                          [x]    |
+------------------------------------------------------------------+
|  Model: [Sequential Model v]                                      |
|  Data: [training_data    v]                                       |
|                                                                   |
|  CV STRATEGY                                                      |
|  (o) K-Fold  ( ) Stratified K-Fold  ( ) Time Series               |
|  ( ) Leave-One-Out  ( ) Repeated K-Fold  ( ) Group K-Fold         |
|                                                                   |
|  Parameters:                                                      |
|  Number of Folds: [5     ]                                        |
|  Shuffle: [x]                                                     |
|  Random State: [42    ]                                           |
|                                                                   |
|  Scoring Metrics:                                                 |
|  [x] Accuracy  [x] F1  [x] Precision  [x] Recall  [ ] ROC-AUC     |
|                                                                   |
|  [ Run Cross-Validation ]                                         |
|                                                                   |
|  RESULTS                                                          |
|  +-----------------------------------------------------------+   |
|  | Fold | Accuracy | F1-Score | Precision | Recall | Time    |   |
|  +-----------------------------------------------------------+   |
|  | 1    | 0.9234   | 0.9156   | 0.9245    | 0.9068 | 12.3s   |   |
|  | 2    | 0.9312   | 0.9234   | 0.9312    | 0.9156 | 11.8s   |   |
|  | 3    | 0.9189   | 0.9112   | 0.9189    | 0.9036 | 12.1s   |   |
|  | 4    | 0.9267   | 0.9189   | 0.9267    | 0.9112 | 12.5s   |   |
|  | 5    | 0.9245   | 0.9167   | 0.9234    | 0.9101 | 11.9s   |   |
|  +-----------------------------------------------------------+   |
|  | Mean | 0.9249   | 0.9172   | 0.9249    | 0.9095 | 60.6s   |   |
|  | Std  | 0.0043   | 0.0043   | 0.0043    | 0.0043 | -       |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  LEARNING CURVE                                                   |
|  [Show Learning Curve]  [Show Validation Curve]                   |
|                                                                   |
|  [Export Results]  [Generate Report]                              |
+------------------------------------------------------------------+
```

### Model Comparison Panel

```
+------------------------------------------------------------------+
|  Model Comparison                                          [x]    |
+------------------------------------------------------------------+
|  Models to Compare:                                               |
|  [x] Sequential_v1    [x] Sequential_v2    [x] CNN_baseline       |
|  [ ] Random_Forest    [ ] XGBoost          [ ] Ensemble           |
|                                                                   |
|  Test Data: [test_dataset v]                                      |
|                                                                   |
|  [ Compare Models ]                                               |
|                                                                   |
|  PERFORMANCE COMPARISON                                           |
|  +-----------------------------------------------------------+   |
|  | Model          | Accuracy | F1     | AUC    | Params      |   |
|  +-----------------------------------------------------------+   |
|  | Sequential_v1  | 0.9234   | 0.9156 | 0.9567 | 125,456     |   |
|  | Sequential_v2  | 0.9312   | 0.9234 | 0.9623 | 234,567     |   |
|  | CNN_baseline   | 0.9456   | 0.9389 | 0.9712 | 1,234,567   |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  EFFICIENCY COMPARISON                                            |
|  +-----------------------------------------------------------+   |
|  | Model          | Train Time | Inference | Memory (MB)    |   |
|  +-----------------------------------------------------------+   |
|  | Sequential_v1  | 45.2s      | 2.3ms     | 12.5           |   |
|  | Sequential_v2  | 89.7s      | 3.1ms     | 23.4           |   |
|  | CNN_baseline   | 234.5s     | 5.6ms     | 156.7          |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  STATISTICAL SIGNIFICANCE (McNemar's Test)                        |
|  +-----------------------------------------------------------+   |
|  | Comparison              | p-value | Significant?          |   |
|  +-----------------------------------------------------------+   |
|  | v1 vs v2                | 0.0234  | Yes (p < 0.05)        |   |
|  | v1 vs CNN               | 0.0012  | Yes (p < 0.01)        |   |
|  | v2 vs CNN               | 0.0456  | Yes (p < 0.05)        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Export Comparison]  [Radar Chart]  [Box Plot]                   |
+------------------------------------------------------------------+
```

### Error Analysis Panel

```
+------------------------------------------------------------------+
|  Error Analysis                                            [x]    |
+------------------------------------------------------------------+
|  Model: [trained_model v]                                         |
|  Test Data: [test_set  v]                                         |
|                                                                   |
|  [ Analyze Errors ]                                               |
|                                                                   |
|  MISCLASSIFICATION SUMMARY                                        |
|  Total Samples: 2,323                                             |
|  Correct: 2,148 (92.47%)                                          |
|  Errors: 175 (7.53%)                                              |
|                                                                   |
|  ERROR BREAKDOWN BY CLASS                                         |
|  +-----------------------------------------------------------+   |
|  | True Class | Errors | Most Confused With | Error Rate     |   |
|  +-----------------------------------------------------------+   |
|  | Cat        | 45     | Dog (32), Bird (13)| 8.2%           |   |
|  | Dog        | 38     | Cat (25), Wolf (13)| 7.1%           |   |
|  | Bird       | 52     | Plane (31), Cat (21)| 9.8%          |   |
|  | Plane      | 40     | Bird (28), Car (12)| 6.5%           |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  HARDEST SAMPLES (Lowest Confidence)                              |
|  +-----------------------------------------------------------+   |
|  | Sample ID | True | Predicted | Confidence | View          |   |
|  +-----------------------------------------------------------+   |
|  | 1456      | Cat  | Dog       | 0.51       | [View]        |   |
|  | 2891      | Bird | Plane     | 0.52       | [View]        |   |
|  | 0923      | Dog  | Cat       | 0.53       | [View]        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  ERROR PATTERNS                                                   |
|  [x] Show image samples  [x] Feature importance for errors        |
|                                                                   |
|  [Export Error Analysis]  [Show All Errors]  [Retrain on Hard]    |
+------------------------------------------------------------------+
```

## Evaluation Metrics Reference

### Classification Metrics

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | Higher |
| **Precision** | TP/(TP+FP) | 0-1 | Higher |
| **Recall** | TP/(TP+FN) | 0-1 | Higher |
| **F1-Score** | 2*(P*R)/(P+R) | 0-1 | Higher |
| **ROC-AUC** | Area under ROC curve | 0-1 | Higher |
| **PR-AUC** | Area under PR curve | 0-1 | Higher |
| **Log Loss** | Cross-entropy loss | 0-infinity | Lower |
| **Matthews CC** | Balanced measure | -1 to 1 | Higher |

### Regression Metrics

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| **MSE** | Mean squared error | 0-infinity | Lower |
| **RMSE** | Root mean squared error | 0-infinity | Lower |
| **MAE** | Mean absolute error | 0-infinity | Lower |
| **MAPE** | Mean absolute percentage error | 0-100% | Lower |
| **R-squared** | Variance explained | 0-1 | Higher |

## Scripting Functions

### Classification Evaluation

```python
from cyxwiz.evaluation import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Basic metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC-AUC (requires probabilities)
auc = roc_auc_score(y_true, y_proba)

# Full report
report = classification_report(y_true, y_pred)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
```

### Regression Evaluation

```python
from cyxwiz.evaluation import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Cross-Validation

```python
from cyxwiz.evaluation import cross_val_score, KFold, StratifiedKFold

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Custom CV strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train and evaluate
```

### ROC Curve

```python
from cyxwiz.evaluation import roc_curve, auc
import pycyxwiz.plot as plt

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

## Integration with Node Editor

### Evaluation Nodes

| Node | Inputs | Outputs |
|------|--------|---------|
| **Accuracy** | Predictions, Labels | Accuracy score |
| **Classification Report** | Predictions, Labels | Report dict |
| **Confusion Matrix** | Predictions, Labels | Matrix tensor |
| **ROC Curve** | Probabilities, Labels | FPR, TPR, AUC |
| **Cross Validate** | Model, Data | Scores array |

### Example Pipeline

```
[Model Output] -> [Evaluation Node] -> [Metrics Display]
       |                 |
       v                 v
   [Test Data]    [Confusion Matrix] -> [Visualization]
```

---

**Next**: [Transformations Tools](../transformations/index.md) | [Statistics Tools](../statistics/index.md)
