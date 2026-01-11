# Transformation Tools

Data transformation and preprocessing tools for preparing datasets for machine learning.

## Overview

The Transformation tools provide:
- **Scaling & Normalization** - StandardScaler, MinMaxScaler, RobustScaler
- **Encoding** - OneHot, Label, Target encoding
- **Feature Engineering** - Polynomial features, binning
- **Missing Data** - Imputation strategies
- **Image Transforms** - Augmentation, resizing, normalization

## Tools Reference

### Scaling Tools

#### StandardScaler Panel

```
+------------------------------------------------------------------+
|  Standard Scaler                                           [x]    |
+------------------------------------------------------------------+
|  Input Column(s): [x] price [x] area [x] rooms [ ] category       |
|                                                                   |
|  PARAMETERS                                                       |
|  [x] With Mean (center data)                                      |
|  [x] With Std (scale to unit variance)                            |
|                                                                   |
|  [ Fit ]  [ Transform ]  [ Fit & Transform ]                      |
|                                                                   |
|  FITTED PARAMETERS                                                |
|  +-----------------------------------------------------------+   |
|  | Column  | Mean        | Std Dev     | Fitted              |   |
|  +-----------------------------------------------------------+   |
|  | price   | 45,234.56   | 12,456.78   | Yes                 |   |
|  | area    | 1,234.56    | 345.67      | Yes                 |   |
|  | rooms   | 3.45        | 1.23        | Yes                 |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  PREVIEW (First 5 rows)                                           |
|  +-----------------------------------------------------------+   |
|  | Row | price (orig) | price (scaled) | area (scaled)       |   |
|  +-----------------------------------------------------------+   |
|  | 0   | 55,000       | 0.784          | 1.234               |   |
|  | 1   | 42,000       | -0.260         | -0.567              |   |
|  | 2   | 38,500       | -0.541         | -0.890              |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Save Scaler]  [Apply to New Data]  [Inverse Transform]          |
+------------------------------------------------------------------+
```

### Scaling Methods Comparison

| Scaler | Formula | Use Case | Outlier Sensitive |
|--------|---------|----------|-------------------|
| **Standard** | (x - mean) / std | Most algorithms | Yes |
| **MinMax** | (x - min) / (max - min) | Neural networks | Yes |
| **MaxAbs** | x / max(abs(x)) | Sparse data | Yes |
| **Robust** | (x - median) / IQR | Outlier presence | No |
| **Normalizer** | x / norm(x) | Text/sparse | No |

### Encoding Tools

#### One-Hot Encoding Panel

```
+------------------------------------------------------------------+
|  One-Hot Encoding                                          [x]    |
+------------------------------------------------------------------+
|  Categorical Columns: [x] color [x] size [x] brand [ ] id         |
|                                                                   |
|  OPTIONS                                                          |
|  [x] Drop first category (avoid multicollinearity)                |
|  [x] Handle unknown categories (ignore)                           |
|  Max categories per column: [20    ]                              |
|                                                                   |
|  [ Fit & Transform ]                                              |
|                                                                   |
|  ENCODING MAPPING                                                 |
|  +-----------------------------------------------------------+   |
|  | Column | Categories | New Columns                         |   |
|  +-----------------------------------------------------------+   |
|  | color  | 5          | color_blue, color_green, color_red, |   |
|  |        |            | color_yellow                        |   |
|  | size   | 3          | size_M, size_L                      |   |
|  | brand  | 8          | brand_A, brand_B, brand_C, ...      |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Original columns: 3                                              |
|  New columns: 14 (after dropping first)                           |
|                                                                   |
|  PREVIEW                                                          |
|  +-----------------------------------------------------------+   |
|  | color | color_blue | color_green | color_red | ...        |   |
|  +-----------------------------------------------------------+   |
|  | red   | 0          | 0           | 1         | ...        |   |
|  | blue  | 1          | 0           | 0         | ...        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Save Encoder]  [Transform New Data]                             |
+------------------------------------------------------------------+
```

#### Target Encoding Panel

```
+------------------------------------------------------------------+
|  Target Encoding                                           [x]    |
+------------------------------------------------------------------+
|  Categorical Column: [category       v]                           |
|  Target Column: [price             v]                             |
|                                                                   |
|  SMOOTHING                                                        |
|  Method: (o) Mean  ( ) Leave-One-Out  ( ) Weight of Evidence      |
|  Smoothing Factor: [1.0   ]                                       |
|  Min Samples: [5     ]                                            |
|                                                                   |
|  [ Fit & Transform ]                                              |
|                                                                   |
|  ENCODING VALUES                                                  |
|  +-----------------------------------------------------------+   |
|  | Category    | Count  | Target Mean | Encoded Value        |   |
|  +-----------------------------------------------------------+   |
|  | Electronics | 1,234  | 456.78      | 452.34               |   |
|  | Clothing    | 2,345  | 234.56      | 235.12               |   |
|  | Home        | 1,567  | 345.67      | 344.89               |   |
|  | Sports      | 891    | 289.12      | 290.45               |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Global Mean: 331.53 (used for unknown categories)                |
|                                                                   |
|  [Save Encoder]  [Transform]                                      |
+------------------------------------------------------------------+
```

### Missing Data Handling

#### Imputer Panel

```
+------------------------------------------------------------------+
|  Missing Value Imputer                                     [x]    |
+------------------------------------------------------------------+
|  DATA SUMMARY                                                     |
|  +-----------------------------------------------------------+   |
|  | Column     | Type     | Missing | % Missing | Strategy    |   |
|  +-----------------------------------------------------------+   |
|  | price      | Numeric  | 45      | 1.5%      | Mean        |   |
|  | area       | Numeric  | 23      | 0.8%      | Median      |   |
|  | category   | Categor. | 12      | 0.4%      | Mode        |   |
|  | rooms      | Numeric  | 0       | 0.0%      | -           |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  IMPUTATION STRATEGY                                              |
|  Numeric columns:                                                 |
|  ( ) Mean  (o) Median  ( ) Constant  ( ) KNN  ( ) Iterative      |
|                                                                   |
|  Categorical columns:                                             |
|  (o) Most Frequent  ( ) Constant                                  |
|                                                                   |
|  Constant value: [        ]                                       |
|  KNN neighbors: [5     ]                                          |
|                                                                   |
|  [ Preview Imputation ]  [ Apply ]                                |
|                                                                   |
|  IMPUTED VALUES PREVIEW                                           |
|  +-----------------------------------------------------------+   |
|  | Row  | Column   | Original | Imputed Value               |   |
|  +-----------------------------------------------------------+   |
|  | 45   | price    | NaN      | 45,234.56 (median)          |   |
|  | 123  | category | NaN      | Electronics (mode)          |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Save Imputer]  [Apply to New Data]                              |
+------------------------------------------------------------------+
```

### Image Transforms

#### Image Augmentation Panel

```
+------------------------------------------------------------------+
|  Image Augmentation                                        [x]    |
+------------------------------------------------------------------+
|  GEOMETRIC TRANSFORMS                                             |
|  +-----------------------------------------------------------+   |
|  | Transform        | Enabled | Parameters                   |   |
|  +-----------------------------------------------------------+   |
|  | Random Crop      | [x]     | Size: 224x224               |   |
|  | Random Flip      | [x]     | Horizontal: Yes, Vertical: No|   |
|  | Random Rotation  | [x]     | Max degrees: 15             |   |
|  | Random Scale     | [ ]     | Range: 0.8-1.2              |   |
|  | Random Shear     | [ ]     | Max: 10 degrees             |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  COLOR TRANSFORMS                                                 |
|  +-----------------------------------------------------------+   |
|  | Transform         | Enabled | Parameters                  |   |
|  +-----------------------------------------------------------+   |
|  | Color Jitter      | [x]     | Brightness: 0.2, Contrast: 0.2|  |
|  | Random Grayscale  | [ ]     | Probability: 0.1            |   |
|  | Gaussian Blur     | [x]     | Kernel: 3, Sigma: 0.1-2.0   |   |
|  | Random Erasing    | [ ]     | Probability: 0.5            |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  NORMALIZATION                                                    |
|  Preset: (o) ImageNet  ( ) CIFAR-10  ( ) Custom                   |
|  Mean: [0.485, 0.456, 0.406]                                      |
|  Std:  [0.229, 0.224, 0.225]                                      |
|                                                                   |
|  PREVIEW                                                          |
|  +-------+-------+-------+-------+                                |
|  | Orig  | Aug 1 | Aug 2 | Aug 3 |                                |
|  +-------+-------+-------+-------+                                |
|  | [img] | [img] | [img] | [img] |                                |
|  +-------+-------+-------+-------+                                |
|                                                                   |
|  [Save Transform Pipeline]  [Apply to Dataset]                    |
+------------------------------------------------------------------+
```

### Transform Presets

CyxWiz includes pre-configured transform pipelines:

| Preset | Description | Transforms |
|--------|-------------|------------|
| **ImageNet** | Standard image classification | Resize, CenterCrop, Normalize |
| **CIFAR-10** | Small image classification | RandomCrop, RandomFlip, Normalize |
| **Medical** | Medical imaging | Intensity normalization, augmentation |
| **Satellite** | Remote sensing | Multi-spectral normalization |
| **Self-Supervised** | Contrastive learning | Strong augmentation pair |

## Scripting Functions

### Scaling

```python
from cyxwiz.transforms import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transform new data
X_new_scaled = scaler.transform(X_new)

# Inverse transform
X_original = scaler.inverse_transform(X_scaled)
```

### Encoding

```python
from cyxwiz.transforms import OneHotEncoder, LabelEncoder, TargetEncoder

# One-hot encoding
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['category', 'color']])

# Label encoding
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Target encoding
target_enc = TargetEncoder(smoothing=1.0)
X_encoded = target_enc.fit_transform(X['category'], y)
```

### Imputation

```python
from cyxwiz.transforms import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)
```

### Image Transforms

```python
from cyxwiz.transforms.image import (
    Compose, Resize, RandomCrop, RandomHorizontalFlip,
    ColorJitter, Normalize, ToTensor
)

# Define transform pipeline
transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply to image
transformed_image = transform(image)
```

### Pipeline

```python
from cyxwiz.transforms import Pipeline, ColumnTransformer

# Create pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

# Column transformer
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

# Fit and transform
X_processed = preprocessor.fit_transform(X)
```

## Integration with Node Editor

### Transform Nodes

| Node | Type | Parameters |
|------|------|------------|
| **StandardScaler** | Scaling | with_mean, with_std |
| **MinMaxScaler** | Scaling | feature_range |
| **OneHotEncoder** | Encoding | drop, max_categories |
| **Imputer** | Missing | strategy, fill_value |
| **ImageAugment** | Image | transforms list |

### Example Pipeline

```
[Raw Data] -> [Imputer] -> [Scaler] -> [Encoder] -> [Model]
                                          |
                                    [OneHot for categoricals]
```

## Best Practices

### Scaling Guidelines

1. **Always scale** before distance-based algorithms (KNN, SVM, Neural Networks)
2. **Fit on training data only** - transform test data with same parameters
3. **Use RobustScaler** when outliers are present
4. **MinMaxScaler** for neural networks (bounded activations)

### Encoding Guidelines

1. **One-Hot** for low cardinality categoricals (< 20 categories)
2. **Target Encoding** for high cardinality (> 20 categories)
3. **Drop first** category to avoid multicollinearity in linear models
4. **Handle unknowns** for production robustness

### Image Transform Guidelines

1. **Training**: Use augmentation for regularization
2. **Validation/Test**: Use deterministic transforms only
3. **Normalization**: Match pre-trained model statistics
4. **Strong augmentation**: For limited data or self-supervised learning

---

**Next**: [Linear Algebra Tools](../linear-algebra/index.md) | [Data Science Tools](../data-science/index.md)
