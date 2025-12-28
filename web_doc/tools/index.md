# Tools Reference

CyxWiz Engine includes a comprehensive suite of 70+ data science and analysis tools, accessible via the **Tools** menu or the **Command Palette** (Ctrl+P).

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| [Model Analysis](model-analysis.md) | 3 | Architecture inspection |
| [Data Science](data-science/index.md) | 4 | Data exploration and quality |
| [Statistics](statistics/index.md) | 4 | Statistical analysis |
| [Clustering](clustering/index.md) | 5 | Unsupervised learning |
| [Model Evaluation](model-evaluation/index.md) | 5 | Performance metrics |
| [Transformations](transformations/index.md) | 5 | Data preprocessing |
| [Linear Algebra](linear-algebra/index.md) | 5 | Matrix operations |
| [Signal Processing](signal-processing/index.md) | 5 | Frequency analysis |
| [Optimization](optimization/index.md) | 6 | Mathematical optimization |
| [Time Series](time-series/index.md) | 5 | Temporal analysis |
| [Text Processing](text-processing/index.md) | 5 | NLP tools |
| [Utilities](utilities/index.md) | 6 | General purpose tools |

## Quick Access

### Command Palette (Recommended)

Press `Ctrl+P` and type the tool name:

```
> correlation    -> Correlation Matrix
> kmeans         -> K-Means Clustering
> fft            -> FFT Analysis
```

### Tools Menu

**Tools** > [Category] > [Tool Name]

## Tool Interface

All tools share a consistent interface:

```
+------------------------------------------------------------------+
|  Tool Name                                               [x] [-] |
+------------------------------------------------------------------+
|                                                                   |
|  [Input Section]                                                  |
|  +-----------------------+                                        |
|  | Data Source: [v]      |   Select dataset or array             |
|  | Column: [v]           |   Select variable                     |
|  +-----------------------+                                        |
|                                                                   |
|  [Parameters Section]                                             |
|  +-----------------------+                                        |
|  | Parameter 1: [___]    |                                        |
|  | Parameter 2: [___]    |                                        |
|  +-----------------------+                                        |
|                                                                   |
|  [Actions]                                                        |
|  [  Run  ]  [ Export ]  [ Clear ]                                |
|                                                                   |
|  [Results Section]                                                |
|  +---------------------------------------------------------+     |
|  | Results displayed here (tables, plots, statistics)       |     |
|  |                                                          |     |
|  +---------------------------------------------------------+     |
|                                                                   |
+------------------------------------------------------------------+
```

## Model Analysis Tools

### Model Summary
View architecture details:
- Layer names and types
- Parameter counts
- Output shapes
- Memory estimation

### Architecture Diagram
Visual representation:
- Layer connections
- Skip connections
- Parameter flow
- Export to image

### Learning Rate Finder
Find optimal LR:
- Exponential LR sweep
- Loss vs LR plot
- Suggested LR range

## Data Science Tools

### Data Profiler
Comprehensive overview:
- Data types
- Missing values
- Distributions
- Correlations
- Outliers

### Correlation Matrix
Feature relationships:
- Pearson/Spearman/Kendall
- Heatmap visualization
- Significance tests

### Missing Value Analysis
Handle missing data:
- Detect patterns
- Visualize missingness
- Imputation options

### Outlier Detection
Find anomalies:
- Z-score method
- IQR method
- Isolation Forest
- LOF

## Statistics Tools

### Descriptive Statistics
Summary metrics:
- Mean, median, mode
- Std dev, variance
- Skewness, kurtosis
- Quartiles

### Hypothesis Testing
Statistical tests:
- t-test (one/two sample)
- ANOVA
- Chi-square
- Mann-Whitney U

### Distribution Fitter
Fit distributions:
- Normal, log-normal
- Exponential, gamma
- Poisson, binomial
- Goodness of fit tests

### Regression Analysis
Fit regression models:
- Linear regression
- Polynomial regression
- Residual analysis
- R-squared, RMSE

## Clustering Tools

### K-Means Clustering
Classic clustering:
- Configurable K
- Elbow method
- Cluster visualization
- Centroids export

### DBSCAN
Density-based:
- Eps/MinPts params
- Noise detection
- Arbitrary shapes

### Hierarchical Clustering
Dendrogram-based:
- Linkage methods
- Distance metrics
- Cut-off threshold

### Gaussian Mixture Model
Probabilistic:
- Soft assignments
- BIC/AIC selection
- Covariance types

### Cluster Evaluation
Quality metrics:
- Silhouette score
- Davies-Bouldin
- Calinski-Harabasz

## Model Evaluation Tools

### Confusion Matrix
Classification analysis:
- True/false positives
- Precision, recall
- F1 score
- Per-class metrics

### ROC-AUC Curve
Binary classification:
- ROC curve plot
- AUC calculation
- Threshold selection

### Precision-Recall Curve
Imbalanced data:
- PR curve plot
- Average precision
- Threshold analysis

### Cross-Validation
Model validation:
- K-fold CV
- Stratified CV
- Leave-one-out
- Score aggregation

### Learning Curves
Training analysis:
- Train vs val curves
- Bias-variance diagnosis
- Sample efficiency

## Data Transformation Tools

### Normalization
Min-max scaling:
- Scale to [0,1]
- Custom range
- Feature-wise

### Standardization
Z-score:
- Zero mean
- Unit variance
- Robust option

### Log Transform
Handle skew:
- Natural log
- Log10
- Log1p

### Box-Cox Transform
Power transform:
- Optimal lambda
- Handle zeros
- Inverse transform

### Feature Scaling
Combined options:
- Multiple methods
- Column selection
- Preview results

## Linear Algebra Tools

### Matrix Calculator
Matrix operations:
- Addition, multiplication
- Inverse, transpose
- Determinant
- Rank

### Eigendecomposition
Eigen analysis:
- Eigenvalues
- Eigenvectors
- Visualization

### SVD
Singular values:
- U, S, V matrices
- Rank approximation
- Explained variance

### QR Decomposition
Factorization:
- Q, R matrices
- Numerical stability

### Cholesky Decomposition
Positive definite:
- Lower triangular
- Efficient solving

## Signal Processing Tools

### FFT Analysis
Frequency domain:
- Magnitude spectrum
- Phase spectrum
- Power spectral density

### Spectrogram
Time-frequency:
- STFT analysis
- Window selection
- Colormap visualization

### Filter Designer
Digital filters:
- Low/high/band pass
- FIR/IIR
- Frequency response

### Convolution
Signal filtering:
- 1D/2D convolution
- Custom kernels
- Padding modes

### Wavelet Transform
Multi-resolution:
- Wavelet families
- Decomposition levels
- Reconstruction

## Optimization Tools

### Gradient Descent Visualizer
Learning visualization:
- 2D contour plots
- Trajectory animation
- Optimizer comparison

### Convexity Analyzer
Function analysis:
- Convexity check
- Hessian analysis
- Saddle points

### Linear Programming
LP solver:
- Objective function
- Constraints
- Optimal solution

### Quadratic Programming
QP solver:
- Quadratic objective
- Linear constraints

### Differentiation
Numerical derivatives:
- First/second order
- Gradient computation

### Integration
Numerical integration:
- Definite integrals
- Multiple methods

## Time Series Tools

### Decomposition
Component analysis:
- Trend extraction
- Seasonal patterns
- Residuals

### ACF/PACF
Autocorrelation:
- ACF plot
- PACF plot
- Lag selection

### Stationarity Testing
Statistical tests:
- ADF test
- KPSS test
- Differencing

### Seasonality Detection
Pattern finding:
- Period detection
- Strength measure
- Visualization

### Forecasting
Prediction:
- ARIMA
- Exponential smoothing
- Evaluation metrics

## Text Processing Tools

### Tokenization
Text splitting:
- Word tokens
- Sentence tokens
- Custom patterns

### Word Frequency
Term analysis:
- Word counts
- N-grams
- Stopword removal

### TF-IDF
Term importance:
- TF-IDF matrix
- Document similarity
- Feature extraction

### Embeddings
Vector representations:
- Word2Vec style
- Sentence embeddings
- Dimensionality reduction

### Sentiment Analysis
Opinion mining:
- Polarity scores
- Subjectivity
- Visualization

## Utility Tools

### Calculator
Scientific calculator:
- Basic operations
- Functions
- Variable storage

### Unit Converter
Conversions:
- Length, mass, time
- Temperature
- Data sizes

### Random Generator
Random data:
- Distributions
- Seed control
- Batch generation

### Hash Generator
Hashing:
- MD5, SHA-256
- File hashing
- Comparison

### JSON Viewer
JSON handling:
- Pretty print
- Validation
- Path navigation

### Regex Tester
Pattern matching:
- Live testing
- Match highlighting
- Common patterns

## Tool Data Flow

Tools can work with:

1. **Loaded Datasets** - From Dataset Panel
2. **Python Variables** - From Console
3. **File Input** - Direct file loading
4. **Manual Input** - Typed data

Results can be:
1. **Exported to CSV** - Save results
2. **Sent to Console** - As Python variables
3. **Visualized** - In plot windows
4. **Copied** - To clipboard

## Backend Integration

Most tools use GPU-accelerated algorithms from `cyxwiz-backend`:

```cpp
// Example: K-Means via ArrayFire
af::array data = loadData();
af::array centroids, labels;
af::kmeans(centroids, labels, data, k, max_iters);
```

See [Backend API](../backend/index.md) for algorithm details.

---

**Next**: [Data Science Tools](data-science/index.md) | [Statistics Tools](statistics/index.md)
