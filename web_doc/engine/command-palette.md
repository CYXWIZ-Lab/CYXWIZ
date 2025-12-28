# Command Palette

The Command Palette provides instant access to all CyxWiz Engine features through a searchable interface.

## Opening the Command Palette

**Shortcut:** `Ctrl+P` (Windows/Linux) or `Cmd+P` (macOS)

Or via menu: **View > Command Palette**

## Interface

```
+--------------------------------------------------+
|  > Search commands...                        [x] |
+--------------------------------------------------+
|  [icon] K-Means Clustering         Clustering    |
|  [icon] DBSCAN Clustering          Clustering    |
|  [icon] Gaussian Mixture Model     Clustering    |
|  [icon] Hierarchical Clustering    Clustering    |
|  [icon] FFT Analysis               Signal Proc.  |
|  [icon] Spectrogram                Signal Proc.  |
|  ... (scrollable list)                           |
+--------------------------------------------------+
```

## Usage

1. Press `Ctrl+P` to open
2. Start typing to filter commands
3. Use `Up/Down` arrows to navigate
4. Press `Enter` to execute selected command
5. Press `Escape` to close without executing

## Search Features

### Fuzzy Matching

The palette uses fuzzy matching, so you don't need exact matches:

| Type | Matches |
|------|---------|
| `kmeans` | K-Means Clustering |
| `fft` | FFT Analysis |
| `lr` | Learning Rate Finder |
| `conf mat` | Confusion Matrix |

### Category Filtering

Prefix search with category to filter:

| Prefix | Category |
|--------|----------|
| `clust:` | Clustering tools |
| `stat:` | Statistics tools |
| `sig:` | Signal processing |
| `time:` | Time series |
| `text:` | Text processing |
| `util:` | Utilities |

Example: `clust:kmeans` shows only clustering tools matching "kmeans"

### Keyword Search

Each tool has associated keywords:

| Tool | Keywords |
|------|----------|
| K-Means | cluster, kmeans, machine learning, unsupervised |
| FFT | fourier, frequency, spectrum, signal |
| ROC-AUC | receiver, operating, curve, classification |

## Available Commands

### Model Analysis

| Command | Description |
|---------|-------------|
| Model Summary | View model architecture summary |
| Architecture Diagram | Visual model diagram |
| LR Finder | Find optimal learning rate |

### Data Science

| Command | Description |
|---------|-------------|
| Data Profiler | Comprehensive data analysis |
| Correlation Matrix | Feature correlations |
| Missing Value Analysis | Find and handle missing data |
| Outlier Detection | Identify outliers |

### Statistics

| Command | Description |
|---------|-------------|
| Descriptive Statistics | Mean, std, quartiles |
| Hypothesis Testing | Statistical tests |
| Distribution Fitter | Fit probability distributions |
| Regression Analysis | Linear/polynomial regression |

### Clustering

| Command | Description |
|---------|-------------|
| K-Means Clustering | K-Means algorithm |
| DBSCAN | Density-based clustering |
| Hierarchical Clustering | Dendrogram clustering |
| GMM | Gaussian Mixture Model |
| Cluster Evaluation | Silhouette, Davies-Bouldin |

### Model Evaluation

| Command | Description |
|---------|-------------|
| Confusion Matrix | Classification matrix |
| ROC-AUC Curve | Receiver operating curve |
| PR Curve | Precision-recall curve |
| Cross-Validation | K-fold validation |
| Learning Curves | Training progress analysis |

### Data Transformations

| Command | Description |
|---------|-------------|
| Normalization | Min-max scaling |
| Standardization | Z-score normalization |
| Log Transform | Logarithmic transform |
| Box-Cox Transform | Power transform |
| Feature Scaling | General scaling options |

### Linear Algebra

| Command | Description |
|---------|-------------|
| Matrix Calculator | Matrix operations |
| Eigendecomposition | Eigen analysis |
| SVD | Singular value decomposition |
| QR Decomposition | QR factorization |
| Cholesky Decomposition | Cholesky factorization |

### Signal Processing

| Command | Description |
|---------|-------------|
| FFT Analysis | Fast Fourier Transform |
| Spectrogram | Time-frequency analysis |
| Filter Designer | Design digital filters |
| Convolution | Signal convolution |
| Wavelet Transform | Wavelet analysis |

### Optimization

| Command | Description |
|---------|-------------|
| Gradient Descent Viz | Visualize optimization |
| Convexity Analysis | Check function convexity |
| Linear Programming | LP solver |
| Quadratic Programming | QP solver |

### Calculus

| Command | Description |
|---------|-------------|
| Numerical Differentiation | Compute derivatives |
| Numerical Integration | Compute integrals |

### Time Series

| Command | Description |
|---------|-------------|
| Time Series Decomposition | Trend/seasonal split |
| ACF/PACF | Autocorrelation analysis |
| Stationarity Test | ADF/KPSS tests |
| Seasonality Detection | Find seasonal patterns |
| Forecasting | Time series prediction |

### Text Processing

| Command | Description |
|---------|-------------|
| Tokenization | Text tokenization |
| Word Frequency | Word count analysis |
| TF-IDF | Term frequency analysis |
| Embeddings | Word/sentence embeddings |
| Sentiment Analysis | Sentiment classification |

### Utilities

| Command | Description |
|---------|-------------|
| Calculator | Scientific calculator |
| Unit Converter | Unit conversions |
| Random Generator | Random number generation |
| Hash Generator | Hash computation |
| JSON Viewer | JSON formatting/viewing |
| Regex Tester | Regular expression testing |

### View Commands

| Command | Description |
|---------|-------------|
| Show Node Editor | Toggle node editor |
| Show Script Editor | Toggle script editor |
| Show Console | Toggle console |
| Show Properties | Toggle properties |
| Show Asset Browser | Toggle asset browser |
| Reset Layout | Reset to default layout |

### File Commands

| Command | Description |
|---------|-------------|
| New Project | Create new project |
| Open Project | Open existing project |
| New Script | Create new script |
| Save | Save current file |
| Save All | Save all open files |

### Training Commands

| Command | Description |
|---------|-------------|
| Start Training | Begin training |
| Stop Training | Stop training |
| Connect to Server | Connect to Central Server |

## Command Structure

Each command entry contains:

```cpp
struct ToolEntry {
    std::string name;       // Display name
    std::string category;   // Tool category
    std::string keywords;   // Search keywords
    std::string icon;       // FontAwesome icon
    std::string shortcut;   // Keyboard shortcut (if any)
    std::function<void()> callback;  // Action to execute
};
```

## Match Scoring

The palette ranks results by match quality:

1. **Exact prefix match** - Highest score
2. **Word boundary match** - High score
3. **Substring match** - Medium score
4. **Character sequence** - Lower score
5. **Keyword match** - Additional bonus

Example scores for query "km":
- "K-Means" = 100 (prefix match)
- "DBSCAN" = 0 (no match)
- "K-Means Clustering" = 100 (word match)

## Customization

### Adding Custom Commands

Developers can add commands via the API:

```cpp
// In your panel initialization
ToolEntry entry;
entry.name = "My Custom Tool";
entry.category = "Custom";
entry.keywords = "custom tool example";
entry.icon = ICON_FA_STAR;
entry.callback = []() {
    // Your tool logic
};

toolbar->RegisterTool(entry);
```

### Keyboard Shortcut for Commands

Commands with shortcuts show them in the palette:

```
+--------------------------------------------------+
|  [icon] New Project              Ctrl+Shift+N    |
|  [icon] Open Project             Ctrl+Shift+O    |
+--------------------------------------------------+
```

## Tips

1. **Type Less**: Fuzzy search finds results with minimal typing
2. **Use Initials**: "cm" finds "Confusion Matrix"
3. **Category Prefix**: Narrow results with category prefixes
4. **Recent First**: Recently used commands appear higher
5. **Escape to Cancel**: Won't execute anything

---

**Next**: [Themes](themes.md) | [Node Editor](node-editor/index.md)
