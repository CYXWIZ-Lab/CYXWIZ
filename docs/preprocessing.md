# CyxWiz Preprocessing Module

## Overview

The Preprocessing Module is a core system component in CyxWiz that provides comprehensive data preparation tools for all supported data types. It sits between data loading (Asset Browser) and model training, enabling users to clean, transform, and augment their data through a visual interface.

### System Flow

```
Asset Browser (Load Data)
        ↓
Preprocessing Module (Clean & Transform)
        ↓
Dataset Manager (Batch & Feed)
        ↓
Training Pipeline (Model Training)
```

### Two Preprocessing Paths

1. **Visual Preprocessing Module** (This Document)
   - Built-in system tools
   - Visual node-based pipeline
   - Beginner-friendly
   - Real-time preview
   - Supports all data types

2. **Script-Based Preprocessing** (Advanced Users)
   - Python scripting in Script Editor
   - DuckDB for SQL-based transformations
   - Polars for high-performance DataFrame operations
   - Full programmatic control
   - Custom transformations

---

## Architecture

### Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessing Module                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Input   │ → │  Stage 1 │ → │  Stage 2 │ → │  Output  │     │
│  │  Loader  │   │  (Clean) │   │(Transform│   │  Writer  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Preview Panel                         │    │
│  │   Before: [████████]    After: [████████]               │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

```cpp
// Preprocessing Pipeline
class PreprocessingPipeline {
    std::vector<PreprocessingStage> stages_;
    DataType input_type_;
    DataType output_type_;

public:
    void AddStage(std::unique_ptr<PreprocessingStage> stage);
    void RemoveStage(size_t index);
    void ReorderStage(size_t from, size_t to);

    PreviewResult Preview(const DataSample& sample);
    void Process(const Dataset& input, Dataset& output);
    void ProcessAsync(const Dataset& input, ProgressCallback callback);

    // Serialization
    nlohmann::json ToJson() const;
    static PreprocessingPipeline FromJson(const nlohmann::json& j);
};

// Base class for all preprocessing operations
class PreprocessingStage {
public:
    virtual ~PreprocessingStage() = default;
    virtual std::string GetName() const = 0;
    virtual std::string GetCategory() const = 0;
    virtual DataType GetInputType() const = 0;
    virtual DataType GetOutputType() const = 0;

    virtual void Configure(const nlohmann::json& params) = 0;
    virtual nlohmann::json GetConfiguration() const = 0;

    virtual DataSample Process(const DataSample& input) = 0;
    virtual void RenderUI() = 0;  // ImGui configuration UI
};
```

---

## Text Preprocessing Tools

### 1. Text Cleaning

| Tool | Description | Parameters |
|------|-------------|------------|
| **Remove Whitespace** | Strip leading/trailing whitespace, normalize internal spaces | `trim_start`, `trim_end`, `normalize_internal` |
| **Remove Punctuation** | Remove or replace punctuation marks | `keep_sentence_endings`, `replacement_char` |
| **Remove Numbers** | Remove or replace numeric characters | `keep_decimals`, `replacement` |
| **Remove Special Characters** | Remove non-alphanumeric characters | `allowed_chars`, `replacement` |
| **Remove HTML/XML Tags** | Strip markup tags from text | `keep_content`, `decode_entities` |
| **Remove URLs** | Detect and remove/replace URLs | `replacement`, `keep_domain` |
| **Remove Emails** | Detect and remove/replace email addresses | `replacement`, `mask_pattern` |
| **Remove Emojis** | Remove or replace emoji characters | `replacement`, `keep_emoticons` |
| **Fix Unicode** | Normalize unicode, fix encoding issues | `normalization_form` (NFC, NFD, NFKC, NFKD) |
| **Remove Duplicates** | Remove duplicate lines/sentences | `scope` (line, sentence, document) |

### 2. Text Normalization

| Tool | Description | Parameters |
|------|-------------|------------|
| **Lowercase** | Convert text to lowercase | `locale` |
| **Uppercase** | Convert text to uppercase | `locale` |
| **Title Case** | Capitalize first letter of each word | `exceptions` (list of words to skip) |
| **Sentence Case** | Capitalize first letter of sentences | - |
| **ASCII Transliteration** | Convert non-ASCII to ASCII equivalents | `language` |
| **Expand Contractions** | "don't" → "do not" | `language`, `custom_mappings` |
| **Spell Correction** | Fix common spelling errors | `dictionary`, `max_edit_distance` |
| **Normalize Numbers** | "1000" → "one thousand" or vice versa | `direction`, `locale` |

### 3. Tokenization

| Tool | Description | Parameters |
|------|-------------|------------|
| **Word Tokenizer** | Split text into words | `language`, `keep_punctuation` |
| **Sentence Tokenizer** | Split text into sentences | `language`, `abbreviations` |
| **Subword Tokenizer (BPE)** | Byte-Pair Encoding tokenization | `vocab_size`, `min_frequency` |
| **WordPiece Tokenizer** | BERT-style tokenization | `vocab_size`, `unk_token` |
| **SentencePiece** | Unigram/BPE tokenization | `model_type`, `vocab_size` |
| **Character Tokenizer** | Split into individual characters | `include_spaces` |
| **Regex Tokenizer** | Custom regex-based splitting | `pattern`, `flags` |

### 4. Text Filtering

| Tool | Description | Parameters |
|------|-------------|------------|
| **Stopword Removal** | Remove common words | `language`, `custom_stopwords` |
| **Rare Word Removal** | Remove infrequent words | `min_frequency`, `min_document_frequency` |
| **Length Filter** | Filter by text/token length | `min_length`, `max_length`, `unit` (char/word/token) |
| **Language Filter** | Keep only specific languages | `languages`, `confidence_threshold` |
| **Quality Filter** | Remove low-quality text | `min_word_length_avg`, `max_special_char_ratio` |
| **Profanity Filter** | Remove or mask profanity | `word_list`, `replacement`, `mask_char` |

### 5. Text Transformation

| Tool | Description | Parameters |
|------|-------------|------------|
| **Stemming** | Reduce words to stems (Porter, Snowball) | `algorithm`, `language` |
| **Lemmatization** | Reduce words to lemmas | `language`, `pos_tag` |
| **N-gram Generation** | Generate word/character n-grams | `n`, `type` (word/char), `padding` |
| **Text Augmentation** | Synonym replacement, back-translation | `method`, `aug_ratio` |
| **Masking** | Mask tokens for MLM training | `mask_ratio`, `mask_token` |
| **Truncation/Padding** | Standardize sequence length | `max_length`, `padding_side`, `truncation_side` |

### 6. Vectorization (Preview)

| Tool | Description | Parameters |
|------|-------------|------------|
| **Bag of Words** | Word frequency vectors | `max_features`, `binary` |
| **TF-IDF** | Term frequency-inverse document frequency | `max_features`, `norm`, `sublinear_tf` |
| **Word Embeddings** | Word2Vec, GloVe, FastText | `model`, `dimension`, `pretrained_path` |

---

## Image Preprocessing Tools

### 1. Basic Transformations

| Tool | Description | Parameters |
|------|-------------|------------|
| **Resize** | Change image dimensions | `width`, `height`, `interpolation` (nearest, bilinear, bicubic, lanczos) |
| **Crop** | Extract region from image | `x`, `y`, `width`, `height`, `mode` (center, random, corner) |
| **Pad** | Add padding to image | `top`, `bottom`, `left`, `right`, `mode` (constant, reflect, replicate) |
| **Rotate** | Rotate image | `angle`, `expand`, `fill_color` |
| **Flip** | Mirror image | `direction` (horizontal, vertical, both) |
| **Transpose** | Swap axes | `method` (rotate_90, rotate_180, rotate_270, flip_left_right) |

### 2. Color Adjustments

| Tool | Description | Parameters |
|------|-------------|------------|
| **Grayscale** | Convert to grayscale | `method` (luminosity, average, lightness) |
| **Color Space Conversion** | Convert between color spaces | `from`, `to` (RGB, BGR, HSV, LAB, YUV, CMYK) |
| **Brightness** | Adjust brightness | `factor` (-1.0 to 1.0) |
| **Contrast** | Adjust contrast | `factor` (0.0 to 2.0) |
| **Saturation** | Adjust color saturation | `factor` (0.0 to 2.0) |
| **Hue** | Shift hue values | `shift` (-180 to 180) |
| **Gamma Correction** | Apply gamma curve | `gamma` (0.1 to 10.0) |
| **Color Balance** | Adjust RGB channels independently | `red`, `green`, `blue` factors |
| **Auto Contrast** | Automatic contrast stretching | `cutoff_low`, `cutoff_high` |
| **Histogram Equalization** | Enhance contrast via histogram | `method` (standard, CLAHE), `clip_limit` |
| **White Balance** | Correct color temperature | `method` (gray_world, white_patch, custom), `temperature` |

### 3. Filtering & Enhancement

| Tool | Description | Parameters |
|------|-------------|------------|
| **Blur** | Apply blur filter | `kernel_size`, `type` (gaussian, box, median, bilateral) |
| **Sharpen** | Enhance edges | `amount`, `radius`, `threshold` |
| **Denoise** | Remove noise | `method` (gaussian, bilateral, nlm), `strength` |
| **Edge Detection** | Detect edges | `method` (sobel, canny, laplacian), `threshold` |
| **Morphological Operations** | Erosion, dilation, opening, closing | `operation`, `kernel_size`, `iterations` |
| **Unsharp Mask** | Sharpen via unsharp masking | `radius`, `amount`, `threshold` |

### 4. Normalization

| Tool | Description | Parameters |
|------|-------------|------------|
| **Min-Max Normalization** | Scale to [0, 1] or [0, 255] | `output_range` |
| **Z-Score Normalization** | Standardize (mean=0, std=1) | `per_channel`, `mean`, `std` |
| **ImageNet Normalization** | Standard ImageNet preprocessing | - (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| **Percentile Normalization** | Normalize based on percentiles | `low_percentile`, `high_percentile` |

### 5. Augmentation

| Tool | Description | Parameters |
|------|-------------|------------|
| **Random Crop** | Random region extraction | `size`, `scale`, `ratio` |
| **Random Flip** | Random horizontal/vertical flip | `p_horizontal`, `p_vertical` |
| **Random Rotation** | Random rotation within range | `degrees` (-45, 45), `expand` |
| **Random Affine** | Random affine transformation | `degrees`, `translate`, `scale`, `shear` |
| **Random Perspective** | Random perspective distortion | `distortion_scale` |
| **Random Erasing** | Random rectangular masking | `p`, `scale`, `ratio`, `value` |
| **Color Jitter** | Random color adjustments | `brightness`, `contrast`, `saturation`, `hue` |
| **Gaussian Noise** | Add random noise | `mean`, `std` |
| **Cutout** | Random square masking | `n_holes`, `length` |
| **Mixup** | Blend two images | `alpha` |
| **CutMix** | Replace region with another image | `alpha` |
| **AutoAugment** | Learned augmentation policies | `policy` (imagenet, cifar10, svhn) |
| **RandAugment** | Random augmentation selection | `n`, `m` (magnitude) |
| **TrivialAugment** | Simple random augmentation | `num_magnitude_bins` |

### 6. Format Conversion

| Tool | Description | Parameters |
|------|-------------|------------|
| **Channel Order** | Reorder channels | `order` (CHW, HWC) |
| **Data Type** | Convert data type | `dtype` (uint8, float32, float16) |
| **Bit Depth** | Change bit depth | `bits` (8, 16, 32) |

---

## Audio Preprocessing Tools

### 1. Loading & Format

| Tool | Description | Parameters |
|------|-------------|------------|
| **Resample** | Change sample rate | `target_sr` (8000, 16000, 22050, 44100, 48000) |
| **Convert Channels** | Mono/stereo conversion | `mode` (mono, stereo), `mix_method` |
| **Bit Depth Conversion** | Change audio bit depth | `bits` (16, 24, 32) |
| **Format Conversion** | Convert between formats | `format` (wav, mp3, flac, ogg) |

### 2. Trimming & Segmentation

| Tool | Description | Parameters |
|------|-------------|------------|
| **Trim Silence** | Remove leading/trailing silence | `top_db`, `frame_length`, `hop_length` |
| **Split on Silence** | Split audio at silent regions | `min_silence_len`, `silence_thresh`, `keep_silence` |
| **Fixed Length Crop** | Extract fixed duration | `duration`, `offset`, `mode` (start, center, random) |
| **Pad/Truncate** | Standardize duration | `target_length`, `pad_mode` (zero, reflect, wrap) |
| **Voice Activity Detection** | Detect speech regions | `method` (energy, webrtc), `aggressiveness` |

### 3. Volume & Dynamics

| Tool | Description | Parameters |
|------|-------------|------------|
| **Normalize Volume** | Normalize audio level | `method` (peak, rms, lufs), `target_level` |
| **Gain** | Adjust volume | `gain_db` |
| **Compression** | Dynamic range compression | `threshold`, `ratio`, `attack`, `release` |
| **Limiter** | Prevent clipping | `threshold`, `release` |
| **Fade In/Out** | Apply fade effects | `fade_in_duration`, `fade_out_duration`, `curve` |
| **De-esser** | Reduce sibilance | `frequency`, `threshold` |

### 4. Noise Reduction

| Tool | Description | Parameters |
|------|-------------|------------|
| **Noise Gate** | Remove low-level noise | `threshold`, `attack`, `release` |
| **Spectral Noise Reduction** | Reduce broadband noise | `noise_profile`, `reduction_amount` |
| **High-pass Filter** | Remove low frequencies | `cutoff_freq`, `order` |
| **Low-pass Filter** | Remove high frequencies | `cutoff_freq`, `order` |
| **Band-pass Filter** | Keep frequency range | `low_cutoff`, `high_cutoff`, `order` |
| **Notch Filter** | Remove specific frequency | `frequency`, `q_factor` |
| **Hum Removal** | Remove power line hum | `frequency` (50 or 60 Hz), `harmonics` |

### 5. Feature Extraction

| Tool | Description | Parameters |
|------|-------------|------------|
| **Spectrogram** | Time-frequency representation | `n_fft`, `hop_length`, `win_length`, `window` |
| **Mel Spectrogram** | Mel-scaled spectrogram | `n_mels`, `fmin`, `fmax`, `n_fft`, `hop_length` |
| **MFCC** | Mel-frequency cepstral coefficients | `n_mfcc`, `n_mels`, `dct_type` |
| **Chroma** | Pitch class distribution | `n_chroma`, `hop_length` |
| **Spectral Centroid** | Center of mass of spectrum | `n_fft`, `hop_length` |
| **Spectral Bandwidth** | Variance around centroid | `n_fft`, `hop_length` |
| **Zero Crossing Rate** | Rate of sign changes | `frame_length`, `hop_length` |
| **RMS Energy** | Root mean square energy | `frame_length`, `hop_length` |
| **Tonnetz** | Tonal centroid features | `chroma_type` |
| **Tempogram** | Tempo estimation | `hop_length`, `win_length` |

### 6. Augmentation

| Tool | Description | Parameters |
|------|-------------|------------|
| **Time Stretch** | Change speed without pitch | `rate` (0.5 to 2.0) |
| **Pitch Shift** | Change pitch without speed | `semitones` (-12 to 12) |
| **Add Noise** | Add background noise | `noise_type` (white, pink, brown), `snr_db` |
| **Room Reverb** | Add reverb effect | `room_size`, `damping`, `wet_level` |
| **Time Masking** | Mask time segments (SpecAugment) | `max_mask_length`, `num_masks` |
| **Frequency Masking** | Mask frequency bands (SpecAugment) | `max_mask_length`, `num_masks` |
| **Polarity Inversion** | Flip waveform polarity | - |
| **Random Gain** | Random volume adjustment | `min_gain_db`, `max_gain_db` |

---

## Video Preprocessing Tools

### 1. Frame Extraction

| Tool | Description | Parameters |
|------|-------------|------------|
| **Extract All Frames** | Extract every frame | `output_format` (png, jpg) |
| **Fixed FPS Extraction** | Extract at specific framerate | `target_fps` (1, 5, 10, 24, 30) |
| **Keyframe Extraction** | Extract only keyframes (I-frames) | - |
| **Scene Detection** | Extract frames at scene changes | `threshold`, `min_scene_length` |
| **Uniform Sampling** | Extract N evenly-spaced frames | `num_frames` |
| **Random Sampling** | Extract N random frames | `num_frames`, `seed` |
| **Temporal Sliding Window** | Extract overlapping windows | `window_size`, `stride` |

### 2. Video Transformations

| Tool | Description | Parameters |
|------|-------------|------------|
| **Resize** | Change video resolution | `width`, `height`, `interpolation` |
| **Crop** | Extract region from video | `x`, `y`, `width`, `height` |
| **Rotate** | Rotate video | `angle` (90, 180, 270) |
| **Flip** | Mirror video | `direction` (horizontal, vertical) |
| **Trim** | Cut video segment | `start_time`, `end_time` |
| **Speed Change** | Adjust playback speed | `factor` (0.25 to 4.0) |
| **Frame Rate Change** | Convert frame rate | `target_fps`, `interpolation` |
| **Aspect Ratio** | Change aspect ratio | `ratio`, `method` (pad, crop, stretch) |

### 3. Color & Quality

| Tool | Description | Parameters |
|------|-------------|------------|
| **Grayscale** | Convert to grayscale | - |
| **Brightness/Contrast** | Adjust per-frame | `brightness`, `contrast` |
| **Color Correction** | Apply color grading | `preset` or custom curves |
| **Deinterlace** | Remove interlacing | `method` (blend, bob, yadif) |
| **Denoise** | Temporal/spatial denoising | `strength`, `method` |
| **Stabilization** | Reduce camera shake | `smoothing`, `crop_ratio` |
| **Super Resolution** | Upscale video | `scale`, `model` |

### 4. Audio Track

| Tool | Description | Parameters |
|------|-------------|------------|
| **Extract Audio** | Extract audio track | `format` (wav, mp3), `sample_rate` |
| **Remove Audio** | Strip audio from video | - |
| **Replace Audio** | Replace audio track | `audio_path` |
| **Audio-Video Sync** | Fix sync issues | `offset_ms` |

### 5. Temporal Features

| Tool | Description | Parameters |
|------|-------------|------------|
| **Optical Flow** | Calculate motion between frames | `method` (farneback, lucas_kanade, raft) |
| **Temporal Difference** | Frame-to-frame difference | `order` (1, 2) |
| **Motion History** | Accumulated motion image | `duration`, `decay` |
| **Temporal Pooling** | Aggregate frames | `method` (mean, max, attention), `window_size` |

### 6. Augmentation

| Tool | Description | Parameters |
|------|-------------|------------|
| **Random Temporal Crop** | Random time segment | `duration` |
| **Random Spatial Crop** | Random region per frame | `size` |
| **Temporal Jitter** | Vary playback speed randomly | `range` |
| **Frame Dropout** | Randomly skip frames | `dropout_rate` |
| **Color Augmentation** | Per-frame color jitter | `brightness`, `contrast`, `saturation`, `hue` |
| **Temporal Masking** | Mask random frames | `mask_ratio` |

---

## Tabular Data Preprocessing

### 1. Missing Value Handling

| Tool | Description | Parameters |
|------|-------------|------------|
| **Drop Missing** | Remove rows/columns with missing values | `axis`, `how` (any, all), `threshold` |
| **Fill Missing** | Fill with constant value | `value`, `method` |
| **Imputation** | Statistical imputation | `strategy` (mean, median, mode, constant), `columns` |
| **Forward/Backward Fill** | Fill with previous/next value | `method` (ffill, bfill), `limit` |
| **Interpolation** | Interpolate missing values | `method` (linear, polynomial, spline) |
| **KNN Imputation** | K-nearest neighbors imputation | `n_neighbors`, `weights` |
| **Model-based Imputation** | Use ML model to predict missing | `model_type`, `columns` |

### 2. Encoding

| Tool | Description | Parameters |
|------|-------------|------------|
| **Label Encoding** | Convert categories to integers | `columns` |
| **One-Hot Encoding** | Create binary columns | `columns`, `drop_first`, `sparse` |
| **Ordinal Encoding** | Encode with order preserved | `columns`, `order` |
| **Target Encoding** | Encode based on target mean | `columns`, `smoothing` |
| **Frequency Encoding** | Encode based on frequency | `columns` |
| **Binary Encoding** | Encode as binary representations | `columns` |
| **Hash Encoding** | Hash-based encoding | `columns`, `n_components` |

### 3. Scaling & Normalization

| Tool | Description | Parameters |
|------|-------------|------------|
| **Min-Max Scaling** | Scale to [0, 1] | `columns`, `feature_range` |
| **Standard Scaling** | Z-score normalization | `columns`, `with_mean`, `with_std` |
| **Robust Scaling** | Scale using median/IQR | `columns`, `quantile_range` |
| **Max-Abs Scaling** | Scale by max absolute value | `columns` |
| **Log Transform** | Apply log transformation | `columns`, `base` |
| **Box-Cox Transform** | Power transformation | `columns`, `lambda` |
| **Yeo-Johnson Transform** | Extended power transformation | `columns` |
| **Quantile Transform** | Transform to uniform/normal distribution | `columns`, `output_distribution` |

### 4. Outlier Handling

| Tool | Description | Parameters |
|------|-------------|------------|
| **Z-Score Filter** | Remove based on z-score | `threshold`, `columns` |
| **IQR Filter** | Remove based on interquartile range | `factor` (default 1.5), `columns` |
| **Percentile Clip** | Clip values at percentiles | `lower`, `upper`, `columns` |
| **Isolation Forest** | Detect outliers with ML | `contamination`, `columns` |
| **LOF (Local Outlier Factor)** | Density-based outlier detection | `n_neighbors`, `contamination` |
| **Winsorization** | Replace outliers with percentile values | `limits`, `columns` |

### 5. Feature Engineering

| Tool | Description | Parameters |
|------|-------------|------------|
| **Polynomial Features** | Generate polynomial combinations | `degree`, `interaction_only` |
| **Binning** | Discretize continuous variables | `n_bins`, `strategy` (uniform, quantile, kmeans) |
| **Date/Time Features** | Extract date components | `columns`, `features` (year, month, day, hour, weekday, etc.) |
| **Text Features** | Extract from text columns | `columns`, `features` (length, word_count, etc.) |
| **Aggregation** | Group-by aggregations | `group_by`, `agg_columns`, `functions` |
| **Rolling Statistics** | Moving window calculations | `window`, `columns`, `functions` |
| **Lag Features** | Create lagged values | `columns`, `lags` |
| **Difference Features** | Calculate differences | `columns`, `periods` |

### 6. Feature Selection

| Tool | Description | Parameters |
|------|-------------|------------|
| **Variance Threshold** | Remove low-variance features | `threshold` |
| **Correlation Filter** | Remove highly correlated features | `threshold` |
| **SelectKBest** | Select top K features | `k`, `score_func` |
| **Recursive Feature Elimination** | RFE with estimator | `n_features`, `estimator` |
| **Feature Importance** | Select by model importance | `model`, `threshold` |
| **LASSO Selection** | L1-based selection | `alpha` |

### 7. Sampling

| Tool | Description | Parameters |
|------|-------------|------------|
| **Random Sampling** | Random subset | `n` or `fraction`, `replace` |
| **Stratified Sampling** | Preserve class distribution | `n` or `fraction`, `stratify_column` |
| **Undersampling** | Reduce majority class | `strategy`, `random_state` |
| **Oversampling (SMOTE)** | Generate minority samples | `sampling_strategy`, `k_neighbors` |
| **Train-Test Split** | Split into train/test sets | `test_size`, `stratify`, `shuffle` |
| **K-Fold Split** | Create K-fold cross-validation splits | `n_splits`, `shuffle` |

---

## Statistical & Data Analysis

The Preprocessing Module includes comprehensive statistical analysis and exploratory data analysis (EDA) tools. These help users understand their data before and after transformations, identify issues, and make informed preprocessing decisions.

### 1. Descriptive Statistics

| Tool | Description | Output |
|------|-------------|--------|
| **Summary Statistics** | Compute mean, median, mode, std, variance, min, max, range | Statistics table per column |
| **Percentiles** | Calculate percentile values (25th, 50th, 75th, custom) | Percentile table |
| **Skewness** | Measure asymmetry of distribution | Skewness value per column |
| **Kurtosis** | Measure tailedness of distribution | Kurtosis value per column |
| **Count Statistics** | Total count, unique count, missing count, duplicates | Count summary |
| **Data Types** | Infer and display column data types | Type summary with recommendations |

### 2. Distribution Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Histogram** | Frequency distribution visualization | `bins`, `density`, `cumulative` |
| **Density Plot (KDE)** | Kernel density estimation | `bandwidth`, `kernel` |
| **Box Plot** | Quartile visualization with outliers | `orientation`, `show_outliers` |
| **Violin Plot** | Distribution shape visualization | `scale`, `inner` |
| **Q-Q Plot** | Quantile-quantile plot for normality check | `distribution` (normal, uniform, etc.) |
| **P-P Plot** | Probability-probability plot | `distribution` |
| **ECDF** | Empirical cumulative distribution function | - |
| **Rug Plot** | Individual data points on axis | `height`, `alpha` |

### 3. Correlation Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Pearson Correlation** | Linear correlation coefficient | `columns` |
| **Spearman Correlation** | Rank-based correlation | `columns` |
| **Kendall Correlation** | Ordinal association | `columns` |
| **Point-Biserial** | Correlation between binary and continuous | `binary_col`, `continuous_col` |
| **Correlation Matrix** | Pairwise correlations heatmap | `method`, `threshold_highlight` |
| **Partial Correlation** | Correlation controlling for variables | `columns`, `control_vars` |
| **Cross-Correlation** | Time-lagged correlation | `lag_range` |
| **Mutual Information** | Non-linear dependency measure | `n_neighbors` |

### 4. Hypothesis Testing

| Tool | Description | Parameters |
|------|-------------|------------|
| **t-Test (One Sample)** | Test mean against value | `column`, `population_mean` |
| **t-Test (Two Sample)** | Compare means of two groups | `group_col`, `value_col`, `equal_var` |
| **t-Test (Paired)** | Compare paired observations | `col1`, `col2` |
| **ANOVA (One-Way)** | Compare means across groups | `group_col`, `value_col` |
| **ANOVA (Two-Way)** | Two-factor analysis of variance | `factor1`, `factor2`, `value_col` |
| **Chi-Square Test** | Test independence of categorical variables | `col1`, `col2` |
| **Fisher's Exact Test** | Small sample categorical test | `col1`, `col2` |
| **Mann-Whitney U** | Non-parametric two-sample test | `group_col`, `value_col` |
| **Wilcoxon Signed-Rank** | Non-parametric paired test | `col1`, `col2` |
| **Kruskal-Wallis** | Non-parametric ANOVA | `group_col`, `value_col` |
| **Levene's Test** | Test equality of variances | `group_col`, `value_col` |
| **Bartlett's Test** | Test homogeneity of variances | `group_col`, `value_col` |

### 5. Normality Tests

| Tool | Description | Parameters |
|------|-------------|------------|
| **Shapiro-Wilk** | Test for normality (small samples) | `column` |
| **D'Agostino-Pearson** | Test using skewness and kurtosis | `column` |
| **Kolmogorov-Smirnov** | Compare to theoretical distribution | `column`, `distribution` |
| **Anderson-Darling** | Weighted KS test | `column`, `distribution` |
| **Jarque-Bera** | Test based on skewness/kurtosis | `column` |
| **Lilliefors** | KS test with estimated parameters | `column` |

### 6. Time Series Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Trend Detection** | Identify upward/downward trends | `column`, `method` (linear, polynomial) |
| **Seasonality Detection** | Detect periodic patterns | `column`, `period` |
| **Decomposition** | Separate trend, seasonal, residual | `column`, `model` (additive, multiplicative) |
| **Autocorrelation (ACF)** | Correlation with lagged values | `column`, `max_lag` |
| **Partial Autocorrelation (PACF)** | Direct correlation at each lag | `column`, `max_lag` |
| **Stationarity Test (ADF)** | Augmented Dickey-Fuller test | `column`, `regression` |
| **Stationarity Test (KPSS)** | Kwiatkowski-Phillips-Schmidt-Shin | `column`, `regression` |
| **Granger Causality** | Test predictive causality | `col1`, `col2`, `max_lag` |
| **Cointegration Test** | Test for long-run equilibrium | `columns`, `method` |
| **Change Point Detection** | Detect distribution changes | `column`, `method` (PELT, BOCPD) |

### 7. Data Profiling (Auto-EDA)

| Tool | Description | Output |
|------|-------------|--------|
| **Auto Profile** | Generate comprehensive data report | HTML/PDF report |
| **Data Quality Score** | Overall data quality assessment | Score 0-100 with breakdown |
| **Missing Data Analysis** | Patterns and mechanisms of missingness | Heatmap, MCAR/MAR/MNAR assessment |
| **Cardinality Analysis** | Unique values and high-cardinality detection | Cardinality report |
| **Data Type Inference** | Smart type detection with suggestions | Type recommendations |
| **Constant Column Detection** | Find columns with single value | List of constant columns |
| **Duplicate Detection** | Find duplicate rows/patterns | Duplicate report |
| **Memory Usage Analysis** | Estimate and optimize memory | Memory report with optimization suggestions |

### 8. Visualization Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| **Scatter Plot** | Two-variable relationship | `x`, `y`, `color`, `size`, `trendline` |
| **Scatter Matrix** | Pairwise scatter plots | `columns`, `diagonal` |
| **Heatmap** | 2D color-coded matrix | `data`, `colormap`, `annotations` |
| **Bar Chart** | Categorical frequency | `column`, `orientation`, `sort` |
| **Line Chart** | Sequential data | `x`, `y`, `markers` |
| **Area Chart** | Stacked area visualization | `columns`, `stacked` |
| **Pie Chart** | Proportion visualization | `column`, `explode` |
| **Parallel Coordinates** | Multi-dimensional visualization | `columns`, `color_by` |
| **Andrews Curves** | High-dimensional visualization | `columns`, `class_column` |
| **RadViz** | Radial visualization | `columns`, `class_column` |
| **Joint Plot** | Bivariate + marginal distributions | `x`, `y`, `kind` |
| **Pair Plot** | Pairwise relationships grid | `columns`, `hue`, `diag_kind` |

### 9. Anomaly & Outlier Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Z-Score Analysis** | Distance from mean in std units | `threshold`, `columns` |
| **IQR Analysis** | Interquartile range method | `factor`, `columns` |
| **Isolation Forest** | Tree-based anomaly detection | `contamination`, `n_estimators` |
| **Local Outlier Factor** | Density-based outlier detection | `n_neighbors`, `contamination` |
| **DBSCAN Outliers** | Clustering-based detection | `eps`, `min_samples` |
| **One-Class SVM** | Support vector outlier detection | `nu`, `kernel` |
| **Mahalanobis Distance** | Multivariate distance measure | `columns` |
| **Outlier Visualization** | Visual highlighting of outliers | `method`, `columns` |

### 10. Feature Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Variance Analysis** | Identify low-variance features | `threshold` |
| **Information Gain** | Feature importance for classification | `target_column` |
| **Chi-Square Feature Scores** | Categorical feature importance | `target_column` |
| **ANOVA F-Scores** | Numerical feature importance | `target_column` |
| **Mutual Information** | Non-linear feature importance | `target_column`, `discrete_features` |
| **Permutation Importance** | Model-agnostic importance | `model`, `target_column` |
| **VIF (Variance Inflation Factor)** | Multicollinearity detection | `columns` |
| **Condition Number** | Matrix stability measure | `columns` |

### 11. Comparison & Drift Analysis

| Tool | Description | Parameters |
|------|-------------|------------|
| **Dataset Comparison** | Compare two datasets | `dataset1`, `dataset2` |
| **Distribution Comparison** | Compare column distributions | `col1`, `col2`, `test` |
| **Population Stability Index (PSI)** | Measure distribution shift | `baseline`, `current`, `buckets` |
| **KL Divergence** | Measure distribution difference | `col1`, `col2` |
| **JS Divergence** | Symmetric KL divergence | `col1`, `col2` |
| **Wasserstein Distance** | Earth mover's distance | `col1`, `col2` |
| **Data Drift Detection** | Detect concept/data drift | `reference`, `current`, `method` |
| **Feature Drift Report** | Per-feature drift analysis | `reference`, `current` |

### 12. Report Generation

| Tool | Description | Output Format |
|------|-------------|---------------|
| **Summary Report** | Quick overview of dataset | Text/Markdown |
| **Full EDA Report** | Comprehensive analysis report | HTML/PDF |
| **Quality Report** | Data quality assessment | HTML/PDF |
| **Comparison Report** | Before/after transformation comparison | HTML/PDF |
| **Statistical Test Report** | Hypothesis test results | HTML/PDF |
| **Export Statistics** | Export computed statistics | CSV/JSON/Excel |

### Statistical Analysis UI

```cpp
class StatisticalAnalysisPanel : public Panel {
    Dataset* current_dataset_;
    StatisticsCache cache_;

public:
    void Render() override {
        RenderDatasetSelector();
        RenderAnalysisCategories();
        RenderResultsPanel();
        RenderVisualizationPanel();
    }

private:
    void RenderAnalysisCategories() {
        if (ImGui::TreeNode("Descriptive Statistics")) {
            if (ImGui::Button("Compute Summary")) {
                cache_.summary = ComputeSummaryStatistics(current_dataset_);
            }
            if (ImGui::Button("Show Distributions")) {
                show_distribution_plots_ = true;
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Correlation Analysis")) {
            static int corr_method = 0;
            ImGui::Combo("Method", &corr_method, "Pearson\0Spearman\0Kendall\0");
            if (ImGui::Button("Compute Correlation Matrix")) {
                cache_.correlation = ComputeCorrelation(current_dataset_, corr_method);
            }
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Hypothesis Testing")) {
            // Test selection and configuration
            RenderHypothesisTestUI();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Time Series")) {
            RenderTimeSeriesAnalysisUI();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Auto EDA")) {
            if (ImGui::Button("Generate Full Report")) {
                GenerateEDAReport(current_dataset_);
            }
            ImGui::TreePop();
        }
    }
};
```

### Integration with Preprocessing Pipeline

Statistical analysis integrates seamlessly with preprocessing:

```cpp
// Example: Analyze before/after preprocessing
PreprocessingPipeline pipeline;
pipeline.AddStage<NormalizeStage>();
pipeline.AddStage<OutlierRemovalStage>();

// Compute statistics before
auto stats_before = ComputeStatistics(original_dataset);

// Apply preprocessing
auto processed_dataset = pipeline.Process(original_dataset);

// Compute statistics after
auto stats_after = ComputeStatistics(processed_dataset);

// Generate comparison report
GenerateComparisonReport(stats_before, stats_after, "preprocessing_impact.html");
```

### Script-Based Statistical Analysis

For advanced users, full statistical libraries are available via scripting:

```python
import cyxwiz
import polars as pl
from scipy import stats
import numpy as np

# Load dataset
df = pl.read_csv("project://datasets/sales.csv")

# Descriptive statistics
print(df.describe())

# Normality test
for col in df.select(pl.col(pl.Float64)).columns:
    stat, p_value = stats.shapiro(df[col].to_numpy())
    print(f"{col}: Shapiro-Wilk p-value = {p_value:.4f}")

# Correlation matrix
corr_matrix = df.select(pl.col(pl.Float64)).to_pandas().corr()
print(corr_matrix)

# t-test example
group_a = df.filter(pl.col("group") == "A")["value"].to_numpy()
group_b = df.filter(pl.col("group") == "B")["value"].to_numpy()
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-test: t={t_stat:.4f}, p={p_value:.4f}")

# Time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df["value"].to_numpy(), period=12)
```

---

## Integration with CyxWiz

### Asset Browser Integration

```cpp
// When user right-clicks a file in Asset Browser
void AssetBrowser::OnContextMenu(const std::string& file_path) {
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Open with Preprocessing Module")) {
            PreprocessingModule::Instance().OpenFile(file_path);
        }
        ImGui::EndPopup();
    }
}
```

### Preprocessing Panel UI

```cpp
class PreprocessingPanel : public Panel {
    PreprocessingPipeline pipeline_;
    DataSample current_sample_;
    PreviewResult preview_;

public:
    void Render() override {
        RenderToolPalette();     // Left: available tools by category
        RenderPipelineView();    // Center: current pipeline stages
        RenderPreviewPanel();    // Right: before/after preview
        RenderConfigPanel();     // Bottom: selected stage configuration
    }

private:
    void RenderToolPalette() {
        // Tree view of tools by category
        if (ImGui::TreeNode("Text")) {
            if (ImGui::Selectable("Lowercase")) AddStage<LowercaseStage>();
            if (ImGui::Selectable("Remove Punctuation")) AddStage<RemovePunctuationStage>();
            // ...
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Image")) {
            if (ImGui::Selectable("Resize")) AddStage<ResizeStage>();
            if (ImGui::Selectable("Normalize")) AddStage<NormalizeStage>();
            // ...
            ImGui::TreePop();
        }
        // ... more categories
    }

    void RenderPipelineView() {
        // Vertical list of stages with drag-reorder
        for (size_t i = 0; i < pipeline_.StageCount(); i++) {
            auto& stage = pipeline_.GetStage(i);

            // Drag handle
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload("STAGE", &i, sizeof(size_t));
                ImGui::Text("Moving: %s", stage.GetName().c_str());
                ImGui::EndDragDropSource();
            }

            // Stage display
            ImGui::Selectable(stage.GetName().c_str(), selected_stage_ == i);

            // Delete button
            ImGui::SameLine();
            if (ImGui::SmallButton("X")) {
                pipeline_.RemoveStage(i);
            }
        }
    }

    void RenderPreviewPanel() {
        // Split view: before | after
        ImGui::Columns(2);

        ImGui::Text("Before");
        RenderDataPreview(current_sample_);

        ImGui::NextColumn();

        ImGui::Text("After");
        RenderDataPreview(preview_.output);

        ImGui::Columns(1);
    }
};
```

### Pipeline Serialization

Preprocessing pipelines are saved as JSON and can be:
1. Saved with project in `.cyxwiz` file
2. Exported as standalone `.cyxprep` files
3. Shared in model marketplace

```json
{
  "name": "Image Classification Preprocessing",
  "version": "1.0",
  "input_type": "image",
  "output_type": "tensor",
  "stages": [
    {
      "type": "Resize",
      "params": {
        "width": 224,
        "height": 224,
        "interpolation": "bilinear"
      }
    },
    {
      "type": "Normalize",
      "params": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
      }
    },
    {
      "type": "ToTensor",
      "params": {
        "dtype": "float32",
        "channel_order": "CHW"
      }
    }
  ]
}
```

---

## Script-Based Preprocessing (Advanced)

For advanced users who prefer code-based preprocessing, CyxWiz integrates with:

### DuckDB Integration

```python
import duckdb
import cyxwiz

# Load data from Asset Browser
con = duckdb.connect()
df = con.execute("""
    SELECT * FROM read_csv('project://datasets/sales.csv')
    WHERE amount > 0
""").fetchdf()

# SQL-based transformations
result = con.execute("""
    SELECT
        customer_id,
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as total_amount,
        COUNT(*) as order_count
    FROM df
    GROUP BY customer_id, month
    ORDER BY month
""").fetchdf()

# Register as dataset
cyxwiz.datasets.register("monthly_sales", result)
```

### Polars Integration

```python
import polars as pl
import cyxwiz

# Load data
df = pl.read_csv("project://datasets/transactions.csv")

# Polars transformations (much faster than pandas)
result = (
    df
    .filter(pl.col("amount") > 0)
    .with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("category").str.to_lowercase(),
        pl.col("amount").log().alias("log_amount")
    ])
    .group_by("category")
    .agg([
        pl.col("amount").mean().alias("avg_amount"),
        pl.col("amount").std().alias("std_amount"),
        pl.count().alias("count")
    ])
    .sort("avg_amount", descending=True)
)

# Save processed data
result.write_parquet("project://datasets/processed/category_stats.parquet")
```

### Custom Preprocessing Functions

```python
import cyxwiz
from cyxwiz.preprocessing import register_preprocessor

@register_preprocessor("custom_text_clean")
def clean_text(text: str, options: dict) -> str:
    """Custom text cleaning function"""
    # Remove extra whitespace
    text = " ".join(text.split())

    # Custom replacements
    replacements = options.get("replacements", {})
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# Now available in Preprocessing Module UI under "Custom" category
```

---

## Built-in Presets

### Image Classification

```yaml
name: ImageNet Preprocessing
stages:
  - Resize: {size: 256, mode: shortest_edge}
  - CenterCrop: {size: 224}
  - ToTensor: {}
  - Normalize: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
```

### Text Classification (BERT)

```yaml
name: BERT Preprocessing
stages:
  - Lowercase: {}
  - RemovePunctuation: {keep_sentence_endings: true}
  - Tokenize: {tokenizer: wordpiece, vocab: bert-base-uncased}
  - Truncate: {max_length: 512}
  - Pad: {max_length: 512, pad_token: "[PAD]"}
```

### Audio Classification

```yaml
name: Audio Mel Spectrogram
stages:
  - Resample: {target_sr: 16000}
  - MonoConvert: {}
  - TrimSilence: {top_db: 20}
  - PadOrTruncate: {target_length: 160000}
  - MelSpectrogram: {n_mels: 128, n_fft: 2048, hop_length: 512}
  - AmplitudeToDB: {}
  - Normalize: {method: minmax}
```

### Video Action Recognition

```yaml
name: Video RGB Frames
stages:
  - UniformSample: {num_frames: 16}
  - Resize: {size: [256, 256]}
  - CenterCrop: {size: 224}
  - Normalize: {mean: [0.45, 0.45, 0.45], std: [0.225, 0.225, 0.225]}
  - ToTensor: {channel_order: TCHW}
```

### Tabular AutoML

```yaml
name: Tabular Auto-Preprocessing
stages:
  - DetectTypes: {}  # Auto-detect column types
  - HandleMissing: {numeric: median, categorical: mode}
  - EncodeCategories: {method: target_encoding, min_samples: 10}
  - ScaleNumeric: {method: robust}
  - HandleOutliers: {method: iqr, factor: 1.5}
```

---

## Comparison with Other Systems

| Feature | CyxWiz | scikit-learn | TensorFlow | PyTorch | Hugging Face |
|---------|--------|--------------|------------|---------|--------------|
| Visual Pipeline Builder | ✅ | ❌ | ❌ | ❌ | ❌ |
| Text Preprocessing | ✅ | Limited | ✅ | Limited | ✅ |
| Image Preprocessing | ✅ | ❌ | ✅ | ✅ | ✅ |
| Audio Preprocessing | ✅ | ❌ | Limited | ✅ | ✅ |
| Video Preprocessing | ✅ | ❌ | Limited | ✅ | Limited |
| Tabular Preprocessing | ✅ | ✅ | Limited | Limited | Limited |
| Real-time Preview | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pipeline Export/Import | ✅ | ✅ | ✅ | Limited | ✅ |
| GPU Acceleration | ✅ | ❌ | ✅ | ✅ | ✅ |
| DuckDB/Polars Integration | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Implementation Phases

### Phase 1: Core Framework
- [ ] PreprocessingStage base class
- [ ] PreprocessingPipeline container
- [ ] JSON serialization
- [ ] Basic UI panel structure

### Phase 2: Text Tools
- [ ] Text cleaning tools (whitespace, punctuation, etc.)
- [ ] Tokenizers (word, sentence, BPE)
- [ ] Normalization tools
- [ ] Preview for text data

### Phase 3: Image Tools
- [ ] Basic transforms (resize, crop, rotate)
- [ ] Color adjustments
- [ ] Normalization
- [ ] Common augmentations
- [ ] Preview for image data

### Phase 4: Audio Tools
- [ ] FFmpeg integration
- [ ] Basic audio transforms
- [ ] Feature extraction (spectrogram, MFCC)
- [ ] Audio augmentation
- [ ] Preview for audio data

### Phase 5: Video Tools
- [ ] Frame extraction
- [ ] Video transforms
- [ ] Temporal features
- [ ] Video augmentation
- [ ] Preview for video data

### Phase 6: Tabular Tools
- [ ] Missing value handling
- [ ] Encoding methods
- [ ] Scaling/normalization
- [ ] Feature engineering
- [ ] Feature selection

### Phase 7: Advanced Features
- [ ] DuckDB integration
- [ ] Polars integration
- [ ] Custom preprocessor registration
- [ ] GPU-accelerated preprocessing
- [ ] Batch processing with progress
- [ ] Pipeline marketplace sharing

---

## Summary

The CyxWiz Preprocessing Module provides:

1. **Unified Interface** - Single visual tool for all data types
2. **Comprehensive Tools** - Industry-standard preprocessing operations
3. **Real-time Preview** - See transformations before applying
4. **Pipeline Management** - Save, load, share preprocessing pipelines
5. **Flexible Workflow** - Visual UI for beginners, scripting for advanced users
6. **Performance** - GPU acceleration where applicable
7. **Integration** - Seamless flow from Asset Browser to Training
