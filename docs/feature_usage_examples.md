# CyxWiz Engine — Feature Reference & Usage Examples

Comprehensive reference for all features in the CyxWiz Engine and `pycyxwiz` Python module.

---

## Table of Contents

1. [Neural Network Layers](#1-neural-network-layers)
2. [Activation Functions](#2-activation-functions)
3. [Loss Functions](#3-loss-functions)
4. [Optimizers](#4-optimizers)
5. [Learning Rate Schedulers](#5-learning-rate-schedulers)
6. [Text Tokenization](#6-text-tokenization)
7. [Upsampling Layers](#7-upsampling-layers)
8. [Time-Series Windowing](#8-time-series-windowing)
9. [Audio Processing](#9-audio-processing)
10. [RL Environment Interface](#10-rl-environment-interface)
11. [Dataset Manager](#11-dataset-manager)
12. [Node Editor & Code Generation](#12-node-editor--code-generation)
13. [Data Augmentation](#13-data-augmentation)
14. [Annotation System](#14-annotation-system)
15. [Model Export](#15-model-export)
16. [Python Scripting Console](#16-python-scripting-console)

---

## 1. Neural Network Layers

### Available Layers

| Layer | Parameters | Use Case |
|-------|-----------|----------|
| `DenseLayer` | in_features, out_features | Fully connected |
| `Conv2DLayer` | in_ch, out_ch, kernel, stride, padding | Image feature extraction |
| `Conv1DLayer` | in_ch, out_ch, kernel, stride, padding | Sequence feature extraction |
| `ConvTranspose2DLayer` | in_ch, out_ch, kernel, stride, padding | Decoder / upsampling |
| `MaxPool2DLayer` | kernel, stride | Spatial downsampling |
| `AvgPool2DLayer` | kernel, stride | Spatial downsampling (smooth) |
| `GlobalAvgPool2DLayer` | — | Reduce spatial to 1x1 |
| `FlattenLayer` | — | Reshape to 1D |
| `DropoutLayer` | rate | Regularization |
| `BatchNorm2DLayer` | num_features | Normalize activations |
| `LayerNormLayer` | normalized_shape | Transformer normalization |
| `InstanceNorm2DLayer` | num_features | Style transfer normalization |
| `GroupNormLayer` | num_groups, num_channels | Group normalization |
| `EmbeddingLayer` | vocab_size, embed_dim | Token → vector lookup |
| `LSTMLayer` | input_size, hidden_size, num_layers, bidirectional | Sequence modeling |
| `GRULayer` | input_size, hidden_size, num_layers, bidirectional | Sequence modeling (lighter) |
| `MultiHeadAttentionLayer` | embed_dim, num_heads | Self/cross attention |
| `TransformerEncoderLayer` | d_model, nhead, dim_ff | Full encoder block |
| `TransformerDecoderLayer` | d_model, nhead, dim_ff | Full decoder block |
| `Upsample2DLayer` | scale_factor, mode | Spatial upscaling |
| `PixelShuffleLayer` | upscale_factor | Sub-pixel convolution |

### Python Console

```python
import pycyxwiz as cyx

# Build a CNN classifier
model = cyx.Sequential()
model.add_conv2d(3, 32, kernel_size=3, stride=1, padding=1)
model.add_batchnorm2d(32)
model.add_relu()
model.add_maxpool2d(2, 2)
model.add_conv2d(32, 64, kernel_size=3, padding=1)
model.add_relu()
model.add_maxpool2d(2, 2)
model.add_flatten()
model.add_linear(64 * 8 * 8, 128)
model.add_relu()
model.add_dropout(0.5)
model.add_linear(128, 10)
model.add_softmax()

# Transformer encoder
embedding = cyx.EmbeddingLayer(vocab_size=30000, embed_dim=512)
encoder = cyx.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
```

### Node Editor

Right-click canvas → **Add Node** → browse categories:
- **Layers**: Dense, Conv2D, Conv1D, MaxPool2D, AvgPool2D, GlobalAvgPool, Flatten, Dropout
- **Normalization**: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
- **Sequence**: Embedding, LSTM, GRU, MultiHeadAttention, TransformerEncoder, TransformerDecoder
- **Upsampling**: ConvTranspose2D, Upsample, PixelShuffle

---

## 2. Activation Functions

| Activation | Description |
|-----------|-------------|
| ReLU | `max(0, x)` — default choice |
| LeakyReLU | Small slope for negatives |
| ELU | Exponential for negatives |
| GELU | Gaussian error — used in Transformers |
| Swish / SiLU | `x * sigmoid(x)` — modern networks |
| Mish | `x * tanh(softplus(x))` |
| Hardswish | Efficient approximation of Swish |
| SELU | Self-normalizing networks |
| PReLU | Learnable negative slope |
| Sigmoid | Output range (0, 1) |
| Tanh | Output range (-1, 1) |
| Softmax | Probability distribution |

```python
import pycyxwiz as cyx

relu = cyx.CreateActivation(cyx.ActivationType.ReLU)
gelu = cyx.CreateActivation(cyx.ActivationType.GELU)
leaky = cyx.CreateActivation(cyx.ActivationType.LeakyReLU, alpha=0.1)
```

---

## 3. Loss Functions

| Loss | Use Case |
|------|----------|
| MSE | Regression |
| L1 | Regression (robust to outliers) |
| SmoothL1 / Huber | Regression (hybrid L1/L2) |
| CrossEntropy | Multi-class classification |
| BinaryCrossEntropy | Binary classification |
| BCEWithLogits | BCE + Sigmoid (numerically stable) |
| NLLLoss | Negative log-likelihood |
| KLDivergence | Distribution matching (VAE) |
| CosineEmbedding | Similarity learning |
| Focal | Class-imbalanced classification |
| Triplet | Metric learning (anchor/positive/negative) |
| Contrastive | Similarity learning (pairs) |

```python
import pycyxwiz as cyx

# Classification
loss = cyx.CrossEntropyLoss()
output = loss.forward(predictions, targets)

# Class-imbalanced dataset
focal = cyx.FocalLoss(alpha=0.25, gamma=2.0)

# Metric learning
triplet = cyx.TripletLoss(margin=1.0, distance=cyx.DistanceType.Euclidean)

# Contrastive learning
contrastive = cyx.ContrastiveLoss(margin=1.0)
```

---

## 4. Optimizers

| Optimizer | Key Parameters |
|-----------|---------------|
| SGD | lr, momentum, weight_decay, nesterov |
| Adam | lr, beta1=0.9, beta2=0.999, eps=1e-8 |
| AdamW | lr, beta1, beta2, weight_decay=0.01 |
| RMSprop | lr, alpha=0.99, eps=1e-8 |
| AdaGrad | lr, eps=1e-10 |
| NAdam | lr, beta1, beta2 (Adam with Nesterov) |
| Adadelta | lr=1.0, rho=0.9 |
| LAMB | lr (layer-wise adaptive for large batches) |

```python
import pycyxwiz as cyx

optimizer = cyx.Adam(model.get_parameters(), lr=0.001)
# or
optimizer = cyx.AdamW(model.get_parameters(), lr=3e-4, weight_decay=0.01)
# or
optimizer = cyx.SGD(model.get_parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model.forward(x)
    loss_val = loss.forward(output, target)
    grad = loss.backward(output, target)
    model.backward(grad)
    optimizer.step(model.get_parameters(), model.get_gradients())
```

---

## 5. Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| StepLR | Decay by `gamma` every `step_size` epochs |
| ExponentialLR | Decay by `gamma` every epoch |
| CosineAnnealing | Cosine annealing with warm restarts |
| ReduceLROnPlateau | Reduce when metric stops improving |
| LinearWarmup | Linear warmup then decay |
| OneCycleLR | Super-convergence 1cycle policy |

```python
import pycyxwiz as cyx

scheduler = cyx.StepLR(optimizer, step_size=30, gamma=0.1)
# or
scheduler = cyx.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

for epoch in range(100):
    train_one_epoch()
    scheduler.step(epoch)
    print(f"LR: {scheduler.get_lr()}")
```

---

## 6. Text Tokenization

### Python Console

```python
import pycyxwiz as cyx

# Build vocabulary from documents
tok = cyx.Tokenizer(cyx.TokenizerType.Word)
tok.set_lowercase(True)
tok.set_max_length(128)
tok.set_padding(True)
tok.set_add_bos(True)
tok.set_add_eos(True)

documents = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "A bird flew over the house"
]
tok.train(documents, min_freq=1, max_vocab_size=10000)

# Encode single text
ids = tok.encode("the cat ran")
print(ids)  # [2, 4, 5, 12, 3, 0, 0, ...]  (BOS=2, EOS=3, PAD=0)

# Decode back to text
text = tok.decode(ids)
print(text)  # "the cat ran"

# Batch encode and pad
batch = tok.encode_batch(["hello world", "short"])
padded = tok.pad_batch(batch, max_length=10)

# Save/load vocabulary
vocab = tok.get_vocabulary()
vocab.save_to_file("my_vocab.txt")

# Tokenizer types
word_tok = cyx.Tokenizer(cyx.TokenizerType.Word)        # Split on word boundaries
ws_tok = cyx.Tokenizer(cyx.TokenizerType.Whitespace)    # Split on whitespace only
char_tok = cyx.Tokenizer(cyx.TokenizerType.Character)   # Character-level tokens
```

**Special tokens:** `[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3`

### Node Editor

1. Add **DatasetInput** node → load a text CSV file
2. Add **TextTokenizer** node → set type (Word/Whitespace/Character), max_length, lowercase
3. Add **TextPadding** node → connect to tokenizer output
4. Add **TextVocabulary** node → set vocab size
5. Connect to **Embedding** → LSTM/Transformer pipeline

---

## 7. Upsampling Layers

### Python Console

```python
import pycyxwiz as cyx

# ConvTranspose2D: learnable upsampling for decoders
deconv = cyx.ConvTranspose2DLayer(512, 256, kernel_size=4, stride=2, padding=1)
# Output size: (H-1)*stride - 2*padding + kernel + output_padding = 2*H

# Upsample2D: non-learnable spatial scaling
up_nearest = cyx.Upsample2DLayer(scale_factor=2, mode=cyx.UpsampleMode.Nearest)
up_bilinear = cyx.Upsample2DLayer(scale_factor=2, mode=cyx.UpsampleMode.Bilinear)

# PixelShuffle: rearrange channels to spatial (super-resolution)
# Input: [H, W, C*r^2, N] → Output: [H*r, W*r, C, N]
ps = cyx.PixelShuffleLayer(upscale_factor=2)
```

### Node Editor — U-Net Decoder

```
Encoder:  Input → Conv → Pool → Conv → Pool → Bottleneck
                                                    ↓
Decoder:  Output ← Conv ← ConvTranspose2D ← Conv ← ConvTranspose2D
```

1. Build encoder path: **Conv2D** → **MaxPool2D** → **Conv2D** → **MaxPool2D**
2. Add **ConvTranspose2D** nodes for decoder (kernel=4, stride=2, padding=1)
3. Or use **Upsample** + **Conv2D** (U-Net pattern)
4. For super-resolution: **Conv2D** (C*r^2 output channels) → **PixelShuffle**

---

## 8. Time-Series Windowing

### Python Console

```python
import pycyxwiz as cyx

# Single variable: stock prices, temperature, etc.
data = [100.0, 102.5, 101.3, 105.0, 103.2, 107.1, 106.5, 110.0, 108.3, 112.0]

config = cyx.TimeSeries.WindowConfig()
config.window_size = 3        # Use 3 past values as input
config.forecast_horizon = 1   # Predict 1 step ahead
config.stride = 1             # Step by 1 between windows

result = cyx.TimeSeries.create_windows(data, config)
print(f"Windows: {result.num_windows}")
# X[0] = [100.0, 102.5, 101.3] → y[0] = [105.0]
# X[1] = [102.5, 101.3, 105.0] → y[1] = [103.2]

# Add engineered features
config.lag_values = [1, 7]           # Lag-1 and lag-7 features
config.rolling_windows = [7]         # 7-period rolling mean/std
config.add_diff_features = True      # First-order differencing
result = cyx.TimeSeries.create_windows(data, config)

# Chronological split (no data leakage!)
train_end, val_end = cyx.TimeSeries.chronological_split(
    result.num_windows, train_ratio=0.7, val_ratio=0.15
)
# train: [0, train_end), val: [train_end, val_end), test: [val_end, end)

# Multivariate: multiple input columns
multi_data = [
    [100, 102, 101, 105, 103],  # price (target)
    [1000, 1200, 900, 1500, 1100],  # volume
    [0.5, 0.6, 0.4, 0.7, 0.3],  # sentiment
]
result = cyx.TimeSeries.create_multivariate_windows(
    multi_data, target_col=0, config=config
)
```

### Node Editor

1. **DatasetInput** → load CSV with time-series data
2. **TimeSeriesWindow** → set window_size, forecast_horizon, stride
3. **TimeSeriesFeatures** → enable lag, rolling mean/std, differencing
4. **TimeSeriesSplit** → chronological train/val/test split
5. Connect to **LSTM** / **GRU** → **Dense** → **Output**

```
DatasetInput → TimeSeriesWindow → TimeSeriesFeatures → TimeSeriesSplit → LSTM → Dense → Output
```

---

## 9. Audio Processing

### Python Console

```python
import pycyxwiz as cyx

# Load audio (WAV, FLAC, OGG, AIFF)
audio = cyx.AudioProcessing.load_audio("speech.wav", target_sr=16000)
print(f"Duration: {audio.duration_seconds:.2f}s, Samples: {audio.num_samples}")

# Spectrogram (frequency vs. time)
spec_cfg = cyx.SpectrogramConfig()
spec_cfg.n_fft = 512
spec_cfg.hop_length = 256
spec = cyx.AudioProcessing.compute_spectrogram(audio, spec_cfg)
print(f"Spectrogram: [{spec.rows} x {spec.cols}]")  # [257 x time_frames]

# Mel Spectrogram (perceptually weighted — best for ML)
mel_cfg = cyx.MelConfig()
mel_cfg.n_fft = 1024
mel_cfg.hop_length = 512
mel_cfg.n_mels = 80
mel_cfg.fmin = 0.0
mel_cfg.fmax = 8000.0
mel = cyx.AudioProcessing.compute_mel_spectrogram(audio, mel_cfg)
print(f"Mel: [{mel.rows} x {mel.cols}]")  # [80 x time_frames]

# MFCCs (compact features for speech)
mfcc_cfg = cyx.MFCCConfig()
mfcc_cfg.n_mfcc = 13
mfcc = cyx.AudioProcessing.compute_mfcc(audio, mfcc_cfg)
print(f"MFCC: [{mfcc.rows} x {mfcc.cols}]")  # [13 x time_frames]

# Audio augmentation
noisy = cyx.AudioProcessing.add_noise(audio, snr_db=20.0)
stretched = cyx.AudioProcessing.time_stretch(audio, rate=1.2)
shifted = cyx.AudioProcessing.pitch_shift(audio, semitones=2.0)
normalized = cyx.AudioProcessing.normalize(audio)
trimmed = cyx.AudioProcessing.trim_silence(audio, threshold_db=-40.0)
```

### Node Editor — Audio Classification Pipeline

```
AudioInput → MelSpectrogram → Conv2D → Pool → Conv2D → Pool → Flatten → Dense → Softmax
```

1. **AudioInput** node → set sample_rate=16000, max_duration
2. Feature extraction: **Spectrogram**, **MelSpectrogram**, or **MFCC** node
3. Optional: **AudioAugmentation** node (noise, stretch, pitch shift)
4. Connect features to **Conv2D** or **LSTM** classifier

---

## 10. RL Environment Interface

### Python Console

```python
import pycyxwiz as cyx

# Replay buffer for experience replay (DQN, SAC, etc.)
buffer = cyx.ReplayBuffer(capacity=50000, seed=42)

# Simulate collecting transitions
import random
state = [0.1, 0.2, 0.3, 0.4]
for step in range(1000):
    action = [random.randint(0, 1)]
    reward = random.random()
    next_state = [s + random.gauss(0, 0.01) for s in state]
    done = random.random() < 0.01

    buffer.push(state, action, reward, next_state, done)
    state = next_state if not done else [0.1, 0.2, 0.3, 0.4]

print(f"Buffer size: {len(buffer)}")  # 1000

# Sample a training batch
if buffer.can_sample(32):
    batch = buffer.sample(32)
    print(f"States: {len(batch.states)}, Rewards: {batch.rewards[:5]}")

# Epsilon-greedy exploration schedule
eps = cyx.EpsilonSchedule(start=1.0, end=0.01, decay_steps=10000)
for i in range(10000):
    eps.step()
print(f"Final epsilon: {eps.epsilon:.3f}")  # ~0.01
```

### GymConnector (C++ Engine Bridge)

```cpp
// Connect to OpenAI Gymnasium via embedded Python
auto gym = std::make_unique<cyxwiz::GymConnector>(&scripting_engine);

gym->CreateEnv("CartPole-v1");
auto info = gym->GetEnvInfo();
// info.observation_dim=4, info.num_actions=2, info.discrete_actions=true

auto obs = gym->Reset(/*seed=*/42);
float total_reward = 0;
bool done = false;

while (!done) {
    int action = SelectAction(obs);  // Your policy
    auto result = gym->Step(action);
    obs = result.observation;
    total_reward += result.reward;
    done = result.done || result.truncated;
}

// For continuous action spaces:
auto result = gym->StepContinuous({0.5f, -0.3f});
```

### Node Editor — DQN Pipeline

```
GymEnvironment → ReplayBuffer → PolicyNetwork → Loss → Optimizer
                                  ↑
                         ValueNetwork (target)
```

1. **GymEnvironment** → env name (e.g., "CartPole-v1")
2. **ReplayBuffer** → capacity, batch_size
3. **PolicyNetwork** / **ValueNetwork** → Dense layers
4. **RLTraining** → algorithm (DQN/PPO), gamma, epsilon schedule

---

## 11. Dataset Manager

### Supported Dataset Types

| Type | Format |
|------|--------|
| CSV | `.csv` comma-separated values |
| TSV | `.tsv` tab-separated values |
| JSON | `.json` data files |
| TXT | `.txt` plain text |
| ImageFolder | Directory of images (class per subfolder) |
| ImageCSV | Images + CSV label file |
| MNIST | Standard MNIST format |
| FashionMNIST | Fashion-MNIST format |
| CIFAR-10 | CIFAR-10 binary format |
| CIFAR-100 | CIFAR-100 binary format |
| HDF5 | `.h5`, `.hdf5` hierarchical data |
| ARFF | `.arff` Weka format |
| HuggingFace | HuggingFace datasets |
| Kaggle | Kaggle datasets |
| Custom | User-defined schema |

### Dataset Manager UI (3-Pane Layout)

```
+------------------------------------------------------------------+
| TOOLBAR: [Refresh] [Memory Bar] [Search]               [Settings]|
+----------+-------------------------------------------------------+
| SIDEBAR  | TABS: Preview | Prepare | Pipeline | Training |       |
+----------+       Evaluate | Export | Details                      |
| Dataset  |-------------------------------------------------------+
| Tree     |                                                       |
| - mnist  |  [Active Tab Content]                                 |
|   train  |                                                       |
|   val    |  Each tab provides focused functionality:              |
|   test   |  Preview: sample images, statistics, class distribution|
|          |  Prepare: data cleaning, normalization, encoding       |
+----------+  Pipeline: augmentation configuration                 +
| STATUS BAR: Ready | mnist (60,000 samples) | Memory: 2.1 GB      |
+------------------------------------------------------------------+
```

### Tabs

| Tab | Features |
|-----|----------|
| **Preview** | Sample grid, class distribution chart, basic stats |
| **Prepare** | Missing value handling, normalization, feature encoding |
| **Pipeline** | Augmentation preset selection, custom transform chain |
| **Training** | Local training with real-time loss/accuracy plots |
| **Evaluate** | Confusion matrix, regression metrics, per-class accuracy |
| **Export** | Model export (CyxModel, ONNX, Safetensors, GGUF) |
| **Details** | Dataset metadata, file paths, memory usage |

### Analytics Features

- Class distribution bar charts
- Color/grayscale histograms
- Brightness/contrast statistics
- Outlier detection (IQR method)
- Quality analysis (blur, noise, exposure)
- Duplicate detection (pHash, aHash, dHash)

---

## 12. Node Editor & Code Generation

### Adding Nodes

- **Right-click** canvas → Add Node menu (categorized)
- **Ctrl+Space** → Quick search (type to filter by name or keyword)

### Node Categories

| Category | Nodes |
|----------|-------|
| Input/Output | DatasetInput, Output |
| Data Pipeline | DataLoader, Augmentation, DataSplit, TensorReshape, Normalize, OneHotEncode |
| Text Processing | TextTokenizer, TextVocabulary, TextPadding |
| Layers | Dense, Conv2D, Conv1D, MaxPool2D, AvgPool2D, GlobalAvgPool, Flatten, Dropout |
| Normalization | BatchNorm, LayerNorm, InstanceNorm, GroupNorm |
| Sequence | Embedding, LSTM, GRU, MultiHeadAttention, TransformerEncoder, TransformerDecoder |
| Upsampling | ConvTranspose2D, Upsample, PixelShuffle |
| Activations | ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax, Swish, Mish |
| Arithmetic | Add (residual connections) |
| Loss | CrossEntropyLoss, MSELoss, FocalLoss, TripletLoss |
| Optimizers | SGD, Adam, AdamW, RMSprop |
| Time-Series | TimeSeriesWindow, TimeSeriesFeatures, TimeSeriesSplit |
| Audio | AudioInput, Spectrogram, MelSpectrogram, MFCC, AudioAugmentation |
| RL | GymEnvironment, ReplayBuffer, PolicyNetwork, ValueNetwork, RLTraining |

### Code Generation

Select a framework from the toolbar and click **Generate Code**:

| Target | Output |
|--------|--------|
| **PyTorch** | `torch.nn.Module` class with `forward()` method |
| **TensorFlow** | `tf.keras.Model` subclass |
| **Keras** | `tf.keras.Sequential` or Functional API |
| **PyCyxWiz** | Native `pycyxwiz` script using CyxWiz backend |

Generated code is displayed in the Script Editor panel and can be saved as `.py`.

### Subgraphs

Select multiple nodes → Right-click → **Create Subgraph** to group repeating blocks (e.g., ResNet blocks, encoder layers).

---

## 13. Data Augmentation

### Preset Augmentation Pipelines

Apply via the **Pipeline** tab in Dataset Manager or the **Augmentation** node:

| Preset | Transforms | Best For |
|--------|-----------|----------|
| ImageNet | RandomCrop(224), HorizontalFlip, ColorJitter, Normalize | General image classification |
| CIFAR-10 | RandomCrop(32,pad=4), HorizontalFlip, Normalize | Small image classification |
| Medical | ElasticDeform, RandomRotate(15), GaussianBlur | Medical imaging |
| Satellite | RandomRotate(90), VerticalFlip, HorizontalFlip | Remote sensing |
| OCR | RandomAffine, GaussianNoise, Erode/Dilate | Text recognition |
| Self-Supervised | ColorJitter(strong), GaussianBlur, RandomGrayscale | SimCLR/BYOL pretraining |
| Detection | RandomCrop, HorizontalFlip, ColorJitter, RandomScale | Object detection |
| Segmentation | RandomCrop, RandomRotate, HorizontalFlip | Semantic segmentation |

### Audio Augmentation

| Transform | Parameters |
|-----------|-----------|
| AddNoise | snr_db (signal-to-noise ratio) |
| TimeStretch | rate (playback speed multiplier) |
| PitchShift | semitones (pitch change) |
| Normalize | Peak normalize to [-1, 1] |
| TrimSilence | threshold_db (silence threshold) |

---

## 14. Annotation System

### Features

- **Batch navigation**: Prev/Next/Go-to for dataset images
- **Class management**: Add/select/remove class labels
- **Annotation tools**: Polygon, bounding box, brush mask
- **Per-image annotations**: View, select, delete
- **Export formats**: COCO JSON, YOLO TXT, Pascal VOC XML

### Python API

```python
# In the engine (C++ API)
ann_mgr = DataRegistry.Instance().GetAnnotationManager()

# Export annotations
ann_mgr.ExportCOCO("my_dataset", "output/coco.json")    # For Detectron2, MMDet
ann_mgr.ExportYOLO("my_dataset", "output/yolo/")        # For YOLOv5/v8
ann_mgr.ExportVOC("my_dataset", "output/voc/")           # For Pascal VOC tools
```

### Training with Annotations

```cpp
// C++ — get segmentation masks for training
DatasetBatcher batcher(dataset, 32, DatasetSplit::Train);
while (batcher.HasNext()) {
    AnnotatedBatch batch = batcher.GetNextAnnotatedBatch("my_dataset");
    // batch.images: [B, H, W, C] — input images
    // batch.masks:  [B, H, W]    — segmentation masks (class IDs per pixel)
}
```

---

## 15. Model Export

### Supported Export Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CyxModel | `.cyx` | Native CyxWiz format (full model + optimizer state) |
| ONNX | `.onnx` | Open Neural Network Exchange (interop with PyTorch, TF, etc.) |
| Safetensors | `.safetensors` | HuggingFace safe tensor serialization |
| GGUF | `.gguf` | GGML format for llama.cpp inference |

Export via the **Export** tab in Dataset Manager after training, or via code:

```python
import pycyxwiz as cyx

model.save("model.cyx")           # Native format
model.export_onnx("model.onnx")   # ONNX export
```

---

## 16. Python Scripting Console

The engine includes an embedded Python interpreter (via pybind11) accessible from the **Console** panel.

### Quick Start

```python
import pycyxwiz as cyx

# Check backend
print(cyx.get_device_info())

# Initialize with GPU
cyx.initialize()
cyx.set_device(0)  # GPU 0

# Full training example
model = cyx.Sequential()
model.add_linear(784, 256)
model.add_relu()
model.add_linear(256, 10)
model.add_softmax()

optimizer = cyx.Adam(model.get_parameters(), lr=0.001)
loss_fn = cyx.CrossEntropyLoss()

model.train()
for epoch in range(10):
    # Forward
    output = model.forward(x_batch)
    loss_val = loss_fn.forward(output, y_batch)

    # Backward
    grad = loss_fn.backward(output, y_batch)
    model.backward(grad)

    # Update
    optimizer.step(model.get_parameters(), model.get_gradients())
    optimizer.zero_grad()

    print(f"Epoch {epoch}: loss={loss_val:.4f}")

model.eval()
model.save("trained_model.cyx")
```

---

## End-to-End Examples

### Example 1: Image Classification (CIFAR-10)

**Node Editor:**
```
DatasetInput(CIFAR-10) → Augmentation(CIFAR preset)
    → Conv2D(3→32) → BatchNorm → ReLU → MaxPool
    → Conv2D(32→64) → BatchNorm → ReLU → MaxPool
    → Flatten → Dense(1024→256) → ReLU → Dropout(0.5)
    → Dense(256→10) → Softmax
    → CrossEntropyLoss ← OneHotEncode ← DatasetInput
    → Adam(lr=0.001)
```

### Example 2: Text Sentiment Analysis

**Node Editor:**
```
DatasetInput(CSV) → TextTokenizer(Word, max=128, lowercase)
    → TextPadding(128) → Embedding(10000→128)
    → LSTM(128, hidden=64, bidirectional=True)
    → Dense(128→64) → ReLU → Dense(64→2) → Softmax
    → CrossEntropyLoss → AdamW(lr=3e-4)
```

### Example 3: Stock Price Forecasting

**Node Editor:**
```
DatasetInput(CSV) → TimeSeriesWindow(window=30, horizon=7)
    → TimeSeriesFeatures(lags=[1,7,30], rolling=[7,14], diff=True)
    → TimeSeriesSplit(train=0.7, val=0.15)
    → LSTM(features→64, layers=2) → Dense(64→7)
    → MSELoss → Adam(lr=0.001)
```

### Example 4: Audio Classification

**Node Editor:**
```
AudioInput(sr=16000, max_dur=4.0) → MelSpectrogram(n_mels=80)
    → AudioAugmentation(noise, stretch)
    → Conv2D(1→32) → MaxPool → Conv2D(32→64) → MaxPool
    → Flatten → Dense(→128) → ReLU → Dense(→num_classes) → Softmax
    → CrossEntropyLoss → Adam(lr=0.001)
```

### Example 5: Semantic Segmentation (U-Net)

**Node Editor:**
```
DatasetInput(ImageFolder) → Augmentation(Segmentation preset)
    Encoder: Conv2D(3→64) → Conv2D(64→64) → MaxPool
           → Conv2D(64→128) → Conv2D(128→128) → MaxPool
    Bottleneck: Conv2D(128→256) → Conv2D(256→256)
    Decoder: ConvTranspose2D(256→128) → Conv2D(256→128) → Conv2D(128→128)
           → ConvTranspose2D(128→64) → Conv2D(128→64) → Conv2D(64→num_classes)
    → CrossEntropyLoss → Adam(lr=0.001)
```
