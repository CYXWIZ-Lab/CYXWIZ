# OpenML Dataset Analysis for CyxWiz Engine

Analysis of the 48 most popular OpenML datasets mapped against CyxWiz Engine capabilities.

**Source:** [OpenML Dataset Search (sorted by runs)](https://www.openml.org/search?type=data&sort=runs&status=active)
**Reference:** [48 Most Popular Open ML Datasets - Daily Dose of DS](https://blog.dailydoseofds.com/p/48-most-popular-open-ml-datasets)

---

## Fully Supported (Build Now)

These datasets can be loaded, modeled, trained, and evaluated with the current engine:

| Dataset | Task | CyxWiz Approach |
|---------|------|-----------------|
| **MNIST** | Digit Classification | Built-in loader, Conv2D + Dense, CrossEntropyLoss, full training pipeline |
| **CIFAR-10/100** | Image Classification | Built-in loader, Conv2D stacks + BatchNorm + Dropout, 13 augmentation presets |
| **Iris** | 3-class Classification | CSV loader, Dense layers, CrossEntropyLoss |
| **UCI Adult** | Income Classification | CSV/Parquet loader, Dense + Embedding, BCEWithLogitsLoss |
| **Wine Quality** | Classification/Regression | CSV loader, Dense network, MSE or CrossEntropy |
| **Titanic** | Survival Classification | CSV loader, Dense + Dropout, BCELoss |
| **California Housing** | Regression | CSV loader, Dense layers, MSELoss or SmoothL1Loss |
| **Diabetes** | Regression | CSV loader, Dense network, MSELoss |
| **IMDb Reviews** | Sentiment Classification | Embedding + LSTM/GRU (bidirectional), BCEWithLogitsLoss |
| **ImageNet** | Image Classification | Image loader, deep Conv2D networks, augmentation presets (ImageNet preset exists) |
| **PASCAL VOC** | Detection/Segmentation | Image loader + annotation system (COCO/YOLO/VOC export), Conv2D + pretrained YOLO nodes |
| **COCO** | Detection/Segmentation | AnnotationManager with COCO JSON import, DNNDetect nodes, FocalLoss for class imbalance |

**Key strengths:** Tabular data (CSV/Parquet/HDF5), image classification, object detection via DNN nodes, semantic segmentation via annotation system.

---

## Partially Supported (Need Some Work)

| Dataset | Task | What Works | What's Missing |
|---------|------|------------|----------------|
| **COCO Captions** | Image Captioning | TransformerDecoder, Embedding, attention layers all exist | No image-text pairing pipeline, no beam search decoding |
| **VQA v2.0** | Visual QA | MultiHeadAttention, Embedding, Conv2D | No multimodal fusion node, no answer vocabulary handling |
| **SQuAD** | Question Answering | TransformerEncoder, Embedding, PositionalEncoding | No tokenizer integration, no span extraction head |
| **CoNLL-2003** | NER | LSTM/GRU + Dense, Embedding | No CRF layer, no BIO tag handling |
| **WikiText-103** | Language Modeling | Full Transformer stack (encoder+decoder), Embedding | No text tokenizer, no autoregressive generation |
| **Rossmann Sales** | Time Series | LSTM/GRU layers exist | No time-series specific data loader or windowing |
| **Cityscapes** | Semantic Segmentation | Annotation system exists, Conv2D | No upsampling/transposed conv node for U-Net style architecture |

---

## Not Currently Feasible

| Dataset | Task | Why |
|---------|------|-----|
| **OpenAI Gym / Atari / D4RL** | Reinforcement Learning | No RL environment integration, no policy gradient or Q-learning |
| **CARLA / MineRL** | RL Simulation | No simulator interface |
| **LAION-5B** | Contrastive Learning | Scale too large, no CLIP-style dual encoder pipeline |
| **AudioSet** | Audio Classification | No audio data loader or spectrogram processing |
| **HowTo100M / MovieQA** | Video Understanding | No video frame extraction or temporal modeling |
| **Criteo 1TB** | Click Prediction | Scale requires distributed training (P2P exists but not for data-parallel) |

---

## Recommended Build Projects

### 1. Image Classification Pipeline
- **Datasets:** MNIST, CIFAR-10, ImageNet
- **Architecture:** DataInput -> Augmentation -> Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> (repeat) -> Flatten -> Dense -> Softmax
- **Loss:** CrossEntropyLoss + Adam optimizer + CosineAnnealing scheduler

### 2. Tabular ML Suite
- **Datasets:** Titanic, Adult, Wine, California Housing, Diabetes, Iris
- **Architecture:** DataInput -> Normalize -> Dense -> ReLU -> Dropout -> Dense -> Output
- **Use FocalLoss** for imbalanced datasets (Adult, Titanic)

### 3. Object Detection with Pretrained Models
- **Datasets:** PASCAL VOC, COCO
- **Architecture:** Use PretrainedYOLO / DNNDetect nodes for inference, AnnotationManager for labeling new data

### 4. Sentiment Analysis (NLP)
- **Datasets:** IMDb Reviews
- **Architecture:** DataInput -> Embedding -> LSTM(bidirectional=true) -> Dense -> Sigmoid
- **Loss:** BCEWithLogitsLoss + AdamW

### 5. Metric Learning
- **Datasets:** Any image dataset with classes
- **Architecture:** Conv2D encoder -> TripletLoss or ContrastiveLoss
- **Use case:** Image similarity search, face verification

---

## Readiness Summary

| Category | Datasets | CyxWiz Readiness |
|----------|----------|-----------------|
| **Tabular Classification/Regression** | 6 datasets | Ready |
| **Image Classification** | 4 datasets | Ready |
| **Object Detection/Segmentation** | 3 datasets | Ready (inference + annotation) |
| **Sentiment/Text Classification** | 2 datasets | Ready (with Embedding+RNN) |
| **Sequence Modeling (NLP)** | 4 datasets | Partial (layers exist, no tokenizer) |
| **Multimodal** | 4 datasets | Partial (components exist, no fusion pipeline) |
| **Reinforcement Learning** | 8 datasets | Not supported |
| **Audio/Video** | 3 datasets | Not supported |

**Bottom line:** ~15 of the 48 popular datasets are fully buildable today. Adding a text tokenizer node would unlock another ~6 NLP datasets.

---

## Gap Analysis: What to Build Next

Priority features to expand dataset coverage:

1. **Text Tokenizer Node** - Unlocks SQuAD, WikiText, CoNLL, GLUE (6 datasets)
2. **Transposed Conv / Upsample Node** - Unlocks U-Net for Cityscapes segmentation
3. **Time-Series Windowing** - Unlocks Rossmann and similar forecasting datasets
4. **Audio Spectrogram Loader** - Unlocks AudioSet
5. **RL Environment Interface** - Unlocks 8 RL datasets (large effort)
