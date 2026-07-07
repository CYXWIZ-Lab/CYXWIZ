## 10) Data-domain paths (from node graph to runtime behavior)

### 10.1 image domain
Observed constraints and behavior:
- image preprocessing nodes must resolve required transforms (e.g. resize where needed),
- image flattening expectations before dense transitions when required by model design,
- input shape expectations validated with compile-level output checks.

### 10.2 text domain
- tokenizer / padding / vectorizer nodes can map into preprocessing contracts,
- tokenizer-related nodes participate in materialization metadata.

### 10.3 sequence/time-series domain
- time-series window behavior can override default split/feature extraction,
- sequence dataset and split behavior is handled through dedicated branch in launcher/executor.

### 10.4 audio domain
- audio transforms and spectrogram-style nodes are represented,
- runtime behavior currently constrained by capability matrix and compile-time compatibility.

### 10.5 legacy / unsupported domains
- Many nodes are UI-visible but not yet in stable training/compiler support.
- Unsupported nodes should be explicitly flagged before training and/or materialization.

---
