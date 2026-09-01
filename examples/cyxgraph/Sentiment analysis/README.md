# CyxGraph Examples

Example graph files (`.cyxgraph`) for CyxWiz Engine.

## Files

| File | Description |
|------|-------------|
| `mnist_mlp.cyxgraph` | End-to-end MLP classifier for `mnist_784.csv`. Loads, normalizes, splits, batches, and trains. |
| `mnist_classifier.cyxgraph` | CNN-based MNIST digit classifier (legacy - may need updating). |
| `sentiment_analysis_gru_classifier.cyxgraph` | Text sentiment classifier for `D:/demo/mrcj/datasets/sentiment analysis/sentiment_mental_health.csv`. Uses text tokenization, vocabulary building, padding, Embedding, and GRU layers. |
| `sentiment_analysis_inference.py` | Local inference helper for the sentiment graph. Uses the prep metadata/vocab file and calls the embedded predict endpoint. |
| `sentiment_analysis_tfidf_mlp_classifier.cyxgraph` | Canonical unigram TF-IDF benchmark graph. |
| `sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph` | Canonical unigram+bigram candidate graph. |
| `benchmark_sentiment_text_features.py` | Reproducible sklearn feature-isolation benchmark for the two TF-IDF configurations. |
| `sentiment_text_feature_benchmark.json` | Recorded full-dataset reference result, including dataset hash and limitations. |

## Canonical graph ownership

The `.cyxgraph` files directly in this directory are the tracked source of
truth. A nested `sentiment/` directory is a local Studio project workspace; it
may contain copied graphs, environments, caches, and checkpoints and is ignored
by Git. Promote an intentional graph edit back to the canonical top-level file
instead of committing a second copy.

## Sentiment vocab path

The sentiment graph points its `TextVocabulary` node at the dataset-side 10k
vocab file:

`D:/demo/mrcj/datasets/sentiment analysis/sentiment_analysis_vocab.txt`

Generate it with:

```bash
python examples/cyxgraph/Sentiment analysis/prepare_sentiment_analysis_demo.py
```

That script now writes the vocab and metadata into the dataset folder by
default so the graph, prep step, and inference helper all read the same files.

## MNIST MLP (recommended for verifying end-to-end training)

### What the pipeline contains

```text
DataInput (mnist_784.csv, label=class)
  -> Data -> Normalize(mean=0, std=255) -> Dense(128) -> ReLU
  -> Dense(64) -> ReLU
  -> Dense(10) -> CrossEntropyLoss -> Adam -> Output
  -> Labels

DataSplit (80/10/10, seed=42)     [config node, read at training start]
DataLoader (batch=128, shuffle)   [config node, read at training start]
```

The `DataSplit` and `DataLoader` nodes are config nodes: `GraphCompiler` reads their parameters at training start. They don't have to be physically linked into the data flow.

### Parameters (editable when applying the pattern)

| Param | Default | Notes |
|-------|---------|-------|
| `file_path` | `D:/demo/mrcj/datasets/mnist_784.csv` | Absolute path to the CSV file. |
| `batch_size` | `128` | DataLoader batch size. |
| `hidden_units` | `128` | First hidden layer width. |
| `learning_rate` | `0.001` | Adam learning rate. |
| `epochs` | `10` | Training epochs (read from optimizer node for now). |

### How to run it

1. Open CyxWiz Engine.
2. File -> Open -> `examples/cyxgraph/mnist_mlp.cyxgraph`. The graph loads onto the canvas.
3. Double-click the **MNIST CSV** (DataInput) node -> confirm `file_path` points at your copy of `mnist_784.csv` -> click **Apply**. This loads the dataset into `DataRegistry` under the name `mnist_784`. You should see "Applied" feedback and row/column counts.
4. Optionally, double-click **DataSplit** and **DataLoader** to review or tweak the split ratios and batch size.
5. Hit **Train**. The training dashboard should open and show loss curves.

Expected: loss drops from ~2.3 (random) toward <0.1, accuracy climbs above 95% within a few epochs.

### What `mnist_784.csv` should look like

- 785 columns, CSV header row: `"pixel1","pixel2",...,"pixel784","class"`
- 70,000 data rows (60k train + 10k test in the standard split - we reshuffle 80/10/10 here)
- Pixel values are integers in `[0, 255]`. The Normalize node divides by 255 to scale into `[0, 1]`.
- `class` column is an integer digit in `[0, 9]`.

## MNIST Classifier (CNN - legacy)

```text
DataInput -> Split -> Normalize -> Conv2D(32) -> ReLU -> MaxPool
         -> Conv2D(64) -> ReLU -> MaxPool -> Flatten
         -> Dense(128) -> ReLU -> Dropout(0.5) -> Dense(10)
         -> CrossEntropyLoss -> Adam -> Output
```

Not yet updated for the new DataLoader/DataSplit wiring. Use the MLP version until this is refreshed.
## Model choice: LSTM vs TF-IDF + Dense MLP

The current supported neural sequence baseline is:

```text
TextTokenizer
-> Embedding
-> single-direction LSTM
-> Dense classifier
-> CrossEntropy
-> AdamW
```

For the GPU-fit LSTM profile, use conservative dimensions first:

```text
max_length=64
embedding_dim=64
hidden_size=32
batch_size=64
```

This reduces ArrayFire/CUDA generated-kernel pressure. Larger profiles such as
`max_length=128`, `embedding_dim=128`, and `hidden_size=64` may train, but the
compiler can CPU-route the LSTM when the generated CUDA kernel formal-parameter
estimate exceeds the current safe limit.

If the goal is best practical sentiment accuracy, the stronger baseline is
usually:

```text
DataInput
-> TFIDFVectorizer
-> DataSplit
-> DataLoader
-> Dense 256
-> ReLU
-> Dropout
-> Dense 128
-> ReLU
-> Dropout
-> Dense 7
-> CrossEntropy
-> AdamW
```

### What TFIDFVectorizer does

`TFIDFVectorizer` turns raw text into normal numeric feature columns that a Dense
network can train on directly.

It does two things:

1. It builds a vocabulary from the text column.
2. It converts every row of text into a fixed-size vector of word-importance
   scores.

TF-IDF means **term frequency - inverse document frequency**.

Term frequency asks:

```text
How often does this word appear in this document?
```

Inverse document frequency asks:

```text
How rare is this word across the whole dataset?
```

The combined score is high when a word appears in the current document but is
not common everywhere. That makes it useful for classification.

Example:

```text
Text: "I feel anxious and hopeless every night"
```

Common words such as `I`, `and`, `every` usually get low value. More
class-informative words such as `anxious`, `hopeless`, or `night` can get higher
value if they help distinguish one label from another.

A vectorized row may conceptually become:

```text
tfidf_anxious=0.42
tfidf_hopeless=0.51
tfidf_sleep=0.00
tfidf_angry=0.00
tfidf_happy=0.00
...
y=<class id>
```

In the actual engine schema, the columns are usually named generically:

```text
tfidf_0, tfidf_1, tfidf_2, ..., tfidf_N, y
```

`max_features` controls how many TF-IDF columns are kept. For example,
`max_features=2000` means each text row becomes a 2000-feature numeric vector.

### Why Dense MLP works well after TF-IDF

After TF-IDF, the model is no longer processing a token sequence. It is
processing a regular numeric feature vector.

That means the classifier can be a standard feed-forward network:

```text
Dense 256 -> ReLU -> Dropout -> Dense 128 -> ReLU -> Dense 7
```

The first Dense layer learns combinations of useful words. For example, it can
learn that groups of terms related to anxiety, sleep, depression, stress, or
positive affect push the prediction toward different classes.

This often works very well for sentiment/status classification because many
labels are strongly tied to discriminative words and short phrases.

### Why TF-IDF can beat LSTM/GRU here

LSTM and GRU learn from ordered token sequences:

```text
word_1 -> word_2 -> word_3 -> ... -> prediction
```

That is powerful, but in the current engine recurrent CUDA placement is still
limited. Large recurrent shapes may route to CPU because ArrayFire generated CUDA
kernels can exceed the formal-parameter limit.

TF-IDF avoids that recurrent bottleneck:

```text
raw text -> fixed numeric vector -> Dense GPU-friendly classifier
```

For this dataset, TF-IDF + Dense MLP may be the most pragmatic route toward an
80% accuracy target because it is simpler, faster, and easier to optimize.

Tradeoffs:

- LSTM can model token order and context, but recurrent CUDA placement is still
  constrained in the current engine.
- GRU is conservatively CPU-routed until a GRU-specific JIT probe or fused/native
  recurrent CUDA path is implemented.
- TF-IDF + Dense MLP ignores most word order, but trains faster and avoids the
  recurrent-kernel bottleneck.
- For an 80% accuracy target, try TF-IDF + Dense MLP on a build where the
  training compiler maps `TFIDFVectorizer.max_features` to the model input
  width and the Arrow materializer runs the TF-IDF operator before batching.

Recommended workflow:

1. Use the GPU-fit LSTM graph to validate the current neural text path.
2. Watch compile placement for `cuda_jit_param_overflow_risk`.
3. If LSTM remains CPU-routed or validation accuracy stalls, switch to the
   TF-IDF + Dense MLP architecture.
4. Keep validation/test data unresampled; class balancing belongs to the
   DataLoader training sampler only.


