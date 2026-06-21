# CyxGraph Examples

Example graph files (`.cyxgraph`) for CyxWiz Engine.

## Files

| File | Description |
|------|-------------|
| `mnist_mlp.cyxgraph` | End-to-end MLP classifier for `mnist_784.csv`. Loads, normalizes, splits, batches, and trains. |
| `mnist_classifier.cyxgraph` | CNN-based MNIST digit classifier (legacy - may need updating). |
| `sentiment_analysis_gru_classifier.cyxgraph` | Text sentiment classifier for `D:/demo/mrcj/datasets/sentiment analysis/sentiment_mental_health.csv`. Uses text tokenization, vocabulary building, padding, Embedding, and GRU layers. |
| `sentiment_analysis_inference.py` | Local inference helper for the sentiment graph. Uses the prep metadata/vocab file and calls the embedded predict endpoint. |

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
