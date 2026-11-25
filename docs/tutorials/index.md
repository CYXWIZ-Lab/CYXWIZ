---
title: Tutorials
description: A six-lesson learning path from the main window to an exported model, then lessons on text, audio, time series, Python and more.
type: tutorial
area: general
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - ../install/index.md
  - ../main-window/start-page.md
  - ../user-guide/graph-editor/index.md
---

These lessons teach the CyxWiz Engine by doing. Each one runs start to
finish on a fresh install with a graph and a small dataset from the
`examples` folder, and each ends with links to the guides that go further.
Take them in order the first time.

## The learning path

1. [Tour the main window](tour-the-main-window.md) (10 minutes): create a
   project and learn where the canvas, the panels, the menus and the status
   bar are.
2. [Train your first model from a starter graph](first-model-from-a-starter.md)
   (15 minutes): open the **Multiclass image classification** starter, look
   at each node, compile it, train it and read the Training Dashboard.
3. [Bring in and prepare your own data](prepare-your-own-data.md)
   (20 minutes): load a table into Data Studio, profile it, fill and drop
   its missing values, and check it.
4. [Build a graph yourself](build-a-graph-yourself.md) (20 minutes): find,
   place and connect the nodes of a classifier, set their properties and
   compile it.
5. [Train, evaluate and understand the results](train-and-evaluate.md)
   (20 minutes): test a model on rows it never saw, read the confusion
   matrix and the classes, and see which inputs drove a prediction.
6. [Save, export and use a model](save-export-and-use.md) (15 minutes):
   save the graph, export the trained model, write it as PyTorch code, and
   test it again in a later session.

## Lessons by kind of data

Each of these opens another task starter of the start page and explains
what is particular to its data. Take lesson 2 first.

- [Classify text](text-classification.md): review sentences, a tokenizer
  and an embedding (**Text classification** starter).
- [Classify audio](audio-classification.md): spoken digits from their mel
  spectrogram (**Audio classification** starter).
- [Train a time-series forecaster](time-series-forecasting.md): monthly airline
  passengers, windows and a chronological split (**Time-series
  forecasting** starter).

## Python

- [Train a model from the Script Editor](python-in-the-script-editor.md):
  the digits network of lesson 2, written with `pycyxwiz` and run inside
  the Engine.

## More lessons

These go further than the learning path. Take lessons 1 to 5 first.

- [Find customer segments with analysis nodes](analysis-demo-graph.md)
  (10 minutes): scale a customer table, project it with PCA, and compare
  K-Means and DBSCAN with the true segments on three plots.
- [Train a small ResNet image classifier from scratch](image-classifier-resnet.md):
  residual blocks, image steps once and per batch, label smoothing and a
  test score. Needs your own two-class image folder and, in practice, a GPU.
- [Compare two TF-IDF text classifiers](sentiment-text-classification.md):
  unigrams against unigrams and bigrams, stop words and negation, and a fair
  comparison. Needs your own labelled text CSV.

## What the lessons use

| Lesson | Graph | Data in `examples/datasets` |
| --- | --- | --- |
| 1, 2, 5, 6 | **Multiclass image classification** (`examples/cyxgraph/mnist_mlp.cyxgraph`) | `digits_8x8.csv`: 1,797 handwritten digits, 8x8 pixels |
| 3, 4 | The graph you build; the finished version is `examples/cyxgraph/missing_values_fix.cyxgraph` | `patients_missing.csv`: 300 made-up rows with gaps |
| Text | **Text classification** (`review_sentiment_classifier.cyxgraph`) | `review_sentences.csv`: 3,000 review sentences |
| Audio | **Audio classification** (`speech_command_classifier.cyxgraph`) | `spoken_digits/`: 400 clips |
| Time series | **Time-series forecasting** (`timeseries/airline_passengers_dense.cyxgraph`) | `airline_passengers.csv`: 144 months |
| Python | `examples/python/digits_mlp.py` | `digits_8x8.csv` |
| Customer segments | `analysis/customer_segments_analysis.cyxgraph` | `customer_segments.csv` beside the graph: 248 made-up customers |
| ResNet | `cats_dogs_resnet_scratch.cyxgraph` | Not shipped: a two-class image folder of your own |
| TF-IDF | The two TF-IDF graphs in `Sentiment analysis/` | Not shipped: a labelled text CSV of your own |

The samples are small so that every lesson trains in seconds to a few
minutes on a CPU. They are for learning the Engine, not for judging a
model. Where each one comes from, and its licence, is in
[the datasets README](../../examples/datasets/README.md).

## Get the examples

The Engine looks for the `examples` folder next to its program. When the
start page lists the five **Task starter graphs**, you have it.

If the start page says "No starter graphs found: the examples/cyxgraph
folder is not next to this Engine.", your package has no `examples` folder
(the 1.0.7 packages do not ship one):

1. Open the [CyxWiz repository on GitHub](https://github.com/CYXWIZ-Lab/CYXWIZ)
   and download it (**Code > Download ZIP**).
2. Copy the `examples` folder from the download into the Engine's install
   folder, next to the Engine program.
3. Start the Engine again. The starters appear on the start page.

## Before you start

- Install the Engine: [Install the Engine](../install/index.md).
- A GPU is not needed. Every lesson runs on the CPU; with a GPU pack the
  Engine uses the GPU. See [Add a GPU pack and verify it](../install/gpu-packs.md).
- Python is needed only for the Python lesson. See
  [First launch](../install/first-launch.md).
