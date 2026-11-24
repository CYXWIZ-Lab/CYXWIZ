---
title: Learn word vectors from a text column
description: Learn word2vec, fastText or GloVe vectors from a text column in Explore > Text, find the nearest words and look at the 3D word map.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - text.md
  - ../../usage/word_embeddings.md
---

The **Embeddings** look of Data Studio's **Explore > Text** tab learns word
vectors from a text column, so that each word sits near the words used in
the same places. It trains with the Word Embeddings node's trainers on the
device, shows the nearest words to a word you type, and opens a 3D map of
the words. **Tools > Text > Word Embeddings** opens it. Picking the column
and the groups is as in [Explore a text column](text.md).

## Before you start

- A loaded dataset with a text column, shown in Data Studio's rail.

## Learn the vectors

1. In **Data Studio > Explore**, open the **Text** tab, pick the **Text
   column**, then the **Embeddings** look.
2. Pick what to **Learn**: word2vec, fastText (with character n-grams) or
   GloVe.
3. For word2vec and fastText, pick the **Architecture**: **skip-gram** (each
   word predicts its context; better for rare words, the default) or
   **CBOW** (the context predicts the word; faster).
4. Set **Size** (100 numbers per word), **Window** (5; GloVe 10), **Words
   seen at least** (5) and **Epochs** (5; GloVe 25).
5. Pick what to **Learn from**: **a sample** (the default: rows drawn at
   random, seed 1, until 500,000 words, so the look takes seconds) or
   **every row** (a large column takes minutes).
6. Select **Learn**. A progress bar and **Stop** show while it learns.

The look does not learn by itself: after a change, select **Learn again**
(off, with the reason on hover, while those settings are learned or
learning). Until the new vectors are learned, the last ones stay shown, and
the page says so.

## Read the vectors

The page shows the words learned, the vectors, the rows learned from
("4,298 of 52,681 rows (a sample)") and the words read.

- **Nearest words**: type a word, and its nearest words by the cosine of
  the vectors show (**show** N): words used in the same places, opposites
  included.
- A turning preview of **the word map**, with the word and its neighbours
  marked.

## Keep the vectors

- **Keep as a node** adds the Word Embeddings node with these settings
  (source, architecture, size, window, min count, epochs, seed) as a
  branch. It learns the vectors again from every row when the graph runs,
  and saves them for an Embedding layer (see
  [word embeddings](../../usage/word_embeddings.md)).
- **Save vectors (.vec)** writes these vectors in the word2vec text format
  (a word and its numbers per line), which the Word Embeddings node can
  load.

The learned vectors live in the Engine's memory: after a restart, select
**Learn again** or load the saved `.vec` file.

## Look at the word map

**Open the word map** opens the words in 3D in a window of its own, over
Data Studio at 80% of the main window; dock it, float it, or fill the
screen. The same map is the **Map** tab of the Word Embeddings node's
dialog, on the node's own vectors (see
[word embeddings](../../usage/word_embeddings.md#map)).

| Part | What it does |
| --- | --- |
| **Words on the map** | The 1,500 commonest words of three letters or more without the common English words (100 to 5,000), plus the word looked up and its neighbours. |
| **Look up a word** | Enter, or double-click a word on the map: the word in the accent colour, its **neighbours shown** (10) ringed, labelled and joined to it, the view moved around them; the list of neighbours with their cosine (click one to look it up). |
| **Layout** | **t-SNE** (the default: near words stay near; distances between far clusters mean little), **UMAP**, or **PCA** (the three widest directions: faithful but crowded; the share of the spread each axis holds is said). The layout is of the unit vectors, so t-SNE and UMAP measure as the cosine does; t-SNE uses scikit-learn's automatic learning rate (rows / 48, at least 50) and reports its iterations; a new layout can be stopped. |
| **View** | 3D (drag to turn, wheel to zoom) or flat (from above); **Reset view**; **Turn slowly**. |
| **Colour** | **clusters** (k-means of 8 clusters, 2 to 20, each named by its three commonest words), **leans to a group** (with a group column: a word leans to the group where it is in twice its share of the rows, in 30 of its rows or more; grey otherwise) or **one colour**. |
| **Labels** | The word and its neighbours, and room for more (the most common words first), none over another, with a short line to its dot when a label moved aside. |
| **Export PNG** | Saves the map as an image. |

## Related

- [Explore a text column](text.md)
- [Word embeddings node](../../usage/word_embeddings.md)
- [Explore your data](explore.md)
