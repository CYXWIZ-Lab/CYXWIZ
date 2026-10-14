# Word Embeddings

The **Word Embeddings** node gives every word a vector of numbers, so words
used alike sit close together ("therapy" near "counseling", "cbt", "emdr").
It reads vectors someone already trained (GloVe, fastText, word2vec files) or
trains word2vec or GloVe on a text column of your data, on the selected device.

```
Data Input (texts) -> Word Embeddings -> Vectors -> (an Embedding layer loads the saved file)
                                      -> Rows    -> Logistic Regression / SVM / KNN / Plot
```

It is a Text node. Its input pin takes a table with a text column (needed for
training and for row vectors; a pretrained file alone needs no input rows).

## Outputs

| Pin | What it holds |
| --- | --- |
| **Vectors** | A table `word, v_0 .. v_{d-1}`: one row per word. With a vocabulary file, one row per vocabulary token in token-id order, so row *i* is token id *i*. |
| **Rows** | With **Row vectors** on: the input table plus `emb_0 .. emb_{d-1}`, the mean vector of each row's words. Words without a vector are skipped; a row with none gets zeros. |

The vectors are always saved as word2vec text (`.vec`). With **Save vectors**
empty they go to `artifacts/word_embeddings/<node id>_<node name>.vec` in the
project, replaced at each run, which is where an Embedding layer finds them.

## Using it

1. Add **Word Embeddings** (Nodes > Text) and connect the table with the text
   to its **Text** pin.
2. Double-click it (or **Open Dialog** in Properties). The dialog has four
   tabs: **Source**, **Vocabulary**, **Nearest words**, **Outputs**.
3. Choose the source:
   - **Pretrained file**: choose the vector file. The dialog reads it at once
     (in Task View) and shows its format, size and read time, e.g.
     "word2vec / fastText text · 16,334 words · 100 values each · read 0.5 s".
     **Read at most** limits a large file to its first words (0 reads all).
   - **Train word2vec**: choose the text column (**Load inputs** runs the
     steps before it so the dialog can list the columns and rows), the model
     (Skip-gram or CBOW) and its settings, then **Train**. Training runs as a
     task (Task View); the dialog shows the epoch and progress, **Cancel
     training** stops it, and when it ends the loss per epoch is drawn.
   - **Train GloVe**: the same text column, then vector size, window, minimum
     count, epochs and seed (Advanced: count cap, weighting power, learning
     rate, batch). Training runs in two stages, both in Task View: counting
     the word pairs ("Counting word pairs"), then the epochs ("Training GloVe,
     epoch 4 of 25, cost 0.01603"). When it ends the cost per epoch is drawn;
     the summary shows the first and last cost and the curve's tooltip lists
     every epoch.
4. **Vocabulary**: the Text Tokenizer's vocabulary file (filled in from the
   graph's Text Tokenizer when there is one). The vectors are then lined up
   with it: the Source tab shows how many tokens were found as written, found
   in lowercase, and missing, with the missing ones listed (a token the font
   cannot draw, such as an emoji, shows as its code point, `U+1F60A`).
   [PAD] gets zeros and [UNK] the mean vector; missing words get zeros or
   random values (**Missing words**). Empty keeps the file's own words.
5. **Nearest words**: type a word and press Enter for its 10 nearest words by
   cosine similarity, a quick check that the vectors make sense.
6. **Outputs**: where the vectors are saved, and **Row vectors** for the Rows
   pin.

Words are split as the Text Tokenizer's Word mode splits them: lowercased,
runs of letters and digits.

## Supported files

| Format | Recognised by |
| --- | --- |
| GloVe text (`.txt`) | no header; a word then its values on each line |
| fastText / word2vec text (`.vec`) | a first line `count dim`, then a word and its values per line |
| word2vec binary (`.bin`) | the `.bin` extension |

Files are read as gensim's `KeyedVectors.load_word2vec_format` reads them; a
repeated word keeps its first vector.

## word2vec settings

gensim's names and defaults (`Word2Vec`): vector size 100, window 5,
minimum count 5, negative samples 5, epochs 5, down-sampling 1e-3, learning
rate 0.025 falling to 0.0001, plus **Batch** (examples per device step, 512)
and **Seed** (the same seed repeats a run on one device).

Training runs in batches on the device. One step applies the summed gradient
of its batch's examples; larger batches than 512 diverge (a frequent word
gets the sum of too many gradients), so keep the default.

**Speed (measured 2026-10-14)** on the sentiment set below (53,043 rows,
6.0 M words, 24.6 M skip-gram examples per epoch): about 115 s per epoch on a
GTX 1050 Ti (CUDA), against 26 s per epoch for gensim on 4 CPU threads. Five
epochs take about 10 minutes here. The vectors are good (see the example),
but a 1-epoch run shares 37% of each word's 10 nearest neighbours with
gensim, while two gensim runs with different seeds share 60%. Faster, closer
training is open work (TOFIX152).

## GloVe settings

The Stanford GloVe defaults (`glove.c` / `cooccur.c`): vector size 100,
window 10, minimum count 5, epochs 25, count cap (x_max) 100, weighting power
0.75, learning rate 0.05 (AdaGrad), plus **Batch** (word pairs per device
step, 8192) and **Seed**. Vector size, minimum count and seed are shared
with word2vec; the window, epochs and the rest are GloVe's own (keys
`glove_window`, `glove_epochs`, `x_max`, `weighting_power`,
`glove_learning_rate`, `glove_batch`).

How it learns: every pair of words within the window of each other in one
row is counted, a pair `d` words apart adding `1/d`. Then each word gets two
vectors and two biases, fitted so that their dot product plus the biases
matches the log of the pair's count, pairs weighted by
`min(1, count / x_max)^0.75`. The saved vector of a word is the sum of its
two vectors, as the reference writes them.

**Speed and quality (measured 2026-10-14)** on the sentiment set (16,334
words, 7,968,113 word pairs) on a GTX 1050 Ti (CUDA): counting 10.6 s, then
about 11 s per epoch; 25 epochs in 319 s (269 s in the Engine). The reference
(`scripts/glove_reference.py`, the same algorithm on the CPU) takes 42 s to
count and 832 s to train. The cost per pair ends at 0.00613 (reference
0.00614), and the two models share 87% of each word's 10 nearest neighbours
(top 1,000 words). Nearest words of "therapy": cbt 0.782, sessions 0.777,
medication 0.772, counselling 0.762.

After 1 or 2 epochs every word still points the same way (cosine near 1.0
with all words, in the reference too); use the full 25 epochs.

## With an Embedding layer

In the Embedding node's dialog, **Weights** tab:

- **Random**: the table starts random and is learned in training.
- **From a Word Embeddings node**: choose the node. Its saved vectors become
  the weights file and set the shape (vocabulary size and dimension; the
  Shape tab shows them read-only). Run the Word Embeddings node first so its
  vectors exist.
- **File**: a weights file of your own: a matrix of numbers (one row per
  token id, spaces or commas) or word vectors as text (`.vec`, GloVe `.txt`;
  the words are dropped, rows kept in file order). The dialog reads its shape
  at once and offers **Use the file's shape** when it differs from the node.

**Freeze loaded weights** keeps the table as loaded and trains the rest of
the model. **Build starter weights** (folded) writes a random table of the
node's shape and chooses it. A relative weights file is read from the
project folder.

## Example graph: a classifier starting from word vectors

`examples/cyxgraph/Sentiment analysis/sentiment_analysis_lstm_word_vectors.cyxgraph`:

```
Data Input (statement, status) -> Text Tokenizer (vocabulary file) -> Data Split -> Data Loader
     |                                                                               |
     +-> Word Embeddings (train word2vec or GloVe, same vocabulary file)             v
            saves artifacts/word_embeddings/sentiment_word_vectors.vec ------> Embedding (5,000 x 100, loads the file)
                                                                               -> LSTM -> Dense -> ... -> Output
```

The three pieces share one vocabulary file, so row *i* of the vectors is the
vector of token id *i*:

- the **Text Tokenizer** turns words into ids with `sentiment_analysis_vocab.txt`;
- the **Word Embeddings** node lines its vectors up with the same file and
  saves them;
- the **Embedding** loads the saved file (Weights > From a Word Embeddings
  node), which also sets its size to 5,000 x 100.

The graph carries notes on the canvas, and `examples/cyxgraph/Sentiment analysis/word_vectors_graph.md` explains it node by node (how the vocabulary file is made, the line-up step, word vectors versus the Embedding).

Steps: open the graph, double-click **Word vectors (word2vec)** and **Train**
(2 epochs, about 4 minutes on a GTX 1050 Ti), then **Train** the graph. Freeze
is off, so the vectors are fine-tuned with the rest of the model; turn it on
to keep them as trained. Training does not run the Word Embeddings node: if
its file is missing, the Embedding stops with "could not open embedding
weights file".

## Example: the mental-health sentiment set

`D:/demo/mrcj/datasets/sentiment analysis/sentiment_mental_health.csv`
(column `statement`, 53,043 rows) with its tokenizer vocabulary
`sentiment_analysis_vocab.txt` (5,000 tokens) and vectors trained on it by
gensim (100 values, window 5, minimum count 5, 5 epochs):

- Pretrained, lined up with the vocabulary: 4,919 found as written, 0 in
  lowercase, 79 missing (mostly punctuation, digits and symbols, and
  [BOS] / [EOS]).
- Nearest words of "therapy": counseling 0.796, counselling 0.786, cbt 0.763,
  emdr 0.760, dbt 0.714 (identical to gensim's `most_similar`).
- Row vectors: the Rows pin gives the 53,043 rows with `emb_0 .. emb_99`,
  ready for a classifier or a Plot.

The check graph is `word_embeddings_check.cyxgraph` in the test project
(pretrained node with a Row vectors plot, a training node, an Embedding).

## Checking it

- `test_computation_truth_word_embeddings`: reading, lining up, nearest words,
  row means and word2vec steps against gensim 4.4.0 / PyTorch fixtures; GloVe
  counts, AdaGrad steps and group quality against numpy / PyTorch
  (`reference/generate_glove_fixtures.py`).
- `test_word_embeddings_operator`: the node (lined-up table, saved file, Rows,
  long rows averaged in several device blocks, word2vec and GloVe training
  repeat, GloVe's two progress stages, refusals).
- `test_word_embeddings_presentation`: the dialog's wording.
- `benchmark_word_vectors <word2vec|glove> <tokens.txt> [epochs] [batch] [out.vec]`
  and `scripts/word2vec_neighbour_overlap.py`: speed and agreement with gensim
  or the GloVe reference.
