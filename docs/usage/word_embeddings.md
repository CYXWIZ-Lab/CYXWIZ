# Word Embeddings

The **Word Embeddings** node gives every word a vector of numbers, so words
used alike sit close together ("therapy" near "counseling", "cbt", "emdr").
It reads vectors someone already trained (GloVe, fastText, word2vec files and
fastText models) or trains word2vec, GloVe or fastText on a text column of your
data, on the selected device.

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
project, replaced each time it trains, which is where an Embedding layer
finds them.

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
     task (Task View); the dialog shows the epoch, the progress and the time
     left, **Cancel training** stops it, and **Show in Training Dashboard**
     opens the run's live view (below), where the loss per epoch is drawn.
   - **Train GloVe**: the same text column, then vector size, window, minimum
     count, epochs and seed (Advanced: count cap, weighting power, learning
     rate, batch). Training runs in two stages, both in Task View: counting
     the word pairs ("Counting word pairs"), then the epochs ("Training GloVe,
     epoch 4 of 25, cost 0.01603"). The dialog shows the counted pairs and the
     time left ("Counted 8,024,345 word pairs · about 3 min 11 s left"); the
     cost per epoch is drawn on the Training Dashboard. When it ends the
     summary shows the first and last cost.
   - **Train fastText**: word2vec that also learns vectors for the pieces of
     each word (character n-grams of 3 to 6 letters, such as `<th`, `the`,
     `rap`, `apy>`; `<` and `>` mark the ends), so a word it never saw, or a
     misspelling, still gets a vector from its pieces. The settings are
     word2vec's (model, window, negative samples, epochs, learning rate) plus
     **Pieces** (n-gram lengths) and, under Advanced, **Buckets** (2,000,000,
     with the device memory they take shown). The run shows on the Training
     Dashboard like word2vec's, with a **Pieces** card.
4. **Vocabulary**: the Text Tokenizer's vocabulary file (filled in from the
   graph's Text Tokenizer when there is one). The vectors are then lined up
   with it: the Source tab shows how many tokens were found as written, found
   in lowercase, and missing, with the missing ones listed (a token the font
   cannot draw, such as an emoji, shows as its code point, `U+1F60A`).
   [PAD] gets zeros and [UNK] the mean vector; missing words get zeros or
   random values (**Missing words**). Empty keeps the file's own words.
   With fastText (trained, or a fastText `.bin`), **Words not trained**
   builds a token without a vector from its pieces (**Build from pieces**,
   the default) and the line-up counts them as "built from pieces"; tokens
   with no letter or digit (punctuation, symbols, emoji) follow the Missing
   words rule either way.
5. **Nearest words**: type a word and press Enter for its 10 nearest words by
   cosine similarity, a quick check that the vectors make sense. With fastText
   a word the vectors lack still gets neighbours from its pieces, noted "not
   in the training words · from 26 pieces" (the model of this session's run,
   else the model saved with the vectors, or the pretrained `.bin` read again
   for that word).
6. **Outputs**: where the vectors are saved, and **Row vectors** for the Rows
   pin.

Words are split as the Text Tokenizer's Word mode splits them: lowercased,
runs of letters and digits of any script, read by character ("café",
"naïve" and Cyrillic words stay whole); punctuation such as a curly
apostrophe, and emoji, end a word. A vocabulary file built before 2026-10-16
can hold single-byte tokens (the pieces of such characters); rebuild it.

## Trained once, reused after

A training run writes a small record beside the saved vectors,
`<vectors>.vec.training.json`. It holds:

- a fingerprint of the text;
- the settings that shape the vectors;
- the vocabulary file's content;
- when the vectors were trained, the summary and the line-up.

The next run of the step compares these with its own: **Train** in the
dialog, a pipeline run, or a run of a later step. When nothing changed it
reads the saved file instead of training again. On the sentiment set that
takes under a second instead of minutes. The dialog then says "Reused the
saved vectors in 0.4 s", and no new run appears on the Training Dashboard.

Training runs again when:

- the text changes (another file, changed rows, another text column);
- a setting of the method changes (vector size, window, epochs, seed, ...),
  or the Missing words rules when a vocabulary is used;
- the vocabulary file is chosen, removed or edited;
- the saved `.vec` is replaced or edited;
- fastText: the saved model (`.bin`) is replaced, edited or deleted.

**Row vectors**, the save path, and settings of the other methods (GloVe's
window for a word2vec run) do not count. To train again with nothing changed,
change the seed. A new Engine version whose trainers give different vectors
retrains once.

After the project is reopened, the dialog shows the saved training ("Trained
2026-10-15 19:40 in 92.3 s", its summary and line-up), and **Nearest words**
looks words up in the saved vectors. When the settings in the dialog differ
from that training's, it says "The settings changed since this training".

A fastText training also saves its model beside the vectors, in fastText's
own `.bin` format: `sentiment_fasttext.vec` gets `sentiment_fasttext.bin`
(a vectors path that already ends in `.bin` gets `<name>.bin.model.bin`).
The `.vec` holds the trained words only; the model keeps the pieces' rows, so
after a reopen **Nearest words** still builds words the training never saw
("anxietyy" -> anxiety). The model is about as large on disk as the buckets
are on the device (800 MB at 2,000,000 buckets and 100 values); fewer buckets
make it smaller. fastText and gensim open it too
(`gensim.models.fasttext.load_facebook_model`), and the Word Embeddings node
reads it as a pretrained fastText model.

## On the Training Dashboard

Each word2vec or GloVe training run shows on the Training Dashboard as its
own run, whether it started from the dialog's **Train** or while a graph's
training prepares its data. A **Run** switcher at the top of the dashboard
lists the model run and the latest word-vector runs; a word-vector run never
changes the model run's charts, cards or Run Comparison table. The dashboard
shows the run that just started; once a model starts training it goes back to
the model run, unless you picked a run yourself.

A word-vector run shows:

- **Status**: Preparing (GloVe counting), Training, Completed, Failed or
  Cancelled, the node and method, the epoch ("Epoch 7 / 25 · 18 remaining")
  and the progress. **Cancel** stops the run; **Actions**: Export curve CSV,
  Export runs CSV (to the project's `exports` folder), Open node dialog.
- **Steps**: Counting word pairs (GloVe: "8,024,345 pairs · 10.6 s"),
  Training, Lining up with the vocabulary ("4,919 found · 79 missing"),
  Saved (the file name).
- **Cards**: the latest cost per pair (GloVe) or loss per example (word2vec)
  with its change since epoch 1 and its best epoch; the time remaining (from
  the epoch times) or the total time; the vocabulary (words seen min-count
  times, values each); the word pairs (GloVe) or the examples of the last
  epoch (word2vec).
- **Chart**: the cost or loss of each epoch, drawn as each epoch ends, with
  every planned epoch on the axis (hover a point for its value; the chart's
  own tools give a log axis, export and a window of its own).
- When finished: **Result** (the saved file, how the vocabulary lined up, the
  special tokens, and the Embedding layers that load the file) and a **Quick
  check** of the nearest words of the vocabulary's most frequent word of 7+
  letters (type another word and press Enter; **More words in the node
  dialog** opens the Nearest words tab's dialog).
- **Word vector runs**: every run of the session (method, settings, words,
  pairs or examples per epoch, first and last cost or loss, time, file);
  click a row to show it.

## Supported files

| Format | Recognised by |
| --- | --- |
| GloVe text (`.txt`) | no header; a word then its values on each line |
| fastText / word2vec text (`.vec`) | a first line `count dim`, then a word and its values per line |
| word2vec binary (`.bin`) | the `.bin` extension |
| fastText model (`.bin`) | the `.bin` extension and fastText's magic number (793712314) at the start |

Text and word2vec files are read as gensim's
`KeyedVectors.load_word2vec_format` reads them; a repeated word keeps its first
vector.

A fastText model (fastText's own `.bin`, version 12, as fastText and gensim's
`save_facebook_model` write it) holds the words' and the pieces' rows. With a
vocabulary file only the rows the vocabulary needs are read (each token's word
row and its pieces' rows), so a 7 GB model such as `cc.en.300.bin` is not read
whole; a token the model lacks is built from its pieces. [UNK]'s mean is the
mean of the tokens found. The dialog shows "fastText model (.bin) · 16,334
words · 2,000,000 pieces · 100 values each · 813 MB · read 3.1 s". Quantized
(`.ftz`) and supervised (classifier) models are refused with a reason; use
their `.vec` file.

## word2vec settings

gensim's names and defaults (`Word2Vec`): vector size 100, window 5,
minimum count 5, negative samples 5, epochs 5, down-sampling 1e-3, learning
rate 0.025 falling to 0.0001, plus **Batch** (examples per device step,
2048) and **Seed** (the same seed repeats a run on one device).

Training runs in batches on the device. One step applies the summed gradient
of its batch's examples, with gensim's update rules (in CBOW every context word
takes the whole gradient). Each epoch's examples are shuffled first, so a
frequent word's examples spread over many steps; in corpus order large
batches diverged. The learning rate falls linearly over the run, as gensim's
does. A step uses at most 8 examples per vocabulary word, so a small
vocabulary (tens of words) takes smaller steps and stays stable; a real
vocabulary keeps the whole batch. Batches of 512 give the same quality more
slowly; 8192 is faster but further from gensim.

**Speed and quality (measured 2026-10-15)** on the sentiment set below (53,043
rows, 16,334 words, 24.6 M skip-gram examples per epoch), skip-gram, 5 epochs,
GTX 1050 Ti (CUDA): 247 s (about 49 s per epoch; before this work 115 s per
epoch), against 68 s for gensim on 4 CPU threads. The run shares 59% of each
word's 10 nearest neighbours with gensim (top 1,000 words); two gensim runs
with different seeds share 65%. Nearest words of "therapy": counseling 0.814,
cbt 0.802, counselling 0.788, emdr 0.765, dbt 0.752. On 20,000 posts (3
epochs) the same changes took a run from 134 s and 39% shared to 54 s and
55% (gensim seed to seed 60%), and CBOW from 40% to 62% shared (gensim 63%).

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

## fastText settings

gensim's FastText names and defaults: model (skip-gram or CBOW), vector size
100, window 5, minimum count 5, negative samples 5, epochs 5, down-sampling
1e-3, learning rate 0.025 to 0.0001, batch 2048 and seed, all shared with
word2vec (same keys), plus `min_n` 3 and `max_n` 6 (piece lengths), `bucket`
2,000,000 (rows the pieces are hashed into, fastText's FNV-1a hash; 4 x vector
size bytes each on the device, 800 MB at 100 values) and `unseen_words`
(`pieces` or `missing`).

How it learns: as word2vec, but a word's input vector is the mean of its own
row and its pieces' rows. In skip-gram the centre word, with its pieces,
predicts each context word; in CBOW the context words' rows and pieces predict
the centre. Every row of the mean takes the whole gradient, as fastText and
gensim update. Batches train stably because each epoch's examples are
shuffled (as for word2vec); in corpus order a common piece such as `ing>`
summed hundreds of updates in one step and diverged. A word's saved vector is
the mean of its row and its pieces' rows; a word outside the vocabulary gets
the mean of its pieces' rows.

**Speed and quality (measured 2026-10-15)** on the sentiment set (16,334
words, 24.6 million skip-gram examples per epoch, 2,000,000 buckets) on a
GTX 1050 Ti (CUDA): about 178 s per epoch, 5 epochs in 890 s (1,152 s before
the changes below); gensim 4.4.0 takes 395.5 s on one CPU thread (106.8 s on
four). The loss per example falls from 2.551 to 2.428. The two models share
71% of each word's 10 nearest neighbours (top 1,000 words; 57% before; two
gensim runs with different seeds share 77%).

Each epoch's examples are shuffled by centre word: a centre's examples stay
together, so its piece rows are read once for all of them (shuffling every
example was 301 s per epoch, same quality). After building an epoch's
examples the trainers return ArrayFire's cached scratch memory to the device:
kept, it filled the 4 GB card and every step waited on memory (453 s per
epoch).
Words the training never saw, from their pieces: anxietyy -> anxiety 0.990,
depressionn -> depression 0.986, overthinkingg -> overthinking 0.985,
psychiatristt -> psychiatrist 0.997, counsellingg -> counselling 0.970.
Neighbours lean towards words that share letters ("therapy": therapys,
hypnotherapy, radiotherapy, psychotherapy; "depressed": repressed,
deppressed, oppressed), fastText's trade-off.

The device keeps the word and piece rows ((words + buckets) x vector size x 4
bytes: 807 MB here). Each skip-gram batch reads a word's rows from a padded
table: the words of 98% of the text use a narrow one, the long tail (up to 99
rows, such as `jwkdrbp6bk528dudk`) runs in batches of its own.

On this project no vocabulary token is built from pieces: the vocabulary file
comes from the same posts and its 5,000 tokens are all frequent words (even at
minimum count 20). Pieces fill in for a vocabulary from other text, for rare
words below the minimum count, and with pretrained fastText models.

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
(2 epochs, under 2 minutes on a GTX 1050 Ti), then **Train** the graph. Freeze
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
  (`reference/generate_glove_fixtures.py`); fastText hash, pieces, steps,
  vectors, group quality (words and unseen variants), a trained model saved
  as `.bin` and read back (header, words, the same vectors for known and
  unseen words) and a fastText `.bin`
  (header, vectors, a vocabulary lined up with pieces) against gensim /
  PyTorch (`reference/generate_fasttext_fixtures.py`). Runs on CUDA, OpenCL
  and CPU.
- `test_word_embeddings_operator`: the node (lined-up table, saved file, Rows,
  long rows averaged in several device blocks, word2vec and GloVe training
  repeat, GloVe's two progress stages, the dashboard's figures: settings,
  vocabulary, pairs, each epoch's loss or cost, the line-up, the end;
  fastText trained and lined up with a token built from pieces, Words not
  trained as missing, the model kept for the dialog and saved as `.bin`
  (reused with both files, trained again when the `.bin` is deleted), a
  fastText `.bin` read
  for a vocabulary; refusals).
- `test_word_embeddings_presentation`: the dialog's wording (with the
  pieces in the line-up, the fastText file summary, the unseen-word note).
- `test_word_vector_runs`: the dashboard's word-vector runs, on the real
  GloVe run (status, steps, cards, time left, result, table rows, CSV) and
  word2vec, fastText (Pieces card), cancelled and failed runs.
- `benchmark_word_vectors <word2vec|glove|fasttext> <tokens.txt> [epochs] [batch] [out.vec] [buckets]`
  and `scripts/word2vec_neighbour_overlap.py`: speed and agreement with gensim
  or the GloVe reference; fastText also prints the neighbours of misspellings
  it never saw.
- `scripts/arrayfire_probes/sum_by_key_runs`: a standalone check of
  `sumByKey` on sorted keys with runs (run after an ArrayFire upgrade).
