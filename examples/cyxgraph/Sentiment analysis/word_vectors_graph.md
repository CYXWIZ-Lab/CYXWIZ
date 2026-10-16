# The word-vectors sentiment graph, node by node

Graph: `sentiment_analysis_lstm_word_vectors.cyxgraph`. It classifies posts
from `sentiment_mental_health.csv` into 7 labels (Anxiety, Normal,
Depression, Suicidal, Stress, Bipolar, Personality disorder), with the
Embedding layer starting from word2vec vectors trained on the same posts.

```
Data Input ──> Tokenizer ──> Split ──> Data Loader ──> Embedding ──> LSTM ──> Dense ... ──> CrossEntropy ──> AdamW
     │                                                     ▲
     └──> Word vectors (word2vec) ── saves the .vec file ──┘ (loaded at the start of training)
```

To run it: train the **Word vectors** node first (double-click it, **Train**),
then press **Train** on the canvas.

## 1. Data Input

`sentiment_mental_health.csv`: 53,043 posts. Column `statement` is the text,
`status` the label. It feeds two nodes: the Tokenizer (the training path) and
the Word vectors node (a side branch that prepares the Embedding's start).

## 2. Word vectors (word2vec)

**Job:** learn a vector of 100 numbers for each word, so that words used the
same way get similar numbers. It learns from your posts alone, without the
labels.

1. **Read the words.** The `statement` column, lowercased, split into words
   (runs of letters and digits).
2. **Vocabulary for training.** Every word is counted; words seen at least 5
   times are kept (16,321 words).
3. **Train (skip-gram, 2 epochs).** For each word of each post, its neighbours
   within 5 words are its context. word2vec keeps two tables of vectors and
   nudges them so a word's vector predicts its real neighbours (their dot
   product goes up) and not 5 random "noise" words drawn for each pair (theirs
   goes down). That is about 24.6 million word-neighbour pairs per epoch.
   Words with the same company end up close: "therapy" ends next to
   counselling, counseling, medication, therapist.
4. **Line up with the vocabulary file** (section 4 below).

**GloVe instead:** choose **Train GloVe** on the Source tab. GloVe first
counts how often each pair of words appears within 10 words of each other
(about 8 million distinct pairs here, counted in 11 s), then fits the
vectors to those counts over 25 epochs: about 4.5 minutes on a GTX 1050 Ti.
"therapy" then ends next to cbt, sessions, medication and counselling. The
saved file, the line-up and the Embedding are the same.

**fastText instead:** choose **Train fastText**. It is word2vec that also
learns vectors for the pieces of each word (3 to 6 letters such as `<th`,
`the`, `apy>`), so a word the posts never contain, or a misspelling, still
gets a vector from its pieces: "anxietyy" lands next to anxiety (0.99),
"depressionn" next to depression. 5 epochs take about 15 minutes on a GTX
1050 Ti (2 epochs about 6). Its neighbours lean towards words that share
letters ("therapy": therapys, hypnotherapy, psychotherapy). On this project
almost every vocabulary token is a trained word; the line-up shows the rest
as "built from pieces".

**Output:** a table `word, v_0 ... v_99` with one row per vocabulary token, in
token-id order, saved as `artifacts/word_embeddings/sentiment_word_vectors.vec`.

## 3. Tokenizer (word mode)

**Job:** turn text into numbers.

1. Lowercase each post and split it into words.
2. Look each word up in `sentiment_analysis_vocab.txt`; its line number is its
   id. A word not in the file becomes 1 (`[UNK]`).
3. Cut or pad every post to 64 ids (padding id 0, `[PAD]`).
4. Turn the label into a class number 0 to 6.

"i can't sleep" becomes something like `[4, 61, 893, 0, 0, ...]`. The output is
a table: 64 id columns plus the label.

## How the vocabulary file is made

`prepare_sentiment_analysis_demo.py` (in this folder) writes it:

1. Each post is lowercased and split by the regular expression
   `[A-Za-z0-9_]+|[^\w\s]`: runs of ASCII letters, digits and `_`, **and every
   single punctuation mark or symbol on its own**. So `,` `.` `?` `!` and emoji
   are tokens too.
2. Every token is counted over all posts.
3. Tokens seen fewer than 5 times are dropped.
4. The rest are sorted, most frequent first (ties alphabetically), and cut so
   that, with the 4 special tokens, the file holds 5,000 lines.
5. The special tokens come first, then the sorted tokens, one per line:

   | Line (= id) | Token |
   | --- | --- |
   | 0 | `[PAD]` (padding) |
   | 1 | `[UNK]` (unknown word) |
   | 2 | `[BOS]` (begin of sequence) |
   | 3 | `[EOS]` (end of sequence) |
   | 4 | `i` (the most frequent token) |
   | 5 | `.` |
   | 6 | `to` |
   | 7 | `and` |
   | 8 | `,` |
   | 9 | `the` |

The file is the agreement between the nodes: id 9 means `the` for the
Tokenizer, for the Word vectors node and for the Embedding.

## 4. Lining the vectors up with the vocabulary

word2vec ends with its **own** list of 16,321 words, ordered by how often they
appear. The Embedding instead needs a table where **row i is the vector of
token id i** of the vocabulary file, because the Tokenizer hands it ids. So the
Word vectors node builds a new 5,000-row table, one row per line of the file:

1. For each token of the file, in order (id 0, 1, 2, ...), find its vector
   among word2vec's words:
   - found as written: copy that vector into row *id* (4,919 tokens);
   - otherwise, with "Match lowercase" on, look it up in lowercase (0 here,
     the file is already lowercase).
2. Special tokens get fixed rows: `[PAD]` all zeros (padding carries no
   meaning), `[UNK]` the average of all vectors (a neutral "some word").
3. Tokens with no vector get zeros (or small random values with "Missing
   words: Random"). Here 79: the punctuation marks and symbols (`.` `,` `?`
   ...), because word2vec's split keeps only letters and digits, plus
   `[BOS]`/`[EOS]` and a few odd characters.
4. The table is written in id order to the `.vec` file.

The Vocabulary tab of the node's dialog shows these counts and lists the
missing tokens. The 79 zero rows are not fixed for ever: with Freeze off the
Embedding learns them during training like every other row.

Lining up is what makes "pretrained" vectors usable at all: GloVe or fastText
files downloaded from elsewhere have their own word lists, and the same step
reorders them to your vocabulary.

## 5. Split and Data Loader

Split: 80% training, 10% validation, 10% test, stratified so each label keeps
its share (seed 42). Data Loader: batches of 64 posts, a 64 x 64 grid of ids
plus the labels.

## 6. Embedding (word2vec vectors)

**Job:** turn ids into vectors inside the model. It is a table of 5,000 rows x
100 numbers, and those numbers are weights the model trains. For each batch it
replaces every id with its row: 64 posts x 64 words x 100 numbers go to the
LSTM.

- Started from random numbers, it would have to learn word meanings from the 7
  labels alone.
- Here its starting table is the Word vectors file (Weights: **From a Word
  Embeddings node**), so similar words already start close together.
- Freeze is off: training keeps adjusting the vectors for this task.

## Word vectors node versus Embedding layer

| | Word vectors node | Embedding layer |
| --- | --- | --- |
| What it is | A data-preparation step | Part of the model |
| When it runs | Once, before training | On every batch of every epoch |
| Learns from | Which words appear near each other (no labels) | The sentiment labels, through the loss |
| Output | A saved file | Vectors for the LSTM |
| Changes during training | No | Yes (Freeze off) |

The word vectors are a general dictionary built by reading all the posts; the
Embedding is the model's own copy of it, refined for telling the 7 labels
apart. They meet only through the saved file, which works because all three
nodes use the same vocabulary file.

## After the Embedding

The LSTM reads the 64 vectors in order and keeps one summary of the post. Dense
layers turn it into 7 scores, CrossEntropy compares them with the true label,
and AdamW updates every weight, the Embedding's included.

## Result (2026-10-14, GTX 1050 Ti, CUDA)

Word vectors: trained in 206.5 s (2 epochs). Training: early stop at epoch 10
of 20 after 10 minutes; best validation accuracy 73.8% (epoch 7); training
accuracy 86.8%, so the model starts to overfit after a few epochs. Comparing
with **Weights: Random** shows what the word vectors add.
