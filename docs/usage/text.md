# Text

One text column read five ways: which words it uses, what sets each group of
rows apart, how positive or negative the rows read, what a tokenizer makes of
them, and which words are used alike. Data Studio's Explore tab **Text**
answers those on any table ([Data Studio](data_studio.md#text)); the nodes
below do the same inside a graph, so the answer runs with the rest of the
pipeline.

Tools > Text opens the tab on the right look: **Tokenization** opens Tokens,
**Word Frequency** opens Words, **TF-IDF Analysis** opens Terms, **Word
Embeddings** opens Embeddings and **Sentiment Analysis** opens Sentiment;
Data > Tokenization opens Tokens too. **Language Model Generation** stays a
window of its own ([language model generation](lm_generation.md)).

The figures are the same as Python gives. `test_computation_truth_text` checks
the words, the stop words, the Porter stems and VADER against Python's `re`
and `str.lower`, scikit-learn's `ENGLISH_STOP_WORDS`, NLTK's `PorterStemmer`
and vaderSentiment 3.3.2 (1,516 word cases, 3,065 VADER cases, 6,096 stems);
`test_text_explore` checks Terms against scikit-learn's `TfidfVectorizer`,
Sentiment against vaderSentiment and the nearest words against the cosine of
the vectors. The tab calls the nodes' own code: Terms counts with the TF-IDF
node's rule, Tokens trains the Tokenizer node's tokenizer, Embeddings trains
with the Word Embeddings node's trainers on the device.

## The column and the groups

Pick a **Text column** (its average length and distinct values show on hover)
and, optionally, **Group by**: a category or the label, with 2 to 200 values.
With a group column, the words, the terms and the sentiment are also shown per
group: the 12 largest groups, the rest together as "other", and rows
without a group as "(missing)". Rows whose text is missing (NULL, or a text
the project marks as missing) or only spaces are counted and left out; every
look says how many it read ("52,681 rows with text (362 missing left out)").

Each look runs as a job on its own: it shows its progress, and **Stop** ends
it ("Stopped. Change a setting to run again.").

## Words, stop words and stems

Every look splits the text the same way, and the TF-IDF and Count Vectorizer
nodes split it that way too:

- a **word** is a run of letters and digits of any script, lower-cased, two or
  more long; an apostrophe between two runs is kept ("don't", "i've"), and a
  typographic apostrophe (’) reads as one. Python:
  `re.finditer(r"[^\W_]+(?:'[^\W_]+)*", text)`, lower-cased. scikit-learn's
  default `token_pattern` (`\b\w\w+\b`) differs on apostrophes ("don't" gives
  "don") and the underscore (a word character there);
- the **common English words** (stop words) are scikit-learn's
  `ENGLISH_STOP_WORDS` (318) and their contractions (54: "don't", "i'm", ...).
  They include "not", "no" and "nor": leave them in when a negation matters
  (see [text vectorizers](text_vectorizers.md#n-grams-and-stop-words));
- a **stem** is the Porter (1980) stem ("feeling" and "feels" count as
  "feel"), NLTK's `PorterStemmer(mode="ORIGINAL_ALGORITHM")`.

## Words

The commonest words of the column, or word pairs (**Count**: words | word
pairs), with the common English words left out or kept and by stem when asked;
**Show the top** N (5 to 200). The page shows the words read, the distinct
ones, the words per row (median, 95th percentile, longest), a bar chart of the
top words, a histogram of the words per row (up to a round cap near the 95th
percentile; the last bar holds the longer rows), each group's commonest words
and the commonest pairs. **Export CSV** writes rank, word, count and share;
**Copy** copies the sentence.

## Terms

TF-IDF over the rows, as the TF-IDF node weighs it: a term's count in the row
times its smoothed idf, ln((1 + n) / (1 + df)) + 1, the row then normalised
(**Normalise**: L2, L1 or none). **Terms**: words, or words and pairs; a term
counts when it is **in at least** N rows (min_df, 5 by default).

- With a group column, **what sets each group apart**: the terms whose mean
  TF-IDF in the group is the most above the table's (each group's six, the
  scores on hover). Without one, the terms that weigh most over the table.
- **A row's terms** (**Row**: the table row, 1 by default) and **the rows that
  read most like it**: the five nearest by the cosine of their TF-IDF rows,
  with their group.
- **Text quality** the terms show: rows with HTML entities (`&amp;`), links,
  repeated rows, and rows with broken characters (UTF-8 read as Latin-1 and
  saved again, such as "Ã©" or "ðÿ", or U+FFFD). Clean them with a Text Clean
  step first when they top the terms.

**Add as a step** adds the TF-IDF node with these settings (the 2,000
commonest terms, max_features) after the data in use: a table of tfidf_0 ..
with the label and the split. **Export CSV** writes group, term, mean TF-IDF
and the lift.

## Sentiment

VADER (Hutto and Gilbert, 2014) on every row: a rated lexicon with rules for
negation ("not good" reads negative), boosters ("very"), capitals, "but",
emoticons and emojis. Each row gets a **compound** score from -1 to 1 and the
shares of its weight that read negative, neutral and positive. A row reads
**positive** at the compound **Positive from** (0.05) and above, **negative**
at **negative at** (-0.05) and below, and neutral between: VADER's own
thresholds. vaderSentiment rounds the compound to 4 places before such a
comparison; the tab compares the unrounded score, so a row at exactly the
threshold after rounding can fall on the other side.

The page shows the negative, positive and neutral rows and the mean compound,
a histogram of the compound over -1 .. 1, a row word by word (each word with a
valence after the rules coloured and scored; **Row** picks it) and, with a
group column, each group's rows, median compound and negative and positive
shares. VADER was made for short social-media text: long rows add up many
words and read more strongly.

**Add as a step** adds the Sentiment Analysis node: the columns
`<text>_compound`, `<text>_positive`, `<text>_neutral`, `<text>_negative`
(numbers) and `<text>_sentiment` ("positive", "neutral" or "negative") beside
the others, the label and the split kept. **Export CSV** writes row, group,
compound and sentiment for every row.

## Tokens

What a tokenizer makes of the rows, with the Tokenizer node's tokenizers
trained on this column as the node trains them: **Tokenizer** (Word by
default; Whitespace, Character, Byte BPE, WordPiece, and SentencePiece BPE /
Unigram when the SentencePiece provider is installed), **Vocabulary** (the
largest vocabulary, max_vocab_size: 10,000; 0 keeps every token), **seen at
least** N times (min_word_freq: 2) and the **Length limit** (max_length: 256
tokens).

The page shows the tokens per row (median, 95th percentile, longest), the rows
the limit cuts (and the share it would cut at twice the limit), the vocabulary
and the share of tokens it covers (not the unknown token), a histogram of the
tokens per row with the limit marked, and a row as the tokenizer reads it:
each token with its id, an unknown token marked (the first 64; **Row** picks
the row). **Add as a step** adds the Tokenizer node with these settings:
token id columns (max_length per row) with the label and the split, for an
Embedding, LSTM or Transformer model. **Copy tokens** copies the row's tokens
and ids.

## Embeddings

Word vectors learned from the column, so that each word sits near the words
used in the same places. **Learn**: word2vec, fastText (with character
n-grams) or GloVe, the Word Embeddings node's trainers on the device.
**Architecture** (word2vec and fastText): **skip-gram** (each word predicts
its context; better for rare words, the default) or **CBOW** (the context
predicts the word; faster). **Size** (100 numbers per word), **Window** (5;
GloVe 10), **Words seen at least** (5), **Epochs** (5; GloVe 25).

**Learn from**: **a sample** (the default: rows drawn at random, seed 1,
until 500,000 words, so the look takes seconds) or **every row** (a large
column takes minutes). The look does not learn by itself: **Learn** starts it
with the settings shown and **Learn again** after a change (off, with the
reason on hover, while those settings are learned or learning); a progress bar
and **Stop** show while it learns. Until the new vectors are learned, the last
ones stay shown, and the page says so.

The page shows the words learned, the vectors, the rows learned from ("4,298
of 52,681 rows (a sample)") and the words read; **Nearest words**: type a word
and its nearest words by the cosine of the vectors show (**show** N), words
used in the same places, opposites included; and a turning preview of **the
word map** with the word and its neighbours marked.

**Keep as a node** adds the Word Embeddings node with these settings (source,
architecture, size, window, min count, epochs, seed) as a branch: it learns
the vectors again from every row when the graph runs, and saves them for an
Embedding layer ([word embeddings](word_embeddings.md)). **Save vectors
(.vec)** writes these vectors in the word2vec text format (a word and its
numbers per line), which the Word Embeddings node can load. The learned vectors
live in the Engine's memory: after a restart, Learn again or load the saved
.vec.

## The word map

**Open the word map** opens the words in 3D in a window of its own, over Data
Studio at 80% of the main window; dock it, float it, or fill the screen. The
same map is the **Map** tab of the Word Embeddings node's dialog, on the
node's own vectors ([word embeddings](word_embeddings.md#map)).

- **Words on the map**: the 1,500 commonest words of three letters or more
  without the common English words (100 to 5,000), plus the word looked up and
  its neighbours.
- **Look up a word** (Enter, or double-click a word on the map): the word in
  the accent colour, its **neighbours shown** (10) ringed, labelled and joined
  to it, the view moved around them; the list of neighbours with their cosine
  (click one to look it up).
- **Layout**: **t-SNE** (the default: near words stay near; distances between
  far clusters mean little), **UMAP**, or **PCA** (the three widest directions:
  faithful but crowded; the share of the spread each axis holds is said). The
  layout is of the unit vectors, so t-SNE and UMAP measure as the cosine does;
  t-SNE uses scikit-learn's automatic learning rate (rows / 48, at least 50)
  and reports its iterations; a new layout can be stopped.
- **View**: 3D (drag to turn, wheel to zoom) or flat (from above); **Reset
  view**; **Turn slowly**.
- **Colour**: **clusters** (k-means of 8 clusters, 2 to 20, each named by its
  three commonest words), **leans to a group** (with a group column: a word
  leans to the group where it is in twice its share of the rows, in 30 of its
  rows or more; grey otherwise) or **one colour**.
- **Labels**: the word and its neighbours, and room for more (the most
  common words first), none over another, with a short line to its dot when a
  label moved aside.
- **Export PNG**.

## The nodes

| Look | Node | What it adds |
| --- | --- | --- |
| Terms | TF-IDF (`TFIDFVectorizer`) | tfidf_0 .. columns ([text vectorizers](text_vectorizers.md)) |
| Sentiment | Sentiment Analysis (`SentimentAnalyzer`) | `<text>_compound`, `_positive`, `_neutral`, `_negative`, `_sentiment` beside the others |
| Tokens | Tokenizer (`TextTokenizer`) | token id columns |
| Embeddings | Word Embeddings (`WordEmbeddings`) | word vectors for an Embedding layer ([word embeddings](word_embeddings.md)) |

Prepare's **More nodes** offers these four and Count Vectorizer. Text Clean
keeps the letters, digits and marks of every script when it removes special
characters.

## Python

`pycyxwiz.text` has the same rules: `words`, `terms`, `english_stop_words`,
`stem` and `vader` (`python_doc/pycyxwiz.md` section 9.1).

```python
import pycyxwiz.text as tx

tx.terms("not good at all", stop_words=None, ngram_range=(1, 2))
# ['not', 'good', 'at', 'all', 'not good', 'good at', 'at all']
tx.vader("The food was not good.")["compound"]  # -0.3412...
```

A runnable example on a few reviews written for it (words per label, the
negation pairs, stems, VADER per review, and where a lexicon misreads):
`examples/python/text_words_and_sentiment.py`.
