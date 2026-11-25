---
title: Explore a text column
description: Read a text column's words, distinctive terms, sentiment and tokens in Explore > Text, and add the matching text nodes.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - text-embeddings.md
  - explore.md
  - ../../usage/text_vectorizers.md
redirect_from:
  - text.md
---

Data Studio's **Explore > Text** tab reads one text column five ways: which
words it uses (**Words**), what sets each group of rows apart (**Terms**),
how positive or negative the rows read (**Sentiment**), what a tokenizer
makes of them (**Tokens**), and which words are used alike
(**Embeddings**, see [Learn word vectors from a text
column](text-embeddings.md)). Each look puts its node into the graph, so the
answer runs with the rest of the pipeline.

**Tools > Text** opens the tab on a look: **Tokenization** opens Tokens,
**Word Frequency** opens Words, **TF-IDF Analysis** opens Terms, **Word
Embeddings** opens Embeddings and **Sentiment Analysis** opens Sentiment;
**Data > Tokenization** opens Tokens too. **Language Model Generation**
stays a window of its own (see
[Language model generation](../evaluation/language-model-generation.md)).

The figures are the same as Python gives: the words, the stop words, the
Porter stems and VADER are checked against Python's `re` and `str.lower`,
scikit-learn's `ENGLISH_STOP_WORDS`, NLTK's `PorterStemmer` and
vaderSentiment 3.3.2; Terms against scikit-learn's `TfidfVectorizer`. The
tab calls the nodes' own code: Terms counts with the TF-IDF node's rule, and
Tokens trains the Tokenizer node's tokenizer.

## Before you start

- A loaded dataset with a text column, shown in Data Studio's rail.

## Read a text column

1. In **Data Studio > Explore**, open the **Text** tab.
2. Pick a **Text column** (its average length and distinct values show on
   hover).
3. Optional: pick **Group by**, a category or the label with 2 to 200
   values. The words, the terms and the sentiment are then also shown per
   group: the 12 largest groups, the rest together as "other", and rows
   without a group as "(missing)".
4. Pick a look. Each runs as a job of its own: it shows its progress, and
   **Stop** ends it ("Stopped. Change a setting to run again.").
5. Optional: **Add as a step** puts the look's node after the data in use
   (see [The nodes](#the-nodes)).

Rows whose text is missing (NULL, or a text the project marks as missing)
or only spaces are counted and left out; every look says how many it read
("52,681 rows with text (362 missing left out)").

## Words, stop words and stems

Every look splits the text the same way, and the TF-IDF and Count
Vectorizer nodes split it that way too:

- a **word** is a run of letters and digits of any script, lower-cased, two
  or more long; an apostrophe between two runs is kept ("don't", "i've"),
  and a typographic apostrophe (’) reads as one. In Python:
  `re.finditer(r"[^\W_]+(?:'[^\W_]+)*", text)`, lower-cased. scikit-learn's
  default `token_pattern` (`\b\w\w+\b`) differs on apostrophes ("don't"
  gives "don") and the underscore (a word character there);
- the **common English words** (stop words) are scikit-learn's
  `ENGLISH_STOP_WORDS` (318) and their contractions (54: "don't", "i'm",
  ...). They include "not", "no" and "nor": leave them in when a negation
  matters (see
  [text vectorizers](../../usage/text_vectorizers.md#n-grams-and-stop-words));
- a **stem** is the Porter (1980) stem ("feeling" and "feels" count as
  "feel"), NLTK's `PorterStemmer(mode="ORIGINAL_ALGORITHM")`.

## Words

The commonest words of the column, or word pairs (**Count**: words | word
pairs), with the common English words left out or kept, and by stem when
asked; **Show the top** N (5 to 200). The page shows the words read, the
distinct ones, the words per row (median, 95th percentile, longest), a bar
chart of the top words, a histogram of the words per row (up to a round cap
near the 95th percentile; the last bar holds the longer rows), each group's
commonest words and the commonest pairs. **Export CSV** writes rank, word,
count and share; **Copy** copies the sentence.

## Terms

TF-IDF over the rows with the TF-IDF node's weights: a term's count in the
row times its smoothed idf, ln((1 + n) / (1 + df)) + 1, the row then
normalised (**Normalise**: L2, L1 or none) over every term the look keeps
(the node keeps its `max_features` commonest, so its normalised values can
differ a little). **Terms**: words, or words and pairs; a term counts when
it is **in at least** N rows (min_df, 5 by default).

- With a group column, **what sets each group apart**: the terms whose mean
  TF-IDF in the group is the most above the table's (each group's six, the
  scores on hover). Without one, the terms that weigh most over the table.
- **A row's terms** (**Row**: the table row, 1 by default) and **the rows
  that read most like it**: the five nearest by the cosine of their TF-IDF
  rows, with their group.
- **Text quality** the terms show: rows with HTML entities (`&amp;`), links,
  repeated rows, and rows with broken characters (UTF-8 read as Latin-1 and
  saved again, such as "Ã©" or "ðÿ", or U+FFFD). Clean them with a Text
  Clean step first when they top the terms.

**Add as a step** adds the TF-IDF node with these settings (the 2,000
commonest terms, max_features) after the data in use: a table of tfidf_0 ..
with the label and the split. **Export CSV** writes group, term, mean
TF-IDF and the lift.

## Sentiment

VADER (Hutto and Gilbert, 2014) on every row: a rated lexicon with rules for
negation ("not good" reads negative), boosters ("very"), capitals, "but",
emoticons and emojis. Each row gets a **compound** score from -1 to 1 and
the shares of its weight that read negative, neutral and positive. A row
reads **positive** at the compound **Positive from** (0.05) and above,
**negative** at **negative at** (-0.05) and below, and neutral between:
VADER's own thresholds. vaderSentiment rounds the compound to 4 places
before such a comparison; the tab compares the unrounded score, so a row at
exactly the threshold after rounding can fall on the other side.

The page shows the negative, positive and neutral rows and the mean
compound, a histogram of the compound over -1 .. 1, a row word by word (each
word with a valence after the rules coloured and scored; **Row** picks it)
and, with a group column, each group's rows, median compound and negative
and positive shares. VADER was made for short social-media text: long rows
add up many words and read more strongly.

**Add as a step** adds the Sentiment Analysis node: the columns
`<text>_compound`, `<text>_positive`, `<text>_neutral`, `<text>_negative`
(numbers) and `<text>_sentiment` ("positive", "neutral" or "negative")
beside the others, the label and the split kept. **Export CSV** writes row,
group, compound and sentiment for every row.

## Tokens

What a tokenizer makes of the rows, with the Tokenizer node's tokenizers
trained on this column as the node trains them: **Tokenizer** (Word by
default; Whitespace, Character, Byte BPE, WordPiece, and SentencePiece BPE /
Unigram when the SentencePiece provider is installed), **Vocabulary** (the
largest vocabulary, max_vocab_size: 10,000), **seen at least** N times
(min_word_freq: 2) and the **Length limit** (max_length: 256 tokens).

The page shows the tokens per row (median, 95th percentile, longest), the
rows the limit cuts (and the share it would cut at twice the limit), the
vocabulary and the share of tokens it covers (not the unknown token), a
histogram of the tokens per row with the limit marked, and a row as the
tokenizer reads it: each token with its id, an unknown token marked (the
first 64; **Row** picks the row). **Add as a step** adds the Tokenizer node
with these settings: token id columns (max_length per row) with the label
and the split, for an Embedding, LSTM or Transformer model. **Copy tokens**
copies the row's tokens and ids.

## The nodes

| Look | Node | What it adds |
| --- | --- | --- |
| Terms | TF-IDF (`TFIDFVectorizer`) | tfidf_0 .. columns ([text vectorizers](../../usage/text_vectorizers.md)) |
| Sentiment | Sentiment Analysis (`SentimentAnalyzer`) | `<text>_compound`, `_positive`, `_neutral`, `_negative`, `_sentiment` beside the others |
| Tokens | Tokenizer (`TextTokenizer`) | token id columns |
| Embeddings | Word Embeddings (`WordEmbeddings`) | word vectors for an Embedding layer ([word embeddings](../../usage/word_embeddings.md)) |

Prepare's **More nodes** offers these four and Count Vectorizer. Text Clean
keeps the letters, digits and marks of every script when it removes special
characters.

## Python

`pycyxwiz.text` has the same rules: `words`, `terms`, `english_stop_words`,
`stem` and `vader` (see the
[Python API reference](../../sdk/python-api/index.md#pycyxwiztext)).

```python
import pycyxwiz.text as tx

tx.terms("not good at all", stop_words=None, ngram_range=(1, 2))
# ['not', 'good', 'at', 'all', 'not good', 'good at', 'at all']
tx.vader("The food was not good.")["compound"]  # -0.3412...
```

A runnable example on a few reviews written for it (words per label, the
negation pairs, stems, VADER per review, and where a lexicon misreads):
`examples/python/text_words_and_sentiment.py`.

## Related

- [Learn word vectors from a text column](text-embeddings.md)
- [Text vectorizer nodes](../../usage/text_vectorizers.md)
- [Explore your data](explore.md)
