---
redirect_from:
  - ../../../usage/text_vectorizers.md
related:
  - ../../../user-guide/data-studio/text.md
  - ../../../tutorials/sentiment-text-classification.md
---

## Text vectorizers

[TF-IDF](../text/tf-idf.md) and [Count Vectorizer](../text/count-vectorizer.md)
turn a text column into one number column per term, for classic NLP models.
Their numbers follow scikit-learn: raw counts, IDF, binary presence, the
vocabulary cap and the optional row normalization. Data Studio's
**Explore > Text > Terms** weighs terms with the same code and adds the
TF-IDF node with its settings
([Text](../../../user-guide/data-studio/text.md#terms)).

Use the [Tokenizer](../text/tokenizer.md) instead when the model expects
token ids (an Embedding, recurrent, attention or transformer path): the
vectorizers give feature columns, not token sequences.

### Words and terms

A row's terms are counted as Data Studio's Text tab counts them (also
`pycyxwiz.text.terms`):

1. **Words**: runs of letters and digits of any script, lower-cased, two or
   more long; an apostrophe between two runs is kept ("don't"), and a
   typographic apostrophe reads as one. In Python:
   `re.finditer(r"[^\W_]+(?:'[^\W_]+)*", text)`, lower-cased. scikit-learn's
   default `token_pattern` (`\b\w\w+\b`) differs: it splits "don't" into
   "don" and keeps the underscore.
2. **Stop words** (`stop_words=english`): scikit-learn's
   `ENGLISH_STOP_WORDS` and their contractions, left out before the n-grams.
3. **N-grams** (`ngram_range`): the word n-grams, shortest first, joined by a
   space.

When `max_features` caps the vocabulary, both nodes keep the most frequent
terms, with a fixed tie-break, and write the columns in alphabetical order.

### TF-IDF and counts

TF-IDF multiplies each raw term count by the IDF (when `use_idf` is on),
then applies the row normalization (`l1`, `l2` or `none`). With smoothed
IDF the weight is:

```text
log((1 + document_count) / (1 + document_frequency)) + 1
```

Count Vectorizer gives the raw term counts, then the optional normalization.
That is not the count divided by the number of words in the row. With
`binary` on, it gives 0/1 presence instead of counts.

### N-grams and stop words

| `ngram_range` | Terms of "not very good" (no stop words) |
| --- | --- |
| `1,1` | not, very, good |
| `1,2` | the above, plus "not very", "very good" |
| `2,2` | "not very", "very good" |
| `1,3` | the above, plus "not very good" |

The widest n-gram is three words. A malformed, reversed or unsupported range
fails before any data is read. Graphs saved with `ngram_min` and `ngram_max`
still load when `ngram_range` is absent; when both are present,
`ngram_range` wins.

English stop words are removed before the n-grams are formed, so a pair
never spans a removed word. The list includes negations such as `not`, `no`
and `nor`, so for sentiment use `ngram_range=1,2` and `stop_words=none`:
that keeps phrase features such as "not good" and "no hope".

### Dense or sparse output

`output_format` picks how the features travel:

- `dense` (default) writes one Arrow column per term. The table takes about
  `rows x max_features x 4` bytes, and building it needs more (token maps,
  vocabulary statistics, labels, batches). Compile and the run's memory
  check report the estimate, and the node stops before tokenizing when the
  configured or available memory would be exceeded.
- `sparse` keeps only the non-zero values (compressed sparse rows on the
  host); training turns just the current batch into a dense tensor. Use it
  for wide vocabularies. It works when the vectorizer is the last
  preprocessing step and the model's first layer is **Dense**, on
  non-sequence batches. Compile refuses, on the node: a second sparse
  vectorizer on the path, a step after it that needs a table, a first model
  layer other than Dense, sequence or time-series training, and a Dev or
  Test dataset that is not sparse. The run stops when the data has no
  labels, when a Dev or Test vocabulary differs from Training's, or when a
  regression target has more than one column. A
  [Truncated SVD](../analytics/truncated-svd.md) right after the vectorizer
  reads the sparse features and gives dense columns, which lifts these
  limits.

### Fitted state

With **Save fitted state** on, a `fit_transform` run writes the selected
vocabulary and, for TF-IDF, the IDF weights to a `.cyxstate.json` file
(**State artifact path**). **Mode** `transform_only` reads it, keeps the fitted column order and width,
refuses node settings that do not match, and gives 0 for terms the fit did
not see, so validation, test and new data get the training columns.

The numbers are checked against scikit-learn 1.8 fixtures.
