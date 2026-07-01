# Text Vectorizer Nodes

CyxWiz currently supports dense text vectorizer outputs for classic NLP graphs.

## TFIDFVectorizer

Use `TFIDFVectorizer` when term importance should depend on both local term
frequency and how rare the term is across the corpus.

Good fit:

- sentiment classification
- topic classification
- document categorization
- classic dense MLP baselines

Important parameters:

```text
text_col       required text column
label_col      optional label column copied to y
max_features   dense output width cap
min_df         minimum document frequency
ngram_range    1,1 | 1,2 | 2,2
stop_words     english | none
norm           l1 | l2 | none
use_idf        true | false
smooth_idf     true | false
output_format  dense
```

For sentiment tasks, consider:

```text
ngram_range=1,2
stop_words=none
```

This preserves negation phrases such as `not good`.

## CountVectorizer

Use `CountVectorizer` when raw term frequency or term presence is preferable to
TF-IDF weighting.

Good fit:

- simple bag-of-words baselines
- linear/classic ML models
- feature ablation against TF-IDF
- binary term-presence experiments

Important parameters:

```text
text_col       required text column
label_col      optional label column copied to y
max_features   dense output width cap
ngram_range    1,1 | 1,2 | 2,2
stop_words     english | none
norm           l1 | l2 | none
binary         true | false
output_format  dense
```

With:

```text
binary=true
norm=none
```

features are 0/1 term-presence values.

With:

```text
binary=true
norm=l1 or l2
```

the row is converted to presence values first, then normalized.

## TextTokenizer

Use `TextTokenizer` when the model expects token IDs rather than dense feature
columns.

Good fit:

- sequence models
- embedding layers
- recurrent/attention-style experiments
- token-level NLP workflows

Do not use TF-IDF or CountVectorizer when the next model layer expects token
IDs. Use `TextTokenizer` for that path.

## Dense memory cost

Dense vectorizers allocate feature columns roughly as:

```text
rows * max_features * 4 bytes
```

This is only the raw float32 feature matrix. Actual peak memory can be higher
because materialization also needs vocabulary maps, token counts, Arrow builders,
labels, and training batches.

Example:

```text
52,681 rows * 8,000 features * 4 bytes ~= 1.69 GB
```

Before scaling beyond this size, reduce `max_features`, sample rows, or wait for
the sparse feature path.

## Understanding unigrams, bigrams, and stop words

The `ngram_range` property controls whether the vectorizer uses single words,
two-word phrases, or both.

```text
ngram_range=1,1
```

Uses unigrams only. A unigram is one token:

```text
good
bad
panic
help
not
```

```text
ngram_range=1,2
```

Uses unigrams and bigrams. A bigram is two neighboring tokens:

```text
not good
feel bad
panic attack
need help
no hope
```

```text
ngram_range=2,2
```

Uses bigrams only.

For sentiment tasks, `ngram_range=1,2` is useful because phrases can change the
meaning of individual words:

```text
good
not good
```

These should not be treated as the same signal.

The `stop_words` property controls whether common English words are removed.

```text
stop_words=english
```

Removes common English stop words before features are built.

```text
stop_words=none
```

Keeps all tokenized words. This is often better for sentiment because words such
as `not` and `no` can be important.

To check a TF-IDF or CountVectorizer node in Studio, open the node properties
and inspect:

```text
ngram_range
stop_words
```

For the sentiment candidate graph, the intended settings are:

```text
ngram_range=1,2
stop_words=none
```

That means the graph uses single words plus two-word phrases, and it keeps
negation words such as `not` and `no`.

## Sparse output status

The current executable contract is:

```text
output_format=dense
```

Sparse output is planned, but it needs a real storage and tensor path before it
can be enabled. Requests such as `output_format=sparse` fail closed instead of
silently producing dense output.

Implementation tracking:

```text
docs/Data Studio/tofix45.md
```
