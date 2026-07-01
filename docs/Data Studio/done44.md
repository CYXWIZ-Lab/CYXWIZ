# done44 - Text Feature Quality: N-grams, Dense Contract, and Sentiment Accuracy

## Status

Done. Dense text-feature support is in place. Real sparse execution is split
into `tofix45` because it requires storage, batching, tensor, and layer-math
changes.

## Problem

The sentiment TF-IDF MLP graph improved training accuracy but validation/test
accuracy stayed near 74%. The engine currently supports dense unigram TF-IDF,
but does not support common text-classification upgrades such as bigrams,
sparse feature matrices, or binary count features.

For sentiment analysis, unigrams alone often miss useful phrases:

- `not good`
- `no hope`
- `panic attack`
- `feel better`
- `want help`
- `not okay`

The current `TFIDFVectorizer` rejects `ngram_range` values other than `1,1`.
This blocks a standard next experiment: `ngram_range=1,2`.

## Goal

Improve CyxWiz text feature support so graph authors can build stronger classic
NLP baselines before moving to heavier sequence/transformer models.

## First Implementation Slice

Add n-gram support to:

- `TFIDFVectorizer`
- `CountVectorizer`

Supported parameters:

- `ngram_range`
- `ngram_min`
- `ngram_max`

Initial target:

```text
1,1 unigram only
1,2 unigram + bigram
2,2 bigram only
```

The output may remain dense for now, but preflight/materialization memory
guardrails from `tofix43` should eventually protect large n-gram expansions.

## Progress

Implemented:

- `TFIDFVectorizer` accepts `ngram_range=1,1`, `1,2`, and `2,2`.
- `TFIDFVectorizer` accepts equivalent `ngram_min` and `ngram_max` parameters.
- `TFIDFVectorizer` supports `stop_words=english` and `stop_words=none`.
- `CountVectorizer` accepts `ngram_range=1,1`, `1,2`, and `2,2`.
- `CountVectorizer` accepts equivalent `ngram_min` and `ngram_max` parameters.
- `CountVectorizer` supports `stop_words=english` and `stop_words=none`.
- `CountVectorizer` supports `binary=true` for term-presence features.
- New node defaults include the real text-feature parameters instead of hidden
  or stale legacy parameters.
- Node metadata/properties expose the actual operator contract:
  `text_col`, `label_col`, `max_features`, `norm`, `ngram_range`,
  `ngram_min`, `ngram_max`, and `stop_words`.
- Node metadata now explains dense memory cost, phrase-sensitive n-grams, and
  when TF-IDF or CountVectorizer is the right feature path.
- `output_format=dense` is now explicit for TF-IDF and CountVectorizer.
- `output_format=sparse` fails closed with a clear validation/configuration
  error instead of silently falling back to dense output.
- Runtime capability validation knows supported `ngram_range` values and
  `stop_words` choices before execution.
- Graph compiler warns for large dense text-vectorizer materialization using
  rows, max_features, n-gram range, stop-word mode, and estimated float32
  feature allocation.
- Pipeline execution logs the same dense materialization warning when the
  input Arrow table is available immediately before running TF-IDF,
  CountVectorizer, or legacy TextVectorize.
- `ngram_range` is canonical when present; `ngram_min` and `ngram_max` remain
  compatibility inputs when `ngram_range` is absent.
- Invalid n-gram ranges fail closed with a clear materializer/operator error.
- Sentiment benchmark graph remains unigram-only for baseline comparison.
- New sentiment candidate graph uses unigram+bigram features with `stop_words=none`.
- Focused tests cover operator reset behavior, metadata/runtime contract, and
  pipeline routing for text vectorizer n-gram support.

Graph files:

- `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph`
  is the unigram benchmark reference.
- `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph`
  is the active unigram+bigram candidate for accuracy testing.
- `docs/usage/sentiment_text_classification.md` documents how to compare the
  benchmark and candidate graphs.
- `docs/usage/text_vectorizers.md` documents TF-IDF, CountVectorizer,
  TextTokenizer, n-grams, binary counts, and dense memory cost.

Follow-ups:

- Real sparse storage/tensor path implementation.
- Hard memory budget/fail-fast policy from `tofix43`.

## Follow-up Features

### Sparse Feature Path

- Sparse TF-IDF output.
- Sparse Arrow or internal sparse tensor representation.
- Sparse-aware Linear/Dense input path.
- Avoid dense `rows * features` RAM explosion.

Status: split out. The current engine path is dense:

```text
TFIDFVectorizer / CountVectorizer
  -> dense Arrow float columns
  -> ArrowDataset
  -> ArrowDatasetBatcher
  -> dense Tensor
  -> Dense/Linear layer
```

Sparse support must not be faked inside the vectorizer only. It needs a real
storage and training path:

- sparse dataset representation
- sparse mini-batch representation
- sparse-to-dense fallback rules
- sparse-aware first linear layer or explicit densify node
- debugger/materializer visibility for nnz, density, and memory estimates

Until that exists, `output_format=sparse` fails closed.

### CountVectorizer Binary Mode

Support:

```text
binary=true
```

This is useful when term presence matters more than term frequency.

Status: implemented for the dense CountVectorizer path. With `norm=none`,
features are 0/1 presence values. With `norm=l1` or `norm=l2`, the presence row
is normalized after binary conversion.

### Better Text Feature Preflight

Compiler/preflight should estimate:

- expected vocabulary size
- requested max features
- n-gram range
- dense memory estimate
- likely risk level

### Graph UX

Node properties should clearly explain:

- unigrams
- bigrams
- n-gram range
- dense memory cost
- when to use TF-IDF vs CountVectorizer vs TextTokenizer

## Acceptance Criteria

- `TFIDFVectorizer` accepts `ngram_range=1,2`.
- `TFIDFVectorizer` accepts matching `ngram_min=1`, `ngram_max=2`.
- `CountVectorizer` accepts `ngram_range=1,2`.
- Compiler/materializer errors remain clear for invalid n-gram settings.
- Existing `1,1` graphs behave the same.
- Tests cover unigram, bigram, and mixed unigram+bigram behavior.
- Sentiment graph can be updated to try TF-IDF unigram+bigram features.

## Notes

This ticket is accuracy-driven. It is separate from `tofix43`, which is about
memory safety. However, n-grams can greatly increase vocabulary size, so the two
tickets are related.

Close condition met: dense text-feature quality work is implemented, tested,
documented, and the sparse follow-up ticket exists as `tofix45`.
