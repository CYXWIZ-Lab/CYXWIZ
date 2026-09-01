# Sentiment Text Classification Graphs

CyxWiz keeps one unigram benchmark and one unigram-plus-bigram candidate so
text-feature experiments can be compared without silently replacing the
reference graph.

## Benchmark graph

```text
examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph
```

Current TF-IDF settings:

```text
max_features=8000
min_df=2
ngram_range=1,1
norm=l2
output_format=dense
```

Use this graph as the stable unigram reference.

## Candidate graph

```text
examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph
```

Current TF-IDF settings:

```text
max_features=8000
min_df=1
ngram_range=1,2
stop_words=none
norm=l2
output_format=dense
```

`stop_words=none` is intentional: the engine's English stop-word list removes
negation words, which would prevent features such as `not good` from being
built.

## Reproducible comparison

Use the same dataset snapshot, split, seed, runtime device, batch size, epoch
limit, and early-stopping configuration for both graphs. Record at least:

- requested and effective runtime backend/device
- completed and best epoch
- best validation loss and accuracy
- final test loss and accuracy
- training/validation accuracy gap
- stop reason and fallback count
- wall-clock training time

The candidate should replace the benchmark only when validation/test evidence
improves, not merely training accuracy. Record failed or unavailable device
routes as explicit skips; do not silently pass through another backend.

## Reference feature benchmark

On 2026-09-01, the two tracked feature configurations were evaluated on the
52,681-row dataset with the same stratified 80/10/10 split and one fixed sparse
scikit-learn `SGDClassifier`. This isolates text-feature quality; it is not a
CyxWiz MLP or device-lifecycle result.

| Features | Validation accuracy | Test accuracy |
|---|---:|---:|
| Unigram benchmark | 75.57% | 76.62% |
| Unigram+bigram candidate | 78.28% | 77.76% |
| Candidate change | +2.71 points | +1.14 points |

The dataset SHA-256, class counts, split sizes, estimator budget, sparse-matrix
sizes, timings, and full-precision results are recorded in
`examples/cyxgraph/Sentiment analysis/sentiment_text_feature_benchmark.json`.
Regenerate the comparison with `benchmark_sentiment_text_features.py`.

The candidate graph remains an experiment rather than an automatic replacement.
A live Engine run must still validate the MLP, requested/effective device,
fallback count, training lifecycle, and final test result before promotion.

## Dense-memory note

For 52,681 rows and 8,000 float32 features, raw dense feature values require
about 1.69 GB before temporary vocabulary structures, Arrow builders, labels,
and training batches. The vectorizer performs a blocking memory preflight; if
the run is rejected, reduce `max_features`, reduce the input rows, or use a
machine with sufficient memory. Sparse execution remains a separate end-to-end
feature tracked by Tofix45.
