# Sentiment Text Classification Graphs

CyxWiz keeps two sentiment TF-IDF MLP graphs for comparison.

## Benchmark graph

Path:

```text
examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph
```

Purpose:

- Stable reference graph.
- Uses unigram-only TF-IDF.
- Use this to compare whether newer text-feature changes actually improve validation and test accuracy.

Key feature settings:

```text
max_features=8000
min_df=2
ngram_range=1,1
norm=l2
```

## Candidate graph

Path:

```text
examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph
```

Purpose:

- Active accuracy experiment.
- Uses unigram+bigram TF-IDF.
- Keeps negation words by setting `stop_words=none`.

Key feature settings:

```text
max_features=8000
min_df=2
ngram_range=1,2
stop_words=none
norm=l2
```

## How to compare

Train the benchmark first, record:

- best validation accuracy
- final test accuracy
- early stopping epoch
- train/validation gap

Then train the candidate and compare the same fields.

The candidate should replace the benchmark only if it improves validation/test
accuracy, not just training accuracy.

## Notes

TF-IDF materialization is CPU/RAM based today. Dense memory cost is roughly:

```text
rows * max_features * 4 bytes
```

For 52,681 rows and 8,000 features, raw dense feature values are about 1.69 GB
before Arrow builders, temporary term maps, labels, and training batches.
