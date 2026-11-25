---
title: Compare two TF-IDF text classifiers
description: Train a unigram and a unigram-plus-bigram TF-IDF classifier on the same split, and compare them fairly before replacing one with the other.
type: tutorial
area: training
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - text-classification.md
  - ../user-guide/data-studio/text.md
  - ../usage/text_vectorizers.md
redirect_from:
  - ../usage/sentiment_text_classification.md
---

In this lesson you train two text classifiers that differ only in their
text features: one counts single words (unigrams), the other also counts
word pairs (bigrams). You learn why stop words matter for sentiment, how
much memory a dense TF-IDF table takes, and how to compare two graphs so
that the better one wins on evidence, not on training accuracy.

**You need:** lesson [Classify text](text-classification.md) done, the two
graphs in `examples/cyxgraph/Sentiment analysis/`, and a CSV with a text
column `statement` and a label column `status`. The graphs were built for
the 52,681-row "Sentiment Analysis for Mental Health" set published on
Kaggle, with seven classes (Anxiety, Bipolar, Depression, Normal,
Personality disorder, Stress, Suicidal); it is not shipped with the
Engine. About 30 minutes, most of it training.

## 1. Look at the text first

1. Load the CSV in Data Studio and open **Explore > Text** on the
   `statement` column (see [Explore a text column](../user-guide/data-studio/text.md)).
2. Read the commonest words and pairs, what sets each class apart
   (**Terms**), VADER sentiment per class, and the token lengths.

Pairs such as "not good" carry meaning a single word does not. That is the
idea the second graph tests.

## 2. Open the unigram graph

1. Open `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph`.
2. Double-click **Sentiment CSV**, choose your CSV, check that the text
   column is `statement` and the label column `status`, and click
   **Apply**. The saved path in the example is not on your computer, so
   this step is needed.
3. If your CSV has another number of classes, set the units of **Dense 7**
   to that number.

| Node | Setting |
| --- | --- |
| **TF-IDF 8000 unigram benchmark** | `max_features` 8000, `min_df` 2, `ngram_range` 1,1, `norm` l2, English stop words |
| **Split 80/10/10** | Stratified, seed 42 |
| **DataLoader** | Batch 128, up to 30 epochs, early stopping after 7 epochs without improvement, seed 42 |
| Layers | Dense 128 -> ReLU -> Dropout 0.3 -> Dense 64 -> ReLU -> Dropout 0.3 -> Dense 7 |
| Loss and optimizer | Cross entropy, AdamW (learning rate 0.0005) |

## 3. Train and test it

1. Press **F7**, then **F5**.
2. When it finishes, press **F8** to test it on the held-out rows.
3. Write down, from the Training Dashboard and **Results > Test**:
   - the device the run used;
   - the epoch it stopped at and its best epoch;
   - the best validation loss and accuracy;
   - the test loss and accuracy;
   - the gap between training and validation accuracy;
   - the training time.

## 4. Train the unigram-plus-bigram graph

1. Open `sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph`
   from the same folder and point **Sentiment CSV** at the same file, as in
   part 2.
2. Look at its TF-IDF node, **Candidate TF-IDF 8000 unigram+bigram tuned**:
   `min_df` 1, `ngram_range` 1,2 and `stop_words` none. Its layers are
   smaller: Dense 64 -> Dense 32 -> Dense 7.
3. Train and test it as in part 3, on the same device, and write down the
   same figures.

`stop_words` none is deliberate. The English stop-word list (scikit-learn's
`ENGLISH_STOP_WORDS` and their contractions) removes negation words such
as `not`, `no` and `nor`, so with it a feature such as `not good` could
never be built.

## 5. Compare the two

Use the same data file, split, seed, device, batch size, epoch limit and
early stopping for both graphs; the two examples already share them. Then:

- Prefer the second graph only when its **validation and test** figures
  are better. Higher training accuracy alone is not evidence.
- If a run fell back to another device, or a device was not available,
  note it: do not compare a GPU run with a CPU run.

For reference, the two feature sets were also compared outside the Engine,
on the full 52,681 rows with the same stratified 80/10/10 split and one
scikit-learn `SGDClassifier`. This isolates the quality of the features;
it is not a result of the Engine's network:

| Features | Validation accuracy | Test accuracy |
| --- | ---: | ---: |
| Unigrams | 75.57 % | 76.62 % |
| Unigrams and bigrams | 78.28 % | 77.76 % |
| Change | +2.71 points | +1.14 points |

`sentiment_text_feature_benchmark.json` in the same folder records the
dataset's SHA-256, class counts, split sizes and the full results;
`benchmark_sentiment_text_features.py --dataset <your CSV>` repeats the
comparison.

## Mind the memory

TF-IDF writes one column per kept term. For 52,681 rows and 8,000 float32
features, the dense values alone take about 1.69 GB, before the
vocabulary, the Arrow builders, the labels and the training batches. The
TF-IDF node checks the memory it needs before it builds the table and
refuses a run that does not fit. Then lower `max_features`, use fewer
rows, or set `output_format` to *sparse*, which keeps only the non-zero
values (see [Text vectorizers](../usage/text_vectorizers.md)).

## Next steps

- [Explore a text column](../user-guide/data-studio/text.md): Data
  Studio's **Explore > Text**.
- [Text vectorizers](../usage/text_vectorizers.md): every TF-IDF and Count
  setting.
- [Test a trained model](../user-guide/evaluation/test-a-trained-model.md)
