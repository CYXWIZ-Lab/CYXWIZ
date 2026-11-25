---
title: Classify text
description: Train the Text classification starter on 3,000 review sentences and learn what a tokenizer and an embedding do.
type: tutorial
area: training
audience: user
status: experimental
since: 1.0.8
owner: docs
last_reviewed: 2026-11-25
related:
  - index.md
  - first-model-from-a-starter.md
  - ../user-guide/data-studio/text.md
  - ../user-guide/evaluation/test-results.md
---

In this lesson you train a model that tells positive review sentences from
negative ones. A model reads numbers, not words, so the graph first turns
each sentence into a row of word numbers (a tokenizer), then each word
number into a learned vector (an embedding).

**You need:** lesson 2 done, the **Text classification** starter
(`examples/cyxgraph/review_sentiment_classifier.cyxgraph`) and its data
`examples/datasets/review_sentences.csv`: 3,000 sentences from Amazon, IMDb
and Yelp reviews, half positive, half negative. About 15 minutes.

## 1. Open the starter

1. On an empty canvas, select **Open an example graph**, then **Text
   classification** (or use the start page's **Task starter graphs**).
2. Press **F** to fit the graph in view.

## 2. Follow a sentence through the graph

| Node | What it does here |
| --- | --- |
| **Data Input** | Reads the file as **Text**: the text column `sentence` and the label column `sentiment` (*positive* or *negative*). |
| **Tokenizer (word)** | Splits each sentence into words, lowercases them, and gives each of the 5,000 most frequent words (seen at least twice) a number. Each sentence becomes 48 numbers: longer ones are cut, shorter ones padded with 0. |
| **Stratified Split** | 80 / 10 / 10, the same share of positive and negative in each part, seed 42. |
| **DataLoader** | Batches of 64, shuffled, 8 epochs. |
| **Embedding** | Turns each word number into a vector of 64 learned values; 0 (padding) stays empty. Its size follows the tokenizer's vocabulary, 5,000, whatever the node's name says. |
| **Flatten** | Lines up the 48 vectors into one row of values. |
| **Dense 128**, **ReLU**, **Dropout** | A hidden layer. Dropout switches off half of its values at random while training, so the model cannot lean on a few words. |
| **Dense N** | Two scores, one per class. |
| **CrossEntropy**, **Adam**, **Output** | The loss, the optimizer (learning rate 0.0002) and the end of the model, as in lesson 2. |

Double-click the tokenizer: its settings are in a dialog, **Text Tokenizer
Configuration**.

## 3. Train and test

1. Press **F7**. The compile report should say **Ready to train**.
2. Press **F5**. The eight epochs take about a minute or less on a CPU.
3. Read the Training Dashboard. On such short sentences, expect a held-out
   accuracy well above chance (50 %) but far from perfect.
4. Press **F8**. **Results > Test** opens.

## 4. Read the mistakes

1. In **Results > Test**, select **Classes**. The classes carry their names,
   *negative* and *positive*, taken from the label column.
2. Select **Errors**. It counts the wrong rows and lists the most confident
   mistakes first.
3. Select **Predictions**: each test row with its row in the dataset, the
   true and the predicted class and the confidence. It can list the
   mistakes only, most confident first.

Sentences with irony ("not bad at all") or words the tokenizer never kept
are the usual mistakes: the model sees only which words appear where, not
what they mean together.

## 5. Try a change

1. Click the **Dropout** node and set **Rate** to `0.3`.
2. Press **F5**, then **F8**, and compare with **Train > Test > Compare
   Test Results...**.

## Next steps

- Clean and explore text in Data Studio: [Text](../user-guide/data-studio/text.md)
- Every field of the tokenizer and the embedding: [Node dialogs](../main-window/node-dialogs.md)
- Bigger language models: [Fine-tune language models](../user-guide/training/fine-tune-language-models.md)
- [Classify audio](audio-classification.md)
