---
title: Classify audio
description: Train the Audio classification starter on 400 spoken digits and learn how a mel spectrogram turns sound into features.
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
  - ../user-guide/data-studio/split-images-and-audio.md
  - ../user-guide/training/checkpoints-and-early-stopping.md
---

In this lesson you train a model that recognises the spoken digits zero to
nine. A clip of sound is a long row of air-pressure samples; the graph
turns it into a mel spectrogram, a picture of which pitches sound when,
and a small network learns the digits from it.

**You need:** lesson 2 done, the **Audio classification** starter
(`examples/cyxgraph/speech_command_classifier.cyxgraph`) and its data
`examples/datasets/spoken_digits`: 400 WAV clips at 8 kHz, the digits 0-9
spoken by four speakers, one folder per digit (Free Spoken Digit Dataset).
About 15 minutes.

## 1. Open the starter and its data

1. On an empty canvas, select **Open an example graph**, then **Audio
   classification**. Press **F** to fit it in view.
2. Double-click the **Data Input**. On **Source**, **File or folder** is the
   `spoken_digits` folder and **Detected** says what it found: class
   folders `0` to `9`. The rows are used as **Audio**, at a sample rate of
   8,000 per second, each clip cut or padded to 1 second.
3. Select **OK**. It loads the clips and closes the dialog.

The label of each clip is the name of its folder. To train on your own
sounds, point the Data Input at a folder of class folders the same way.

## 2. Follow a clip through the graph

| Node | What it does here |
| --- | --- |
| **Train/Val Split** | 80 / 10 / 10 into training, validation and test clips (the name says Train/Val; it keeps a test part too). |
| **DataLoader** | Batches of 16, shuffled, up to 60 epochs; it stops early when the validation loss has not improved for 15 checks. |
| **Mel Spectrogram** | Cuts the clip into overlapping windows of 512 samples, 256 apart, and measures each window's energy in 40 pitch bands spaced as the ear hears them, on a log scale. |
| **Normalize** | Keeps the values as they are here (mean 0, std 1); set it to your data's mean and spread when you change the data. |
| **Flatten** | Lines up the spectrogram (bands x windows) into one row. |
| **BatchNorm** | Scales each of those values to a common range from batch to batch. Without it this network learns very slowly. |
| **Dense 256**, **ReLU**, **Dropout 0.3**, **Dense 128**, **ReLU**, **Dropout 0.2** | Two hidden layers, with dropout against learning the 320 training clips by heart. |
| **Dense N**, **CrossEntropy**, **Adam**, **Output** | Ten scores, the loss, the optimizer (learning rate 0.001) and the end, as in lesson 2. |

When an audio graph has no feature node, the compile report offers **Add
Mel Spectrogram**, **Add Spectrogram** and **Add MFCC** to add one.

## 3. Check the shapes

Press **F7**, then open **Configuration** in the report. The input per
sample is the size of the flattened spectrogram; the parameters line shows
that most of them sit in the first Dense layer.

## 4. Train and test

1. Press **F5**. Watch the dashboard: with 60 epochs planned, the run may
   end with "Stopped early: the validation loss did not improve for 15
   epochs." That is early stopping doing its job, and the active model is
   the best validation checkpoint.
2. Press **F8**. In **Results > Test**, open **Confusion matrix**. Digits
   that sound alike are the usual mix-ups.

With only four speakers, the test clips come from the same voices as the
training clips. A new voice scores lower: test on a speaker the model never
heard before you trust it.

## 5. Try a change

1. Click the **Mel Spectrogram** node and set **N Mels** to `64`.
2. Press **F5**, then **F8**, and compare with **Train > Test > Compare
   Test Results...**.

## Next steps

- Split audio by speaker or by file: [Split images and audio](../user-guide/data-studio/split-images-and-audio.md)
- Early stopping and the best checkpoint: [Checkpoints, early stopping and resume](../user-guide/training/checkpoints-and-early-stopping.md)
- Look at signals in Data Studio: [Signal processing](../user-guide/data-studio/signal-processing.md)
- [Train a time-series forecaster](time-series-forecasting.md)
