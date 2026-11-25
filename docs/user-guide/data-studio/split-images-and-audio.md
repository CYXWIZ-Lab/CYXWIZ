---
title: Split images and audio
description: Split an image or audio dataset into train, validation and test with a Data Split, and see the parts before training.
type: how-to
area: data-studio
audience: user
status: experimental
since: 1.0.8
owner: data-studio
last_reviewed: 2026-11-25
related:
  - quality-analyzer.md
  - validate.md
  - ../evaluation/test-a-trained-model.md
redirect_from:
  - split-images-and-audio.md
---

A **Data Split** node splits an image or audio dataset into train,
validation and test, the same way it splits a table. The split is known
before training starts, so you can check that every class lands in every
part.

## Before you start

- An image or audio Data Input, loaded. An image dataset's index is built
  when the images load (see [Bring in data](bring-in-data.md)); the counts
  come from it.
- A **Data Split** node after the Data Input (and after a
  [Quality Analyzer](quality-analyzer.md), when you use one).

## Steps

1. On the canvas, select the **Data Split**.
2. In Properties (or the Data Split's dialog), set **Train ratio**, **Val
   ratio** and **Test ratio**.
   They should add up to 1; the compile warns when they do not.
3. Leave **Stratified** on (the default) to split each class on its own, so
   every part keeps the classes in the same shares as the whole set. Off,
   the whole set is shuffled once and cut.
4. Optional: change **Seed**. The same seed gives the same parts in every
   run, on every machine.
5. Compile (F7) and read the split in **Properties > As compiled** or in
   **Data Studio > ML readiness** (see [See the split before
   training](#see-the-split-before-training)).

The Data Split's seed is not the Data Loader's: changing the batch order
(the loader's seed) never moves an image from one part to another. A graph
without a Data Split uses 80 / 10 / 10 in one shuffle (seed 42), as tables
do.

The images split are the ones training reads: unreadable files and images
a Quality Analyzer leaves out are taken out first.

## What each part is for

- **Train**: the model learns from these. Only they are shuffled each epoch
  and only they get the random transforms (see
  [image transforms](../../usage/image_transforms.md)).
- **Validation**: scored after every epoch, in the same order each time.
- **Test**: kept out of training and scored once at the end of training,
  and by **Train > Test > Run Test** (F8) and **Run Quick Test** (see
  [Test a trained model](../evaluation/test-a-trained-model.md)).

## Use another dataset for validation or test

Wire another loaded image Data Input into the Data Split's **Validation** or
**Test** input, and that dataset is the whole part. The training dataset
then splits into the parts that are left (with a wired Test, 80 / 10 / 10
becomes 90 / 10 / 0 of the training images). A wired dataset goes through
the same image steps (Resize, Normalize, ...) as the training images. Its
classes must be the training dataset's (same names, same order); otherwise
the compile refuses it, since the model's outputs would not line up.

## See the split before training

- **Properties > As compiled** on the Data Split: a table with a row per
  part (Train, Validation, Test) and a column per class (up to six classes;
  more show the totals only), a note for each wired dataset and for the
  images left out, and **Show the split in Data Studio**.
- The compile's messages: *Images: 20,000 train · 2,500 validation · 2,500
  test, by class (seed 42).* A class too small for a part's share gets a
  warning (*Class 'b' has 2 images: its validation part is empty*) with the
  fix: more images, or a larger ratio.
- **Data Studio > ML readiness**: a card per part (its share and each
  class's count, or *all of 'holdout'* for a wired dataset), a table of each
  class in each part with its share of that part, and a verdict. The parts
  count as balanced when every class's share in every part is within 2
  points of its share of the whole set; otherwise the first class that is
  off is named. The counts follow the Data Split's settings as you change
  them. More in [Check a dataset before training](validate.md).

## Audio

An audio dataset (a folder of class folders) splits the same way, with the
same three parts, the same settings and the same rule. Its test part is
scored at the end of training as for images. A loaded audio Data Input
wired into the Data Split's Validation or Test input is that part whole,
with the same features. Training refuses it, saying why, when its classes
differ from the training dataset's (names and order) or its features have
another shape: load it with the same sample rate and duration.

## Related

- [Leave poor images out with the Quality Analyzer](quality-analyzer.md)
- [Check a dataset before training](validate.md)
- [Test a trained model](../evaluation/test-a-trained-model.md)
- [Explore your data](explore.md)
