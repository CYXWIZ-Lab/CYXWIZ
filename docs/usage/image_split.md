# Splitting images and audio

A **Data Split** node splits an image or audio dataset into train,
validation and test, the same way it splits a table:

- **Train ratio**, **Val ratio** and **Test ratio** say each part's share.
  They should add up to 1; the compile warns when they do not.
- **Stratified** (on by default) splits each class on its own, so every part
  keeps the classes in the same shares as the whole set. Off, the whole set
  is shuffled once and cut.
- **Seed** picks which images go to which part. The same seed gives the same
  parts in every run, on every machine. It is not the Data Loader's seed:
  changing the batch order (the loader's seed) never moves an image from one
  part to another.

A graph without a Data Split uses 80 / 10 / 10 in one shuffle (seed 42), as
tables do.

The images split are the ones training reads: unreadable files and images
a Quality Analyzer leaves out are taken out first.

## What each part is for

- **Train**: the model learns from these. Only they are shuffled each epoch
  and only they get the random transforms (see image_transforms.md).
- **Validation**: scored after every epoch, in the same order each time.
- **Test**: kept out of training and scored once at the end of training,
  and by **Train > Test > Run Test** (F8) and **Run Quick Test** (see
  testing.md).

## A dataset for validation or test

Wire another loaded image Data Input into the Data Split's **Validation** or
**Test** input and that dataset is the whole part. The training dataset then
splits into the parts that are left (with a wired Test, 80 / 10 / 10 becomes
90 / 10 / 0 of the training images). A wired dataset goes through the same
image steps (Resize, Normalize, ...) as the training images. Its classes must
be the training dataset's (same names, same order); otherwise the compile
refuses it, since the model's outputs would not line up.

## Seeing the split before training

The counts come from the image index (Data Studio > Bring in builds it when
the images load), so they are known before training starts:

- **Properties > As compiled** on the Data Split: a table with a row per part
  (Train, Validation, Test) and a column per class (up to six classes; more
  show the totals only), a note for each wired dataset and for the images
  left out, and **Show the split in Data Studio**.
- The compile's messages: *Images: 20,000 train · 2,500 validation · 2,500
  test, by class (seed 42).* A class too small for a part's share gets a
  warning (*Class 'b' has 2 images: its validation part is empty*) with the
  fix: more images, or a larger ratio.
- **Data Studio > ML readiness**: a card per part (its share and each class's
  count, or *all of 'holdout'* for a wired dataset), a table of each class in
  each part with its share of that part, and a verdict. The parts count as
  balanced when every class's share in every part is within 2 points of its
  share of the whole set; otherwise the first class that is off is named.
  The counts follow the Data Split's settings as you change them.

## Audio

An audio dataset (a folder of class folders) splits the same way, with the
same three parts, the same settings and the same rule. Its test part is
scored at the end of training as for images. A loaded audio Data Input
wired into the Data Split's Validation or Test input is that part whole,
with the same features; training refuses it, saying why, when its classes
differ from the training dataset's (names and order) or its features have
another shape (load it with the same sample rate and duration).
