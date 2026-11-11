# A Gabor filter bank: 11-channel images into a CNN

`cats_dogs_gabor_bank.cyxgraph` is a small cats-vs-dogs classifier whose
images carry eight more channels: **Image Gabor** runs a bank of striped
kernels (4 orientations at 2 wavelengths) over each photo and, with **Keep
image** on, adds one channel per filter after R, G and B. The first Conv2D
reads `[64, 64, 11]`.

```
Data Input -> Split 80/10/10 -> Loader (batch 64, 8 epochs) -> Resize 64
  -> Image Gabor (4 orientations, wavelengths 3 and 6, keep image)
  -> Horizontal Flip -> Normalize (same: mean 0.5, std 0.5)
  -> Conv 16 -> BN -> ReLU -> Conv 32 /2 -> BN -> ReLU -> Conv 64 /2 -> BN -> ReLU
  -> Global Avg Pool -> Dense 1 -> BCE with Logits -> AdamW -> Output
```

Each filter answers where the image has stripes of its direction and spacing:
fur, whiskers, the edge of an ear. A filter at 0 degrees answers to vertical
stripes, at 90 degrees to horizontal ones.

## Data

Any folder of labelled images with two classes. It was made for the Kaggle
"Dogs vs. Cats" set (25,000 JPEGs with a CSV of `image,labels`); the example
saves no path.

## Run it

1. Open the graph (**File > Open**, or the Asset Browser).
2. Double-click **Cats_Dogs Data**, pick your image folder and how its labels
   are given, then **OK**.
3. **Data Studio > Prepare** shows each step on a sample: the Image Gabor row
   has the photo with `+8`; hover it for the sheet (R, G, B, then
   `channel 4 · gabor 0 deg 3 px` to `channel 11 · gabor 135 deg 6 px`).
4. **Train**. The images are prepared once with their eleven channels.

## Try

- Turn **Keep image** off: the model sees the bank alone, `[64, 64, 8]`.
- **Orientations** 8, or **Wavelengths** `3, 6, 12`: more channels, at most 32.
  A long wavelength needs a large kernel; the node says so when it does not
  fit the image.
- **Response** `even` or `odd`: one phase, stored around mid-grey, instead of
  the energy of both.
- The Horizontal Flip after the bank mirrors each channel's picture but does
  not swap the 45 and 135 degree channels; read those two as a pair.
- After a Local Debug run with Layer capture on, the Studio Debugger's
  Inspector shows the sample as the same sheet.

Rules for images with more than three channels:
`docs/usage/image_transforms.md`, section More than three channels.
