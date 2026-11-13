# Image Filter: a hand-typed kernel in front of a CNN

`cats_dogs_image_filter.cyxgraph` is a small cats-vs-dogs classifier whose
images first pass through an **Image Filter** node: one fixed 3 x 3 kernel,
typed by hand, run over every image once when the images are prepared.

```
Data Input -> Split 80/10/10 -> Loader (batch 64, 8 epochs) -> Resize 64
  -> Image Filter (custom: -1 -1 -1; -1 9 -1; -1 -1 -1)
  -> Horizontal Flip -> Normalize (ImageNet)
  -> Conv 16 -> BN -> ReLU -> Conv 32 /2 -> BN -> ReLU -> Conv 64 /2 -> BN -> ReLU
  -> Global Avg Pool -> Dense 1 -> BCE with Logits -> AdamW -> Output
```

## Data

Any folder of labelled images with two classes. It was made for the Kaggle
"Dogs vs. Cats" set (25,000 JPEGs, `cat.N.jpg` / `dog.N.jpg`, with a CSV of
`image,labels`); the example saves no path.

## Run it

1. Open the graph (**File > Open**, or the Asset Browser).
2. Double-click **Cats_Dogs Data**, pick your image folder and how its labels
   are given (class folders, or a flat folder with a labels CSV), then **OK**.
3. **Train**. The first run prepares the images (Resize and the filter, once);
   later runs with the same settings read them from `cache/images`.

## Try

- Select **Image Filter** and change **Kernel** in Properties:
  `outline` with **Output** `absolute` gives an edge map; `emboss` with
  `offset` a relief; `kirsch` with a **Direction** answers to edges on one
  side only; `identity` is the plain image (a baseline to compare with).
- Type your own under `custom`: rows split by `;`, numbers by spaces.
  `1 2 1; 2 4 2; 1 2 1` with **Normalize** on is a soft blur.
- **Data Studio > Explore > Images > What the model sees** shows a picked
  image after each step, so you see the kernel's effect before training.
- After a Local Debug run, **Studio Debugger > Model > Layers > Filters** on
  **Conv 16** shows the kernels the network learned on top of yours.

Settings and rules: `docs/usage/image_transforms.md`, section Image Filter.
