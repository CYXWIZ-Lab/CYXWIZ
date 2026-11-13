# RGB + edges: a 4-channel image into a CNN

`cats_dogs_edges_channel.cyxgraph` is a small cats-vs-dogs classifier whose
images carry one more channel: **Image Edges** with **Keep image** on keeps
the photo and adds its Canny edge map, so the first Conv2D reads
`[64, 64, 4]` (R, G, B, edges).

```
Data Input -> Split 80/10/10 -> Loader (batch 64, 8 epochs) -> Resize 64
  -> Image Edges (canny, keep image) -> Horizontal Flip
  -> Normalize (same: mean 0.5, std 0.5)
  -> Conv 16 -> BN -> ReLU -> Conv 32 /2 -> BN -> ReLU -> Conv 64 /2 -> BN -> ReLU
  -> Global Avg Pool -> Dense 1 -> BCE with Logits -> AdamW -> Output
```

## Data

Any folder of labelled images with two classes. It was made for the Kaggle
"Dogs vs. Cats" set (25,000 JPEGs with a CSV of `image,labels`); the example
saves no path.

## Run it

1. Open the graph (**File > Open**, or the Asset Browser).
2. Double-click **Cats_Dogs Data**, pick your image folder and how its labels
   are given, then **OK**.
3. **Data Studio > Prepare** shows each step on a sample: the Image Edges row
   has the photo with `+1`; hover it for the sheet (R, G, B and
   `channel 4 · edges`).
4. **Train**. The images are prepared once with their four channels.

## Try

- Turn **Keep image** off: the model sees the edge map alone, `[64, 64, 1]`.
- Change **Method** to `sobel` with **Direction** `x`, or to `log` / `dog`.
- Set Normalize to **Per channel**: it asks for four values, named R, G, B and
  edges (**Data Studio > Explore > Profile > Use in Normalize** fills them).
- After a Local Debug run with Layer capture on, the Studio Debugger's
  Inspector shows the sample as the same sheet, and **Where it looks** covers
  all four channels.

Rules for images with more than three channels:
`docs/usage/image_transforms.md`, section More than three channels.
