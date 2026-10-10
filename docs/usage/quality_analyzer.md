# Quality Analyzer

The **Quality Analyzer** leaves poor images out of training: blurry ones, ones
that are too dark or too bright, flat low-contrast ones, and near-duplicates.
It is a filter on the dataset. The files stay on disk; training skips them.

```
Data Input (images) -> Quality Analyzer -> Data Split -> Data Loader
  -> Resize 64 -> ... -> Conv2D -> ...
```

Put it before **Data Split**: it decides which images exist for training,
and the Split then shares out only the images that pass (a rejected image
never lands in validation or test). Its pins carry the dataset, like the Data
Input's. One Quality Analyzer per graph. It works on image Data Inputs (folders of
image files, with class subfolders or a labels CSV).

## Using it

1. Load the images in the Data Input and add a **Resize** node.
2. Double-click the Quality Analyzer (or **Open Dialog** in Properties).
3. Click **Analyze**. Every image is decoded at the Resize size, as the model
   sees it, and measured once on the GPU. The work also shows in Task View,
   so you can close the dialog and keep working; **Stop** ends it.
4. The dialog shows how many images pass, a row per reason with the worst
   examples, a histogram per measurement with the cut marked, and a warning
   when one class loses a clearly larger share than another.
5. Change a check and the verdict updates at once: the measurements are kept,
   so only adding or changing images, or changing the Resize size, needs
   **Analyze again**.
6. **OK** saves the checks. Training then leaves the rejected files out of
   every split.

Training refuses to start when the node has no analysis for the current
images at the current Resize size, and says so on the Quality Analyzer:
"the images are not analyzed yet" or "the images or the Resize size changed
since the analysis". It also refuses checks that leave out every image.

## The checks

| Check | Measured as | Default |
| --- | --- | --- |
| Blur | Variance of the 3x3 Laplacian of the luminance; low = blurry | reject below 700 |
| Brightness | Mean luminance, 0 to 255 | reject below 50 or above 220 |
| Contrast | Standard deviation of the luminance / 255 | reject below 0.12 |
| Near-duplicates | 64-bit difference hash; images within 4 bits match | on; the first of each group stays |

The luminance is `0.299 R + 0.587 G + 0.114 B` (OpenCV's RGB to grey). The
measurements match OpenCV: `cv2.Laplacian(ksize=1)` with its default
reflect-101 border, and `cv2.resize(INTER_AREA)` to 9 x 8 for the hash.

The numbers depend on the image size. The blur threshold of 100 often quoted
for OpenCV is meant for full-size photos; at 64 x 64 almost every image is far
above it. The defaults here come from measuring a real set at the Resize size
(see below); look at the blur histogram and the examples before you move them.

An image that cannot be decoded measures as black and is left out as too dark.

## Where the analysis is kept

In `<project>/cache/quality/` when a project is open, otherwise in
`<temp>/cyxwiz/quality/`. Each file is named after a key made from every
image's path, size and modification time and the Resize size, so a changed
dataset never reuses an old analysis. Deleting the folder only means analyzing
again.

## Example: cats and dogs

300 images (150 cats, 150 dogs) at Resize 64 x 64, default checks:

- Measured in about 0.5 s on a GeForce GTX 1050 Ti (CUDA).
- 22 of 300 left out: 14 blurry, 9 low contrast, 2 too dark, 1 too bright;
  4 fail more than one check. No near-duplicates.
- Cats lose 16, dogs 6, so the dialog warns that rejections lean to one class.
- Blur at 64 x 64: 5th percentile 713, median 1778. Brightness: median 119.
  Contrast: median 0.211.

## Notes

- Measured on the GPU with ArrayFire; without a GPU the same code runs on
  ArrayFire's CPU backend. A build without ArrayFire refuses the node.
- Code export does not include the filter yet and says so; train these graphs
  in Studio.
- Checked against OpenCV: `image_quality_opencv_parity` compares every
  measurement on CUDA, OpenCL and the ArrayFire CPU backend.
  `image_transform_graph_contract` covers the training refusals (no analysis,
  a new Resize size, an added image, bad checks, two analyzers) and the batcher
  leaving the files out.
