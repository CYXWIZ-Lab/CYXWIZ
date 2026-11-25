## How the bank works

Image Gabor runs a bank of striped kernels over every image and gives one
channel per kernel. Each kernel is a small patch of stripes under a Gaussian
window; it answers where the image has stripes of that direction and
spacing, which is what fur, grass, fabric and the edges of one direction
look like up close.

- **Orientations** spread over 180 degrees: 4 gives 0, 45, 90 and 135. The
  angle is the direction the brightness changes in, so 0 degrees answers to
  vertical stripes and 90 to horizontal ones.
- **Wavelengths** are the stripes' spacing in pixels per cycle, split by
  commas or spaces, each 2 or more and given once. Every wavelength gets the
  whole set of orientations: `3, 5` with 4 orientations is 8 channels, at
  most 32.
- **Bandwidth** (octaves) sets the window's size from the wavelength: at 1
  the window's sigma is about 0.56 wavelengths. A smaller bandwidth is a
  wider window that answers to a narrower range of spacings. **Aspect** is
  the window's width across the stripes over its length along them.
- The kernel reaches two sigmas of the window's longer side each way: at
  bandwidth 1 and aspect 0.5, 15 x 15 for wavelength 3 and 25 x 25 for 5. A
  kernel larger than the image's shorter side is refused with the size it
  needs: use a shorter wavelength, a wider bandwidth or a rounder aspect.
- **Response**: `energy` is the size of both phases together (the even,
  cosine kernel and the odd, sine one), so it does not depend on where a
  stripe falls. `even` and `odd` keep one phase with its sign, stored around
  mid-grey (128 = no response). Every kernel has zero mean, so a flat area
  answers 0, and the stored values are scaled so the strongest possible
  answer just fills the range.
- The bank runs once, before the random steps. A later flip or rotation
  moves each channel's picture but does not swap the channels: after a
  horizontal flip the `45 deg` channel holds what a bank on the flipped
  photo would put in `135 deg`. With flips in the graph, read the diagonal
  channels as a pair.
- Colour images are read as grey. **Keep image** keeps the image's channels
  and adds the bank's after them (`[H, W, 3 + 8]`); without it the model
  sees the bank alone.
- The channels are named like `gabor 45 deg 3 px`: each wavelength, its
  orientations in turn. Normalize's Per channel rows and the image sheets use
  the names. See
  [More than three channels](../categories/preprocessing.md#more-than-three-channels).

```text
Data Input (images) -> Resize 64 -> Image Gabor (4 orientations, 3 and 5 px, keep image)
  -> Normalize (same 0.5 / 0.5) -> Conv2D -> ...
```

`examples/cyxgraph/cats_dogs_gabor_bank.cyxgraph` is a small cats-vs-dogs
CNN that reads the photo and the bank; `cats_dogs_gabor_bank.md` next to it
explains it.
