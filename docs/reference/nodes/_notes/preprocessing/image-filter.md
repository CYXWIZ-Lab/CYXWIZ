## Kernels

Image Filter runs one kernel you choose over every image, each channel
alone (`cv::filter2D`): each output pixel is the sum of its neighbours times
the kernel, the kernel's centre on the pixel. It is what a Conv2D filter
does, with numbers you set instead of learned ones, so it shows what a
kernel does before a network learns its own (compare with the Studio
Debugger's layer filters).

| `preset` | Kernel | What you get |
| --- | --- | --- |
| `identity` | `0 0 0; 0 1 0; 0 0 0` | the image unchanged |
| `box` | nine times 1/9 | a 3 x 3 average (blur) |
| `sharpen` | `0 -1 0; -1 5 -1; 0 -1 0` | stronger edges, same brightness |
| `emboss` | `-2 -1 0; -1 1 1; 0 1 2` | a relief lit from the top-left |
| `outline` | `-1 -1 -1; -1 8 -1; -1 -1 -1` | edges in every direction |
| `prewitt_x`, `prewitt_y` | `-1 0 1` in each row; `-1 -1 -1; 0 0 0; 1 1 1` | left-right and top-bottom changes |
| `roberts_x`, `roberts_y` | `1 0; 0 -1` and `0 1; -1 0` | diagonal changes, 2 x 2 |
| `kirsch` | `5 5 5; -3 0 -3; -3 -3 -3` turned to `direction` | edges brighter on that side, strong |
| `robinson` | `1 2 1; 0 0 0; -1 -2 -1` turned to `direction` | the same, gentler (Sobel weights) |
| `custom` | the `kernel` you type | anything |

- **Typing a kernel.** Rows are split by `;` (or a new line), numbers by
  spaces or commas: `1 2 1; 2 4 2; 1 2 1`. Rows must have the same length; a
  kernel may be 1 to 31 numbers each way and need not be square, but not
  larger than the image. `normalize` divides it by its sum, so
  `1 2 1; 2 4 2; 1 2 1` becomes a blur that keeps the brightness; a kernel
  that sums to 0 is left as typed.
- **It is a correlation**, as in OpenCV and in Conv2D: the kernel is laid on
  the image as written, not flipped. Flip your kernel yourself for a
  textbook convolution; symmetric kernels are the same either way.
- **`direction`** (Kirsch and Robinson) is the side the edge is brighter
  on: `n` answers to a bright band above a dark one, `e` to bright on the
  right, and so on, in steps of 45 degrees.
- **`border`** is what the kernel reads past the edge: `reflect_101`
  mirrors without repeating the edge pixel (OpenCV's default), `reflect`
  mirrors with it, `replicate` repeats the edge pixel, `zero` reads black.
- **`output`.** The images are stored as 0 to 255, and an edge kernel
  answers with negative numbers too. `clip` cuts below 0 and above 255
  (right for blurs and sharpening); `absolute` keeps the size of the answer
  (an edge map, either sign); `offset` stores it around mid-grey, 128 = no
  answer (keeps the sign; this is how emboss is usually shown). With `clip`
  and a kernel that sums to 0, the compile report says so.
- The channels stay: a colour image stays colour. Put **Grayscale** before
  the filter for one channel.

```text
Data Input (images) -> Resize 64 -> Image Filter (outline, absolute)
  -> Normalize -> Conv2D -> ...
```

`examples/cyxgraph/cats_dogs_image_filter.cyxgraph` is a small cats-vs-dogs
CNN with a typed kernel in front; `cats_dogs_image_filter.md` next to it
explains it.
