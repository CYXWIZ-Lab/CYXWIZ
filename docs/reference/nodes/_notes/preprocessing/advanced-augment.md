## Methods

**Method** picks one augmentation; Properties shows only that method's
settings. Every method runs on training images only, with `probability`
(set it to 1 for torchvision's behaviour); validation and test pass
through unchanged.

- **cutout**: a square centred at a random pixel, clipped at the edges
  (DeVries & Taylor).
- **random_erasing**: a box of random area and aspect ratio (torchvision
  `RandomErasing`); after ten misses the image stays as it is.
- **mixup**, **cutmix**: mix whole batches, see below.
- **randaugment**, **autoaugment**, **trivialaugment**, **augmix**: policies
  of image ops, see below.

The erasing `value` is a pixel value, written before Normalize.

### MixUp and CutMix

As `torchvision.transforms.v2.MixUp` and `CutMix`:

- One weight `lambda` per batch, drawn from Beta(`alpha`, `alpha`); `alpha`
  1.0 draws it uniformly from 0 to 1.
- Each image is mixed with the image before it in the batch (the first with
  the last). mixup blends the two: `lambda * own + (1 - lambda) * other`.
  cutmix pastes a box from the other image covering `1 - lambda` of the
  area, and `lambda` becomes the share left unpasted.
- The labels are mixed with the same `lambda`, so a label becomes, for
  example, 70% cat and 30% dog. Cross Entropy learns from these soft labels,
  so the loss must be Cross Entropy; the compiler refuses other losses.
- `probability` is the chance that a batch is mixed. One mixup or cutmix
  per graph.
- Training accuracy counts a mixed image as right when the model picks the
  class with the larger share of its label.

### RandAugment

As `torchvision.transforms.v2.RandAugment`: each training image takes
`num_ops` ops, each picked at random from 14:

| Op | Strength at `magnitude` m (of 30) |
| --- | --- |
| Identity | none |
| Shear X / Y | shear factor `0.3 m / 30`, either way, about the top-left corner |
| Translate X / Y | `150 / 331` of the width / height `x m / 30` pixels, either way |
| Rotate | `30 m / 30` degrees, either way |
| Brightness / Color / Contrast / Sharpness | factor `1 +/- 0.9 m / 30` |
| Posterize | keeps `8 - round(m / 7.5)` bits |
| Solarize | inverts pixels at or above `1 - m / 30` |
| AutoContrast | stretches each channel to 0..1 |
| Equalize | equalizes each channel's histogram (as uint8, like torchvision) |

The default `num_ops` 2 and `magnitude` 9 are torchvision's. Shear,
translate and rotate use nearest sampling and fill with 0.

### AutoAugment, TrivialAugment and AugMix

These pick ops from the same list with their own strengths:

| Method | Follows | What each image gets |
| --- | --- | --- |
| autoaugment | `v2.AutoAugment` | one of the policy's 25 learned pairs (`policy`: imagenet, cifar10 or svhn); each op runs with its own chance and strength, and the policies also invert images |
| trivialaugment | `v2.TrivialAugmentWide` | one op of 14 at a random strength bin, on wider scales (shear 0.99, shift 32 pixels, rotate 135 degrees, factors 1 +/- 0.99) |
| augmix | `v2.AugMix` | `mixture_width` chains of `chain_depth` ops (-1: 1 to 3) at strength bins below `severity`, mixed with Dirichlet(`alpha`) weights and blended with the image by Beta(`alpha`, `alpha`); never inverts |

autoaugment and trivialaugment sample with nearest pixels, augmix with
bilinear ones, as torchvision does. All three keep the image's shape and
need a 1- or 3-channel image.
