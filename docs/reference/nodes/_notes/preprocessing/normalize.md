## Per-channel statistics

Normalize comes last, after the image steps, and runs on each batch on the
GPU: every value becomes `(x - mean) / std`. **Statistics** picks the
numbers:

| Statistics | Numbers | For |
| --- | --- | --- |
| Same for every channel | `mean`, `std` | Tables, audio, text features, and images when one pair is enough |
| Per channel | `channel_mean`, `channel_std`: one typed row per channel (R, G, B, or grey) | Images, with each channel's own statistics |
| ImageNet | 0.485, 0.456, 0.406 and 0.229, 0.224, 0.225 (torchvision), read-only | Models pretrained on ImageNet; pick Per channel to change a value, and the numbers carry over |

- The rows follow the images' channels where Normalize sits: after
  Grayscale or Image Edges there is one (grey); after a step that adds
  channels there is one row per channel, with the channel's name.
- Data Studio's **Explore > Profile > Channel statistics > Use in
  Normalize** fills Per channel from the set this graph trains from
  ([Profile](../../../user-guide/data-studio/profile.md)).
- The compiler refuses Per channel and ImageNet on data that is not images,
  a count of rows that does not match the channels, ImageNet on images that
  are not RGB, and a std that is not above 0. ImageNet adds an Info note:
  for training from scratch, the data's own statistics centre it better.
- The Compile popup's setup list and the **As compiled** card show the
  values used.
