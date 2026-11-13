# ResNet-18 from ImageNet weights: fine-tuning a cats-vs-dogs classifier

`cats_dogs_resnet18_finetune.cyxgraph` is torchvision's ResNet-18, node for
node, started from the published ImageNet weights. The backbone is frozen and
only a new one-logit head trains.

```
Data Input -> Quality Analyzer -> Split 80/10/10 -> Loader (batch 64, 3 epochs,
  Start from: weights/resnet18-f37072fd.pth) -> Resize 72 -> Center Crop 64
  -> Random Resized Crop -> Horizontal Flip -> Color Jitter -> Normalize (ImageNet)
  -> conv1 7x7 /2 -> bn1 -> relu -> maxpool 3 /2 pad 1
  -> layer1.0 ... layer4.1 (8 residual blocks; frozen)
  -> avgpool -> fc (1 logit, starts fresh) -> BCE with Logits -> AdamW -> Output
```

Every layer node is named as torchvision names the layer (`conv1`, `bn1`,
`layer1.0.conv1`, `layer2.0.downsample.0`, `fc`), so the weight file fills the
graph by name. The convolutions have **Bias** off, as torchvision's do. The
graph is written by `scripts/examples/torch_to_cyxgraph.py` from
`torchvision.models.resnet18()`.

## Data and weights

- Any folder of labelled images with two classes. It was made for the Kaggle
  "Dogs vs. Cats" set (25,000 JPEGs with a CSV of `image,labels`); the example
  saves no data path.
- The weight file is not shipped (45 MB). Download torchvision's ImageNet
  weights for ResNet-18 from
  `https://download.pytorch.org/models/resnet18-f37072fd.pth` and put it in the
  project's `weights` folder, or point the Data Loader at it (Checkpoints >
  Start from > A weights file). PyTorch does not need to be installed.

## Run it

1. Open the graph.
2. Double-click **Cats_Dogs Data**, pick your image folder and how its labels
   are given, then **OK**.
3. Double-click **Quality Analyzer** and **Analyze** (once per image size).
4. Double-click **Data Loader**, Checkpoints tab: the card under the weights
   file should say **40 of 41 layers take their weights**, and **fc starts
   fresh** (the file's head has 1,000 classes, this one has 1).
5. **Train**. The line under the canvas reads **Trainable 513 · Frozen
   11,176,512**: only the head learns.

## What one run gave

Kaggle Dogs vs. Cats, this PC (GeForce GTX 1050 Ti, CUDA), 2026-11-13. The
Quality Analyzer left out 1,099 images; 19,120 train / 2,389 validation /
2,392 test.

| Epoch | Train loss | Train acc | Val loss | Val acc |
| --- | --- | --- | --- | --- |
| 1 | 0.543 | 78.36 % | 0.365 | 84.18 % |
| 2 | 0.361 | 84.54 % | 0.323 | 85.77 % |
| 3 | 0.339 | 85.03 % | 0.321 | 85.81 % |

**Test accuracy 86.29 %** (loss 0.316), about 140 s per epoch. The run log
says: `started from resnet18-f37072fd.pth (PyTorch state-dict): 40 of 41
layers take weights from the file; 1 starts fresh: fc`.

That is three epochs of one linear layer on 64 x 64 images, which is far
smaller than the 224 x 224 the weights were trained on.

## Second stage: train every layer

1. **Nodes > Freeze > Unfreeze All** (click the canvas first).
2. Data Loader > Checkpoints: **Start from** > **A checkpoint of an earlier
   run**, and type the first run's `best` folder
   (`checkpoints/<run>/train-<id>/best`).
3. Set AdamW's learning rate to 0.0001 and **Train**.

The same PC, three more epochs, every layer training (11,177,025 weights),
about 220 s per epoch. The run log says `initialized model weights from
.../best (source epoch 3, val_loss 0.32057858)`.

| Epoch | Train loss | Train acc | Val loss | Val acc |
| --- | --- | --- | --- | --- |
| 1 | 0.354 | 84.31 % | 0.258 | 89.07 % |
| 2 | 0.226 | 90.45 % | 0.220 | 90.79 % |
| 3 | 0.182 | 92.48 % | 0.213 | 91.00 % |

**Test accuracy 92.60 %** (loss 0.176). For comparison,
`cats_dogs_resnet_scratch.cyxgraph` trains a smaller network from random
weights for 20 epochs and reaches 92.64 %; here six epochs in all reach the
same.

## Try
- **Larger images.** Resize 128 and Center Crop 112 bring the images closer to
  what the weights expect; preparing them takes longer.
- **Freeze less.** **Nodes > Freeze > Freeze all but the last N layers...**
  keeps the last blocks training.

See [docs/usage/pretrained_weights.md](../../docs/usage/pretrained_weights.md).
