# MNIST digits: the Model window and Results > Test

**Graph:** `mnist_cnn.cyxgraph`. **Docs:** [The Model window](../../../docs/usage/model_tools.md),
[Testing a trained model](../../../docs/usage/testing.md).

A small CNN on the MNIST test digits, read from a CSV of pixel columns. It is
the graph the Model window's guide uses for its numbers:

```
MNIST rows (mnist_test.csv, label + 784 pixels)
  -> Split 80 / 10 / 10 (stratified, seed 42) -> Batches of 64, 3 epochs
  -> Scale to 0..1 (Normalize, mean 0, std 255)
  -> Image 28 x 28 (Reshape 28,28,1: each row read as a 28 x 28 x 1 image)
  -> Conv2D 32 -> ReLU -> MaxPool -> Conv2D 64 -> ReLU -> MaxPool
  -> Flatten -> Dense 128 -> ReLU -> Dropout 0.5 -> Dense 10
  -> Cross entropy -> Adam 0.001
```

## Get the data

The digits are not in the repository. `get_mnist.py` downloads the MNIST test
set (about 2 MB, checked by MD5) and writes `mnist_test.csv` here (10,000 rows,
`label,pixel0,...,pixel783`, 17.4 MB):

```
python get_mnist.py
```

Standard library only. `--out <path>` writes it elsewhere; then point the
Data Input at that file.

## Run it

1. Open the graph (File > Open, or `cyxwiz-engine <project> examples/cyxgraph/model_tools/mnist_cnn.cyxgraph`).
2. Data Studio > Bring in: **Open dialog** on MNIST rows, then **OK** (it
   loads the 10,000 rows; the graph compiles: 225,034 parameters).
3. **Tools > Model > Learning Rate Finder**, **Run**: it stops
   early at step 85 of 100 (under half a minute) and suggests 0.0085 (the lowest loss, 1.56, is
   at 0.0955).
4. **Train > Start Training** (F5): 3 epochs.
5. **Train > Test > Run Test** (F8): accuracy 98.01 % on 1,007 test rows.

Then, on this machine (NVIDIA CUDA; another device can differ in the last
digits):

| Where | What it shows |
| --- | --- |
| Model > Summary | 225,034 parameters (879.0 KB), 2.63 M multiply-adds a digit, activations 16.73 MB at batch 64 |
| Model > Health | 47 of Dense 128's 128 units dead after its ReLU (53 -> 47 -> 47 by epoch); gradients in range |
| Model > Explain | Row 1014, a 6 read as 5 with 98.8 %: input, Grad-CAM of Conv2D 64, saliency, Integrated Gradients |
| Model > Projector | Dense 128 by PCA: 19.6 % and 17.7 % of the variance on the two axes |
| Results > Test > Errors | 20 of 1,007 wrong; the commonest mistake is a Class 6 read as Class 5 (3 rows) |
| Results > Test > Threshold | Class 0 against the rest at 0.5: precision 1.0000, recall 0.9694; a confidence floor of 0.90 answers 94.0 % of the rows (947), 99.37 % of them right |
| Results > Test > Calibration | Calibration error 0.0108; temperature 0.804 brings it to 0.0098 on the held-out half |

**Keep the temperature with the model** keeps it: the next test says "This
test used the model's kept temperature 0.804", the checkpoint's
`metadata.json` and an export carry it, and the local server's `/v1/predict`
returns calibrated probabilities.
