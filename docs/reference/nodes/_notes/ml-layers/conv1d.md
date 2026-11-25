## Convolution over a sequence

Conv1D slides its kernel along one axis. It runs on `[L, C]` samples (a
length `L` with `C` channels), as `torch.nn.Conv1d` on `[N, C, L]`. The
output is `[floor((L + 2p - kernel_size) / stride) + 1, filters]`; `same`
padding keeps `L` at stride 1 (odd kernels), `valid` adds none.

It gets its sequence one of two ways:

- **As the first model layer**, it reads each input row as channels laid end
  to end, torch's `x.view(N, C, L)`:

  | Data | Channels C | Length L |
  | --- | --- | --- |
  | Time Series Window (`input_width` W, `feature_cols`) | 1 + the feature columns | W |
  | Audio features (Spectrogram, Mel, MFCC) | frequency bins (rows) | frames |
  | Table rows of F numbers | 1 | F |

- **After an Embedding** (text), it convolves the `[L, E]` token vectors
  over the tokens with one channel per embedding dimension, torch's
  `x.transpose(1, 2)`. Activations and Dropout may sit between them.

```text
Text:   Data Input -> Embedding -> Conv1D -> ReLU -> Global Max Pool -> Dense -> Cross Entropy
Series: Data Input -> Time Series Window -> Conv1D -> ReLU -> Flatten -> Dense -> MSE
```

Only Conv1D, activations and Dropout run on the sequence. End it before
Dense with **Flatten** (`C x L` values per sample, in torch's `flatten`
order), **Global Avg Pool** (each channel's mean over `L`) or **Global Max
Pool** (its maximum: max-over-time pooling for text). The compiler reports
on the node a Conv1D anywhere else (after Dense or after the Flatten), a
Dense straight after Conv1D, and a kernel longer than the sequence.

A Conv1D graph on table rows and on an Embedding (with Global Avg Pool, and
the text CNN with Global Max Pool) is checked against torch: given torch's
parameters, the output and every parameter gradient match.
