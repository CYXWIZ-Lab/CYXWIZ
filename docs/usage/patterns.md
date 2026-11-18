# Patterns

A pattern is a ready-made graph or part of a graph that you insert into the
canvas. Open the **Patterns** panel with **View > Panels > Patterns panel**
(Ctrl+Shift+P), pick a pattern, change its settings if you want, and click
**Insert Pattern**.

## Full graphs and blocks

Patterns come in two kinds.

- **A full graph** is a whole model, from data to training:
  Data Input -> (Text Tokenizer) -> Data Split -> DataLoader -> (image steps)
  -> layers -> loss -> optimizer -> Output. After inserting it, double-click
  **Data Input**, point it at your data, click **Apply**, then **Train**.
  Each pattern's description says what data it expects.
- **A block** is a few layers to put inside your own graph. It starts with a
  **Block input** node (an Identity layer): link the layer before the block
  to it, and link the block's last layer to whatever comes next. The block's
  description says what input shape it needs.

## The shipped patterns

| Pattern | Kind | Data it expects |
| --- | --- | --- |
| Basic MLP, MLP with Dropout | Full graph | A CSV table: number columns and a `label` column of classes |
| Linear Regression | Full graph | A CSV table: number columns and a number `label` column (MSE loss, SGD) |
| Skip Connections and Merges | Full graph | A CSV table with a `label` column; shows a residual Add, a skip through Concatenate, self-attention with its residual, and an Add of three branches |
| Basic CNN, CNN Image Classifier, LeNet-5 | Full graph | An image folder with one subfolder per class; Resize 32 x 32 and Normalize (mean 0.5, std 0.5) |
| AlexNet, VGG-16, MobileNet v1 | Full graph | An image folder with one subfolder per class; Resize 256, Center Crop 224 and Normalize with the ImageNet statistics |
| Basic RNN, Basic GRU, Basic LSTM, Bidirectional LSTM, Transformer Encoder | Full graph | A CSV with a `text` column and a `label` column; the Text Tokenizer keeps `max_length` tokens |
| Transformer Decoder (next token) | Full graph | A CSV of token ids: `sequence_length` token columns and a `next` column (the token that follows); predicts the next token |
| Causal LM Generation Starter | Full graph | A CSV of token ids, one sequence per row; trained on the next token at every position |
| PPO (MLP 64-64) | Full graph | No data: an Environment node (CartPole by default). See [Training a policy (PPO)](rl_training.md) |
| Conv-BN-ReLU Block, Residual Block | Block | A feature map [height, width, channels]; the Residual Block's input must have `filters` channels |
| Multi-Head Attention Block | Block | A sequence [length, d_model] |

The column names (`label`, `text`, `next`) are only where the patterns start:
change them in the Data Input dialog to match your file.

## Patterns that cannot be used yet

Some patterns need a kind of training the Engine does not have yet. They stay
in the list so you can see them, and inserting one is refused with the
reason:

| Pattern | Needs |
| --- | --- |
| DCGAN, Basic GAN | Training a generator against a discriminator (GAN training) |
| Basic Autoencoder | Training against the input itself (a reconstruction target) |
| VAE Encoder, VAE Decoder | VAE sampling and the KL term |
| U-Net | Segmentation masks as labels (one class per pixel) |
| LSTM Seq2Seq | Two-input sequence-to-sequence training (an encoder input and a shifted decoder input) |

Each becomes a full graph when that training is built.

## Limits

- Inserting a pattern that cannot be used yet, or one that fails, is only
  reported in the log for now. The Patterns panel will mark the full graphs,
  the blocks and the patterns that cannot be used yet.
- There is no command yet to save a selection of your own nodes as a
  pattern.

## For developers: the pattern file

Patterns are `.cyxgraph` files in `cyxwiz-engine/resources/patterns`, in the
template format: `id` (the file name), `name`, `kind` (`graph` or `block`),
`needs` (why it cannot be used yet; left out when it can), `description`,
`sample_shape` (one sample at the first layer, for example `[32, 32, 3]`),
`category`, `tags`, `parameters` (each `$name` in the template takes the
value the user sets), and `template` with `nodes` (registry type names) and
`links` (by node id and pin index).

`test_shipped_patterns` inserts every shipped pattern through the Engine's
pattern library and node factory. It checks that every link lands on a real
pin, that every full graph compiles, that every block compiles inside a small
graph, and that the patterns that cannot be used yet refuse to insert. It
also trains the small ones (Basic MLP, Basic CNN, LeNet-5, Basic RNN) for an
epoch on synthetic data.
