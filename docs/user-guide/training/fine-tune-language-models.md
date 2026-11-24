---
title: Fine-tune a language model
description: Fine-tune a trained language model on prompt and answer examples with graphs only, training on the answer tokens and starting from earlier weights.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - generation-previews.md
  - start-from-pretrained-weights.md
  - checkpoints-and-early-stopping.md
  - ../evaluation/language-model-generation.md
redirect_from:
  - ../../usage/fine_tuning_language_models.md
---

You fine-tune a language model you trained earlier on prompt and answer
examples with graphs only: read the examples from JSON Lines, render them
with SQL, build token windows where only the answer is trained, and start
training from the earlier run's weights. The pieces work separately too:
JSON Lines input and SQL over nested fields are general Data Studio
features, and **Start from** warm-starts any model.

The typical pair of graphs:

```text
Data Input (JSON Lines) -> Preparation Recipe (SQL render + Check)
    -> Text Tokenizer (causal windows, Response column, saved vocabulary)
    -> Export Parquet
Training graph: Data Input (windows) -> Data Loader (Start from, context)
    -> model -> CrossEntropy (ignore_index -100) -> AdamW
```

## Before you start

- A checkpoint of a trained language model and the vocabulary file it was
  trained with. See [Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md).
- Your examples as JSON Lines: one JSON object per line, usually a `.jsonl`
  file, for example a `messages` list of `role` / `content` objects.

## Step 1: read the examples

1. Add a Data Input and open its dialog.
2. Set the format to **JSON Lines**. A `.jsonl` file is recognised by its
   extension; any other name needs `json_lines: true`.
3. **Apply**.

Nested objects become struct columns and arrays become list columns (an
array of objects is a list of structs). Keys that only some rows have are
kept; missing values are null. A plain JSON file (one array or object) is
refused with a message asking for JSON Lines: convert it to one record per
line first.

## Step 2: render the prompts with SQL

In a Preparation Recipe, add a SQL step. It reads list and struct columns
directly, in DuckDB syntax:

```sql
SELECT
    id AS document_id,
    split,
    'System: ' || list_filter(messages, m -> m.role = 'system')[1].content || chr(10) ||
    'User: '   || list_filter(messages, m -> m.role = 'user')[1].content   || chr(10) ||
    'Assistant:' || chr(10)                                                 AS prompt_text,
    list_filter(messages, m -> m.role = 'assistant')[1].content            AS response_text
FROM input
ORDER BY id
```

- The result must be flat (text, numbers, booleans). Returning a list or
  struct column is refused; select the fields you need instead.
- Keep the chat template in one place: generation and chat must render the
  prompt exactly as training did.
- Add a Check so a wrong filter cannot silently drop examples.

See [Prepare data with a recipe](../data-studio/prepare.md).

## Step 3: build supervised windows

In a **Text Tokenizer**, set the output to **Causal document windows** and
fill in:

| Setting | Value |
| --- | --- |
| Text column | The prompt (`prompt_text`) |
| **Response column** (`target_text_col`) | The answer (`response_text`) |
| Document ID / Split column | A unique id per example / train, validation or test |
| Vocabulary file | The saved vocabulary of the model you fine-tune |
| Max length | The training context (every example must fit) |

Each row becomes one window: prompt tokens, then answer tokens and EOS. Only
the answer tokens (and EOS) count toward the loss and accuracy; the prompt
is context. The output carries a `__target_mask` column and the metadata
`cyxwiz.token_windows.supervision = response_only`; training, validation and
**Train > Test** read it through the same loader.

- An example longer than one window fails and names the document (*needs 612
  ids but one window holds 513*). Examples are never split, since a split
  would separate an answer from its question. Raise Max length or shorten
  the example; to find the right length, build once with a large Max length
  and read the window lengths.
- A vocabulary file is required: supervised windows never fit a new
  vocabulary.
- An empty response is an error.
- Training data and its validation and test data must all be supervised, or
  all plain; a mix is refused.

## Step 4: start from the trained model

On the Data Loader, set **Start from** (`initialize_from_checkpoint`) to a
checkpoint folder of the earlier run, for example
`checkpoints/my_base/<run>/best`. A relative path is resolved against the
project, like Data Input paths.

- Only the model weights are loaded. Optimizer state, the learning-rate
  schedule (warm-up included), data order and epochs start fresh. Set a
  fine-tuning learning rate, usually 10 times or more below pre-training.
- The model in the graph must match the checkpoint exactly: same layer
  names, shapes and types. A mismatch stops the run and names the
  parameter.
- The checkpoint's tokenizer must match the training windows. Checkpoints
  record `tokenizer_sha256` and keep the tokenizer in `<checkpoint>/tokenizer/`;
  for older checkpoints the run's generation previews are used. If neither
  exists, the run stops rather than guessing.
- It works for a plain layer stack and for a graph with branches (residual
  blocks). Frozen layers are saved in checkpoints with the rest.
- **Start from** begins a new run; **Resume** continues the same run. Using
  both is refused.

Every checkpoint of a run that trained with a tokenizer keeps it beside its
weights, in `<checkpoint>/tokenizer/`, in the layout a `.cyxmodel` package
uses:

| File | What it holds |
| --- | --- |
| `config.json` | The tokenizer's settings (type, lowercase, BPE details) |
| `vocab.txt` | The vocabulary; its sha256 is the tokenizer's identity (`tokenizer_sha256` in `metadata.json`) |
| `sequence.json` | What training read from the data: the context length, the vocabulary in id order, the output vocabulary size and the pad ids |

Loading a checkpoint (the LM Generation panel, testing, the Generation
tab's Try now) uses this tokenizer, so the weights always meet the tokenizer
they were trained with, even if the training data changed since. A
checkpoint without a `tokenizer/` folder has its tokenizer rebuilt from the
training data and checked against `tokenizer_sha256`; the LM Generation
panel's Details says which (*Tokenizer from: the checkpoint*).

The log shows `initialized model weights from <folder> (source epoch N,
val_loss X); optimizer, schedule and epochs start fresh`, and new
checkpoints record where the run started (`initialized_from`) and its
context length.

## Step 5: watch answers while it trains

On the Data Loader's **Generation** tab, choose **Chat / instruction**, the
template the data was rendered with (**Role lines** with the labels
`System:`, `User:` and your assistant label), the system prompt once, and up
to 8 user messages. Optional: add your format's end (such as `</answer>`)
as a stop text and a check per prompt. The samples show in the Training
Dashboard's Generations tab after each pass. See
[Watch generations while training](generation-previews.md).

## Check that it worked

- The Text Tokenizer output has `__target_mask`, and `__valid_targets`
  equals the number of answer tokens per window.
- At the start of training the log shows the warm-start line, and the first
  loss is far below ln(vocabulary size), where an untrained model starts.
- Validation loss and accuracy are measured on answer tokens only.

## Train at a longer context

Models with rotary positions (RoPE) have no weights tied to the context
length, so a longer context needs no special step:

1. Re-encode the training text with the same vocabulary at the new Max
   length.
2. Set the Data Loader's max sequence length to the new context.
3. **Start from** the shorter-context checkpoint; a short continued training
   at the new length lets the model adapt.

A model with a learned position table fails at Start from with *Checkpoint
shape mismatch for parameter '<name>'*: it cannot change context this way.
A longer context costs memory (a 768 window holds three times the tokens of
a 256 window), so lower the micro-batch and raise gradient accumulation to
keep the effective batch.

## Related

- [Watch generations while training](generation-previews.md)
- [Use a trained language model](../evaluation/language-model-generation.md)
- [Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md)
- [Training](index.md)
