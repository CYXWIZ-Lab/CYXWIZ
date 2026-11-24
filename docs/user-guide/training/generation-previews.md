---
title: Watch generations while training
description: Have a language model answer a few prompts at set points during training, with checks, and read its answers epoch by epoch.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - fine-tune-language-models.md
  - ../evaluation/language-model-generation.md
  - ../../main-window/training-dashboard.md
redirect_from:
  - ../../usage/generation_previews.md
---

While a language model trains, the Engine can ask it a few prompts at set
points and keep what it answers. You see whether it learns to write, to
follow a format and to stop by itself, or whether it loops, epoch by epoch.
It works for any causal language model trained on token windows: a base
model that continues text, or an instruction-tuned (chat) model.

You set previews up on the Data Loader's **Generation** tab and read them in
the Training Dashboard's **Generations** tab while the run trains, and later
in Data Studio's **Results > Generations**. Both show the same view.

## Set up previews

1. Open the Data Loader's dialog and its **Generation** tab.
2. Tick **Generate samples while training**.
3. Choose what the model is:
   - **Continuation · base model**: the prompt is given as it is and the
     model continues it (for example *In the beginning*).
   - **Chat / instruction · fine-tuned model**: each prompt is a user
     message, wrapped in the chat template so the model sees the format it
     was trained on.

   A note next to the switch compares your choice with the training data:
   windows that train on the responses only (made by a Text Tokenizer with a
   response column) suggest chat; plain text windows suggest continuation.
4. Add up to 8 prompts with **+ Add prompt**. Click a row to edit it;
   right-click to move it up or down or to remove it. Optional: give a
   prompt a check (the table that follows).
5. Chat only: pick the chat template and write the system prompt (once, for
   every prompt).
6. Set stopping, decoding and when to generate (the sections that follow),
   then **Apply**.

| Check | Passes when |
| --- | --- |
| **Output contains** | The output contains the text |
| **Output does not contain** | The output does not contain the text |
| **Output matches the pattern** | The regular expression finds a match (`Chapter \d+`) |
| **Stops by itself** | Generation ended at end-of-sequence or a stop text, not at the token limit |

**What the model sees** shows the prompt as the model receives it: the
template's text in the accent colour, your text in the normal colour, and a
green bar where generation starts. On the right, its length against the
context (*187 / 768 tokens*), amber when the new tokens no longer fit, red
when the prompt itself does not. The count shows once the training dataset
is loaded.

The Properties panel shows the main settings (on or off, mode, max new
tokens, every); the prompts, template and decoding are on the tab.

### Chat template

- **Role lines**: one turn per line with a label before it: `System: ...`,
  `User: ...`, then the assistant's label (`Assistant:` by default). Set the
  three labels to what the data was trained with.
- **ChatML**: `<|im_start|>` and `<|im_end|>` markers.
- **Llama 3**: header and end-of-turn markers.
- **From tokenizer** is shown greyed: none of the Engine's tokenizers
  carries a chat template.

Use the template the data was built with; another one shows the model a
format it never learned.

### Stopping

Generation stops at the end-of-sequence token, at the template's own stop
(the next `User:` line, `<|im_end|>`, `<|eot_id|>`), at any **stop text** you
add (for example `</answer>`; up to 8), or at **Max new tokens** (1 to 1024,
capped by the context the prompt leaves). A stop text is cut from the
output.

### Decoding

- **Greedy** always takes the most likely next token. Use it to compare
  epochs: the same weights give the same output, and loops show plainly.
- **Sampling**: temperature, top-k (0 is off), top-p, min-p (0 is off; keeps
  tokens at least that fraction as likely as the most likely one), a seed
  and up to 4 samples per prompt. With a fixed seed each prompt gets the
  same random draws every time, so passes stay comparable.
- **Beam search**: 2 to 8 beams and a length penalty (0 to 2; above 1
  favours longer outputs). It keeps the most likely sequences and returns
  the best one: the same output every time, and slower.
- **Penalties**, for every decoding: **repetition** (1 to 2, off by default)
  makes tokens already in the prompt or output less likely; **frequency**
  and **presence** (0 to 2) lower a token by how often, or whether, the
  output already used it; **no-repeat n-gram** (0 is off) never writes a run
  of that many tokens twice.

### When

- **Every** N **epochs** or **optimizer steps**.
- **Before training starts (baseline)**: what the model says before any
  update (a warm-started model shows what it already knows).
- **After the last epoch**, also when early stopping ends the run.
- **Weights**: **Current**, or **Best so far** (the best validation
  checkpoint is swapped in for the pass and the current weights are put
  back; until a best exists, the current weights are used).

## Try a prompt without training

**Try now** (under What the model sees) runs the tab's current settings on
a saved checkpoint, so a prompt, template or stop text can be checked before
a long run.

1. Load the training data (apply its Data Input): the model and its
   tokenizer come from the graph and that data.
2. Optional: **Choose...** picks a run or checkpoint folder; **Use the
   newest** goes back to the newest run's `best` under the Data Loader's
   checkpoint folder (before any run, the **Start from** checkpoint).
3. Select **Try now**. The model loads in the background on the selected
   device (**Cancel** stops it); the session's active model (Test,
   Inference) is not replaced.

The result shows in the Generations view, and the report is saved next to
the checkpoint in `generation_tries/` (`try.json`, `try_2.json`, ...). The
button is off, with the reason on hover, while training runs, when the data
is not loaded or when there is no checkpoint.

## Read the generations

Each pass writes one file in the run's checkpoint folder,
`<checkpoints>/<run>/generation_previews/`: `baseline.json`, `epoch_4.json`,
`step_1200.json` (a resumed run that repeats a point adds `_2`). The
Training Dashboard shows the tab **Generations (n)** next to **Overview**
once the run has a preview folder; Data Studio's Results lists every run of
the project that wrote previews, newest first.

| View | Shows |
| --- | --- |
| **Latest** | The newest pass (or another under **Show**): tiles for the pass, **checks passed**, **stopped by itself**, **repeating itself** and the **validation loss**; one card per prompt with its output, how it ended (*Stopped by itself · 114 tokens*, *Hit the 128-token limit*, *Stopped at "</answer>"*), a *Repeats "..."* note when a phrase of 3 or more words repeats, the check result, and **What the model saw**. |
| **Compare two points** | Two passes side by side, prompt by prompt. |
| **One prompt over time** | One prompt's output at every pass, with the validation loss next to each. |

**How these were made** lists the mode, template, decoding, token limit,
stop texts, weights, context and time. **Export samples (JSONL)** writes
every sample of the run, one JSON object per line, to `samples.jsonl` in the
same folder.

Each pass also logs one block in the Console at Info level (each sample's
full record, token ids included, at Debug):

```text
Generation · epoch 1/1 · chat (role lines) · greedy · up to 48 new tokens · current weights · 4 prompts · 0.4 s
  1  stopped at "</answer>" · check failed  ""
  ...
  checks 0/4 · stopped by itself 4/4 · repeating 0/4 · saved generation_previews/epoch_1.json
```

## Limits

Previews saved by older Engines (prompt lists in one text field,
`epoch_N.json` without `"schema": 2`) are not read. A Data Loader saved that
way says on its Generation tab that its prompts are not used; add them
again. Old report files are listed under Details as *not shown*.

## Related

- [Fine-tune a language model](fine-tune-language-models.md)
- [Use a trained language model](../evaluation/language-model-generation.md)
- [Training Dashboard](../../main-window/training-dashboard.md)
- [Training](index.md)
