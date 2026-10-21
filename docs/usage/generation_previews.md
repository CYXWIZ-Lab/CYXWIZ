# Generation previews

While a language model trains, the Engine can ask it a few prompts at set
points and keep what it answers. You see whether it is learning to write, to
follow a format, to stop by itself, or whether it loops, epoch by epoch. It
works for any causal language model trained on token windows: a base model
that continues text, or an instruction-tuned (chat) model.

Set them up on the Data Loader's **Generation** tab. Read them in the
Training Dashboard's **Generations** tab while the run trains, and later in
Data Studio > Results > **Generations**. Both show the same view.

## Setting up (Data Loader > Generation)

Tick **Generate samples while training**, then choose what the model is:

- **Continuation · base model**: the prompt is given as it is, and the model
  continues it (for example *In the beginning*).
- **Chat / instruction · fine-tuned model**: each prompt is a user message.
  The Engine wraps it in the chat template, so the model sees the same format
  it was trained on.

A note next to the switch compares your choice with the training data:
windows that train on the responses only (`supervision = response_only`, made
by a Text Tokenizer with a response column) suggest chat; plain text windows
suggest continuation.

### Prompts

Up to 8 prompts. **+ Add prompt** adds one; click a row to edit it;
right-click to move it up or down or to remove it. Each prompt can carry an
optional **check**, shown as passed or failed next to its output:

| Check | Passes when |
| --- | --- |
| Output contains | the output contains the text (*Psalm 34:13*) |
| Output does not contain | the output does not contain the text |
| Output matches the pattern | the regular expression finds a match (`Psalm \d+:\d+`) |
| Stops by itself | generation ended at end-of-sequence or a stop text, not at the token limit |

**What the model sees** shows the prompt as the model receives it: the
template's text in the accent colour, your text in the normal colour, and a
green bar where generation starts. On the right, its length against the
context (*187 / 768 tokens*); amber when the new tokens no longer fit, red
when the prompt itself does not. The count shows once the training dataset
is loaded (apply its Data Input).

### Chat template (chat only)

- **Role lines**: one turn per line with a label before it:
  `System: ...`, `User: ...`, then the assistant's label (`Assistant:` by
  default). Set the three labels to what the data was trained with (for
  example `Berean:`).
- **ChatML**: `<|im_start|>` and `<|im_end|>` markers.
- **Llama 3**: header and end-of-turn markers.
- **From tokenizer** is shown greyed: none of the Engine's tokenizers carries
  a chat template yet.

The **system prompt** is written once and shared by every prompt. Use the
template the data was built with; another one shows the model a format it
never learned.

### Stopping

Generation stops at the end-of-sequence token, at the template's own stop
(the next `User:` line, `<|im_end|>`, `<|eot_id|>`), at any **stop text** you
add (for example `</answer>`; up to 8), or at **Max new tokens** (1 to 1024,
capped by the context the prompt leaves). A stop text is cut from the output.

### Decoding

- **Greedy** always takes the most likely next token. Use it to compare
  epochs: the same weights give the same output, and loops show plainly.
- **Sampling**: temperature, top-k (0 is off), top-p, min-p (0 is off; keeps
  tokens at least that fraction as likely as the most likely one), a seed and
  up to 4 samples per prompt. With a fixed seed each prompt gets the same
  random draws every time, so passes stay comparable.
- **Beam search**: 2 to 8 beams and a length penalty (0 to 2; above 1 favours
  longer outputs). Keeps the most likely sequences instead of single tokens
  and returns the best one; the same output every time, and slower (each beam
  recomputes its sequence).
- **Penalties**, for every decoding: **repetition** (1 to 2, off by default)
  makes tokens already in the prompt or the output less likely; **frequency**
  and **presence** (0 to 2) lower a token by how often, or whether, the output
  already used it; **no-repeat n-gram** (0 is off) never writes a run of that
  many tokens twice.

### When

- **Every** N **epochs** or **optimizer steps**.
- **Before training starts (baseline)**: what the model says before any
  update (a warm-started model shows what it already knows).
- **After the last epoch**: also when early stopping ends the run.
- **Weights**: **Current**, or **Best so far** (the best validation
  checkpoint is swapped in for the pass and the current weights are put back;
  until a best exists, the current weights are used).

The Properties panel shows the main settings (on or off, mode, max new
tokens, every); the prompts, template and decoding are on the tab.

### Try now

**Try now** (under What the model sees) runs the tab's current settings on a
saved checkpoint without training, so a prompt, template or stop text can be
checked before Apply or before a long run. The checkpoint is the newest run's
`best` under the Data Loader's checkpoint folder; before any run it is the
**Start from** checkpoint. **Choose...** picks another run or checkpoint
folder, **Use the newest** goes back. The training data must be loaded (apply
its Data Input): the model and its tokenizer come from the graph and that
data. The model is built on the selected device and loaded in the background
(**Cancel** stops it); the session's active model (Test, Inference) is not
replaced. The result shows in the tab in the Generations view, and the report
is saved next to the checkpoint in `generation_tries/` (`try.json`,
`try_2.json`, ...). The button is off, with the reason on hover, while
training runs, when the data is not loaded or when there is no checkpoint.

## Reading them (Generations)

Each pass writes one file in the run's checkpoint folder,
`<checkpoints>/<run>/generation_previews/`: `baseline.json`, `epoch_4.json`,
`step_1200.json` (a resumed run that repeats a point adds `_2`).

The **Generations** view has three ways to look:

- **Latest**: the newest pass (or pick another under **Show**). Tiles for
  the pass, **checks passed**, **stopped by itself**, **repeating itself**
  and the **validation loss** at that point. One card per prompt with its
  output, how it ended (*Stopped by itself · 114 tokens*, *Hit the 128-token
  limit*, *Stopped at "</answer>"*), a *Repeats "..."* note when a phrase of 3
  or more words repeats, the check result, and **What the model saw**.
- **Compare two points**: two passes side by side, prompt by prompt.
- **One prompt over time**: one prompt's output at every pass, with the
  validation loss next to each.

**How these were made** lists the mode, template, decoding, token limit, stop
texts, weights, context and time. **Export samples (JSONL)** writes every
sample of the run, one JSON object per line, to `samples.jsonl` in the same
folder.

The Training Dashboard shows the tab **Generations (n)** next to
**Overview** once the run has a preview folder, and updates as passes are
written. Data Studio > Results lists every run of the project that wrote
previews, newest first.

## In the Console

Each pass logs one readable block at Info:

```
Generation · epoch 1/1 · chat (role lines) · greedy · up to 48 new tokens · current weights · 4 prompts · 0.4 s
  1  stopped at "</answer>" · check failed  ""
  ...
  checks 0/4 · stopped by itself 4/4 · repeating 0/4 · saved generation_previews/epoch_1.json
```

Each sample's full record (token ids included) is at Debug.

## Files from older Engines

Generation previews saved by Engines before this design (prompt lists in one
text field, `epoch_N.json` without `"schema": 2`) are not read. A Data Loader
saved that way says on its Generation tab that its prompts are not used; add
them again. Old report files are listed under Details as *not shown*.
