# Language Model Generation

Use a trained language model directly in the Engine: chat with a fine-tuned
(instruction) model, or let a base model continue your text. It works for
any causal language model the Engine trains or imports, a layer stack or a
graph with branches, with any of its tokenizers.

Open it with **Tools > Text > Language Model Generation**.

To serve a model to other apps over HTTP, use **Deploy** instead; this
panel is for using the model yourself.

## Loading a model

With no model loaded, the panel offers three sources (the same three are in
**Change model...** once a model is loaded):

| Source | What it loads |
| --- | --- |
| **This session's model** | The model trained in this session, or the one opened with Train > Test > Load Checkpoint. Off while training runs. |
| **A training checkpoint** | A saved run of a graph in the project. Pick the **graph** it was trained from and a **run** (newest first, with its weights, epoch, validation loss and date). **Choose a folder...** lists the runs of another checkpoint folder. |
| **A model package (.cyxmodel)** | An exported model. It carries its tokenizer. |

A checkpoint is built from its graph: the model from the graph's layers, the
tokenizer from the graph's training data (its Data Input files are loaded in
the background; relative paths are relative to the project), then the
weights. Loading runs on the device selected in **Preferences > Devices**,
shows each step, and can be cancelled. The session's active model (Test,
Inference) is not replaced.

While loading, the Engine checks that:

- the model gives next-token scores for every token of its tokenizer;
- the tokenizer is the one the checkpoint was trained with (when the
  checkpoint records it);
- the graph's Data Loader has generation settings (otherwise defaults are
  used).

The dialog opens on the newest graph with a saved run. A graph's own
generation settings that an older Engine saved are not used (Details says
so) and the defaults are; the model still loads.

The bar at the top then shows the model's name, where it came from (*checkpoint ·
best of train-1791984271862*), a summary (epoch, validation loss, context,
vocabulary and tokenizer, chat or continuation) and the device. **Details**
lists every fact, the checks, the checkpoint's path and the last reply.

## Chat

For an instruction-tuned model. Type a message and press **Send** (Enter;
Ctrl+Enter for a new line). The message is sent as the user's turn in the
model's chat template, with the system prompt once at the start and the
earlier turns of the conversation. When the conversation no longer fits the
context, the oldest turns are left out (the system prompt always stays);
the meter under the message shows how many tokens are used and how many new
tokens fit.

Replies stream in as they are generated; **Stop** ends one early (what was
generated is kept). Each reply shows how it ended (*Stopped by itself*, *Hit
the 128-token limit*, *Stopped at "</answer>"*), its speed (*128 tokens ·
3.4 s · 37 tokens/s*) and a *Repeats "..."* note when a phrase of 3 or more
words repeats. Its links:

- **Copy** the reply.
- **Regenerate** the latest reply (sampling draws a new seed each time unless
  the seed is fixed).
- **What the model saw**: the whole prompt, the template's text in the
  accent colour, with its token count and the turns sent or left out.
- **Tokens**: every generated token; click one to see the five most likely
  candidates at that step with their probabilities (after temperature,
  top-k and top-p) and their IDs. **Copy token IDs** copies the reply's IDs.

**Copy all** copies the conversation as text, **Export (JSONL)** writes one
JSON object per reply (message, prompt, output, how it stopped, tokens,
seconds, seed, settings), **New conversation** starts over.

## Continuation

For a base model. Type the text to continue and press **Generate**. The
result shows your text followed by the generated text in another colour.
**Generate again** reruns from the same text; **Continue further** makes
the whole result the new text and continues it. The same footer, **What the
model saw** and **Tokens** as in chat.

Switching between Chat and Continuation starts a new conversation.

## Generation settings

The panel on the right holds the same settings as the Data Loader's
**Generation** tab (see [Generation previews](generation_previews.md)),
loaded from the model's graph:

- **Template** (chat): Role lines with your labels, ChatML or Llama 3, and
  the system prompt.
- **Stop at**: the end-of-sequence token, the template's stop, your stop
  texts (a stop text is cut from the reply) and **Max new tokens**.
- **Decoding**: Greedy; Sampling with temperature, top-k, top-p, min-p and
  seed; or Beam search with the number of beams and a length penalty. The
  penalties (repetition, frequency, presence, no-repeat n-gram) apply to all
  three. See [Generation previews](generation_previews.md#decoding).
- **New seed for each reply** (sampling, on by default): each reply draws
  differently, and its seed is shown under it; **Use this seed** fixes that
  seed, so the same message gives the same reply again. Off, every reply uses
  the seed field. Greedy and beam search always give the same reply.
- **History** (chat): whether earlier turns are sent with each message.
- **Speed**: **KV cache** (on by default) reuses each layer's keys and
  values, so every new token feeds only itself; in a chat the cache is also
  kept between messages, so a new turn feeds only what is new (What the model
  saw says how many tokens came from the previous turn's cache). Off
  recomputes the whole sequence for every token. A graph with branches has no
  KV cache. **Measure speed** runs the message (or the last one) greedily for
  the full max new tokens, with the cache and without, and shows tokens per
  second for each.

**Reset to the graph's** brings back the graph's settings. **Save to the
graph** writes the template, labels, system prompt, stop texts, token limit
and decoding into the graph's Data Loader (its own prompts, cadence and the
on/off switch are kept); reopen the graph on the canvas to see the change.
A package carries no generation settings yet, so it starts with the
defaults.

## Where conversations are kept

A conversation with a checkpoint is saved, after each reply, as a generation
report in the run's folder: `<run>/generation_chats/chat.json` (`chat_2.json`,
... for the next ones). The file has the same form as the training previews,
so it can be read like them. Conversations with a package or a model trained
in this session (not from a checkpoint) are not saved; use **Export (JSONL)**.

## Raw token IDs

Under **Details**, **Raw token IDs** generates from token IDs you type, and
shows what they decode to: for checking a tokenizer or the model itself.
