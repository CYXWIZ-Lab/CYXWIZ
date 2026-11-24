---
title: Chat with or continue text from a language model
description: Load a trained language model in the Language Model Generation panel, chat with it or let it continue text, and tune decoding.
type: how-to
area: evaluation
audience: user
status: stable
since: 1.0.7
owner: evaluation
last_reviewed: 2026-11-25
related:
  - ../training/generation-previews.md
  - ../training/fine-tune-language-models.md
  - test-results.md
redirect_from:
  - ../../usage/lm_generation.md
---

Use a trained language model directly in the Engine: chat with a
fine-tuned (instruction) model, or let a base model continue your text. It
works for any causal language model the Engine trains or imports, a layer
stack or a graph with branches, with any of its tokenizers. Open it with
**Tools > Text > Language Model Generation**.

This panel is for using the model yourself. To serve a model to other
applications over HTTP, use **Deploy** instead.

## Load a model

With no model loaded, the panel offers three sources. The same three are in
**Change model...** once a model is loaded.

| Source | What it loads |
| --- | --- |
| **This session's model** | The model trained in this session, or the one opened with **Train > Test > Load Checkpoint for Testing...**. Off while training runs. Loading it, and a reply, wait while a test, **Model > Explain** or Feature Importance runs on it. |
| **A training checkpoint** | A saved run of a graph in the project. Pick the **graph** it was trained from and a **run** (newest first, with its weights, epoch, validation loss and date). **Choose a folder...** lists the runs of another checkpoint folder. The dialog opens on the newest graph with a saved run. |
| **A model package (.cyxmodel)** | An exported model. It carries its tokenizer. |

A checkpoint is built from its graph: the model from the graph's layers
(its Data Input files load in the background; relative paths are relative
to the project), then the tokenizer the checkpoint keeps (an older
checkpoint rebuilds it from the training data; see
[Fine-tune a language model](../training/fine-tune-language-models.md)),
then the weights. Loading runs on the device selected in **Preferences >
Devices**, shows each step, and can be cancelled. The session's active
model (Test, inference) is not replaced.

While loading, the Engine checks that:

- the model gives next-token scores for every token of its tokenizer;
- the tokenizer is the one the checkpoint was trained with: the one it
  keeps, or one rebuilt from the training data that matches the
  `tokenizer_sha256` it records;
- the graph's Data Loader has generation settings (otherwise the defaults
  are used). Generation settings saved by an older Engine are not used
  (**Details** says so); **Save to the graph** rewrites them in the current
  form.

The bar at the top then shows the model's name, where it came from
(*checkpoint · best of train-1791984271862*), a summary (epoch, validation
loss, context, vocabulary and tokenizer, chat or continuation) and the
device. **Details** lists every fact, the checks, the checkpoint's path and
the last reply.

## Chat with an instruction model

1. Type a message and press **Send** (Enter; Ctrl+Enter for a new line).
   The message is sent as the user's turn in the model's chat template,
   with the system prompt once at the start and the earlier turns.
2. The reply streams in. **Stop** ends it early; what was generated is
   kept.
3. Under each reply: how it ended (*Stopped by itself*, *Hit the 128-token
   limit*, *Stopped at "</answer>"*), its speed (*128 tokens · 3.4 s · 37
   tokens/s*) and a *Repeats "..."* note when a phrase of 3 or more words
   repeats. Its links:
   - **Copy** the reply;
   - **Regenerate** the latest reply (sampling draws a new seed each time
     unless the seed is fixed);
   - **What the model saw**: the whole prompt, the template's text in the
     accent colour, its token count and the turns sent or left out;
   - **Tokens**: every generated token. Click one for the five most likely
     candidates at that step, with their probabilities (after temperature,
     top-k and top-p) and IDs. **Copy token IDs** copies the reply's IDs.

When the conversation no longer fits the context, the oldest turns are
left out; the system prompt always stays. The meter under the message
shows how many tokens are used and how many new tokens fit.

**Copy all** copies the conversation as text, **Export (JSONL)** writes one
JSON object per reply (message, prompt, output, how it stopped, tokens,
seconds, seed, settings), and **New conversation** starts over.

## Continue text with a base model

1. Switch to **Continuation**, type the text to continue and press
   **Generate**. The result shows your text followed by the generated text
   in another colour.
2. **Generate again** reruns from the same text; **Continue further** makes
   the whole result the new text and continues it.

The footer, **What the model saw** and **Tokens** work as in chat.
Switching between Chat and Continuation starts a new conversation.

## Change the generation settings

The panel on the right holds the same settings as the Data Loader's
**Generation** tab ([Generation previews](../training/generation-previews.md)),
loaded from the model's graph:

| Setting | What it sets |
| --- | --- |
| Template (chat) | Role lines with your labels, ChatML or Llama 3, and the system prompt |
| Stop at | The end-of-sequence token, the template's stop, your stop texts (cut from the reply) and **Max new tokens** |
| Decoding | Greedy; Sampling with temperature, top-k, top-p, min-p and seed; or Beam search with the number of beams and a length penalty. The penalties (repetition, frequency, presence, no-repeat n-gram) apply to all three. |
| **New seed for each reply** | Sampling, on by default: each reply draws differently and shows its seed; **Use this seed** fixes it. Off: every reply uses the seed field. Greedy and beam search always give the same reply. |
| History (chat) | Whether earlier turns are sent with each message |
| **KV cache** | On by default: each layer's keys and values are reused, so every new token feeds only itself; in a chat the cache is kept between messages too (What the model saw says how many tokens came from it). Off recomputes the whole sequence for every token. A graph with branches has no KV cache. |

**Measure speed** runs the message (or the last one) greedily for the full
max new tokens, with the cache and without, and shows tokens per second for
each. (Added in 1.0.8.)

**Reset to the graph's** brings back the graph's settings. **Save to the
graph** writes the template, labels, system prompt, stop texts, token limit
and decoding into the graph's Data Loader (its prompts, cadence and on/off
switch are kept); reopen the graph on the canvas to see the change. A
package carries no generation settings, so it starts with the defaults.

## Where conversations are kept

A conversation with a checkpoint is saved after each reply as a generation
report in the run's folder: `<run>/generation_chats/chat.json`
(`chat_2.json`, ... for the next ones). The file has the same form as the
training previews. Conversations with a package or with this session's
model are not saved; use **Export (JSONL)**.

## Check a tokenizer with raw token IDs

Under **Details**, **Raw token IDs** generates from token IDs you type and
shows what they decode to: a way to check a tokenizer or the model itself.

## Related

- [Generation previews](../training/generation-previews.md)
- [Fine-tune a language model](../training/fine-tune-language-models.md)
- [Read test results](test-results.md): Results > Generations
