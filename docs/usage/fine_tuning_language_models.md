# Fine-tuning language models in the Engine

How to fine-tune a trained language model on prompt/answer examples with
graphs only: read the examples from JSON Lines, render them with SQL, build
token windows where only the answer is trained, and start training from an
earlier run's weights. Added in TOFIX143; the worked example is Project
Berean (`examples/thinking_llm/Project Berean`, `docs/sft_windows_v1.md`).

The pieces work separately too: JSON Lines input and SQL over nested fields
are general Data Studio features, and "Start from" warm-starts any model.

## 1. Read JSON Lines records (Data Input)

Data Input reads **JSON Lines**: one JSON object per line, usually a `.jsonl`
file. Set the format to **JSON Lines** (`type: json`); a `.jsonl` file is
recognised by its extension, any other name needs `json_lines: true`.

- Nested objects become struct columns, arrays become list columns
  (an array of objects is a list of structs).
- Keys that only some rows have are kept; missing values are null.
- A plain JSON file (one array or object) is refused with a message asking
  for JSON Lines. Convert it first (one record per line).

## 2. Shape the examples with SQL (Preparation Recipe)

The SQL step can read list and struct columns directly, using DuckDB syntax:

```sql
SELECT
    id AS document_id,
    split,
    'System: ' || list_filter(messages, m -> m.role = 'system')[1].content || chr(10) ||
    'User: '   || list_filter(messages, m -> m.role = 'user')[1].content   || chr(10) ||
    'Berean:'  || chr(10)                                                   AS prompt_text,
    list_filter(messages, m -> m.role = 'assistant')[1].content            AS response_text
FROM input
ORDER BY id
```

- Results must be flat (text, numbers, booleans). Returning a list or struct
  column is refused; select the fields you need instead.
- Keep the chat template in one place: generation and chat must render the
  prompt exactly the same way as training did.
- Add a Row Count Check so a wrong filter cannot silently drop examples.

## 3. Build supervised windows (Text Tokenizer)

Text Tokenizer, Output **Causal document windows**, with a
**Response column** (`target_text_col`):

| Setting | Value |
| --- | --- |
| Text column | the prompt (`prompt_text`) |
| Response column | the answer (`response_text`) |
| Document ID / Split column | unique id per example / train, validation or test |
| Vocabulary file | the **saved vocabulary of the model you fine-tune** |
| Max length | the training context (every example must fit) |

Each row becomes **one window**: prompt tokens, then answer tokens and EOS.
Only the answer tokens (and EOS) count toward the loss and accuracy; the
prompt is context. The output carries a `__target_mask` column and the
metadata `cyxwiz.token_windows.supervision = response_only`; training and
validation read it automatically (Run Test reads windows through the same
loader; an explicit answer-only Run Test check is still open in TOFIX143).

Rules and the messages you may see:
- **An example longer than one window fails** and names the document
  ("needs 612 ids but one window holds 513"). Examples are never split, since a
  split would separate an answer from its question. Raise Max length or shorten
  the example. To find the right length, build once with a large Max length and
  read the window lengths.
- **A vocabulary file is required.** Supervised windows never fit a new
  vocabulary; they must use the tokenizer the model was trained with.
- An empty response is an error.
- Training data and its validation/test data must both be supervised (or both
  plain); a mix is refused.

## 4. Start from a trained model ("Start from")

On the training node (Data Loader), **Start from**
(`initialize_from_checkpoint`) names a checkpoint folder of an earlier run,
for example `checkpoints/my_base/train-1791733219499/best`. A relative path is
resolved against the project, like Data Input paths.

- Only the **model weights** are loaded. Optimizer state, learning-rate
  schedule (including warmup), data order and epochs start fresh; set the
  learning rate for fine-tuning (usually 10x or more below pre-training).
- The model in the graph must match the checkpoint exactly: same layer names,
  shapes and types. A mismatch stops the run and names the parameter.
- For language models the checkpoint's tokenizer must match the training
  windows. New checkpoints record `tokenizer_sha256`; for older ones the run's
  generation previews are used. If neither exists, the run stops rather than
  guessing.
- **Start from** and **Resume** are different: Resume continues the same run;
  Start from begins a new run. Using both is refused.
- New checkpoints record where the run started (`initialized_from`) and the
  context it trained at.

The log shows: `initialized model weights from <folder> (source epoch N,
val_loss X); optimizer, schedule and epochs start fresh`.

## 5. Longer context

Models with rotary positions (RoPE) have no weights tied to the context
length, so a longer context needs no special step:

1. Re-encode the training text with the same vocabulary at the new Max length.
2. Set the training node's max sequence length to the new context.
3. **Start from** the shorter-context checkpoint; a short continued training
   at the new length lets the model adapt.

A model with a learned position table fails at Start from with "Checkpoint
shape mismatch for parameter '<name>'"; that model cannot change context this
way.

A longer context costs memory: a 768 window holds three times the tokens of a
256 window, so lower the micro-batch and raise gradient accumulation to keep
the effective batch.

## Typical graph

```
Data Input (JSON Lines) -> Preparation Recipe (SQL render + Row Count Check)
    -> Text Tokenizer (causal windows, Response column, saved vocabulary)
    -> Export Parquet
Training graph: Data Input (windows) -> Data Loader (Start from, context)
    -> model -> CrossEntropy (ignore_index -100) -> AdamW
```

## Seeing answers during training

Data Loader **Generation previews** generate from the current weights after
chosen epochs. Chat prompts span several lines, so put a line containing only
`---` between prompts; each prompt then keeps its line breaks. Leave an empty
line before `---` when the prompt must end with a newline (as after
`Berean:` above). Without `---`, every line is its own prompt. Up to 8
prompts, 2048 bytes each.

## Checking it worked

- The Text Tokenizer output has `__target_mask`, and `__valid_targets` equals
  the number of answer tokens per window.
- At the start of training the log shows the warm-start line, and the first
  loss is far below ln(vocabulary size) (an untrained model starts there).
- Validation loss and accuracy are measured on answer tokens only.
