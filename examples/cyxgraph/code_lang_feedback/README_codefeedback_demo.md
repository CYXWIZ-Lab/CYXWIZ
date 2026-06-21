# CodeFeedback Multi-Class Text Demo

This demo shows the smallest text model CyxWiz can train today on the CodeFeedback blobs.

The generator now defaults to a filtered, trainable multi-class subset instead
of trying to learn the full long-tail label space in one small GRU run.

It is not a transformer LLM. It is a supervised text classifier built with:
- character tokenization by default
- a fixed vocabulary file
- `Embedding -> GRU -> Dense` classifier head

The goal is to show a complete training and inference loop on real text data, while staying inside the text pipeline that CyxWiz currently supports.

## What We Are Building

The model reads an instruction-style prompt built from:
- `query`
- `answer`

The prep script turns those fields into one text sample like:

```text
[q] Write a function to reverse a string [a] def reverse_string(s): return s[::-1]
```

The label is the source `lang` field. In the generated dataset, the classifier is multi-class and uses all labels found in the blobs.

At inference time, the model returns the most likely `lang` label for a new prompt. It does not generate code. It classifies the prompt.

## Files You Will Use

- Raw source blobs:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\blobs`
- Generated demo data:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data`
- Generated CSV:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_lang_multiclass.csv`
- Generated vocabulary:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_vocab.txt`
- Generated metadata:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_metadata.json`
- Generated graph:
  `D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_lang_classifier_multi.cyxgraph`
- Inference helper:
  `D:\Dev\CyxWiz_Claude\examples\python\codefeedback_text_inference.py`

## 1. Prepare The Data
 python .\blobs\prepare_codefeedback_demo_dataset.py --source .\blobs --out-dir .\data
Run the prep script against the raw blob directory:

```powershell
python D:\Dev\CyxWiz_Claude\examples\python\prepare_codefeedback_demo_dataset.py `
  --source D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\blobs `
  --out-dir D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data `
  --tokenizer-type character `
  --text-mode answer_only `
  --max-length 512
```

By default this writes a filtered multi-class dataset:
- `codefeedback_lang_multiclass.csv`
- `codefeedback_vocab.txt`
- `codefeedback_metadata.json`
- `codefeedback_lang_classifier_multi.cyxgraph`

Default filtering behavior:
- normalizes obvious label aliases such as `cs -> csharp`, `yml -> yaml`, `txt -> plaintext`
- keeps the top 12 normalized labels
- drops labels with fewer than 200 raw examples
- down-samples each remaining class to 1200 rows

This is intentional. The previous "all labels" dataset created 100+ classes with
extreme imbalance and many near-empty labels, which performs poorly with the
current small GRU demo.

For code-language classification, the answer text usually carries the strongest
signal, so the default dataset now trains on `answer_only` samples instead of the
full prompt.

Default tokenizer behavior:
- character-level tokenization
- larger default max length so code syntax survives truncation better
- the inference helper reads `tokenizer_type` from metadata and uses the same tokenization path

Default text mode:
- `answer_only`
- this keeps the code-bearing signal and drops most of the natural-language prompt noise
- the inference helper reads `text_mode` from metadata and builds inputs the same way

If you explicitly want the old broad dataset, run:

```powershell
python D:\Dev\CyxWiz_Claude\examples\python\prepare_codefeedback_demo_dataset.py `
  --source D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\blobs `
  --out-dir D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data `
  --top-k-labels 0 `
  --min-label-count 1 `
  --balance-mode oversample `
  --balance-target 300
```

The metadata file records:
- selected labels
- explicit `label_order` used for inference decoding
- row count
- vocab path
- max sequence length
- class count
- label filtering and alias normalization choices

## 2. What Each Node Does

`DataInput`
- Loads the generated CSV.
- Reads `text` as the input column and `label` as the target column.
- This is the source dataset registration point for training.

`TextTokenizer`
- Defines how text is split into tokens.
- The default demo now uses character tokenization because code syntax is punctuation-heavy and whitespace tokenization was too sparse.
- It also sets the max sequence length used by the pipeline.

`TextVocabulary`
- Loads the fixed vocabulary from `codefeedback_vocab.txt`.
- This keeps token IDs stable between training and inference.
- Without this, the model and the inference script would not agree on token indices.

`TextPadding`
- Pads or truncates every sample to the same sequence length.
- CyxWiz needs a fixed-width tensor for batching.

`DataSplit`
- Splits the data into train, validation, and test partitions.
- The split is stratified so each label is represented in each partition.

`DataLoader`
- Controls batch size, shuffling, epochs, and validation frequency.
- This is the runtime batching step used during training.

`Embedding`
- Converts token IDs into dense vectors.
- This is the first learnable layer.
- It lets the model learn that some characters or tokens are related, even if they are not identical.

`GRU`
- Reads the token sequence and learns order/context patterns.
- This is the sequence modeling layer.
- For this demo it replaces a transformer, which CyxWiz does not fully train end to end yet.

`Dense 64`
- Compresses the GRU output into a smaller feature space.
- Helps the classifier head make the final decision.

`ReLU`
- Adds non-linearity so the model can represent more than a linear mapping.

`Dropout`
- Regularizes the classifier head.
- Reduces overfitting by randomly dropping activations during training.

`Dense N`
- Final classification layer.
- `N` equals the number of labels found in the source data.

`CrossEntropyLoss`
- Training objective for multi-class classification.
- Penalizes the model when the correct label is not assigned the highest score.

`Adam`
- Optimizer used to update model weights.
- Good default choice for this kind of demo.

`Output`
- Graph endpoint marker.
- Not a learnable layer.
- Used to mark the end of the model path.

## 3. Train In CyxWiz

Open the generated graph:

`D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_lang_classifier_multi.cyxgraph`

Then:
1. Open `Data Input`.
2. Confirm the CSV path is correct.
3. Confirm the vocab file path is correct in `TextVocabulary`.
4. Apply the data node.
5. Compile the graph.
6. Train the model.

## 4. How To Test The Model

The inference helper reads the same metadata and vocabulary file, tokenizes the input the same way, pads it to the same length, then sends token IDs to the embedded server.

Single prompt:

```powershell
python D:\Dev\CyxWiz_Claude\examples\python\codefeedback_text_inference.py `
  --metadata D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_metadata.json `
  --query "Write a function to reverse a string" `
  --answer "def reverse_string(s): return s[::-1]"
```

Batch evaluation:

```powershell
python D:\Dev\CyxWiz_Claude\examples\python\codefeedback_text_inference.py `
  --metadata D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_metadata.json `
  --csv-file D:\Dev\DataSet_List\datasets--m-a-p--CodeFeedback-Filtered-Instruction\data\codefeedback_lang_multiclass.csv `
  --num-samples 20
```

What the script prints:
- predicted label
- confidence
- latency
- for batch mode, actual vs predicted per row

## 5. Expected Behavior

This is a classifier, so the output is one label from the training set.

Example:
- Input prompt looks like Python code
- Predicted label: `python`

Another example:
- Input prompt looks like C++ code
- Predicted label: `cpp`

If the model is weak, it may still confuse languages that look similar or share
syntax, but the character tokenizer should behave materially better on code-like
inputs than the previous whitespace-only setup.

## 6. Notes

- This is a teaching demo, not a production LLM.
- The engine currently supports this text classifier path, but not a full transformer training loop end to end.
- If you rerun the prep script, it will regenerate the CSV, vocab, metadata, and graph in the data folder.
