# To Fix 8 - CyxWiz LLM Upgrade Gaps

This document captures the missing pieces found during the CodeFeedback text demo analysis.

Current relationship to `tofix19.md`: keep this file as the focused
LLM/text-generation note. Use `tofix19.md` as the broader source of
truth for unsupported model families.

The short version:

- CyxWiz can already train small supervised text models.
- CyxWiz does not yet support a real end-to-end LLM training flow.
- The UI exposes more NLP/transformer nodes than the backend can actually train.
- The text pipeline still needs persistence, inference, dataset, and graph validation upgrades before it can be presented as a believable LLM path.

---

## 1. What We Confirmed Works Today

These parts are real and usable now:

- Text dataset loading through `TextDataset` and `TextDatasetBatcher`
- Whitespace / word / character tokenization
- Fixed vocabulary loading from a text file
- Padding and truncation to a fixed sequence length
- Supervised text classification training
- `Embedding -> GRU/LSTM -> Dense` style sequence models
- `Embedding -> TransformerEncoder -> Dense` text-classification style
  models
- Embedded HTTP inference for numeric tensor input

This means CyxWiz can already support:

- text classification
- language ID style prediction
- small sequence models
- demo pipelines for NLP-like workflows

It does not yet mean CyxWiz can train a transformer LLM.

---

## 2. What Is Missing For A Real LLM Path

### 2.1 Transformer generation backend

The main missing piece is not every transformer path. The current engine
has a verified encoder-classifier lane, but does not have a decoder or
causal language-model lane.

Observed state:

- `TransformerEncoder` is directionally usable for text classification.
- `TransformerDecoder` and GPT-style causal training are not implemented
  as an end-to-end training path.
- `MultiHeadAttention` as a standalone graph/import surface remains
  guarded until the graph compiler, builder, and runtime contracts are
  aligned.

Impact:

- The UI can suggest broader transformer capability than the trainer
  actually delivers.
- Users can build graphs that look like LLM/generation graphs but cannot
  train them end to end.

### 2.2 Causal / autoregressive language modeling

The current text pipeline is supervised classification, not causal next-token training.

Missing pieces:

- shifted token targets
- causal masking in the training loop
- teacher forcing for decoder-style training
- token-level loss over sequences
- sampling / generation loop for inference

Impact:

- You cannot yet train a GPT-style model.
- You cannot yet generate text from a prompt.
- You can only classify a prompt into a label.

### 2.3 Tokenizer persistence

The training pipeline can build a vocabulary, but the model artifact does not currently package tokenizer state cleanly as a first-class deployment asset.

Missing pieces:

- vocabulary file embedded in the saved model package
- tokenizer configuration embedded in the saved model package
- explicit tokenization metadata in the deployment manifest
- clear versioning for tokenizer + vocab + model bundle

Impact:

- Inference depends on external files.
- Reproducibility is weaker than it should be.
- A real LLM flow needs the tokenizer to ship with the model.

### 2.4 Raw-text inference endpoint

The embedded inference server accepts numeric tensors only.

Missing pieces:

- raw text request body support
- automatic tokenization inside the server
- prompt formatting support
- generation response support for text models

Impact:

- The current Python helper must tokenize text before sending it.
- That is fine for the demo, but it is not a user-friendly LLM API.

### 2.5 HuggingFace dataset support

The current HuggingFace dataset support is still a placeholder/demo path.

Missing pieces:

- a real dataset loader for instruction-tuning corpora
- support for streaming/large dataset ingestion
- mapping of `query`/`answer` style records into supervised or causal training samples
- proper split handling for train/val/test or packed sequences

Impact:

- The CodeFeedback blobs cannot yet be used as a true instruction-tuning dataset inside CyxWiz without conversion.

### 2.6 Graph truth mismatch

The UI exposes more than the backend can train.

Missing pieces:

- a truthful capability registry for nodes
- compile-time blocking for unsupported LLM nodes beyond the guardrails
  already added for unavailable attention and Dense-encoded NER target
  designs
- clear labels for experimental vs supported nodes
- consistency between node visibility, docs, compiler, and builder

Impact:

- Users can build graphs that compile visually but fail semantically.
- Demo quality suffers because the product appears to support more than it really does.

---

## 3. Specific Upgrade Work For CyxWiz

### 3.1 Add a real text generation training path

Work needed:

- add causal LM support
- implement a transformer block path in the backend
- support token-shifted targets
- support attention masks
- add generation sampling for inference

Result:

- CyxWiz can move from text classification to actual language modeling.

### 3.2 Package tokenizer + vocab with the model

Work needed:

- store tokenizer settings in `.cyxmodel`
- store vocabulary file or token map in `.cyxmodel`
- ensure embedded deployment can reconstruct the exact tokenizer

Result:

- The model becomes self-contained and reproducible.

### 3.3 Add raw-text inference

Work needed:

- accept raw text in `/v1/predict`
- tokenize inside the server when the model is a text model
- optionally support generation mode for LLMs

Result:

- Users can test text models without writing a separate tokenizer script.

### 3.4 Upgrade HuggingFace / instruction dataset handling

Work needed:

- real loader for instruction datasets
- support for query/answer training examples
- support for packing, truncation, EOS handling, and prompt templates
- better metadata around labels and targets

Result:

- The CodeFeedback-style corpus can be used as a proper instruction-tuning source.

### 3.5 Make node support truthful

Work needed:

- one support matrix for UI, compiler, builder, and runtime
- hide unsupported transformer nodes until they are real
- mark experimental nodes clearly

Result:

- The graph editor becomes trustworthy.
- Demo graphs stop overpromising.

---

## 4. What We Should Do First

If the goal is a believable LLM story, the first work items should be:

1. Build a tokenizer/vocab bundle into the saved model artifact.
2. Add raw-text inference support for text models.
3. Add a real causal language-model training path.
4. Add a truthful node capability registry so the UI matches the backend.
5. Upgrade dataset loading for instruction-tuning style corpora.

If the goal is only to improve the current demo quality, then the first work items should be:

1. Keep the current text classifier path stable.
2. Improve the graph README and node descriptions.
3. Add test coverage for text dataset loading and vocab reuse.
4. Make the embedded test flow cleaner for classification.

---

## 5. Practical Conclusion

CyxWiz is already strong enough for:

- small NLP classifiers
- sequence models
- graph-driven text demos

CyxWiz is not yet strong enough for:

- true LLM training
- transformer-based instruction tuning
- prompt-to-text generation

The current CodeFeedback demo is useful because it shows the nearest working path.
The next product step is to turn that path into a real, self-contained text-model workflow.
