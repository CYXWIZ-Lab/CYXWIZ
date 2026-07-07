# tofix56 - Task-aware compiler graph validation contract

## Status

Open.

## Why this ticket exists

The causal language-model example exposed a real compiler design limitation.
CyxWiz already has useful compiler guardrails that catch bad graph structure, unsupported engine paths, missing labels, incompatible output shapes, and backend placement uncertainty.

The problem is that some rules are still written as if every training graph is a classic supervised classification/regression graph with an explicit label column.

That assumption is not generic enough for the engine we are building.

CyxWiz needs to support multiple training contracts:

- supervised classification
- supervised regression
- unsupervised clustering
- dimensionality reduction
- autoencoders
- self-supervised learning
- causal language modeling
- masked language modeling
- sequence tagging
- metric learning
- generative models

Thanks to the causal-LM example, we can separate compiler warnings/errors into:

- valid structural graph errors
- valid engine capability guards
- noisy warnings caused by supervised-only assumptions
- missing first-class task-contract checks

## Evidence from the causal-LM example

Example graph:

`examples/cyxgraph/text/causal_lm_decoder_generation_experiment.cyxgraph`

Compiler output showed errors/warnings around:

- missing label pin connections
- missing CrossEntropy targets
- DataInput label output not consumed
- missing label column
- CrossEntropy class count mismatch
- TransformerDecoder causal/generative training guard
- unsupported LayerNorm in SequentialModel
- unsupported TimeDistributed training wrapper
- sequence shape uncertainty
- backend placement uncertainty for TransformerDecoder / PositionalEncoding / TimeDistributed
- generic pre-train inspection recommendation

Some of these are correct. Some are too generic.

## Valid compiler behavior that should remain

### Structural graph errors

These are valid and should remain errors unless a task contract explicitly explains how the data is generated internally:

- `DataSplit` required labels but received only features.
- `DataLoader` required labels but received only features.
- `CrossEntropyLoss` required targets but received none.
- `DataInput` produced labels that were not consumed.
- Model output size did not match CrossEntropy class count.

For supervised graphs, these checks are correct.

### Engine capability guards

These are valid and should remain truth-based compiler errors/warnings:

- `LayerNorm` visible in analysis but unsupported by `ModelBuilder/SequentialModel`.
- `TimeDistributed` token-level training unsupported by Studio training.
- `TimeDistributed` sequence shape could not be proven.
- Backend placement unknown for newer sequence/transformer nodes.

The compiler should not pretend unsupported training paths work.

## Noisy or incomplete compiler behavior

### Label-column warning assumes supervised learning

Current behavior:

`No label column selected - training will use the last column as label by default`

This is correct for supervised graphs, but noisy or wrong for:

- unsupervised learning
- self-supervised learning
- causal language modeling
- masked language modeling
- autoencoders
- clustering
- embedding pretraining

Better behavior:

- supervised classification: require explicit label column or warn/error
- supervised regression: require explicit target column or warn/error
- unsupervised clustering: no label column required
- autoencoder: target is input
- causal LM: targets are shifted tokens
- masked LM: targets are masked tokens
- sequence tagging: labels are sequence-aligned tags
- metric learning: labels/pairs/triplets depend on metric-learning contract

### Generic data inspection warning is useful but too broad

Current behavior:

`No pre-train data inspection node found`

This is useful as a recommendation, but should become task-aware:

- classification: recommend class balance, value counts, missing labels
- regression: recommend target distribution, outlier checks
- text classification: recommend text length, label balance, vocabulary preview
- causal LM: recommend token length, vocab/OOV stats, sequence-target alignment
- clustering: recommend feature distribution and scaling checks
- autoencoder: recommend reconstruction input shape and normalization checks

### Causal decoder guard should check the full contract

The compiler currently rejects decoder/generative training when it sees generative hints such as `causal_mask`.

That is safe, but incomplete.

If the graph explicitly declares a full causal-LM task contract, the compiler should validate that contract instead of rejecting based on one flag.

Required causal-LM contract should include:

- `model_task=causal_lm` or `task_type=causal_lm`
- token sequence input shape known
- vocabulary size known
- shifted next-token targets declared
- token-level loss declared
- output logits shape compatible with vocab size
- padding/ignore index declared
- tokenizer/vocabulary packaging declared for export/generation
- generation metadata declared

If all required pieces are present and supported by the engine, compile.

If pieces are missing, report targeted errors.

## Desired design

Introduce a first-class graph task contract.

Recommended graph-level or DataInput/Output-level keys:

```text
task_type
model_task
target_mode
target_source
label_required
sequence_length
vocab_size
ignore_index
tokenizer_asset_path
vocabulary_asset_path
generation_enabled
```

Suggested task types:

```text
supervised_classification
supervised_regression
unsupervised_clustering
dimensionality_reduction
autoencoder
self_supervised
causal_lm
masked_lm
sequence_tagging
metric_learning
generative
```

## Compiler validation matrix

Add a task-aware validation matrix.

Each task should define:

- whether labels are required
- where targets come from
- allowed losses
- allowed output shapes
- required metadata
- required preprocessing/materialization behavior
- compatible metrics
- train/test/inference contract
- export metadata contract

Example:

```text
causal_lm:
  labels_required: false
  target_source: shifted_input_tokens
  allowed_losses: CrossEntropyLoss
  output_shape: [batch, seq, vocab]
  required_metadata: vocab_size, sequence_length, ignore_index
  inference: prompt -> tokenizer -> token ids -> generation -> decode
```

Example:

```text
unsupervised_clustering:
  labels_required: false
  target_source: none
  allowed_losses: none or clustering objective
  output_shape: cluster assignments or embeddings
  required_metadata: feature_columns
  inference: features -> cluster id / distance
```

## Error taxonomy

Compiler diagnostics should be categorized more precisely:

- `structural_error`
- `task_contract_error`
- `capability_error`
- `shape_error`
- `data_contract_error`
- `recommendation`
- `backend_placement_warning`
- `export_contract_warning`

This matters for Studio UX.

Users should know whether a message means:

- graph is invalid
- graph is valid but unsupported by current engine
- graph can run but may fall back to CPU
- graph can run but has data-quality risks
- graph needs a clearer task declaration

## Studio UX requirements

The compiler/debugger should show:

- detected task type
- whether labels are required
- target source
- loss compatibility
- output-shape expectation
- generated targets, if any
- backend placement plan
- unsupported engine capability, if any
- actionable fix guidance

For causal LM, Studio should say something like:

```text
Task: causal_lm
Labels: not required
Targets: shifted input tokens
Loss: token-level CrossEntropy
Output: vocab logits
Status: blocked because TimeDistributed/token-level training wrapper is not implemented
```

That is more truthful than warning that no label column was selected.

## Implementation plan

1. Add `TaskContract` model.
2. Add task detection from graph-level, DataInput, Output, and loss metadata.
3. Add explicit task declaration fields to saved graph parameters.
4. Replace generic label-column checks with task-aware label/target validation.
5. Replace generic output-class inference with task-aware output-shape inference.
6. Add task-specific data inspection recommendations.
7. Update compiler diagnostics with typed categories.
8. Update Studio compiler/debugger panel to display task contract truth.
9. Add tests for supervised, unsupervised, autoencoder, sequence tagging, metric learning, and causal-LM graphs.
10. Update usage docs with examples of label-required vs label-free training tasks.

## Tests required

Add tests for:

- supervised classification with missing label column still warns/errors
- unsupervised clustering does not warn about missing labels
- autoencoder accepts input-as-target contract
- causal LM accepts shifted-target declaration without a label column
- causal LM rejects missing vocab size
- causal LM rejects incompatible output vocab size
- causal LM reports unsupported TimeDistributed/token-level wrapper as capability error, not label-column error
- masked LM requires masked-token target metadata
- sequence tagging requires token-aligned tag targets
- compiler diagnostic categories are stable

## Success criteria

This ticket is done when:

- compiler validation is task-aware
- label warnings only appear when labels are actually required
- self-supervised and unsupervised graphs do not receive supervised-only warnings
- causal-LM graphs receive targeted causal-LM diagnostics
- unsupported engine paths are still blocked truthfully
- Studio shows the task contract clearly
- tests cover valid and invalid task contracts

## Relationship to other tickets

This ticket does not implement the full causal-LM training stack by itself.

It prepares the compiler and Studio validation layer so future tickets such as full LM-stack inference/training, TimeDistributed token-level training, tokenizer packaging, and transformer generation workflows can report truthful, task-aware diagnostics.
