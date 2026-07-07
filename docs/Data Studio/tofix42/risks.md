# tofix42 Risks

## Hallucinated Engine Capabilities

Risk: the assistant claims CyxWiz supports nodes, properties, debugger lenses,
GPU behavior, graph edits, or export features that do not exist.

Controls:

- retrieve supported node/property truth before answering
- prefer current source and structured traces over docs
- require citations for engine-specific claims
- use explicit unknown/unsupported language
- test refusal behavior for unsupported capabilities

## Stale Source Knowledge

Risk: a fine-tuned or cached model answers from old CyxWiz behavior.

Controls:

- do not start with fine-tuning
- rebuild or incrementally update the local index
- store content hashes and timestamps
- show index freshness in Studio
- keep retrieval mandatory even after future fine-tuning

## Weak Retrieval

Risk: the answer model receives irrelevant snippets and produces plausible but
wrong explanations.

Controls:

- hybrid lexical/vector retrieval
- exact-match boosts for symbols, paths, node types, run ids, and error text
- intent-aware retrieval routing
- trust-rank reranking
- evaluation fixtures for known source/debugger/graph questions

## Runtime Trace Misinterpretation

Risk: the assistant treats raw logs or incomplete traces as proof.

Controls:

- structured traces outrank logs
- answers separate fact from inference
- cite run id, trace phase, role, status, and node fields
- state when a trace proves the symptom but not the root cause
- consume future structured diagnostics instead of plain text where possible

## Security and Mutation Risk

Risk: users trust the assistant to edit graphs/source or run actions before the
approval model is ready.

Controls:

- first milestone is read-only
- graph drafts are proposals only
- source edits are out of scope
- no automatic training launch
- no delete/rewrite operations for traces or logs
- explicit user approval required for any future mutation path

## Privacy and Data Leakage

Risk: private dataset rows, logs, paths, or source code are sent outside the
machine or indexed too broadly.

Controls:

- local-only model and embeddings by default
- no network dependency
- do not index private dataset rows by default
- do not index binary model weights
- provide index delete/rebuild controls
- use redaction hooks for support-bundle-like records

## Performance and Memory

Risk: indexing and local generation slow Studio or consume too much memory.

Controls:

- optional assistant module
- background, cancelable indexing
- bounded default roots
- quantized small model
- separate embedding and answer model
- cache embeddings by content hash
- avoid loading model during training-critical paths unless requested

## User Experience Overclaim

Risk: Studio UI labels imply the assistant fully understands the engine or can
fully trace every graph.

Controls:

- precise labels such as `Explain selected trace`, `Draft graph`, and
  `Find likely source`
- avoid labels like `fix automatically`, `complete graph trace`, or
  `understands everything`
- show evidence and missing evidence
- preserve tofix32 distinction between real, synthetic, smoke, and estimated
  traces

## Graph Generation Invalidity

Risk: generated `.cyxgraph` drafts use invalid pins, unsupported properties, or
bad training configuration.

Controls:

- retrieve validated examples in the same domain
- retrieve node/property definitions before generation
- constrain output to supported nodes
- run compiler/preflight in a later user-approved phase
- label drafts as drafts until validated
- collect invalid drafts as evaluation failures

## Fine-Tuning Misuse

Risk: fine-tuning starts too early and bakes in wrong or stale behavior.

Controls:

- explicitly defer fine-tuning
- collect reviewed examples only after RAG is working
- require source citations or validation evidence for examples
- compare fine-tuned model against base model plus retrieval
- do not remove retrieval after fine-tuning

## Diagnostic Contract Gaps

Risk: source-aware answers are limited because current engine diagnostics are
not structured enough.

Controls:

- integrate with `DebugTraceRecord`, `DebugRunStore`, and
  `TrainingTraceCollector` first
- consume future tofix26/tofix32 structured diagnostics when available
- answer from logs only with lower confidence
- recommend missing trace/diagnostic fields as engine follow-up

## Risk Acceptance Criteria

- Assistant output cites evidence or states that evidence is missing.
- Unsupported features are refused or qualified.
- Graph/source mutation is impossible in the first milestone.
- Local model/index failure does not affect training.
- Indexing has clear privacy and rebuild boundaries.
- Fine-tuning is not part of the first implementation.
