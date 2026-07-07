# tofix42 Model and Runtime Design

## Position

The first assistant should be a local small model plus retrieval. It should not
start as a fine-tuned CyxWiz model.

The model's job is to read retrieved CyxWiz context, synthesize an answer, cite
the evidence, and avoid inventing engine behavior. The retrieval layer is the
truth layer.

## Runtime Requirements

The first runtime should be:

- local and offline-capable
- optional, not required for training
- CPU-capable
- quantized
- small enough for normal developer machines
- restartable without losing the workspace index
- isolated from graph mutation and source mutation

The assistant module should fail closed. If the model or index is unavailable,
Studio should show that the assistant is unavailable rather than degrading
training, graph editing, or debugger behavior.

## Model Classes to Evaluate

### Small Instruct Model

Use a general local instruct model for the first prototype.

Target profile:

- 3B to 8B parameter range
- 4-bit or 5-bit quantization
- context window large enough for multiple source and trace snippets
- good code and JSON following behavior
- acceptable CPU latency for short answers

This is the likely first path because the assistant mostly needs to answer from
retrieved context, not memorize CyxWiz.

### Code-Oriented Instruct Model

Evaluate a code-oriented small model for source questions and scripting help.

Strengths:

- better C++ source explanation
- better Python snippet generation
- better JSON and graph skeleton generation

Risks:

- may give weaker user-facing ML explanations
- may overfit to generic framework patterns instead of CyxWiz truth

### Embedding Model

The embedding model should be separate from the answer model.

Requirements:

- local CPU inference
- stable embeddings across restarts
- good code and documentation retrieval
- supports short queries and longer chunks
- cheap enough for incremental indexing

The embedding model can be smaller than the answer model. It should not require
network access.

## Runtime Backends

### llama.cpp-Style Backend

Good first candidate:

- mature quantized CPU path
- easy model file deployment
- works offline
- supports memory-constrained machines

Integration should be through a narrow assistant runtime interface so CyxWiz can
replace the backend later.

### ONNX Runtime Backend

Useful candidate if CyxWiz wants one runtime family for other model workloads.

Strengths:

- portable
- familiar runtime packaging
- potential GPU provider support later

Risks:

- text generation model support and quantized deployment may be more complex
  than llama.cpp-style runtimes for the first prototype

### Future CyxWiz-Native Runtime

Long term, CyxWiz may run assistant inference through its own model packaging
or execution stack. That should be a later phase after the assistant product
contract is proven.

Do not block the first assistant on native CyxWiz text-generation inference.

## Prompt Contract

The runtime prompt should enforce:

- answer only from retrieved CyxWiz context when making engine-specific claims
- cite source files, docs, graph files, trace ids, or run ids
- separate fact, inference, and recommendation
- say when evidence is missing or stale
- prefer current source and structured traces over docs
- do not claim unsupported graph nodes, properties, diagnostics, or backend
  capabilities exist
- do not mutate files or graphs

Recommended answer sections for technical questions:

- `Evidence`
- `Answer`
- `Recommendation`
- `Unsupported or unknown`

Studio UI can render these sections more compactly.

## Graph Generation Mode

Graph generation should be a constrained mode, not free-form prose.

Before drafting a graph, the model needs retrieved context for:

- supported node types
- node properties and defaults
- valid pin/link patterns
- graph examples in the same domain
- compiler/preflight constraints

The assistant output should be a draft with warnings, not an automatic graph
edit. Later phases can pass the draft through preflight and ask the user before
applying it.

## Tool and Action Boundaries

The local model should be read-only in the first milestone.

Allowed:

- explain a warning or failure
- summarize a run
- point to likely source files
- draft a graph
- draft a Python snippet
- recommend graph changes
- recommend debugger follow-up

Not allowed in the first milestone:

- editing source files
- editing graph files
- starting training
- deleting or rewriting traces/logs
- installing models or dependencies without explicit user action
- sending code, traces, logs, or data to network services

## Future Fine-Tuning

Fine-tuning is only justified after:

- RAG quality has been measured
- bad answers have been collected
- source-grounded QA pairs exist
- graph drafts have been validated by preflight
- debugger examples have structured trace evidence
- maintenance cost is accepted

Even after fine-tuning, retrieval remains mandatory because CyxWiz source and
graph contracts change.

## Model Acceptance Criteria

- Runs locally without network access.
- Answers from retrieved context with citations.
- Handles source, docs, graph, and trace snippets in one prompt.
- Refuses or qualifies unsupported engine-specific claims.
- Produces deterministic-enough graph drafts when context is supplied.
- Keeps assistant failure isolated from engine training and Studio workflows.
