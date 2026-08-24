# Modularity and Source-Growth Guardrails

Use these guardrails when adding behavior to an existing codebase, especially
when the likely target file is already large or owns several responsibilities.

## Inspect Before Editing

1. Read the target file and its public interface completely enough to identify
   its responsibility and invariants.
2. Find callers, implementations of the same concept, related tests, and the
   dependency direction.
3. Check whether the requested behavior already exists behind another UI,
   service, adapter, or legacy path.
4. Identify user-owned or unrelated changes before editing shared files.

Do not begin by choosing a new class. Begin by describing the data and control
flow that must change.

## Choose a Module Boundary

A module should have:

- one cohesive reason to change;
- explicit inputs, outputs, ownership, and failure behavior;
- a narrow interface that hides replaceable implementation details;
- tests that can exercise the boundary without constructing the entire
  application;
- dependencies pointing toward stable domain contracts rather than UI or
  platform details.

Prefer capability names such as `data_preview_service`, `query_session`, or
`file_dialog_adapter`. Avoid vague containers such as `misc_utils`,
`common_helpers`, `global_manager`, or `base_service`.

## Treat Size as a Review Trigger

Line count is not a design metric, but it is useful evidence. Apply these as
review triggers for hand-written code, excluding generated code and large
declarative tables:

- Around 500 lines: inspect cohesion before adding a new feature or state.
- Around 1,000 lines: default to extracting a responsibility unless the file
  is one cohesive algorithm or protocol implementation.
- A change adding roughly 150 or more lines to an already large file: require
  an explicit boundary decision in the plan or review.
- A function that no longer fits on one screen, has multiple phases, or mixes
  policy with mechanics: extract named operations or a focused collaborator.

Never split only to satisfy a number. A poor split creates forwarding wrappers,
shared mutable state, circular dependencies, or files that cannot be understood
alone. Extract along ownership and invariant boundaries.

## Separate Common Responsibilities

Keep these concerns separate when they can change independently:

- UI rendering and domain decisions;
- parsing and validation;
- storage and business logic;
- synchronous computation and task orchestration;
- platform-independent policy and platform adapters;
- transport DTOs and domain models;
- analytical tables and training tensors;
- source data, working views, and published outputs.

Small private helpers are appropriate when they clarify a cohesive algorithm.
A new module is appropriate when behavior has its own state, lifecycle,
dependencies, tests, or reuse boundary.

## Control Dependencies

- Keep dependencies acyclic and point them toward stable contracts.
- Pass required collaborators explicitly; avoid service location and hidden
  globals unless the application architecture already owns that lifecycle.
- Expose the minimum public API. Keep implementation details private.
- Prefer composition over inheritance for replaceable behavior.
- Avoid framework types in domain interfaces when a small project-owned type
  preserves independence.
- Add an extension point only for a demonstrated variation.

## Change Existing Large Files Safely

1. Characterize current behavior with tests or observable evidence.
2. Extract one cohesive responsibility without changing behavior.
3. Validate the extraction.
4. Implement the requested change through the new or corrected boundary.
5. Remove obsolete paths after consumers migrate; do not leave permanent
   parallel implementations.

Avoid combining a broad rewrite with a feature unless the current boundary
makes a safe feature impossible. Preserve public behavior deliberately and
document any migration or compatibility period.

## Review Checklist

- Does every changed file have a clear purpose?
- Did the change reduce or increase the number of sources of truth?
- Are state ownership and lifecycle visible from interfaces?
- Can the new module be tested independently?
- Did extraction create circular dependencies or excessive forwarding?
- Is legacy behavior removed, time-bounded, or supported by a real requirement?
- Is optional functionality absent from the common runtime path?
