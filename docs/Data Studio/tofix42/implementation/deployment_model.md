# tofix42 Deployment Model

## Purpose

Decide how the local CyxWiz source-aware assistant should ship before it is
integrated into Studio surfaces such as the Debugger.

This note separates:

- engine/runtime coupling
- knowledge/corpus packaging
- Studio integration surface
- optional plugin packaging

## Short Answer

Yes: deployment should be designed before Studio integration.

The current assistant is useful, but it should not be welded into the engine
core first. The correct first shipping shape is:

- read-only
- optional
- local-first
- replaceable
- packaged separately from the training/runtime core

That means:

- the engine does not depend on the assistant
- Studio can call the assistant when available
- the assistant can be disabled or omitted without breaking engine behavior

## Recommended Shipping Shape

Ship it as an optional local assistant module or plugin, not as a mandatory
engine subsystem.

Recommended shape:

1. `CyxWiz engine core`
   - training, graph compile, runtime, debugger data production
2. `Assistant package`
   - retrieval index
   - prompt/answer adapter
   - local model runtime configuration
   - validation/test helpers
3. `Studio integration point`
   - one narrow caller such as:
     - `Explain selected trace`
     - `Explain terminal reason`
     - `Find related source`

This keeps the assistant useful without making engine correctness depend on it.

## Why Not Put It Directly In the Engine Core

Because the assistant has a different change rate and failure mode than the
engine.

Engine core responsibilities:

- deterministic execution
- graph compile/runtime behavior
- debugger trace production
- training correctness

Assistant responsibilities:

- retrieval
- evidence packaging
- model answering
- explanation UX

If these are tightly coupled:

- model/runtime issues can contaminate core product behavior
- assistant deployment becomes harder to version
- offline/debug builds become heavier
- future replacement becomes expensive

So the assistant should consume engine outputs, not become one of them.

## Plugin vs Built-In Module

There are two realistic options.

### Option A: Optional Plugin

Best first shipping choice.

Properties:

- installable/enabled separately
- Studio detects whether it exists
- Studio shows assistant actions only when available
- assistant can evolve independently

Good for:

- internal teams first
- experimental rollout
- keeping the engine lean
- easy disable/fallback

### Option B: Bundled but Optional Module

Also reasonable later.

Properties:

- shipped with Studio installer
- still disabled or hidden by policy/config if needed
- tighter packaging, but still not a hard dependency

Good for:

- mature assistant behavior
- stable local runtime packaging
- simpler end-user install experience

## Recommended Rollout Order

1. Internal command/test harness
   - already done
2. Optional plugin or sidecar package
3. One narrow Studio Debugger action
4. Broader Studio surfaces after usage proves value

Do not invert that order.

## Does It Need the Entire Codebase

Not necessarily.

This assistant is based on indexed knowledge, not magic access to all source at
all times.

You have three deployment choices:

### 1. Internal developer mode

Use the real repository checkout.

Best for:

- engine developers
- source debugging
- architecture questions

Properties:

- index built from actual source/docs/examples
- highest fidelity
- requires local repo access

### 2. Curated shipped knowledge pack

Ship a prebuilt index or curated document pack.

Best for:

- Studio users who should not need the full repo
- customer-facing or non-dev installs

Properties:

- package only approved docs/source excerpts/example graphs
- avoid shipping the whole source tree if not desired
- reproducible and versioned with the engine release

This is usually the right production answer.

### 3. Hybrid mode

Use shipped knowledge by default, and allow developer override to point at a
real repo checkout.

Best overall long-term shape.

Properties:

- normal users get the curated pack
- developers can opt into full local source indexing

## Recommended Knowledge Packaging

For production-like shipping, do not require the full raw codebase in the
install.

Instead ship one of:

- a prebuilt retrieval index
- a curated source/docs/example subset
- a versioned assistant knowledge pack

That pack should contain only what the assistant needs:

- approved docs
- selected source excerpts or indexed chunks
- selected example graphs
- optional structured diagnostics metadata

This keeps packaging smaller and reduces accidental IP overexposure.

## Versioning Rule

The assistant knowledge pack must be versioned with the engine build it
describes.

Minimum rule:

- assistant knowledge version must match engine release version or commit band

Otherwise the assistant may explain old behavior against a newer runtime.

## Runtime Topology

Recommended local topology:

```text
Studio UI / Debugger
        |
        v
assistant plugin or module
        |
        +--> retrieval index / knowledge pack
        |
        +--> local answer adapter
        |
        +--> optional local model server
```

Important:

- Studio talks to the assistant
- assistant talks to retrieval/model pieces
- engine core is not blocked by assistant availability

## Failure Behavior

If the assistant is unavailable:

- Studio still works
- Debugger still works
- training still works
- graph compile still works

The assistant should fail soft:

- hide assistant actions, or
- show "assistant unavailable", or
- return retrieval-only context when the model is unavailable

## Best First Product Use

The strongest first product use is still Studio Debugger explanation.

Why:

- it uses structured context you already have
- it is read-only
- users want explanation, not mutation
- it is easy to keep bounded

Recommended first action:

- `Explain selected trace`

Recommended second action:

- `Explain training terminal reason`

## Recommendation

Use this deployment decision:

```text
Ship the assistant as an optional local plugin/module with a versioned
knowledge pack. Do not make engine runtime depend on it. Integrate first into
one narrow Studio Debugger explanation action.
```

## What This Means For the Plan

Before real Studio integration, define:

1. plugin/module boundary
2. knowledge-pack contents
3. versioning rule
4. fallback behavior when unavailable
5. first Studio action

Only then implement the first UI entry point.
