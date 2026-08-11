---
name: lean-software-guardrails
description: Apply lean software engineering guardrails based on Niklaus Wirth's "A Plea for Lean Software." Use when planning, implementing, reviewing, refactoring, or debugging coding and software engineering projects where complexity, feature creep, bloated dependencies, weak decomposition, extensibility, performance, maintainability, or engineering discipline matter.
---

# Lean Software Guardrails

## Core Stance

Keep software small, understandable, and extensible. Treat every feature,
dependency, abstraction, layer, background service, generated artifact, and
configuration knob as a cost until it proves essential.

Software growth often tracks available hardware rather than user value.
Counter that tendency with disciplined design, simple primitives, coherent
modules, strong interfaces, and iterative refinement.

Read [wirth-lean-software.md](references/wirth-lean-software.md) when work
involves architecture, code review, major refactoring, framework choice, or
adding functionality.

## Operating Workflow

1. Identify the essential user or system capability.
2. Separate essentials from conveniences, compatibility burdens, visual
   embellishments, and nice-to-have features.
3. Prefer the smallest design that solves the essential capability clearly.
4. Keep extension points narrow and typed. Extend by composing modules, not by
   broadening the core.
5. Move optional behavior outside the core when it can be loaded, configured,
   or composed only when needed.
6. Challenge abstractions that cannot be explained through concrete data,
   operations, and module boundaries.
7. Review for avoidable complexity before optimizing for speed, novelty, or
   completeness.
8. Validate with tests, examples, and readable code paths that prove the
   simpler design is sufficient.

## Guardrail Checklist

Before making or approving a change, ask:

- Does this add functionality users need now, or serve only a possible future?
- Does this make the common path simpler or harder to understand?
- Can this behavior be a module, adapter, command, plugin, or optional
  integration instead of core logic?
- Can types, schemas, or tests enforce the boundary?
- Does a new abstraction remove real duplication or merely rename complexity?
- Is the implementation larger because it compensates for unclear design?
- Can one capable engineer understand the changed area without tribal
  knowledge?
- Did time pressure create an addition that should be a small redesign?

## Coding Guidance

Prefer:

- small modules with explicit imports, exports, ownership, and invariants;
- strong typing or clear runtime validation at module boundaries;
- straightforward data structures before framework-heavy machinery;
- iterative refinement: design, implement, inspect, simplify, then polish;
- tests that prove behavior without overconstraining implementation;
- publication-quality names, examples, and error paths.

Avoid:

- monoliths where every optional feature is permanently loaded;
- feature accumulation as a proxy for product quality;
- compatibility layers that preserve old concepts while duplicating them;
- broad hooks that expose internals or weaken invariants;
- dependencies that save a few lines but add a large conceptual surface;
- premature generality, hidden global state, and just-in-case configuration.

## Review Output

Lead with concrete risks:

- `Complexity`: unnecessary features, abstractions, dependencies, or states.
- `Core bloat`: logic that should be optional or modular.
- `Weak boundary`: unclear ownership, typing, validation, or API contracts.
- `Missed simplification`: a smaller design that preserves the goal.
- `Validation gap`: missing tests or examples for the intended behavior.

Then propose the smallest practical correction that preserves the user-visible
goal.
