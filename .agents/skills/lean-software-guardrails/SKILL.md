---
name: lean-software-guardrails
description: Apply lean, modular engineering guardrails when architecture, source growth, dependency cost, ownership boundaries, C++ safety or performance, web correctness, or maintainability are material concerns. Use for substantial software changes, large-file refactoring, new abstractions or dependencies, and performance-sensitive C++ or web work.
---

# Lean Software Guardrails

## Core Stance

Keep software small, understandable, testable, and extensible. Treat every
feature, dependency, abstraction, layer, background service, generated
artifact, configuration knob, and new source file as a cost until it proves
essential.

Prefer cohesive modules with narrow typed boundaries. Do not solve growth by
continuing to append unrelated responsibilities to an already large file.

Read the references that match the task:

- Read [wirth-lean-software.md](references/wirth-lean-software.md) for
  architecture, major refactoring, framework choice, or feature expansion.
- Read [modularity.md](references/modularity.md) before adding substantial
  behavior to an existing file or changing module boundaries.
- Read [cpp-practices.md](references/cpp-practices.md) for C or C++ design,
  implementation, review, optimization, concurrency, or portability work.
- Read [web-practices.md](references/web-practices.md) for HTML, CSS,
  JavaScript, TypeScript, browser UI, or web-framework work.

## Operating Workflow

1. Inspect the existing source, callers, tests, ownership, and local
   conventions before designing or coding.
2. State the essential capability and the invariants that must remain true.
3. Separate required behavior from conveniences, compatibility burdens,
   visual embellishments, and speculative extension points.
4. Locate the smallest cohesive module that should own the behavior. If the
   apparent target is already large or has mixed responsibilities, evaluate
   extraction before adding more code.
5. Define data flow, ownership, errors, concurrency, and platform boundaries
   explicitly. Use types and tests to enforce them.
6. Reuse a sound existing primitive or service when it fits. Do not create a
   parallel implementation or compatibility layer without a migration need.
7. Implement the smallest complete vertical change. Keep optional behavior
   outside the core and load or activate it only when needed.
8. Inspect the resulting diff for duplication, file growth, hidden state,
   unnecessary allocation, dependency expansion, and weakened boundaries.
9. Validate behavior, failure paths, integration points, and relevant
   performance claims in proportion to risk.
10. Simplify names, interfaces, code paths, and documentation before declaring
    the work complete.

## Mandatory Guardrails

- Before adding substantial code to a large or mixed-responsibility file,
  state its current responsibility and choose explicitly whether to extend,
  extract, refactor the boundary first, or reject unnecessary behavior.
- Treat file and function size as investigation signals, not automatic quality
  verdicts. Generated code, declarative tables, and cohesive algorithms may be
  legitimately large.
- Do not mix UI rendering, domain logic, persistence, I/O, task orchestration,
  and platform integration in one component when narrow interfaces can
  separate them.
- Do not create generic `utils`, `helpers`, `common`, or manager modules as a
  dumping ground. Name modules after the capability or invariant they own.
- Do not duplicate an existing source of truth. Migrate consumers toward one
  canonical contract.
- Do not hide expensive copies, blocking I/O, background work, global state,
  or platform fallback behind innocent-looking APIs.
- Do not claim optimization without a baseline, a representative measurement,
  and correctness validation.
- Do not accept warnings, ignored errors, swallowed exceptions, unsafe casts,
  or disabled checks as normal completion conditions.
- Do not add a dependency or framework for behavior that a small local module
  can express clearly and safely.

## Design and Review Questions

Before making or approving a change, ask:

- What user or system capability is essential now?
- Which module owns the invariant, and which modules only consume it?
- Is the proposed file growing because it is cohesive, or because ownership is
  unclear?
- Can invalid states be prevented through types, schemas, or construction?
- Does the abstraction remove real duplication or merely rename complexity?
- Are lifetime, thread, cancellation, and failure behavior explicit?
- Can optional behavior remain outside the common path?
- Can one capable engineer understand and test the changed area without tribal
  knowledge?
- What should be removed, extracted, or simplified before completion?

## Validation Expectations

- Run the narrowest relevant formatter, compiler, static analysis, and tests.
- Add or update tests at the module boundary, not only inside implementation
  details.
- Cover success, invalid input, failure, cancellation, lifetime, and platform
  behavior when relevant.
- Measure memory, CPU, latency, allocation, bundle size, or compile-time effects
  only when the change makes a related claim.
- Record what was not validated and why; never convert an unverified assumption
  into a claim.

## Review Output

Lead with concrete risks using the applicable labels:

- `Complexity`: unnecessary states, abstractions, dependencies, or code paths.
- `Module growth`: a file, class, or function accumulating another
  responsibility.
- `Weak boundary`: unclear ownership, typing, validation, lifetime, or API
  contract.
- `C++ safety`: resource, lifetime, undefined-behavior, concurrency, ABI, or
  portability risk.
- `Web correctness`: accessibility, security, state, browser, API, or rendering
  risk.
- `Performance evidence`: an optimization claim without representative proof.
- `Missed simplification`: a smaller design that preserves the goal.
- `Validation gap`: missing tests, tools, examples, or platform coverage.

Then propose the smallest practical correction that preserves the user-visible
goal and identify any deliberate tradeoff.
