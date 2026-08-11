# Source-Derived Lean Software Guidance

This reference condenses Niklaus Wirth's 1995 article "A Plea for Lean
Software."

## Central Argument

Software tends to grow because hardware growth makes waste affordable. The
result is often slower, larger, less understandable software whose extra
functionality does not match essential user value. Correct this with
disciplined methodology and a return to essentials.

Do not let available compute, modern frameworks, large memory, or package
ecosystems justify avoidable complexity.

## Causes of Software Bloat

- Teams add features users do not need because quantity is easier to market
  than quality.
- Incompatibility with the original system concept produces cumbersome
  behavior.
- Monolithic design forces every possible feature into one system.
- Visual polish and convenience features can hide real cost.
- Systems that require tribal knowledge create avoidable dependence.
- Time pressure encourages additions and corrections instead of redesign.
- Larger teams add communication cost and can weaken design coherence.

## Design Principles

### Concentrate on Essentials

Identify the essential model and workflow. Remove or defer anything that does
not contribute directly to power, flexibility, or clear user convenience.

### Use Strong Language and Type Support

Prefer designs where invalid states and incompatible operations are caught
early. Strong typing, explicit interfaces, and well-defined boundaries reduce
risk during change.

### Decompose into Coherent Modules

Invest design effort in decomposition. Give each module a precise interface,
clear imports and exports, and a responsibility understandable in isolation.

### Extend Without Broadening the Core

Let new modules and types integrate through narrow boundaries. Keep the core
small and activate optional capabilities only when demanded.

### Prefer Primitives Over Proliferation

Choose a small set of flexible primitives. Avoid many similar operations,
options, modes, and special cases when composition is sufficient.

### Simplify Through Iteration

Design, implement, inspect, and refine. After implementation, make the result
smaller, clearer, and more coherent before calling it complete.

## Modern Engineering Lessons

1. Use type systems, static checks, schemas, and tests to expose mistakes
   before runtime.
2. Avoid duplicated logic and unclear ownership.
3. Preserve compatibility through narrow modules rather than exposed
   internals.
4. Prefer carefully chosen primitives to many derivative APIs.
5. Keep systems understandable by an individual engineer or small team.
6. Treat high communication overhead as a possible design smell.
7. Make complexity and size reduction explicit engineering goals.
8. Improve design through direct implementation and iterative review.
9. Polish code until it is clear, maintainable, and fit for publication.

## Practical Review Prompts

- What is the smallest essential capability?
- Which paths exist only because features accumulated?
- Which module boundary would simplify ownership and reasoning?
- Which types, schemas, or tests can replace convention or comments?
- Can optional behavior move behind a module, plugin, or lazy integration?
- What would be removed if CPU, memory, dependencies, and attention were
  scarce?
- Does the design teach a clear concept or depend on hidden behavior?
- What should be simplified before completion?
