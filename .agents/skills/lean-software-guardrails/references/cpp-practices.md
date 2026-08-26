# Modern C++ Engineering Guardrails

Apply these practices in the context of the project's supported C++ standard,
toolchain, ABI, and established error model. Do not introduce a new library or
language feature merely to appear modern.

## Ownership and Lifetime

- Prefer RAII and the Rule of Zero. Make acquisition and release follow object
  lifetime.
- Prefer values for small owned objects, `std::unique_ptr` for exclusive dynamic
  ownership, and `std::shared_ptr` only for genuine shared lifetime.
- Treat raw pointers and references as non-owning and make nullable versus
  required parameters clear.
- Avoid manual `new`/`delete`, detached threads, owning raw pointers, and
  callbacks that outlive captured state.
- Use views such as `std::span` and `std::string_view` only when the referenced
  storage is guaranteed to outlive the view.
- Make destruction order and callback unregistration explicit for GUI,
  plugin, device, and asynchronous resources.

## Types and Interfaces

- Prefer the narrowest meaningful domain type over loosely related booleans,
  strings, integer tags, or parallel vectors.
- Use `enum class`, `std::optional`, and `std::variant` when they model real
  states supported by the project standard.
- Prevent invalid states in constructors or factories. Validate external data
  at the boundary once.
- Use `const` to communicate immutability and references for required
  non-owning inputs. Avoid returning references to temporary or unstable
  storage.
- Prefer value semantics and explicit move-only types over shared mutable
  ownership.
- Mark `noexcept` only when the complete operation satisfies the guarantee.
- Preserve one consistent project error contract: exceptions, status values,
  or result types. Do not silently mix models or swallow failures.

## Resource and Memory Discipline

- Avoid hidden full-data copies and materialization. Name expensive operations
  and document ownership transfer.
- Pass large immutable objects by reference or view; move owned results when
  appropriate.
- Reserve containers only when a reliable size estimate exists. Do not trade
  unbounded capacity for speculative speed.
- Prefer contiguous storage and simple data layouts on measured hot paths.
- Keep host/device transfers, synchronization, allocation, and fallback
  boundaries explicit.
- Do not cache without an owner, bound, invalidation rule, and measured need.

## Concurrency

- Give every task and thread an owner, cancellation path, completion path, and
  bounded shutdown behavior.
- Prefer the project's structured task system over ad hoc `std::thread`
  creation.
- Do not mutate GUI state from workers. Publish immutable or synchronized
  results to the owning thread.
- Protect invariants, not individual variables. An atomic flag does not make a
  compound state thread-safe.
- Avoid holding locks during blocking I/O, callbacks, rendering, or expensive
  computation.
- Define lock ordering when multiple locks are unavoidable.
- Make thread-safety part of the public contract and test cancellation and
  destruction races where practical.

## Correctness and Undefined Behavior

- Initialize state and use compiler warnings as defects to investigate.
- Check bounds, narrowing conversions, signed/unsigned interactions, overflow,
  iterator validity, alignment, aliasing, and object lifetime.
- Prefer explicit checked conversions. Avoid C-style casts and unjustified
  `reinterpret_cast` or `const_cast`.
- Avoid undefined behavior as an optimization technique.
- Treat null, empty, missing, invalid, and zero as distinct states when the
  domain distinguishes them.
- Preserve type, precision, null, and schema semantics across conversion
  boundaries.

## Headers, Templates, and Build Cost

- Keep headers self-contained and include only required declarations.
- Forward-declare when ownership and destruction rules allow it; do not obscure
  dependencies to save trivial compile time.
- Keep non-template implementation out of headers unless inline behavior is
  essential.
- Use templates for type-safe reusable algorithms, not to hide ordinary runtime
  polymorphism or create unreadable error surfaces.
- Keep macros local and minimal. Prefer constants, functions, templates, and
  scoped feature configuration.
- Respect ABI boundaries and isolate C APIs, third-party types, and
  platform-specific code behind focused adapters.

## Performance and Portability

- Establish a representative baseline before optimizing.
- Profile the actual workload and identify CPU, allocation, memory-bandwidth,
  I/O, synchronization, or device-transfer cost before changing design.
- Prefer algorithmic and data-layout improvements over clever syntax.
- Validate numerical parity and failure behavior after optimization.
- Use standard C++ for common behavior and narrow platform adapters for OS
  facilities. Test supported Windows, macOS, and Linux paths.
- Do not weaken correctness, clarity, or portability for an unmeasured gain.

## Validation

Use the tools already supported by the repository where applicable:

- formatter and compiler warnings;
- unit and integration tests;
- static analysis;
- AddressSanitizer and UndefinedBehaviorSanitizer;
- ThreadSanitizer for supported concurrency tests;
- representative benchmarks and memory profiles.

Report which configurations and platforms were actually validated.
