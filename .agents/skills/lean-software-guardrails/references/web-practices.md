# Web Engineering Guardrails

Apply these practices to HTML, CSS, JavaScript, TypeScript, browser UI, and
component frameworks. Follow the repository's framework, package manager,
browser support, formatting, linting, and test conventions before adding new
tools.

## Structure and State

- Build small cohesive components around user-visible responsibilities, not
  arbitrary visual fragments.
- Keep domain logic, data access, state transitions, and rendering separable
  and independently testable.
- Keep state local until multiple consumers genuinely need shared ownership.
- Derive values instead of synchronizing duplicate state.
- Model loading, empty, success, stale, cancelled, and failure states
  explicitly.
- Prefer composition and plain functions over deep component inheritance or
  framework-specific base classes.
- Split large components by responsibility and data flow, not by line count
  alone.

## TypeScript and JavaScript

- Prefer TypeScript with strict checking for maintained application code when
  the project supports it.
- Avoid `any`, unchecked casts, non-null assertions, and stringly typed state;
  narrow `unknown` data at trust boundaries.
- Validate network, storage, URL, worker, and plugin data at runtime even when
  compile-time types exist.
- Keep functions pure when practical and make side effects explicit.
- Use immutable updates where framework change detection depends on identity.
- Handle promise rejection, cancellation, component disposal, and stale
  responses. Do not let older requests overwrite newer state.
- Avoid adding a state-management or utility dependency for behavior the
  language or current framework already expresses clearly.

## HTML, Accessibility, and Interaction

- Use semantic HTML before adding ARIA. Preserve native keyboard and form
  behavior.
- Provide accessible names, labels, focus order, visible focus, keyboard
  operation, and meaningful status/error announcements.
- Do not encode meaning through color alone. Maintain usable contrast and
  scalable text.
- Keep interaction available across mouse, keyboard, touch, and assistive
  technology where applicable.
- Preserve user input on recoverable errors and explain corrective action.

## CSS and Responsive Layout

- Prefer scoped styles, documented design tokens, and predictable layout
  primitives over growing global stylesheets.
- Avoid highly specific selectors, pervasive `!important`, and duplicated
  magic values.
- Design for content growth, localization, zoom, reduced motion, dark/light
  themes, and narrow viewports when the product supports them.
- Use CSS for layout and presentation; do not move stable styling decisions
  into repeated JavaScript calculations.
- Remove unused rules and components instead of leaving permanent legacy
  layers.

## APIs, Security, and Privacy

- Treat all external and persisted data as untrusted. Validate and encode at
  the boundary appropriate to its destination.
- Avoid unsafe HTML injection. Sanitize only with a maintained, proven policy
  when rich HTML is required.
- Never place secrets in client bundles, source maps, logs, URLs, or browser
  storage.
- Apply authentication and authorization on the server; client checks are user
  experience, not security boundaries.
- Use safe cookie, CSRF, CORS, content-security, and transport settings where
  the architecture requires them.
- Minimize collection and retention of personal or sensitive data.
- Avoid leaking internal errors, paths, tokens, or user data into telemetry.

## Performance

- Measure real user flows before optimizing and establish budgets for bundle
  size, initial render, interaction latency, and memory where relevant.
- Avoid unnecessary rerenders, duplicate requests, unbounded lists, retained
  subscriptions, and event-listener leaks.
- Page, stream, aggregate, or virtualize large data rather than materializing
  it into the DOM.
- Lazy-load optional routes and heavy features when it improves measured
  startup behavior.
- Optimize images, fonts, and network caching without sacrificing correctness
  or accessibility.
- Do not add memoization, virtualization, code splitting, or caching without a
  demonstrated workload and invalidation plan.

## Browser and Lifecycle Correctness

- Use standards-based APIs and isolate browser or platform variations behind
  small adapters.
- Clean up timers, observers, event listeners, workers, subscriptions, object
  URLs, and in-flight requests.
- Define behavior for offline, slow, retried, duplicated, and out-of-order
  requests where applicable.
- Preserve progressive enhancement or a clear unsupported-browser message for
  the project's browser policy.

## Validation

Use the repository's existing tools where applicable:

- formatter, linter, and strict type checking;
- unit tests for domain logic and state transitions;
- component tests for interaction and accessibility;
- integration tests for API boundaries and failure paths;
- end-to-end tests for essential user journeys;
- accessibility checks plus keyboard/manual verification;
- bundle analysis and browser performance profiling for performance claims;
- supported-browser and responsive-layout checks.

Prefer a small number of meaningful tests over snapshots that record incidental
markup or styling.
