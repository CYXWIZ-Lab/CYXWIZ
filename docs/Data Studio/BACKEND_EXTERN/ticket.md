# BackendExtern Proposed Ticket Breakdown

## Ticket BE-0: Design acceptance

Confirm this package's boundaries, LTS policy, legal ownership, platform
starting point, and first commercial model/provider. Required before coding.

## Ticket BE-1: Runtime contract types and fake worker

Add Engine-owned protocol v1 types, manifest validation, run state machine,
fake worker harness, and unit tests. No Python/JAX/PyTorch dependency.

Acceptance: native tests stay green; unsupported/malformed worker behavior
fails closed; cancellation and Engine shutdown are deterministic.

## Ticket BE-2: Worker manager and local IPC

Implement managed worker process startup, authenticated local endpoint or
inherited private channel, request/result/event validation, timeout, stop, and
support-bundle diagnostic records.

Acceptance: worker crash, timeout, malformed reply, and Engine close are
tested; no orphan worker remains.

## Ticket BE-3: Runtime catalog and installer

Implement signed/catalogued runtime metadata, lockfile hash verification,
project runtime pinning, health probes, and uninstall/cache controls.

Acceptance: exact runtime is reproducible; a missing or incompatible runtime
cannot be silently upgraded.

## Ticket BE-4: External graph contracts

Add node metadata/pins, compiler and materializer preflight, run-report
artifact, and missing-provider compatibility behavior.

Acceptance: legacy/native graph execution remains unchanged; saved external
nodes remain inspectable and fail closed without their provider.

## Ticket BE-5: First approved provider

Implement one inference-only provider, CPU reference tests, Data Studio node,
model provenance, and commercial-license catalog gating.

Acceptance: a supported table-to-prediction path is reproducible and its
result states are visible in Studio.

## Ticket BE-6: GPU validation

Add one explicitly pinned worker framework/device pair with diagnostics,
compatibility checks, and failure containment.

Acceptance: CPU fallback/disable behavior is truthful; failure cannot damage
the native ArrayFire session.

## Ticket BE-7: Provider SDK review

Decide whether two shipped providers justify a public provider SDK. If not,
continue with official providers only.

