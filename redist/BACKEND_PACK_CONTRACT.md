# CyxWiz Backend Pack Contract

## Status

Schema 1 is the frozen desktop release boundary for Ticket 88.
The implementation is dependency-free and lives in
`scripts/backend_pack_contract.py`. A schema change requires a new
`schema_version`; readers must fail closed on unknown versions or fields.

## Runtime Layout

CyxWiz owns one application-local runtime root selected by an app-level native
bootstrapper:

```text
cyxwiz-runtime-bootstrapper[.exe]     # app-level launcher
cyxwiz-installer[.exe]
cyxwiz-backend-pack-installer[.exe]
runtime/
  trust/
    trusted-keys.json
  catalogs/
    current.json
    manifests/
      <pack-id>.json
  base/
    <base-pack-id>/
  packs/
    cuda/<pack-id>/
    opencl/<pack-id>/
    oneapi/<pack-id>/
  staging/
    <transaction-id>/
  rollback/
    <runtime-set-id>/
  active-runtime.json
```

The stable bootstrapper is installed beside `runtime/` and retained inside the
signed versioned base as its verified publication source. Fresh base delivery
rechecks its signed size/hash and atomically publishes it before activating
`active-runtime.json`. It resolves the selected Engine from that state; a base
archive does not contain an independently launchable PATH-mutating script.
The bootstrapper resolves installer/repair executables from the active base so
the full GUI and helper dependency closure stays versioned instead of being
duplicated at the product root. The helper remains a separate process that
does not link the backend or ArrayFire runtime, so it can replace an inactive
pack without keeping any pack DLL loaded.

Windows and POSIX launchers enforce the same state and child-process contract.
Linux and macOS resolve the active base through the shared runtime validator,
replace inherited loader paths with only the active package directories,
remove ArrayFire, Python, and loader-injection overrides, and use exact
`fork`/`exec` argument vectors for Engine, Installer, and deferred Repair.
They do not invoke a shell or restore the legacy PATH-mutating launch script.

The graphical `cyxwiz-installer` component manager remains in the signed
versioned base and is resolved by the stable bootstrapper (`.exe` on Windows).
It owns Recommended, CPU-only, and Custom package consent and launches the
same-base signed delivery helper with an exact pack ID.
It uses a portable ImGui/GLFW shell and a narrow desktop adapter; it must not
link the CyxWiz compute backend or ArrayFire. Windows uses WinHTTP and
`CreateProcessW`; Linux and macOS use the same certificate-verifying HTTPS
client and an exact `fork`/`exec` helper invocation. All three preserve the
same signed catalog, immutable staging, isolated qualification, atomic
activation, and no-global-environment contract. Recommendation is deliberately
conservative: an OS with no trusted hardware classification defaults to the
CPU-only choice instead of guessing an accelerator pack.

A fresh setup may keep its app-bundled trust store and signed catalog under a
separate absolute metadata root while the customer selects an empty runtime
destination. The exact helper verifies that source, publishes the parsed trust
store and verified manifest files into the destination cache with the signed
catalog published last, and only then stages and qualifies the CPU base.
Maintenance reads metadata from the installed runtime cache. Current-user
installation is the least-privilege default; an all-users scope is an explicit
choice and requires platform authorization. Neither scope changes the global
loader environment.

Installer packaging accepts that app-bundled source through
`CYXWIZ_INSTALLER_BOOTSTRAP_METADATA_DIR`. Its contents are installed below
`runtime/` and must contain `trust/trusted-keys.json`,
`catalogs/current.json`, and one cached manifest for every catalog entry. A
staged installer without a verified CPU base and at least one optional pack is
not a production-capable component manager and must fail package verification.
Release jobs provide release-signed metadata; native CI uses ephemeral keys and
`packages.invalid` URLs only to exercise parsing, signatures, and UI discovery.

The CPU backend is part of the required base and cannot be represented as an
optional pack. A process resolves exactly one base and at most one pack per
optional backend. Activation replaces `active-runtime.json` atomically only
after all staged files, notices, manifests, and compatibility gates pass.

No normal workflow modifies the machine-wide `PATH`, `LD_LIBRARY_PATH`, or
equivalent. The launcher supplies a bounded child-only loader environment.
Hardware drivers and vendor providers remain host prerequisites.

`trusted-keys.json` is an app-bundled schema-1 document with exactly
`schema_version` and `keys`. Each key entry contains exactly `key_id`,
`algorithm=ed25519`, the 32-byte raw public key as 43-character unpadded
base64url, one or both unique roles (`catalog`, `pack`), and a boolean
`revoked`. Key IDs are unique, private keys are never present, and unknown
fields or roles fail closed. Application updates may revoke a bundled key;
catalog policy independently blocks or revokes individual packs.

## Signed Envelope

Pack manifests and catalogs use the same envelope:

```json
{
  "schema_version": 1,
  "kind": "cyxwiz-backend-pack-manifest",
  "signed": {},
  "signatures": [
    {
      "key_id": "release-2026",
      "algorithm": "ed25519",
      "value": "<unpadded-base64url-signature>"
    }
  ]
}
```

The signature input is the UTF-8 result of `canonical_json_bytes(signed)`.
Schema 1 permits no floating-point values, sorts object keys, removes
insignificant whitespace, and preserves array order. The signature and hash
implementations are release-tool concerns; runtime code consumes only a
reviewed verifier and never private signing keys.

HTTPS protects transport but is not package authenticity. A pack is trusted
only when its manifest signature chains to the app-bundled trust root, its
catalog entry is current and signed, and its archive and component hashes
match.

Schema 1 resolves a pack archive beside its signed manifest. For an online
entry, replace the final path segment of the catalog-authorized HTTPS manifest
URL with the manifest's signed `archive.file_name`; credentials, queries,
fragments, directory-valued manifest URLs, and nested archive names are
rejected. Offline media places that same archive file beside its copied signed
manifest. This deterministic rule is the only implicit artifact-source
mapping and introduces no unsigned mirror or redirect authority.

Online and offline artifacts enter one acquisition transaction. The service
writes only to a sibling `.part` file, resumes from its retained byte length,
requires the signed final byte size and SHA-256, flushes the completed file,
and atomically publishes it without replacing an existing destination. HTTPS
requests reject credentials and fragments, disable redirects, use bounded
timeouts, and require exact `Content-Length` plus exact `Content-Range` for a
resume. A changed remote object is rejected by the final signed hash.

ZIP extraction accepts only regular, non-link entries whose canonical UTF-8
paths, sizes, and case-folded uniqueness exactly match the signed component
inventory. Directory, traversal, drive/ADS, sparse, hard-link, symbolic-link,
duplicate, missing, and unexpected entries fail closed. Extracted bytes and
free-space/caller budgets are bounded before activation, every component is
rehashed, and any failed or cancelled extraction removes its private staging
directory.

## Pack Manifest

The signed pack body contains exactly:

- `pack_id`, `pack_kind`, `backend`, and `package_version`;
- `platform`, `architecture`, and `runtime_set_id`;
- compatible CyxWiz release minimum/maximum;
- exact ArrayFire version and ABI identity;
- companion base ID and conflicting pack IDs;
- compatibility requirements and release support state;
- every component path, byte size, SHA-256, source, and executable flag;
- license/notice entries that reference packaged component paths;
- archive file name, byte size, and SHA-256;
- generation time in UTC.

`pack_kind=base` requires `backend=cpu` and a null companion base. Every
`backend_pack` requires a distinct companion base. Component paths are
canonical relative POSIX paths. Absolute paths, drive prefixes, backslashes,
`.`/`..`, and case-insensitive duplicates are rejected before extraction.
Installer-supplied scripts are not an extension mechanism and unknown fields
fail closed.

The compatibility object contains:

- required device kinds and CPU features;
- provider types;
- minimum and tested driver ranges;
- minimum physical-identity confidence required for cross-backend matching;
- eligible recommendation-target backends (never automatic substitution);
- Ticket 91 operation-matrix ID;
- training-certification scope;
- `supported`, `diagnostic`, `blocked`, or `revoked` support status.

Physical identity and confidence are measured locally by Ticket 89/91. They
are not copied from a catalog as authorization. A catalog can constrain
device kind, provider, driver, runtime, and matrix requirements; the local
qualification cache proves the exact machine route.

## Catalog

The signed catalog body contains exactly:

- catalog ID, creation time, and expiry time;
- minimum supported CyxWiz client version;
- pack ID, HTTPS manifest URL, manifest SHA-256, signing-key ID, and support
  status for each published pack.

Duplicate pack IDs, insecure URLs, unknown states, expired catalogs, revoked
keys, and unsupported client versions fail closed. Offline installation uses
the same signed catalog and manifests copied onto trusted media; it does not
introduce an unsigned metadata format.

The application-local catalog cache has one deterministic read boundary:
`catalogs/current.json` is the complete current signed catalog envelope and
`catalogs/manifests/<pack-id>.json` is the cached signed manifest authorized by
that catalog entry. Metadata publication replaces complete files atomically;
the runtime never follows a mutable unsigned pointer or derives a manifest path
from its HTTPS host. A valid catalog remains browsable when an individual
manifest is absent or invalid, but delivery for that entry stays disabled.

## Enterprise, Offline, and Proxy Policy

Runtime inspection is local-only. `show backend packs`,
`show backend compatibility`, and `show backend support-bundle [1-100]` read
the active runtime, local signed catalog cache, retained device inventory, and
qualification snapshot without opening a network connection or starting a
probe. The bounded support output is shareable text: it omits filesystem
paths, catalog URLs, proxy values, credentials, tokens, and internal ticket
keys, and it never uploads automatically.

An enterprise administrator can install without network access by placing the
unchanged signed catalog at `catalogs/current.json`, each unchanged signed
manifest at `catalogs/manifests/<pack-id>.json`, and the signed archive beside
its manifest. The exact packaged helper is then invoked with an absolute
runtime root, one catalog-authorized pack ID, and `--offline`. Offline mode
uses the same signature, release-policy, byte-size, SHA-256, extraction,
qualification, and atomic-activation checks as HTTPS delivery. It does not
accept an unsigned catalog, a rewritten URL, or a local trust override.

Online acquisition begins only after explicit installer consent. On Windows,
WinHTTP uses the operating system's automatic proxy configuration. On Linux
and macOS, schema 1 uses direct certificate-verified HTTPS and has no explicit
proxy-credential input; proxy-only deployments must use the offline workflow.
Proxy URLs and credentials are never catalog or manifest fields and must not
be copied into support output. Redirects remain disabled on every platform.
Failure to reach the network leaves the verified local catalog and installed
runtime inspectable and does not downgrade to unsigned metadata.

## Active Runtime

`active-runtime.json` is local transaction state, not signed publication
metadata. It records schema version, runtime-set ID, monotonic generation,
base-pack ID, and at most one pack ID for each optional backend. It never
stores mutable device ordinals or qualification verdicts.

Pack activation supplies pack/runtime identity to Ticket 91 and invalidates
only affected route evidence. Ticket 91 remains the sole owner of isolated
verification, route diagnostics, training authorization, and selected-route
configuration.

The non-GUI lifecycle service is the composition boundary for catalog read,
exact user-selected pack delivery, extraction, staging, qualification policy,
activation, rollback, repair, and removal. It exposes the verified catalog to
hardware-aware recommendation consumers but never silently selects a pack.
Before qualification it constructs the exact prospective runtime identity and
passes that identity to the shared qualification adapter. It activates only a
`supported` pack with a qualified result, and only if the active runtime state
is unchanged when qualification completes. Diagnostic, missing-adapter,
failed, cancelled, and stale-evidence results leave a complete pack inactive.

## Customer Verification Summary

The consolidated installer reads the shared machine-local qualification
snapshot; it does not run a second probe registry. It presents route outcomes
from typed status and count fields and never renders internal matrix IDs,
evidence IDs, benchmark IDs, ticket names, or raw probe messages as customer
text. Crashes, timeouts, failed operations, unavailable operations, stale
evidence, and incomplete evidence remain distinct results with bounded next
actions.

A route may be labeled `Best measured` only when at least two active routes
passed the complete operation contract and contain finite positive samples
from the same fixed CyxWiz performance benchmark. One measured route is shown
as evidence without a comparative claim. Failed, inactive, unmatched, stale,
or differently benchmarked routes are never performance recommendations.

## Repair and Removal

Versioned pack directories are immutable while active. Repair first verifies
and stages the complete signed payload, deactivates the affected route,
rewrites any rollback reference to a complete retained runtime, quarantines
the corrupt directory, publishes the repaired directory, and only then may
reactivate it. It never overwrites an active directory in place.

The Engine queues Repair against the exact active backend and pack identity.
After that Engine process exits, the minimal bootstrapper validates the queued
identity and launches the platform's `cyxwiz-backend-pack-installer` executable.
The helper reuses
the signed catalog, manifest, extraction, lifecycle, and qualification
contracts. Qualification runs in the isolated route-probe child process; the
helper activates the repaired pack only when that exact candidate passes.
Failure leaves the complete pack inactive and retains the queued request for a
later retry.

Removal accepts optional backend packs only. It holds the shared runtime
mutation lease, deactivates an exact active pack, removes any rollback
reference to that pack, and atomically moves the directory into a private
per-pack quarantine before deletion. A process interruption may leave only an
inactive quarantine; repeating removal validates active and rollback state
before finishing that cleanup. The CPU base, active packs, rollback-protected
packs, links, and paths resolving outside the runtime root are never deleted.

## Trust and Rotation Policy

- Production pack and catalog signatures use Ed25519 detached signatures.
- Private keys remain outside source, build trees, packages, and application
  installations.
- The application ships a small versioned public-key trust root.
- A catalog may reference only an already trusted signing key.
- Key rotation overlaps old and new public keys for one release window; the
  new key is trusted before it becomes the sole signer.
- Pack revocation is distributed in a catalog signed by a still-trusted
  catalog key. Signing-key revocation is delivered only through a trusted
  application update that replaces the app-bundled trust root.
- Offline media includes the current app-bundled trust root and signed catalog
  needed to evaluate key/pack revocation and expiry at publication time.
- Downgrades require explicit policy and cannot select a revoked pack, key,
  runtime set, or unsupported CyxWiz release.
- Clock or expiry uncertainty fails closed for connected updates and requires
  an explicit offline-administrator workflow.

## Redistribution Policy

Every component records its source and has a corresponding packaged notice.
Release engineering must review redistribution rights for the exact versions
being shipped. The base may include CyxWiz, Python 3.12, ArrayFire unified/CPU,
MKL/TBB, and required VC++ application runtimes. Optional packs may include
legally redistributable ArrayFire plugins and their user-mode runtime closure.
Hardware-vendor display, kernel, or compute drivers are never bundled.

Missing or incomplete notices fail packaging. A new runtime component cannot
be accepted by adding an undocumented wildcard; its source, closure, notice,
and compatibility requirements must be reviewed and represented explicitly.

## Validation Ownership

`backend_pack_contract.py` validates schema shape and semantic invariants
without executing package code. Native services own archive extraction,
signature verification, transaction staging, rollback, repair, and removal.
Runtime probing remains owned by Ticket 91. All consumers must use these
contracts rather than widen them inside GUI code.
