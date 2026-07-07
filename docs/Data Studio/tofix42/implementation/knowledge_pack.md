# Knowledge Pack

## Purpose

Define the shipped retrieval data format for the CyxWiz assistant.

The assistant should not rescan the full codebase on every startup. It should
load a prepared, versioned knowledge pack optimized for local retrieval.

## Design Goal

The knowledge pack should be:

- local
- versioned
- fast to load
- easy to validate
- suitable for read-only retrieval
- independent from the live source tree at runtime

## What a Knowledge Pack Is

A knowledge pack is the retrieval-ready dataset the assistant backend uses at
query time.

It is not:

- the whole repo copied raw into the plugin
- a live filesystem crawler
- a model checkpoint

It is:

- indexed chunks
- retrieval metadata
- version and manifest data
- optional structured diagnostics metadata

## Why It Exists

Without a knowledge pack, the plugin would need to:

- scan the repo on startup
- re-chunk files repeatedly
- rebuild scoring data locally each time

That is slow, fragile, and hard to version against engine releases.

The knowledge pack turns source/docs/examples into a stable retrieval asset.

## Recommended Deployment Modes

### Curated shipped pack

Default product mode.

Use a prebuilt, versioned pack generated during release packaging.

### Developer override

Optional internal mode.

Allow a developer to point the backend at:

- a local repo checkout
- a locally rebuilt index

This is useful for engine development, but should not be the required default.

## Content Scope

The knowledge pack should include retrieval content that helps answer CyxWiz
questions accurately.

### Include

- engine source chunks
- docs chunks
- example graph chunks
- graph-node and operator descriptions
- structured diagnostic and trace schema excerpts
- selected test or validation excerpts only when they encode real behavior

### Exclude

- binaries
- generated build outputs
- temporary files
- unrelated tooling caches
- giant raw artifacts that do not improve answer quality

## Source Types

Each chunk should declare its source type.

Recommended source types:

- `source`
- `markdown`
- `cyxgraph`
- `cyxgraph_node`
- `trace_schema`
- `diagnostic`
- `command_help`
- `test`

This enables filtered retrieval and better ranking.

## Pack Structure

Suggested on-disk layout:

```text
knowledge_pack/
  manifest.json
  chunks.jsonl
  lexicon.json
  postings.json
  metadata.json
  diagnostics.json
```

The exact file breakdown may change, but the pack should keep:

- manifest/version data separate
- chunk content separate
- retrieval index structures separate

## Minimum Manifest Fields

`manifest.json` should contain at least:

```json
{
  "schema": "cyxwiz.assistant.knowledge_pack.v1",
  "engine_version": "0.0.0",
  "build_id": "example-build",
  "content_revision": "git-or-release-id",
  "created_at_utc": "2026-07-03T00:00:00Z",
  "chunk_count": 0,
  "source_file_count": 0,
  "source_types": ["source", "markdown"],
  "retrieval_backend": "lexical-v1"
}
```

## Chunk Shape

Each chunk should contain enough information for:

- retrieval
- citation rendering
- debugging
- pack validation

Recommended chunk shape:

```json
{
  "id": "chunk-000001",
  "source_type": "source",
  "path": "cyxwiz-engine/src/core/debug_trace_record.h",
  "line_start": 31,
  "line_end": 46,
  "title": "DebugTraceRecord",
  "text": "struct DebugTraceRecord { ... }",
  "tags": ["debugger", "trace", "core"],
  "symbol": "DebugTraceRecord",
  "content_hash": "sha256-or-similar"
}
```

Optional extra fields:

- `graph_name`
- `node_type`
- `diagnostic_code`
- `section_name`

## Retrieval Index Data

The pack should include prebuilt retrieval structures so query-time work stays
small.

At minimum for the current lexical approach:

- normalized token lexicon
- postings/inverted index
- document frequency data
- chunk metadata table

Do not require the plugin to rebuild these at load time unless a developer
override is explicitly requested.

## Pack Build Pipeline

Recommended build stages:

1. collect allowed input files
2. classify by source type
3. chunk by file/symbol/section/graph structure
4. normalize and tag chunks
5. build retrieval structures
6. write manifest and assets
7. validate counts, hashes, and version metadata

## Chunking Rules

The pack builder should prefer semantic chunking over arbitrary fixed windows.

Examples:

- C++ source: symbol or nearby block boundaries
- markdown: heading/section boundaries
- graph files: graph, node, or property blocks
- trace schemas: record or field definition blocks

Overlap is acceptable where needed to preserve boundary-local meaning.

## Versioning Rule

The knowledge pack must be tied to the engine build it describes.

Minimum rule:

- pack `engine_version` must match the engine release or approved version band

Better rule:

- include both release version and source/content revision

The backend should warn or reject when the pack is out of policy.

## Validation Rules

The backend should check on load:

- manifest schema is supported
- chunk store exists
- retrieval assets exist
- chunk count matches manifest
- engine version/build policy matches
- content hashes are structurally valid

If validation fails, the assistant should fail soft and not block engine use.

## Query-Time Behavior

At runtime, the backend should:

1. load the manifest
2. load retrieval structures
3. answer queries against chunks
4. return explicit citations

It should not:

- reparse the repo by default
- write new pack files implicitly
- mutate source content

## Shipping Rules

For production-like builds:

- ship the knowledge pack with the assistant package or installer
- keep it versioned with the engine release
- allow optional replacement only in developer mode

If the pack is missing:

- assistant features should be disabled or degraded
- engine features should continue normally

## Recommended First Pack Scope

For the first real product version, include:

- `cyxwiz-engine/src/core/**/*.{h,hpp,cpp,cc,cxx}`
- selected docs under `docs/`
- example graphs under `examples/cyxgraph/`
- selected structured diagnostics/traces metadata

That mirrors the retrieval value already proven in the tofix42 harness.

## Future Expansion

Later packs may add:

- command help metadata
- richer debugger schema metadata
- curated tutorial content
- structured operator capability summaries

Add these only when they improve answer quality materially.

## Recommendation

Use this model:

```text
Build a versioned, precomputed retrieval knowledge pack from approved CyxWiz
source/docs/examples. Ship that pack with the assistant. Treat live repo
indexing as a developer override, not the default runtime path.
```

That is the cleanest path from the current RAG proof to a shippable assistant
retrieval asset.
