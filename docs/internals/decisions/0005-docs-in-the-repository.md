---
title: "Decision 0005: The docs live in the repository"
description: Why all CyxWiz documentation is Markdown in docs/, changed with the code and rendered by the website.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - README.md
  - ../../README.md
---

**Status:** accepted (2026-11-24).

## Context

Documentation lived in several places (a website, usage guides, internal
notes) and drifted from the code. Users and contributors need one complete,
current reference, modelled on the Python and Java documentation.

## Decision

- `docs/` in the Engine repository is the only CyxWiz documentation. The
  Engine website renders it: stable from the latest release tag, next from
  the integration branch.
- Every page has checked front matter and one type (tutorial, how-to,
  reference, explanation, troubleshooting). Diagrams are Mermaid in the
  Markdown.
- A user-visible change updates its page and adds a release-note fragment
  in the same commit. Reference pages that can be generated from the code
  are generated.
- `docs/sync.yaml` records the commit each area's docs were last checked
  against.
- Internal design notes stay private; contributor explanations of public
  code go in `docs/internals/`.

## Consequences

- `python scripts/docs/check_docs.py` runs on every docs change.
- There is no copy of the docs per release; pages say `since` and note
  changes inline.

See [How the docs are organised](../../README.md).
