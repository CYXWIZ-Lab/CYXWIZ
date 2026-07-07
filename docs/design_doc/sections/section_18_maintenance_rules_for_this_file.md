## 18) Maintenance rules for this file

## 18.1 Scope discipline

This file is the canonical contract source for engine internals and should remain
- architecture-first, then contract-first, then behavior-first,
- ASCII-only diagrams/tables for portability,
- small and explicit deltas per edit.

## 18.2 Mandatory update set per design edit

Every time a design behavior changes:

1. Add or update at least one section that owns the behavior contract.
2. Add source anchor list under that section pointing to the concrete file path(s).
3. Update one of the phase registers (`section_19`..`section_28`) with:
   - status (`S/B/L` or `open`),
   - reason string,
   - release impact (`allowed`, `warn`, `blocked`).
4. Update section `0b` and 1: phase snapshot tables if scope changes.
5. Update `section_29` traceability when section boundaries move.

## 18.3 Versioning and traceability rule

- This index uses `Created` timestamp and phase snapshots only.
- For each behavioral change, add a small note near the modified subsection:
  - `Source:` file path,
  - `When:` date,
  - `Why:` concise intent.
- If a section is retracted/renamed:
  - add a short pointer section in the receiver section,
  - keep legacy section file only with explicit deprecation note, do not delete it without preserving links.

## 18.4 Evidence quality bar

Use this acceptance bar:

- at least one source file path for every claim,
- at least one example or contract in pseudo-ASCII,
- at least one explicit `allowed/warn/blocked` status for behavior changes.

## 18.5 Review gates before merge

- If any section adds or changes a node in `NodeType` contracts, run/record the corresponding node-registry and runtime-catalog evidence.
- If any section changes training lifecycle semantics, update:
  - graph compile section,
  - training execution section,
  - trace/error section,
  - one walkthrough example if launch behavior shifts.

## 18.6 Maintenance anti-patterns (do not do)

- Do not add generic “good to have” text without a source anchor.
- Do not mix editor implementation detail with runtime contract unless explicitly labeled.
- Do not convert ASCII diagrams to image-based references.
- Do not collapse multiple risk states into a single “done” note.
