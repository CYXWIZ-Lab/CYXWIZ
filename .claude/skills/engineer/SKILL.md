---
name: engineer
description: Professional coding agent for CyxWiz. Before implementing any new feature or non-trivial task, this skill enforces a design-first workflow — understand the codebase, identify existing abstractions, apply software-design principles (SOLID, DRY, single-responsibility), and plan for long-term maintainability instead of feature-cargo-culting. Invoke with /engineer for any new feature, refactor, or multi-file change.
---

# Engineer Skill — design-first coding for CyxWiz

## Purpose

Stop repeating the DataInputDialog::Apply disaster. For the full day of
2026-04-17 we fought a 700-line category switch that grew by copy-pasting
each new data type's async block. Every fix introduced subtle drift.
Unifying it into a polymorphic `DataLoader` hierarchy took 8 commits
across multiple parallel sessions and would have been 15 minutes at the
time the second async branch was added — IF someone had paused.

This skill is the pause. It runs BEFORE you write code for any new
feature or non-trivial change. It produces a short design note that you
then execute against, instead of diving into the first file that looks
relevant.

## When to invoke

MANDATORY before:
- Any new feature (new node, new loader, new UI panel, new dialog)
- Any file that will grow past 200 lines
- Any change that will touch 3+ files
- Any place the user's task description contains "add another" or "same
  as X but for Y"
- Any commit that would duplicate an existing pattern for a new type

OPTIONAL but recommended for:
- Bug fixes that span multiple modules (drift is likely)
- Anywhere you find yourself copy-pasting a block and changing 20% of it

SKIP for:
- Trivial typo / comment / formatting changes
- Single-file, single-function changes under 20 lines
- Build-wrapper / test-only changes

## The process (do these in order, no skipping)

### Step 1 — Understand (read before you write)

Before writing ANY implementation code:

1. Find the closest existing analogue. If you're adding an Audio path,
   read the Text path end-to-end first. If you're adding a chart node,
   read the full BarChart files. Do not skim — read.
2. Grep for every place the existing analogue appears. Duplicated
   switches, hand-maintained lists, per-type `if` chains are signals the
   abstraction is weak or missing. Catalog them.
3. Read the relevant tofix entries in `docs/Data Studio/tofix.md`. If
   the area you're about to touch is already flagged as architecturally
   broken, that changes the plan.
4. Read the plans folder `docs/plans/`. Big refactors may already be
   designed and waiting for a session — don't freelance around them.

Output of this step: a 3-6 line summary of what's there already, what
pattern it follows, and what's likely to drift if you just copy-paste.

### Step 2 — Apply the principles (structured, not reflexive)

Run through this checklist before picking an approach. Every "yes"
means the naive copy-paste plan is wrong:

- Is there a **switch on type** (NodeType / file_category / backend tag)
  in 3+ places for the same concern? → Polymorphism wanted. Extract an
  interface, not another switch case.
- Are you **duplicating a block** from a sibling type (Text copied to
  Image, etc.)? → The duplicate 80% wants to be a helper or base class;
  extract that, then put only the 20% difference in the concrete
  subclass.
- Are you **adding a field to a god-struct** that already has per-type
  special fields? → The struct wants to become a variant / hierarchy.
- Is the caller-side dispatch a **hand-maintained list** (switch / `if`
  chain / whitelist)? → The callee-side should own the decision via a
  virtual method or a registered factory.
- Is a change going to require **edits in N unrelated files** (registry,
  UI, loader, dispatch)? → A single indirection is missing; find where
  to put it so the N files reduce to 1.
- Is there a **stateful global** being read from multiple places? →
  Consider injecting or scoping. At minimum, write the invariant.
- Is any concern **crossing layer boundaries** (UI code reading registry
  internals, training code reaching into dialog state)? → Pull the
  concern to the right layer; don't plumb it sideways.
- Is there a **feature flag or hand-kept whitelist**? → Derive it from
  the actual registered state. We had 4 such lists in this codebase,
  all of which drifted.

Output of this step: the principles violated by the naive plan, and the
principle-driven alternative.

### Step 3 — Plan (write it down before coding)

Write a short plan — either inline in chat, as a file under
`docs/plans/`, or as a comment block in the target file. It must state:

- **The abstraction**: what new class / interface / helper you're
  introducing, or what existing one you're extending. Name it.
- **The migration order**: if this is more than one commit, sequence the
  commits so each one builds and runs.
- **The drift prevention**: how does the design make the N-places
  problem disappear? "Adding a new X requires editing Y places" — what
  is Y, and is it less than before?
- **The rollback boundary**: if the refactor has to be reverted mid-way,
  what's the last commit that leaves the tree in a good state?

For refactors > ~3 commits, write the plan into `docs/plans/<name>.md`
following the pattern of existing plans there.

### Step 4 — Implement against the plan (not around it)

- Land each commit so it compiles and runs. Use `scripts/rebuild.sh` or
  `scripts/rebuild.cmd` — never raw `cmake --build ... | tail` (the
  pipeline-exit-code trap cost us 1 hour of zombie-binary debugging
  2026-04-17).
- After each commit, verify the binary mtime advanced (the rebuild
  wrappers do this automatically — trust the warning they print).
- If you deviate from the plan mid-stream, update the plan OR revert to
  the last good commit and revise. Do not let the plan and the code
  diverge silently.

### Step 5 — Reflect (prevent this from happening again)

After landing, update:

- **tofix.md** if you closed an entry or surfaced a new one (new
  concerns to flag for future sessions).
- **CLAUDE.md** if you changed an invariant or registered a new
  "single point of truth" (future sessions need to know the shape).
- **docs/plans/** if a multi-commit plan was completed — mark
  ~~LANDED~~ with commit pointers.
- **Auto-memory** (`memory/MEMORY.md`) if the decision is load-bearing
  for future sessions (e.g., architectural choices, user preferences).

## Anti-patterns we've actually hit (specific to this codebase)

Learn these by name. When you recognize the shape, you know you're
about to make the same mistake:

### 1. The 700-line category switch

`DataInputDialog::Apply` grew to 700 lines with 4 near-identical async
blocks (tabular, text, image, audio). Each new data type copy-pasted
the previous one and drifted on subtle per-category fields. Fix: a
polymorphic `DataLoader` hierarchy with concrete loaders per category.
If you see yourself adding a 5th if-else branch to this kind of
structure, STOP and extract.

### 2. Three hand-maintained lists

Adding BarChart required editing `NodeMetadataRegistry`,
`BuildSearchableNodes`, AND `ShowCategorizedNodeMenu`. Missing one of
the three made the node invisible in one UI. Fix: single source of
truth (the registry); the other consumers iterate it. Rule: if adding a
thing to the system requires edits in N unrelated files, one of those
N is the missing abstraction.

### 3. The pipeline-exit-code trap

`cmake --build ... | tail -3` returns tail's exit code (always 0), so
a real C2440 compile error reported "success" and the stale `.exe` kept
running. We shipped a zombie binary for an hour. Fix: use
`scripts/rebuild.sh` / `scripts/rebuild.cmd`. Rule: never pipe a
verification command through tail/head and trust the exit code.

### 4. The whitelist that disagrees with the factory

`ShouldShowOpenDialogButton` had 19 NodeTypes; the factory had 9
dialogs registered. 10 types showed a button that did silently nothing.
Fix: derive from `factory.HasDialog(type)`. Rule: if an authority
already exists (factory, registry, enum), don't maintain a parallel
list — ask the authority.

### 5. The config-extractor masquerading as an operator

`TextTokenizer` etc. were built as nodes with pins but no data flow —
the compiler scans them for config then the real work runs inside
TextDatasetBatcher. The pins visually lie. Fix B in tofix tracks this.
Rule: if a node has pins, data must actually flow through them; if it's
a config knob, it belongs in Properties or a dialog, not the graph.

### 6. Registries plural

5 parallel maps in `DataRegistry` (arrow_datasets_, parquet_datasets_,
image_datasets_, audio_datasets_, text_datasets_) each with their own
`Register/Unregister/Get/Is/Clear` methods. Every downstream site has
to try all 5 to find the owner. Fix (planned): routing sidecar
`name_to_category_` so `ResolveCategory(name)` is a single lookup.
Rule: parallel same-shape maps keyed differently are a missing
`std::variant` or a missing indirection.

## Output format

When invoked, produce (before writing code):

```
## Engineer: <task description>

### Understand
<3-6 lines on existing analogues, patterns, drift signals>

### Principles at risk
- <principle>: <how the naive approach violates it>
- ...

### Plan
1. <step 1 — what abstraction, what commit>
2. <step 2 — next commit>
...

### Drift prevention
<How does this design make the "add a new X" story single-place?>

### Rollback boundary
<Last safe commit if mid-refactor revert is needed>
```

Only after this is written and (optionally) approved by the user should
implementation begin.

## Rules

- Never write feature code without going through steps 1-3 first, even
  if the task seems small. Small tasks that skip design grow into the
  700-line switch.
- If the principles checklist (step 2) yields >1 violation, pause and
  explicitly choose to either (a) do the refactor first, (b) write a
  tofix entry for the debt, or (c) accept the violation with a
  justification comment. Never silently add another drift point.
- If the user says "just add X quickly", push back once with the
  engineering concern. If they re-confirm the quick-add, add a tofix
  entry describing the debt you took on.
- Prefer extending an existing abstraction over introducing a new one.
  Premature abstraction is also bad — but "extract after 3 duplicates"
  is the right threshold for this codebase.
- The plan does not have to be long. Four lines of abstraction + four
  lines of migration is enough for most tasks. Long plans are for 8+
  commit refactors; those live under `docs/plans/`.

## Verdicts

- **PROCEED** — understand step done, principles checked, plan written.
  Implementation can start.
- **REFACTOR FIRST** — the naive implementation would compound
  existing drift. Do the refactor as a separate commit sequence
  before the feature work.
- **DEBT-ACCEPTED** — user explicitly chose the quick path; tofix
  entry filed; proceed to implementation.

After implementation, run `/audit` and `/commit` as usual.
