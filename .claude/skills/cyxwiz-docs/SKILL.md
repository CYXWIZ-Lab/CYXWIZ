---
name: cyxwiz-docs
description: Write, update, move or review CyxWiz Engine documentation in docs/ (the source of the Engine website). Use whenever a change affects anything a user, plugin developer, operator or contributor can see (panel, menu, node, preference, file format, SDK or Python API, server node, CyxCloud), when the user asks for docs, a guide, a tutorial or release notes, when moving or deleting doc pages, and when merging a branch that touched docs/. Covers the docs tree, page types and templates, required front matter, same-change and merge rules, deprecation, what must never be published, and the check to run.
---

# CyxWiz documentation

The docs in `docs/` are the only CyxWiz Engine documentation. The Engine
website renders them (stable from the release tag, next from the integration
branch). Read `docs/README.md` first: it holds the tree, the front matter and
the workflow. This skill is the short form for agents.

The goal is Python- or Java-documentation completeness: a reader can install
the Engine, learn it, do any task, look up any node or setting, extend it,
and understand how it works inside, without reading the code.

## Before writing

1. **Search first.** `grep -ril "<topic>" docs/` for an existing page. Update
   it rather than writing a second one. Old pages may still be in `usage/`,
   `extending/`, `architecture/`; until a page has moved, update it there.
2. **Write from the code and a running Engine, never from memory.** Check
   every label, menu path, node name, default and limit against the source
   (`cyxwiz-engine/src`, `cyxwiz-backend`, `plugins/`) or the screen.
3. **Pick one type** and start from its template in
   `docs/internals/docs/templates/`: tutorial, how-to, reference,
   explanation, troubleshooting. Do not mix types on one page; link instead.
4. **Pick the folder** from the tree in `docs/README.md`. Do not create a
   folder for a single future page.

## Writing

5. **Front matter is complete and true.** title, description, type, area,
   status, since, owner, last_reviewed are required. `since` is the version
   that ships the behaviour; `status` matches reality; set `last_reviewed` to
   today only after checking the page against the code.
6. **Write for the reader.** Second person, active voice, present tense,
   short sentences, the key point first. Sentence-case headings under 70
   characters; never "Overview" or "Introduction" as a title.
7. **Procedures.** Numbered steps, one action each: location first, then the
   action, then the result. UI labels in **bold** exactly as shown; menu
   paths as **File > Open Project**. Mark optional steps "Optional:". No
   "simply", "just", "easy", "please", "above/below".
8. **Code.** Fenced blocks always name the language. File names, paths,
   parameters and values in `code` font. Every code block runs as shown;
   prefer snippets taken from committed, tested files in `examples/`.
9. **Real examples.** Use committed example projects and datasets, never
   foo/bar or personal paths. Say where the data comes from or how to get it.
10. **Diagrams as code.** Mermaid inside the Markdown, never exported images
    of diagrams.
11. **Screenshots only when the UI is the point**: default theme, cropped,
    alt text, the Engine version in the file name, under the area's
    `images/` folder. Retake them in the same change when the UI changes.
12. **Tell the truth.** Document what the build does now, including limits
    and unsupported devices. Never announce plans. A bug is not behaviour:
    add it to `docs/known-issues.md` and report it.
13. **Keep pages under about 1000 words.** Split by task, not by length.

## Never publish

14. No internal ticket numbers (TOFIX...), branch names, local paths
    (`D:\...`), machine names, IP addresses, ssh aliases, private repository
    details, secrets or personal data.
15. Nothing about dropped features: crypto, tokens, wallets, staking,
    Solana, the model marketplace or model publishing, Hugging Face.
16. Internal design notes stay in the private engineering notes; contributor
    explanations of public code go in `docs/internals/`.

## Same change, branches and merges

17. **Docs change with the code.** A user-visible change updates its page and
    adds a fragment in `docs/release-notes/next/` (template
    `release-note-fragment.md`) in the same commit. If no docs are needed,
    say why in the commit or ticket.
18. **On a feature branch**, new pages carry `status: experimental` and the
    planned `since`. **When merging**, resolve docs conflicts in the same
    merge: one page per topic (no duplicates from two branches), the newer
    true behaviour wins, links and the index updated, then run the check on
    the merge result.
19. **Generated pages** (`generated: true`: nodes, preferences, menus,
    shortcuts, SDK and Python API) are never edited by hand. Change the code
    or the generator under `scripts/docs/`.

20a. **Keep the sync marker true.** `docs/sync.yaml` records, per area, the commit
    its docs were last checked against (`synced_at`). Run
    `python scripts/docs/docs_drift.py --files` to see what changed since; after
    updating an area's pages against the code, set its `synced_at` to the commit
    you checked (on the integration branch; on a feature branch, at merge).
    Never bump it without checking.

## Moving and removing

20. **Move with a redirect.** Add the old path to `redirect_from`, fix every
    inbound link (`grep -rn "<old name>" docs/ cyxwiz-engine/src`), and keep
    paths the Engine's Help menu or tests link to.
21. **Deprecate, then remove.** `status: deprecated`, `until`,
    `replaced_by` and a warning at the top; on removal, a stub pointing to
    the replacement, deleted one release later. Never leave an orphan page.
22. **Moving or deleting existing pages needs the owner's approval**, given
    as a keep/move/merge/delete list.

## Before handing back

23. Run `python scripts/docs/check_docs.py` and fix every error.
24. Report the pages changed, what each was checked against (build, version,
    source files), and anything found that needs fixing in the code (add it
    to the private findings file and the ticket, not to the docs).
25. Pages written by an agent are drafts until the area owner reviews them.
