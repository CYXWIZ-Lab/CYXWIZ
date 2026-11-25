# How the CyxWiz documentation is organised

This folder is the single source of the CyxWiz Engine documentation. The
Engine website renders it: **stable** from the latest release tag, **next**
from the integration branch. Nothing about the Engine is documented anywhere
else; a page that is not here does not exist.

The model is the Python and Java documentation: a tutorial to learn, guides
to get things done, a complete reference (much of it generated from the
code), explanations of how things work, and the internals for contributors.

## The tree

| Folder | What goes in it | Page type |
| --- | --- | --- |
| `index.md` | The docs landing page: the entry points below | landing |
| `install/` | Download, the installer, GPU packs, first launch, command line, update, uninstall | how-to |
| `tutorials/` | A learning path and end-to-end lessons, each built on a committed example in `examples/` | tutorial |
| `user-guide/<area>/` | How to do things, one folder per area: `graph-editor`, `data-studio`, `training`, `evaluation`, `deployment`, `simulation`, `scripting`, `plugins`, `remote` | how-to |
| `main-window/` | The window layout, menus, toolbar, status bar, and one page per panel and dialog | reference |
| `concepts/` | How things work and why: how graphs run, the data pipeline, compute back ends, seeds and determinism, model formats | explanation |
| `reference/nodes/` | Every node: an index, a page per category, a page per node (generated) | reference |
| `reference/graph-spec/` | The `.cyxgraph` format, pin types, compile and execute rules, the node contract | reference |
| `reference/preferences/` | One page per Preferences tab (generated) | reference |
| `reference/` (files) | `shortcuts.md`, `menus.md`, `command-line.md` (generated), `file-formats/`, `error-messages.md` | reference |
| `sdk/` | Plugins and Python nodes: `get-started/`, `guides/`, and the generated `cpp-api/`, `python-api/`, `c-api/` | mixed |
| `server-and-cloud/` | The server node and CyxCloud: install, operate, storage, sign-in, security | how-to, reference |
| `distributed-training/` | Remote and distributed training: concepts, setting up machines, running jobs, limits | mixed |
| `troubleshooting/` | By symptom: the exact error text, the cause, the fix | troubleshooting |
| `whats-new/` | One page per release, written for users | release-notes |
| `release-notes/` | Per version, assembled from the fragments in `release-notes/next/` | release-notes |
| `faq.md`, `known-issues.md`, `glossary.md`, `deprecations.md` | As named | mixed |
| `internals/` | For contributors: architecture (with diagrams), code layout, build, tests, how to add a node or a panel, the protocol, decision records (`internals/decisions/`), and how to write docs (`internals/docs/`) | explanation, how-to |
| `security/` | The security model: what opened files, plugins, Python nodes and remote jobs can do; server node hardening; reporting a vulnerability | explanation, how-to |
| `legal/` | Licensing FAQ (AGPL, the commercial licence, the Apache SDK) and third-party notices | reference |
| `about/` | Support and versioning, privacy (what the apps send) | reference |

Generated pages say so in their front matter (`generated: true`) and are
never edited by hand: change the code or the generator instead.

### Pages still in the old locations

The guides in `usage/`, `extending/`, `architecture/` and `research/` are
moving into the tree above, one area at a time, each with a redirect from
its old path. Until a page has moved, its old location is the one to
update. Do not copy a page into the new tree while the old one still exists.

## Every page has front matter

```yaml
---
title: Prepare data in Data Studio
description: Clean, filter and split a dataset with Preparation Recipe steps.
type: how-to
area: data-studio
audience: user
status: stable
since: 1.0.7
owner: data-studio
last_reviewed: 2026-11-24
related: []
redirect_from: []
---
```

| Field | Required | Values |
| --- | --- | --- |
| `title` | yes | Sentence case, under 70 characters; also the page heading |
| `description` | yes | One sentence, under 160 characters, starting with a verb or a noun |
| `type` | yes | `tutorial`, `how-to`, `reference`, `explanation`, `landing`, `troubleshooting`, `release-notes` |
| `area` | yes | `general`, `install`, `graph-editor`, `data-studio`, `training`, `evaluation`, `deployment`, `simulation`, `scripting`, `plugins`, `remote`, `sdk`, `server-and-cloud`, `internals`, `security`, `legal`, `about` |
| `audience` | no | `user` (default), `plugin-developer`, `operator`, `contributor`, `organisation` (legal and procurement readers) |
| `status` | yes | `experimental`, `beta`, `stable`, `deprecated`, `removed` |
| `since` | yes | The first Engine version with this behaviour, for example `1.0.7` |
| `until`, `replaced_by` | when deprecated or removed | The version, and the page that replaces it |
| `owner` | yes | The area owner |
| `last_reviewed` | yes | `YYYY-MM-DD`: the day the page was last checked against the code |
| `related` | no | Up to five relative paths |
| `redirect_from` | no | Old paths of this page, after a move |
| `generated` | no | `true` on generated pages |

## Keeping the docs current

1. **Docs change with the code.** A change anyone can see (a panel, a node,
   a preference, a file format, the SDK or Python API) updates its page and
   adds a release-note fragment in `release-notes/next/` in the same commit.
2. **Feature branches carry their own pages**, marked `status: experimental`
   with `since:` set to the planned version. Whoever merges the branch
   resolves docs conflicts in the same merge and runs the docs check on the
   result.
3. **No copy of the docs per release.** Pages say `since`, and "Changed in
   1.0.8" notes go inline. The website builds stable from the release tag
   and next from the integration branch.
4. **Deprecate, then remove.** Set `status: deprecated`, `until` and
   `replaced_by`, and add a warning at the top. On removal, cut the page to a
   stub that points to the replacement, then delete the stub one release
   later. No page is ever left orphaned.
5. **Reviews.** A page past its review date (180 days, 90 for fast-moving
   areas) is checked against the code and its `last_reviewed` updated.
   Pages written by an agent are drafts until the area owner reviews them.
6. **Truth over completeness.** Document what the build does now, with its
   limits. Bugs go to `known-issues.md` and the issue tracker, not into the
   description of how something works.

7. **Know where the docs ended.** `sync.yaml` records, per area, the commit the
   area's docs were last checked against. `python scripts/docs/docs_drift.py`
   lists every commit since then that touches the area's code: that list is
   what the docs may need to catch up with. After updating an area, set its
   `synced_at` to the commit you checked.

Run the check before you hand in a change:

```bash
python scripts/docs/check_docs.py
```

It checks the front matter and every internal link. Agents follow the docs
skill in `.claude/skills/cyxwiz-docs/SKILL.md`; page templates are in
`internals/docs/templates/`.

## What never goes in these docs

Internal ticket numbers, branch names, local paths, machine names or IP
addresses, private repository details, secrets, and plans that are not
built. Internal design notes stay in the private engineering notes.
