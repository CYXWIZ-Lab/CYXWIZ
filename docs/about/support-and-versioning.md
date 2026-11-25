---
title: Support and versions
description: How CyxWiz versions and releases are numbered, which release gets fixes during the alpha, and how to get help or report a bug.
type: reference
area: about
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - privacy.md
  - ../security/report-a-vulnerability.md
  - ../install/update-and-uninstall.md
  - ../legal/licensing-faq.md
redirect_from: []
---

CyxWiz is in **alpha**. It is usable and tested, but features, file formats
and the plugin API can still change between releases.

## Version numbers and releases

- **Product version**: three numbers, for example `1.0.7`. The Engine shows
  it in **Help > About CyxWiz...**. These docs give the version a behaviour first
  shipped in as `since`, and note later changes as "Changed in 1.0.8".
- **Releases** are published as pre-releases on the
  [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases), each with
  a tag named `cyxwiz-installer-alpha-<date>.<n>`, for example
  `cyxwiz-installer-alpha-20261018.1`.
- **GPU packs** have their own revisions inside a release. The installer
  shows **Update available** when a newer signed pack exists; see
  [Update or uninstall](../install/update-and-uninstall.md).
- **The plugin API** has its own version (for example plugin API 1.4). A
  plugin must be built for the Engine that loads it: the Engine refuses a
  plugin built differently.

## Which releases get fixes

During the alpha, **only the latest release gets fixes**, security fixes
included. A fix ships in a new release; earlier releases are not patched.
Update to the newest release before you report a problem.

## How stable each feature is

Every page of these docs has a status:

| Status | Means |
| --- | --- |
| **stable** | Works as documented; changes are noted on the page. |
| **beta** | Complete, still being checked; details may change. |
| **experimental** | New, may change or be removed in a later release. |
| **deprecated** | Still works, will be removed; the page names the replacement. |

## Get help

| You want to | Go to |
| --- | --- |
| Look up how something works | These docs; start at the [docs home](../index.md) |
| Check whether a problem is known | [Known issues](../known-issues.md) |
| Ask a usage or design question | [GitHub Discussions](https://github.com/CYXWIZ-Lab/CYXWIZ/discussions), when available |
| Report a bug or propose a feature | [GitHub Issues](https://github.com/CYXWIZ-Lab/CYXWIZ/issues), or **Help > Report Issue...** in the Engine |
| Report a security problem | Privately by email: [Report a vulnerability](../security/report-a-vulnerability.md) |
| Ask about the commercial licence | [cyxwiz.ai](https://cyxwiz.ai); see the [Licensing FAQ](../legal/licensing-faq.md) |

## What to put in a bug report

- The CyxWiz version (**Help > About CyxWiz...**) or source commit, the operating
  system and whether you build from source.
- The compute device: the route you chose and the one in use
  (**Preferences > Devices**), with any fallback reason.
- Exact steps, what you expected and what happened.
- The relevant log lines, and a small graph or dataset that shows the
  problem when you may share it.

Do not post passwords, tokens, personal data, private datasets or private
business documents.

## Related

- [What CyxWiz sends over the network](privacy.md)
- [Report a vulnerability](../security/report-a-vulnerability.md)
- [Update or uninstall](../install/update-and-uninstall.md)
