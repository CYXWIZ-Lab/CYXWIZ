---
title: Licensing FAQ
description: Answers on the GNU AGPL licence of CyxWiz, its network clause, the plugin and runtime permissions, the Apache plugin SDK and the commercial licence.
type: explanation
area: legal
audience: organisation
status: stable
since: 1.0.7
owner: legal
last_reviewed: 2026-11-25
related:
  - third-party-notices.md
  - ../about/support-and-versioning.md
redirect_from: []
---

This page explains how CyxWiz is licensed, in plain words. It is not legal
advice, and the licence texts decide: `LICENSE` (GNU AGPL),
`LICENSE-ADDITIONAL-PERMISSIONS.md` and `LICENSE-COMMERCIAL.md` in the
source repository and in each release package. Ask your own lawyer about
your situation.

## What licence is CyxWiz under?

| Part | Licence |
| --- | --- |
| CyxWiz Engine (`cyxwiz-engine`), the shared backend (`cyxwiz-backend`), the installer, the Python bindings, the CyxWiz Server Node (`cyxwiz-server-node`) | GNU Affero General Public License, version 3 or later (AGPL-3.0-or-later), with the additional permissions below |
| The CyxWiz Plugin SDK: files marked `SPDX-License-Identifier: Apache-2.0` and the SDK folder shipped with each package | Apache License 2.0 |
| Third-party components | Their own licences; see [Third-party notices](third-party-notices.md) |
| Services CYXWIZ runs and does not distribute (the central server, CyxCloud, GPU pack distribution, signing) | Not in the repository and not covered by these licences |

The copyright holder is CYXWIZ COMPUTER SYSTEMS L.L.C - S.P.C.

## Can I use CyxWiz at work, for free?

Yes. The AGPL lets anyone run, study and change the program, for any
purpose, including commercial work. The AGPL covers the program; in general
it does not cover the models, graphs and data you make with it.

## When do I have to publish source code?

Only when you **distribute** CyxWiz or a modified version, or let others
**use a modified version over a network**:

- **Using it as it is**, on your own machines or your organisation's: no
  obligation.
- **Changing it for your own use** without giving it to anyone: no
  obligation.
- **Giving copies to others** (inside a product, on a download page, to a
  customer): you must give them the complete source of that version under
  the AGPL.
- **Running a modified version that others use over a network**: section 13
  of the AGPL (the network clause) applies.

## What does the network clause mean for server nodes and hosted use?

Section 13 says: if you modify the program and users interact with it
remotely over a network, you must offer those users the complete source of
your modified version, under the AGPL.

- **A server node you run unchanged**, lending compute to Engine users:
  no source obligation beyond the AGPL's general ones.
- **A modified server node** that Engine users connect to: you must offer
  those users the source of your modified node.
- **CyxWiz, or a modified version, offered as a hosted service** (for
  example the Engine's training or serving behind your web service): if it is
  modified, you must offer its users the source.

## Can I write a closed-source plugin?

Yes. The additional permission in section 2.1 of
`LICENSE-ADDITIONAL-PERMISSIONS.md` says: a plugin that talks to CyxWiz only
through the Plugin SDK's interfaces, and is loaded at run time, is not a work
based on CyxWiz. You may license it as you like, including proprietary terms.

The SDK itself is Apache-2.0: you may build, distribute and sell plugins that
include its headers, link its import library or start from its template.
The permission does not cover changes to CyxWiz itself; those stay under the
AGPL.

## Does the AGPL reach ArrayFire, CUDA or oneAPI?

No. Section 2.2 lets you combine CyxWiz with ArrayFire, the Intel oneAPI
runtime libraries (oneMKL and the DPC++ / SYCL runtime), the NVIDIA CUDA
runtime libraries, and the libraries they depend on, as their vendors
distribute them, without those libraries becoming subject to the AGPL. A
modified version of those libraries not distributed by its vendor is not
covered.

## When do I need the commercial licence?

When you want to build on CyxWiz without the AGPL's obligations, for example
to:

- ship CyxWiz, or a product built from its code, without publishing the
  source;
- run a modified version as a service without offering its source.

The commercial licence is based on the CYXWIZ Commercial Source and
Evaluation License, with the production and commercial rights set by a
signed agreement. It covers the code CYXWIZ owns. Third-party components
keep their own licences, and some cannot go into a closed product without a
separate licence from their authors: FFTW, for example, is under the GNU GPL.

To ask about it, contact CYXWIZ through [cyxwiz.ai](https://cyxwiz.ai).

## Can I contribute code?

Because CYXWIZ offers CyxWiz under both licences, it needs the right to
license contributions both ways. Outside contributions therefore need a
signed contributor licence agreement. Open an issue before you send a pull
request; see `CONTRIBUTING.md` in the source repository.

## Can I use the name and logo?

"CyxWiz", the CyxWiz logo and the installer identity are trademarks of
CYXWIZ COMPUTER SYSTEMS L.L.C - S.P.C. The AGPL grants no right to use them.
A modified version must be distributed under a different name; it may say
that it is based on CyxWiz.

## Related

- [Third-party notices](third-party-notices.md)
- [Support and versions](../about/support-and-versioning.md)
