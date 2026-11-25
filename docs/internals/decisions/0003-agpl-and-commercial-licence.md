---
title: "Decision 0003: AGPL, a commercial licence and an Apache SDK"
description: Why the Engine and Server Node are AGPL-3.0, the plugin SDK is Apache-2.0, and a commercial licence is offered.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - README.md
---

**Status:** accepted (2026-11-20). A legal review of the wording is open.

## Context

CyxWiz wants the Engine and the Server Node to stay open, including when
someone runs a modified version as a network service, while letting
companies write closed plugins, and letting CYXWIZ license the code to
those who cannot accept copyleft terms.

## Decision

- The Engine, the backend, the installer, the Python bindings and the
  Server Node are licensed under the **GNU AGPL v3.0 or later** (`LICENSE`).
- The **plugin SDK** files (marked `SPDX-License-Identifier: Apache-2.0`)
  are licensed under **Apache-2.0**.
- Additional permissions under AGPL section 7
  (`LICENSE-ADDITIONAL-PERMISSIONS.md`) say that a plugin using only the
  SDK interfaces is not a work based on the Engine, and allow combining the
  Engine with ArrayFire and the Intel oneAPI and NVIDIA CUDA runtimes.
- A **commercial licence** (`LICENSE-COMMERCIAL.md`) is offered for use
  without the AGPL obligations.
- Outside contributions need a contributor licence agreement, so they can
  be offered under both licences.

## Consequences

- A modified Engine or Server Node offered to users over a network must
  publish its complete source.
- Third-party components keep their own licences. Some (FFTW, under the
  GPL) cannot be part of a closed product without their authors' licence.
- The CyxWiz name and logo are trademarks and are not licensed by the AGPL.
