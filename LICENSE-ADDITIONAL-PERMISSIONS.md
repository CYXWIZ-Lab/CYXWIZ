# CyxWiz licence: scope and additional permissions

Copyright (c) 2026 CYXWIZ COMPUTER SYSTEMS L.L.C - S.P.C.

CyxWiz Engine is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version. The licence text is in [LICENSE](LICENSE).

CyxWiz Engine is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
General Public License for more details.

## 1. What this licence covers

"The Program" means the software in this repository that CYXWIZ owns, including the CyxWiz Engine
(`cyxwiz-engine`), the shared backend (`cyxwiz-backend`), the installer, the Python bindings and the
CyxWiz Server Node (`cyxwiz-server-node`).

These are not part of the Program and keep their own licences:

- **The CyxWiz Plugin SDK**: the files that carry `SPDX-License-Identifier: Apache-2.0` and the SDK
  folder shipped with each package (`cmake/sdk/LICENSE`, `cmake/sdk/NOTICE`). They are licensed under
  the Apache License 2.0.
- **Third-party components**, under their own licences and notices (see `THIRD_PARTY_LICENSES/`
  and the notices shipped with each package).
- **Services CYXWIZ runs and does not distribute** (the central server, CyxCloud, payments, GPU
  pack distribution, signing), which are not in this repository.

## 2. Additional permissions (GNU AGPL version 3, section 7)

As the copyright holder, CYXWIZ grants the following additional permissions. Anyone who conveys the
Program or a modified version may remove them from their copy, as section 7 allows.

### 2.1 Plugins

A plugin that interacts with the Program only through the interfaces of the CyxWiz Plugin SDK (the
Apache-2.0 files described in section 1), and is loaded by the Program at run time, is not a work
based on the Program. You may convey such a plugin under terms of your choice, including
proprietary terms. This permission does not cover changes to the Program itself, which remain under
the GNU AGPL.

### 2.2 Compute runtimes

You may link or combine the Program with the following libraries, and convey the combination,
without those libraries becoming subject to the GNU AGPL and without their source code being part of
the Corresponding Source: ArrayFire, the Intel oneAPI runtime libraries (including oneMKL and the
DPC++ / SYCL runtime), the NVIDIA CUDA runtime libraries, and the libraries these depend on, as
distributed by their vendors in object form. This permission does not cover any modified version of
those libraries that is not distributed by its vendor.

## 3. Commercial licence

CYXWIZ also offers the Program under a commercial licence for those who do not want the obligations
of the GNU AGPL. See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

## 4. Trademark

"CyxWiz", the CyxWiz logo and the installer identity are trademarks of CYXWIZ COMPUTER SYSTEMS
L.L.C - S.P.C. The GNU AGPL does not grant any right to use them. A modified version must be
distributed under a different name; it may state that it is based on CyxWiz.
