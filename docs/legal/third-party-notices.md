---
title: Third-party notices
description: Where the licence notices of the libraries CyxWiz ships are kept, and the main third-party components with their licences.
type: reference
area: legal
audience: organisation
status: stable
since: 1.0.7
owner: legal
last_reviewed: 2026-11-25
related:
  - licensing-faq.md
redirect_from: []
---

CyxWiz is built on open-source and vendor libraries. Each keeps its own
licence. The licence notices travel with every release package; this page
says where they are and lists the main components.

## Where the notices are

In an installed CyxWiz or an unpacked release package:

| Folder or file | Holds |
| --- | --- |
| `LICENSE`, `LICENSE-ADDITIONAL-PERMISSIONS.md`, `LICENSE-COMMERCIAL.md` | CyxWiz's own licence (see the [Licensing FAQ](licensing-faq.md)) |
| `THIRD_PARTY_LICENSES/vcpkg/` | The copyright and licence file of every library built through vcpkg, one file per package |
| `THIRD_PARTY_LICENSES/ArrayFire/` | ArrayFire's licences |
| `THIRD_PARTY_LICENSES/Intel/` | The notices for the Intel oneAPI runtimes (oneMKL, DPC++ / SYCL, TBB, UMF), including Intel's list of redistributable files |
| `THIRD_PARTY_LICENSES/NVIDIA/` | The notices for the NVIDIA CUDA runtimes, in packages that bundle them |
| The bundled Python runtime's `LICENSE` | Python's licence |
| `MUJOCO_LICENSE.txt`, `MUJOCO_THIRD_PARTY_NOTICES.txt` beside the MuJoCo plugin | MuJoCo's licence and its own third-party notices |
| The SDK folder's `LICENSE` and `NOTICE` | The Apache-2.0 licence of the Plugin SDK, including the Apache Arrow C data interface structs it holds |

The release tooling refuses to build a package when one of these notice sets
is missing.

## Main components

The licence named here is the one the project publishes; the notice file in
the package is the authoritative text for the version shipped.

| Component | Used for | Licence |
| --- | --- | --- |
| ArrayFire | GPU and CPU compute (CUDA, OpenCL, oneAPI, CPU) | BSD 3-Clause |
| NVIDIA CUDA runtime libraries | CUDA GPU pack | NVIDIA licence (redistributable runtime) |
| Intel oneAPI runtimes (oneMKL, DPC++ / SYCL, TBB) | CPU maths and the oneAPI GPU pack | Intel licence (redistributable runtime) |
| Apache Arrow | Tables in memory and Parquet files | Apache 2.0 |
| DuckDB | SQL in Data Studio | MIT |
| Dear ImGui, ImPlot | The user interface and plots | MIT |
| imnodes | The node editor canvas | MIT |
| ImPlot3D | 3D plots | MIT |
| ImGuiColorTextEdit | The Script Editor's text widget | MIT |
| GLFW, glad | Windows and OpenGL loading | zlib; MIT |
| stb | Image reading and writing | Public domain or MIT |
| gRPC, Protocol Buffers | Messages between the Engine, server nodes and services | Apache 2.0; BSD 3-Clause |
| OpenSSL | TLS and hashing | Apache 2.0 |
| cpp-httplib, jwt-cpp | HTTP calls and sign-in tokens | MIT |
| pybind11 | The `pycyxwiz` Python bindings | BSD 3-Clause |
| Python (bundled runtime) | Scripting and Python nodes | Python Software Foundation License |
| FFTW | Fourier transforms | GNU GPL 2.0 or later |
| OpenCV | Image processing (CPU) | Apache 2.0 |
| libsndfile | Audio files | GNU LGPL 2.1 |
| HighFive (HDF5) | HDF5 files | Boost Software License |
| OpenXLSX | Excel files | BSD 3-Clause |
| libarchive | Archives | BSD 2-Clause |
| SQLite | Local databases | Public domain |
| HiGHS | Linear and mixed-integer optimisation | MIT |
| ONNX, ONNX Runtime | ONNX models | Apache 2.0; MIT |
| llama.cpp | Local language models | MIT |
| spdlog, fmt, nlohmann/json, yaml-cpp | Logging, formatting, JSON and YAML | MIT |
| FTXUI, libvterm | The server node's text interface and terminal | MIT |
| Native File Dialog Extended | File dialogs | zlib |
| TweetNaCl | Plugin signature checks | Public domain |
| MuJoCo (plugin) | Physics simulation | Apache 2.0 |

Libraries used only to build and test (for example Catch2) are not shipped.

## FFTW and closed products

FFTW is licensed under the GNU GPL. It is compatible with CyxWiz's AGPL, but
a product licensed on closed terms under the commercial licence cannot
include it without a separate licence from FFTW's authors. See the
[Licensing FAQ](licensing-faq.md#when-do-i-need-the-commercial-licence).

## Related

- [Licensing FAQ](licensing-faq.md)
