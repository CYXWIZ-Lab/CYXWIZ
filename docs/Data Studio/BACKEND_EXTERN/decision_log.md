# BackendExtern Architecture Decision Log

These decisions are accepted for planning. Altering one requires a new review,
because each protects the LTS boundary.

| ID | Decision | Rationale | Status |
| --- | --- | --- | --- |
| ADR-BE-01 | ArrayFire/native C++ remains the native compute path | Avoid multi-framework operator parity | Accepted |
| ADR-BE-02 | Curated framework work runs in a separate Python worker process | Contain crashes, GPU/CUDA/XLA state, and dependency conflicts | Accepted |
| ADR-BE-03 | BackendExtern core owns protocol/lifecycle; providers are optional | Project/task/provenance/security boundaries need stable Engine ownership | Accepted |
| ADR-BE-04 | Script Editor remains embedded/user-managed | Preserve flexibility without expanding LTS promise to arbitrary packages | Accepted |
| ADR-BE-05 | Runtime support is immutable and version-pinned | Old projects must not silently change framework behavior | Accepted |
| ADR-BE-06 | CPU-first, then one validated GPU tuple | Avoid an untestable hardware matrix | Accepted |
| ADR-BE-07 | Data begins as file/artifact exchange, not GPU zero-copy | Preserve isolation and simplify memory ownership | Accepted |
| ADR-BE-08 | TabFM is not a default commercial provider without rights clearance | Its published weight license is non-commercial | Accepted |
| ADR-BE-09 | v1 supports inference only | Training adds a separate lifecycle/checkpoint/optimizer contract | Accepted |
| ADR-BE-10 | No remote code or arbitrary provider Python protocol | Models must not gain code execution through import | Accepted |

## Open decisions before BE-1

1. Which operating system is the only supported target for the first managed
   runtime release?
2. Will the installer download packages or require an offline approved bundle?
3. What is the exact first commercially approved provider/model?
4. Are concurrent workers forbidden, queued, or allowed only on CPU?
5. What is the retention policy for cached models, input artifacts, and run
   reports?

