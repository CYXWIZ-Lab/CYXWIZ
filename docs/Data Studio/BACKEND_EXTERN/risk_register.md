# BackendExtern Risk Register

| Risk | Why it matters | Required mitigation |
| --- | --- | --- |
| Core bloat | Framework code spreads through native operators | Keep frameworks in workers; core sees only protocol types |
| GPU conflict | ArrayFire/PyTorch/JAX may have incompatible runtime state | Separate process; CPU-first; explicit GPU matrix |
| Dependency drift | Latest framework upgrade breaks old project | Immutable runtime IDs and lockfiles |
| Model format ambiguity | Weights alone do not identify architecture | Named provider adapter validates all schema/tensors/config |
| Unsafe model code | Hub repos may ship executable Python/pickle | No remote code; safetensors/approved formats first; allow-list adapters |
| Licensing mistake | Paid product may ship restricted weights | Legal catalog gate; no default TabFM commercial distribution |
| Orphan workers | Engine close leaves process/GPU memory behind | Process groups/job objects, shutdown tests, bounded termination |
| Silent fallback | User believes GPU/native execution occurred | Run report always records plane/framework/device/reason |
| Support explosion | Every package/platform becomes a promise | Support named tuples only; user-managed path separate |
| Result corruption | Worker returns malicious/partial output | Hash/schema/path validation before import |

## Review gate

No implementation ticket proceeds if it widens a core interface merely to make
one framework easier to call. The provider should absorb framework-specific
complexity unless at least two shipped providers prove a shared, stable need.

