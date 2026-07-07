## 3) Core data contracts

### 3.1 Node catalog contract (editor-side)
Defined in `node_editor.h`:
- `NodeType`: broad typed enum of node families.
- `PinType`: semantic pin type used in graph connections.
- `MLNode`: node instance with:
  - `uuid` / node id
  - `name`
  - `nodeType`
  - `inputs`, `outputs` (`Pin`)
  - `parameters` (typed key/value + dynamic metadata)
  - optional `expanded`, `position`, layout metadata

Important capability:
- Editor metadata is intentionally permissive, but runtime compiles only what is compatible with active contracts.

### 3.2 Pin contract
Pins are the only stable cross-layer interface between graph and compiler:
- name (e.g. `"Data"`, `"Labels"`, `"Loss"`, `"Predictions"`, `"Optimizer"`)
- direction (input/output)
- expected type hints and defaults
- dynamic updates during graph load/import.

### 3.3 Runtime node contract
Defined in `node_executors/node_executor.h` and factory/provider interfaces:
- `NodeExecutor` implements behavior by contract.
- `node_executor_factory` binds node categories to concrete runtime behavior.
- `INodeProvider` permits external providers to add:
  - node descriptors
  - parameters
  - pin metadata
  - optional runtime behavior.

Interpretation:
- Editor-side node richness does not imply runtime support; capability tables are required.

---
