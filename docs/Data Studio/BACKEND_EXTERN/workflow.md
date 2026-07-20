# BackendExtern Workflows

## 1. First use and runtime installation

```text
User selects a managed external node
  -> Engine checks its provider/runtime requirement
  -> missing runtime: show size, platforms, licenses, support date
  -> user approves install
  -> verified installer creates managed environment outside Engine process
  -> worker health probe validates protocol and supported device report
  -> runtime becomes available
```

Do not auto-install a GPU runtime merely because a GPU appears present. Device
and package choice must be explicit and diagnosable.

## 2. Node execution

```text
Data Studio table -> External Model node
  -> compile validates provider, runtime, input schema, and model policy
  -> Engine writes immutable input artifact
  -> BackendExternService starts/reuses matching worker
  -> worker loads approved model revision and runs requested operation
  -> events stream to task UI and node status
  -> Engine validates result artifact and imports predictions/metrics
  -> graph continues through native CyxWiz nodes
```

## 3. Project reopen and reproducibility

```text
Open project
  -> read stored runtime/model/provider provenance
  -> exact runtime exists and passes health probe: runnable
  -> runtime missing but catalog can install exact version: install-needed
  -> exact version no longer available: blocked with migration guidance
  -> never silently replace with latest framework/model
```

## 4. User-script workflow

The Script Editor continues to call the current `ScriptingEngine` and current
project Python environment. A script may use PyTorch/JAX/Flax if the user
installs it, but it is labelled `user-managed` and does not become a managed
external model node automatically.

Future work may offer a command to run an approved script through a worker,
but arbitrary script execution is outside Runtime Contract v1.

## 5. Error workflow

Every visible error must say where it failed:

- runtime unavailable;
- runtime health probe failed;
- provider/model not installed;
- model license disallows selected product path;
- input schema invalid;
- worker crash/timeout/cancelled;
- framework/device failure;
- result schema invalid.

No error may be reported as generic native `ArrayFire` placement failure when
the work was external.

