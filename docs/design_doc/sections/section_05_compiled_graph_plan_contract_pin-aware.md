## 5) Compiled graph plan contract (pin-aware)

`CompiledGraphPlan` is a typed executable view of the selected training path.

```text
+----------------------+
| CompiledGraphPlan     |
+----------+-----------+
           |
           +-- compiled_nodes : [CompiledGraphNode]
           |        - graph_node_id
           |        - layer_id / operator id
           |        - node_type
           |
           +-- compiled_edges : [CompiledGraphEdge]
           |        - source node id, destination node id
           |        - from_pin, to_pin
           |
           +-- data/label/loss/optimizer pins
           |        - DataPin
           |        - LabelsPin
           |        - LossPin
           |        - PredictionsPin
           |        - Target/Optimizer-specific pins
           |
           +-- model_path_node_ids (linearized training path)
           +-- source_node_ids / sink_node_ids
```

Build behavior:
- `BuildCompiledGraphPlan(...)` receives selected nodes + sorted order + edge list.
- The plan normalizes role ids and keeps pin-level references for executor alignment.
- Exposes helper lookups for plan introspection and debug reporting.

---
