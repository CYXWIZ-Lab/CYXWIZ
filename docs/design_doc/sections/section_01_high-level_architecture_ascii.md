## 1) High-level architecture (ASCII)

```
   +--------------------+     +---------------------+
   |   Desktop Shell    |---->|   App/bootstrap     |
   | (desktop runtime)  |     |  CyxWizApp          |
   +--------------------+     +----------+----------+
                                         |
                                         v
+----------------------+          +-------+------------------------------+
|   GUI / Node Editor  |<-------->|             MainWindow               |
| (graph, pins, params) |          | Compile/Run orchestration           |
+----------+-----------+          +---------------+----------------------+
           |                                          |
           v                                          v
  +--------+---------+                      +---------+----------------+
  | Editor Graph IR  |                      | Compile/Training UI State  |
  +--------+---------+                      +-----------+---------------+
           |                                              |
           |                                              v
           |                                   +----------+--------------+
           +---------------------------------->| GraphCompiler            |
                                               | +Validation             |
                                               | +TrainingConfiguration  |
                                               +----+----------+---------+
                                                    |          |
                                                    |          v
                                                    |   +------+-------------------+
                                                    |   | CompiledGraphPlan         |
                                                    |   +------+-------------------+
                                                           |   |        |
                                                           |   |        |
                                                           |   |        v
                                                           |   |  PreflightValidator
                                                           |   |
                                   +-----------------------+------------+
                                   |                                |
                                   v                                v
                        +----------+--------+              +--------+-----------+
                        | GraphExecutable   |              | TrainingManager    |
                        | Model / GraphExec |              | lifecycle control  |
                        +----------+--------+              +----------+---------+
                                   |                                  |
                                   v                                  v
                    +--------------+---------------+          +-------+--------------------+
                    | PipelineMaterializer         |          | TrainingExecutor           |
                    | Runtime capability gating    |          | Arrow/Parquet/Legacy path  |
                    +--------------+---------------+          +-----------+----------------+
                                   |                                      |
                                   v                                      v
                    +--------------+---------------+          +-----------+----------------+
                    | Backend / dataset adapters   |          | Backends: CPU/GPU/Cuda  |
                    | (Arrow-centric passes)       |          | Metrics/Checkpoint       |
                    +------------------------------+          +-------------------------+
```

Interpretation:
- The compiler and preflight stage is the strict gate.
- Execution and training use different concrete executors but consume a common compiled contract.

---
