## 14) End-to-end dataflow state machine (training path)

```text
IDLE
  |
  +--> COMPILE_REQUEST
        |
        +--> VALIDATE_GRAPH
             |
             +--error--> COMPILE_BLOCKED
             |
             +--> BUILD_TRAINING_CONFIG
                   |
                   +--> COMPILE_SUCCESS
                          |
                          +--> PREFLIGHT
                               |
                               +--error--> PREFLIGHT_BLOCKED
                               |
                               +--> MATERIALIZE
                                    |
                                    +--error--> MATERIALIZATION_BLOCKED
                                    |
                                    +--> EXECUTOR_INIT
                                         |
                                         +--> TRAINING_LOOP
                                                |
                                                +--> PAUSED | STOPPED | FAILED | SUCCEEDED
```

---
