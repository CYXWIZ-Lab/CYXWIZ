# Reading Log

Use this file as the running extraction log. Keep entries short and action-oriented.

## Entry Template

```md
## Chapter N: Title

Read scope:

- Pages or sections read.

Production ideas:

- Idea:
- CyxWiz component:
- Why it matters:
- Smallest useful task:
- Validation:
- Risk/complexity:

Deferred ideas:

- Idea:
- Reason deferred:
```

## Initial Pass: Table of Contents and Repo Fit

Read scope:

- Front matter and table of contents sampled from the root PDF.
- Local context from `README.md`, `cyxwiz-backend/README.md`, `cyxwiz-backend/TODO.md`, and `cyxwiz-server-node/README.md`.

Production ideas:

- Idea: Treat performance as a full-stack property across hardware, runtime, networking, storage, scheduling, and application behavior.
- CyxWiz component: both backend and server node.
- Why it matters: CyxWiz is a distributed compute platform, so raw ML code speed is not enough for production reliability.
- Smallest useful task: add benchmark and metrics contracts before implementing new optimization features.
- Validation: repeatable benchmark output and structured node metrics.
- Risk/complexity: chasing advanced GPU techniques before the node can explain failures or slow jobs.

- Idea: Separate backend compute primitives from server-node deployment policy.
- CyxWiz component: backend public API and server-node deployment manager.
- Why it matters: the backend should remain reusable by engine and node, while job admission, retries, queues, and SLOs belong to the node/orchestrator layer.
- Smallest useful task: document and enforce the deployment lifecycle state machine.
- Validation: lifecycle tests for create, run, stop, fail, complete, and cleanup.
- Risk/complexity: putting scheduling policy into the backend would make the shared library harder to reuse.

- Idea: Introduce useful-work metrics.
- CyxWiz component: metrics collector, metrics storage, node heartbeat, central server scheduling.
- Why it matters: GPU utilization alone can hide queue stalls, model loading delays, artifact transfer failures, and bad scheduling.
- Smallest useful task: add goodput-oriented fields to job metrics.
- Validation: a completed job report includes accepted work, failed work, queue time, execution time, and failure category.
- Risk/complexity: metrics can become noisy unless the schema is small and stable.

Deferred ideas:

- Idea: disaggregated prefill/decode and KV cache pools.
- Reason deferred: valuable for large LLM inference, but CyxWiz first needs basic inference SLOs, batching, memory budgets, and observability.

- Idea: custom Triton kernels.
- Reason deferred: backend currently depends on ArrayFire and should not add a second low-level kernel stack until benchmarks prove a specific gap.

- Idea: Kubernetes topology tuning.
- Reason deferred: deployment-level concern; document readiness first, then add production deployment profiles later.

