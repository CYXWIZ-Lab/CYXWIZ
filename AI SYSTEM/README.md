# AI System Notes for CyxWiz

This folder captures practical engineering notes from:

`AI Systems Performance Engineering: Optimizing Model Training and Inference Workloads with GPUs, CUDA, and PyTorch` by Chris Fregly, O'Reilly, first release 2025-11-11.

The goal is not to summarize the whole book. The goal is to convert the book into production work for:

- `cyxwiz-backend`: the shared ML compute library used by the engine and server node.
- `cyxwiz-server-node`: the distributed worker that registers with the central server, loads models, runs jobs, exposes metrics, and serves deployment APIs.

## Working Method

The book is large, so notes should be built incrementally:

1. Read one chapter or one focused section.
2. Extract only production-relevant ideas.
3. Map each idea to a CyxWiz component.
4. Turn the idea into a small testable engineering task.
5. Reject ideas that add broad complexity before CyxWiz has the observability and benchmarks to prove they are needed.

## Current Documents

- [book-to-cyxwiz-map.md](book-to-cyxwiz-map.md): maps the book's major topics to backend and server-node production areas.
- [production-roadmap.md](production-roadmap.md): staged roadmap for turning CyxWiz into a production-grade AI compute node.
- [reading-log.md](reading-log.md): template and running log for chapter-by-chapter extraction.

## Production Principle

CyxWiz should become production-grade through measurement first, then targeted optimization. The backend and node should avoid accumulating every advanced AI systems feature at once. The first durable production layer is:

- clear device and memory contracts,
- reproducible benchmarks,
- reliable job lifecycle management,
- metrics that distinguish useful work from raw utilization,
- safe fallbacks when GPU, model, network, or storage conditions fail.

