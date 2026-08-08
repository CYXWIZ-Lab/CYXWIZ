# Kaggriculture Engine Benchmark

## Purpose

Use Kaggriculture as an agent/runtime benchmark for CyxWiz. Unlike RSNA Knee,
this is not primarily a supervised model-training challenge. Public descriptions
frame it as a farming simulation where an autonomous AI agent manages resources
and competes in a dynamic environment.

The public Kaggle page is not readable from this environment beyond the page
title. Exact rules, environment API, submission packaging, scoring, and runtime
limits must be confirmed from the Kaggle competition tabs or downloaded starter
kit.

## Challenge Shape

Expected workload:

- Autonomous agent decision loop.
- Environment state parsing.
- Action selection under resource constraints.
- Multi-step planning and strategy.
- Simulation rollouts, debugging, and replay.
- Kaggle-compatible agent packaging.
- Possibly LLM/tool-use constraints from the Google/Kaggle agent course.

## Current Engine Fit

Usable now:

- Python scripting surface for external agent code.
- Basic RL scaffolding: replay buffer, epsilon schedule, RL training executor,
  and RL script generation.
- Training/metric dashboard patterns that can be reused for simulation metrics.
- Plugin architecture that can represent environment nodes.
- Central/server-node architecture that could later run simulation jobs.

## Limitations Exposed

### Environment Loop

Current limitation:

- RL support is currently oriented around MuJoCo/PPO experiments.
- The C++ RL executor still contains placeholder zero-action stepping in its
  native loop.
- Existing tests already indicate missing RL training contract and missing
  environment-loop support for policy paths.

Smallest fix:

- Add a generic `AgentEnvironment` interface:
  `reset()`, `observe()`, `legal_actions()`, `step(action)`, `score()`,
  `done()`, and `serialize_replay()`.

### Agent Packaging

Current limitation:

- There is no Kaggle agent-submission wrapper.
- There is no runtime contract for a submitted policy function or agent class.

Smallest fix:

- Add an `experiments/kaggriculture/` harness that can import a starter-kit
  environment and run a local agent against it.
- Keep final Kaggle submission as a lean Python package generated from that
  harness.

### Observability

Current limitation:

- Training metrics exist, but simulation agents need step traces, resource
  curves, action histograms, terminal-state reasons, and replay inspection.

Smallest fix:

- Add a simulation trace CSV/JSONL format and dashboard importer.

### Planning And Tool Use

Current limitation:

- CyxWiz has model and scripting surfaces, but no first-class agent planner,
  memory store, tool registry, or safety budget for external tool calls.

Smallest fix:

- Start with deterministic heuristic agents and policy functions.
- Add optional LLM/tool-use only after the local environment loop and replay
  are stable.

## Recommended First Slice

Start with a local benchmark harness:

1. Download or copy the Kaggriculture starter kit into `experiments/kaggriculture/`.
2. Wrap the environment behind a minimal Python `AgentEnvironment` adapter.
3. Implement one deterministic baseline agent.
4. Log every episode to JSONL.
5. Import score/reward/resource curves into the CyxWiz dashboard.

This tests a core engine question:

- Can CyxWiz run, observe, and compare autonomous agent simulations?

If yes, the next limitation is planner quality. If no, the engine needs a
generic environment-loop and replay system before more agent features are worth
adding.

## Data Needed Next

Collect from Kaggle:

- starter notebook or starter kit
- environment API
- submission format
- scoring rule
- runtime/package restrictions
- allowed model/API/tool usage
