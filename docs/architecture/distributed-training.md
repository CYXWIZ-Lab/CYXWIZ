# Distributed training

Status: designed, not built yet. This page is the public summary; the work is
tracked internally.

## What it is

Training one model on several CyxWiz server nodes at once. Nodes are
community machines with different speeds, memory and graphics devices,
joined over home networks or the internet. The user states what the job
needs; the platform finds the nodes, shows the expected time and price before
anything starts, and returns one trained model.

Renting a single node and training on it (Deploy > Connect to Server) stays
as it is.

## How it trains: rounds

Synchronising after every training step, as datacenter systems do, would
spend most of the time on the network at home-network speeds. CyxWiz trains
in **rounds** instead, the approach of DiLoCo (Douillard et al., 2023,
arXiv:2311.08105) and INTELLECT-1 (Prime Intellect, 2024,
arXiv:2412.01152):

1. Every node starts the round from the same model.
2. Each node trains on its own share of the data until the round ends. A
   faster node simply does more steps.
3. Each node sends the change it made, compressed. The changes are combined
   (weighted by how much data each node trained on) and applied to the
   model.

So the model is exchanged once per round, not once per step. A node that
drops out only misses that round. Training stays in float32, the type every
supported device has.

A model too large for any one node will later be split by layers across
several nodes. The page that offers it will say plainly that pooled memory
is not the speed of one large card.

## Data

Datasets stay in the user's CyxCloud space. Each node fetches only its own
share for the round, with access limited to that job.

## Checking it

A PyTorch reference of the round algorithm defines the expected results;
the Engine's implementation must match it, as single-node layers do:

- `cyxwiz-engine/tests/computation_truth/reference/round_training_reference.py`
- `cyxwiz-engine/tests/computation_truth/reference/generate_round_training_fixtures.py`
- `cyxwiz-engine/tests/computation_truth/fixtures/round_training_pytorch.json`

The test lab is described in [distributed-test-lab.md](distributed-test-lab.md).
