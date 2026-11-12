# Distributed training: engineering design

Status: **design accepted 2026-11-12** (D1, D2, D3, D5, D7 as recommended; D4 and D6 open). No code is written against it yet.
Ticket: TOFIX166 (parent TOFIX163). Last updated 2026-11-12.

This page decides how CyxWiz trains one model on several server nodes. It
takes into account what the code has today, the lab's measured network, the
AI SYSTEM book (Fregly, *AI Systems Performance Engineering*) and published
decentralized-training research. Every number from outside has its source
in [Sources](#sources).

## 1. The problem in our terms

A user wants to train one model faster, or bigger, than one rented node
allows. The nodes are community machines:

- **Mixed speed.** An Intel Iris Xe laptop, an Iris Pro Mac, a GTX 1050 Ti
  and CPU-only Linux boxes differ by up to about 10x in speed for the same
  job.
- **Mixed memory.** From 1.5 GB to about 8 GB usable for training.
- **Mixed numeric support.** The Intel iGPUs have no float64, and float16
  is not usable everywhere. **Float32 is the only common type.**
- **Slow, uneven links.** Measured in the lab on 2026-11-12, file copies
  over ssh ran at 65-130 Mbit/s per machine. Ping was 7 ms to the Dell and
  72 ms to the Mac (Wi-Fi). Home internet uplinks are often 10-50 Mbit/s.
- **Unreliable.** Laptops sleep (the Mac did, 2026-11-11), owners close the
  node, batteries run out.
- **Untrusted.** A node is a stranger's machine (TOFIX176, TOFIX180).

The datacenter methods assume the opposite on every point: identical GPUs,
100+ GB/s links, nodes that never leave. The book says so plainly:

- More devices over weak links can be *slower* than fewer (p. 128).
- The slowest node sets the pace (pp. 123-124).
- Sharded training needs "a very fast multinode fabric" (p. 577).

## 2. What exists today

The code inventory was taken on 2026-11-12 from branch distributed-training.

| Piece | State |
| --- | --- |
| Remote training | One reservation = one node = one job (RunGraphTrainingJob on the node). Two nodes = two separate models. |
| Backend DDP stack (`cyxwiz-backend/include/cyxwiz/distributed/`) | Built and about 3.8k lines: ProcessGroup (AllReduce, Broadcast, ...), ring all-reduce over raw TCP, optional NCCL, DDP wrapper, DistributedSampler, DistributedTrainer. **No C++ caller and no C++ test.** The TCP transport reaches one host only (every rank dials `master_addr`), and has no authentication and no TLS. It wraps `SequentialModel`, not the graph executor (`IExecutableModel`). |
| Server node | No distributed code (`supports_distributed(false)`, `job_execution_service.cpp:65`). Never opens a connection to another node. |
| Protocol | No rank, world size, peers, group reservation or shard assignment. `ReserveNodeRequest` takes one node. |
| Training loop hook | `TrainingExecutor::AccumulateGradientsAndMaybeStep` (`training_executor.cpp:2169-2304`). Gradient exchange would plug in near `:2264`, periodic weight averaging after `:2295`. Parameters are accessed as `map<string, Tensor>`; there is **no flat weight-vector API**. Optimizers can export and import their state (`optimizer_base.h:32-62`). |
| Checkpoints | Format v2: named float32 tensors, optimizer state, runtime.json, SHA-256 manifest, refusal on a different graph. Written on the node every epoch and every 500 steps, and **deleted when the job ends**; only the epoch and hash reach the Engine. No path exists to move one between nodes. |
| Data | A node fetches whole files from CyxCloud (`cloud_dataset_fetcher`). There is no training-partition code (CyxCloud "shards" are erasure-coded storage pieces, not training splits). |
| Speed measure | The node benchmark reports `compute_score` (tokens/s) to central. Each job reports `samples_per_second`. |

**Conclusion.** No layer of the multi-node path exists in the product: no
protocol, no scheduling, no node-to-node path, no partition and no
aggregation. The backend DDP stack is the wrong shape for our network. It
synchronises every step, assumes one host, and is unauthenticated, so it is
not the base of this design (see [D6](#9-decisions-for-the-owner)).

## 3. Options weighed

Ψ is the model size in bytes. For a 10M-parameter float32 model Ψ = 40 MB;
for 100M parameters Ψ = 400 MB. "Per worker" means what each node sends
(and receives).

| Method | Traffic per worker | At 100 Mbit/s, 100M model | Mixed speed | Node leaves | Verdict |
| --- | --- | --- | --- | --- | --- |
| Synchronous DDP (ring all-reduce) | about 2Ψ **every step** | about 64 s per step, against under 1 s of compute | slowest node sets pace | the whole group stops | No |
| ZeRO-3 / FSDP (pool memory) | about 3Ψ **every step** | about 96 s per step | as DDP | as DDP | No (book p. 577) |
| DDP + compression (PowerSGD, DeMo, DisTrO) | 85-850x less, but **still every step** | seconds per step | slowest node sets pace | stops | Later, as an add-on at most |
| **Local SGD / DiLoCo** | about Ψ **per round** (a round = hundreds of steps) | 32 s fp32, 8 s int8 per round of 10-30 min | each node does as many steps as its speed allows | the round closes without it | **Yes: first** |
| Pipeline / SWARM (pool memory) | activations at each cut, not parameters | depends on batch × width; workable on a LAN | faster peers take more micro-batches | micro-batch resent to another peer | **Yes: second, for models too big for any node** |
| Tensor parallel | per-layer collectives | needs NVLink-class links | - | - | No |

Evidence for the chosen first method:

- **DiLoCo** (Douillard et al., 2023): 8 workers with H = 500 inner steps
  sent about 500x less data than data parallel and *beat* the baseline's
  perplexity. Rounds of H = 1000 cost only 2.9 %. Dropping half the
  updates at random cost 2.1 %.
- **OpenDiLoCo** ran across 2 continents at 90-95 % compute utilisation.
- **INTELLECT-1** trained 10B parameters across 3 continents with int8
  pseudo-gradients (400x less traffic than data parallel) at 83-96 %
  utilisation.
- **Asynchronous Local-SGD** sets each worker's step count from its speed,
  which is exactly our mixed-speed case.

## 4. Chosen design: rounds (DiLoCo with time-based rounds)

### 4.1 The algorithm

The **coordinator** keeps the global weights θ and the outer momentum m.
Each **worker** (a server node) keeps its own inner optimizer state
(AdamW, or whatever the graph's optimizer is).

```
round r:
  coordinator -> every worker:  θ_r (int8 or fp32), round deadline T, data chunk list
  worker i:
     θ_i := θ_r                      (keeps its own optimizer state across rounds)
     train inner steps on its chunks until the deadline T (fast nodes do more steps)
     Δ_i := θ_r − θ_i                (pseudo-gradient), n_i := samples trained
     send int8(Δ_i), n_i, loss stats
  coordinator (at deadline, with whoever reported):
     Δ := robust weighted mean of Δ_i, weights n_i      (trimmed mean, §6)
     m := μ·m + Δ ;  θ_{r+1} := θ_r − η_out·(Δ + μ·m)    (Nesterov; η_out 0.7, μ 0.9 to start)
     checkpoint θ_{r+1}, m to CyxCloud
```

- **Time-based rounds** absorb mixed speed: each worker trains for T, then
  its update is weighted by the samples it trained (FedNova and DiLoCo
  weighting).
- **Dropouts** cost little: a worker that misses the deadline is left out
  of that round. It rejoins at the next round with θ_{r+1}.
- **One exchange per round**, so a 72 ms Wi-Fi ping does not matter.
- **Float32 everywhere.** int8 is a wire format only, quantised on the CPU
  (the INTELLECT-1 scheme: μ ± 6σ, 256 buckets). Reduction is in float32.

### 4.2 Roles and topology

```
 Engine (user) ──submit job──► central (scheduler) ──picks group, issues tokens──┐
                                                                                 ▼
                      ┌────────────── coordinator (round service) ◄──────────────┘
                      │   θ, momentum, round state; checkpoints to CyxCloud
            θ_r ▲▼ Δ_i (gRPC/TLS, job-scoped token)
     ┌────────────┼──────────────┬──────────────┐
  node A        node B         node C   ...   (workers: fetch their chunks
 (Dell)         (Mac)          (HP)            from CyxCloud, train, send Δ)
```

- **Star topology, not a ring.** Every worker talks only to the
  coordinator. Robust aggregation needs all updates in one place. A star
  tolerates a missing worker, and no node needs to accept connections from
  another node, which is better for home NAT and firewalls.
- **The coordinator runs on the platform side, not on a user's node.** An
  untrusted node must never hold the global weights or do the averaging.
  It is a separate service next to central, because central holds policy,
  not the data path (book guardrail). It is stateless between rounds: θ and
  m are checkpointed to CyxCloud every round, so a restarted coordinator
  resumes.
- **Scaling.** With 16 workers the coordinator receives about 16 × 100 MB
  per round for a 100M model (about 13 s at 1 Gbit/s). Beyond that, regional
  relays aggregate first (hierarchical reduction, book pp. 126-130).

### 4.3 Data: chunks handed out per round

- The dataset in the user's CyxCloud space is split into **chunks** of a
  fixed number of rows (Arrow IPC files). This is the training partition,
  separate from CyxCloud's storage shards.
- Each round the coordinator gives each worker a chunk list sized to its
  measured speed (a work queue, not a fixed `rank::world` stride, so a slow
  node is never the bottleneck of an epoch).
- Workers fetch only their chunks, with a short-lived token scoped to the
  job and the chunk ids (TOFIX164 fetcher; TOFIX176 client-side
  encryption).
- An "epoch" is when every chunk has been trained once. The coordinator
  tracks it.

### 4.4 Scheduling: who decides

1. The user submits a **training request**: the graph, a dataset in
   CyxCloud, and either a deadline or a budget. The user does not pick
   nodes.
2. Central sizes the model's memory with the existing memory probe, then
   picks nodes that fit it. Each node has a measured job speed (a short
   warm-up on admission, plus its benchmark); central groups nodes of
   similar link quality and **shows the estimate first**: nodes, expected
   time, price.
3. The user accepts. One job id, one group reservation, one price, one
   Leave, one progress view with a row per node.
4. Manual mode stays: "rent this node" is today's single-node flow.

### 4.5 Failure handling (fail fast, never hang: book pp. 143-144)

| Event | Behaviour |
| --- | --- |
| Worker misses the round deadline | Its round is dropped; logged; reputation noted. |
| Worker gone 2 rounds | Removed; central may add a replacement, which starts from the last θ checkpoint. |
| Coordinator restarts | Reloads θ, m and round number from CyxCloud and resumes the round. |
| All workers gone | The job pauses; resumable from the last round checkpoint. |
| Update fails validation (§6) | Rejected; the node is flagged. |

Membership changes only at round boundaries: no live re-forming of a
collective (book p. 140).

### 4.6 What the user gets

The final θ goes into the user's CyxCloud space and the Engine downloads it.
A progress view shows loss per round, samples per node, nodes that dropped,
and the time split into network, data fetch and compute (book pp. 596-598:
measure every step's split).

## 5. Pooled memory ("a big GPU from small ones")

For a model that fits no single node: **pipeline stages across nodes**
(SWARM-style), in phase 4.

- Each stage holds a block of layers. Only activations cross the network,
  not parameters.
- Several peers can serve one stage, so faster peers take more
  micro-batches, and a failed peer's micro-batch goes to another.
- **Honesty rule.** The memory pools; the speed does not equal one real big
  card. The estimate shown before payment uses measured link speed. On our
  lab LAN this suits models whose layers are large; on home internet it is
  slow.
- Memory sizing for DiLoCo: a node needs about 4Ψ for float32 weights,
  gradients and two Adam moments, plus activations. A 1.5 GB node can join
  only small models (about 50M parameters or less) unless it serves a
  pipeline stage.

## 6. Trust: updates from strangers

A poisoned Δ_i shifts the global model. Defences, cheapest first:

1. **Robust aggregation.** Trimmed mean or centered clipping instead of a
   plain mean (Yin et al. 2018; Karimireddy et al. 2021). Cost O(n·d).
   Caveat: small coordinated attacks get past it (Baruch et al. 2019).
2. **Plausibility checks** on each Δ_i: its norm relative to the others,
   loss reported versus loss re-measured on a held-out batch by the
   coordinator.
3. **Spot redundancy.** A fraction p of chunks is trained by two nodes and
   their results are compared with a tolerance. Cost: (1 + p) times the
   compute.

Bitwise verification (Gensyn Verde) needs identical results across
hardware. ArrayFire on CUDA, OpenCL, oneAPI and CPU will not give that, so
we compare with a tolerance. Reputation (already in central) records the
outcomes.

## 7. Changes per component

| Component | Change |
| --- | --- |
| Protocol | A `training_group` service (coordinator ⇄ worker: JoinRound, SubmitUpdate, round state); `JobConfig` gains the group id, coordinator endpoint, chunk list and round deadline; a group reservation in reservation.proto. |
| Training core | A **round hook** in `TrainingExecutor`: stop at the deadline, export parameters as one flat float32 vector (new helper over `GetParameters`), apply the new θ (`SetParameters`) and keep the optimizer state. Used by every training loop that steps. |
| Server node | Worker mode: fetch chunks, train until the deadline, quantise and send Δ, receive θ. Opens outgoing connections only. |
| Coordinator (new, server side) | Round state machine, int8 decode, robust aggregation, Nesterov outer step, θ and m checkpoint to CyxCloud, per-node metrics. |
| Central | Group placement from memory fit, measured speed and link; group reservation; billing per group; reputation updates from round outcomes. |
| CyxCloud | Chunked dataset layout; job-scoped chunk tokens; θ checkpoint storage. |
| Engine | Training request (deadline/budget), the estimate card, the group progress view. Manual single-node rental unchanged. |

## 8. Verification plan

Design comes before code, and every stage is proven before the next.

1. **Measure the lab properly (Analyse).**
   - Bandwidth and latency between every pair of machines, and to the
     server side.
   - Per-node job speed on one reference graph.
   - The book's comm-vs-compute test (pp. 188-189).
2. **PyTorch oracle.** A Python reference of the exact round algorithm
   (same chunk assignment, seed, float32, int8 scheme). One round on our
   engine must match the oracle's θ_{r+1} within tolerance, as the
   single-node layers already do.
3. **One machine, two workers.** Two daemons on this PC plus a local
   coordinator: rounds, a worker dropping, a coordinator restart.
4. **Lab.** Dell + Mac (+ HP CPU) training one model. Compare the loss
   per wall-clock hour with one node alone; record the time split.
5. **Hostile.** A worker sending a scaled or random Δ must be rejected
   (TOFIX203 break-it-first).

## 9. Decisions for the owner

| # | Decision | Owner, 2026-11-12 |
| --- | --- | --- |
| D1 | The platform places jobs (user states the need); manual rental stays as a second path | Accepted |
| D2 | First algorithm: time-based DiLoCo rounds; pipeline (pooled memory) second | Accepted |
| D3 | The coordinator runs on the platform side (next to central), never on a user's node | Accepted |
| D4 | Pricing of a group: sum of node-hours shown up front, or a price per job | Open (business; needed by phase 2) |
| D5 | First target: Dell + Mac training one model, matched against the PyTorch oracle | Accepted |
| D6 | The unused backend DDP stack (single host, every-step sync, no auth): keep it as SDK API for same-host use, or remove it. It is not the base of this design. | Open (it ships in the SDK headers) |
| D7 | A node serves one job at a time while in a group | Accepted (revisit later) |

## 10. Phases

1. **Phase 0, Analyse:** lab measurements, PyTorch oracle, protocol draft.
2. **Phase 1:** synchronous time-based rounds, fp32 updates, trimmed mean,
   one machine, then the lab.
3. **Phase 2:**
   - int8 updates;
   - chunk work queue sized by speed;
   - group reservation and billing;
   - the Engine's estimate and group views.
4. **Phase 3:**
   - streaming fragments that overlap communication with compute
     (Streaming DiLoCo);
   - asynchronous rounds with staleness correction;
   - spot redundancy.
5. **Phase 4:** pipeline stages for models that fit no node.

## Sources

- Fregly, *AI Systems Performance Engineering* (internal/repository-private/AI SYSTEM), pages as cited: ch. 4 pp. 103-162, ch. 13 pp. 570-603, p. 140, pp. 143-144, p. 951.
- Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language Models", 2023, arXiv:2311.08105.
- Jaghouar, Ong, Hagemann, "OpenDiLoCo", 2024, arXiv:2407.07852.
- Prime Intellect, "INTELLECT-1 Technical Report", 2024, arXiv:2412.01152.
- Douillard et al., "Streaming DiLoCo", 2025, arXiv:2501.18512.
- Liu, Douillard et al., "Asynchronous Local-SGD Training for Language Modeling", 2024, arXiv:2401.09135.
- Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization" (FedNova), 2020, arXiv:2007.07481.
- Ryabinin et al., "SWARM Parallelism", 2023, arXiv:2301.11913.
- Borzunov et al., "Petals", 2022, arXiv:2209.01188.
- Rajbhandari et al., "ZeRO", 2020, arXiv:1910.02054.
- Peng, Quesnelle, Kingma, "DeMo: Decoupled Momentum Optimization", 2024, arXiv:2411.19870.
- Vogels, Karimireddy, Jaggi, "PowerSGD", 2019, arXiv:1905.13727.
- Yin et al., "Byzantine-Robust Distributed Learning", 2018, arXiv:1803.01498.
- Karimireddy, He, Jaggi, "Learning from History for Byzantine Robust Optimization", 2021, arXiv:2012.10333.
- Baruch et al., "A Little Is Enough", 2019, arXiv:1902.06156.
- Gensyn, "Verde: Verification via Refereed Delegation", 2025, arXiv:2502.19405.
