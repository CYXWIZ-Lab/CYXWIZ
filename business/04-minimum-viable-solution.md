# CyxWiz Minimum Viable Solution (MVS)

## What is MVS?

The **Minimum Viable Solution** is the smallest version of CyxWiz that:
1. Solves the core problem (expensive ML compute)
2. Delivers value to both sides of the marketplace (buyers + sellers)
3. Validates our key assumptions
4. Generates initial revenue
5. Provides learning for iteration

**Philosophy**: Build the smallest thing that proves the model works, then iterate based on real usage.

---

## Core Hypothesis to Validate

Before building, we must validate these assumptions:

| # | Hypothesis | How MVS Validates |
|---|------------|-------------------|
| 1 | People will pay for decentralized ML compute | Users complete paid jobs |
| 2 | GPU owners will run nodes for projected earnings | Nodes stay online, accept jobs |
| 3 | P2P compute can be reliable enough | Job success rate > 90% |
| 4 | We can price 50%+ below cloud | Users confirm savings |
| 5 | The UX is simple enough for adoption | Users complete jobs without support |

---

## MVS Scope Definition

### What's IN (Must Have)

```
┌─────────────────────────────────────────────────────────────┐
│                    MVS Feature Set                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ENGINE (Desktop Client)                                    │
│  ───────────────────────                                    │
│  ✓ User authentication (wallet-based)                       │
│  ✓ Simple job submission (upload script + data)             │
│  ✓ Job monitoring (status, logs, progress)                  │
│  ✓ Results download                                         │
│  ✓ Basic cost estimation                                    │
│  ✓ Wallet integration (deposit/withdraw)                    │
│                                                             │
│  SERVER NODE (Compute Provider)                             │
│  ──────────────────────────────                             │
│  ✓ One-click installation                                   │
│  ✓ Auto-registration with network                           │
│  ✓ Job execution (Python/PyTorch scripts)                   │
│  ✓ Resource reporting (GPU, memory, utilization)            │
│  ✓ Earnings dashboard                                       │
│  ✓ Basic availability settings (on/off)                     │
│                                                             │
│  CENTRAL SERVER (Orchestrator)                              │
│  ────────────────────────────                               │
│  ✓ Node registry (join/leave/heartbeat)                     │
│  ✓ Job queue and assignment                                 │
│  ✓ Basic matching (requirements → available nodes)          │
│  ✓ Payment escrow (hold until complete)                     │
│  ✓ Job verification (did it complete?)                      │
│  ✓ Simple dispute handling (refund on failure)              │
│                                                             │
│  BLOCKCHAIN                                                 │
│  ──────────                                                 │
│  ✓ Wallet connection (Phantom/Solflare)                     │
│  ✓ Payment escrow smart contract                            │
│  ✓ Job payment settlement                                   │
│  ✓ Basic transaction history                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What's OUT (Post-MVS)

| Feature | Why Deferred |
|---------|--------------|
| Visual node editor | Complex; script upload works for MVS |
| Advanced job scheduling | Basic FIFO queue sufficient initially |
| Reputation system | Need transaction history first |
| Token staking | Adds complexity; simple escrow first |
| Federated learning | Advanced feature, not core value prop |
| Model marketplace | Separate product, build later |
| Team/collaboration features | Single user focus first |
| Mobile apps | Desktop sufficient for target users |
| Multi-GPU distributed training | Single-node jobs first |
| Custom container support | Standard PyTorch environment first |
| Advanced analytics | Basic logs sufficient |
| White-label/API access | Direct usage first |

---

## MVS User Journeys

### Journey 1: ML Developer (Compute Buyer)

```
┌─────────────────────────────────────────────────────────────┐
│              Compute Buyer Journey (MVS)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DISCOVER                                                │
│     └─► Finds CyxWiz (word of mouth, search, social)        │
│                                                             │
│  2. SIGN UP                                                 │
│     └─► Downloads Engine                                    │
│     └─► Connects Solana wallet                              │
│     └─► Deposits funds (SOL or USDC)                        │
│                                                             │
│  3. SUBMIT JOB                                              │
│     └─► Uploads Python training script                      │
│     └─► Uploads dataset (or provides URL)                   │
│     └─► Specifies requirements (GPU type, memory)           │
│     └─► Reviews cost estimate                               │
│     └─► Confirms and submits                                │
│                                                             │
│  4. MONITOR                                                 │
│     └─► Watches job status (queued → running → complete)    │
│     └─► Views real-time logs                                │
│     └─► Sees cost accumulating                              │
│                                                             │
│  5. RETRIEVE                                                │
│     └─► Downloads trained model                             │
│     └─► Downloads logs and metrics                          │
│     └─► Reviews final cost                                  │
│                                                             │
│  6. REPEAT                                                  │
│     └─► Submits more jobs                                   │
│     └─► Refers others                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Success Metrics**:
- Time from signup to first job: < 15 minutes
- Job submission flow: < 5 minutes
- Support tickets per job: < 0.1

---

### Journey 2: GPU Owner (Compute Seller)

```
┌─────────────────────────────────────────────────────────────┐
│              Compute Seller Journey (MVS)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. DISCOVER                                                │
│     └─► Learns about earning with idle GPU                  │
│                                                             │
│  2. INSTALL                                                 │
│     └─► Downloads Server Node installer                     │
│     └─► Runs installer (auto-detects GPU)                   │
│     └─► Connects Solana wallet for payments                 │
│                                                             │
│  3. CONFIGURE                                               │
│     └─► Sets availability (always on / scheduled)           │
│     └─► Sets minimum acceptable price (or accepts default)  │
│     └─► Verifies hardware detection                         │
│                                                             │
│  4. OPERATE                                                 │
│     └─► Node registers with network                         │
│     └─► Receives and executes jobs automatically            │
│     └─► Views earnings dashboard                            │
│                                                             │
│  5. EARN                                                    │
│     └─► Payments arrive after job completion                │
│     └─► Withdraws to wallet anytime                         │
│                                                             │
│  6. SCALE (optional)                                        │
│     └─► Adds more GPUs/machines                             │
│     └─► Refers other GPU owners                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Success Metrics**:
- Time from download to first job received: < 30 minutes
- Node uptime: > 95%
- Earnings vs. projection: within 20%

---

## MVS Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MVS Architecture                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────────┐                        │
│                      │   Engine    │                        │
│                      │  (Desktop)  │                        │
│                      └──────┬──────┘                        │
│                             │ gRPC                          │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Central Server                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │    Job     │ │    Node    │ │  Payment   │       │   │
│  │  │  Service   │ │  Registry  │ │  Service   │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘       │   │
│  │         │              │              │              │   │
│  │         └──────────────┼──────────────┘              │   │
│  │                        │                             │   │
│  │                 ┌──────┴──────┐                      │   │
│  │                 │  PostgreSQL │                      │   │
│  │                 └─────────────┘                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                             │                               │
│              ┌──────────────┼──────────────┐                │
│              │ gRPC         │ gRPC         │ gRPC           │
│              ▼              ▼              ▼                │
│       ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│       │  Server   │  │  Server   │  │  Server   │          │
│       │  Node 1   │  │  Node 2   │  │  Node N   │          │
│       └───────────┘  └───────────┘  └───────────┘          │
│                                                             │
│                             │                               │
│                             ▼                               │
│                      ┌─────────────┐                        │
│                      │   Solana    │                        │
│                      │ Blockchain  │                        │
│                      └─────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Specifications (MVS)

#### Engine (Desktop Client)
| Aspect | MVS Specification |
|--------|-------------------|
| Platform | Windows first, Linux second |
| Auth | Solana wallet signature |
| Job Types | Python scripts with PyTorch |
| Data Upload | Direct upload < 1GB, URL for larger |
| UI | Functional, not polished |

#### Server Node
| Aspect | MVS Specification |
|--------|-------------------|
| Platform | Windows and Linux |
| GPU Support | NVIDIA (CUDA) only |
| Environment | Pre-configured PyTorch container |
| Isolation | Docker container per job |
| Monitoring | Basic resource reporting |

#### Central Server
| Aspect | MVS Specification |
|--------|-------------------|
| Hosting | Single server (cloud VM) |
| Database | PostgreSQL |
| Scaling | Vertical only (upgrade VM) |
| Redundancy | Daily backups, no HA |

#### Blockchain
| Aspect | MVS Specification |
|--------|-------------------|
| Network | Solana Devnet → Mainnet |
| Payments | SOL and USDC |
| Contract | Simple escrow (no streaming) |

---

## MVS Feature Details

### Job Submission (Engine)

**Input Requirements**:
```
├── training_script.py     (required)
├── requirements.txt       (optional, common packages pre-installed)
├── data/                  (required)
│   └── [dataset files]
└── config.json            (optional)
    ├── gpu_type: "any" | "rtx3080" | "rtx4090" | ...
    ├── min_vram: 8  (GB)
    ├── max_price: 1.00  ($/hour)
    └── timeout: 3600  (seconds)
```

**Supported Frameworks (Pre-installed)**:
- PyTorch 2.x
- NumPy, Pandas, Scikit-learn
- Transformers (HuggingFace)
- Common utilities

### Job Execution (Server Node)

**Execution Flow**:
```
1. Receive job assignment from Central Server
2. Pull job files from secure storage
3. Create isolated Docker container
4. Install any additional requirements
5. Execute training script
6. Stream logs to Central Server
7. Upload results (model, metrics)
8. Clean up container and files
9. Report completion
```

**Resource Limits**:
- Max runtime: 24 hours (configurable)
- Max output size: 10 GB
- Network: Outbound only (no inbound connections)

### Payment Flow (Blockchain)

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Submit  │───►│ Escrow  │───►│Complete │───►│ Settle  │
│   Job   │    │ Created │    │   Job   │    │ Payment │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
  User pays     Funds held     Verified by    90% → Node
  estimated     in escrow      completion     10% → Platform
  cost          contract       check          Excess → User
```

**Pricing Model (MVS)**:
- Fixed rate per GPU-hour (set by platform initially)
- Example: RTX 4090 = $0.80/hour
- User pays upfront estimate
- Refund excess or charge deficit on completion

---

## MVS Success Criteria

### Launch Criteria (Ready to Release)

| Criteria | Target |
|----------|--------|
| Job success rate (internal testing) | > 95% |
| End-to-end job time (simple test) | < 10 min overhead |
| Node installation success rate | > 90% |
| Payment settlement accuracy | 100% |
| Critical bugs | 0 |
| Security audit | Passed |

### Post-Launch Success (First 90 Days)

| Metric | Target | Stretch |
|--------|--------|---------|
| Registered nodes | 50 | 200 |
| Active nodes (daily) | 20 | 100 |
| Jobs completed | 500 | 2,000 |
| Unique paying users | 100 | 500 |
| Gross transaction volume | $10K | $50K |
| Job success rate | > 90% | > 95% |
| NPS score | > 30 | > 50 |
| Node churn (monthly) | < 20% | < 10% |

---

## What MVS is NOT

To stay focused, explicitly exclude:

| Not Building | Why |
|--------------|-----|
| Perfect UX | Functional > beautiful for MVS |
| Every GPU type | NVIDIA CUDA only covers 90% of market |
| Global scale | Single region initially |
| Enterprise features | Focus on individuals and small teams |
| Mobile apps | Desktop users are primary target |
| Advanced ML features | Script execution is sufficient |
| Complex pricing | Fixed rates simplify launch |

---

## MVS Development Phases

### Phase 1: Core Infrastructure
- [ ] Central Server: Node registry + heartbeat
- [ ] Central Server: Job queue + assignment
- [ ] Server Node: Registration + job execution
- [ ] Protocol: All gRPC services defined

### Phase 2: Payment Integration
- [ ] Escrow smart contract (Solana)
- [ ] Central Server: Payment service
- [ ] Engine: Wallet connection
- [ ] End-to-end payment flow

### Phase 3: Engine Client
- [ ] Job submission UI
- [ ] Job monitoring UI
- [ ] Wallet/balance management
- [ ] Results download

### Phase 4: Node Experience
- [ ] One-click installer
- [ ] Earnings dashboard
- [ ] Availability settings
- [ ] Auto-updates

### Phase 5: Testing & Hardening
- [ ] Internal dogfooding
- [ ] Security audit
- [ ] Performance testing
- [ ] Bug fixes

### Phase 6: Soft Launch
- [ ] Closed beta (invite only)
- [ ] Gather feedback
- [ ] Iterate on critical issues
- [ ] Prepare for public launch

---

## Risk Mitigation for MVS

| Risk | Mitigation in MVS |
|------|-------------------|
| Jobs fail frequently | Automatic retry, full refund on failure |
| Nodes go offline mid-job | Checkpoint support, job reassignment |
| Malicious nodes | Docker isolation, result verification |
| Malicious jobs | Resource limits, sandboxing |
| Payment disputes | Simple rule: failure = refund |
| Not enough nodes | Seed with own hardware initially |
| Not enough jobs | Subsidize early users |

---

## Post-MVS Roadmap Preview

After MVS validation, prioritize based on user feedback:

**Likely Next Features**:
1. Visual node editor (differentiation)
2. Reputation system (quality)
3. Multi-GPU distributed training (capability)
4. More GPU types - AMD, Intel (reach)
5. Advanced scheduling (efficiency)
6. Team features (enterprise path)

---

## Key Questions to Answer with MVS

1. **Will they pay?** → Track conversion and revenue
2. **Will they stay?** → Track retention and repeat usage
3. **Will they refer?** → Track organic growth
4. **What breaks?** → Track failures and support tickets
5. **What's missing?** → Track feature requests
6. **What's confusing?** → Track drop-off points

---

*Document created: 2025-11-25*
*Status: Draft - Ready for technical planning*
