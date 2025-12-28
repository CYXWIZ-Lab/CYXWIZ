# CyxWiz Target Market Analysis

## Executive Summary

CyxWiz operates a **two-sided marketplace** requiring both supply (GPU compute) and demand (ML training jobs). Our target market spans multiple segments across both sides, with a primary focus on **underserved developers** and **idle GPU owners**.

---

## Market Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CYXWIZ TARGET MARKET                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │   CYXWIZ    │                            │
│                      │ MARKETPLACE │                            │
│                      └──────┬──────┘                            │
│                             │                                   │
│          ┌──────────────────┼──────────────────┐                │
│          │                  │                  │                │
│          ▼                  ▼                  ▼                │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │   COMPUTE   │   │    MODEL    │   │   COMPUTE   │          │
│   │   BUYERS    │   │ MARKETPLACE │   │   SELLERS   │          │
│   │  (Demand)   │   │  (Creators  │   │  (Supply)   │          │
│   │             │   │  & Buyers)  │   │             │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│   WHO NEEDS          WHO MONETIZES       WHO PROVIDES           │
│   GPU COMPUTE        AI MODELS           GPU COMPUTE            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Primary Target Markets

### Market 1: Compute Buyers (Demand Side)

These are people and organizations who need GPU compute for ML training.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE BUYER SEGMENTS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TIER 1: PRIMARY TARGETS (Launch Focus)                        │
│   ───────────────────────────────────────                       │
│   • Independent ML developers                                   │
│   • Graduate students / PhD researchers                         │
│   • AI hobbyists and enthusiasts                                │
│   • Early-stage AI startups (pre-seed to seed)                  │
│                                                                 │
│   TIER 2: SECONDARY TARGETS (Growth Phase)                      │
│   ────────────────────────────────────────                      │
│   • Funded AI startups (Series A-B)                             │
│   • Small ML teams at tech companies                            │
│   • University research labs                                    │
│   • Bootcamp students and learners                              │
│                                                                 │
│   TIER 3: FUTURE TARGETS (Scale Phase)                          │
│   ──────────────────────────────────────                        │
│   • Enterprise ML teams                                         │
│   • Large research institutions                                 │
│   • Government agencies                                         │
│   • Fortune 500 AI initiatives                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Segment A: Independent ML Developers

```
┌─────────────────────────────────────────────────────────────────┐
│              SEGMENT: INDEPENDENT ML DEVELOPERS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • Solo developers building AI products                        │
│   • Freelancers offering ML services                            │
│   • Side-project builders                                       │
│   • Open source contributors                                    │
│   • Kaggle competitors                                          │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Age: 22-40                                                  │
│   • Location: Global (US, Europe, India, SEA heavy)             │
│   • Income: $50K-150K (or equivalent)                           │
│   • Technical: High (can write ML code)                         │
│   • Budget: $50-500/month personal spend                        │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • Global: 2-5 million                                         │
│   • Addressable: 500K-1M                                        │
│   • Year 1 target: 5,000-10,000                                 │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • Using Google Colab (free tier limits)                       │
│   • Occasional AWS/GCP (expensive, complex)                     │
│   • Lambda Labs, RunPod (if they know about them)               │
│   • Local GPU (limited by hardware)                             │
│   • Vast.ai (tech-savvy early adopters)                         │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "Colab keeps disconnecting"                                 │
│   • "AWS is way too expensive for personal projects"            │
│   • "I can't afford to experiment freely"                       │
│   • "Setting up cloud infrastructure is a nightmare"            │
│   • "I have ideas but no compute budget"                        │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • Affordable compute ($0.50-1/hr, not $3+)                    │
│   • Simple setup (minutes, not hours)                           │
│   • Pay only for what they use                                  │
│   • No commitments or minimums                                  │
│   • Reliable enough for real work                               │
│                                                                 │
│   HOW TO REACH THEM:                                            │
│   ──────────────────                                            │
│   • Reddit: r/MachineLearning, r/LocalLLaMA, r/learnmachinelearning │
│   • Twitter/X: ML community, AI influencers                     │
│   • YouTube: ML tutorials, tech reviewers                       │
│   • Discord: ML servers, Hugging Face                           │
│   • Hacker News: Technical launches                             │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Train ML models for 80% less than AWS.                       │
│    No setup. No commitment. Just compute."                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Indie Developer Persona**:

| Attribute | Details |
|-----------|---------|
| **Name** | Alex Chen |
| **Age** | 28 |
| **Role** | Freelance ML Engineer |
| **Location** | Austin, TX |
| **Income** | $95K/year |
| **GPU Budget** | $200/month |
| **Current Solution** | Colab Pro + occasional Lambda Labs |
| **Frustration** | "I have 10 project ideas but can only afford to try 2" |
| **Dream** | "Experiment freely like I work at Google" |
| **Quote** | "If compute was cheaper, I'd ship 5x more projects" |

---

### Segment B: Students & Academic Researchers

```
┌─────────────────────────────────────────────────────────────────┐
│              SEGMENT: STUDENTS & RESEARCHERS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • PhD students in ML/AI/CS                                    │
│   • Masters students                                            │
│   • Undergraduate researchers                                   │
│   • Postdocs and research scientists                            │
│   • Lab managers                                                │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Age: 20-35                                                  │
│   • Location: University hubs globally                          │
│   • Income: $25K-80K (stipend/salary)                           │
│   • Technical: Very high                                        │
│   • Budget: Grant-funded, $100-2,000/month                      │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • Global ML/AI grad students: 200K+                           │
│   • Researchers needing GPU: 100K+                              │
│   • Addressable: 50K-100K                                       │
│   • Year 1 target: 2,000-5,000                                  │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • University cluster (long queues, shared)                    │
│   • Cloud credits (limited, run out)                            │
│   • Google Colab (unreliable for long jobs)                     │
│   • Lab GPUs (limited, competitive access)                      │
│   • Begging for more compute                                    │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "The cluster queue is 2 weeks long"                         │
│   • "My cloud credits ran out mid-experiment"                   │
│   • "I can't reproduce results without more compute"            │
│   • "Competing with other students for GPU time"                │
│   • "Grant money doesn't stretch far enough"                    │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • Instant access (no queues)                                  │
│   • Affordable (grant money limited)                            │
│   • Long-running jobs supported                                 │
│   • Easy to cite/document for papers                            │
│   • Reproducibility features                                    │
│                                                                 │
│   HOW TO REACH THEM:                                            │
│   ──────────────────                                            │
│   • University partnerships                                     │
│   • Academic conferences (NeurIPS, ICML, etc.)                  │
│   • Twitter/X academic community                                │
│   • Lab-to-lab word of mouth                                    │
│   • Professor recommendations                                   │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Skip the cluster queue. Train your models now.               │
│    80% cheaper than cloud. Perfect for research."               │
│                                                                 │
│   SPECIAL OFFER:                                                │
│   ──────────────                                                │
│   • 50% discount for .edu emails                                │
│   • Free credits for published research                         │
│   • Academic citation support                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Researcher Persona**:

| Attribute | Details |
|-----------|---------|
| **Name** | Priya Sharma |
| **Age** | 26 |
| **Role** | PhD Candidate, Computer Vision |
| **Location** | Stanford University |
| **Income** | $45K stipend |
| **GPU Budget** | $500/month (grant-funded) |
| **Current Solution** | University cluster + Colab |
| **Frustration** | "I wait 2 weeks for cluster access, then my job fails" |
| **Dream** | "Run experiments whenever I have an idea" |
| **Quote** | "My research timeline is bottlenecked by compute access" |

---

### Segment C: AI Startups

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEGMENT: AI STARTUPS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • Pre-seed to Series B AI/ML startups                         │
│   • AI-first product companies                                  │
│   • ML infrastructure companies                                 │
│   • AI consulting/services firms                                │
│   • AI research startups                                        │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Company size: 2-100 employees                               │
│   • Funding: $0-50M raised                                      │
│   • Location: SF, NYC, London, Berlin, Tel Aviv, Bangalore      │
│   • Technical: Founding team usually ML experts                 │
│   • Budget: $1,000-100,000/month on compute                     │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • Global AI startups: 50,000+                                 │
│   • Funded AI startups: 10,000+                                 │
│   • Addressable: 5,000-10,000                                   │
│   • Year 1 target: 100-500                                      │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • AWS/GCP (most common, expensive)                            │
│   • Lambda Labs (popular alternative)                           │
│   • CoreWeave (for scale)                                       │
│   • Reserved instances (locked in)                              │
│   • On-prem (some, for cost control)                            │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "GPU costs are 30% of our burn rate"                        │
│   • "We can't iterate as fast as we need to"                    │
│   • "Cloud contracts lock us in"                                │
│   • "Scaling up doubles our costs"                              │
│   • "Investors question our unit economics"                     │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • Significant cost reduction (50%+)                           │
│   • Reliability (can't afford failures)                         │
│   • Scale on demand                                             │
│   • No lock-in (flexibility)                                    │
│   • Team features (collaboration)                               │
│                                                                 │
│   HOW TO REACH THEM:                                            │
│   ──────────────────                                            │
│   • VC/accelerator introductions                                │
│   • Startup community (YC, Techstars alumni)                    │
│   • LinkedIn outreach to founders/CTOs                          │
│   • AI/ML conferences                                           │
│   • Content marketing (case studies)                            │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Cut GPU costs 70% without sacrificing reliability.           │
│    Extend your runway. Impress your investors."                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Startup Persona**:

| Attribute | Details |
|-----------|---------|
| **Name** | Marcus Johnson |
| **Age** | 34 |
| **Role** | CTO & Co-founder |
| **Company** | AI SaaS Startup (Series A) |
| **Location** | San Francisco |
| **Team Size** | 15 (5 ML engineers) |
| **GPU Budget** | $25,000/month |
| **Current Solution** | AWS + Lambda Labs |
| **Frustration** | "Compute is our #2 expense after salaries" |
| **Dream** | "Same capabilities at half the cost" |
| **Quote** | "If we cut compute costs in half, we extend runway by 8 months" |

---

### Segment D: AI Hobbyists & Enthusiasts

```
┌─────────────────────────────────────────────────────────────────┐
│                SEGMENT: HOBBYISTS & ENTHUSIASTS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • AI curious professionals (non-ML day job)                   │
│   • Weekend warriors building AI projects                       │
│   • Stable Diffusion / local LLM enthusiasts                    │
│   • Tech-savvy generalists exploring AI                         │
│   • Career transitioners into ML                                │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Age: 25-50                                                  │
│   • Location: Global, tech-savvy regions                        │
│   • Income: $60K-200K (day job, not AI)                         │
│   • Technical: Medium-high (can follow tutorials)               │
│   • Budget: $20-200/month (hobby budget)                        │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • AI hobbyists globally: 5-10 million                         │
│   • Those who'd pay for compute: 1-2 million                    │
│   • Addressable: 500K                                           │
│   • Year 1 target: 3,000-5,000                                  │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • Local GPU (if they have one)                                │
│   • Google Colab free tier                                      │
│   • Not willing to pay AWS prices                               │
│   • Frustrated by limitations                                   │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "My laptop can't handle serious training"                   │
│   • "Colab disconnects before my job finishes"                  │
│   • "I can't justify $50/month just to play around"             │
│   • "I want to try fine-tuning but don't have hardware"         │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Powerful GPU compute for the price of Netflix.               │
│    Train real models. No expensive hardware needed."            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Market 2: Compute Sellers (Supply Side)

These are people and organizations who have idle GPUs and want to monetize them.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE SELLER SEGMENTS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TIER 1: PRIMARY TARGETS (Launch Focus)                        │
│   ───────────────────────────────────────                       │
│   • Gamers with high-end GPUs                                   │
│   • Ex-cryptocurrency miners                                    │
│   • Tech enthusiasts with spare hardware                        │
│                                                                 │
│   TIER 2: SECONDARY TARGETS (Growth Phase)                      │
│   ────────────────────────────────────────                      │
│   • Small businesses with GPU workstations                      │
│   • Design/creative agencies                                    │
│   • Small data centers                                          │
│                                                                 │
│   TIER 3: FUTURE TARGETS (Scale Phase)                          │
│   ──────────────────────────────────────                        │
│   • Large data centers                                          │
│   • Enterprise IT departments                                   │
│   • Cloud providers (excess capacity)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Segment E: Gamers with High-End GPUs

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEGMENT: GAMERS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • PC gaming enthusiasts                                       │
│   • Owners of RTX 30/40 series GPUs                             │
│   • People who upgraded for gaming but use < 10%                │
│   • Tech-savvy, comfortable with software                       │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Age: 18-40                                                  │
│   • Location: US, Europe, Asia (developed markets)              │
│   • Income: $40K-150K                                           │
│   • GPU: RTX 3070/3080/3090/4080/4090                           │
│   • GPU utilization: <10% average                               │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • RTX 30/40 series owners: 50+ million                        │
│   • High-end (3080+): 15-20 million                             │
│   • Would consider monetizing: 5-10%                            │
│   • Addressable: 1-2 million                                    │
│   • Year 1 target: 5,000-10,000                                 │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • GPU sits idle 90% of time                                   │
│   • Maybe tried mining (unprofitable now)                       │
│   • Aware hardware depreciates                                  │
│   • Open to passive income ideas                                │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "I paid $1,500 for this GPU and it just sits there"         │
│   • "Mining doesn't work anymore"                               │
│   • "Hardware is depreciating while I sleep"                    │
│   • "I want to make money but not a second job"                 │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • Passive income ($100-300/month)                             │
│   • Easy setup (not complicated)                                │
│   • Doesn't interfere with gaming                               │
│   • Trustworthy (no sketchy software)                           │
│   • Control over when it runs                                   │
│                                                                 │
│   HOW TO REACH THEM:                                            │
│   ──────────────────                                            │
│   • Reddit: r/pcmasterrace, r/nvidia, r/AMD                     │
│   • YouTube: Tech/gaming channels                               │
│   • Discord: Gaming communities                                 │
│   • Twitter: Gaming/tech accounts                               │
│   • PC hardware forums                                          │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Your RTX 4090 can earn $150/month while you sleep.           │
│    One-click setup. You control when it runs."                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Gamer Persona**:

| Attribute | Details |
|-----------|---------|
| **Name** | Jake Williams |
| **Age** | 27 |
| **Role** | Software Developer (not ML) |
| **Location** | Seattle, WA |
| **Income** | $120K |
| **GPU** | RTX 4080 ($1,200) |
| **Usage** | 3 hours/day gaming, 90% idle |
| **Frustration** | "My GPU cost more than my rent and mostly does nothing" |
| **Dream** | "My hardware pays for itself" |
| **Quote** | "If I could make $100/month with zero effort, I'm in" |

---

### Segment F: Ex-Cryptocurrency Miners

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEGMENT: EX-MINERS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • Former Ethereum/crypto GPU miners                           │
│   • Own multiple GPUs (mining rigs)                             │
│   • Hardware sitting idle since Ethereum merge                  │
│   • Looking for new revenue streams                             │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Age: 22-45                                                  │
│   • Location: Global (energy cost aware)                        │
│   • Investment: $5K-100K in hardware                            │
│   • GPUs: 2-50+ cards (RTX 3060-3090 common)                    │
│   • Technical: High (ran mining operations)                     │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • Ex-GPU miners globally: 1-2 million                         │
│   • With significant hardware: 500K                             │
│   • Addressable: 100K-200K                                      │
│   • Year 1 target: 1,000-3,000                                  │
│                                                                 │
│   CURRENT BEHAVIOR:                                             │
│   ─────────────────                                             │
│   • Hardware sitting idle (unprofitable to mine)                │
│   • Some sold GPUs (many still holding)                         │
│   • Tried altcoin mining (marginal)                             │
│   • Looking for alternatives                                    │
│   • Know about Vast.ai, Render, etc.                            │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "I have $30K in GPUs earning nothing"                       │
│   • "Mining is dead but I don't want to sell at a loss"         │
│   • "Power costs more than I earn"                              │
│   • "There has to be something better"                          │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • ROI on existing hardware                                    │
│   • Better returns than mining                                  │
│   • Stable, predictable income                                  │
│   • Easy to manage multiple GPUs                                │
│   • Worth the electricity cost                                  │
│                                                                 │
│   HOW TO REACH THEM:                                            │
│   ──────────────────                                            │
│   • Reddit: r/gpumining, r/EtherMining                          │
│   • Discord: Mining communities                                 │
│   • Twitter: Crypto/mining accounts                             │
│   • Mining forums and Telegram groups                           │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Mining is dead. ML compute pays better.                      │
│    Your rigs can earn $50-100/GPU/month—more than mining."      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ex-Miner Persona**:

| Attribute | Details |
|-----------|---------|
| **Name** | Ryan Martinez |
| **Age** | 35 |
| **Role** | IT Manager |
| **Location** | Denver, CO |
| **Investment** | $25K in mining rig (8x RTX 3080) |
| **Current Revenue** | $0 (mining unprofitable) |
| **Frustration** | "My rig was making $2K/month, now it's a paperweight" |
| **Dream** | "Recover my investment, then profit" |
| **Quote** | "I'll try anything that's better than letting it collect dust" |

---

### Segment G: Small Businesses with GPUs

```
┌─────────────────────────────────────────────────────────────────┐
│               SEGMENT: SMALL BUSINESSES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WHO THEY ARE:                                                 │
│   ─────────────                                                 │
│   • Design/creative agencies                                    │
│   • Video production companies                                  │
│   • Architecture/engineering firms                              │
│   • Small game development studios                              │
│   • AI/ML consulting firms                                      │
│                                                                 │
│   DEMOGRAPHICS:                                                 │
│   ─────────────                                                 │
│   • Company size: 5-50 employees                                │
│   • GPUs: 5-50 workstations with GPUs                           │
│   • Utilization: 20-40% during business hours                   │
│   • Idle time: Nights, weekends = 70% of time                   │
│                                                                 │
│   SIZE ESTIMATE:                                                │
│   ──────────────                                                │
│   • Small businesses with GPU workstations: 500K+               │
│   • Interested in monetization: 10-20%                          │
│   • Addressable: 50K-100K                                       │
│   • Year 1 target: 100-500 businesses                           │
│                                                                 │
│   PAIN POINTS:                                                  │
│   ────────────                                                  │
│   • "These workstations cost $5K each and sit idle at night"    │
│   • "IT is a cost center, not a profit center"                  │
│   • "We have spare capacity but no way to use it"               │
│                                                                 │
│   WHAT THEY WANT:                                               │
│   ───────────────                                               │
│   • Turn IT costs into revenue                                  │
│   • No disruption to employees                                  │
│   • Enterprise-grade security                                   │
│   • Easy management across fleet                                │
│   • Meaningful revenue ($1,000+/month)                          │
│                                                                 │
│   MESSAGING:                                                    │
│   ──────────                                                    │
│   "Your workstations earn money while everyone's asleep.        │
│    Turn IT from cost center to profit center."                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Market 3: Model Marketplace

```
┌─────────────────────────────────────────────────────────────────┐
│                MODEL MARKETPLACE SEGMENTS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   MODEL CREATORS:                                               │
│   ───────────────                                               │
│   • ML engineers with trained models                            │
│   • Researchers with specialized models                         │
│   • AI studios producing models                                 │
│   • Open source contributors wanting income                     │
│                                                                 │
│   MODEL BUYERS:                                                 │
│   ──────────────                                                │
│   • Developers needing pre-trained models                       │
│   • Startups wanting to save training time                      │
│   • Enterprises needing specialized models                      │
│   • Agencies building client solutions                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Target Market Summary

### Primary Targets (Year 1 Focus)

| Segment | Side | Size | Priority | Why |
|---------|------|------|----------|-----|
| Indie ML Developers | Demand | 500K-1M | #1 | Large, underserved, vocal |
| Gamers with GPUs | Supply | 1-2M | #1 | Easy to reach, large base |
| Students/Researchers | Demand | 50K-100K | #2 | Budget-constrained, influential |
| Ex-Miners | Supply | 100K-200K | #2 | Have hardware, motivated |
| AI Hobbyists | Demand | 500K | #3 | Volume, low touch |

### Secondary Targets (Year 2-3)

| Segment | Side | Size | Priority | Why |
|---------|------|------|----------|-----|
| AI Startups | Demand | 5K-10K | #1 | High value, scalable |
| Small Businesses | Supply | 50K-100K | #2 | Multiple GPUs per customer |
| Research Institutions | Demand | 5K | #3 | Large budgets, validation |

### Future Targets (Year 3+)

| Segment | Side | Size | Priority | Why |
|---------|------|------|----------|-----|
| Enterprises | Demand | 10K | #1 | Highest value |
| Data Centers | Supply | 1K | #1 | Massive scale |

---

## Market Sizing

### Total Addressable Market (TAM)

```
GLOBAL GPU COMPUTE MARKET
─────────────────────────

Cloud GPU Market (2024): $15B
Cloud GPU Market (2028): $50B+
CAGR: 35%

ML Training specifically: $8B (2024) → $25B (2028)
```

### Serviceable Addressable Market (SAM)

```
SEGMENTS WE CAN SERVE
─────────────────────

Indie developers:        $500M (spend constrained by cost)
Students/researchers:    $300M (grant-limited)
AI startups:             $2B
Hobbyists:               $200M
Small enterprises:       $1B
────────────────────────────────
Total SAM:               ~$4B
```

### Serviceable Obtainable Market (SOM)

```
REALISTIC YEAR 1-5 TARGETS
──────────────────────────

Year 1: $500K   (0.01% of SAM)
Year 2: $3M     (0.075% of SAM)
Year 3: $15M    (0.4% of SAM)
Year 5: $60M    (1.5% of SAM)
```

---

## Geographic Focus

### Phase 1: English-Speaking Markets

| Region | Priority | Why |
|--------|----------|-----|
| United States | #1 | Largest market, early adopters |
| United Kingdom | #2 | English-speaking, tech-savvy |
| Canada | #3 | Similar to US |
| Australia | #4 | English-speaking, growing tech |

### Phase 2: Europe & India

| Region | Priority | Why |
|--------|----------|-----|
| Germany | #1 | Large tech market |
| India | #1 | Huge developer population |
| Netherlands | #2 | Tech hub |
| France | #3 | Growing AI ecosystem |

### Phase 3: Asia-Pacific

| Region | Priority | Why |
|--------|----------|-----|
| Singapore | #1 | AI hub, English |
| Japan | #2 | Tech-savvy, gamers |
| South Korea | #3 | Gaming culture |

---

## Key Insights

### Why These Markets?

```
1. UNDERSERVED BY INCUMBENTS
   • AWS/GCP too expensive for individuals
   • No good options between "free but limited" and "enterprise"
   • Vast.ai exists but has UX/trust issues

2. MOTIVATED TO SWITCH
   • Clear cost savings (80%)
   • Pain is acute (budget constraints)
   • Alternatives are known to be lacking

3. REACHABLE
   • Concentrated in online communities
   • Respond to word-of-mouth
   • Technical enough to try new tools

4. GROWING
   • AI interest exploding
   • More people learning ML
   • More GPUs in consumer hands
```

### Why NOT Enterprise First?

```
ENTERPRISE CHALLENGES:
• Long sales cycles (6-18 months)
• Requires sales team
• Compliance/security requirements
• Distracts from product
• Can't validate product quickly

START B2C, ADD B2B LATER:
• Validate product with fast feedback
• Build case studies
• Prove reliability at scale
• Then sell to enterprise with proof
```

---

*Document created: 2025-11-25*
*Status: Ready for execution*
