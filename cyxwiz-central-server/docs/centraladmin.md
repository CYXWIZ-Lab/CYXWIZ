# Central Server Admin Integration Design Document

**Version:** 1.0
**Date:** December 11, 2025
**Status:** Design Phase
**Authors:** CyxWiz Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture](#2-current-architecture)
3. [Target Architecture](#3-target-architecture)
4. [API Gateway Design](#4-api-gateway-design)
5. [REST API Specification](#5-rest-api-specification)
6. [Admin Dashboard Pages](#6-admin-dashboard-pages)
7. [Database Schema Alignment](#7-database-schema-alignment)
8. [Real-time Features](#8-real-time-features)
9. [Security & Authentication](#9-security--authentication)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Testing Strategy](#11-testing-strategy)
12. [Deployment Considerations](#12-deployment-considerations)

---

## 1. Executive Summary

### 1.1 Goal

Integrate the Central Server's monitoring and management capabilities into the existing Web Admin Dashboard, providing a unified administration interface for the CyxWiz platform.

### 1.2 Decision

**Chosen Approach:** Web Admin Integration (Option B)

**Rationale:**
- Admin dashboard already has 90% of required features (devices, compute, transactions)
- Single source of truth for platform administration
- Browser-based access without desktop installation
- Consistent UX across all admin functions
- Reduced maintenance burden (one codebase)

### 1.3 Scope

| In Scope | Out of Scope |
|----------|--------------|
| System health monitoring | Desktop GUI for Central Server |
| Blockchain wallet/transactions view | TUI replacement (keep for ops) |
| Scheduler metrics & controls | User-facing dashboard changes |
| Real-time node status | Mobile admin app |
| Infrastructure alerts | |

### 1.4 Key Stakeholders

- **Backend Team:** Central Server REST API development
- **Frontend Team:** Admin dashboard new pages
- **DevOps:** Deployment and networking
- **Security:** Authentication and authorization

---

## 1.5 Implementation Status

> **Status:** Phase 1 Complete (December 2025)

### Implemented REST API Endpoints

| Endpoint | Status | Description |
|----------|--------|-------------|
| `GET /api/v1/health` | ✅ Done | System health with service status |
| `GET /api/v1/infrastructure/stats` | ✅ Done | Network-wide statistics |
| `GET /api/v1/nodes/live` | ✅ Done | Paginated live nodes with filters |
| `GET /api/v1/nodes/stats` | ✅ Done | Node counts by status |
| `GET /api/v1/nodes/:id/metrics` | ✅ Done | Node metrics history |
| `GET /api/v1/scheduler/status` | ✅ Done | Scheduler running status |
| `GET /api/v1/scheduler/queue` | ✅ Done | Jobs in queue |
| `GET /api/v1/scheduler/assignments` | ✅ Done | Recent job assignments |
| `GET /api/v1/scheduler/throughput` | ✅ Done | Processing rate stats |
| `GET /api/v1/blockchain/wallet` | ✅ Done | Platform wallet info |
| `GET /api/v1/blockchain/transactions` | ✅ Done | Transaction history |
| `GET /api/v1/blockchain/escrows` | ✅ Done | Active escrows |
| `GET /api/v1/blockchain/stats` | ✅ Done | Blockchain statistics |

### Implemented Admin Pages

| Page | URL | Description |
|------|-----|-------------|
| Infrastructure | `/infrastructure` | System health, service status, node/job stats |
| Network Nodes | `/network-nodes` | Live nodes with filtering, search, pagination |
| Scheduler | `/scheduler` | Queue depth, assignments, throughput metrics |
| Blockchain | `/blockchain` | Wallet info, transactions, escrows, payment stats |

### How to Run

#### 1. Start the Central Server (REST API on port 8080)

```bash
# Navigate to Central Server directory
cd D:\Dev\CyxWiz_Claude\cyxwiz-central-server

# Run in default mode (gRPC + REST)
cargo run

# Or run in TUI mode (still serves REST API)
cargo run -- --tui
```

The server will start:
- **REST API**: http://localhost:8080
- **gRPC**: localhost:50051

#### 2. Start the Admin Website (port 3005)

```bash
# Navigate to admin app
cd D:\Dev\cyxwiz_web\apps\admin

# Start development server
npm run dev
```

The admin will be available at: http://localhost:3005

#### 3. Access the New Pages

Once both servers are running:

- **Infrastructure**: http://localhost:3005/infrastructure
- **Network Nodes**: http://localhost:3005/network-nodes
- **Scheduler**: http://localhost:3005/scheduler
- **Blockchain**: http://localhost:3005/blockchain

### Environment Configuration

Add to `apps/admin/.env.local`:

```env
CENTRAL_SERVER_URL=http://localhost:8080
```

### File Locations

**Central Server (Rust):**
```
cyxwiz-central-server/src/api/rest/
├── mod.rs                    # Router + CORS
├── dashboard.rs              # Legacy endpoints
└── v1/
    ├── mod.rs                # V1 router composition
    ├── infrastructure.rs     # Health, stats endpoints
    ├── nodes.rs              # Live nodes, metrics
    ├── scheduler.rs          # Queue, assignments, throughput
    └── blockchain.rs         # Wallet, transactions, escrows
```

**Admin Website (Next.js):**
```
apps/admin/src/
├── lib/
│   ├── central-server.ts           # API client
│   └── actions/
│       ├── infrastructure.ts       # Infrastructure actions
│       ├── nodes-central.ts        # Nodes actions
│       ├── scheduler.ts            # Scheduler actions
│       └── blockchain.ts           # Blockchain actions
├── app/(dashboard)/
│   ├── infrastructure/page.tsx     # Infrastructure page
│   ├── network-nodes/
│   │   ├── page.tsx                # Nodes page
│   │   ├── nodes-search.tsx        # Search component
│   │   └── nodes-table.tsx         # Table component
│   ├── scheduler/page.tsx          # Scheduler page
│   └── blockchain/page.tsx         # Blockchain page
└── components/layout/
    └── admin-sidebar.tsx           # Navigation (updated)
```

---

## 2. Current Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CURRENT STATE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐         ┌──────────────────┐
    │   CyxWiz Engine  │         │  Server Nodes    │
    │   (Desktop App)  │         │  (Compute Workers)│
    └────────┬─────────┘         └────────┬─────────┘
             │ gRPC                        │ gRPC
             └────────────┬────────────────┘
                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     CENTRAL SERVER (Rust)                                │
    │                 D:\Dev\CyxWiz_Claude\cyxwiz-central-server               │
    │  ┌────────────┬────────────┬────────────┬────────────┬────────────┐    │
    │  │  gRPC API  │  Scheduler │ Node Reg   │  Payments  │  Database  │    │
    │  │  :50051    │            │            │  (Solana)  │  (SQLite)  │    │
    │  └────────────┴────────────┴────────────┴────────────┴────────────┘    │
    │                                                                          │
    │  Interfaces: CLI mode, TUI mode (--tui)                                 │
    │  NO REST API currently exposed for admin                                │
    └─────────────────────────────────────────────────────────────────────────┘
                          ║
                          ║ (No connection currently)
                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      WEB PLATFORM (cyxwiz_web)                           │
    │                      D:\Dev\cyxwiz_web                                   │
    │                                                                          │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
    │  │   Web App       │    │   Admin Panel   │    │    API Server   │     │
    │  │   (Next.js)     │    │   (Next.js)     │    │    (Rust/Axum)  │     │
    │  │   /apps/web     │    │   /apps/admin   │    │    /apps/api    │     │
    │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
    │                                │                        │               │
    │                                │ Server Actions         │               │
    │                                └────────────────────────┘               │
    │                                                                          │
    │  Admin has: /devices, /compute, /users, /transactions, /logs, /system   │
    │  Missing: Infrastructure health, Blockchain view, Scheduler internals   │
    └─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Current Data Flow

| Component | Data Source | Update Frequency |
|-----------|-------------|------------------|
| Admin /devices | Web API → PostgreSQL | On request |
| Admin /compute | Web API → PostgreSQL | On request |
| Central Server TUI | Direct DB + Solana RPC | 1 second |
| Central Server gRPC | Direct DB | Real-time |

### 2.3 Gap Analysis

**What Admin Dashboard is Missing:**

| Feature | Currently In | Should Be In |
|---------|--------------|--------------|
| System health (DB, Redis, Solana) | TUI only | Admin |
| Wallet balance & info | TUI only | Admin |
| Blockchain transactions | TUI only | Admin |
| Scheduler queue depth | Internal only | Admin |
| Job assignment details | Internal only | Admin |
| Node heartbeat status | gRPC only | Admin |
| Real-time metrics | TUI only | Admin |

---

## 3. Target Architecture

### 3.1 Integrated Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TARGET STATE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐         ┌──────────────────┐
    │   CyxWiz Engine  │         │  Server Nodes    │
    │   (Desktop App)  │         │  (Compute Workers)│
    └────────┬─────────┘         └────────┬─────────┘
             │ gRPC                        │ gRPC
             └────────────┬────────────────┘
                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     CENTRAL SERVER (Rust)                                │
    │  ┌────────────┬────────────┬────────────┬────────────┬────────────┐    │
    │  │  gRPC API  │  REST API  │  Scheduler │  Payments  │  Database  │    │
    │  │  :50051    │  :8080 NEW │            │  (Solana)  │            │    │
    │  └────────────┴─────┬──────┴────────────┴────────────┴────────────┘    │
    │                     │                                                    │
    │  TUI mode: Still available for ops/debugging (--tui flag)               │
    └─────────────────────┼────────────────────────────────────────────────────┘
                          │
                          │ REST API (internal network)
                          │ /api/v1/admin/*
                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      WEB PLATFORM (cyxwiz_web)                           │
    │                                                                          │
    │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
    │  │   Web App       │    │   Admin Panel   │    │    API Server   │     │
    │  │   (Next.js)     │    │   (Next.js)     │    │    (Rust/Axum)  │     │
    │  └─────────────────┘    └────────┬────────┘    └────────┬────────┘     │
    │                                  │                      │               │
    │                    Server Actions│    Proxy to Central  │               │
    │                                  └──────────────────────┘               │
    │                                                                          │
    │  NEW Admin Pages:                                                        │
    │  ├── /infrastructure  (System health, service status)                   │
    │  ├── /blockchain      (Wallet, transactions, escrow)                    │
    │  ├── /scheduler       (Queue, assignments, throughput)                  │
    │  └── /nodes/live      (Real-time heartbeat, geographic view)           │
    └─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Communication Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMUNICATION PATTERNS                                │
└─────────────────────────────────────────────────────────────────────────────┘

Pattern A: Direct REST (Recommended for MVP)
─────────────────────────────────────────────
Admin Dashboard ──HTTP──▶ Central Server REST API ──▶ Response
                         (via internal network)

Pattern B: Web API Gateway (Recommended for Production)
───────────────────────────────────────────────────────
Admin Dashboard ──HTTP──▶ Web API ──gRPC──▶ Central Server ──▶ Response
                         (gateway)

Pattern C: WebSocket for Real-time
──────────────────────────────────
Admin Dashboard ◀──WS──▶ Central Server ──▶ Push updates
                         (health, metrics, alerts)
```

### 3.3 Chosen Pattern: Hybrid Approach

**Phase 1 (MVP):** Pattern A - Direct REST
- Central Server exposes REST API on :8080
- Admin dashboard calls directly (internal network)
- Simpler to implement, faster to deploy

**Phase 2 (Production):** Pattern B - Gateway
- Web API acts as gateway/proxy
- Single entry point for all admin API calls
- Better security (Central Server not exposed)
- Rate limiting, caching at gateway level

**Phase 3 (Enhancement):** Pattern C - Real-time
- WebSocket connection for live updates
- Push notifications for alerts
- Real-time dashboard metrics

---

## 4. API Gateway Design

### 4.1 Central Server REST API Module

**Location:** `cyxwiz-central-server/src/api/rest/`

```
src/api/rest/
├── mod.rs              # Module exports
├── router.rs           # Axum router setup
├── handlers/
│   ├── mod.rs
│   ├── health.rs       # System health endpoints
│   ├── blockchain.rs   # Wallet & transaction endpoints
│   ├── scheduler.rs    # Scheduler metrics endpoints
│   └── nodes.rs        # Node status endpoints
├── middleware/
│   ├── mod.rs
│   ├── auth.rs         # Admin authentication
│   └── cors.rs         # CORS configuration
└── models/
    ├── mod.rs
    ├── responses.rs    # API response types
    └── requests.rs     # API request types
```

### 4.2 REST API Base URL

```
Production:  https://api.cyxwiz.com/central/v1
Development: http://localhost:8080/api/v1
```

### 4.3 Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Admin     │     │   Admin     │     │  Web API    │     │  Central    │
│   Browser   │     │  Dashboard  │     │  (Gateway)  │     │   Server    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  1. Login         │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │  2. Verify admin  │                   │
       │                   │──────────────────▶│                   │
       │                   │  3. JWT token     │                   │
       │◀──────────────────│◀──────────────────│                   │
       │                   │                   │                   │
       │  4. API request   │                   │                   │
       │  + JWT            │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │  5. Forward +     │                   │
       │                   │     service token │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │  6. Verify &      │
       │                   │                   │     process       │
       │                   │                   │──────────────────▶│
       │                   │                   │  7. Response      │
       │◀──────────────────│◀──────────────────│◀──────────────────│
       │                   │                   │                   │
```

---

## 5. REST API Specification

### 5.1 Health & Infrastructure

#### GET /api/v1/health
System health check for all services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-11T08:50:41Z",
  "uptime_seconds": 86400,
  "version": "0.1.0",
  "services": {
    "database": {
      "status": "healthy",
      "latency_ms": 2,
      "type": "postgresql"
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 1,
      "queue_length": 5
    },
    "solana_rpc": {
      "status": "healthy",
      "latency_ms": 200,
      "network": "devnet",
      "endpoint": "api.devnet.solana.com"
    },
    "grpc_server": {
      "status": "running",
      "port": 50051,
      "active_connections": 12
    }
  }
}
```

#### GET /api/v1/metrics
Prometheus-compatible metrics endpoint.

**Response:** Plain text metrics format

#### GET /api/v1/infrastructure/stats
Detailed infrastructure statistics.

**Response:**
```json
{
  "scheduler": {
    "queue_depth": 15,
    "jobs_processed_24h": 1250,
    "avg_processing_time_ms": 45000,
    "active_workers": 8
  },
  "network": {
    "total_nodes": 150,
    "online_nodes": 142,
    "total_gpu_count": 320,
    "total_vram_gb": 2560
  },
  "storage": {
    "database_size_mb": 512,
    "models_storage_gb": 45.2
  }
}
```

### 5.2 Blockchain & Payments

#### GET /api/v1/blockchain/wallet
Platform wallet information.

**Response:**
```json
{
  "payer_pubkey": "4Y5HWB9W9SELq3Yoyf7mK7KF5kTbuaGxd2BMvn3AyAG8",
  "balance_lamports": 11315500000,
  "balance_sol": 11.3155,
  "program_id": "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA",
  "network": "devnet",
  "rpc_endpoint": "https://api.devnet.solana.com"
}
```

#### GET /api/v1/blockchain/transactions
Recent blockchain transactions.

**Query Parameters:**
- `limit` (int, default: 20, max: 100)
- `offset` (int, default: 0)
- `status` (string: all|pending|completed|failed)
- `type` (string: all|escrow|release|refund)

**Response:**
```json
{
  "transactions": [
    {
      "id": "uuid",
      "signature": "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9g...",
      "type": "escrow_lock",
      "job_id": "uuid",
      "amount_lamports": 100000000,
      "amount_sol": 0.1,
      "status": "confirmed",
      "created_at": "2025-12-11T08:30:00Z",
      "confirmed_at": "2025-12-11T08:30:15Z",
      "from_pubkey": "4Y5HWB9W...",
      "to_pubkey": "DefY4GG3..."
    }
  ],
  "total": 156,
  "page": 1,
  "total_pages": 8
}
```

#### GET /api/v1/blockchain/escrows
Active escrow accounts.

**Response:**
```json
{
  "escrows": [
    {
      "job_id": "uuid",
      "escrow_pubkey": "...",
      "amount_lamports": 500000000,
      "status": "locked",
      "client_pubkey": "...",
      "node_pubkey": "...",
      "created_at": "2025-12-11T08:00:00Z",
      "expires_at": "2025-12-12T08:00:00Z"
    }
  ],
  "total_locked_sol": 125.5
}
```

#### GET /api/v1/blockchain/stats
Payment statistics for dashboard.

**Response:**
```json
{
  "total_volume_24h_sol": 1234.56,
  "total_volume_7d_sol": 8500.00,
  "total_transactions_24h": 450,
  "platform_fees_24h_sol": 123.45,
  "node_payouts_24h_sol": 1111.11,
  "avg_job_cost_sol": 2.74,
  "pending_escrows_count": 25,
  "pending_escrows_value_sol": 68.75
}
```

### 5.3 Scheduler

#### GET /api/v1/scheduler/status
Current scheduler status.

**Response:**
```json
{
  "status": "running",
  "queue_depth": 15,
  "processing_rate_per_minute": 12.5,
  "last_assignment_at": "2025-12-11T08:49:30Z",
  "config": {
    "max_retries": 3,
    "assignment_timeout_seconds": 300,
    "heartbeat_interval_seconds": 30
  }
}
```

#### GET /api/v1/scheduler/queue
Jobs currently in queue.

**Response:**
```json
{
  "queued_jobs": [
    {
      "job_id": "uuid",
      "priority": "high",
      "type": "training",
      "queued_at": "2025-12-11T08:45:00Z",
      "wait_time_seconds": 300,
      "requirements": {
        "gpu_count": 2,
        "vram_gb": 16,
        "device_type": "cuda"
      }
    }
  ],
  "total_queued": 15,
  "by_priority": {
    "critical": 2,
    "high": 5,
    "normal": 6,
    "low": 2
  },
  "by_type": {
    "training": 10,
    "inference": 3,
    "fine_tuning": 2
  }
}
```

#### GET /api/v1/scheduler/assignments
Current job-to-node assignments.

**Response:**
```json
{
  "assignments": [
    {
      "job_id": "uuid",
      "node_id": "uuid",
      "node_name": "gpu-node-42",
      "assigned_at": "2025-12-11T08:30:00Z",
      "status": "running",
      "progress_percent": 45,
      "eta_seconds": 1800
    }
  ],
  "total_active": 8
}
```

#### GET /api/v1/scheduler/throughput
Historical throughput data for graphs.

**Query Parameters:**
- `period` (string: hour|day|week|month)
- `interval` (string: minute|hour|day)

**Response:**
```json
{
  "period": "day",
  "interval": "hour",
  "data_points": [
    {
      "timestamp": "2025-12-11T00:00:00Z",
      "jobs_completed": 45,
      "jobs_failed": 2,
      "avg_duration_seconds": 3600
    }
  ]
}
```

### 5.4 Nodes (Real-time)

#### GET /api/v1/nodes/live
Live node status with recent heartbeats.

**Response:**
```json
{
  "nodes": [
    {
      "id": "uuid",
      "name": "gpu-node-42",
      "status": "online",
      "last_heartbeat": "2025-12-11T08:50:30Z",
      "heartbeat_age_seconds": 10,
      "current_job_id": "uuid",
      "utilization": {
        "cpu_percent": 45,
        "gpu_percent": 85,
        "memory_percent": 60,
        "vram_percent": 72
      },
      "specs": {
        "gpu_model": "RTX 4090",
        "gpu_count": 2,
        "vram_gb": 48,
        "cpu_cores": 16
      },
      "location": {
        "country": "US",
        "region": "us-west-2"
      }
    }
  ],
  "summary": {
    "online": 142,
    "offline": 8,
    "busy": 95,
    "idle": 47
  }
}
```

#### GET /api/v1/nodes/{node_id}/metrics
Detailed metrics for a specific node.

**Response:**
```json
{
  "node_id": "uuid",
  "current": {
    "cpu_percent": 45,
    "gpu_percent": 85,
    "memory_used_gb": 24,
    "memory_total_gb": 64,
    "vram_used_gb": 34,
    "vram_total_gb": 48,
    "network_rx_mbps": 125,
    "network_tx_mbps": 45
  },
  "history_1h": [
    {
      "timestamp": "2025-12-11T08:00:00Z",
      "cpu_percent": 40,
      "gpu_percent": 80
    }
  ]
}
```

### 5.5 Alerts & Events

#### GET /api/v1/alerts
Active system alerts.

**Response:**
```json
{
  "alerts": [
    {
      "id": "uuid",
      "severity": "warning",
      "type": "low_balance",
      "message": "Platform wallet balance below 10 SOL",
      "created_at": "2025-12-11T08:00:00Z",
      "acknowledged": false,
      "metadata": {
        "current_balance_sol": 8.5,
        "threshold_sol": 10
      }
    }
  ],
  "counts": {
    "critical": 0,
    "warning": 2,
    "info": 5
  }
}
```

#### POST /api/v1/alerts/{alert_id}/acknowledge
Acknowledge an alert.

**Response:**
```json
{
  "success": true,
  "acknowledged_by": "admin@cyxwiz.com",
  "acknowledged_at": "2025-12-11T08:51:00Z"
}
```

---

## 6. Admin Dashboard Pages

### 6.1 New Page Structure

```
apps/admin/src/app/(dashboard)/
├── infrastructure/           # NEW
│   ├── page.tsx             # System health overview
│   ├── health-cards.tsx     # Service health cards
│   ├── metrics-chart.tsx    # Real-time metrics graph
│   └── alerts-panel.tsx     # Active alerts
│
├── blockchain/              # NEW
│   ├── page.tsx             # Blockchain overview
│   ├── wallet-card.tsx      # Wallet info & balance
│   ├── transactions-table.tsx
│   ├── transactions-search.tsx
│   ├── escrows-table.tsx    # Active escrows
│   └── stats-cards.tsx      # Payment statistics
│
├── scheduler/               # NEW
│   ├── page.tsx             # Scheduler overview
│   ├── queue-table.tsx      # Jobs in queue
│   ├── assignments-table.tsx # Active assignments
│   ├── throughput-chart.tsx # Historical throughput
│   └── config-panel.tsx     # Scheduler settings
│
├── nodes/                   # ENHANCED
│   ├── live/                # NEW subdirectory
│   │   ├── page.tsx         # Real-time node view
│   │   ├── node-map.tsx     # Geographic distribution
│   │   └── heartbeat-list.tsx
│   └── [nodeId]/
│       └── metrics/         # NEW
│           └── page.tsx     # Node detail metrics
│
└── devices/                 # EXISTING (enhanced)
    └── ... (add real-time indicators)
```

### 6.2 Page Designs

#### 6.2.1 Infrastructure Page (`/infrastructure`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Infrastructure Health                                                        │
│ Monitor system services and infrastructure status                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  PostgreSQL  │ │    Redis     │ │  Solana RPC  │ │ gRPC Server  │       │
│  │  ● HEALTHY   │ │  ● HEALTHY   │ │  ● HEALTHY   │ │  ● RUNNING   │       │
│  │  2ms         │ │  1ms         │ │  200ms       │ │  :50051      │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ System Metrics (Last Hour)                                              │ │
│  │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │ │                        📈 Live Graph                                │ │ │
│  │ │   Jobs/min ───  Latency ─── Queue Depth ───                        │ │ │
│  │ │                                                                     │ │ │
│  │ └────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────┐ ┌──────────────────────────────────────┐│
│  │ Active Alerts (2)              │ │ Quick Stats                          ││
│  │ ⚠️ Low wallet balance          │ │ Uptime: 15d 4h 23m                   ││
│  │ ⚠️ Node offline > 5min         │ │ Jobs (24h): 1,250                    ││
│  │                                │ │ Nodes Online: 142/150                ││
│  └────────────────────────────────┘ └──────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Blockchain Page (`/blockchain`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Blockchain & Payments                                                        │
│ Monitor wallet, transactions, and escrow accounts                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────┐ ┌────────────────────────────────────┐│
│  │ Platform Wallet                  │ │ Network Info                        ││
│  │                                  │ │                                     ││
│  │ Address: 4Y5HWB...yAG8  [Copy]   │ │ Network: Devnet                     ││
│  │ Balance: 11.3155 SOL             │ │ Program: DefY4G...sp2dA             ││
│  │                                  │ │ RPC: api.devnet.solana.com          ││
│  │ [Request Airdrop] [View on Solscan] │ Latency: 200ms ●                  ││
│  └──────────────────────────────────┘ └────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Volume 24h  │ │ Txns 24h    │ │ Platform Fee│ │ Pending     │           │
│  │ 1,234 SOL   │ │ 450         │ │ 123.45 SOL  │ │ 25 escrows  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Recent Transactions                                        [Filters ▼] │ │
│  │ ┌──────────┬──────────┬──────────┬──────────┬──────────┬────────────┐ │ │
│  │ │ Time     │ Type     │ Job ID   │ Amount   │ Status   │ Signature  │ │ │
│  │ ├──────────┼──────────┼──────────┼──────────┼──────────┼────────────┤ │ │
│  │ │ 08:50:15 │ ESCROW   │ abc123.. │ 0.5 SOL  │ ● CONF   │ 5eykt4...  │ │ │
│  │ │ 08:45:30 │ RELEASE  │ def456.. │ 0.45 SOL │ ● CONF   │ 3xYpQ2...  │ │ │
│  │ │ 08:40:00 │ REFUND   │ ghi789.. │ 0.3 SOL  │ ● PEND   │ 7zKmN8...  │ │ │
│  │ └──────────┴──────────┴──────────┴──────────┴──────────┴────────────┘ │ │
│  │                                                          [< 1 2 3 >]  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.3 Scheduler Page (`/scheduler`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Job Scheduler                                                                │
│ Monitor job queue, assignments, and throughput                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Queue Depth │ │ Processing  │ │ Active Jobs │ │ Avg Wait    │           │
│  │ 15 jobs     │ │ 12.5/min    │ │ 8 running   │ │ 45 seconds  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Job Throughput (Last 24 Hours)                                         │ │
│  │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │ │     ▂▃▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▃▂▁▂▃▄▅▆▇█                            │ │ │
│  │ │     ─────────────────────────────────────────────────────────────── │ │ │
│  │ │     00:00    04:00    08:00    12:00    16:00    20:00    24:00    │ │ │
│  │ └────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────┐ ┌────────────────────────────────────┐│
│  │ Jobs in Queue                    │ │ Active Assignments                  ││
│  │ ┌────────────────────────────┐   │ │ ┌────────────────────────────────┐ ││
│  │ │ Job ID   │ Priority │ Wait │   │ │ │ Job      │ Node     │ Progress │ ││
│  │ │ abc123   │ 🔴 HIGH  │ 2m   │   │ │ │ xyz789   │ node-42  │ ███░ 75% │ ││
│  │ │ def456   │ 🟡 NORM  │ 5m   │   │ │ │ uvw456   │ node-15  │ ██░░ 45% │ ││
│  │ │ ghi789   │ 🟢 LOW   │ 8m   │   │ │ │ rst123   │ node-08  │ █░░░ 20% │ ││
│  │ └────────────────────────────┘   │ │ └────────────────────────────────┘ ││
│  └──────────────────────────────────┘ └────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.4 Live Nodes Page (`/nodes/live`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Live Node Status                                                             │
│ Real-time view of compute nodes with heartbeat monitoring                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Online      │ │ Busy        │ │ Idle        │ │ Offline     │           │
│  │ 🟢 142      │ │ 🔵 95       │ │ ⚪ 47       │ │ 🔴 8        │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Node Map                                                    [List View] │ │
│  │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │ │                                                                     │ │ │
│  │ │         🟢🟢     🟢🔴              🟢🟢🟢                           │ │ │
│  │ │      🟢🟢🟢🟢         🟢🟢🟢🟢      🟢🟢                            │ │ │
│  │ │                              🟢🟢🟢                                  │ │ │
│  │ │                                                        🟢🟢🟢🟢     │ │ │
│  │ │                                                                     │ │ │
│  │ └────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Heartbeat Feed (Live)                                        [Pause]   │ │
│  │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ 08:51:02 │ node-42  │ 🟢 heartbeat │ GPU: 85% │ Job: xyz789        │ │ │
│  │ │ 08:51:01 │ node-15  │ 🟢 heartbeat │ GPU: 72% │ Job: uvw456        │ │ │
│  │ │ 08:51:00 │ node-08  │ 🟢 heartbeat │ GPU: 45% │ Job: rst123        │ │ │
│  │ │ 08:50:58 │ node-99  │ 🔴 timeout   │ Last seen: 35s ago            │ │ │
│  │ └────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Sidebar Navigation Update

Add to `admin-sidebar.tsx`:

```tsx
const navigationItems = [
  // ... existing items ...

  // NEW: Infrastructure section
  {
    title: "Infrastructure",
    items: [
      {
        title: "System Health",
        href: "/infrastructure",
        icon: Activity,
      },
      {
        title: "Blockchain",
        href: "/blockchain",
        icon: Wallet,
      },
      {
        title: "Scheduler",
        href: "/scheduler",
        icon: Clock,
      },
      {
        title: "Live Nodes",
        href: "/nodes/live",
        icon: Radio,
      },
    ],
  },
];
```

---

## 7. Database Schema Alignment

### 7.1 Current Schemas

The Web API and Central Server may have separate databases. We need to ensure they can share data or sync appropriately.

**Option A: Shared Database (Recommended)**
- Both services connect to same PostgreSQL instance
- Central Server uses same schema as Web API
- No sync needed, real-time consistency

**Option B: Database Replication**
- Central Server has its own DB
- Changes replicated to Web API DB
- Eventual consistency

**Option C: API-only Access**
- Web API never touches Central Server DB
- All data fetched via REST API
- Simplest but higher latency

### 7.2 Schema Additions

If using shared database, add these tables:

```sql
-- System health snapshots for historical data
CREATE TABLE system_health_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    db_latency_ms INTEGER,
    redis_latency_ms INTEGER,
    solana_latency_ms INTEGER,
    queue_depth INTEGER,
    online_nodes INTEGER,
    active_jobs INTEGER
);

-- Create index for time-series queries
CREATE INDEX idx_health_snapshots_timestamp
ON system_health_snapshots(timestamp DESC);

-- Blockchain transactions log (supplements on-chain data)
CREATE TABLE blockchain_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signature VARCHAR(128) UNIQUE,
    tx_type VARCHAR(32) NOT NULL, -- escrow_lock, escrow_release, refund
    job_id UUID REFERENCES jobs(id),
    amount_lamports BIGINT NOT NULL,
    from_pubkey VARCHAR(64),
    to_pubkey VARCHAR(64),
    status VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confirmed_at TIMESTAMPTZ,
    error_message TEXT
);

-- Alerts table
CREATE TABLE system_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    severity VARCHAR(16) NOT NULL, -- critical, warning, info
    alert_type VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255)
);
```

---

## 8. Real-time Features

### 8.1 WebSocket Implementation

For live updates, implement WebSocket support in Central Server:

```rust
// src/api/websocket/mod.rs
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to events
    let mut health_rx = state.health_broadcast.subscribe();
    let mut node_rx = state.node_events.subscribe();

    loop {
        tokio::select! {
            Ok(health) = health_rx.recv() => {
                let msg = serde_json::to_string(&health).unwrap();
                sender.send(Message::Text(msg)).await.ok();
            }
            Ok(event) = node_rx.recv() => {
                let msg = serde_json::to_string(&event).unwrap();
                sender.send(Message::Text(msg)).await.ok();
            }
        }
    }
}
```

### 8.2 Event Types

```typescript
// Frontend event types
type WebSocketEvent =
  | { type: 'health_update'; data: SystemHealth }
  | { type: 'node_heartbeat'; data: NodeHeartbeat }
  | { type: 'job_status_change'; data: JobStatusChange }
  | { type: 'alert'; data: SystemAlert }
  | { type: 'blockchain_tx'; data: BlockchainTransaction };
```

### 8.3 Frontend Hook

```typescript
// apps/admin/src/hooks/use-central-server.ts
export function useCentralServerEvents() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [nodes, setNodes] = useState<NodeHeartbeat[]>([]);

  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'health_update':
          setHealth(data.data);
          break;
        case 'node_heartbeat':
          setNodes(prev => updateNodes(prev, data.data));
          break;
        // ... handle other events
      }
    };

    return () => ws.close();
  }, []);

  return { health, nodes };
}
```

---

## 9. Security & Authentication

### 9.1 Authentication Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Layer 1: Admin Session (NextAuth)
─────────────────────────────────
Admin logs into dashboard → NextAuth session created
Session includes: admin_id, email, role, permissions

Layer 2: Service Token (Internal)
─────────────────────────────────
Admin dashboard → Web API → Central Server
Web API generates short-lived service token for Central Server calls

Layer 3: API Key (Alternative)
──────────────────────────────
For direct Central Server access (ops, scripts)
API key stored in environment, rotated regularly
```

### 9.2 Role-Based Access Control

```typescript
// Permission levels for Central Server endpoints
const permissions = {
  // Read-only access
  'infrastructure:read': ['admin', 'viewer'],
  'blockchain:read': ['admin', 'finance', 'viewer'],
  'scheduler:read': ['admin', 'ops', 'viewer'],
  'nodes:read': ['admin', 'ops', 'viewer'],

  // Write access
  'scheduler:manage': ['admin', 'ops'],
  'alerts:acknowledge': ['admin', 'ops'],
  'blockchain:airdrop': ['admin', 'finance'],

  // Dangerous operations
  'scheduler:pause': ['admin'],
  'nodes:disconnect': ['admin'],
  'system:restart': ['admin'],
};
```

### 9.3 Central Server Auth Middleware

```rust
// src/api/rest/middleware/auth.rs
pub async fn admin_auth(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = auth.token();

    // Verify service token from Web API
    let claims = state.jwt_manager
        .verify_service_token(token)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Check required permissions
    let required_permission = get_required_permission(&request);
    if !claims.permissions.contains(&required_permission) {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}
```

### 9.4 Network Security

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NETWORK CONFIGURATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Production Setup:
─────────────────

                    Internet
                        │
                        ▼
              ┌─────────────────┐
              │   Cloudflare    │
              │   (CDN + WAF)   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Load Balancer  │
              │   (nginx/ALB)   │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Web App │  │ Admin   │  │ Web API │
    │ :3000   │  │ :3001   │  │ :8000   │
    └─────────┘  └─────────┘  └────┬────┘
                                   │
                         Internal Network
                                   │
                             ┌─────▼─────┐
                             │  Central  │
                             │  Server   │
                             │ :8080 REST│
                             │ :50051 gRPC
                             └───────────┘

Key Points:
- Central Server NOT exposed to internet
- Web API is the only gateway to Central Server
- Internal network communication only
- TLS everywhere (even internal)
```

---

## 10. Implementation Roadmap

### 10.1 Phase 1: Foundation (Week 1-2)

**Goal:** REST API in Central Server + basic admin pages

| Task | Owner | Priority | Effort |
|------|-------|----------|--------|
| Add Axum REST router to Central Server | Backend | P0 | 2d |
| Implement `/health` endpoint | Backend | P0 | 0.5d |
| Implement `/blockchain/wallet` endpoint | Backend | P0 | 1d |
| Implement `/scheduler/status` endpoint | Backend | P0 | 1d |
| Create `/infrastructure` admin page | Frontend | P0 | 2d |
| Add health status cards component | Frontend | P0 | 1d |
| Service token auth middleware | Backend | P0 | 1d |
| Update admin sidebar navigation | Frontend | P1 | 0.5d |

**Deliverable:** Basic infrastructure page showing system health

### 10.2 Phase 2: Blockchain Integration (Week 3-4)

**Goal:** Full blockchain view in admin

| Task | Owner | Priority | Effort |
|------|-------|----------|--------|
| Implement `/blockchain/transactions` endpoint | Backend | P0 | 2d |
| Implement `/blockchain/escrows` endpoint | Backend | P0 | 1d |
| Implement `/blockchain/stats` endpoint | Backend | P0 | 1d |
| Create `/blockchain` admin page | Frontend | P0 | 2d |
| Wallet card component | Frontend | P0 | 1d |
| Transactions table with filters | Frontend | P0 | 2d |
| Escrows table component | Frontend | P1 | 1d |
| Payment stats cards | Frontend | P1 | 1d |

**Deliverable:** Complete blockchain admin page

### 10.3 Phase 3: Scheduler Dashboard (Week 5-6)

**Goal:** Scheduler monitoring and control

| Task | Owner | Priority | Effort |
|------|-------|----------|--------|
| Implement `/scheduler/queue` endpoint | Backend | P0 | 1d |
| Implement `/scheduler/assignments` endpoint | Backend | P0 | 1d |
| Implement `/scheduler/throughput` endpoint | Backend | P0 | 2d |
| Create `/scheduler` admin page | Frontend | P0 | 2d |
| Queue table component | Frontend | P0 | 1d |
| Assignments table component | Frontend | P0 | 1d |
| Throughput chart (recharts) | Frontend | P0 | 2d |
| Job priority indicators | Frontend | P1 | 0.5d |

**Deliverable:** Complete scheduler admin page

### 10.4 Phase 4: Live Nodes & Real-time (Week 7-8)

**Goal:** Real-time node monitoring

| Task | Owner | Priority | Effort |
|------|-------|----------|--------|
| Implement `/nodes/live` endpoint | Backend | P0 | 1d |
| Implement `/nodes/{id}/metrics` endpoint | Backend | P0 | 1d |
| WebSocket server in Central Server | Backend | P0 | 3d |
| Create `/nodes/live` admin page | Frontend | P0 | 2d |
| Node map component (optional) | Frontend | P2 | 3d |
| Heartbeat feed component | Frontend | P0 | 2d |
| Real-time connection hook | Frontend | P0 | 2d |
| Node detail metrics page | Frontend | P1 | 2d |

**Deliverable:** Real-time node monitoring

### 10.5 Phase 5: Alerts & Polish (Week 9-10)

**Goal:** Alerting system and UX polish

| Task | Owner | Priority | Effort |
|------|-------|----------|--------|
| Implement `/alerts` endpoints | Backend | P0 | 2d |
| Alert generation logic | Backend | P0 | 2d |
| Alerts panel component | Frontend | P0 | 2d |
| Toast notifications for alerts | Frontend | P0 | 1d |
| Dashboard widgets for overview | Frontend | P1 | 2d |
| Error handling improvements | Both | P0 | 2d |
| Performance optimization | Both | P1 | 2d |
| Documentation | Both | P1 | 2d |

**Deliverable:** Production-ready admin integration

### 10.6 Gantt Chart Overview

```
Week:     1    2    3    4    5    6    7    8    9    10
          ─────────────────────────────────────────────────
Phase 1   ████████
Phase 2             ████████
Phase 3                      ████████
Phase 4                               ████████
Phase 5                                        ████████
          ─────────────────────────────────────────────────
```

---

## 11. Testing Strategy

### 11.1 API Testing

```rust
// Central Server API tests
#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();

        let response = server
            .get("/api/v1/health")
            .add_header("Authorization", "Bearer test-token")
            .await;

        response.assert_status_ok();
        response.assert_json_contains(&json!({
            "status": "healthy"
        }));
    }

    #[tokio::test]
    async fn test_blockchain_wallet() {
        let app = create_test_app().await;
        let server = TestServer::new(app).unwrap();

        let response = server
            .get("/api/v1/blockchain/wallet")
            .add_header("Authorization", "Bearer test-token")
            .await;

        response.assert_status_ok();
        response.assert_json_contains(&json!({
            "network": "devnet"
        }));
    }
}
```

### 11.2 Frontend Testing

```typescript
// Admin page tests
import { render, screen } from '@testing-library/react';
import { InfrastructurePage } from './page';

describe('Infrastructure Page', () => {
  it('displays health status cards', async () => {
    render(<InfrastructurePage />);

    expect(await screen.findByText('PostgreSQL')).toBeInTheDocument();
    expect(await screen.findByText('Redis')).toBeInTheDocument();
    expect(await screen.findByText('Solana RPC')).toBeInTheDocument();
  });

  it('shows alert when service unhealthy', async () => {
    // Mock unhealthy response
    server.use(
      rest.get('/api/v1/health', (req, res, ctx) => {
        return res(ctx.json({
          services: {
            redis: { status: 'unhealthy' }
          }
        }));
      })
    );

    render(<InfrastructurePage />);

    expect(await screen.findByText('UNHEALTHY')).toBeInTheDocument();
  });
});
```

### 11.3 Integration Testing

```bash
# Docker compose for integration tests
docker-compose -f docker-compose.test.yml up -d

# Run Central Server
cargo run --release &

# Run integration tests
npm run test:integration --prefix apps/admin

# Teardown
docker-compose -f docker-compose.test.yml down
```

### 11.4 Load Testing

```yaml
# k6 load test for Central Server API
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 50 },
    { duration: '3m', target: 50 },
    { duration: '1m', target: 0 },
  ],
};

export default function () {
  const res = http.get('http://localhost:8080/api/v1/health', {
    headers: { 'Authorization': 'Bearer test-token' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 100ms': (r) => r.timings.duration < 100,
  });
}
```

---

## 12. Deployment Considerations

### 12.1 Environment Variables

```bash
# Central Server (.env)
DATABASE_URL=postgresql://user:pass@localhost/cyxwiz
REDIS_URL=redis://localhost:6379
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_KEYPAIR_PATH=/etc/cyxwiz/keypair.json
SOLANA_PROGRAM_ID=DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA

# REST API Config
REST_API_PORT=8080
REST_API_HOST=0.0.0.0
SERVICE_TOKEN_SECRET=your-secret-here
CORS_ORIGINS=https://admin.cyxwiz.com

# gRPC Config
GRPC_PORT=50051
```

### 12.2 Docker Configuration

```dockerfile
# Central Server Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/cyxwiz-central-server /usr/local/bin/
EXPOSE 8080 50051
CMD ["cyxwiz-central-server"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  central-server:
    build: ./cyxwiz-central-server
    ports:
      - "8080:8080"   # REST API
      - "50051:50051" # gRPC
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
    networks:
      - internal

  admin:
    build: ./cyxwiz_web/apps/admin
    ports:
      - "3001:3000"
    environment:
      - CENTRAL_SERVER_URL=http://central-server:8080
    networks:
      - internal
      - external

networks:
  internal:
    internal: true
  external:
```

### 12.3 Health Checks

```yaml
# Kubernetes readiness/liveness probes
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: central-server
    livenessProbe:
      httpGet:
        path: /api/v1/health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /api/v1/health
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
```

### 12.4 Monitoring

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'central-server'
    static_configs:
      - targets: ['central-server:8080']
    metrics_path: '/api/v1/metrics'
```

---

## Appendix A: API Response Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful request |
| 201 | Created | Resource created |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Error | Server error |
| 503 | Service Unavailable | Service down |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Central Server** | Rust service that orchestrates the compute network |
| **TUI** | Terminal User Interface (ratatui-based) |
| **Escrow** | On-chain account holding payment until job completion |
| **Heartbeat** | Periodic signal from nodes indicating they're alive |
| **Scheduler** | Component that matches jobs to available nodes |
| **Lamports** | Smallest unit of SOL (1 SOL = 1,000,000,000 lamports) |

---

## Appendix C: Related Documents

- [CLAUDE.md](../../../CLAUDE.md) - Project overview and guidelines
- [GRPC_ENABLEMENT_GUIDE.md](../GRPC_ENABLEMENT_GUIDE.md) - gRPC setup guide
- [Solana Integration](../src/blockchain/) - Blockchain client implementation

---

**Document Status:** Draft
**Last Updated:** December 11, 2025
**Next Review:** After Phase 1 completion
