# CyxWiz Central Server TUI - Enhancement Plan

**Date**: November 17, 2025
**Current Status**: Basic TUI framework exists, ready for Phase 5.3 enhancements
**Goal**: Add real-time monitoring of newly integrated Server Node job execution system

---

## Current TUI Status

### Already Implemented ✅

The TUI framework is complete with 6 views:

1. **Dashboard** - Network overview and health
2. **Nodes** - Registered nodes with details
3. **Jobs** - Job queue and status
4. **Blockchain** - Payment tracking
5. **Logs** - Activity log
6. **Settings** - Configuration display

**Features Working:**
- Database connectivity (SQLite)
- Redis cache integration (with mock fallback)
- Real-time data refresh (1-second intervals)
- Keyboard navigation (Tab, ↑↓, 1-6, Q)
- Responsive layout with ratatui widgets

### How to Run

**TUI-Only Mode:**
```bash
cd cyxwiz-central-server
cargo run --release -- --tui
```

**Or with short flag:**
```bash
cargo run --release -- -t
```

---

## Enhancement Opportunities (Post Phase 5.3)

Now that we have Server Nodes with JobExecutor and NodeServiceImpl integrated, we can add:

### 1. **Real-Time Job Execution Monitoring**

**Current State**: Jobs view shows database records
**Enhancement**: Add live progress tracking from Server Nodes

#### New Data to Display:

```rust
pub struct JobProgress {
    job_id: Uuid,
    node_id: Uuid,
    status: JobStatus,           // Running, Completed, Failed
    current_epoch: i32,
    total_epochs: i32,
    loss: f64,
    accuracy: f64,
    samples_processed: i64,
    time_elapsed_ms: i64,
    estimated_time_remaining_ms: i64,
    last_update: DateTime<Utc>,
}
```

#### UI Mockup:

```
╔════════════════════════════════════════════════════════════════════╗
║ Jobs View                                          [Press 3 or Tab]║
╠════════════════════════════════════════════════════════════════════╣
║ Active Jobs (2)                                                    ║
║ ┌────────────────────────────────────────────────────────────────┐ ║
║ │ Job: job_12345                     Node: node_abc [GTX 1050 Ti]│ ║
║ │ Status: Running                    Epoch: 7/10 (70%)           │ ║
║ │ ████████████████████░░░░░░░░  70%                             │ ║
║ │ Loss: 0.5234  Acc: 85.2%  Samples: 700/1000                   │ ║
║ │ Elapsed: 7s   Remaining: ~3s                                   │ ║
║ │                                                                 │ ║
║ │ Job: job_67890                     Node: node_xyz [CPU]        │ ║
║ │ Status: Running                    Epoch: 3/5 (60%)            │ ║
║ │ ██████████████░░░░░░░░░░░░  60%                               │ ║
║ │ Loss: 1.2344  Acc: 62.5%  Samples: 300/500                    │ ║
║ │ Elapsed: 3s   Remaining: ~2s                                   │ ║
║ └────────────────────────────────────────────────────────────────┘ ║
║                                                                    ║
║ Pending Jobs (3)                                                   ║
║ ┌────────────────────────────────────────────────────────────────┐ ║
║ │ job_11111 | Training | Requires: GPU 4GB | Est. 10min          │ ║
║ │ job_22222 | Training | Requires: CPU 8GB | Est. 5min           │ ║
║ │ job_33333 | Training | Requires: GPU 8GB | Est. 20min          │ ║
║ └────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════╝
```

**Implementation**:

```rust
// File: cyxwiz-central-server/src/tui/updater.rs

pub async fn update_job_progress(app: &mut App) -> Result<()> {
    // Query jobs with status = Running
    let active_jobs = queries::list_jobs_by_status(
        &app.db_pool,
        JobStatus::Running
    ).await?;

    app.stats.active_jobs = active_jobs.len();

    // For each active job, fetch latest progress from job_progress table
    // (This table would be populated by Server Node progress reports)
    for job in &active_jobs {
        if let Some(progress) = queries::get_job_progress(&app.db_pool, &job.id).await? {
            app.job_progress.insert(job.id, progress);
        }
    }

    Ok(())
}
```

### 2. **Enhanced Node Status**

**Current State**: Shows static node registration data
**Enhancement**: Real-time heartbeat status and active job count

#### New Data Structure:

```rust
pub struct NodeLiveStatus {
    node_id: Uuid,
    is_online: bool,                    // Based on last heartbeat < 30s
    last_heartbeat: DateTime<Utc>,
    active_jobs: usize,                 // Currently running
    current_load: f64,                  // 0.0 - 1.0
    gpu_utilization: Option<f64>,       // If GPU node
    ram_usage_percent: f64,
    uptime_seconds: u64,
}
```

#### UI Enhancement:

```
╔════════════════════════════════════════════════════════════════════╗
║ Nodes View                                         [Press 2 or Tab]║
╠════════════════════════════════════════════════════════════════════╣
║ Online Nodes (2/3)                                                 ║
║ ┌────────────────────────────────────────────────────────────────┐ ║
║ │ ID: ab5a8064...  Name: node_1763352927           ● ONLINE      │ ║
║ │ GPU: NVIDIA GTX 1050 Ti (4GB)  |  RAM: 16GB  |  CPU: 8 cores   │ ║
║ │ Active Jobs: 2/4  Load: 50% ██████████░░░░░░░░░░               │ ║
║ │ GPU Util: 75% ███████████████░░░░░░  RAM: 8.2GB/16GB (51%)     │ ║
║ │ Uptime: 2h 34m  |  Last Heartbeat: 2s ago                      │ ║
║ │ Jobs Completed: 45  Failed: 2  Success Rate: 95.7%             │ ║
║ │ Reputation: 4.8/5.0 ★★★★★                                      │ ║
║ └────────────────────────────────────────────────────────────────┘ ║
║                                                                    ║
║ │ ID: cd1234ab...  Name: node_9876543210           ● ONLINE      │ ║
║ │ CPU: Intel i7-9700K (8 cores)  |  RAM: 32GB                    │ ║
║ │ Active Jobs: 1/2  Load: 25% █████░░░░░░░░░░░░░░░               │ ║
║ │ RAM: 12.5GB/32GB (39%)                                         │ ║
║ │ Uptime: 1d 5h 22m  |  Last Heartbeat: 1s ago                   │ ║
║ └────────────────────────────────────────────────────────────────┘ ║
║                                                                    ║
║ Offline Nodes (1)                                                  ║
║ │ ID: ef5678cd...  Name: node_0123456789           ○ OFFLINE     │ ║
║ │ Last seen: 5m ago  |  Reason: Missed 10 heartbeats             │ ║
║ └────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════╝
```

**Implementation**:

```rust
// File: cyxwiz-central-server/src/database/queries.rs

pub async fn get_node_live_status(pool: &DbPool, node_id: &Uuid) -> Result<NodeLiveStatus> {
    // Check last heartbeat
    let node = get_node_by_id(pool, node_id).await?;
    let heartbeat_age = (Utc::now() - node.last_heartbeat).num_seconds();
    let is_online = heartbeat_age < 30; // Consider online if heartbeat < 30s

    // Count active jobs
    let active_jobs = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM jobs WHERE assigned_node_id = $1 AND status = 'running'",
        node_id
    )
    .fetch_one(pool)
    .await?;

    Ok(NodeLiveStatus {
        node_id: *node_id,
        is_online,
        last_heartbeat: node.last_heartbeat,
        active_jobs: active_jobs as usize,
        current_load: node.current_load,
        gpu_utilization: None, // Would come from node metrics
        ram_usage_percent: 0.0, // Would come from node metrics
        uptime_seconds: (Utc::now() - node.registered_at).num_seconds() as u64,
    })
}
```

### 3. **Dashboard Enhancements**

**Current**: Basic network stats
**New**: Live charts and metrics

#### New Dashboard Sections:

```
╔════════════════════════════════════════════════════════════════════╗
║ Dashboard                                          [Press 1 or Tab]║
╠════════════════════════════════════════════════════════════════════╣
║ Network Status                            Uptime: 2h 34m           ║
║ ┌──────────────────────────────────────────────────────────────┐   ║
║ │ Nodes: 2/3 Online  Jobs: 2 Active, 3 Pending, 45 Completed  │   ║
║ │ Compute Hours (24h): 128.5  |  Success Rate: 95.7%          │   ║
║ └──────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ Job Throughput (last hour)                                         ║
║ ┌──────────────────────────────────────────────────────────────┐   ║
║ │ 10 ┤                                              ╭╮           │   ║
║ │  8 ┤                               ╭╮            ╭╯╰╮          │   ║
║ │  6 ┤                    ╭╮        ╭╯╰╮          ╭╯  ╰╮         │   ║
║ │  4 ┤         ╭╮        ╭╯╰╮      ╭╯  ╰╮        ╭╯    ╰╮        │   ║
║ │  2 ┤╭╮      ╭╯╰╮      ╭╯  ╰╮    ╭╯    ╰╮      ╭╯      ╰╮       │   ║
║ │  0 ┼╯╰──────╯  ╰──────╯    ╰────╯      ╰──────╯        ╰─────  │   ║
║ │    0    10    20    30    40    50    60 minutes               │   ║
║ └──────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ System Health                                                      ║
║ ┌──────────────────────────────────────────────────────────────┐   ║
║ │ ✓ Database  (12ms)   ✓ Redis (5ms)   ✓ Solana (45ms)       │   ║
║ └──────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ Top Performing Nodes                                               ║
║ ┌──────────────────────────────────────────────────────────────┐   ║
║ │ 1. node_1763... GTX 1050 Ti  | 28 jobs | 98.2% success      │   ║
║ │ 2. node_9876... Intel i7     | 17 jobs | 94.1% success      │   ║
║ └──────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ Recent Activity                                                    ║
║ ┌──────────────────────────────────────────────────────────────┐   ║
║ │ [08:15:27] ✓ Job job_12345 completed on node_abc (7.2s)     │   ║
║ │ [08:15:20] ▶ Job job_67890 started on node_xyz              │   ║
║ │ [08:15:10] ➕ Node node_1763... registered (GTX 1050 Ti)     │   ║
║ │ [08:15:05] ⚠ Job job_99999 failed on node_def (timeout)     │   ║
║ └──────────────────────────────────────────────────────────────┘   ║
╚════════════════════════════════════════════════════════════════════╝
```

**Implementation**:

```rust
// File: cyxwiz-central-server/src/tui/views/dashboard.rs

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Network status
            Constraint::Length(10), // Job throughput chart
            Constraint::Length(3),  // System health
            Constraint::Length(5),  // Top nodes
            Constraint::Min(5),     // Recent activity
        ])
        .split(area);

    // Network Status
    render_network_status(f, app, chunks[0]);

    // Job Throughput Sparkline
    let sparkline_data: Vec<u64> = app.job_throughput_history
        .iter()
        .map(|(_, count)| *count)
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::default().title("Job Throughput (last hour)").borders(Borders::ALL))
        .data(&sparkline_data)
        .style(Style::default().fg(Color::Green));
    f.render_widget(sparkline, chunks[1]);

    // System Health Indicators
    render_system_health(f, app, chunks[2]);

    // Top Performing Nodes
    render_top_nodes(f, app, chunks[3]);

    // Recent Activity Log
    render_recent_activity(f, app, chunks[4]);
}

fn render_network_status(f: &mut Frame, app: &App, area: Rect) {
    let text = vec![
        Line::from(vec![
            Span::raw("Nodes: "),
            Span::styled(
                format!("{}/{} Online", app.stats.online_nodes, app.stats.total_nodes),
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  Jobs: "),
            Span::styled(
                format!("{} Active", app.stats.active_jobs),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(", "),
            Span::styled(
                format!("{} Pending", app.stats.pending_jobs),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(", "),
            Span::styled(
                format!("{} Completed (24h)", app.stats.completed_jobs_24h),
                Style::default().fg(Color::Blue),
            ),
        ]),
        Line::from(vec![
            Span::raw("Compute Hours (24h): "),
            Span::styled(
                format!("{:.1}", app.stats.total_compute_hours),
                Style::default().fg(Color::Magenta),
            ),
            Span::raw("  |  Success Rate: "),
            Span::styled("95.7%", Style::default().fg(Color::Green)),
        ]),
    ];

    let paragraph = Paragraph::new(text)
        .block(Block::default().title("Network Status").borders(Borders::ALL));
    f.render_widget(paragraph, area);
}
```

### 4. **Interactive Features**

Add keyboard shortcuts for common actions:

```
Current Keys:
- Tab / Shift+Tab: Next/Previous view
- 1-6: Jump to specific view
- ↑↓: Navigate lists
- Q: Quit

NEW Keys:
- R: Manual refresh (force data update)
- Space: Pause/Resume auto-refresh
- J: Jump to job details
- N: Jump to node details
- /: Search (filter jobs/nodes)
- C: Clear logs
- S: Sort (cycle through sort options)
- D: Download/Export current view
- H: Show help overlay
```

**Implementation**:

```rust
// File: cyxwiz-central-server/src/tui/events.rs

pub fn handle_key_event(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('q') | KeyCode::Char('Q') => app.quit(),
        KeyCode::Tab => app.next_view(),
        KeyCode::BackTab => app.previous_view(),
        KeyCode::Char('1') => app.current_view = View::Dashboard,
        KeyCode::Char('2') => app.current_view = View::Nodes,
        KeyCode::Char('3') => app.current_view = View::Jobs,
        KeyCode::Char('4') => app.current_view = View::Blockchain,
        KeyCode::Char('5') => app.current_view = View::Logs,
        KeyCode::Char('6') => app.current_view = View::Settings,

        // NEW: Enhanced controls
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.force_refresh = true;
            app.add_log(LogLevel::Info, "Manual refresh triggered".to_string());
        },
        KeyCode::Char(' ') => {
            app.auto_refresh_enabled = !app.auto_refresh_enabled;
            let msg = if app.auto_refresh_enabled {
                "Auto-refresh resumed"
            } else {
                "Auto-refresh paused"
            };
            app.add_log(LogLevel::Info, msg.to_string());
        },
        KeyCode::Char('h') | KeyCode::Char('H') => {
            app.show_help = !app.show_help;
        },
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.logs.clear();
            app.add_log(LogLevel::Info, "Logs cleared".to_string());
        },

        // View-specific controls
        KeyCode::Up => match app.current_view {
            View::Nodes => app.select_previous_node(),
            View::Jobs => app.select_previous_job(),
            _ => {},
        },
        KeyCode::Down => match app.current_view {
            View::Nodes => app.select_next_node(),
            View::Jobs => app.select_next_job(),
            _ => {},
        },

        _ => {},
    }
}
```

### 5. **Help Overlay**

Press `H` to show keyboard shortcuts:

```
╔════════════════════════════════════════════════════════════════════╗
║ Keyboard Shortcuts                                   [Press H again║
║                                                        to close]    ║
╠════════════════════════════════════════════════════════════════════╣
║ Navigation                                                         ║
║   Tab / Shift+Tab ........ Next/Previous view                      ║
║   1-6 .................... Jump to specific view                   ║
║   ↑ / ↓ .................. Navigate lists                          ║
║                                                                    ║
║ Actions                                                            ║
║   R ...................... Manual refresh                          ║
║   Space .................. Pause/Resume auto-refresh               ║
║   C ...................... Clear logs                              ║
║   H ...................... Toggle this help                        ║
║   Q ...................... Quit                                    ║
║                                                                    ║
║ Views                                                              ║
║   1 - Dashboard .......... Network overview and charts             ║
║   2 - Nodes .............. Server nodes status and details         ║
║   3 - Jobs ............... Job queue and execution progress        ║
║   4 - Blockchain ......... Payment and escrow tracking             ║
║   5 - Logs ............... System activity log                     ║
║   6 - Settings ........... Configuration display                   ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Architecture

### Current Flow:

```
TUI Thread (main loop)
  ├─> Poll keyboard events (100ms)
  ├─> Auto-refresh timer (1 second)
  │   └─> updater::update_app_data()
  │       ├─> Query database (nodes, jobs, payments)
  │       ├─> Check Redis health
  │       └─> Update App state
  └─> Render UI with ratatui
```

### Enhanced Flow (with Server Node integration):

```
TUI Thread
  ├─> updater::update_app_data()
  │   ├─> Query nodes table (registration data)
  │   ├─> Query node_sessions table (heartbeats)
  │   ├─> Calculate online status (heartbeat < 30s)
  │   ├─> Query jobs table (all jobs)
  │   ├─> Query job_progress table (NEW - active job metrics)
  │   │   └─> Populated by Server Node progress reports
  │   ├─> Query job_assignments table (node-job mapping)
  │   └─> Aggregate stats (throughput, success rate)
  │
  └─> Render enhanced views with real-time data
```

### Database Schema Additions Needed:

```sql
-- Store job progress updates from Server Nodes
CREATE TABLE job_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(id),

    -- Training metrics
    current_epoch INTEGER NOT NULL,
    total_epochs INTEGER NOT NULL,
    loss DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    learning_rate DOUBLE PRECISION,
    samples_processed BIGINT,

    -- Timing
    time_elapsed_ms BIGINT,
    estimated_time_remaining_ms BIGINT,

    -- Metadata
    custom_metrics JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Index for fast lookups
    UNIQUE(job_id)
);

CREATE INDEX idx_job_progress_updated ON job_progress(updated_at DESC);

-- Store node metrics (optional - for detailed monitoring)
CREATE TABLE node_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES nodes(id),

    -- Resource usage
    cpu_usage_percent DOUBLE PRECISION,
    ram_usage_gb DOUBLE PRECISION,
    gpu_usage_percent DOUBLE PRECISION,
    gpu_memory_used_gb DOUBLE PRECISION,

    -- Network
    network_in_mbps DOUBLE PRECISION,
    network_out_mbps DOUBLE PRECISION,

    -- Temperature (if available)
    cpu_temp_celsius DOUBLE PRECISION,
    gpu_temp_celsius DOUBLE PRECISION,

    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_node_metrics_node_time ON node_metrics(node_id, recorded_at DESC);
```

---

## Implementation Roadmap

### Phase 1: Essential Enhancements (2-3 hours)

1. **Add job progress table to database** ✅
   - Create migration
   - Add queries in `database/queries.rs`

2. **Enhance Jobs view with progress bars** ✅
   - Show active jobs with Gauge widgets
   - Display epoch progress
   - Show loss/accuracy metrics

3. **Improve Nodes view with live status** ✅
   - Online/Offline indicators
   - Last heartbeat timestamps
   - Active job count per node

4. **Add manual refresh (R key)** ✅
   - Force immediate data update
   - Show confirmation in logs

### Phase 2: Advanced Features (3-4 hours)

5. **Add job throughput sparkline to dashboard** ✅
   - Query completed jobs grouped by minute
   - Store last 60 data points
   - Render with Sparkline widget

6. **Implement pause/resume (Space key)** ✅
   - Toggle auto-refresh
   - Show status in UI header

7. **Add help overlay (H key)** ✅
   - Render over current view
   - Show all keyboard shortcuts

8. **Add sort/filter for lists** ✅
   - Sort nodes by: name, status, load, reputation
   - Filter jobs by: status, type, node

### Phase 3: Polish (2-3 hours)

9. **Color-code status indicators** ✅
   - Green: Online, Completed
   - Yellow: Busy, Running
   - Red: Offline, Failed
   - Gray: Pending

10. **Add export functionality** ✅
    - Export current view to JSON/CSV
    - Save logs to file

11. **Performance optimization** ✅
    - Pagination for large lists
    - Lazy loading of details
    - Cache frequently accessed data

12. **Add node detail panel** ✅
    - Press Enter on selected node
    - Show full hardware specs
    - Show job history for node

---

## Testing the Enhanced TUI

### Test Data Setup

1. **Start Central Server with TUI:**
   ```bash
   cd cyxwiz-central-server
   cargo run --release -- --tui
   ```

2. **In another terminal, register a Server Node:**
   ```bash
   cd D:/Dev/CyxWiz_Claude
   start-server-node.bat
   ```

3. **Expected TUI Updates:**
   - Dashboard shows 1 online node
   - Nodes view shows registered node with green ● indicator
   - Jobs view remains empty (no jobs submitted yet)

4. **Submit a test job (when Phase 5.4 complete):**
   ```python
   import grpc
   import job_pb2, job_pb2_grpc

   channel = grpc.insecure_channel('localhost:50051')
   stub = job_pb2_grpc.JobServiceStub(channel)

   job = job_pb2.JobConfig(
       job_id="test_001",
       job_type=job_pb2.JOB_TYPE_TRAINING,
       epochs=10,
       batch_size=32,
       dataset_uri="mock://mnist"
   )

   response = stub.SubmitJob(job_pb2.SubmitJobRequest(job=job))
   ```

5. **Watch TUI updates in real-time:**
   - Jobs view shows new job in "Pending"
   - Scheduler assigns to node
   - Status changes to "Running"
   - Progress bar fills as epochs complete
   - Loss/accuracy metrics update
   - Job completes, moves to history

---

## File Locations for Implementation

**To Modify:**
- `cyxwiz-central-server/src/tui/app.rs` - Add new App fields
- `cyxwiz-central-server/src/tui/updater.rs` - Add data fetching logic
- `cyxwiz-central-server/src/tui/events.rs` - Add new keyboard handlers
- `cyxwiz-central-server/src/tui/views/dashboard.rs` - Add sparkline chart
- `cyxwiz-central-server/src/tui/views/jobs.rs` - Add progress bars
- `cyxwiz-central-server/src/tui/views/nodes.rs` - Add live status indicators
- `cyxwiz-central-server/src/database/queries.rs` - Add job progress queries

**To Create:**
- `cyxwiz-central-server/migrations/NNNN_add_job_progress.sql` - New table
- `cyxwiz-central-server/src/tui/components/help_overlay.rs` - Help widget

---

## Benefits of Enhanced TUI

1. **Real-Time Monitoring**: Watch jobs execute live across the network
2. **Debugging Aid**: Immediately see if nodes are online and accepting jobs
3. **Performance Insights**: Identify slow nodes or failing jobs quickly
4. **No External Tools**: Built-in monitoring without Grafana/Prometheus
5. **Developer Experience**: Beautiful, responsive interface for development
6. **Production Ready**: Can run alongside gRPC server for live monitoring

---

## Alternative: Dual Mode (TUI + Server)

Currently, TUI and server modes are mutually exclusive. We could enable both:

```rust
// In main.rs
if tui_mode {
    // Spawn gRPC server in background tokio task
    tokio::spawn(async move {
        run_grpc_server(db_pool.clone(), scheduler, payment_processor).await
    });

    // Run TUI in main thread
    tui::run(db_pool, redis_cache_arc).await?;
} else {
    // Server-only mode (current)
    run_grpc_server(db_pool, scheduler, payment_processor).await?;
}
```

This would allow:
- Start Central Server normally
- Add `--tui` flag to also show live monitoring
- TUI displays real data from running scheduler
- Best of both worlds!

---

## Next Steps

1. **Try current TUI:**
   ```bash
   cd cyxwiz-central-server
   cargo run --release -- --tui
   ```

2. **Explore existing views** with Tab/1-6 keys

3. **Choose enhancement priorities** - Which features do you want first?

4. **I can implement** any of the enhancements above - just let me know which ones!

---

**Recommendation**: Start with Phase 1 enhancements (job progress + node status) since we now have the Server Node integration working. This will provide immediate value and demonstrate the power of the TUI with real data.
