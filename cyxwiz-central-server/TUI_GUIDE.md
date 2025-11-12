# CyxWiz Central Server - TUI Guide

The CyxWiz Central Server includes a powerful Terminal User Interface (TUI) for real-time monitoring and management.

## 🚀 Starting the TUI

### Option 1: Run with TUI flag
```bash
cargo run --release -- --tui
```

### Option 2: Run compiled binary
```bash
./target/release/cyxwiz-central-server --tui
```

## 📊 Views

The TUI has 6 main views accessible via number keys or Tab:

### 1. Dashboard [Press 1]
- **Network Statistics**: Total nodes, online nodes, active/pending jobs
- **System Health**: Database, Redis, Solana RPC status with latency
- **Job Throughput Graph**: Sparkline showing jobs per minute over last hour
- **Top Nodes**: Best performing nodes by reputation
- **Recent Activity**: Live log of important events

### 2. Nodes View [Press 2]
- **Node Table**: All registered compute nodes with status, reputation, GPU info, current load
- **Selection**: Use `↑`/`↓` arrows or `j`/`k` to navigate
- **Node Details**: Bottom panel shows detailed info about selected node
- **Status Indicators**:
  - `● ONL` = Online (green)
  - `○ OFF` = Offline (gray)
  - `● BSY` = Busy (yellow)
  - `⚠ MNT` = Maintenance (magenta)

### 3. Jobs View [Press 3]
- **Job Table**: All jobs with status, assigned node, progress
- **Selection**: Use `↑`/`↓` arrows to navigate
- **Job Details**: Bottom panel shows requirements, cost, duration, retry count
- **Progress Bar**: Visual representation of job completion
- **Status Icons**:
  - `⏸ PENDING` = Waiting for assignment
  - `→ ASSIGNED` = Assigned to node
  - `▶ RUNNING` = Currently executing
  - `✓ COMPLETE` = Successfully finished
  - `✗ FAILED` = Job failed
  - `⊗ CANCELLED` = User cancelled

### 4. Blockchain View [Press 4]
- **Wallet Info**: Payer address, SOL balance, CYXWIZ token balance
- **Network Stats**: RPC endpoint, latency, block height, TPS
- **Recent Transactions**: Escrow creation, payment distribution, refunds
- **Payment Distribution**: 24-hour volume breakdown (90% nodes, 10% platform)

### 5. Logs View [Press 5]
- **Real-time Logs**: All server events with timestamps
- **Color Coding**:
  - Green `✓` = Success
  - Blue `ℹ` = Info
  - Yellow `⚠` = Warning
  - Red `✗` = Error
- **Auto-scroll**: Shows most recent 100 log entries

### 6. Settings [Press 6]
- **Configuration Display**: Current server settings
- Server addresses (gRPC, REST)
- Database connection details
- Redis configuration
- Blockchain network info
- Scheduler parameters

## ⌨️ Keyboard Shortcuts

### Navigation
- `1-6` or `Tab` - Switch between views
- `Shift+Tab` - Previous view
- `↑` / `k` - Move up in lists
- `↓` / `j` - Move down in lists

### Actions
- `R` or `F5` - Manual refresh (auto-refreshes every second)
- `Q` or `Esc` - Quit TUI
- `Ctrl+C` - Force quit

### Future Shortcuts (Planned)
- `Enter` - View detailed modal for selected item
- `D` - Delete node/cancel job (with confirmation)
- `Space` - Multi-select items
- `/` - Search/filter

## 🎨 UI Layout

```
╔═══════════════════════════════════════════════════════════════════╗
║ Header: Version, Uptime, Last Update                             ║
╠═══════════════════════════════════════════════════════════════════╣
║ Tab Bar: [1] Dashboard [2] Nodes [3] Jobs [4] Blockchain ...     ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║                      Main Content Area                            ║
║                      (Current View)                               ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║ Footer: Keyboard shortcuts help                                  ║
╚═══════════════════════════════════════════════════════════════════╝
```

## 🔄 Data Updates

- **Auto-refresh**: Every 1 second
- **Real-time**: Direct database access (no HTTP overhead)
- **Efficient**: Only fetches changed data
- **Manual refresh**: Press `R` or `F5`

## 🎯 Use Cases

### System Administrator
```bash
# SSH into production server
ssh admin@server.cyxwiz.com

# Launch TUI for monitoring
cd /opt/cyxwiz-central-server
./cyxwiz-central-server --tui

# Monitor node health, check job queue, verify blockchain connectivity
```

### Developer
```bash
# Development mode with TUI
cargo run -- --tui

# Debug job scheduling in real-time
# Watch logs for errors
# Monitor database latency
```

### DevOps
```bash
# Check server health after deployment
./cyxwiz-central-server --tui

# Verify:
# - All services are running (green indicators)
# - Nodes are connecting
# - Jobs are being processed
# - No errors in logs
```

## 📈 Performance Monitoring

The TUI provides real-time metrics:

1. **Database Health**
   - Connection status
   - Query latency (ms)
   - Pool utilization

2. **Redis Health**
   - Connection status
   - Command latency (ms)
   - Queue length

3. **Blockchain Health**
   - RPC endpoint status
   - Transaction latency
   - Block height tracking

4. **Job Throughput**
   - Jobs per minute (sparkline graph)
   - Success rate
   - Average completion time

5. **Node Performance**
   - Online/offline ratio
   - Average reputation score
   - Load distribution

## 🐛 Troubleshooting

### TUI won't start
```
Error: Failed to initialize terminal
```
**Solution**: Ensure your terminal supports ANSI escape codes. Try:
```bash
export TERM=xterm-256color
```

### Display corruption
**Solution**: Resize terminal window or press `Ctrl+L` to redraw

### Can't see colors
**Solution**: Use a terminal that supports 256 colors (e.g., iTerm2, Windows Terminal, gnome-terminal)

### High CPU usage
**Solution**: The TUI updates every second. This is normal. If concerned, use REST API instead.

### Data not updating
**Solution**:
1. Check database connection in "System Health"
2. Press `R` to force refresh
3. Check server logs for errors

## 🔐 SSH Access

The TUI works perfectly over SSH:

```bash
# From local machine
ssh -t user@server.cyxwiz.com './cyxwiz-central-server --tui'

# With tmux for persistent session
ssh user@server.cyxwiz.com
tmux new -s cyxwiz-monitor
./cyxwiz-central-server --tui
# Detach: Ctrl+B then D
# Reattach: tmux attach -t cyxwiz-monitor
```

## 🆚 TUI vs Web Dashboard

| Feature | TUI | Web Dashboard |
|---------|-----|---------------|
| Speed | Very fast (direct DB) | Fast (HTTP API) |
| Access | Local/SSH only | Remote (browser) |
| Multiple users | One at a time | Unlimited |
| Resource usage | Low | Medium |
| Setup | None | Separate React app |
| Use case | Sysadmin, debugging | Business analytics |

## 💡 Tips & Tricks

1. **Use tmux/screen** for persistent monitoring sessions
2. **Pipe output** to file for logging (not implemented yet)
3. **Combine with** `watch` for periodic snapshots
4. **SSH tunneling** for secure remote access
5. **Keyboard-only** operation for efficiency

## 🚧 Planned Features

- [ ] Search/filter functionality in tables
- [ ] Modal popups for detailed views
- [ ] Interactive job cancellation
- [ ] Node management (delete, restart)
- [ ] Custom refresh intervals
- [ ] Export data to CSV
- [ ] Help modal (`H` key)
- [ ] Configuration editor
- [ ] Alert notifications
- [ ] Historical graphs (longer timeframes)

## 📝 Examples

### Monitor Production Server
```bash
# Check everything is healthy
./cyxwiz-central-server --tui

# Switch to Nodes view (press 2)
# Look for offline nodes (red indicators)
# Check reputation scores

# Switch to Jobs view (press 3)
# Verify jobs are being processed
# Look for failed jobs (red)

# Switch to Logs view (press 5)
# Check for errors
```

### Debug Job Scheduling
```bash
# Run in TUI mode
cargo run -- --tui

# Watch Dashboard for:
# - Pending jobs count increasing
# - Jobs being assigned to nodes
# - Activity log showing assignments

# Switch to Jobs view
# Watch progress bars update
# Check estimated vs actual cost
```

## 🔗 Integration

The TUI and REST API run simultaneously:

```bash
# Terminal 1: Run server normally
cargo run --release

# Terminal 2: Monitor with TUI
cargo run --bin cyxwiz-monitor --release  # (future: separate monitor binary)

# OR: Run with TUI
cargo run --release -- --tui  # (server + TUI in same process)
```

## 📚 Further Reading

- [Main README](README.md) - Full project documentation
- [API Documentation](README.md#api-endpoints) - REST API reference
- [Architecture](README.md#architecture) - System design
- [Troubleshooting](README.md#troubleshooting) - Common issues

---

**Happy Monitoring!** 🚀

For questions or issues, open a ticket at: https://github.com/cyxwiz/cyxwiz/issues
