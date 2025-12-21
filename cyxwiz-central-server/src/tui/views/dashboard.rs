use crate::tui::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // Stats cards
            Constraint::Length(12), // Job throughput graph
            Constraint::Length(8),  // Top nodes table
            Constraint::Min(0),     // Recent activity
        ])
        .split(area);

    render_stats_cards(f, app, chunks[0]);
    render_job_throughput(f, app, chunks[1]);
    render_top_nodes(f, app, chunks[2]);
    render_recent_activity(f, app, chunks[3]);
}

fn render_stats_cards(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left: Network Statistics
    let network_stats = vec![
        format!("Total Nodes:        {}", app.stats.total_nodes),
        format!(
            "Online Nodes:       {} ({:.1}%)",
            app.stats.online_nodes,
            if app.stats.total_nodes > 0 {
                (app.stats.online_nodes as f64 / app.stats.total_nodes as f64) * 100.0
            } else {
                0.0
            }
        ),
        format!("Active Jobs:        {}", app.stats.active_jobs),
        format!("Pending Jobs:       {}", app.stats.pending_jobs),
        format!("Completed (24h):    {}", app.stats.completed_jobs_24h),
    ];

    let network_block = Paragraph::new(network_stats.join("\n"))
        .block(
            Block::default()
                .title("Network Statistics")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .style(Style::default().fg(Color::White));

    f.render_widget(network_block, chunks[0]);

    // Right: System Health
    let db_status = if app.stats.db_healthy {
        Span::styled("● HEALTHY", Style::default().fg(Color::Green))
    } else {
        Span::styled("● DOWN", Style::default().fg(Color::Red))
    };

    let redis_status = if app.stats.redis_healthy {
        Span::styled("● HEALTHY", Style::default().fg(Color::Green))
    } else {
        Span::styled("● DOWN", Style::default().fg(Color::Red))
    };

    let solana_status = if app.stats.solana_healthy {
        Span::styled("● HEALTHY", Style::default().fg(Color::Green))
    } else {
        Span::styled("● DOWN", Style::default().fg(Color::Red))
    };

    let health_lines = vec![
        Line::from(vec![
            Span::raw("PostgreSQL:  "),
            db_status,
            Span::raw(format!("   {}ms", app.stats.db_latency_ms)),
        ]),
        Line::from(vec![
            Span::raw("Redis:       "),
            redis_status,
            Span::raw(format!("   {}ms", app.stats.redis_latency_ms)),
        ]),
        Line::from(vec![
            Span::raw("Solana RPC:  "),
            solana_status,
            Span::raw(format!("   {}ms", app.stats.solana_latency_ms)),
        ]),
        Line::from(Span::raw(format!("gRPC Server: ● RUNNING :50051"))),
        Line::from(Span::raw(format!("REST API:    ● RUNNING :8080"))),
    ];

    let health_block = Paragraph::new(health_lines)
        .block(
            Block::default()
                .title("System Health")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .style(Style::default().fg(Color::White));

    f.render_widget(health_block, chunks[1]);
}

fn render_job_throughput(f: &mut Frame, app: &App, area: Rect) {
    // Mock data for sparkline (in production, use app.job_throughput_history)
    let data: Vec<u64> = vec![50, 55, 60, 58, 70, 75, 80, 85, 90, 88, 95, 100, 110, 105, 115, 120];

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title("Job Throughput (Last Hour)")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .data(&data)
        .style(Style::default().fg(Color::Yellow))
        .max(150);

    f.render_widget(sparkline, area);
}

fn render_top_nodes(f: &mut Frame, app: &App, area: Rect) {
    let header = vec![
        "Rank │ Node ID    │ Reputation │ Jobs   │ Uptime │ GPU          │ Load",
        "─────┼────────────┼────────────┼────────┼────────┼──────────────┼─────",
    ];

    let mut items: Vec<ListItem> = header
        .into_iter()
        .map(|h| ListItem::new(h).style(Style::default().add_modifier(Modifier::BOLD)))
        .collect();

    // Display top 3 nodes
    for (idx, node) in app.nodes.iter().take(3).enumerate() {
        let stars = "★".repeat((node.reputation_score * 5.0) as usize);
        let node_id = &node.id.to_string()[..8];
        let gpu = node
            .gpu_model
            .as_ref()
            .map(|g| {
                let g_short = if g.len() > 12 { &g[..12] } else { g };
                format!("{} {}G", g_short, node.gpu_memory_gb.unwrap_or(0))
            })
            .unwrap_or_else(|| "CPU Only".to_string());

        let load_percent = (node.current_load * 100.0) as u8;
        let load_bar = "▓".repeat((load_percent / 20) as usize);

        let line = format!(
            " {:2}   │ {}  │ {} {:.2} │ {:6} │ {:5.1}% │ {:14} │ {:3}%",
            idx + 1,
            node_id,
            stars,
            node.reputation_score,
            node.total_jobs_completed,
            node.uptime_percentage,
            gpu,
            load_percent
        );

        items.push(ListItem::new(line));
    }

    let list = List::new(items).block(
        Block::default()
            .title("Top Nodes (by Reputation)")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(list, area);
}

fn render_recent_activity(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .rev()
        .take(10)
        .map(|log| {
            let time = log.timestamp.format("%H:%M:%S");
            let (icon, color) = match log.level {
                crate::tui::app::LogLevel::Success => ("✓", Color::Green),
                crate::tui::app::LogLevel::Info => ("→", Color::Blue),
                crate::tui::app::LogLevel::Warn => ("⚠", Color::Yellow),
                crate::tui::app::LogLevel::Error => ("✗", Color::Red),
            };

            let line = format!("[{}] {} {}", time, icon, log.message);
            ListItem::new(line).style(Style::default().fg(color))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title("Recent Activity")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(list, area);
}
