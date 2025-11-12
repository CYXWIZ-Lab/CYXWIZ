use crate::tui::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Row, Table},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(10)])
        .split(area);

    render_nodes_table(f, app, chunks[0]);

    // Render node details if a node is selected
    if let Some(node) = app.get_selected_node() {
        render_node_details(f, node, chunks[1]);
    }
}

fn render_nodes_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = [
        "ID", "Name", "Status", "Rep.", "Jobs", "GPU", "Load", "Location",
    ]
    .iter()
    .map(|h| {
        ratatui::widgets::Cell::from(*h).style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
    });

    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let rows = app.nodes.iter().enumerate().map(|(idx, node)| {
        let node_id_str = node.id.to_string();
        let node_id = if node_id_str.len() >= 8 {
            node_id_str[..8].to_string()
        } else {
            node_id_str.clone()
        };
        let name = if node.name.len() > 10 {
            format!("{}...", &node.name[..7])
        } else {
            node.name.clone()
        };

        let status_text = match node.status {
            crate::database::models::NodeStatus::Online => "● ONL",
            crate::database::models::NodeStatus::Offline => "○ OFF",
            crate::database::models::NodeStatus::Busy => "● BSY",
            crate::database::models::NodeStatus::Maintenance => "⚠ MNT",
        };

        let status_color = match node.status {
            crate::database::models::NodeStatus::Online => Color::Green,
            crate::database::models::NodeStatus::Offline => Color::Gray,
            crate::database::models::NodeStatus::Busy => Color::Yellow,
            crate::database::models::NodeStatus::Maintenance => Color::Magenta,
        };

        let gpu = node
            .gpu_model
            .as_ref()
            .map(|g| {
                let short = if g.len() > 10 { &g[..10] } else { g };
                format!("{} {}G", short, node.gpu_memory_gb.unwrap_or(0))
            })
            .unwrap_or_else(|| "CPU Only".to_string());

        let load_bars = "▓".repeat(((node.current_load * 4.0) as usize).min(4));
        let load_empty = "░".repeat(4 - load_bars.len());
        let load_display = format!("{}{}", load_bars, load_empty);

        let region = node.region.as_ref().map(|r| {
            if r.len() > 2 { &r[..2] } else { r }
        }).unwrap_or("??");

        let cells = vec![
            ratatui::widgets::Cell::from(node_id),
            ratatui::widgets::Cell::from(name),
            ratatui::widgets::Cell::from(status_text).style(Style::default().fg(status_color)),
            ratatui::widgets::Cell::from(format!("{:.2}", node.reputation_score)),
            ratatui::widgets::Cell::from(format!("{}", node.total_jobs_completed)),
            ratatui::widgets::Cell::from(gpu),
            ratatui::widgets::Cell::from(load_display),
            ratatui::widgets::Cell::from(region.to_string()),
        ];

        let style = if idx == app.selected_node_index {
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        Row::new(cells).style(style).height(1)
    });

    let widths = [
        Constraint::Length(8),
        Constraint::Length(12),
        Constraint::Length(6),
        Constraint::Length(5),
        Constraint::Length(6),
        Constraint::Length(15),
        Constraint::Length(6),
        Constraint::Length(4),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title(format!(
                    "Nodes View ({} Total, {} Online)",
                    app.stats.total_nodes, app.stats.online_nodes
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(table, area);
}

fn render_node_details(f: &mut Frame, node: &crate::database::models::Node, area: Rect) {
    let details = vec![
        Line::from(vec![
            Span::styled("Node ID: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(node.id.to_string()),
        ]),
        Line::from(vec![
            Span::styled("Wallet: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&node.wallet_address),
        ]),
        Line::from(vec![
            Span::styled("Hardware: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "{} CPU cores, {} GB RAM",
                node.cpu_cores, node.ram_gb
            )),
        ]),
        Line::from(vec![
            Span::styled("GPU: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(
                node.gpu_model
                    .as_ref()
                    .map(|g| format!("{} ({}GB)", g, node.gpu_memory_gb.unwrap_or(0)))
                    .unwrap_or_else(|| "None".to_string()),
            ),
        ]),
        Line::from(vec![
            Span::styled("Reputation: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{:.2}", node.reputation_score),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Jobs: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "{} completed, {} failed",
                node.total_jobs_completed, node.total_jobs_failed
            )),
        ]),
        Line::from(vec![
            Span::styled("Uptime: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{:.2}%", node.uptime_percentage)),
        ]),
    ];

    let paragraph = Paragraph::new(details).block(
        Block::default()
            .title("Selected Node Details")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );

    f.render_widget(paragraph, area);
}
