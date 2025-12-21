use crate::tui::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(12)])
        .split(area);

    render_jobs_table(f, app, chunks[0]);

    // Render job details if a job is selected
    if let Some(job) = app.get_selected_job() {
        render_job_details(f, job, chunks[1]);
    }
}

fn render_jobs_table(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = [
        "Job ID", "User", "Type", "Status", "Node", "Progress", "Cost",
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

    let rows = app.jobs.iter().enumerate().map(|(idx, job)| {
        let job_id_str = job.id.to_string();
        let job_id = if job_id_str.len() >= 8 {
            job_id_str[..8].to_string()
        } else {
            job_id_str.clone()
        };
        let user_short = if job.user_wallet.len() > 8 {
            format!("{}...", &job.user_wallet[..5])
        } else {
            job.user_wallet.clone()
        };

        let (status_text, status_color) = match job.status {
            crate::database::models::JobStatus::Pending => ("⏸ PENDING", Color::Gray),
            crate::database::models::JobStatus::Assigned => ("→ ASSIGNED", Color::Blue),
            crate::database::models::JobStatus::Running => ("▶ RUNNING", Color::Green),
            crate::database::models::JobStatus::Completed => ("✓ COMPLETE", Color::Cyan),
            crate::database::models::JobStatus::Failed => ("✗ FAILED", Color::Red),
            crate::database::models::JobStatus::Cancelled => ("⊗ CANCELLED", Color::Magenta),
        };

        let node_display = job
            .assigned_node_id
            .as_ref()
            .map(|id| {
                let node_str = id.to_string();
                if node_str.len() >= 8 {
                    node_str[..8].to_string()
                } else {
                    node_str
                }
            })
            .unwrap_or_else(|| "-".to_string());

        // Calculate progress (mock for now)
        let progress = match job.status {
            crate::database::models::JobStatus::Completed => 100,
            crate::database::models::JobStatus::Running => 50,
            crate::database::models::JobStatus::Assigned => 10,
            _ => 0,
        };

        let progress_bars = "▓".repeat((progress / 12).min(8));
        let progress_empty = "░".repeat(8 - progress_bars.len());
        let progress_display = format!("{}{}", progress_bars, progress_empty);

        let cost_display = format!("{:.3}", job.estimated_cost as f64 / 1_000_000_000.0);

        let cells = vec![
            ratatui::widgets::Cell::from(job_id),
            ratatui::widgets::Cell::from(user_short),
            ratatui::widgets::Cell::from(job.job_type.clone()),
            ratatui::widgets::Cell::from(status_text).style(Style::default().fg(status_color)),
            ratatui::widgets::Cell::from(node_display),
            ratatui::widgets::Cell::from(progress_display),
            ratatui::widgets::Cell::from(cost_display),
        ];

        let style = if idx == app.selected_job_index {
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
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(8),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title(format!(
                    "Jobs View ({} Active, {} Pending)",
                    app.stats.active_jobs, app.stats.pending_jobs
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

fn render_job_details(f: &mut Frame, job: &crate::database::models::Job, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(2)])
        .split(area);

    let details = vec![
        Line::from(vec![
            Span::styled("Job ID: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(job.id.to_string()),
        ]),
        Line::from(vec![
            Span::styled("User Wallet: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&job.user_wallet),
        ]),
        Line::from(vec![
            Span::styled("Type: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&job.job_type),
        ]),
        Line::from(vec![
            Span::styled("Requirements: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "{} RAM, {}",
                format_gb(job.required_ram_gb),
                if job.required_gpu {
                    format!("GPU ({})", format_gb(job.required_gpu_memory_gb.unwrap_or(0)))
                } else {
                    "CPU Only".to_string()
                }
            )),
        ]),
        Line::from(vec![
            Span::styled("Estimated Cost: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{:.3} CYXWIZ", job.estimated_cost as f64 / 1_000_000_000.0),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::styled("Duration: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format_duration(job.estimated_duration_seconds)),
        ]),
        Line::from(vec![
            Span::styled("Retry Count: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{}/3", job.retry_count)),
        ]),
    ];

    let paragraph = Paragraph::new(details).block(
        Block::default()
            .title("Selected Job Details")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );

    f.render_widget(paragraph, chunks[0]);

    // Progress gauge
    let progress = match job.status {
        crate::database::models::JobStatus::Completed => 100,
        crate::database::models::JobStatus::Running => 50,
        crate::database::models::JobStatus::Assigned => 10,
        _ => 0,
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::NONE))
        .gauge_style(
            Style::default()
                .fg(Color::Green)
                .bg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .percent(progress)
        .label(format!("{}%", progress));

    f.render_widget(gauge, chunks[1]);
}

fn format_gb(gb: i32) -> String {
    format!("{}GB", gb)
}

fn format_duration(seconds: i32) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;

    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}
