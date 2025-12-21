use crate::database::models::PaymentStatus;
use crate::tui::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame,
};

/// Truncate pubkey for display: "4Y5HWB...yAG8"
fn truncate_pubkey(pubkey: &str) -> String {
    if pubkey.len() > 12 {
        format!("{}...{}", &pubkey[..6], &pubkey[pubkey.len()-4..])
    } else {
        pubkey.to_string()
    }
}

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // Wallet info
            Constraint::Min(0),     // Transactions
            Constraint::Length(8),  // Payment stats
        ])
        .split(area);

    render_wallet_info(f, app, chunks[0]);
    render_transactions(f, app, chunks[1]);
    render_payment_stats(f, app, chunks[2]);
}

fn render_wallet_info(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Calculate balance in SOL
    let balance_sol = app.blockchain_info.balance_lamports as f64 / 1_000_000_000.0;

    // Left: Wallet Info
    let wallet_lines = vec![
        Line::from(vec![
            Span::styled("Payer Address: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(truncate_pubkey(&app.blockchain_info.payer_pubkey)),
        ]),
        Line::from(vec![
            Span::styled("Balance: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:.4} SOL", balance_sol), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Program ID: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(truncate_pubkey(&app.blockchain_info.program_id)),
        ]),
        Line::from(vec![
            Span::styled("Status: ", Style::default().add_modifier(Modifier::BOLD)),
            if app.stats.solana_healthy {
                Span::styled("Connected", Style::default().fg(Color::Green))
            } else {
                Span::styled("Disconnected", Style::default().fg(Color::Red))
            },
        ]),
    ];

    let wallet_block = Paragraph::new(wallet_lines).block(
        Block::default()
            .title("Wallet Info")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(wallet_block, chunks[0]);

    // Right: Network Stats
    let latency_color = if app.stats.solana_latency_ms < 200 {
        Color::Green
    } else if app.stats.solana_latency_ms < 500 {
        Color::Yellow
    } else {
        Color::Red
    };

    let network_lines = vec![
        Line::from(vec![
            Span::styled("RPC Endpoint: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("devnet.solana.com"),
        ]),
        Line::from(vec![
            Span::styled("Latency: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}ms", app.stats.solana_latency_ms), Style::default().fg(latency_color)),
        ]),
        Line::from(vec![
            Span::styled("Network: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&app.blockchain_info.network),
        ]),
        Line::from(vec![
            Span::styled("Platform Fee: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("10%"),
        ]),
    ];

    let network_block = Paragraph::new(network_lines).block(
        Block::default()
            .title("Network Stats")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(network_block, chunks[1]);
}

fn render_transactions(f: &mut Frame, app: &App, area: Rect) {
    let header_cells = [
        "Time", "Type", "Job ID", "Amount", "Status", "Signature",
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

    // Use real payment data from app.payments
    let rows: Vec<Row> = if app.payments.is_empty() {
        // Show placeholder when no payments
        vec![Row::new(vec![
            ratatui::widgets::Cell::from("--"),
            ratatui::widgets::Cell::from("--"),
            ratatui::widgets::Cell::from("No transactions"),
            ratatui::widgets::Cell::from("--"),
            ratatui::widgets::Cell::from("--"),
            ratatui::widgets::Cell::from("--"),
        ]).style(Style::default().fg(Color::DarkGray))]
    } else {
        app.payments.iter().take(10).map(|payment| {
            let time = payment.created_at.format("%H:%M:%S").to_string();
            let tx_type = match payment.status {
                PaymentStatus::Locked => "ESCROW",
                PaymentStatus::Completed => "COMPLETE",
                PaymentStatus::Streaming => "STREAM",
                PaymentStatus::Refunded => "REFUND",
                PaymentStatus::Pending => "PENDING",
                PaymentStatus::Failed => "FAILED",
            };
            let job_id = truncate_pubkey(&payment.job_id.to_string());
            let amount_sol = payment.amount as f64 / 1_000_000_000.0;

            let (status_str, status_color) = match payment.status {
                PaymentStatus::Completed => ("CONF", Color::Green),
                PaymentStatus::Locked => ("LOCK", Color::Yellow),
                PaymentStatus::Streaming => ("STRM", Color::Yellow),
                PaymentStatus::Pending => ("PEND", Color::Yellow),
                PaymentStatus::Refunded => ("RFND", Color::Cyan),
                PaymentStatus::Failed => ("FAIL", Color::Red),
            };

            let sig = payment.escrow_tx_hash.as_ref()
                .map(|s| truncate_pubkey(s))
                .unwrap_or_else(|| "-".to_string());

            let cells = vec![
                ratatui::widgets::Cell::from(time),
                ratatui::widgets::Cell::from(tx_type),
                ratatui::widgets::Cell::from(job_id),
                ratatui::widgets::Cell::from(format!("{:.4} SOL", amount_sol)),
                ratatui::widgets::Cell::from(status_str).style(Style::default().fg(status_color)),
                ratatui::widgets::Cell::from(sig),
            ];

            Row::new(cells).height(1)
        }).collect()
    };

    let widths = [
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(6),
        Constraint::Length(14),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title(format!("Recent Transactions ({})", app.payments.len()))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );

    f.render_widget(table, area);
}

fn render_payment_stats(f: &mut Frame, _app: &App, area: Rect) {
    let stats = vec![
        Line::from(vec![
            Span::styled(
                "Payment Distribution (Last 24h)",
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Total Volume: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled("1,234.56 CYXWIZ", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::raw("  ████████████████████████████████████████████ 90% Nodes"),
        ]),
        Line::from(vec![Span::raw("  ████ 10% Platform Fee")]),
    ];

    let paragraph = Paragraph::new(stats).block(
        Block::default()
            .title("Payment Distribution")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(paragraph, area);
}
