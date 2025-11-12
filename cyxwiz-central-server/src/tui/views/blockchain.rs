use crate::tui::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
    Frame,
};

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

    // Left: Wallet Info
    let wallet_lines = vec![
        Line::from(vec![
            Span::styled("Payer Address: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("4f2a3b...8e9d (mock)"),
        ]),
        Line::from(vec![
            Span::styled("Balance: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled("123.456 SOL", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("CYXWIZ Tokens: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled("45,234.56 CYXWIZ", Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Program ID: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("CyxWiz1111...1111"),
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
    let network_lines = vec![
        Line::from(vec![
            Span::styled("RPC Endpoint: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("api.devnet.solana"),
        ]),
        Line::from(vec![
            Span::styled("Latency: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{}ms", app.stats.solana_latency_ms)),
        ]),
        Line::from(vec![
            Span::styled("Block Height: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("234,567,890"),
        ]),
        Line::from(vec![
            Span::styled("TPS: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("~2,500"),
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

    // Mock transaction data
    let mock_transactions = vec![
        ("16:03:45", "ESCROW", "job-a3f2", "0.456", "✓ CONF", "4f2a3b..."),
        ("16:03:42", "DISTRIBUTE", "job-b8d1", "2.345", "✓ CONF", "8b1c9e..."),
        ("16:03:39", "STREAM", "job-c9a5", "0.123", "⏳ PEND", "2d9f7a..."),
        ("16:03:35", "REFUND", "job-d2f6", "1.234", "✓ CONF", "6e3b4c..."),
    ];

    let rows = mock_transactions.iter().map(|(time, tx_type, job_id, amount, status, sig)| {
        let status_color = if status.contains("✓") {
            Color::Green
        } else if status.contains("⏳") {
            Color::Yellow
        } else {
            Color::Red
        };

        let cells = vec![
            ratatui::widgets::Cell::from(*time),
            ratatui::widgets::Cell::from(*tx_type),
            ratatui::widgets::Cell::from(*job_id),
            ratatui::widgets::Cell::from(format!("{} CYX", amount)),
            ratatui::widgets::Cell::from(*status).style(Style::default().fg(status_color)),
            ratatui::widgets::Cell::from(*sig),
        ];

        Row::new(cells).height(1)
    });

    let widths = [
        Constraint::Length(8),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .title("Recent Transactions")
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
