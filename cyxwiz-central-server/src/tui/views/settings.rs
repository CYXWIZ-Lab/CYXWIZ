use crate::tui::app::App;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let settings = vec![
        Line::from(vec![
            Span::styled("Configuration", Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow)),
        ]),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Server:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("  gRPC Address:    0.0.0.0:50051")),
        Line::from(Span::raw("  REST Address:    0.0.0.0:8080")),
        Line::from(Span::raw("  Max Connections: 1000")),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Database:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("  URL:             postgres://localhost/cyxwiz")),
        Line::from(Span::raw("  Max Connections: 20")),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Redis:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("  URL:             redis://localhost:6379")),
        Line::from(Span::raw("  Pool Size:       10")),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Blockchain:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("  Network:         Solana Devnet")),
        Line::from(Span::raw("  RPC:             api.devnet.solana.com")),
        Line::from(Span::raw("")),
        Line::from(vec![
            Span::styled("Scheduler:", Style::default().add_modifier(Modifier::BOLD)),
        ]),
        Line::from(Span::raw("  Poll Interval:   1000ms")),
        Line::from(Span::raw("  Heartbeat Timeout: 30000ms")),
        Line::from(Span::raw("  Max Retries:     3")),
    ];

    let paragraph = Paragraph::new(settings).block(
        Block::default()
            .title("Settings & Configuration")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(paragraph, area);
}
