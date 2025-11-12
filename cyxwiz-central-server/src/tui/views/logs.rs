use crate::tui::app::App;
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem},
    Frame,
};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .logs
        .iter()
        .rev()
        .map(|log| {
            let time = log.timestamp.format("%H:%M:%S");
            let level = log.level.as_str();

            let (color, icon) = match log.level {
                crate::tui::app::LogLevel::Success => (Color::Green, "✓"),
                crate::tui::app::LogLevel::Info => (Color::Blue, "ℹ"),
                crate::tui::app::LogLevel::Warn => (Color::Yellow, "⚠"),
                crate::tui::app::LogLevel::Error => (Color::Red, "✗"),
            };

            let line = format!("[{}] {} {} {}", time, icon, level, log.message);
            ListItem::new(line).style(Style::default().fg(color))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(format!("Server Logs ({} entries)", app.logs.len()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(list, area);
}
