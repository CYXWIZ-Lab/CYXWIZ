use super::{
    app::{App, LogLevel, View},
    events::{handle_key_event, poll_events, AppEvent},
    updater::update_app_data,
    views,
};
use crate::blockchain::SolanaClient;
use crate::cache::RedisCache;
use crate::error::Result;
use crossterm::{
    event::DisableMouseCapture,
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs},
    Terminal,
};
use crate::database::DbPool;
use std::{
    io::{self, Stdout},
    sync::Arc,
    time::Duration,
};
use tokio::sync::RwLock;

pub async fn run(
    db_pool: DbPool,
    cache: Arc<RwLock<RedisCache>>,
    solana_client: Option<Arc<SolanaClient>>,
) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new(db_pool, cache, solana_client);
    app.add_log(LogLevel::Success, "TUI started successfully".to_string());

    // Initial data fetch
    if let Err(e) = update_app_data(&mut app).await {
        app.add_log(LogLevel::Error, format!("Failed to update data: {}", e));
    }

    // Run the UI loop
    let result = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("Error: {}", e);
    }

    Ok(())
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
) -> Result<()> {
    let mut last_update = std::time::Instant::now();
    let update_interval = Duration::from_secs(1);

    loop {
        // Draw UI
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Header
                    Constraint::Length(3), // Tab bar
                    Constraint::Min(0),    // Content
                    Constraint::Length(1), // Footer
                ])
                .split(f.size());

            render_header(f, app, chunks[0]);
            render_tabs(f, app, chunks[1]);
            render_content(f, app, chunks[2]);
            render_footer(f, chunks[3]);
        })?;

        // Handle events
        if let Some(event) = poll_events(Duration::from_millis(100))? {
            match event {
                AppEvent::Key(key) => handle_key_event(app, key),
                AppEvent::Tick => {
                    // Auto-update data every second
                    if last_update.elapsed() >= update_interval {
                        if let Err(e) = update_app_data(app).await {
                            app.add_log(
                                LogLevel::Error,
                                format!("Failed to update data: {}", e),
                            );
                        }
                        last_update = std::time::Instant::now();
                    }
                }
                AppEvent::Resize => {}
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

fn render_header(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                " CyxWiz Central Server ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("v{}  ", env!("CARGO_PKG_VERSION"))),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("  Uptime: {}  ", app.format_uptime())),
            Span::styled("│", Style::default().fg(Color::DarkGray)),
            Span::raw(format!(
                "  Last Update: {}",
                app.last_update.format("%H:%M:%S")
            )),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(header, area);
}

fn render_tabs(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let titles = vec![
        "[1] Dashboard",
        "[2] Nodes",
        "[3] Jobs",
        "[4] Blockchain",
        "[5] Logs",
        "[6] Settings",
    ];

    let selected_index = match app.current_view {
        View::Dashboard => 0,
        View::Nodes => 1,
        View::Jobs => 2,
        View::Blockchain => 3,
        View::Logs => 4,
        View::Settings => 5,
    };

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .select(selected_index)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
                .bg(Color::DarkGray),
        );

    f.render_widget(tabs, area);
}

fn render_content(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    match app.current_view {
        View::Dashboard => views::dashboard::render(f, app, area),
        View::Nodes => views::nodes::render(f, app, area),
        View::Jobs => views::jobs::render(f, app, area),
        View::Blockchain => views::blockchain::render(f, app, area),
        View::Logs => views::logs::render(f, app, area),
        View::Settings => views::settings::render(f, app, area),
    }
}

fn render_footer(f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
    let footer = Paragraph::new(Span::raw(
        " [Tab] Switch Views  [↑↓] Navigate  [R] Refresh  [Q] Quit  [H] Help ",
    ))
    .style(Style::default().fg(Color::DarkGray));

    f.render_widget(footer, area);
}
