use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, KeyEventKind};
use std::time::Duration;

pub enum AppEvent {
    Tick,
    Key(KeyEvent),
    Resize,
}

pub fn poll_events(timeout: Duration) -> std::io::Result<Option<AppEvent>> {
    if event::poll(timeout)? {
        match event::read()? {
            Event::Key(key) => {
                // Only process key press events, not key release or repeat events
                // This prevents double-processing of Tab and other keys
                if key.kind == KeyEventKind::Press {
                    Ok(Some(AppEvent::Key(key)))
                } else {
                    Ok(None)
                }
            }
            Event::Resize(_, _) => Ok(Some(AppEvent::Resize)),
            _ => Ok(None),
        }
    } else {
        Ok(Some(AppEvent::Tick))
    }
}

pub fn handle_key_event(app: &mut crate::tui::app::App, key: KeyEvent) {
    use crate::tui::app::View;

    match key.code {
        // Quit
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            app.quit();
        }
        KeyCode::Esc => {
            app.quit();
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.quit();
        }

        // Navigation between views
        KeyCode::Tab => {
            app.next_view();
        }
        KeyCode::BackTab => {
            app.previous_view();
        }
        KeyCode::Char('1') => app.current_view = View::Dashboard,
        KeyCode::Char('2') => app.current_view = View::Nodes,
        KeyCode::Char('3') => app.current_view = View::Jobs,
        KeyCode::Char('4') => app.current_view = View::Blockchain,
        KeyCode::Char('5') => app.current_view = View::Logs,
        KeyCode::Char('6') => app.current_view = View::Settings,

        // Navigation within views
        KeyCode::Up | KeyCode::Char('k') => match app.current_view {
            View::Nodes => app.select_previous_node(),
            View::Jobs => app.select_previous_job(),
            _ => {}
        },
        KeyCode::Down | KeyCode::Char('j') => match app.current_view {
            View::Nodes => app.select_next_node(),
            View::Jobs => app.select_next_job(),
            _ => {}
        },

        // Refresh
        KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::F(5) => {
            // Trigger manual refresh (will be handled in update loop)
        }

        _ => {}
    }
}
