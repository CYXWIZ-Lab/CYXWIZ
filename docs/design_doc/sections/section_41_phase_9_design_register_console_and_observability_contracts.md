# 41) Phase 9 design register (console and observability contracts)

## 41.1 Objective
Define the end-to-end observability contract from process bootstrap logging into the GUI console, including log sink wiring, UI rendering rules, async task-to-UI event flow, and command execution behavior in the console panel.

## 41.2 ASCII flow: logging and console event path

```text
main()
  -> spdlog logger created with:
      [engine_log.txt sink] + [stdout sink]
  -> logger set as default
  -> backend+plugin init
  -> CyxWizApp.Run()
     -> window + ImGui init
     -> MainWindow created
     -> ConsoleSinkMt added to logger.sinks()
        -> spdlog events fan into console panel live
     -> when app shuts down:
        -> remove extra sinks from logger
        -> destroy UI and context
```

```text
Console::ExecCommand("pip install ...")
  -> AddLog("> pip ...", Info)
  -> ExecutePipCommand()
     -> validate project context
     -> spawn async task via AsyncTaskManager::RunAsync
        -> child process IO captured in task thread
        -> AddInfo/AddSuccess/AddError into Console
     -> completion callback runs on main thread
        -> optional spdlog error path
```

## 41.3 Logging bootstrap contract (entrypoint)

### 41.3.1 Default logger setup in `main.cpp`
- A shared file sink is created at `<cwd>/engine_log.txt`.
- A colored stdout sink is created for console/terminal output.
- Both are passed into one named logger (`"cyxwiz"`) and registered via `spdlog::set_default_logger`.
- Default level is `info`, with `flush_on(info)`.
- `CYXWIZ_DEBUG` changes default level to `debug`.

### 41.3.2 Early observability behavior
- All startup events before UI construction are preserved in file+terminal sinks.
- GUI-specific sink attachment happens later, so no console panel dependency is required for bootstrap.
- Startup logs include launch cwd, device enumeration, and plugin path configuration.

## 41.4 Console panel contract

### 41.4.1 Data model and lifecycle
- `gui::Console` stores `LogEntry { message, level, timestamp }`.
- Log levels are:
  - `Info`, `Warning`, `Error`, `Success`, `Debug`.
- `AddLog` is guarded by mutex (`log_mutex_`).
- History is bounded at 1000 entries; oldest records are erased when exceeded.
- Constructor intentionally avoids logging because it may run before an active ImGui frame.

### 41.4.2 Render contract
- Panel visibility is controlled by `show_window_`; closed = no render.
- Initial first frame emits three bootstrap messages:
  - `CyxWiz Console initialized`
  - `Type 'help' for available commands`
  - `Ready`
- Tab contract:
  - `All`, `Info`, `Warnings`, `Errors`, `Success`.
- Interaction contract:
  - `Clear` empties entries and emits `Console cleared`.
  - `Copy All` serializes full log text and puts it on clipboard.
  - Double click/copy context on any visible line copies message/full line.
  - 2 second green "Copied!" transient state.
- Auto-scroll behavior is shared by all tabs and can be disabled per user.

### 41.4.3 Timestamp and formatting contract
- Timestamp uses `ImGui::GetTime()` at insertion time.
- Prefix and color mapping are deterministic by level:
  - `[INFO]`, `[WARN]`, `[ERROR]`, `[OK]`, `[DEBUG]`.

## 41.5 Spdlog sink bridge contract

- `ConsoleSink<Mutex>` inherits `spdlog::sinks::base_sink`.
- `sink_it_` receives raw spdlog message, applies formatter, strips one trailing newline.
- Level mapping:
  - `trace|debug|info` -> `AddInfo`.
  - `warn` -> `AddWarning`.
  - `err|critical` -> `AddError`.
  - default -> `AddInfo`.
- `flush_` is explicit no-op (GUI updates through live append).

## 41.6 Runtime attachment and detachment contract

### 41.6.1 Attach point
- In `CyxWizApp::Render()`, after `MainWindow` is created and ready:
  - obtain `main_window_->GetConsole()`,
  - emit startup console messages,
  - create `ConsoleSinkMt` and `push_back` into default logger sink list.
- This attachment is late-bound to avoid UI dependency before console exists.

### 41.6.2 Detach point
- In `CyxWizApp::Shutdown()`:
  - remove non-base sinks by resizing `logger->sinks()` to one entry.
  - this prevents dangling console pointer after `main_window_` destruction.

### 41.6.3 Failure-safe teardown
- UI and renderer are destroyed after detaching sink.
- `Shutdown()` terminates via `_exit(0)` after major cleanup to avoid destructor-order side effects.

## 41.7 Console command and process execution contract

### 41.7.1 Command parser
- `ExecCommand` adds command echo `> <cmd>` with `Info`.
- Built-in support:
  - `help`
  - `clear`
  - `test`
- Unknown command path logs explicit error and help reminder.

### 41.7.2 pip command execution contract
- A pip command is detected when input prefix is `pip` or `pip3`.
- Requires active project.
- Resolves pip binary in project virtual env:
  - Windows: `<project>/python/Scripts/pip.exe`
  - Unix: `<project>/python/bin/pip`
- If missing venv pip, it emits explicit error + recovery hint.
- Command is executed as:
  - informational preamble (`Executing`, `Running in background...`)
  - async task in `AsyncTaskManager`.
- Windows path uses `CreatePipe` + `CreateProcessA`, reads stdout/stderr in real time.
- Unix path uses `popen` loop over lines.
- Cancellation is checked from the task object; on cancel:
  - Windows attempts process terminate,
  - Unix closes pipe early,
  - logs `Command cancelled by user`.
- Exit outcome:
  - 0 -> `Command completed successfully` (`AddSuccess`)
  - non-zero -> `Command failed with exit code: X` (`AddError`) and `MarkFailed`.

### 41.7.3 Completion callback contract
- Completion callback may log spdlog error on failure.
- No UI manipulation is done directly inside worker thread.
- Long-running command status remains non-blocking to main ImGui loop.

## 41.8 Async task observability contract

- `AsyncTask::ReportProgress` updates progress/message and records into `TrainingTraceCollector`.
- `RunAsync` marks completed if worker returns without explicit cancel/fail.
- UI consumes task callbacks through:
  - `CyxWizApp::Update()` calls `AsyncTaskManager::ProcessCompletedCallbacks`.
- This guarantees task completion side effects run on main thread.

## 41.9 Lean guardrail assessment
- The design is minimal:
  - one logger surface (default logger + sinks),
  - one UI sink adaptor,
  - one text command parser,
  - one async run path for external process commands.
- Explicitly bounded queue and lifecycle boundaries lower risk of unbounded memory growth.
- Open question / technical debt:
  - `AddLog` depends on `ImGui::GetTime` while being callable from task threads; this is currently synchronized by mutex for container state but relies on ImGui timing stability.

## 41.10 Evidence anchors

| Contract | Evidence |
|---|---|
| Logger bootstrap and default sink list | `cyxwiz-engine/src/main.cpp` |
| Main window and log sink attachment/detachment | `cyxwiz-engine/src/application.cpp` |
| Console panel API, buffering, command model, tabs, copy, and pip hooks | `cyxwiz-engine/src/gui/console.h`, `cyxwiz-engine/src/gui/console.cpp` |
| Spdlog bridge adapter | `cyxwiz-engine/src/gui/console_sink.h` |
| Console access from main window | `cyxwiz-engine/src/gui/main_window.h` |
| Async task threading + completion callback path | `cyxwiz-engine/src/core/async_task_manager.h`, `cyxwiz-engine/src/core/async_task_manager.cpp` |
