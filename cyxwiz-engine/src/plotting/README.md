# Engine plotting code: current state

Status as of 2026-10-02 (TOFIX134 P0 done). User-facing usage is in
`docs/usage/plotting.md`. The replacement design (plot model, data service,
one renderer, Plot and Dashboard nodes sharing one window) is TOFIX134;
decision D3 retires this folder's PlotManager, PlotWindow and the
`cyxwiz_plotting` Python module.

## What works

| Path | Code | Notes |
| --- | --- | --- |
| matplotlib `plt.show()` -> Plot Output | `scripting/scripting_engine.cpp` (capture), `scripting/plot_inbox.h`, `gui/panels/plot_output_panel.cpp` | Script runs publish figures; Plot Output takes them every frame, also while closed. Notebook cells keep figures inline. |
| Table Viewer Quick Plot | `gui/panels/table_viewer_plot.cpp` | ImPlot; "Plot with Python" writes a valid script via `core/plot_script`. |
| Training Dashboard | `gui/panels/training_plot_panel.cpp` | ImPlot directly; draws cached series reduced by `core/series_decimation`. Fed by canvas training only. |
| RL Training Dashboard | `gui/panels/training_dashboard.cpp` | Uses PlotManager; fed by `pycyxwiz.rl_update_metric` through `ThreadInbox` in the scripting engine. |
| Visualization panel, Data Explorer scatter/correlation | `gui/panels/visualization_panel.cpp`, Data Explorer | x/y paired by row (`core/paired_columns`). |
| PlotManager | `plotting/plot_manager.*` | Draws by plot type with the config's labels, legend and grid (`DrawPlot`); test `plot_manager_contract`. |
| MatplotlibBackend | `plotting/backends/matplotlib_backend.cpp` | Private namespace, escaped text, figures closed; test `matplotlib_backend_safety`. |

## What does not work

- `cyxwiz_plotting` (the pybind11 module in `python/`) is a separate
  binary with its own PlotManager: nothing it creates reaches an Engine
  window. `show_plot` raises an error that says so.
- The ~22 plot node templates in the node palette draw nothing; the Bar
  Chart node plots the raw source of the first upstream Data Input.
- No 3D plotting (ImPlot3D is decision D1, later phase).

## Tests

`plot_manager_contract`, `matplotlib_backend_safety`, `test_plot_script_contract`,
`test_paired_columns_contract`, `test_plot_inbox_contract`,
`test_series_decimation_contract` (ctest in build-gui).
