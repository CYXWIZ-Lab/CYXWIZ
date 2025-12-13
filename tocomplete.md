# CyxWiz Engine - Incomplete TODOs

> **Generated:** December 14, 2025
> **Total Items:** ~140+ unimplemented features
> **Source:** `cyxwiz-engine/src/` (excludes `external/`)

---

## Summary

| Category | Count | Priority |
|----------|-------|----------|
| GUI Panels/Toolbar | ~80 | Medium |
| Plotting | ~15 | Medium |
| Asset Browser | ~9 | Low |
| Model Import/Export | ~7 | **High** |
| Scripting/Python | ~7 | Medium |
| Core/Application | ~5 | Low |
| Wallet/Blockchain | ~4 | Low |
| Node Editor | ~4 | Medium |
| Platform-specific | ~4 | Medium |

---

## High Priority

### Model Import/Export

- [ ] `core/model_exporter.cpp:122` - Implement optimizer state export when Optimizer::GetState() is added
- [ ] `core/model_exporter.cpp:195` - Implement ONNX export using onnxruntime
- [ ] `core/model_exporter.cpp:422` - Validate that all layer types are supported in ONNX
- [ ] `core/model_exporter.cpp:478` - Get CyxWiz version from version header
- [ ] `core/model_importer.cpp:280` - Implement ONNX import using onnxruntime
- [ ] `core/formats/cyxmodel_format.cpp:679` - Implement ZIP archive creation with minizip
- [ ] `core/formats/cyxmodel_format.cpp:689` - Implement ZIP archive extraction with minizip

### Scripting Security

- [ ] `scripting/python_sandbox.cpp:189` - Implement file access restrictions
- [ ] `scripting/python_sandbox.cpp:297` - Get actual memory usage
- [ ] `scripting/python_sandbox.cpp:321` - Check memory limit (requires platform-specific code)
- [ ] `scripting/python_sandbox.cpp:357` - Implement proper timeout using Python signal module or subprocess

---

## Medium Priority

### GUI - Toolbar Nodes Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 26 - Add dense layer
- [ ] Line 29 - Add conv layer
- [ ] Line 32 - Add pooling layer
- [ ] Line 35 - Add dropout
- [ ] Line 38 - Add batch norm
- [ ] Line 41 - Add attention
- [ ] Line 49 - Group nodes
- [ ] Line 53 - Ungroup nodes
- [ ] Line 59 - Duplicate selected
- [ ] Line 63 - Delete selected nodes

### GUI - Toolbar Train Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 83 - Start training
- [ ] Line 87 - Pause training
- [ ] Line 91 - Stop training
- [ ] Line 97 - Open training settings
- [ ] Line 101 - Open optimizer settings

### GUI - Toolbar Dataset Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 117 - Create dataset
- [ ] Line 123 - Preprocess dataset
- [ ] Line 127 - Tokenize dataset
- [ ] Line 131 - Data augmentation
- [ ] Line 137 - Show statistics

### GUI - Toolbar Script Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 147 - Show Python console
- [ ] Line 153 - Create new script
- [ ] Line 157 - Run script
- [ ] Line 163 - Open script editor

### GUI - Toolbar Deploy Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 296 - Quantize INT8
- [ ] Line 299 - Quantize INT4
- [ ] Line 302 - Quantize FP16
- [ ] Line 316 - Publish model to marketplace

### GUI - Toolbar Help Menu

> File: `gui/panels/toolbar_other_menus.cpp`

- [ ] Line 373 - Open docs
- [ ] Line 377 - Show shortcuts
- [ ] Line 381 - Open API docs
- [ ] Line 387 - Open issue tracker
- [ ] Line 391 - Check updates

### GUI - Plotting

- [ ] `gui/panels/plot_window.cpp:556` - 3D plotting (plot3D)
- [ ] `gui/panels/plot_window.cpp:561` - 3D plotting (surface)
- [ ] `gui/panels/plot_window.cpp:566` - 3D plotting (contour3D)
- [ ] `gui/panels/plot_test_panel.cpp:258` - Implement statistical plots rendering
- [ ] `gui/panels/plot_test_panel.cpp:275` - Add data to plot (need AddRawData method)
- [ ] `gui/panels/plot_test_panel.cpp:286` - Implement Q-Q plot
- [ ] `gui/panels/plot_test_panel.cpp:291` - Implement box plot
- [ ] `gui/panels/plot_test_panel.cpp:414` - Add other data types
- [ ] `gui/panels/plot_test_panel.cpp:458` - Add histogram data to plot
- [ ] `gui/panels/plot_test_panel.cpp:466` - Add scatter data to plot
- [ ] `gui/panels/plot_test_panel.cpp:473` - Add heatmap data to plot
- [ ] `gui/panels/plot_test_panel.cpp:479` - Implement file export
- [ ] `gui/panels/training_plot_panel.cpp:240` - Implement screenshot/export functionality
- [ ] `plotting/backends/implot_backend.cpp:215` - Implement custom box plot rendering
- [ ] `plotting/backends/implot_backend.cpp:325` - Implement screenshot capture using framebuffer
- [ ] `plotting/plot_manager.cpp:437` - Shutdown Python interpreter

### GUI - Memory Monitoring

- [ ] `gui/panels/memory_monitor.cpp:60` - Integrate with ArrayFire or CUDA to get actual GPU memory
- [ ] `gui/panels/memory_panel.cpp:561` - Query actual GPU info from ArrayFire

### GUI - Node Editor

- [ ] `gui/node_editor.cpp:1633` - Add custom overlay drawing for matching but not selected nodes
- [ ] `gui/node_editor_codegen.cpp:19` - Show error dialog to user
- [ ] `gui/node_editor_io.cpp:780` - Show error dialog to user
- [ ] `gui/node_editor_io.cpp:831` - Show error dialog to user

### Scripting

- [ ] `scripting/script_manager.h:2` - Script management, auto-completion, syntax highlighting
- [ ] `scripting/script_manager.cpp:2` - Implement script management
- [ ] `scripting/debugger.cpp:490` - Implement call stack update from Python

---

## Low Priority

### Core/Application

- [ ] `application.cpp:116` - Process command line arguments
- [ ] `application.cpp:184` - ViewportsEnable causes crash on Windows - needs investigation

### GUI - Asset Browser

> File: `gui/asset_browser.cpp`

- [ ] Line 24 - Implement search filtering
- [ ] Line 36 - Show import dialog
- [ ] Line 41 - Create new folder
- [ ] Line 121 - Open asset in appropriate editor
- [ ] Line 127 - Show rename dialog
- [ ] Line 133 - Show confirmation dialog
- [ ] Line 140 - Show properties panel
- [ ] Line 147 - Scan filesystem and rebuild tree
- [ ] Line 151 - Add asset to the appropriate category

### GUI - Console

- [ ] `gui/console.cpp:269` - Integrate with Python engine or command processor

### GUI - Toolbar View Menu

- [ ] `gui/panels/toolbar_view_menu.cpp:230` - Toggle fullscreen mode

### GUI - Toolbar Main

- [ ] `gui/panels/toolbar.cpp:661` - Actual authentication API call
- [ ] `gui/panels/toolbar.cpp:956` - Find previous
- [ ] `gui/panels/toolbar.cpp:1221` - Implement actual replace in files (requires file modification)

### GUI - Misc Panels

- [ ] `gui/panels/custom_node_editor.cpp:251` - Prompt to save
- [ ] `gui/panels/custom_node_editor.cpp:329` - Open file dialog
- [ ] `gui/panels/dataset_panel.cpp:2178` - Trigger connection
- [ ] `gui/panels/feature_scaling_panel.cpp:490` - Implement export functionality
- [ ] `gui/panels/output_renderer.cpp:587` - Parse table data from output.data
- [ ] `gui/panels/theme_editor.cpp:507` - Open file dialog

### GUI - Script Editor

- [ ] `gui/panels/script_editor.cpp:2251` - Implement backward search

### GUI - Wallet/Blockchain

- [ ] `gui/panels/wallet_panel.cpp:101` - Open URL in browser
- [ ] `gui/panels/wallet_panel.cpp:251` - Call gRPC WalletService.ConnectWallet
- [ ] `gui/panels/wallet_panel.cpp:276` - Call gRPC WalletService.GetBalance
- [ ] `gui/panels/wallet_panel.cpp:283` - Call gRPC WalletService.GetTransactionHistory

### Data Handling

- [ ] `data/data_table.cpp:373` - Implement Excel loading via Python openpyxl

### NAS

- [ ] `core/nas_evaluator.cpp:536` - Update links to include new node

---

## Platform-Specific

> These need implementation for Linux/macOS

- [ ] `gui/panels/script_editor.cpp:1571` - Implement for Linux/macOS using native dialogs or portable file browser
- [ ] `gui/panels/script_editor.cpp:1596` - Implement for Linux/macOS
- [ ] `gui/panels/test_results_panel.cpp:500` - Add platform-specific file dialog for other platforms
- [ ] `gui/panels/test_results_panel.cpp:524` - Add platform-specific file dialog for other platforms

---

## Notes

1. **Toolbar menu items** are the largest category - many menu items currently just have placeholder TODOs
2. **ONNX import/export** is critical for interoperability with other ML frameworks
3. **Python sandbox security** needs attention for safe script execution
4. **GPU memory monitoring** requires ArrayFire/CUDA API integration
5. Some TODOs may be partially implemented - verify before starting work

---

## Progress Tracking

Use this file to track completion. Mark items with `[x]` when done and commit.

Last updated: December 14, 2025
