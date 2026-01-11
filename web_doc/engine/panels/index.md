# Engine Panels Reference

Complete reference for all GUI panels in the CyxWiz Engine desktop application.

## Panel Overview

| Panel | Description | Default Location |
|-------|-------------|------------------|
| [Node Editor](../node-editor/index.md) | Visual ML pipeline builder | Center |
| [Properties](#properties-panel) | Node/selection properties | Right |
| [Console](#console-panel) | Python REPL and output | Bottom |
| [Asset Browser](#asset-browser) | Project file navigation | Left |
| [Script Editor](#script-editor) | Code editor | Center (tab) |
| [Table Viewer](#table-viewer) | Data table viewing | Center (tab) |
| [Training Dashboard](#training-dashboard) | Training visualization | Center (tab) |
| [Plot Window](#plot-window) | Data visualization | Floating |
| [Dataset Manager](#dataset-manager) | Dataset configuration | Right (tab) |

## Core Panels

### Properties Panel

Displays and edits properties of selected nodes or objects.

```
+------------------------------------------------------------------+
|  Properties                                                [x]    |
+------------------------------------------------------------------+
|  SELECTED: Dense Layer (node_15)                                  |
|                                                                   |
|  LAYER CONFIGURATION                                              |
|  +-----------------------------------------------------------+   |
|  | Property      | Value                                      |   |
|  +-----------------------------------------------------------+   |
|  | Name          | [dense_1                ]                  |   |
|  | Units         | [128                    ]                  |   |
|  | Activation    | [ReLU                v]                    |   |
|  | Use Bias      | [x]                                        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  INITIALIZATION                                                   |
|  +-----------------------------------------------------------+   |
|  | Kernel Init   | [He Normal           v]                    |   |
|  | Bias Init     | [Zeros               v]                    |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  REGULARIZATION                                                   |
|  +-----------------------------------------------------------+   |
|  | Kernel L2     | [0.0001               ]                    |   |
|  | Bias L2       | [0.0                  ]                    |   |
|  | Activity L2   | [0.0                  ]                    |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  SHAPE INFERENCE                                                  |
|  +-----------------------------------------------------------+   |
|  | Input Shape   | (batch, 784)                               |   |
|  | Output Shape  | (batch, 128)                               |   |
|  | Parameters    | 100,480                                    |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Apply]  [Reset]  [Delete Node]                                  |
+------------------------------------------------------------------+
```

**Features:**
- Dynamic property display based on selection
- Real-time shape inference
- Parameter count calculation
- Validation with error highlighting

### Console Panel

Interactive Python console with output display.

```
+------------------------------------------------------------------+
|  Console                                            [Clear] [x]   |
+------------------------------------------------------------------+
|  >>> import pycyxwiz as cyx                                         |
|  >>> model = cyx.Sequential()                                     |
|  >>> model.add(cyx.layers.Dense(128, activation='relu'))          |
|  >>> model.summary()                                              |
|  Model: Sequential                                                |
|  _________________________________________________________________|
|  Layer (type)                Output Shape              Param #    |
|  =================================================================|
|  dense (Dense)               (None, 128)               100480     |
|  =================================================================|
|  Total params: 100,480                                            |
|  Trainable params: 100,480                                        |
|  Non-trainable params: 0                                          |
|  _________________________________________________________________|
|  >>> # Training example                                           |
|  >>> history = model.fit(X_train, y_train, epochs=10)             |
|  Epoch 1/10 - loss: 0.5432 - accuracy: 0.8234                     |
|  Epoch 2/10 - loss: 0.3456 - accuracy: 0.8912                     |
|  ...                                                              |
|                                                                   |
|  +-----------------------------------------------------------+   |
|  | >>> _                                                      |   |
|  +-----------------------------------------------------------+   |
|  [Run]  [Cancel]  [History: ^/v]                                  |
+------------------------------------------------------------------+
```

**Features:**
- Syntax highlighting
- Auto-completion (Tab)
- Command history (Up/Down arrows)
- Async execution with cancellation
- Variable inspection
- Output capture (stdout/stderr)

### Asset Browser

Project file navigation and management.

```
+------------------------------------------------------------------+
|  Asset Browser                                             [x]    |
+------------------------------------------------------------------+
|  Project: MyMLProject                                             |
|  +-----------------------------------------------------------+   |
|  | Filter: [All Files     v]  [Search...             ]        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  PROJECT ROOT                                                     |
|  +- models/                                                       |
|  |  +- classifier_v1.cyxgraph                                     |
|  |  +- classifier_v2.cyxgraph                                     |
|  |  +- autoencoder.cyxgraph                                       |
|  +- data/                                                         |
|  |  +- train.csv                                                  |
|  |  +- test.csv                                                   |
|  |  +- images/                                                    |
|  |     +- class_0/                                                |
|  |     +- class_1/                                                |
|  +- scripts/                                                      |
|  |  +- preprocess.py                                              |
|  |  +- train.py                                                   |
|  |  +- evaluate.py                                                |
|  +- outputs/                                                      |
|     +- checkpoints/                                               |
|     +- logs/                                                      |
|                                                                   |
|  SELECTED: train.csv                                              |
|  Size: 12.5 MB | Modified: 2024-01-15 14:32                       |
|                                                                   |
|  [Open]  [View in Table]  [Delete]  [Rename]                      |
+------------------------------------------------------------------+
```

**Features:**
- Hierarchical file tree
- File type filtering (Scripts, Models, Datasets, etc.)
- Search functionality
- Context menu actions
- Drag-and-drop support
- Quick preview

### Script Editor

Full-featured code editor for Python and CyxWiz scripts.

```
+------------------------------------------------------------------+
|  Script Editor - train.py *                              [x]      |
+------------------------------------------------------------------+
|  File: /projects/MyML/scripts/train.py  [Save] [Run] [Debug]      |
+------------------------------------------------------------------+
|  1 | import pycyxwiz as cyx                                         |
|  2 | import pycyxwiz.transforms as T                                |
|  3 |                                                              |
|  4 | # Load dataset                                               |
|  5 | train_data = cyx.datasets.load('data/train.csv')             |
|  6 | test_data = cyx.datasets.load('data/test.csv')               |
|  7 |                                                              |
|  8 | # Define transforms                                          |
|  9 | transform = T.Compose([                                      |
| 10 |     T.StandardScaler(),                                      |
| 11 |     T.ToTensor()                                             |
| 12 | ])                                                           |
| 13 |                                                              |
| 14 | # Create model from graph                                    |
| 15 | model = cyx.load_graph('models/classifier_v1.cyxgraph')      |
| 16 |                                                              |
| 17 | # Train                                                      |
| 18 | model.compile(                                               |
| 19 |     optimizer=cyx.optimizers.Adam(lr=0.001),                 |
| 20 |     loss='categorical_crossentropy',                         |
| 21 |     metrics=['accuracy']                                     |
| 22 | )                                                            |
| 23 |                                                              |
| 24 | history = model.fit(                                         |
| 25 |     train_data,                                              |
| 26 |     epochs=50,                                               |
| 27 |     validation_data=test_data,                               |
| 28 |     callbacks=[                                              |
| 29 |         cyx.callbacks.EarlyStopping(patience=5),             |
| 30 |         cyx.callbacks.ModelCheckpoint('best_model.h5')       |
| 31 |     ]                                                        |
| 32 | )                                                            |
+------------------------------------------------------------------+
|  Ln 24, Col 12  |  Python  |  UTF-8  |  Spaces: 4                 |
+------------------------------------------------------------------+
```

**Features:**
- Syntax highlighting (Python, JSON, YAML)
- Auto-indentation
- Bracket matching
- Line numbers
- Async file loading with progress
- Multiple tabs
- Unsaved changes indicator

### Table Viewer

Data table viewing and analysis.

```
+------------------------------------------------------------------+
|  Table Viewer                                              [x]    |
+------------------------------------------------------------------+
|  [train.csv] [test.csv] [predictions.csv] [+]                     |
+------------------------------------------------------------------+
|  Rows: 10,000  |  Columns: 15  |  Size: 12.5 MB                   |
|                                                                   |
|  Filter: [column:price > 1000            ]  [Apply] [Clear]       |
|                                                                   |
|  +-----------------------------------------------------------+   |
|  |   | id    | name       | price   | category | stock | ... |   |
|  +-----------------------------------------------------------+   |
|  | 1 | P001  | Widget A   | 29.99   | Home     | 150   | ... |   |
|  | 2 | P002  | Widget B   | 49.99   | Tech     | 89    | ... |   |
|  | 3 | P003  | Widget C   | 19.99   | Home     | 234   | ... |   |
|  | 4 | P004  | Widget D   | 99.99   | Tech     | 45    | ... |   |
|  | 5 | P005  | Widget E   | 14.99   | Garden   | 567   | ... |   |
|  | 6 | P006  | Widget F   | 79.99   | Tech     | 23    | ... |   |
|  | 7 | P007  | Widget G   | 34.99   | Home     | 189   | ... |   |
|  | 8 | P008  | Widget H   | 59.99   | Garden   | 78    | ... |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Page: [< 1 2 3 4 5 ... 100 >]  Rows per page: [100 v]            |
|                                                                   |
|  COLUMN STATISTICS (price)                                        |
|  Mean: 45.67 | Median: 39.99 | Std: 23.45 | Min: 9.99 | Max: 199.99|
|                                                                   |
|  [Export Selection]  [Sort]  [Statistics]  [Visualize]            |
+------------------------------------------------------------------+
```

**Features:**
- Multi-tab interface
- Pagination for large files
- Column sorting
- Filtering with expressions
- Column statistics
- Selection and export
- Async loading with progress

### Training Dashboard

Real-time training visualization and control.

```
+------------------------------------------------------------------+
|  Training Dashboard                                        [x]    |
+------------------------------------------------------------------+
|  Model: Sequential_classifier  |  Status: Training  |  Epoch: 23/50|
+------------------------------------------------------------------+
|                                                                   |
|  LOSS                                 ACCURACY                    |
|  +----------------------------+  +----------------------------+   |
|  | 2.0|*                      |  | 1.0|          ************|   |
|  |    | *  Train              |  |    |      ****            |   |
|  | 1.5|  **                   |  | 0.8|    ***               |   |
|  |    |    ** Val             |  |    |  **                  |   |
|  | 1.0|      ***              |  | 0.6|**                    |   |
|  |    |         ****          |  |    |                      |   |
|  | 0.5|             *****     |  | 0.4|                      |   |
|  |    |                  ***  |  |    |  Train --- Val       |   |
|  +----------------------------+  +----------------------------+   |
|                                                                   |
|  CURRENT METRICS                                                  |
|  +-----------------------------------------------------------+   |
|  | Metric           | Train      | Validation  | Best        |   |
|  +-----------------------------------------------------------+   |
|  | Loss             | 0.3456     | 0.4123      | 0.3234      |   |
|  | Accuracy         | 0.8912     | 0.8567      | 0.8789      |   |
|  | Learning Rate    | 0.0001     | -           | -           |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  PROGRESS                                                         |
|  Epoch: [========================>         ] 23/50 (46%)          |
|  Batch: [=============>                    ] 156/500 (31%)        |
|  ETA: 12 min 34 sec                                               |
|                                                                   |
|  [Pause]  [Stop]  [Save Checkpoint]  [Adjust LR]                  |
+------------------------------------------------------------------+
```

**Features:**
- Real-time loss/metric plotting
- Training controls (pause, stop)
- Checkpoint saving
- Learning rate adjustment
- Resource monitoring
- ETA calculation

### Dataset Manager

Dataset configuration and memory management.

```
+------------------------------------------------------------------+
|  Dataset Manager                                           [x]    |
+------------------------------------------------------------------+
|  LOADED DATASETS                                                  |
|  +-----------------------------------------------------------+   |
|  | Name          | Type   | Shape         | Memory  | Status |   |
|  +-----------------------------------------------------------+   |
|  | train_images  | Image  | (60000,28,28) | 45.0 MB | Loaded |   |
|  | train_labels  | Array  | (60000,)      | 0.2 MB  | Loaded |   |
|  | test_images   | Image  | (10000,28,28) | 7.5 MB  | Loaded |   |
|  | test_labels   | Array  | (10000,)      | 0.04 MB | Loaded |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  Total Memory: 52.74 MB / 500 MB limit                            |
|  [==========                                        ] 10.5%       |
|                                                                   |
|  DATASET CONFIGURATION                                            |
|  Selected: train_images                                           |
|  +-----------------------------------------------------------+   |
|  | Batch Size        | [32                   ]                |   |
|  | Shuffle           | [x]                                   |   |
|  | Drop Last         | [ ]                                   |   |
|  | Num Workers       | [4                    ]                |   |
|  | Prefetch Factor   | [2                    ]                |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  TRANSFORMS                                                       |
|  +-----------------------------------------------------------+   |
|  | 1. Normalize      | mean=0.5, std=0.5                      |   |
|  | 2. RandomFlip     | horizontal=True                       |   |
|  | 3. ToTensor       | dtype=float32                         |   |
|  +-----------------------------------------------------------+   |
|  [Add Transform]  [Edit]  [Remove]                                |
|                                                                   |
|  [Load Dataset]  [Unload]  [Preview]  [Apply to Training]         |
+------------------------------------------------------------------+
```

**Features:**
- Dataset loading and unloading
- Memory limit management (LRU eviction)
- Transform pipeline configuration
- Batch configuration
- Preview functionality
- Integration with training

### Plot Window

Standalone plotting window for data visualization.

```
+------------------------------------------------------------------+
|  Plot Window                                               [x]    |
+------------------------------------------------------------------+
|  [+] Add Series  [Clear]  [Export]  [Settings]                    |
+------------------------------------------------------------------+
|                                                                   |
|  LOSS OVER TRAINING                                               |
|                                                                   |
|  Loss                                                             |
|  2.5 |                                                            |
|      | *                                                          |
|  2.0 |  *                                                         |
|      |   *                                                        |
|  1.5 |    **                                                      |
|      |      **                                                    |
|  1.0 |        ***                                                 |
|      |           ****                                             |
|  0.5 |               *****                                        |
|      |                    ********                                |
|  0.0 +------------------------------------------------------>     |
|      0        20        40        60        80       100          |
|                           Epoch                                   |
|                                                                   |
|  Legend: [*] Training Loss  [+] Validation Loss                   |
|                                                                   |
|  TOOLS                                                            |
|  [Zoom In] [Zoom Out] [Pan] [Reset] [Select Region]               |
|                                                                   |
|  Data Points: 100  |  X Range: [0, 100]  |  Y Range: [0, 2.5]     |
+------------------------------------------------------------------+
```

**Features:**
- Multiple plot types (line, scatter, bar, histogram)
- Interactive zoom and pan
- Multiple series support
- Legend management
- Export to image/data

## Panel Management

### Opening Panels

| Method | Description |
|--------|-------------|
| **View Menu** | `View > Panels > [Panel Name]` |
| **Command Palette** | `Ctrl+Shift+P > "Show [Panel]"` |
| **Keyboard Shortcut** | Panel-specific shortcuts |

### Default Shortcuts

| Panel | Shortcut |
|-------|----------|
| Properties | `Ctrl+1` |
| Console | `Ctrl+2` |
| Asset Browser | `Ctrl+3` |
| Node Editor | `Ctrl+4` |
| Script Editor | `Ctrl+5` |
| Table Viewer | `Ctrl+6` |
| Training Dashboard | `Ctrl+7` |

### Layout Management

- **Save Layout**: `View > Save Layout As...`
- **Load Layout**: `View > Load Layout > [Name]`
- **Reset Layout**: `View > Reset Layout`
- **Default Layouts**: Development, Training, Analysis

---

**Next**: [Node Editor](../node-editor/index.md) | [Menus Reference](../menus.md)
