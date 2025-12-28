# Visualization Tools

Comprehensive plotting and visualization tools powered by ImPlot for real-time data exploration and analysis.

## Overview

The Visualization tools provide:
- **2D Plotting** - Line, scatter, bar, histogram
- **Statistical Plots** - Box plots, violin plots, heatmaps
- **Training Visualization** - Loss curves, metrics, learning rates
- **3D Visualization** - Surface plots, point clouds
- **Interactive Features** - Zoom, pan, tooltips, data selection

## Tools Reference

### Plot Window Panel

```
+------------------------------------------------------------------+
|  Plot Window                                               [x]    |
+------------------------------------------------------------------+
|  [+] Add Plot  [Save Image]  [Export Data]  [Settings]            |
+------------------------------------------------------------------+
|                                                                   |
|  +-----------------------------------+  +----------------------+  |
|  | Training Loss                     |  | Accuracy             |  |
|  |                                   |  |                      |  |
|  | Loss                              |  | Acc                  |  |
|  | 2.0|*                             |  | 1.0|          ******|  |
|  |    | *                            |  |    |      ****      |  |
|  | 1.5|  *                           |  | 0.8|    ***         |  |
|  |    |   **                         |  |    |  **            |  |
|  | 1.0|     ***                      |  | 0.6|**              |  |
|  |    |        ****                  |  |    |               |  |
|  | 0.5|            *****             |  | 0.4|                |  |
|  |    |                 ********     |  |    |                |  |
|  | 0.0+------------------------->    |  | 0.2+--------------->|  |
|  |    0   20   40   60   80  100     |  |    0  20  40  60 80 |  |
|  |            Epoch                  |  |        Epoch        |  |
|  +-----------------------------------+  +----------------------+  |
|                                                                   |
|  +-----------------------------------+  +----------------------+  |
|  | Learning Rate Schedule            |  | GPU Memory           |  |
|  |                                   |  |                      |  |
|  | LR                                |  | GB                   |  |
|  | 0.01|*******                      |  | 8.0|*****            |  |
|  |     |       *****                 |  |    |     ****        |  |
|  |0.005|            ****             |  | 6.0|         ***     |  |
|  |     |                ****         |  |    |            **** |  |
|  |0.001|                    ******   |  | 4.0|                 |  |
|  |     +------------------------>    |  |    +--------------->|  |
|  +-----------------------------------+  +----------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

### Chart Types

#### Line Plot

```python
import pycyxwiz.plot as plt

# Simple line plot
plt.line(x, y, label="Training Loss")

# Multiple series
plt.line(epochs, train_loss, label="Train")
plt.line(epochs, val_loss, label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Progress")
plt.show()
```

#### Scatter Plot

```python
# Basic scatter
plt.scatter(x, y)

# With colors and sizes
plt.scatter(x, y, c=labels, s=sizes, cmap='viridis')
plt.colorbar()

# With transparency
plt.scatter(x, y, alpha=0.5)
```

#### Bar Chart

```python
# Vertical bars
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)

# Horizontal bars
plt.barh(categories, values)

# Grouped bars
plt.bar(x, y1, label='Group 1')
plt.bar(x + width, y2, label='Group 2')
```

#### Histogram

```python
# Basic histogram
plt.histogram(data, bins=50)

# With density
plt.histogram(data, bins=50, density=True, alpha=0.7)

# Multiple histograms
plt.histogram(data1, bins=30, alpha=0.5, label='Class A')
plt.histogram(data2, bins=30, alpha=0.5, label='Class B')
```

### Statistical Visualizations

#### Box Plot Panel

```
+------------------------------------------------------------------+
|  Box Plot                                                  [x]    |
+------------------------------------------------------------------+
|  Data: [dataset        v]                                         |
|  Group By: [category   v]                                         |
|  Value: [price         v]                                         |
|                                                                   |
|  OPTIONS                                                          |
|  [x] Show outliers  [x] Show mean  [ ] Violin overlay             |
|  Orientation: (o) Vertical  ( ) Horizontal                        |
|                                                                   |
|  [ Generate Plot ]                                                |
|                                                                   |
|  BOX PLOT                                                         |
|                                                                   |
|     Electronics  Clothing  Home  Sports  Books                    |
|          |          |        |      |       |                     |
|          +          |        |      +       |                     |
|         -+-        -+-      -+-    -+-     -+-                    |
|        | | |      | | |    | | |  | | |   | | |                   |
|        | o |      |   |    |   |  | o |   |   |                   |
|        |   |      |   |    |   |  |   |   |   |                   |
|        +-+-+      +-+-+    +-+-+  +-+-+   +-+-+                   |
|          |          |        |      |       |                     |
|          +          +        +      +       +                     |
|          o          o        o      o       o                     |
|                                                                   |
|  STATISTICS                                                       |
|  +-----------------------------------------------------------+   |
|  | Category    | Min    | Q1     | Median | Q3     | Max     |   |
|  +-----------------------------------------------------------+   |
|  | Electronics | 150.00 | 299.00 | 456.00 | 678.00 | 1299.00 |   |
|  | Clothing    | 19.99  | 49.99  | 79.99  | 129.99 | 299.99  |   |
|  | Home        | 29.99  | 89.99  | 149.99 | 249.99 | 599.99  |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  [Save Plot]  [Export Statistics]                                 |
+------------------------------------------------------------------+
```

#### Heatmap Panel

```
+------------------------------------------------------------------+
|  Heatmap                                                   [x]    |
+------------------------------------------------------------------+
|  Data: [correlation_matrix v]                                     |
|                                                                   |
|  OPTIONS                                                          |
|  Color Map: [viridis    v]                                        |
|  [x] Show values  [x] Show colorbar                               |
|  Value format: [.2f      ]                                        |
|                                                                   |
|  [ Generate Heatmap ]                                             |
|                                                                   |
|  CORRELATION HEATMAP                                              |
|                                                                   |
|         price   area  rooms   age rating                          |
|  price  [1.00] [0.86] [0.72] [-0.23] [0.57]                       |
|  area   [0.86] [1.00] [0.81] [-0.12] [0.45]                       |
|  rooms  [0.72] [0.81] [1.00] [-0.09] [0.33]                       |
|  age    [-0.23][-0.12][-0.09] [1.00] [-0.46]                      |
|  rating [0.57] [0.45] [0.33] [-0.46] [1.00]                       |
|                                                                   |
|  Color scale: -1.0 [===|===|===|===|===] 1.0                      |
|               Blue        White        Red                         |
|                                                                   |
|  [Save Heatmap]  [Cluster Rows]  [Cluster Columns]                |
+------------------------------------------------------------------+
```

### Training Visualization

#### Real-Time Training Dashboard

```
+------------------------------------------------------------------+
|  Training Dashboard                                        [x]    |
+------------------------------------------------------------------+
|  Model: Sequential_v1        Status: Training (Epoch 45/100)      |
|                                                                   |
|  LOSS CURVES                                                      |
|  +-----------------------------------------------------------+   |
|  | Loss                                                       |   |
|  | 2.0|*                                                      |   |
|  |    |  *  Train                                             |   |
|  | 1.5|   **                                                  |   |
|  |    |     **  Val                                           |   |
|  | 1.0|       ***                                             |   |
|  |    |          ****                                         |   |
|  | 0.5|              ******                                   |   |
|  |    |                    *********                          |   |
|  | 0.0+-------------------------------------------------->    |   |
|  |    0         20         40         60         80     100   |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  METRICS                                                          |
|  +---------------------------+-------------------------------+    |
|  | Train Accuracy  |  XXXXXXXXXX            | 94.5%          |    |
|  | Val Accuracy    |  XXXXXXXXX             | 92.3%          |    |
|  | Learning Rate   |  X                     | 0.001          |    |
|  +---------------------------+-------------------------------+    |
|                                                                   |
|  RESOURCE USAGE                                                   |
|  +---------------------------+-------------------------------+    |
|  | GPU Memory      |  XXXXXXXX              | 6.2/8.0 GB     |    |
|  | GPU Utilization |  XXXXXXXXXX            | 95%            |    |
|  | Batch Time      |                        | 23.4 ms        |    |
|  +---------------------------+-------------------------------+    |
|                                                                   |
|  [Pause]  [Stop]  [Save Checkpoint]  [Export Plots]               |
+------------------------------------------------------------------+
```

### 3D Visualization

#### Surface Plot

```python
import pycyxwiz.plot as plt

# Create meshgrid
X, Y = plt.meshgrid(x_range, y_range)
Z = function(X, Y)

# Surface plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

#### 3D Scatter

```python
# 3D scatter plot (e.g., PCA visualization)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
           c=labels, cmap='tab10')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
```

### Interactive Features

| Feature | Description | Shortcut |
|---------|-------------|----------|
| **Zoom** | Scroll wheel or drag box | Scroll / Shift+Drag |
| **Pan** | Click and drag | Middle Mouse |
| **Reset** | Double-click | Double-click |
| **Tooltip** | Hover over data point | Mouse hover |
| **Select** | Drag to select region | Ctrl+Drag |
| **Export** | Save current view | Ctrl+S |

### Visualization Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| **Training** | Loss/accuracy curves | Model training |
| **Data Explorer** | Scatter/histogram matrix | EDA |
| **Correlation** | Heatmap with clustering | Feature analysis |
| **Time Series** | Line with annotations | Temporal data |
| **Distribution** | KDE + histogram | Distribution analysis |

## Scripting API

### Basic Plotting

```python
import pycyxwiz.plot as plt

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Line plot
axes[0, 0].plot(x, y, label='Data', color='blue', linewidth=2)
axes[0, 0].set_title('Line Plot')
axes[0, 0].legend()

# Scatter plot
axes[0, 1].scatter(x, y, c=colors, s=50, alpha=0.7)
axes[0, 1].set_title('Scatter Plot')

# Histogram
axes[1, 0].hist(data, bins=30, edgecolor='black')
axes[1, 0].set_title('Histogram')

# Bar plot
axes[1, 1].bar(categories, values)
axes[1, 1].set_title('Bar Plot')

plt.tight_layout()
plt.show()
```

### Statistical Plots

```python
# Box plot
plt.boxplot([data1, data2, data3], labels=['A', 'B', 'C'])

# Violin plot
plt.violinplot([data1, data2, data3])

# Heatmap
plt.imshow(matrix, cmap='viridis', aspect='auto')
plt.colorbar()

# Contour plot
plt.contour(X, Y, Z, levels=20)
plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
```

### Real-Time Plotting

```python
# Create real-time plot
plot = plt.RealTimePlot(max_points=1000)

# During training loop
for epoch in range(epochs):
    loss = train_epoch()
    plot.add_point(epoch, loss)
    plot.update()
```

## Integration with Node Editor

### Visualization Nodes

| Node | Inputs | Outputs |
|------|--------|---------|
| **Plot** | Data tensor | Plot image |
| **Histogram** | Data tensor | Distribution plot |
| **Scatter** | X, Y tensors | Scatter plot |
| **Heatmap** | 2D tensor | Heatmap image |
| **Training Plot** | Loss history | Training visualization |

### Example Pipeline

```
[Model Output] -> [Loss History] -> [Training Plot] -> [Display]
                        |
                        v
                  [CSV Export]
```

## Export Options

| Format | Quality | Use Case |
|--------|---------|----------|
| **PNG** | High | Publications, reports |
| **SVG** | Vector | Scalable graphics |
| **PDF** | Vector | Documents |
| **HTML** | Interactive | Web embedding |
| **CSV** | Data only | Further analysis |

---

**Next**: [Data Science Tools](../data-science/index.md) | [Model Evaluation Tools](../model-evaluation/index.md)
