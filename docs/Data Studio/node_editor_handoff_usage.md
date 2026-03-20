# Data Studio to Node Editor Handoff - User Guide

## Overview

The Node Editor Handoff feature allows you to seamlessly deploy processed datasets from Data Studio directly into the ML Node Editor for model building, eliminating the need for manual dataset loading and configuration.

## Quick Start

### Step 1: Process Data in Data Studio

1. Open **Data Studio** panel (View → Data Studio)
2. Switch to **Pipeline** tab
3. Build your data transformation pipeline:
   - Add **FileInput** node
   - Connect data cleaning nodes (FilterRows, RemoveDuplicates, etc.)
   - Add **DeployToNodeEditor** node at the end
   - Connect the final transformation to DeployToNodeEditor

### Step 2: Execute Pipeline

1. Click **"Execute Pipeline"** button
2. Wait for pipeline to complete
3. Check status bar for "Pipeline: Ready"

### Step 3: Deploy to Node Editor

1. Look for the green **"Deploy to Node Editor"** button (appears after successful execution)
2. Click the button
3. Node Editor panel opens automatically
4. Your processed dataset is now ready in a DatasetInput node!

### Step 4: Build ML Model

1. Add layers to your model (Dense, Conv2D, etc.)
2. Connect layers to DatasetInput node
3. Add Loss and Optimizer nodes
4. Click **"Start Training"**

## Example Workflow

### Clean and Deploy MNIST Dataset

```
Pipeline Nodes:
┌────────────────┐
│  FileInput     │  ← Load mnist.csv
└────────┬───────┘
         │
┌────────▼──────────┐
│  RemoveDuplicates │  ← Remove duplicate rows
└────────┬──────────┘
         │
┌────────▼──────────┐
│  FillMissing      │  ← Fill missing values with 0
└────────┬──────────┘
         │
┌────────▼──────────────┐
│  DeployToNodeEditor   │  ← Deploy to Node Editor
└───────────────────────┘

Parameters:
- DeployToNodeEditor > name: "mnist_cleaned"
```

**After Deployment:**

Your Node Editor will have a DatasetInput node configured:
- Dataset: mnist_cleaned
- Split: train
- Ready to connect to your model

## DeployToNodeEditor Node Configuration

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | String | No | `deployed_<node_id>` | Name for the deployed dataset |

### Example Configurations

**Auto-generated name:**
```json
{
  "type": "DeployToNodeEditor",
  "parameters": {}
}
// Results in: "deployed_15" (where 15 is node ID)
```

**Custom name:**
```json
{
  "type": "DeployToNodeEditor",
  "parameters": {
    "name": "properties_clean_v1"
  }
}
// Results in: "properties_clean_v1"
```

## Common Use Cases

### 1. Data Cleaning Pipeline

**Problem:** Raw dataset has missing values, duplicates, and irrelevant columns.

**Solution:**
```
FileInput → SelectColumns → RemoveDuplicates → FillMissing → DeployToNodeEditor
```

### 2. Feature Engineering

**Problem:** Need to create new features before training.

**Solution:**
```
FileInput → GroupBy → Join → StandardScale → DeployToNodeEditor
```

### 3. Multi-Source Data Integration

**Problem:** Combine data from multiple CSV files.

**Solution:**
```
FileInput (sales.csv) ┐
                      ├─→ Join → DeployToNodeEditor
FileInput (users.csv) ┘
```

### 4. Exploratory Analysis → Training

**Problem:** Analyzed data in Data Studio, ready to train model.

**Solution:**
```
FileInput → Analyze (Query/Visualize tabs) → DeployToNodeEditor
```

## Tips and Tricks

### Naming Convention

Use descriptive names for deployed datasets:
- ✅ `mnist_normalized_train`
- ✅ `customer_data_v2`
- ✅ `features_2026_03_20`
- ❌ `deployed_15`
- ❌ `data`
- ❌ `output`

### Pipeline Reusability

Save your pipeline for reuse:
1. Build your cleaning pipeline
2. File → Save Pipeline As...
3. Load it for future datasets with similar structure

### Version Control

Track dataset versions by including dates or version numbers:
```
DeployToNodeEditor > name: "dataset_v1"  (initial)
DeployToNodeEditor > name: "dataset_v2"  (after cleaning)
DeployToNodeEditor > name: "dataset_v3"  (after feature engineering)
```

### Multiple Deployments

You can deploy multiple datasets to Node Editor:
1. Execute first pipeline → Deploy
2. Build second pipeline → Deploy
3. In Node Editor, you now have multiple DatasetInput nodes
4. Choose which dataset to use for training

### Intermediate Results

Deploy intermediate results for comparison:
```
FileInput → Normalization → DeployToNodeEditor (normalized)
                          ↓
                    Standardization → DeployToNodeEditor (standardized)
```

Compare training results with both preprocessing approaches!

## Troubleshooting

### Deploy Button Doesn't Appear

**Cause:** Pipeline didn't execute successfully or doesn't have DeployToNodeEditor node.

**Solution:**
1. Check status bar for errors
2. Verify DeployToNodeEditor node is connected
3. Re-execute pipeline

### "Cannot deploy: no dataset ready" Error

**Cause:** Pipeline execution failed before reaching DeployToNodeEditor node.

**Solution:**
1. Check each node for error indicators (red)
2. Review pipeline execution log
3. Fix upstream nodes and re-execute

### DatasetInput Node Not Visible

**Cause:** Node Editor has many nodes and camera isn't framed.

**Solution:**
1. Press **F** key to frame all nodes
2. Or use Node Editor search (Ctrl+F) to find "Dataset"

### Dataset Shows as Empty in Training

**Cause:** Dataset wasn't properly registered in DataRegistry.

**Solution:**
1. Check Data Studio → Query tab
2. Run `SELECT COUNT(*) FROM <dataset_name>`
3. Verify dataset has rows
4. Re-deploy if count is 0

### Wrong Dataset Loaded

**Cause:** Multiple DeployToNodeEditor nodes in pipeline.

**Solution:**
- Only the **last** DeployToNodeEditor node in execution order will be deployed
- Remove unused DeployToNodeEditor nodes

## Advanced Workflows

### Conditional Deployment

Use FilterRows to create training/validation splits:

```
FileInput → FilterRows (condition: "id % 10 < 8") → DeployToNodeEditor (train)
         └→ FilterRows (condition: "id % 10 >= 8") → DeployToNodeEditor (val)
```

Deploy both, then create two DatasetInput nodes in Node Editor.

### Automated Preprocessing Pipeline

Create a reusable pipeline template:

```json
{
  "name": "Standard Cleaning Pipeline",
  "nodes": [
    {"type": "FileInput", "parameters": {"path": "${INPUT_FILE}"}},
    {"type": "RemoveDuplicates"},
    {"type": "FillMissing", "parameters": {"strategy": "mean"}},
    {"type": "StandardScale"},
    {"type": "DeployToNodeEditor", "parameters": {"name": "${OUTPUT_NAME}"}}
  ]
}
```

Use variables to customize for different datasets.

## Best Practices

1. **Clean Before Deploy:** Always remove duplicates and handle missing values
2. **Name Meaningfully:** Use descriptive names for deployed datasets
3. **Save Pipelines:** Reuse pipelines for similar datasets
4. **Test Small:** Deploy a sample (first 1000 rows) to test your model architecture
5. **Version Control:** Keep track of preprocessing versions
6. **Document:** Add comments in pipeline describing transformations

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+E` | Execute Pipeline |
| `Ctrl+D` | Deploy to Node Editor (when available) |
| `F` | Frame Node Editor to show all nodes |
| `Ctrl+F` | Search for nodes in Node Editor |

## Limitations

- Maximum dataset size: Limited by available RAM
- DeployToNodeEditor must be the last node in pipeline
- Only tabular data (CSV, Parquet, Arrow) supported
- Binary data (images, audio) not supported via this path

## Getting Help

- Check logs: `engine_log.txt` in project directory
- View detailed pipeline execution: Data Studio → Status Bar
- Debug node parameters: Right-click node → Properties

## Next Steps

After deploying your dataset:
1. **Explore Architectures:** Use Pattern Browser to load model templates
2. **AutoML:** Try NAS panel to automatically find best architecture
3. **Hyperparameter Tuning:** Use Grid Search panel
4. **Experiment Tracking:** Enable MLflow plugin to track experiments
