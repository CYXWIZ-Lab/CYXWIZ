# Complete Workflow Tutorial

Learn the full CyxWiz workflow: loading data, building a model, training, and evaluating results.

## Overview

This tutorial covers the end-to-end machine learning workflow:

```
Load Dataset → Build Model → Configure Training → Train → Evaluate → Export
```

We'll train a classifier on sample data using both the visual Node Editor and Python scripting.

## Prerequisites

- Completed [Getting Started](getting-started.md) and [Basic Node Tutorial](basic-node-tutorial.md)
- Sample dataset (we'll create one)

## Step 1: Prepare the Dataset

### Create Sample Data

Open the Console and run:

```python
import numpy as np
import os

# Create sample classification data
np.random.seed(42)
n_samples = 1000
n_features = 20
n_classes = 5

# Generate features
X = np.random.randn(n_samples, n_features).astype(np.float32)

# Generate labels
y = np.random.randint(0, n_classes, n_samples)

# Save to project folder
project_path = "datasets"  # Uses current project path
os.makedirs(project_path, exist_ok=True)

np.save(f"{project_path}/X_train.npy", X[:800])
np.save(f"{project_path}/y_train.npy", y[:800])
np.save(f"{project_path}/X_test.npy", X[800:])
np.save(f"{project_path}/y_test.npy", y[800:])

print("Dataset created!")
print(f"Train: {X[:800].shape}, Test: {X[800:].shape}")
```

### Load Dataset in Script

```python
%%code
import numpy as np
import pycyxwiz as cx

# Load data
X_train = np.load("datasets/X_train.npy")
y_train = np.load("datasets/y_train.npy")
X_test = np.load("datasets/X_test.npy")
y_test = np.load("datasets/y_test.npy")

print(f"Training: {X_train.shape} samples")
print(f"Testing: {X_test.shape} samples")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y_train))}")
```

## Step 2: Build the Model (Visual)

### Using the Node Editor

1. Open **View > Node Editor**
2. Build this architecture:

```
DatasetInput → Dense(64) → ReLU → Dropout(0.3) → Dense(32) → ReLU → Dense(5) → Softmax → Output
```

### Node Configuration

| Node | Configuration |
|------|---------------|
| DatasetInput | Shape: 20, Batch: 32 |
| Dense #1 | Units: 64, Bias: Yes |
| ReLU #1 | - |
| Dropout | Rate: 0.3 |
| Dense #2 | Units: 32, Bias: Yes |
| ReLU #2 | - |
| Dense #3 | Units: 5, Bias: Yes |
| Softmax | Dim: -1 |

### Validate and Export

1. **Nodes > Validate Graph** - Should show green checkmark
2. **Nodes > Generate Code > PyCyxWiz**
3. Copy the generated code for training

## Step 3: Build the Model (Script)

Alternatively, build the model in Python:

```python
%%code
import pycyxwiz as cx

class SimpleClassifier:
    def __init__(self):
        # Layers
        self.dense1 = cx.Dense(20, 64)
        self.relu1 = cx.ReLU()
        self.dropout = cx.Dropout(0.3)
        self.dense2 = cx.Dense(64, 32)
        self.relu2 = cx.ReLU()
        self.dense3 = cx.Dense(32, 5)
        self.softmax = cx.Softmax()

        # Loss function
        self.loss_fn = cx.CrossEntropyLoss()

        # Optimizer
        self.optimizer = cx.Adam(learning_rate=0.001)

    def forward(self, x, training=True):
        x = self.dense1.forward(x)
        x = self.relu1.forward(x)
        if training:
            x = self.dropout.forward(x)
        x = self.dense2.forward(x)
        x = self.relu2.forward(x)
        x = self.dense3.forward(x)
        x = self.softmax.forward(x)
        return x

    def get_parameters(self):
        params = []
        params.extend(self.dense1.get_parameters().values())
        params.extend(self.dense2.get_parameters().values())
        params.extend(self.dense3.get_parameters().values())
        return params

    def get_gradients(self):
        grads = []
        grads.extend(self.dense1.get_gradients().values())
        grads.extend(self.dense2.get_gradients().values())
        grads.extend(self.dense3.get_gradients().values())
        return grads

model = SimpleClassifier()
print("Model created!")
```

## Step 4: Configure Training

```python
%%code
# Training configuration
config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'shuffle': True,
    'validation_split': 0.1
}

print("Training Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
```

## Step 5: Training Loop

```python
%%code
import numpy as np
import matplotlib.pyplot as plt

# Convert to tensors
X_train_tensor = cx.Tensor.from_numpy(X_train)
X_test_tensor = cx.Tensor.from_numpy(X_test)

# One-hot encode labels
def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels].astype(np.float32)

y_train_onehot = one_hot(y_train, 5)
y_test_onehot = one_hot(y_test, 5)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

%%code
# Training loop
n_epochs = 50
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_correct = 0

    # Shuffle data
    indices = np.random.permutation(len(X_train))

    for batch_idx in range(n_batches):
        # Get batch
        start = batch_idx * batch_size
        end = start + batch_size
        batch_indices = indices[start:end]

        X_batch = cx.Tensor.from_numpy(X_train[batch_indices])
        y_batch = cx.Tensor.from_numpy(y_train_onehot[batch_indices])

        # Forward pass
        predictions = model.forward(X_batch, training=True)

        # Compute loss
        loss = model.loss_fn.forward(predictions, y_batch)
        epoch_loss += loss

        # Compute accuracy
        pred_labels = np.argmax(predictions.to_numpy(), axis=1)
        true_labels = y_train[batch_indices]
        epoch_correct += np.sum(pred_labels == true_labels)

        # Backward pass
        grad = model.loss_fn.backward(predictions, y_batch)
        # ... backprop through layers ...

        # Update weights
        model.optimizer.step(model.get_parameters(), model.get_gradients())
        model.optimizer.zero_grad()

    # Compute epoch metrics
    train_loss = epoch_loss / n_batches
    train_acc = epoch_correct / (n_batches * batch_size)

    # Validation (simplified)
    val_pred = model.forward(X_test_tensor, training=False)
    val_loss = model.loss_fn.forward(val_pred, cx.Tensor.from_numpy(y_test_onehot))
    val_labels = np.argmax(val_pred.to_numpy(), axis=1)
    val_acc = np.mean(val_labels == y_test)

    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

print("\nTraining complete!")
```

## Step 6: Visualize Training

```python
%%code
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy plot
axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Step 7: Evaluate the Model

```python
%%code
# Final evaluation on test set
test_predictions = model.forward(X_test_tensor, training=False)
test_pred_labels = np.argmax(test_predictions.to_numpy(), axis=1)

# Compute metrics
from pycyxwiz import stats

results = stats.confusion_matrix(y_test.tolist(), test_pred_labels.tolist())

print("=== Final Evaluation ===")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"Precision: {np.mean(results['precision']):.4f}")
print(f"Recall:    {np.mean(results['recall']):.4f}")
print(f"F1 Score:  {np.mean(results['f1']):.4f}")

%%code
# Confusion matrix visualization
import matplotlib.pyplot as plt
import numpy as np

cm = np.array(results['matrix'])
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

# Labels
classes = [f'Class {i}' for i in range(5)]
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Add text annotations
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black")

ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.colorbar(im)
plt.tight_layout()
plt.show()
```

## Step 8: Save and Export

### Save Model Weights

```python
%%code
import json

# Save model state
model_state = {
    'dense1_weights': model.dense1.get_parameters()['weight'].to_numpy().tolist(),
    'dense1_bias': model.dense1.get_parameters()['bias'].to_numpy().tolist(),
    'dense2_weights': model.dense2.get_parameters()['weight'].to_numpy().tolist(),
    'dense2_bias': model.dense2.get_parameters()['bias'].to_numpy().tolist(),
    'dense3_weights': model.dense3.get_parameters()['weight'].to_numpy().tolist(),
    'dense3_bias': model.dense3.get_parameters()['bias'].to_numpy().tolist(),
}

with open('models/classifier_weights.json', 'w') as f:
    json.dump(model_state, f)

print("Model saved to models/classifier_weights.json")
```

### Export Training History

```python
%%code
import csv

with open('models/training_history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for i in range(len(history['train_loss'])):
        writer.writerow([
            i + 1,
            history['train_loss'][i],
            history['train_acc'][i],
            history['val_loss'][i],
            history['val_acc'][i]
        ])

print("History saved to models/training_history.csv")
```

## Summary

You've completed a full ML workflow in CyxWiz:

| Step | Action |
|------|--------|
| 1 | Created and loaded a dataset |
| 2-3 | Built a neural network (visual and script) |
| 4 | Configured training parameters |
| 5 | Ran the training loop |
| 6 | Visualized training progress |
| 7 | Evaluated model performance |
| 8 | Saved model weights and history |

## What's Next

- Try more complex architectures (CNNs, RNNs)
- Experiment with different optimizers and learning rates
- Load real datasets (MNIST, CIFAR-10)
- Use the Training Dashboard for real-time monitoring
- Explore [distributed training](../engine/training-distributed.md)

---

**Back to**: [Tutorials Index](index.md) | **Next**: [Python Scripting Guide](../engine/scripting/index.md)
