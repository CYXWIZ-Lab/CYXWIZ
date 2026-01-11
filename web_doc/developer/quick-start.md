# Quick Start Guide

Get CyxWiz up and running in minutes. This guide covers the fastest path to building and running all components.

## 5-Minute Setup

### Prerequisites Check

```bash
# Check required tools
cmake --version      # 3.20+
git --version        # 2.0+
rustc --version      # 1.70+
python --version     # 3.8+
```

### Clone and Build

```bash
# Clone repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz

# Run setup script (installs dependencies)
./scripts/setup.sh    # Linux/macOS
# or
setup.bat             # Windows (Developer Command Prompt)

# Build everything
./scripts/build.sh    # Linux/macOS
# or
build.bat             # Windows
```

### Run Components

```bash
# CyxWiz Engine (Desktop Client)
./build/linux-release/bin/cyxwiz-engine

# CyxWiz Server Node (Compute Worker)
./build/linux-release/bin/cyxwiz-server-node

# CyxWiz Central Server (Orchestrator)
cd cyxwiz-central-server && cargo run --release
```

## Your First ML Model

### Using the Node Editor

1. **Launch CyxWiz Engine**
2. **Create New Project**: `File > New Project`
3. **Open Node Editor**: `View > Node Editor`
4. **Build a Simple Network**:

```
[Data Input] -> [Dense 784->128 ReLU] -> [Dense 128->10 Softmax] -> [Model Output]
```

5. **Add Nodes**:
   - Right-click canvas > `Add Node > Data > DataInput`
   - Right-click canvas > `Add Node > Layers > Dense`
   - Connect nodes by dragging from output pins to input pins

6. **Configure Dense Layer**:
   - Select node
   - In Properties panel: Units=128, Activation=ReLU

7. **Generate Code**: `Edit > Generate Code > PyTorch`

### Using Python Scripting

Open the Console panel (`View > Console`) and try:

```python
import pycyxwiz as cyx

# Create a simple model
model = cyx.Sequential([
    cyx.layers.Dense(128, activation='relu', input_shape=(784,)),
    cyx.layers.Dropout(0.2),
    cyx.layers.Dense(64, activation='relu'),
    cyx.layers.Dense(10, activation='softmax')
])

# Print summary
model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load sample data
(X_train, y_train), (X_test, y_test) = cyx.datasets.mnist.load_data()

# Train
history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

## Project Structure

After creating a project, you'll have:

```
MyProject/
├── models/              # .cyxgraph files (node editor graphs)
├── scripts/             # Python scripts
├── data/                # Datasets
├── outputs/
│   ├── checkpoints/     # Model checkpoints
│   └── logs/            # Training logs
└── project.json         # Project configuration
```

## Loading Data

### From File

```python
# CSV
data = cyx.datasets.load_csv('data/train.csv')

# Images from folder
train_data = cyx.datasets.ImageFolder(
    'data/images',
    transform=cyx.transforms.Compose([
        cyx.transforms.Resize(224),
        cyx.transforms.ToTensor(),
        cyx.transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
    ])
)
```

### Built-in Datasets

```python
# MNIST
(X_train, y_train), (X_test, y_test) = cyx.datasets.mnist.load_data()

# CIFAR-10
(X_train, y_train), (X_test, y_test) = cyx.datasets.cifar10.load_data()

# Fashion-MNIST
(X_train, y_train), (X_test, y_test) = cyx.datasets.fashion_mnist.load_data()
```

## Training a Model

### Local Training

```python
# Configure training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        cyx.callbacks.EarlyStopping(patience=3),
        cyx.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        cyx.callbacks.TensorBoard(log_dir='./logs')
    ]
)

# Plot training history
cyx.plot.training_history(history)
```

### Distributed Training (Future)

```python
# Submit to network
job = cyx.submit_job(
    model=model,
    dataset='ipfs://QmXxx...',
    epochs=100,
    payment=10.0  # CYXWIZ tokens
)

# Monitor progress
job.stream_updates(callback=print)

# Get results
results = job.wait()
print(f"Final accuracy: {results.metrics['accuracy']}")
```

## Using GPU

### Check Available Devices

```python
import pycyxwiz as cyx

# List devices
devices = cyx.device.list_devices()
for d in devices:
    print(f"{d.id}: {d.name} ({d.type})")

# Set device
cyx.device.set_device(0)  # First GPU

# Or by type
cyx.device.set_device('cuda', 0)
```

### GPU Training

```python
# Model automatically uses GPU if available
model = cyx.Sequential([...])

# Explicit device placement
with cyx.device.DeviceScope('cuda:0'):
    model.fit(X_train, y_train, epochs=10)
```

## Saving and Loading

### Models

```python
# Save model
model.save('my_model.h5')

# Load model
model = cyx.load_model('my_model.h5')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

### Node Editor Graphs

- Save: `File > Save` or `Ctrl+S`
- Load: `File > Open` or `Ctrl+O`
- Export: `File > Export > PyTorch/TensorFlow/Keras`

## Keyboard Shortcuts

### Global

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+Shift+P` | Command Palette |

### Node Editor

| Shortcut | Action |
|----------|--------|
| `A` | Add Node |
| `X` | Delete Selected |
| `D` | Duplicate |
| `F` | Frame Selected |
| `Space` | Quick Search |

### Console

| Shortcut | Action |
|----------|--------|
| `Enter` | Execute |
| `Ctrl+Enter` | Execute Selection |
| `Ctrl+C` | Cancel |
| `Up/Down` | History |

## Next Steps

1. **Explore Examples**: Check `examples/` folder for sample projects
2. **Read Documentation**: Full docs at [docs.cyxwiz.com](https://docs.cyxwiz.com)
3. **Join Community**: Discord server for help and discussion
4. **Contribute**: See [Contributing Guide](contributing.md)

## Common Tasks

### Create CNN for Image Classification

```python
model = cyx.Sequential([
    cyx.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    cyx.layers.MaxPool2D((2, 2)),
    cyx.layers.Conv2D(64, (3, 3), activation='relu'),
    cyx.layers.MaxPool2D((2, 2)),
    cyx.layers.Conv2D(64, (3, 3), activation='relu'),
    cyx.layers.Flatten(),
    cyx.layers.Dense(64, activation='relu'),
    cyx.layers.Dense(10, activation='softmax')
])
```

### Create RNN for Sequence Data

```python
model = cyx.Sequential([
    cyx.layers.Embedding(10000, 128, input_length=100),
    cyx.layers.LSTM(64, return_sequences=True),
    cyx.layers.LSTM(32),
    cyx.layers.Dense(1, activation='sigmoid')
])
```

### Data Augmentation

```python
augmentation = cyx.Sequential([
    cyx.layers.RandomFlip("horizontal"),
    cyx.layers.RandomRotation(0.1),
    cyx.layers.RandomZoom(0.1),
])

model = cyx.Sequential([
    augmentation,
    cyx.layers.Conv2D(32, (3, 3), activation='relu'),
    # ... rest of model
])
```

## Troubleshooting

### "CUDA not available"

```python
# Check CUDA
print(cyx.device.cuda_available())  # Should be True

# If False, verify:
# 1. NVIDIA driver installed
# 2. CUDA toolkit installed
# 3. ArrayFire built with CUDA
```

### "Out of memory"

```python
# Reduce batch size
model.fit(X, y, batch_size=16)  # Instead of 32

# Clear cache
cyx.device.empty_cache()

# Use mixed precision
model.compile(..., mixed_precision=True)
```

### "Module not found"

```bash
# Ensure Python bindings are built
cmake --build build/linux-release --target pycyxwiz

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./build/linux-release/python
```

---

**Next**: [Building](building.md) | [Installation](installation.md)
