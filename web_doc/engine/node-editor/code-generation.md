# Code Generation

The Node Editor can export your visual model to production-ready Python code in multiple frameworks.

## Supported Frameworks

| Framework | Output | Best For |
|-----------|--------|----------|
| **PyTorch** | `torch.nn.Module` | Research, flexibility |
| **TensorFlow** | `tf.keras.Model` | Production, deployment |
| **Keras** | `keras.Model` | Simplicity, readability |
| **PyCyxWiz** | `pycyxwiz.Sequential` | CyxWiz native training |

## Generating Code

### Method 1: Toolbar Buttons

Click the framework button in the Node Editor toolbar:

```
[Save] [Load] [PyTorch] [TensorFlow] [Keras] [PyCyxWiz] [Validate]
```

### Method 2: Nodes Menu

**Nodes > Generate Code > [Framework]**

### Method 3: Keyboard Shortcut

Configure in Preferences > Keyboard Shortcuts

### Method 4: Context Menu

Right-click on canvas > Generate Code > [Framework]

## Output Location

Generated code is automatically inserted into the Script Editor:
- If a script is open, code is appended
- If no script, a new untitled script is created

## PyTorch Output

### Simple Model

```python
import torch
import torch.nn as nn

class CyxWizModel(nn.Module):
    def __init__(self):
        super(CyxWizModel, self).__init__()
        self.dense_0 = nn.Linear(784, 128)
        self.relu_1 = nn.ReLU()
        self.dense_2 = nn.Linear(128, 64)
        self.relu_3 = nn.ReLU()
        self.dense_4 = nn.Linear(64, 10)
        self.softmax_5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense_0(x)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.relu_3(x)
        x = self.dense_4(x)
        x = self.softmax_5(x)
        return x

# Instantiate model
model = CyxWizModel()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### With Skip Connections

```python
class CyxWizModel(nn.Module):
    def __init__(self):
        super(CyxWizModel, self).__init__()
        self.conv_0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        # Skip connection handled in forward()

    def forward(self, x):
        identity = x

        x = self.conv_0(x)
        x = self.bn_1(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.bn_4(x)

        x = x + identity  # Residual connection
        x = self.relu_2(x)

        return x
```

## TensorFlow Output

### Simple Model

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class CyxWizModel(Model):
    def __init__(self):
        super(CyxWizModel, self).__init__()
        self.dense_0 = layers.Dense(128, activation=None)
        self.relu_1 = layers.ReLU()
        self.dense_2 = layers.Dense(64, activation=None)
        self.relu_3 = layers.ReLU()
        self.dense_4 = layers.Dense(10, activation=None)
        self.softmax_5 = layers.Softmax()

    def call(self, x, training=False):
        x = self.dense_0(x)
        x = self.relu_1(x)
        x = self.dense_2(x)
        x = self.relu_3(x)
        x = self.dense_4(x)
        x = self.softmax_5(x)
        return x

# Instantiate model
model = CyxWizModel()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
```

## Keras Output

### Sequential API (when possible)

```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Functional API (for complex graphs)

```python
import keras
from keras import layers

inputs = keras.Input(shape=(784,))
x = layers.Dense(128)(inputs)
x = layers.ReLU()(x)
x = layers.Dense(64)(x)
x = layers.ReLU()(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

## PyCyxWiz Output

### Native Format

```python
import pycyxwiz as cyx

# Initialize backend
cyx.initialize()

# Build model
model = cyx.Sequential()
model.add(cyx.Dense(128, activation='relu'))
model.add(cyx.Dense(64, activation='relu'))
model.add(cyx.Dense(10, activation='softmax'))

# Configure training
model.compile(
    optimizer=cyx.Adam(lr=0.001),
    loss=cyx.CrossEntropyLoss(),
    metrics=['accuracy']
)

# Train
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)
```

## Code Generation Rules

### Topological Sorting

Nodes are sorted topologically to ensure:
1. Data flows from inputs to outputs
2. Dependencies are resolved correctly
3. No circular references

### Naming Convention

| Node Type | Naming Pattern |
|-----------|----------------|
| Layers | `{type}_{index}` e.g., `dense_0`, `conv_1` |
| Custom names | User-specified names preserved |
| Skip connections | `identity_{index}` for saved tensors |

### Parameter Mapping

Node parameters map to framework parameters:

| Node Param | PyTorch | TensorFlow | Keras |
|------------|---------|------------|-------|
| `units` | `out_features` | `units` | `units` |
| `filters` | `out_channels` | `filters` | `filters` |
| `kernel_size` | `kernel_size` | `kernel_size` | `kernel_size` |
| `activation` | Separate layer | `activation=` | `activation=` |

### Handling Activations

**PyTorch:** Activations are separate modules

```python
self.dense = nn.Linear(128, 64)
self.relu = nn.ReLU()
```

**TensorFlow/Keras:** Can be inline or separate

```python
# Inline
layers.Dense(64, activation='relu')

# Separate
layers.Dense(64)
layers.ReLU()
```

### Skip Connections

```python
# PyTorch
identity = x
x = self.conv(x)
x = x + identity  # Residual

# TensorFlow/Keras
identity = x
x = conv(x)
x = layers.Add()([x, identity])  # Residual
```

### Merge Operations

```python
# PyTorch
x = torch.cat([x1, x2], dim=1)  # Concatenate
x = x1 + x2  # Add
x = x1 * x2  # Multiply

# TensorFlow/Keras
x = layers.Concatenate()([x1, x2])
x = layers.Add()([x1, x2])
x = layers.Multiply()([x1, x2])
```

## Validation Before Generation

The code generator validates the graph:

1. **Has Input** - DatasetInput node present
2. **Has Output** - Output node present
3. **No Cycles** - Acyclic graph
4. **All Connected** - No orphan nodes
5. **Type Compatibility** - Pin types match

Error messages are shown if validation fails.

## Advanced Features

### Custom Layer Support

Custom nodes generate placeholder code:

```python
# Custom layer: MyCustomLayer
class MyCustomLayer(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        # TODO: Implement custom layer
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        return x
```

### Data Augmentation

```python
# PyTorch transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# TensorFlow
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### Training Loop Generation

Option to generate complete training loop:

```python
# PyTorch training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        # ...
```

## Exporting to File

### From Script Editor

1. Generate code to Script Editor
2. **File > Save As**
3. Choose `.py` extension

### Direct Export

**Nodes > Export Code to File**

Saves directly to `.py` file with framework in filename:
- `model_pytorch.py`
- `model_tensorflow.py`
- `model_keras.py`
- `model_pycyxwiz.py`

## Tips

1. **Validate First** - Fix errors before generating
2. **Review Output** - Check generated code for correctness
3. **Test Generated Code** - Run to verify functionality
4. **Customize as Needed** - Generated code is a starting point
5. **Keep Graph Simple** - Complex graphs = complex code

---

**Next**: [Patterns](patterns.md) | [Validation](validation.md)
