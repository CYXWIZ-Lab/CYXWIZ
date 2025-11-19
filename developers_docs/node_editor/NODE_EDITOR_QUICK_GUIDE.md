# CyxWiz Node Editor - Quick Start Guide

## Opening the Node Editor

When you launch CyxWiz Engine, the Node Editor might not be immediately visible. Here's how to access it:

### Method 1: Click the Tab
Look for tabs at the top of panels. You should see tabs like:
- Console
- **Node Editor** ← Click this tab!
- Viewport
- Properties
- Script Editor

### Method 2: Use the View Menu
If you don't see the Node Editor tab:
1. Look for a **View** or **Windows** menu in the menu bar
2. Select **Node Editor** to show the panel

## Node Editor Toolbar Features

Once you have the Node Editor open, you'll see a toolbar with these buttons:

### File Operations
- **Save Graph** - Save your neural network to a .cyxwiz file
- **Load Graph** - Load a previously saved network

### Node Operations
- **Add Dense Layer** - Add a fully connected layer
- **Add ReLU** - Add ReLU activation
- **Clear All** - Clear the entire graph

### Code Generation
- **Framework Dropdown** - Choose PyTorch, TensorFlow, Keras, or PyCyxWiz
- **Generate Code** - Generate code and display in Script Editor
- **Export Code** - Save generated code to a Python file

## Creating Your First Neural Network

1. **Add an Input Node**: Right-click in the canvas → Add Node → Input
2. **Add a Dense Layer**: Click "Add Dense Layer" button or right-click → Add Node → Dense
3. **Add an Output Node**: Right-click → Add Node → Output
4. **Connect Nodes**: Drag from output pin to input pin
5. **Generate Code**: Select framework and click "Generate Code"
6. **Export Code**: Click "Export Code" to save as Python file

## Tips

- **Right-click** on empty canvas to add nodes via context menu
- **Drag** between pins to create connections
- **Select nodes** to see properties in the Properties panel
- **Mouse wheel** to zoom in/out
- The node canvas has an infinite scrollable area

## Troubleshooting

**Can't see buttons?**
- Make sure the "Node Editor" tab is active (clicked)
- The window title should say "Node Editor"
- Buttons appear at the top of the Node Editor panel

**Window is too small?**
- Drag the panel borders to resize
- Undock the panel by dragging its tab

**Export Code not working?**
- Create a valid graph first (Input → Layers → Output)
- All nodes must be connected
- Check the Console for error messages