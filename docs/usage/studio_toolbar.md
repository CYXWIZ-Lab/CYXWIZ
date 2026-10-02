# The CyxWiz Studio bar

The bar at the top of the CyxWiz Studio canvas, one row in groups.

## The graph (left)

- **The graph's name** (the file name, or *Untitled graph*). A dot after the
  name means the graph changed since it was saved. Hover for the full path.
  Click it for the graph menu:
  - **Save** (Ctrl+S): writes the graph's own file. A graph that has no file
    yet asks for a name.
  - **Save As...** (Ctrl+Shift+S): writes a new file; the open file stays as
    it was.
  - **Load...**, **Export...**, **Custom Node Editor**.
  - **Clear the canvas...**: asks first; Ctrl+Z brings the nodes back. The
    file changes only when you save.
- **Save** is also on the bar.

A starter opened from the start page, a pattern template and an imported
model open as *Untitled graph*: Save asks for a name, so the example files
next to the Engine are never overwritten.

Ctrl+S and Ctrl+Shift+S save the graph while the canvas is focused (in the
Script Editor they save the script); see Preferences > Shortcuts.

## Edit and view (middle)

Icons, with their names on hover: **Select all** (Ctrl+A), **Duplicate**
(Ctrl+D) and **Delete** (Delete), dimmed until nodes are selected; zoom out,
the zoom level, zoom in; **Fit** frames all nodes (F); **Minimap** (M,
highlighted while shown); the node, link and selection counts.

## Code and run (right)

- The mode (**Code Gen**, **Data Pipeline**, **Local Training**). Code Gen:
  the framework and **Generate**. Data Pipeline: **Execute Pipeline**, then
  its progress and **Cancel** while it runs. **Export**.
- **Compile** (F7), **Local Debug** (F6) and **Train**. While training,
  Train becomes the epoch and **Stop**.
- Graphs with simulation or RL nodes add **Run Sim** / **Stop Sim** (with the
  simulated time and speed), **Train RL** / **Stop RL** and **Export ONNX**.

## A narrow canvas

When the canvas is too narrow for everything, the edit, code and view tools
(then Local Debug and the simulation / RL tools) move into the **...** menu.
The graph, Save, zoom, Compile and Train always stay on the bar.
