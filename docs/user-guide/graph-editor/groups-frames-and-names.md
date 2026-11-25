---
title: Organise a graph with groups, frames, notes and names
description: Group nodes, draw frames, add notes and rename nodes so a large graph stays readable.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - canvas.md
  - ../../main-window/nodes-panel.md
redirect_from:
  - groups-frames-and-names.md
---

Use this when a graph has grown past one screen and you want to see its
parts at a glance. None of these change what the graph computes.

- A **group** is a set of nodes drawn as one tinted box with a name band
  above it. The Studio Debugger uses group names as headings.
- A **frame** is a box you place and resize yourself, with a title and a
  description.
- A **note** is free text on the canvas.

Groups and notes are saved with the graph. **Frames are not saved**: they
are gone when you close and reopen the graph, and **Clear the canvas**
does not remove them. Use groups or notes for anything you want to keep.

## Make a group

1. On the canvas, click the first node, then **Ctrl+click** each other node
   you want in the group. (Dragging on empty canvas pans the view; there
   is no box selection.)
2. Do one of these:
   - press **Ctrl+G**;
   - right-click empty canvas and choose **Actions > Create Group**;
   - choose **Nodes > Group Selected** in the menu bar;
   - in the **Nodes** panel, under **Studio Tools**, double-click **Group**.

   A new group is called "Group N". Its box follows its nodes when you move
   them.
3. Optional: double-click the name band, or right-click it and choose
   **Rename...**, type a name, then press Enter or click **Save**.

To take one node out, right-click it and choose **Remove from Group**. To
remove the whole group, right-click its name band and choose **Ungroup**,
or select its nodes and press **Ctrl+Shift+G**. The nodes stay on the
canvas.

Groups are not undo steps. **Ctrl+Z** does not bring back a group you
removed or renamed, and after **Clear the canvas** followed by **Ctrl+Z**
the nodes come back without their groups. Make the group again if you
need it.

### Group names in the Studio Debugger

The Studio Debugger's **Model > Layers** list puts layers in sections named
after your groups. In the ResNet example
(`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph`) the sections are
"Stem + stage 1 (32 channels, 64 x 64)", "Stage 2 ...", and so on.

- Group the layers that belong together, such as a stem, each stage and the
  head, and name each group.
- A run keeps the groups as they were when it started; a saved run shows
  its own sections after you change the canvas.
- Consecutive layers in the same group form one section. Layers in no group
  are listed without a heading.
- When a node is in two groups, the first group made is used.

## Add a frame

1. In the **Nodes** panel, under **Studio Tools**, drag **Frame** onto the
   canvas.
2. Drag the frame's title bar to move it, and its bottom-right corner to
   resize it.
3. Optional: double-click the title bar. The **Edit Frame** dialog has
   **Title:** and **Description:** with **Save** and **Cancel**.

Right-click a frame for:

- **Edit Description**: opens **Edit Frame**.
- **Copy**: copies the frame, the nodes inside it and the links between
  them. **Ctrl+V** pastes them in the same layout.
- **Cut**: the same as Copy, then removes the frame and those nodes.
- **Delete**: removes the frame only; its nodes stay.
- **Group Nodes Inside**: groups the nodes whose top-left corner lies inside
  the frame, at least two of them, and names the group after the frame's
  title. With fewer than two nodes nothing happens on the canvas (the log
  says why).

The menu shows **Ctrl+C** and **Ctrl+X** beside Copy and Cut, but those
keys copy and cut the *selected nodes*, not the frame. Use the menu items
for the frame.

A node inside a frame keeps its own clicks: click it to select it,
right-click it for its own menu.

## Add a note

1. In the **Nodes** panel, under **Studio Tools**, drag **Note** onto the
   canvas, or double-click **Note** to place it at a free spot near the
   middle of the view.
2. Double-click the note, or right-click it and choose **Edit...**.
3. In the **Note** dialog, set **Title**, **Text** and **Colour** (the note
   colour or a node family's), then click **Save**. **Delete** removes the
   note.

Right-click a note for **Edit...**, **Minimise** / **Expand** and
**Delete**. A selected note is also deleted with the **Delete** key. Adding,
moving, minimising, editing and deleting a note are each one **Ctrl+Z**
step.

## Rename a node or describe it

- Right-click the node and choose **Rename...**, type the name, then click
  **Save**. You can also click the name at the top of the **Properties**
  panel and type. The name shows on the canvas, in Properties and in the
  Studio Debugger's layer list.
- Right-click the node and choose **Set Description...** to open **Edit
  Node Description**. The text appears below the node on the canvas.

## Copy, cut, paste and duplicate nodes

- **Ctrl+C**, **Ctrl+X** and **Ctrl+V** copy, cut and paste the selected
  nodes with the links between them. Pasted nodes keep their layout
  relative to each other.
- **Ctrl+D** duplicates the selection.
- Pasted and duplicated nodes go to a fixed spot near the top-left of the
  graph area (each paste a little further down and right), not beside the
  originals or at the mouse, and the view does not move to them. If you do
  not see them, press **F** to fit all nodes in the view.
- Nodes inside a subgraph cannot be copied. Save and load the whole graph
  instead.

## Related

- [Move around the canvas](canvas.md)
- [The Nodes panel](../../main-window/nodes-panel.md)
