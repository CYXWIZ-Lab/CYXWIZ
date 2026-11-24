# Groups, frames and names on the canvas

The canvas has three ways to organise a graph. None of them changes what the
graph computes.

- A **group** is a set of nodes shown as one tinted box with a name above
  it. The Studio Debugger uses group names as headings in Model > Layers.
- A **frame** is a box you place and resize yourself, with a title and a
  description. It holds no nodes.
- A **note** is free text on the canvas.

All three are saved with the graph.

## Groups

**Make a group.** Select the nodes, by dragging a box around them or by
Ctrl+clicking each one, then do any of these:

- press **Ctrl+G**;
- right-click the canvas and choose **Create Group**;
- choose **Nodes > Group Selected** in the menu bar;
- double-click the **Group** card in the Nodes panel (Studio Tools).

A new group is called "Group N". The box follows its nodes when you move
them.

**Name a group.** Double-click the band with the group's name, or
right-click that band and choose **Rename...**. Type the name and press
Enter or click **Save**. Undo (Ctrl+Z) takes the old name back.

**Remove nodes or the group.**

- To take one node out, right-click it and choose **Remove from Group**.
- To delete the whole group, right-click its name band and choose
  **Ungroup**, or select its nodes and press **Ctrl+Shift+G**. The nodes
  stay on the canvas.

**From a frame.** A frame's right-click **Group Nodes Inside** groups the
nodes whose top-left corner lies inside the frame, at least two of them.
The group takes the frame's title as its name.

### Group names in the Studio Debugger

The Studio Debugger's **Model > Layers** list puts its layers in sections
named after your groups. In the ResNet example, those sections are "Stem +
stage 1 (32 channels, 64 x 64)", "Stage 2 ...", and so on (see
`examples/cyxgraph/cats_dogs_resnet_scratch.cyxgraph`). How the sections
are formed:

- Group the layers that belong together, such as a stem, each residual
  stage and the head, and give each group a name.
- A run keeps the groups as they were when it started. A saved run shows
  its own sections even after you change the canvas.
- Consecutive layers in the same group form one section. Layers in no group
  are listed without a heading.
- When a node is in two groups, the first group made is the one used.

## Frames

**Make a frame.** Drag the **Frame** card from the Nodes panel (Studio
Tools) onto the canvas.

**Change a frame.**

- Drag its title bar to move it, and its bottom-right corner to resize it.
- Double-click the title bar to change the title and description.

**Right-click a frame** for these:

- **Edit Description**.
- **Copy**: copies the frame and the nodes inside it, with the links
  between those nodes. **Ctrl+V** pastes the frame and nodes in the same
  layout.
- **Cut**: the same as Copy, then removes the frame and those nodes.
- **Delete**: removes the frame only. Its nodes stay.
- **Group Nodes Inside** (see Groups above).

A node inside a frame keeps its own clicks: click it to select it, and
right-click it for its own menu.

## Notes

**Add a note.** Drag the **Note** card (Nodes panel, Studio Tools) onto the
canvas, or double-click it (the note goes to a free spot near the middle of
the view). A single click on the card only says how.

**Edit a note.** Double-click it, or right-click it and choose **Edit...**:
the dialog has its **Title**, **Text** and **Colour** (the note colour or a
node family's), with **Save**, **Cancel** and **Delete**.

**Right-click a note** for **Edit...**, **Minimise** / **Expand** and
**Delete**. A selected note (click it) is deleted with the **Delete** key.

Adding, moving, minimising, editing and deleting a note are each one Undo
(Ctrl+Z) step.

## Copy, cut and paste nodes

- **Ctrl+C**, **Ctrl+X** and **Ctrl+V** copy, cut and paste the selected
  nodes, with the links between them.
- Pasted nodes keep their layout relative to each other.
- **Ctrl+D** duplicates the selection.
- Nodes inside a subgraph cannot be copied yet. Save and load the whole
  graph instead.

## Rename a node

Right-click the node and choose **Rename...**. The name appears on the
canvas, in Properties and in the Studio Debugger's layer list.
