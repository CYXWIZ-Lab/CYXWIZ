# Adding nodes: the Nodes panel, search and the right-click menu

Three places add a node to the graph, and they list nodes the same way:

- the **Nodes** panel (the tab beside Asset Browser): nodes by group, as
  cards or as a list; drag one onto the canvas or double-click it;
- the canvas **Search nodes...** box (type to find; Enter or a click adds the
  highlighted node);
- the canvas **right-click menu**, *Add Node*: a submenu per group.

## What each lists

- **Groups (I/O, Analytics, ML Layers, ...)** hold the nodes the Engine can
  run. A group with nothing that runs is not listed.
- **Advanced operations** holds the table steps a Preparation Recipe is made
  of, by kind: *Rows* (Row Filter, Sorter, Duplicate Remover, Row Sampler,
  Missing Value, Table Cropper, Row to Column Names), *Columns* (Column Filter,
  Rename Columns, Column Appender, Cell Extractor, Cell Updater, String
  Manipulation, Math Formula, Rule Engine, Unpivot), *Text* (Text Clean,
  JSONPath, Regex Tester), *Combine* (GroupBy, Joiner, Row Appender) and *SQL
  and checks* (SQL Query, Check, Data Validator). Data Studio > Prepare adds
  these for you; add one here to build a graph step by step. In the Nodes
  panel it is a folding group at the end; in the menu a submenu after the
  groups.
- **Can't run yet** holds what the Engine cannot run yet: previews ("Coming
  soon"), blocked nodes ("Blocked") and UI-only templates. Hover a node for the
  reason. Each leaves this list when it can run. A saved graph that has one
  still opens and says why it cannot run. In the Nodes panel it is a folding
  group at the end; in the menu a submenu; in the canvas search, matches
  are counted at the end ("2 more can't run yet") and **Show** lists them.

Search lists what runs first (best match first), then up to five table steps,
marked *Advanced operation - Rows* and so on.

## The Nodes panel

From the top:

- **Search nodes (/)**: lists the nodes whose name holds the words first
  (the whole name, then the start of a name, then the rest), and under a
  **Related** line the nodes that only mention them in a keyword or their
  description. **/** goes to the search while the panel has focus; **Esc**
  or the **x** clears it. (Ctrl+F is the canvas's Find.)
- **The filter** narrows the panel: **All** (every node that runs, grouped),
  **Trains a model** (layers, activations, losses, optimizers), **Data
  steps** (loading, preparing, exporting), **Views and panels** (dashboards,
  plots), **Can't run yet** (previews and blocked nodes, with why) and
  **Favourites**. Beside it, the number of nodes in the groups ("250
  nodes"; the folds at the end are not counted) or found ("19 found" while
  searching). With a filter other than *All* every group that has a node is
  shown open.
- **Grid or list**: cards with their icons (as many columns as the panel's
  width holds, the name on up to two lines), or one row per node, denser in
  a narrow panel. **File > Preferences > Nodes panel** sets the view a
  session starts in.
- **Studio Tools**, in one row: **Note** (double-click, or drag onto the
  canvas), **Group** (groups the selected nodes, as Ctrl+G; it is off until
  nodes are selected) and **Frame** (drag onto the canvas).
- **Favourites** and **Recent**, as chips (below).
- **The groups** (I/O, Analytics, ML Layers, ...), each with its number of
  nodes. Click a group's name to open or fold it; the panel remembers which
  are open (I/O and Training on a first run).
- At the end, folded: **Plugins** (the nodes loaded plugins and Python
  bundles add; Apps > a plugin opens this group narrowed to it), **Advanced
  operations**, **Can't run yet** and **Examples** (small graphs to start
  from; double-click one to open it in the Pattern Browser, where you set
  it up and insert it).

Hover a node for what it does, its inputs and outputs (with what each
carries) and one line on where it runs ("Runs in training", "Runs as a data
step", "Can't run yet: why"); click it for every detail in the Info panel.
A small dot marks a node that runs only in part (amber), is blocked (red) or
whose support is not settled yet (grey).
Right-click a node for Add to Canvas, Favourites, Show details and Copy Node
Name.

A node added from the Nodes panel or the canvas search, or a note, goes to a
free spot near the middle of the view, and the view moves to show it when it
is outside. A node added from the right-click menu goes where you clicked.

## Favourites and Recent

At the top of the Nodes panel, below Studio Tools:

- **Favourites**: click the star on a card or row (or right-click it, *Add
  to Favourites*) to keep the node here, as a chip, in the order you starred
  them; click the star again, or right-click the chip, *Remove from
  Favourites*, to take it off. Click a chip to add the node; drag it onto
  the canvas. The filter's *Favourites* shows only these, in their groups.
- **Recent**: the last nodes you added to the canvas, newest first, from the
  panel, the canvas search, the right-click menu or the toolbar menus. Click
  a chip to add the node again, or drag it onto the canvas. Plugin nodes
  are not kept in Favourites or Recent yet.

Both are saved with your Engine settings, so they are the same in every
project. **File > Preferences > Nodes panel** turns Recent off, sets how
many it shows (8 by default, up to 20) and clears it.

## Learning what a node does

- **The Info panel** shows a node's description, support and ports. Hover a
  card in the Nodes panel to preview it; click a card, or click a node on the
  canvas, to keep it there. A canvas node that is removed (Delete, Undo)
  leaves the Info panel.
- **Pins**: hover the small circle just outside a node's box (inputs on the
  left, outputs on the right) for the pin's name, what it carries, how many
  links it has and what it is for.
