# Adding nodes: the Nodes panel, search and the right-click menu

Three places add a node to the graph, and they list nodes the same way:

- the **Nodes** panel (the tab beside Asset Browser): cards by group; drag a
  card onto the canvas or double-click it;
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
  panel it is a section at the end, closed until you open it; in the menu a
  submenu after the groups.
- **Can't run yet** holds what the Engine cannot run yet: previews ("Coming
  soon"), blocked nodes ("Blocked") and UI-only templates. Hover a node for the
  reason. Each leaves this list when it can run. A saved graph that has one
  still opens and says why it cannot run. In the Nodes panel it is a section at
  the end, closed until you open it; in the menu a submenu; in search, matches
  are counted at the end ("2 more can't run yet") and **Show** lists them.

Search lists what runs first (best match first), then up to five table steps,
marked *Advanced operation - Rows* and so on.

The Nodes panel's filter (**All**, **Pipeline**, **Training**, **UI-only**,
**Blocked**, **Favourites**) narrows the cards; with a filter other than
*All*, each group lists every node that passes it, in its own group.

## Favourites and Recent

At the top of the Nodes panel, above Studio Tools:

- **Favourites**: click the star at the top right of a card (or right-click
  the card, *Add to Favourites*) to keep the node here; click the star again,
  or *Remove from Favourites*, to take it off. The filter's *Favourites*
  shows only these, in their groups.
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
  canvas, to keep it there.
- **Pins**: hover the small circle just outside a node's box (inputs on the
  left, outputs on the right) for the pin's name, what it carries, how many
  links it has and what it is for.
