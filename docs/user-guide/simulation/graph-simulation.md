---
title: Run a graph as a simulation
description: Run, pause, step and stop a graph of signal or simulation nodes, and watch its Signal Scopes in the Simulation Monitor.
type: how-to
area: simulation
audience: user
status: experimental
since: 1.0.8
owner: simulation
last_reviewed: 2026-11-25
related:
  - ../../main-window/toolbar.md
  - ../../main-window/menus.md
  - ../../sdk/plugin-api-reference.md
redirect_from:
  - ../../usage/graph_simulation.md
---

Use this when a graph holds signal nodes (**Sine Wave**, **Signal
Slider**, **Signal Scope** and the other Signal nodes) or a plugin's
simulation nodes (such as the MuJoCo Plant). Such a graph runs as a
simulation: the Engine ticks the graph many times a second, and each
**Signal Scope** keeps a trace of what reaches it.

## Before you start

- A graph with at least one signal or simulation node open in CyxWiz
  Studio.
- To see values, a **Signal Scope** with a number output connected to it.

## Run the graph

1. On the Studio bar, click **Run Sim**. The Simulation Monitor opens (see
   [Settings](#settings) to turn that off) and the time starts counting.
2. Optional: click **Pause** to hold the run and **Resume** to go on.
3. Optional: click **Step** to pause the run and advance it one tick.
4. Click **Stop Sim** to end the run.

A run that failed or was refused shows a warning sign beside **Run Sim**;
hover it for the reason. A disabled command also says why on hover.

The same commands are in two other places, and the three always agree:

- The **Simulation** menu. With a simulation graph open it starts with
  "Graph run: &lt;graph&gt;": **Run Sim** (or **Resume**), **Pause**,
  **Step Once**, **Stop**, **Clear Traces**. A plugin's own run controls
  (for example the MuJoCo Viewport's Run) are left out then; its other
  commands stay under its name. With no simulation graph open, the menu
  runs the plugin's model on its own, as its window does.
- The **Simulation Monitor** window: **Run**, **Pause**, **Step**,
  **Stop**, **Clear**.

**Clear Traces** empties every Scope's trace; a run goes on.

## Watch the Signal Scopes

1. Open the **Simulation Monitor** from **Simulation > Simulation
   Monitor** or the sidebar, or run a graph.
2. Read the run's state, time, tick rate and run number at the top.
3. Read one card per Signal Scope: its name, the newest value, the trace,
   and how many samples it holds of its window.
4. Optional: click **Show in graph** on a card to select that Scope on the
   canvas.

Closing the Monitor loses nothing: the run keeps sampling and the traces
are there when you open it again.

A graph with simulation nodes but no Signal Scope says so: add a Signal
Scope and connect a number output to it. A plugin's window, such as the
MuJoCo Viewport, still shows its model.

## Settings

- **Preferences > Simulation**: the **Tick rate** (1 to 1000 ticks a
  second, 60 by default) of the next run, and **Simulation Monitor: Open
  it when a run starts** (on by default). If you close the Monitor during a
  run it stays closed until the next run.
- The tick rate does not change how fast a physics model moves: a model
  keeps to the run's time. A MuJoCo Plant takes as many model steps as fit
  in each tick (several on a short model step, or one every few ticks when
  the model step is longer than a tick), so 4 s of graph time is 4 s of
  physics.
- A Scope's **Window size** (how many samples it keeps, 10 to 100000, 500
  by default) and **Auto scale** are set on the Scope node, in
  **Properties**.

## If something goes wrong

- **Run Sim** is off with "Stop PPO training first.": a reinforcement
  learning run holds the simulation. Stop it, then run the graph.
- "The MuJoCo Viewport is running the model. Stop it there, then run the
  graph.": the MuJoCo plugin refuses a graph run while its viewport runs
  the model. The run does not start, and the Studio bar and the Monitor
  show the reason.
- A plugin node does not run: a plugin node runs in a graph simulation only
  when its plugin says so (`ISimulationProvider` in the
  [plugin API reference](../../sdk/plugin-api-reference.md)).

## Related

- [The Studio bar](../../main-window/toolbar.md)
- [Menus](../../main-window/menus.md)
- [Plugin API reference](../../sdk/plugin-api-reference.md)
