# Graph simulation

A graph with signal nodes (Sine Wave, Signal Slider, Signal Scope and the
other Signal nodes) or a plugin's simulation nodes (such as the MuJoCo
Plant) runs as a simulation: the Engine ticks the graph many times a second
and each **Signal Scope** keeps a trace of what reaches it.

## Run, pause, step, stop

Three places run the same commands and always agree:

- The canvas toolbar: **Run Sim**; during a run **Pause** (or **Resume**),
  **Step**, **Stop Sim** and the time. A run that failed or was refused
  shows a warning sign beside Run Sim; hover it for the reason.
- The **Simulation** menu: with a simulation graph open it starts with
  "Graph run: <graph>": **Run Sim** (or **Resume**), **Pause**, **Step
  Once**, **Stop**, **Clear Traces**. A plugin's own run controls (for
  example the MuJoCo Viewport's Run) are left out then; its other commands
  stay under its name. With no simulation graph open, the menu runs the
  plugin's model on its own, as its window does.
- The **Simulation Monitor** window: **Run**, **Pause**, **Step**, **Stop**,
  **Clear**.

Step pauses a running graph and advances it one tick. Clear empties every
Scope's trace; a run goes on. A disabled command says why on hover.

## Simulation Monitor

Open it from **Simulation > Simulation Monitor**, the sidebar, or by
running a graph. It shows the run's state, time, tick rate and run number,
and one card per Signal Scope: its name, **Show in graph** (selects the
Scope on the canvas), the newest value, the trace and how many samples it
holds of its window. Closing it loses nothing: the run keeps sampling and
the traces are there when you open it again.

A graph with simulation nodes but no Signal Scope says so: add a Signal
Scope and connect a number output to it. A plugin's window, such as the
MuJoCo Viewport, still shows its model.

## Settings

- **Preferences > Simulation**: the **Tick rate** (1 to 1000 ticks a
  second, 60 by default) of the next run, and whether the Monitor opens
  when a run starts (on by default). If you close the Monitor during a run
  it stays closed until the next Run.
- A Scope's **Window size** (how many samples it keeps, 10 to 100000, 500
  by default) and **Auto scale** are set on the Scope node, in Properties.

## Plugins

A plugin node runs in a graph simulation only when its plugin says so (see
`ISimulationProvider` in the [plugin API reference](../extending/plugin-api-reference.md)).
A plugin may refuse a run, for example the MuJoCo plugin while its viewport
runs the model: "The MuJoCo Viewport is running the model. Stop it there,
then run the graph." The run does not start and the toolbar and the
Monitor show the reason.
