# MuJoCo models: the Environment Library and your projects

The MuJoCo Simulation plugin's **Environment Library** (Simulation >
Environment Library) lists the models you can load into the
[MuJoCo Viewport](mujoco_viewport.md) and name in a **MuJoCo Plant** node:

- **Built-in**: the Gymnasium tasks shipped with the plugin (Inverted
  Pendulum, Half Cheetah, Hopper, Walker 2D, ...).
- **Menagerie**: robots from Google DeepMind's
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
  downloaded when you want one.
- **In this project**: models kept in the open project's `mujoco/models/`
  folder.
- **Imported**: models imported from a URL while no project was open.

## Where downloads go

A downloaded robot is **shared by every project**: it is saved once in your
user folder (`C:\Users\<you>\.cyxwiz\menagerie` on Windows, `~/.cyxwiz/menagerie`
on Linux and macOS), not in the project. Robots with their meshes are tens to
hundreds of MB, so five projects using one robot do not keep five copies.

Each download is **one version** of the robot: the Menagerie's commit at the
time you download it, kept in a folder named after the robot and the version
(`franka_emika_panda@2f1a9c3b7d40`). The card shows it:
*Menagerie · shared download · version 2f1a9c3 (2026-11-13)*.

A robot downloaded with an Engine older than this shows *version unknown* and
**Download again (pinned)**: it downloads the current version so projects can
name it. The old folder stays.

## A project names its models, so another PC gets the same one

A MuJoCo Plant node saves its model **by name and version**, never as a path
into your user folder:

| The model | Saved in the graph as |
| --- | --- |
| A built-in task | `builtin:half_cheetah.xml` |
| A Menagerie robot | `menagerie:franka_emika_panda/scene.xml@<version>` |
| A file in the project | `project:mujoco/models/my_arm/arm.xml` |
| A file anywhere else | its full path (works on this PC only) |

The node's **Model** row shows the name (*Robotiq 2F-85 · Menagerie 370c439*)
and where it is: *copy in this project*, *shared download*, *built in*, or *not
in the project: other PCs will not find it*. For such a file, load it in the
MuJoCo Viewport and use Inspect > Model > **Copy into the project** (its folder is
copied to `mujoco/models/<folder>/`), then pick the copy with **Change...**.
The node checks its model on this PC when the graph opens: a status saved on
another PC is never trusted.
**Change...** picks another file; it is saved by its name and version too.

Open the project on another PC (or another account) that does not have that
version, and the node says so: *Not on this PC: MuJoCo Menagerie robotiq_2f85,
version ac6b2b0. The graph cannot run until it is.* **Download it** opens the
Environment Library, where the robot's card says *This project uses version
ac6b2b0* with **Download it**: it downloads exactly that version, and the node
finds it by itself a moment later (its pins come back).

## Copy a robot into the project

**Copy into the project** on a downloaded robot's card (or Inspect > Model in
the Viewport) copies the robot's folder, with its meshes and textures, to
`<project>/mujoco/models/<robot>/`, after a confirmation that gives its size. A project that has a copy of the robot
already keeps it: nothing is replaced (remove that folder to copy another
version). Copies stop at 2 GB.
Use it to:

- share a project as one folder (or zip) that works offline;
- edit the robot's files: the copy is the project's own, no other project sees
  it. Inspect says when the open model is a shared download, so you copy it
  before editing.

The nodes keep naming the robot and its version and use the project's copy
from then on; another PC without the copy downloads that version. A copy is
only offered for a download whose version is known.

**Preferences > Simulation > MuJoCo Viewport > Downloaded models**: *Keep in the
shared folder* (default) or *Copy into the project when used* (a shared
download is copied into the project when you Load it).

## Import from URL

**Import from URL...** takes an MJCF `.xml` file or a GitHub folder. With a
project open it is saved in the project (`mujoco/models/<name>/`, listed under
*In this project*) and travels with it; with no project open it goes to your
user folder (`.cyxwiz/imported/<name>`, listed under *Imported*).

## Keyframes

Keyframes saved in Inspect are kept in the project
(`mujoco/keyframes/<model file name>-<id>.json`), keyed by the model's name and
version, so another PC finds them for the same model, and two robots whose
files share a name (Menagerie's `scene.xml`) keep their own.
Keyframes saved before models were named this way (an older file per model
path) are not read; save them again.
