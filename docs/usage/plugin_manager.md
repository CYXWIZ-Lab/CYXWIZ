# Plugin Manager

**Apps > Plugin Manager** lists every C++ plugin the Engine has loaded and
every Python node bundle it found, one card each. **Apps > Loaded plugins >
a plugin** with no nodes opens it on that plugin's card (one with nodes opens
the Nodes panel on them). Data Studio > Explore > Plugins and a graph with a
missing extension node also open it.

At the top: a search box (name, id, author, description; it filters the
bundles too), **Refresh** (looks in the plugin folders again and starts what
is not running) and **Install...**. The plugins are in name order, so a card
keeps its place when another plugin loads or goes. The footer counts every
state, e.g. *3 plugins: 1 active, 1 waiting, 1 failed*.

## A plugin's card

- **Its name**, version and author, and its state at the right in words:

  | State | What it means | Its button |
  | --- | --- | --- |
  | Active | Started; its nodes, windows, formats and analytics are there | Disable |
  | Not started | Loaded but not started | Start |
  | Waiting | It waits for a plugin it needs (named in its `plugin.json`) and starts when that one does; or it waits for your answer to its permission request | Start (not while it waits for your answer) |
  | Failed | It could not start; the card says why | Try again |
  | Disabled | You turned it off; its nodes and windows are hidden | Enable |

- Its description, then why it is in that state (nothing when Active), then
  what it adds (*Adds: 2 windows, 3 simulation nodes ...*).
- **Load when the Engine starts**, and what that means for its nodes and
  windows after a restart.
- **Permissions (2 allowed)** opens the permissions it asks for, each with
  your answer (below).
- **Actions**: **Open the plugin folder**, **Copy its id**, and
  **Unload...**, which first says what goes (its windows close; graphs with
  its nodes cannot run until it is loaded again) and asks *Keep it* or
  *Unload*.

## Permissions

A plugin declares what it needs in its `plugin.json`. Low-risk permissions
(GPU compute, dataset access, training hooks, UI changes) are allowed without
a question: the card shows *Allowed (low risk, not asked)*. For the others
(files on your computer, the network, system commands, Python)
the Engine asks when the plugin first starts:

- The request names the plugin, its id, version and author, and lists what
  it wants with a line on each. Tick the ones to allow.
- **Allow selected** stores your answers (it is off, with the reason, until
  something is ticked). **Start without these** (also Esc) denies them all.
- The plugin starts either way: without a permission, the parts that need it
  stay off. One request is shown per plugin; pressing Start or Refresh again
  while it asks does not open another.

On the card, each permission that needs an answer has **Allowed**,
**Denied** and **Ask again** (asks at the next start). Answers are kept per
plugin version in `%APPDATA%\cyxwiz\plugin_permissions.json`
(`~/.cyxwiz/` elsewhere). A change takes effect the next time the plugin
starts (Disable, then Enable).

## Installing a plugin

**Install...** opens *Install a plugin*: type or **Browse...** to the folder
that holds the plugin's `plugin.json`, then **Install** (or Enter).

- A problem shows under the path and the dialog stays open with the path as
  you typed it: no folder there, a file instead of a folder, no
  `plugin.json`, a `plugin.json` that cannot be read, or a plugin already
  installed (disabled: press Enable on its card; failed: Try again; loaded:
  Unload it first to load a new build).
- While it installs, the Engine stays responsive and **Cancel** stops it
  before loading starts.
- When some of the plugin's nodes were refused, the dialog says how many and
  why, and stays open; otherwise it closes and the card appears.

To build a plugin see [Extension nodes](extension_nodes.md) and the SDK's
quick start.
