# Keyboard Shortcuts

Complete reference for all keyboard shortcuts in CyxWiz Engine.

## Global Shortcuts

These shortcuts work anywhere in the application.

### Command Palette

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Open Command Palette |

The Command Palette provides quick access to all tools and commands. Type to search, use arrow keys to navigate, Enter to execute.

### File Operations

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Script |
| `Ctrl+O` | Open Script |
| `Ctrl+S` | Save |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+Alt+S` | Save All |
| `Ctrl+Shift+N` | New Project |
| `Ctrl+Shift+O` | Open Project |
| `Alt+F4` | Exit |

### Undo/Redo

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+Shift+Z` | Redo (alternative) |

### Panel Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+1` | Focus Node Editor |
| `Ctrl+2` | Focus Script Editor |
| `Ctrl+3` | Focus Console |
| `Ctrl+4` | Focus Properties |
| `Ctrl+5` | Focus Asset Browser |
| `Ctrl+6` | Focus Viewport |

### Training

| Shortcut | Action |
|----------|--------|
| `F5` | Start Training / Run Script |
| `Shift+F5` | Stop |
| `F6` | Pause |

### Help

| Shortcut | Action |
|----------|--------|
| `F1` | Open Documentation |

---

## Script Editor Shortcuts

### Text Selection

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Select All |
| `Shift+Arrow` | Extend Selection |
| `Ctrl+Shift+Arrow` | Select Word |
| `Ctrl+L` | Select Line |
| `Alt+Click` | Column Selection |

### Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+X` | Cut |
| `Ctrl+C` | Copy |
| `Ctrl+V` | Paste |
| `Ctrl+D` | Duplicate Line |
| `Ctrl+Shift+K` | Delete Line |
| `Alt+Up` | Move Line Up |
| `Alt+Down` | Move Line Down |
| `Tab` | Indent |
| `Shift+Tab` | Outdent |

### Comments

| Shortcut | Action |
|----------|--------|
| `Ctrl+/` | Toggle Line Comment |
| `Ctrl+Shift+/` | Toggle Block Comment |

### Find & Replace

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Find |
| `F3` | Find Next |
| `Shift+F3` | Find Previous |
| `Ctrl+H` | Replace |
| `Ctrl+Shift+F` | Find in Files |
| `Ctrl+Shift+H` | Replace in Files |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+G` | Go to Line |
| `Ctrl+Home` | Go to Start |
| `Ctrl+End` | Go to End |
| `Ctrl+Left` | Word Left |
| `Ctrl+Right` | Word Right |
| `Home` | Line Start |
| `End` | Line End |
| `Page Up` | Page Up |
| `Page Down` | Page Down |

### Execution

| Shortcut | Action |
|----------|--------|
| `F5` | Run Script |
| `Ctrl+Enter` | Run Selection |
| `Shift+F5` | Stop Execution |

---

## Node Editor Shortcuts

### Selection

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Select All Nodes |
| `Escape` | Clear Selection |
| `Click` | Select Node |
| `Ctrl+Click` | Add to Selection |
| `Shift+Click` | Range Select |
| `Drag` | Box Select |

### Clipboard

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Copy Nodes |
| `Ctrl+X` | Cut Nodes |
| `Ctrl+V` | Paste Nodes |
| `Ctrl+D` | Duplicate Nodes |
| `Delete` | Delete Selected |
| `Backspace` | Delete Selected |

### History

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |

### View

| Shortcut | Action |
|----------|--------|
| `F` | Frame Selected |
| `Home` | Frame All |
| `Mouse Wheel` | Zoom |
| `Middle Drag` | Pan |
| `Right Drag` | Pan (alternative) |

### Search

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Search Nodes |
| `F3` | Next Match |
| `Shift+F3` | Previous Match |

### Node Creation

| Shortcut | Action |
|----------|--------|
| `Right-Click` | Open Context Menu |
| `Space` | Open Add Node Menu |
| `A` | Add Node (quick) |

### Connection

| Shortcut | Action |
|----------|--------|
| `Left-Click Drag` | Create Connection |
| `Alt+Click Link` | Delete Connection |

---

## Console Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Execute Command |
| `Up Arrow` | Previous Command |
| `Down Arrow` | Next Command |
| `Ctrl+L` | Clear Console |
| `Ctrl+C` | Cancel Execution |
| `Tab` | Auto-complete |

---

## Asset Browser Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Open Selected |
| `Delete` | Delete Selected |
| `F2` | Rename |
| `Ctrl+C` | Copy Path |
| `Ctrl+Shift+E` | Show in Explorer |

---

## Table Viewer Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Copy Cell |
| `Ctrl+Shift+C` | Copy Row |
| `Ctrl+F` | Filter |
| `Ctrl+E` | Export |
| `Page Up` | Previous Page |
| `Page Down` | Next Page |

---

## Training Dashboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+E` | Export to CSV |
| `R` | Reset Zoom |
| `Left Drag` | Zoom Box |
| `Right Drag` | Pan |

---

## Customizing Shortcuts

Shortcuts can be customized via **Edit > Preferences > Keyboard Shortcuts**.

### Shortcut Configuration

1. Open **Edit > Preferences**
2. Select **Keyboard Shortcuts** tab
3. Find the action to customize
4. Click the shortcut field
5. Press your desired key combination
6. Click **Apply**

### Shortcut Categories

| Category | Description |
|----------|-------------|
| **General** | Global application shortcuts |
| **Script Editor** | Text editing shortcuts |
| **Node Editor** | Graph manipulation shortcuts |
| **Console** | REPL shortcuts |
| **Navigation** | Panel navigation |

### Conflict Resolution

If a shortcut conflicts with an existing binding:
- A warning appears showing the conflict
- You can choose to override or cancel
- Overridden shortcuts become unbound

### Exporting/Importing

Shortcut configurations are saved in:
- Project settings (project-specific)
- User settings (global default)

Export format: JSON

```json
{
  "shortcuts": {
    "file.new": "Ctrl+N",
    "file.open": "Ctrl+O",
    "edit.undo": "Ctrl+Z",
    "edit.redo": "Ctrl+Y"
  }
}
```

---

## Platform Differences

### macOS

On macOS, `Ctrl` is replaced with `Cmd`:

| Windows/Linux | macOS |
|---------------|-------|
| `Ctrl+S` | `Cmd+S` |
| `Ctrl+Z` | `Cmd+Z` |
| `Ctrl+C` | `Cmd+C` |
| `Alt+...` | `Option+...` |

### Function Keys

Some systems may require pressing `Fn` for function keys:
- `Fn+F5` for Run
- `Fn+F1` for Help

---

## Quick Reference Card

```
+----------------------------------------------------------+
|                    CyxWiz Shortcuts                       |
+----------------------------------------------------------+
| GLOBAL                                                    |
| Ctrl+P     Command Palette    Ctrl+S     Save            |
| Ctrl+Z     Undo               Ctrl+Y     Redo            |
| F5         Run/Train          Shift+F5   Stop            |
+----------------------------------------------------------+
| SCRIPT EDITOR                                             |
| Ctrl+D     Duplicate Line     Ctrl+/     Toggle Comment  |
| Ctrl+F     Find               Ctrl+H     Replace         |
| Ctrl+G     Go to Line         Ctrl+Enter Run Selection   |
+----------------------------------------------------------+
| NODE EDITOR                                               |
| Ctrl+A     Select All         Delete     Delete Node     |
| Ctrl+C/X/V Copy/Cut/Paste     Ctrl+D     Duplicate       |
| F          Frame Selected     Home       Frame All       |
| Space      Add Node           Ctrl+Z/Y   Undo/Redo       |
+----------------------------------------------------------+
| PANELS                                                    |
| Ctrl+1     Node Editor        Ctrl+2     Script Editor   |
| Ctrl+3     Console            Ctrl+4     Properties      |
| Ctrl+5     Asset Browser      Ctrl+6     Viewport        |
+----------------------------------------------------------+
```

---

**Next**: [Themes](themes.md) | [Command Palette](command-palette.md)
