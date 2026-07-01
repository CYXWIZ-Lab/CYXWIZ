# CyxWiz Studio UX Template

This document captures the current UX direction for CyxWiz Studio screens. Use it as the baseline for future dashboard, debugger, launcher, and workflow screens.

## Core direction

CyxWiz Studio should feel like a professional ML engineering workstation:

- Dark, focused, low-noise interface.
- Center-based layouts instead of edge-heavy layouts.
- No visible boxed borders unless the border communicates state.
- Clear workspace hierarchy: title, short explanation, primary work area, secondary action lane.
- Engine facts should be visible without requiring users to inspect logs.

## Page structure

Use a centered content frame inside the full application viewport.

- Full viewport background can use dark gradients and subtle glow shapes.
- Main content should sit inside a max-width centered layout.
- Recommended max content width: about `1180` to `1240` px.
- Recommended outer horizontal margin: about `64` px.
- Do not align content directly to the window edge.
- Keep hero title and subtitle aligned to the same centered content x-position.

## Start page template

The Get Started screen uses this structure:

- Full-screen dark background.
- Top hero area:
  - `Get started`
  - One-line product intent subtitle.
- Main centered two-column layout:
  - Left work area for discovery and recent work.
  - Right action lane for project launch actions.
- Bottom-right secondary action:
  - `Continue without project`.

## Left work area

Purpose: help the user find an existing project or open a task starter graph.

Contains:

- Search bar for recent projects.
- Task starter graphs.
- Recent projects.

Rules:

- No visible panel border.
- No heavy child window frame.
- Use spacing and typography to separate sections.
- Starter graph rows should show:
  - icon
  - graph title
  - domain/task label
  - short description
  - primary `Open` action
- Recent projects should show:
  - folder icon
  - project name
  - path
  - last opened time

## Right action lane

Purpose: create or open work quickly.

Contains:

- `Launch workspace` heading.
- Short description.
- Primary `Create a new project` action.
- Workflow lanes:
  - Classic ML workflow
  - Deep Learning workflow
- Domain starters:
  - Tabular project
  - Vision project
  - NLP project
- File actions:
  - Open project or solution
  - Open project folder
  - Clone repository, planned/disabled

Rules:

- The right lane should have breathing room from the left lane and the window edge.
- Use text headings instead of separator bars.
- Avoid boxed borders around the lane.
- Primary action uses electric blue.
- Secondary actions use subdued dark cards/buttons.

## Borders and frames

Default rule: remove borders.

Avoid:

- visible card outlines
- boxed panels
- nested framed children
- divider lines used only for decoration
- unnecessary scrollbars

Allowed:

- subtle background contrast without hard outline
- status borders for warnings/errors/active state
- focus rings or hover state if needed for interaction clarity

## Scroll behavior

Avoid nested scrollbars on the start page.

- The start page should fit normal desktop viewport sizes.
- Prefer layout spacing and content sizing over internal scrolling.
- If content overflows in future screens, use one main page scroll instead of multiple child scroll areas.

## Typography and hierarchy

Use clear hierarchy:

- Large page title.
- Small one-line subtitle.
- Section headings in muted blue/gray.
- Body text in softer blue-gray.
- Primary action labels in high contrast.

Avoid:

- many equal-weight labels
- overuse of separators
- dense text blocks

## Color direction

Base:

- deep navy
- black-blue gradient
- subtle teal/blue glow accents

Actions:

- primary electric blue
- hover state brighter blue
- secondary dark navy card/button

Text:

- main text near white
- secondary text blue-gray
- disabled/planned text lower contrast

## Interaction rules

- Primary action should be obvious.
- Secondary actions should be present but not compete with the primary action.
- Tooltips are useful for paths, planned features, and longer explanations.
- The screen should communicate what the user can do within five seconds.

## Training and debugger UX extension

The same design principles apply to training and debugger screens:

- Show engine truth directly.
- Avoid hiding critical facts in logs.
- Use visible task progress, terminal reasons, checkpoint state, warnings, and active graph node state.
- Use one main scroll surface where possible.
- Avoid nested dashboard panels that hide information.

## Implementation notes for ImGui

For borderless centered screens:

- Use full-screen windows with custom background drawing.
- Use a max-width centered content area.
- Prefer `BeginChild(..., false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)` for layout-only groups.
- Set `ImGuiCol_Border` alpha to zero when the screen should be borderless.
- Prefer text headings over `SeparatorText` when divider lines are not desired.
- Use `ChildBg` only for subtle tonal separation, not framed cards.

