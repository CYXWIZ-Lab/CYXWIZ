# Interactive Tutorials

**Help > Interactive Tutorials** starts a tutorial by name (a done one shows
a tick) or opens the list with **Browse All Tutorials...**.

| Tutorial | Steps | What it covers |
| --- | --- | --- |
| Getting Started | 10 | The menu bar, the Asset Browser, CyxWiz Studio, Properties, the Console, the Script Editor, the Viewport and the Training Dashboard |
| Your First Model | 6 | A Data Input, two Dense layers and an Output node, connected |
| Training Basics | 6 | Loading a dataset, an optimizer, a loss, and starting training |

## The list

Each tutorial shows its description, its status (*6 steps*, *In progress,
step 3 of 6*, *Done*) and a button: **Start**, **Resume** (one closed before
its end, from the step it was closed on, in this session) or **Start again**.
**Reset progress...** asks first, then marks every tutorial as not done.

## A step

A card shows the step's number, title, text and what to do now. When the step
is about a window, that window is outlined in the accent colour and the card
sits beside it; when the window is closed the card says so (View > Panels
opens it). Dots at the bottom of the screen show the steps done and the one
you are on.

- **Next** (or the right arrow) and **Previous** (or the left arrow); the last
  step's button is **Complete**, which marks the tutorial done.
- The close button on the card (or Esc) stops it; **Resume** in the list goes
  back to that step.
- The arrows and Esc work while the card has the focus (it takes it at each
  step; click it again after working in another window).
- You move on by pressing Next: the tutorial does not watch what you do.

Progress is kept in your settings folder, `%APPDATA%\CyxWiz\tutorial_progress.json`
(`~/.cyxwiz/` elsewhere).
