# The MuJoCo Viewport

The MuJoCo plugin's 3D view: it shows a MuJoCo model, runs it, and shows a
trained policy playing. Open it from **Simulation > MuJoCo Viewport**, or it
opens by itself when you press **Play policy** on a MuJoCo task (see
[Train, play and export a policy](rl_walkthrough.md)). The plugin must be
loaded (**Plugin Manager**, MuJoCo Simulation).

## The control row

| Control | What it does |
| --- | --- |
| **Run** / **Pause** / **Resume** | Runs the model in the Viewport, pauses it, goes on. Off while a graph run or a policy playback drives the scene (hover for the reason) |
| **Step** | Pauses and moves the model one step |
| **Reset** | Puts the model back at its start |
| **Stop** | Ends the Viewport's own run |
| **Speed** | Times real time, typed (0.25 to 8); **As fast as possible** steps without waiting |
| **Camera: ...** | The camera the scene is seen through (below) |
| **Record** | A screenshot or a recording (below); **Stop recording** while one runs |
| Status | Running, Paused, Stopped, Policy playback (or the Engine's word for it: Training preview, Training), Driven by the graph run, or Recording with its length and frames |
| **View** | What the scene shows (below) |
| **Stress** / **Hide Stress** | The Stress panel beside the scene: kick the robot, throw objects at it, change the ground (below) |

On a narrow Viewport the row wraps; nothing is hidden.

## The camera

| Choice | The view |
| --- | --- |
| **Free (drag to orbit)** | Drag to orbit, right-drag to pan, scroll to zoom. A model you open starts here, aimed at the whole model; from a model camera it starts at that camera's view |
| **Follow a body** | Keeps one body (its centre of mass) in the middle while it moves, at a distance that frames it and what hangs from it; drag to orbit and scroll to zoom around it. Double-click a body in the scene to follow it (easiest while the scene is still: paused or stopped) |
| **The model's cameras** | The cameras the model file defines, by name. Gymnasium's HalfCheetah, Hopper and Walker2d have one called **track** that runs beside the robot |
| **Reset view** | Back to the free camera, aimed at the whole model |

A policy playback starts with the model's **track** camera, or, for a model
without one, following its root body, so a running robot stays in view.
Change that in **Preferences > Simulation > MuJoCo Viewport**.

## View

| Section | Options |
| --- | --- |
| Show in the scene | Contact points, Contact forces, Joints, Actuators, Centre of mass, Inertia boxes, Transparent, Wireframe, Convex hulls |
| **Frames** | The axes of each body, geom or site, or of the world |
| **Labels** | The names of bodies, joints, geoms, sites or actuators, or the contact points |
| **Groups** | The model's visibility groups 0 to 5 for geoms and sites (a model puts collision shapes or markers in their own groups); MuJoCo shows 0, 1 and 2 |
| **Rendering** | Shadows, Reflections, Sky, Fog, and **Endless floor** (on by default): a model's level floor is drawn wherever the camera looks. A floor plane collides everywhere, but a model draws only its given size (Gymnasium's HalfCheetah: 40 m), so a fast runner would leave the drawn floor. The physics is the same either way |
| **Live numbers during playback** | The episode, step, return and action on the scene (below) |
| **Draw the scene** | Off stops drawing (the model still runs) |
| **Keep these as the default** | The Viewport starts with these options from now on |
| **MuJoCo's options** | Back to MuJoCo's own choices |

Ticking an option keeps the menu open, to set several. The options last for
the session unless you keep them as the default.

## Live numbers during playback

While a policy plays, the top left of the scene shows the played episode:
**Episode** (counted from the start of the playback), **Step** (of the
episode's limit), **Return** (the rewards so far in this episode), **Last
episode** (the return of the episode before, once one has ended) and
**Action** (the last action, up to eight values). While a run trains, the
RL Training Dashboard's **Watch progress** shows a live preview here: the
line under the task's name and the status say **Preview of update 1,580**
and **Training preview**, and the first row says **Networks of** that
update. **Raw training copy** shows copy 0 of the training as it learns
(**Training copy 0**); the Stress panel is off for it (a disturbance would
change what training learns). See the [RL walkthrough](rl_walkthrough.md) (While it trains).

With a second policy playing beside it as a ghost (Policy Playback: **Compare
with**, then **Show as a ghost**), the scene draws the ghost see-through and
tinted, and the numbers get a **Played** and a **Ghost** column: **Policy**
(each one's name), **Episode**, **Step**, **Return** and **Last episode**. A
step back on the timeline shows both where they were, with that step's numbers. See the
[RL walkthrough](rl_walkthrough.md) (Compare two policies side by side).

## Record

**Record > Screenshot (PNG)** saves the scene as it is now.
**Record > Record video...** asks for:

| Choice | Meaning |
| --- | --- |
| **GIF** | An animated GIF at most 640 pixels wide that plays at real speed, for a README or a chat |
| **PNG frames** | One PNG a frame at the Viewport's size, in a folder: for a video editor, or `ffmpeg -framerate 20 -i frame_%06d.png run.mp4` (use the model's frames a second the dialog shows) |
| **Until the episode ends** | Stops by itself when the shown episode ends |
| **Until Stop** | Records until you press **Stop recording** |

A frame is taken each time the simulation moves (one a simulation step,
the rate the dialog shows: 20 a second for a HalfCheetah playback, whose
step is five 0.01 s physics steps), not each time the screen is drawn, so
the GIF plays at the speed of the simulation whatever the Viewport's speed.
When the screen draws slower than the simulation steps, the GIF holds that
frame longer, so its timing stays right. A recording stops by itself at
6,000 GIF frames or 20,000 PNG frames.

The GIF's 256 colours are chosen from the first frame, so start recording
with the scene in view. A 640-pixel GIF of a running HalfCheetah over the
checker floor is about 0.5 MB a second (a 34 s episode: 19 MB); PNG frames
are larger still.

Files go to the open project's **exports** folder, named after the model and
the time (`half_cheetah_20261112-064512.gif`, `..._frames/frame_000001.png`).
Without an open project, Record is off and says so. When a recording is
written, the bottom of the scene says where for a few seconds; the log
has the full path.

MP4 needs the Engine's media support, which is not in this build: record
PNG frames and join them with a video editor or ffmpeg.

## Stress: does the policy cope?

A trained policy has only seen the plain task. The Stress panel (**Stress** in
the control row) disturbs the copy the policy plays, or the Viewport's own
run, so you can watch whether it copes. Every value is a typed field.

| Section | What it does |
| --- | --- |
| **Push** | **Kick back** / **Kick forward** / **Up** / **Any way** (a random direction) change the robot's velocity by the typed **Kick** (m/s): back and forward along +x, the way Gymnasium's runners go. **over** a typed time (seconds) makes it a push held that long, as in Geijtenbeek et al. 2013 (0.2 s); 0 is a sudden kick. **Ctrl+drag** a body in the scene pulls it with a spring while the mouse is down (MuJoCo's own perturbation) |
| **Throw** | A **Ball**, **Box**, **Capsule** or **Cylinder**, or **Any** (one at random each throw), with typed sizes: a ball's diameter; a box's length × width × height; a capsule's or cylinder's length × diameter. Typed mass and speed. **From**: Ahead, Behind, Left, Right or Above, or a typed **Angle** (around the robot: 0 ahead, 90 left, 180 behind, 270 right; and up from level) and **Distance**; the dial at the top right of the scene sets the angle around by click or drag. **Tumble** throws it turned and spinning. It is aimed at where the robot will be when it arrives. Thrown objects stay in the scene until the episode ends; eight can be out at once |
| **Auto disturb** | **A random kick every** N simulated seconds (back, forward, up or any way, at the kick above). **A random throw every** N seconds: the object above with its sizes and mass scaled by a factor in the typed ranges, from any side, level to 60° up (any shape when Any). The ground's seed repeats the same random kicks and throws in each episode |
| **Ground** | **Flat**, **Rough** (random bumps up to 2-15 cm), **Slopes** (hills up to 3-20 degrees), **Steps** (flights of 3-5 steps of 2-15 cm) or **Obstacles** (blocks of 5-30 cm, 2-5 m apart); the heights grow with **Difficulty** (0 to 1); the same **Seed** gives the same ground. It runs along +x from 2 m ahead of the start, for 300 m. **Next episode** puts it in when the episode ends; **Now (new episode)** cuts the episode (counted as cut, not as a fall) and starts one on it |
| **Events** | Each disturbance with its simulated time, and what followed within 3 s: **stayed up**, **flipped over** (the robot's root upside down) or **fell** (the task ended the episode) |

During playback the live numbers show the **Ground** when it is not flat.
**Keep these as the default** saves the panel's choices for next time.

The policy plays on: a disturbance never changes what it observes (the
robot's own joints), only what happens to it. HalfCheetah never ends an
episode by itself, so for it a fall shows as **flipped over**.

## Inspect: look inside the model

**Inspect** (beside Stress; the two share the side of the scene) shows and
changes the Viewport's own run. While a policy plays or a graph run steps the
model, it shows and changes nothing.

- **Select a body** by double-clicking it in the scene (with Inspect closed a
  double-click also follows the body with the camera, as before). The body is
  drawn highlighted with its name and a dot at its centre of mass. Inspect
  shows its **mass** (and the robot's), its **centre of mass** and its
  **speed**; **Follow it with the camera**, **Select parent**, **Clear**.
- **Its joints**: a hinge or slide joint's position is typed in while the run
  is paused (kept within the joint's range; the state is computed again), with
  its velocity, stiffness and damping.
- **Actuators**: each actuator's control typed in (within its range), with
  the force it puts on its joint (gear times the actuator's force: 0.40 on
  HalfCheetah's bthigh is 48 N·m). Paused, the force shows at once; a run
  steps with zeros unless **Hold these controls while it runs** is on.
  **Set all to 0**.
- **Keyframes**: **Save the state** under a name; **Load** puts it back (the
  joints, velocities and controls), **Delete** removes it. They are saved with
  the project in `mujoco/keyframes/<model file name>-<id>.json` (one file a model
  file, also when two models share a file name), so they come back next
  time; a keyframe saved for another version of the model file is refused with
  the reason. A model file's own `<key>` elements are listed too.
- **Physics** (the Viewport's own run): the **timestep** (with the
  environment's frame skip beside it), the **integrator** (Euler, RK4,
  Implicit, Implicit fast), the **solver** (PGS, CG, Newton) and its
  iterations, **gravity** (x, y, z), and switches for contacts, joint limits,
  friction loss, actuation, gravity, springs, dampers, warm start and energy.
  The line beside **Back to the model file's values** says what differs from
  the file. The options stay through a Stress ground and a Reload; opening
  another model file starts with its own. A played policy keeps its
  environment's physics.
- **Compare**: the ghost's name while a second policy plays beside the
  played one (Policy Playback: **Compare with**, then **Show as a ghost**),
  or why it is not drawn (it plays another model file or ground). Choosing
  and removing it is done in Policy Playback.
- **Reload the model file** reads the file again (after editing it on disk),
  keeping the camera and the scene; the state starts over. A file that does not
  build any more leaves the model as it was and says why.

## Timeline: look back at what happened

The strip under the scene keeps the steps of what the Viewport shows: the
last 30 seconds of simulated time of the Viewport's own run, or the whole
episode a policy is playing (a new episode starts it again).

- **Drag along it** (or click), or press **Step back** and **Step**, to show
  an earlier step: the scene goes back to that exact state (the joints,
  velocities, contacts and controls). The Viewport's own run pauses; the scene
  says **A step back** with the shown episode and step, and the numbers under
  it are that step's (its episode, step and the reward summed so far).
- **Back to live** (shown only while a step back is) puts the newest step
  back and runs again if the run was running.
- **Run**, **Resume** or **Step** from a step back goes on from the shown
  step: the kept steps after it are dropped, so you can try again from just
  before a fall, for example after a kick from the Stress panel.
- While a policy plays, a step back holds the scene on that step, and the live
  numbers show that step (a ghost's too); the policy plays on, and **Back to
  live** shows it again.
- The red marks on the track are the Stress panel's **fell** and **flipped
  over** events, at the step they happened.

**Plots** (beside the step buttons) draws the last 10 seconds of the kept
steps under the timeline, all on one scale (the line under the plot says the
range); a line marks the shown step. Pick what it draws:

| Choice | The line |
| --- | --- |
| **Reward** | The reward of each step |
| **Selected joint** | The position of the first joint of the body selected with a double-click (named after the joint once one is selected) |
| **Root speed** | The speed of the robot's root body (m/s) |
| **Controls** | Every actuator's control |
| **Sensors** | The model's first 8 sensor values (shown only for a model with sensors) |

## Preferences > Simulation > MuJoCo Viewport

| Setting | Default |
| --- | --- |
| **Camera for playback** | Model's track camera (else following the root body); or Follow the root body; or Free |
| **Live numbers** | On |
| **Record video** | GIF, until the episode ends |
| **Scene options** | MuJoCo's; **Use MuJoCo's** resets options kept from the View menu |
| **Inspect panel** | Closed; typed controls not held |
| **Plots** | Not shown under the scene; Reward and the selected joint picked |
| **Stress panel** | Closed; a sudden 1.5 m/s kick; a 0.2 m 2 kg ball at 6 m/s from 3 m ahead, no tumble; no random kicks (every 3 s when on) or throws (every 1.5 s, size × 0.5-2, mass × 0.5-3); Rough 0.40 seed 7 offered |

Changing a choice in the Viewport does not change these, except **View >
Keep these as the default** and the Stress panel's **Keep these as the
default**.

## Watch again without training again

Save a trained policy once (**Save policy...** on the RL Training
Dashboard), then **File > Open Model File...** on the `.cyxmodel` and **Play
policy** whenever you want to watch it: no training run is needed. See
[Train, play and export a policy](rl_walkthrough.md).

## Limits

- The Viewport shows one copy of a training run's environments (the first).
- The Stress panel acts on the played policy only, not on a ghost.
- The timeline keeps at most 100,000 steps of the Viewport's own run and
  20,000 of a played episode.
- Training with pushes and rough ground, and the stress test that compares
  two saved policies by difficulty, are in the
  [RL walkthrough](rl_walkthrough.md) (Robustness, and Stress test).
- Recording to MP4 needs the Engine's media support (not in this build).
