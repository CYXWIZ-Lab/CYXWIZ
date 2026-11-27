kind: changed
area: extending
page: usage/mujoco_models.md

**The Simulation (MuJoCo) add-on 1.1.0 runs MuJoCo 4.0.1** (it was 3.17.0). The Gymnasium tasks give the same results as Gymnasium on MuJoCo 4.0.1, to the same closeness as before. Three things differ from 3.17.0: simulation results move in the last digits (up to 4e-13 over a 200-step HalfCheetah run), so a run made on 3.17.0 does not repeat bit for bit; a saved full physics state also holds plastic state for models that use plasticity; and MuJoCo 4 refuses some model files written for MuJoCo 3 (those using `composite`, and some flex features), naming the reason when the model is loaded. The models that come with the add-on load as before, and so do the Menagerie robots that were checked (Google Robot, Kinova Gen3, Robotiq 2F-85); a Menagerie robot downloaded later is checked when it loads, and one MuJoCo 4 refuses says why. Every plugin that uses MuJoCo in one Engine must be built for the same MuJoCo version.
