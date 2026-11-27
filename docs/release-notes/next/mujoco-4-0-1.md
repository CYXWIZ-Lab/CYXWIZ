kind: changed
area: extending
page: usage/mujoco_models.md

**The Simulation (MuJoCo) add-on now runs MuJoCo 4.0.1** (it was 3.17.0). The Gymnasium tasks give the same results as Gymnasium on MuJoCo 4.0.1, to the same closeness as before. Three things differ from 3.17.0: simulation results move in the last digits (up to 4e-13 a step on HalfCheetah), so a run made on 3.17.0 does not repeat bit for bit; a saved full physics state also holds plastic state for models that use plasticity; and MuJoCo 4 refuses some model files written for MuJoCo 3 (those using `composite`, and some flex features), naming the reason when the model is loaded. The models that come with the add-on and the Menagerie robots load as before. Every plugin that uses MuJoCo in one Engine must be built for the same MuJoCo version.
