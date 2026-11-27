kind: fixed
area: training
page: usage/gpu_providers.md

**Linux: the Engine no longer aborts on its first device query when ArrayFire 3.10 with its oneAPI backend is installed** ("free(): invalid pointer"). ArrayFire's backends bring their own copy of the spdlog logging library, and the Engine's copy, a different version, was taking its place. The Engine now keeps its copy private, and the backend's messages still reach the Engine's log and the Console panel. One change on Linux: a plugin that logs through spdlog itself, instead of the plugin context's `LogInfo`, now writes only to the terminal, not to the Engine's log. The Simulation (MuJoCo) add-on's own messages are among them until it moves to `LogInfo`.
