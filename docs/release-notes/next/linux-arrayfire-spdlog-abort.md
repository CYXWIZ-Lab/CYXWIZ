kind: fixed
area: training
page: usage/gpu_providers.md

**Linux: the Engine no longer aborts on its first device query when ArrayFire 3.10 with its oneAPI backend is installed** ("free(): invalid pointer"). Signal-processing and time-series features that use the FFT no longer crash on Linux with ArrayFire's CPU backend. ArrayFire brings its own copy of the spdlog logging library, and its CPU backend its own FFTW, and the Engine's copies were taking their place. The Engine now keeps its copies private, and the backend's messages still reach the Engine's log and the Console panel. One change on Linux: a plugin that logs through spdlog itself, instead of the plugin context's `LogInfo`, now writes only to the terminal, not to the Engine's log. The Simulation (MuJoCo) add-on and the example plugins log through their plugin context, so their messages reach the Engine's log on every platform.
