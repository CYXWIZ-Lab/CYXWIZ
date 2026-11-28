kind: added
area: internals
page: architecture/backend-compute-frameworks.md

Building from source: the `gpu-dev` CMake preset asks for every neural provider whose toolkit is present (CUDA Toolkit, OpenCL SDK, oneAPI DPC++), and every configure now prints a provider table saying which were built and why one was not. The test switch `CYXWIZ_EXPECT_PATH` makes the RL tests and the RL gate tool fail when an operation did not run on the named provider.
