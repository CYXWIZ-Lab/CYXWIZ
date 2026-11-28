kind: fixed
area: training
page: usage/gpu_providers.md

The joint Adam step of reinforcement learning runs on an NVIDIA GPU now refuses a gradient whose shape differs from its parameter's, as the ordinary optimizer step always did. Before, it accepted any gradient with the same number of values.
