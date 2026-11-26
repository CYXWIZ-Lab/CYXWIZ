kind: fixed
area: training
page: usage/seeds_and_reproducibility.md

A training run without a model seed now draws one at its start and records it, so stopping and resuming it continues exactly where it was, as a seeded run does; before, such a resume was marked not exact and its dropout and weight draws differed from the uninterrupted run's. Setting the model seed is still the way to repeat a run from its start.
