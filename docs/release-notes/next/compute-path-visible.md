kind: changed
area: training
page: usage/gpu_providers.md

The Engine now says why a run on a GPU computes on ArrayFire alone. A missing provider has a reason (not installed, not built, from another Engine version, turned off) and one sentence saying what to do, shown in Compile, on the RL Training Dashboard (Run details > Computes as, and the note under the header), in the Console and in the Studio Debugger. The RL note no longer blames small networks when the cause is a missing provider, the dashboard lists PPO's sampling, loss, advantages and Adam step with what computes each (SAC, TD3 and DQN list their Adam step too, also when the optimizers could not step together and never reached the provider), and Compile's provider row is correct under Strict (a provider failure stops a Strict run).
