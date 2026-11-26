kind: fixed
area: training
page: usage/seeds_and_reproducibility.md

A resumed training run now replays the class-balanced rows of each epoch (under- and oversampling, the weighted sampler) and audio augmentation: they come from the DataLoader seed and the epoch number, as the batch order does. Before, they followed how many epochs the process had started, so a run resumed at epoch 30 drew epoch 1's balanced rows while saying the resume was exact. Image and audio training also shows the right number of batches per epoch from epoch 2 on (it showed the validation part's). Balanced and audio runs draw different rows and augmentation than before this release, once.
