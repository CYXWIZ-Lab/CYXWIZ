kind: fixed
area: training
page: usage/model_tools.md

A learning rate range test on a graph with an Embedding Output node no longer writes the embeddings file. The range test trains a throwaway copy of the model at rising learning rates, so the file it wrote replaced the one from your last real training with embeddings from that copy. The run's crash record now also says the range test finished, instead of saying every epoch completed. In a language-model run that saves a resume checkpoint every few steps, a loss that turns NaN between two reports now stops the run at the next checkpoint. Before, checkpoints kept being saved with the broken weights until the next report, and resuming from the latest one failed again straight away.
