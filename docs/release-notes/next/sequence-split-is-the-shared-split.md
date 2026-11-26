kind: fixed
area: training
page: usage/seeds_and_reproducibility.md

Text sequence training (token tagging and causal-LM windows) now splits by the same rule as tables. A Data Split's seed and shuffle decide which sequences go to Train, Validation and Test; before, it cut them in file order whatever its seed. Without a Data Split, validation no longer reads the Train sequences: Train is the first share, Validation the next and Test the rest, as for tables. Before, every part read every sequence, so the validation scores were the training scores.
