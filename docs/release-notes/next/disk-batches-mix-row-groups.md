kind: fixed
area: data
page: usage/big_data.md

Training on a table kept on disk now mixes rows from several row groups in every shuffled batch: each epoch keeps as many row groups in memory as fit in a quarter of the data budget (at least 4) and draws each batch's rows at random from all of them. Before, one row group was read at a time, so a file sorted by class gave batches of one class for long stretches. Causal-LM token windows on disk read row groups in the same windows. A warning says when most row groups of the label column hold one class.
