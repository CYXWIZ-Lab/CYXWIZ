kind: fixed
area: training
page: usage/node_dialogs.md

With gradient accumulation, each epoch's last, partly filled set of batches now gets its optimizer step within that epoch. Before, when batches were prefetched, the epoch could end before the loader reported its end, and those batches' gradients went into the next epoch's first step, after the epoch-end checkpoint. The epoch's reported loss and accuracy also count every batch when the final batch was not a report batch.
