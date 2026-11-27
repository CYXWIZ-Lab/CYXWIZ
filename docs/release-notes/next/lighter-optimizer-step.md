kind: changed
area: training
page: usage/model_tools.md

Each training step does less work besides the model itself. Without gradient accumulation, the optimizer now takes the gradients as the backward pass made them, instead of weighting each by the batch's size and dividing it back. Trained weights can differ from earlier versions in the last digits (the old multiply and divide rounded); results stay within the same tolerance as PyTorch. The Health view's mean gradient per layer is now the mean over the epoch's report steps (the first and last batch and every **Report every** batches), not over every step. The training trace keeps a full event only for report batches; the other batches add their stage times to the totals the Studio Debugger shows.
