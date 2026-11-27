kind: changed
area: training
page: usage/node_dialogs.md

Training no longer waits for the device to read the loss and metrics batch by batch. Regression runs (loss, MAE, RMSE), text runs and every validation and test pass now sum them on the device and read them at the Data Loader's **Report every** points and at the end of each pass, as classification training already did. Regression and text training get faster, most of all on a GPU with a small model. A loss that becomes NaN or infinite now stops a regression or text run at the next report instead of at that batch, and the error names the batches since the report before; set **Report every** to 1 to have it name the batch.
