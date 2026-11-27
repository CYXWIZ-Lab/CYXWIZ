kind: changed
area: training
page: usage/seeds_and_reproducibility.md

Tables, images, audio and text now train through one loop. Every training batch now reseeds the model's random draws from the model seed, the epoch and the batch number, as text runs already did. A batch's dropout masks no longer depend on what drew before it. Tabular, image and audio runs with dropout or noise layers draw different masks than in earlier versions, so their results can move a little once. A plugin's `OnBatchEnd` gets each batch's own `train_loss` for every kind of data, as the plugin API says. The log's loader-timing line now reads "Loader timing epoch N" for every kind of data and also counts the samples.
