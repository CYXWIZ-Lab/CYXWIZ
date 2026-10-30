# CyxWiz Engine user guide

How to use the Engine, one page per feature. Each page describes what the
current build does; limits are stated on the page.

To write your own nodes in Python or C++, see
[Extending CyxWiz](../extending/README.md).

## Getting around

| Page | What it covers |
| --- | --- |
| [The CyxWiz Studio bar](studio_toolbar.md) | The bar at the top of the canvas |
| [Windows of their own](windows.md) | Moving the big windows out of the main window |
| [Asset Browser](asset_browser.md) | The project's folders and files, previews, new files |
| [Properties](properties.md) | The Properties panel for the selected node |
| [Console and logs](console_logs.md) | Logs, the Python REPL and commands |

## Building a graph

| Page | What it covers |
| --- | --- |
| [Adding nodes](node_palette.md) | The Nodes panel, search and the right-click menu |
| [Node dialogs](node_dialogs.md) | Configuring a node in its dialog |
| [Branched graphs](branched_graphs.md) | Split and merge inside a model |
| [Your own nodes](extension_nodes.md) | Python node bundles and C++ plugins in the Engine: install, approve, use |

## Data

| Page | What it covers |
| --- | --- |
| [Where the tools get their data](tool_datasets.md) | The Data Input node as the one way in |
| [Data Studio](data_studio.md) | Bring in, prepare, explore, validate, results, publish |
| [Loaded data](loaded_data.md) | What the Engine holds in memory and when it lets go |
| [Data bigger than memory](big_data.md) | Tables on disk and the data budget |
| [Table Viewer](table_viewer.md) | Sorting, filtering and formatting a table |
| [SQL steps and the SQL Workbench](sql-workbench.md) | SQL steps and guided steps |
| [Scaling and transforms](data_transforms.md) | Scaling and shape changes as graph steps |
| [Image transforms](image_transforms.md) | Image nodes between the Data Input and the model |
| [Quality Analyzer](quality_analyzer.md) | Leaving poor images out of training |
| [Splitting images and audio](image_split.md) | Train, validation and test parts of an image or audio dataset |
| [Text vectorizers](text_vectorizers.md) | TF-IDF and count vectorizers |
| [Word embeddings](word_embeddings.md) | word2vec, GloVe and fastText vectors |

## Exploring and analysing

| Page | What it covers |
| --- | --- |
| [Statistics](statistics.md) | Hypothesis tests and regression |
| [Clustering](clustering.md) | Explore > Groups and the clustering nodes |
| [Dimensionality reduction](dimensionality_reduction.md) | Explore > Projection, PCA, t-SNE, UMAP |
| [Linear algebra tools](linear_algebra.md) | Matrix calculator, SVD, QR, Cholesky, Eigen |
| [Feature Importance](feature_importance.md) | Which columns matter to a model |
| [Analysis demo graph](analysis_demo_graph.md) | Analysis nodes working together on one dataset |
| [Plotting](plotting.md) | Plots, the Plot node and Python plotting |
| [Dashboard node](dashboard.md) | A summary and grid of the data at a point in the graph |

## Models and layers

| Page | What it covers |
| --- | --- |
| [Classic ML models](classic_ml.md) | Naive Bayes, KNN, logistic regression, SVM |
| [Convolutional layers](cnn_layers.md) | Training image graphs as CNNs |
| [Conv3D](conv3d.md) | Convolution over volumes |
| [Cross Attention](cross_attention.md) | One sequence attending to another |
| [Linear Attention](linear_attention.md) | Self-attention with linear cost |
| [Regularization nodes](regularization.md) | Penalties on the model's parameters |
| [Learning-rate schedulers](lr_schedulers.md) | Changing the learning rate during training |
| [Metric learning](metric_learning.md) | Pairs and triplets |
| [Sentiment text classification](sentiment_text_classification.md) | The example text classification graphs |

## Training, testing and results

| Page | What it covers |
| --- | --- |
| [Testing a trained model](testing.md) | Running a test on the test split |
| [Generation previews](generation_previews.md) | Prompting a language model while it trains |
| [Language model generation](lm_generation.md) | Using a trained language model in the Engine |
| [Fine-tuning language models](fine_tuning_language_models.md) | Fine-tuning on prompt/answer examples |
| [Exporting a model](export_import.md) | Export Model and the formats it writes |
| [Server Connection](server-connection.md) | Reserving a server node and training on it |

## Scripting

| Page | What it covers |
| --- | --- |
| [Notebooks in the Script Editor](notebooks.md) | Jupyter notebooks as cells |
| [Writing a module in Python](../extending/python-modules.md) | pycyxwiz modules, training from a script, gradient check |
| [Python nodes](../extending/python-nodes.md) | Turning a Python module into a node for the canvas |
