# CyxWiz Engine user guide

How to use the Engine, one page per feature. Each page describes what the
current build does; limits are stated on the page.

To write your own nodes in Python or C++, see
[Extending CyxWiz](../sdk/index.md).

## Getting around

| Page | What it covers |
| --- | --- |
| [Menus](../main-window/menus.md) | File to Help, the project and your account at the top right |
| [The Studio bar](../main-window/toolbar.md) | The bar at the top of the canvas |
| [The Studio Debugger](../user-guide/training/studio-debugger.md) | Local Debug, layer capture and the layer views |
| [Layouts and docking](../main-window/layouts-and-docking.md) | Arranging panels, moving the big windows out of the main window |
| [Preferences](preferences.md) | How the Engine starts (start page or the last project), recent projects kept, asking before closing |
| [Theme Editor](theme_editor.md) | Every colour the Engine draws, the accent, and saving your own theme |
| [Asset Browser](asset_browser.md) | The project's folders and files, previews, new files |
| [Properties](../main-window/properties.md) | The Properties panel for the selected node |
| [Console and logs](console_logs.md) | Logs, the Python REPL and commands |

## Building a graph

| Page | What it covers |
| --- | --- |
| [Find and add nodes](../user-guide/graph-editor/find-and-add-nodes.md) | Every way to add a node: the Nodes panel, search, the right-click menu |
| [Nodes panel](../main-window/nodes-panel.md) | The Nodes panel in detail |
| [The canvas](../user-guide/graph-editor/canvas.md) | Pan, zoom, select, undo |
| [Connect nodes](../user-guide/graph-editor/connect-nodes.md) | Pins, what connects, refused links |
| [Set node properties](../user-guide/graph-editor/node-properties.md) | The Properties panel and node dialogs |
| [Compile and run](../user-guide/graph-editor/compile-and-run.md) | Compile, the compile report, run modes, what Train does first |
| [Node dialogs](../main-window/node-dialogs.md) | Configuring a node in its dialog |
| [Patterns](patterns.md) | Ready-made full graphs and blocks, what data each expects, the ones not usable yet |
| [Branched graphs](branched_graphs.md) | Split and merge inside a model |
| [Groups, frames and names](../user-guide/graph-editor/groups-frames-and-names.md) | Grouping and naming nodes, frames, copy and paste, renaming |
| [Your own nodes](extension_nodes.md) | Python node bundles and C++ plugins in the Engine: install, approve, use |
| [Graph simulation](graph_simulation.md) | Run Sim, the Simulation menu and the Simulation Monitor |
| [Environments for reinforcement learning](rl_environments.md) | Plugin environments (MuJoCo tasks), their settings, and `pycyxwiz.rl` for Gymnasium and Stable-Baselines3 |
| [Training a policy (PPO)](rl_training.md) | The RL graph (Environment, layers, Policy / Value Head, PPO), its settings, Train and the RL Training Dashboard |
| [The MuJoCo Viewport](mujoco_viewport.md) | The 3D view: run a model, cameras (follow a body, the model's cameras), View options, live numbers, screenshots and recordings, the Stress panel, Inspect (bodies, joints, controls, keyframes, physics), the timeline and plots |
| [MuJoCo models](mujoco_models.md) | The Environment Library; downloads shared by every project and pinned to a version; a project names its models so another PC downloads the same one; Copy into the project; Import from URL |
| [Train, play and export a policy](rl_walkthrough.md) | Step by step: CartPole, InvertedPendulum and HalfCheetah from Studio, playing and saving the policy, export, and what a real robot needs |

## Data

| Page | What it covers |
| --- | --- |
| [Where the tools get their data](../user-guide/data-studio/overview.md) | The Data Input node as the one way in |
| [Data Studio](../user-guide/data-studio/index.md) | Bring in, explore, prepare, validate and write a dataset: every Data Studio page |
| [Loaded data](../concepts/loaded-data.md) | What the Engine holds in memory and when it lets go |
| [Data bigger than memory](../concepts/data-bigger-than-memory.md) | Tables on disk and the data budget |
| [Table Viewer](../main-window/table-viewer.md) | Sorting, filtering and formatting a table |
| [SQL steps and the SQL Workbench](../user-guide/data-studio/sql-workbench.md) | SQL steps and guided steps |
| [Scaling and transforms](data_transforms.md) | Scaling and shape changes as graph steps |
| [Image transforms](image_transforms.md) | Image nodes between the Data Input and the model |
| [Quality Analyzer](../user-guide/data-studio/quality-analyzer.md) | Leaving poor images out of training |
| [Splitting images and audio](../user-guide/data-studio/split-images-and-audio.md) | Train, validation and test parts of an image or audio dataset |
| [Text vectorizers](text_vectorizers.md) | TF-IDF and count vectorizers |
| [Word embeddings](word_embeddings.md) | word2vec, GloVe and fastText vectors |

## Exploring and analysing

| Page | What it covers |
| --- | --- |
| [Statistics](../user-guide/data-studio/statistics.md) | Hypothesis tests and regression |
| [Clustering](../user-guide/data-studio/clustering.md) | Explore > Groups and the clustering nodes |
| [Dimensionality reduction](../user-guide/data-studio/dimensionality-reduction.md) | Explore > Projection, PCA, t-SNE, UMAP |
| [Time series](../user-guide/data-studio/time-series.md) | Explore > Series and Forecast, and the time series nodes |
| [Signal processing](../user-guide/data-studio/signal-processing.md) | Explore > Signal: spectrum, spectrogram, filters, wavelets |
| [Text](../user-guide/data-studio/text.md) | Explore > Text: words, TF-IDF terms, VADER sentiment, tokens, word vectors and the word map |
| [Linear algebra](linear_algebra.md) | Tools > Linear Algebra (matrices typed, pasted or from tables; into the graph), the PCA, Truncated SVD, Whitening, Linear Solve and Linear Projection nodes |
| [Monitoring and diagnostics](monitoring.md) | Tools > Monitoring: the Monitor window (system, the Engine's memory over time, loaded data, frame times), Clear Caches, Free Unused Memory; Tools > Diagnostics: the frame-rate and Python command logs, the frame overlay |
| [Utilities](utilities.md) | Tools > Utilities: calculator, exact unit conversion, seeded random values and UUIDs, hashes of text and files, JSON check and tree, regex with multiline and replace |
| [Optimization and calculus](optimization_calculus.md) | Tools > Optimization and Calculus (optimizer paths, linear and quadratic programs, integrals, derivatives, curvature), the Portfolio Weights node |
| [Feature Importance](../user-guide/evaluation/feature-importance.md) | Which columns matter to a model |
| [Hyperparameter Search](../user-guide/training/hyperparameter-search.md) | Train the graph with different property values and keep the best |
| [Neural Architecture Search](../user-guide/training/neural-architecture-search.md) | Score the model, change its layers, search, train the best versions, put one on the canvas |
| [Analysis demo graph](analysis_demo_graph.md) | Analysis nodes working together on one dataset |
| [Plotting](../user-guide/evaluation/plotting.md) | Plots, the Plot node and Python plotting |
| [Dashboard node](../reference/nodes/categories/visualization.md) | A summary and grid of the data at a point in the graph |

## Models and layers

| Page | What it covers |
| --- | --- |
| [Classic ML models](classic_ml.md) | Naive Bayes, KNN, logistic regression, SVM |
| [Convolutional layers](cnn_layers.md) | Training image graphs as CNNs |
| [Starting from published weights](../user-guide/training/start-from-pretrained-weights.md) | Training from a PyTorch or safetensors weight file |
| [Image classifier: a ResNet from scratch](image_classifier_resnet.md) | The cats vs dogs ResNet example, from data to a test score |
| [Conv3D](conv3d.md) | Convolution over volumes |
| [Cross Attention](cross_attention.md) | One sequence attending to another |
| [Linear Attention](linear_attention.md) | Self-attention with linear cost |
| [Regularization nodes](regularization.md) | Penalties on the model's parameters |
| [Learning-rate schedulers](lr_schedulers.md) | Changing the learning rate during training |
| [Gradient clipping](gradient_clipping.md) | Limiting how far one step can move the weights (as PyTorch's clip_grad_norm_) |
| [Metric learning](../user-guide/training/metric-learning.md) | Pairs and triplets |
| [Sentiment text classification](sentiment_text_classification.md) | The example text classification graphs |

## Training, testing and results

| Page | What it covers |
| --- | --- |
| [Training](../user-guide/training/index.md) | Every training page |
| [Train a model and watch it](../user-guide/training/train-and-watch.md) | Train, pause, resume and stop; what the Training Dashboard shows while it runs |
| [Checkpoints and early stopping](../user-guide/training/checkpoints-and-early-stopping.md) | Where checkpoints go, best and last, early stopping, resume |
| [Losses and optimizers](../user-guide/training/losses-and-optimizers.md) | The loss and optimizer nodes, Optimizer Settings, class imbalance |
| [Evaluation](../user-guide/evaluation/index.md) | Every testing, results and explanation page |
| [The Training Dashboard](../main-window/training-dashboard.md) | The run's state, where it ran and on what data, how it ended, the active model and held-out test, Run details, the cards |
| [Testing a trained model](../user-guide/evaluation/test-a-trained-model.md) | Running a test on the test split; Results > Test: errors, threshold, calibration, a decision kept with the model |
| [Seeds and reproducibility](../concepts/seeds-and-reproducibility.md) | What each seed draws; the same draws on every operating system and device |
| [The Model window](../main-window/model-window.md) | Tools > Model: summary (parameters, multiply-adds, memory), learning-rate range test, Grad-CAM / saliency / Integrated Gradients, training health, projector |
| [Generation previews](../user-guide/training/generation-previews.md) | Prompting a language model while it trains |
| [Language model generation](../user-guide/evaluation/language-model-generation.md) | Using a trained language model in the Engine |
| [Fine-tuning language models](../user-guide/training/fine-tune-language-models.md) | Fine-tuning on prompt/answer examples |
| [Exporting a model](export_import.md) | Export Model and the formats it writes |
| [Server Connection](server-connection.md) | Reserving a server node and training on it |

## Scripting

| Page | What it covers |
| --- | --- |
| [Notebooks in the Script Editor](notebooks.md) | Jupyter notebooks as cells |
| [Writing a module in Python](../sdk/guides/python-modules.md) | pycyxwiz modules, training from a script, gradient check |
| [Python nodes](../sdk/guides/python-nodes.md) | Turning a Python module into a node for the canvas |
