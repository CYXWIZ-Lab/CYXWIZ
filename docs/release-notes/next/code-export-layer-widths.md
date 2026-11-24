kind: fixed
area: graph-editor
page: usage/studio_toolbar.md

Code Gen writes runnable PyTorch for more graphs: a recurrent layer or a LayerNorm set to `AUTO` takes its input width from the graph (an Embedding behind a Dropout, a Data Input's shape), a LayerNorm whose width the data sets becomes a `LazyLayerNorm`, a simple RNN keeps its activation and direction, and a graph with no model layers, or an RL graph that does not compile, stops at its first line with the reason instead of failing further on.
