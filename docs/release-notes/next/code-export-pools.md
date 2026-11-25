kind: fixed
area: graph-editor
page: usage/studio_toolbar.md

Code Gen's PyTorch export writes Avg Pool 2D, Adaptive Avg Pool and Global Avg / Max Pool (on a feature map, or over time on a sequence), and sizes Max Pool as the Engine does (the stride is the window when it is not set; `same` padding resolved). A layer the PyTorch export cannot write yet now stops the script at its first line, naming the node, instead of being left out of the model.
