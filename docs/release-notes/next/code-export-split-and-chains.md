kind: fixed
area: graph-editor
page: usage/branched_graphs.md

Code Gen's PyTorch export writes a Split as `torch.split` and gives each of its outputs to the layer it is linked to; before, both branches after a Split read the whole tensor and the script trained another network. The TensorFlow and Keras exports, which write a straight chain of layers only, now refuse a graph that branches (a Split, a merge or a layer two layers read), naming the node, instead of writing it as a chain. New example: `examples/cyxgraph/branching/blobs_split_heads_classifier.cyxgraph`.
