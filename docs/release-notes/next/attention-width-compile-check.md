kind: fixed
area: graph-editor
page: usage/patterns.md

Two shape mistakes that used to compile and then fail when training started are now errors when the graph compiles, naming both shapes: a Multi-Head Attention, Transformer Encoder, Transformer Decoder or Positional Encoding whose `embed_dim` or `d_model` is not the width of the sequence it reads, and an Add or other merge whose inputs differ when one of them comes through an Identity (as a block's skip connection does). The Residual Block and Multi-Head Attention Block patterns now say they take an input as wide as their `filters` or `d_model` setting.
