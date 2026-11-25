kind: fixed
area: training
page: usage/patterns.md

A graph with a Flatten after a transformer block and a Layer Norm (the Transformer Decoder pattern: decoder, Layer Norm, Flatten, Dense) trains: the Dense after the Flatten reads the whole flattened sequence. It was built for one token's width, and training stopped at the first batch with "Input features mismatch".
