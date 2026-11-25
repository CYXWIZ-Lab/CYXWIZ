## Which attention node

| Node | Attends | Cost per head | Use it for |
| --- | --- | --- | --- |
| [Multi-Head Attention](../attention/multi-head-attention.md) | a sequence over itself, with softmax | `length^2` memory and `length^2 x head_dim` work | the usual self-attention |
| [Cross Attention](../attention/cross-attention.md) | one sequence (Query) over another (Key / Value) | `query length x key length` | a decoder reading an encoder, one branch consulting another |
| [Linear Attention](../attention/linear-attention.md) | a sequence over itself, kernel feature map instead of softmax | `head_dim^2` memory and `length x head_dim^2` work (causal: as Multi-Head) | long sequences, where `length` is larger than `head_dim` |

Linear Attention is a different, cheaper model, not an approximation of
softmax attention: train it from scratch, and do not expect it to reproduce
a Multi-Head Attention model's numbers.

[Transformer Encoder](../attention/transformer-encoder.md) and
[Transformer Decoder](../attention/transformer-decoder.md) are whole blocks
(attention, feed-forward, normalisation) in one node;
[Positional Encoding](../attention/positional-encoding.md) gives the
sequence its order before the first of them.

Which code exports write these nodes is in
[Model formats](../../../concepts/model-formats.md#what-each-framework-writes):
every framework refuses Cross Attention and Linear Attention and says so.
