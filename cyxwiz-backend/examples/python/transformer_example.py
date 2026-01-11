"""
Transformer Layer Example

Demonstrates how to use MultiHeadAttention, TransformerEncoderLayer,
and TransformerDecoderLayer for building transformer architectures.
"""

import pycyxwiz as cx
import numpy as np

# Initialize CyxWiz
cx.initialize()

def main():
    print("=" * 60)
    print("CyxWiz Transformer Layers Example")
    print("=" * 60)

    # Example 1: Multi-Head Self-Attention
    print("\n1. Multi-Head Self-Attention")
    print("-" * 40)

    embed_dim = 256
    num_heads = 8
    batch_size = 4
    seq_length = 20

    attention = cx.MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        use_bias=True
    )

    print(f"   Embedding dim: {attention.embed_dim}")
    print(f"   Num heads: {attention.num_heads}")
    print(f"   Head dim: {attention.head_dim}")  # embed_dim / num_heads

    # Self-attention: same input for Q, K, V
    x = cx.Tensor.random([batch_size, seq_length, embed_dim])
    output = attention.forward(x)

    print(f"   Input shape: {x.shape()}")
    print(f"   Output shape: {output.shape()}")

    # Get attention weights for visualization
    attn_weights = attention.get_attention_weights()
    print(f"   Attention weights shape: {attn_weights.shape()}")

    # Example 2: Cross-Attention (Query from one source, K/V from another)
    print("\n2. Cross-Attention")
    print("-" * 40)

    cross_attn = cx.MultiHeadAttention(embed_dim=256, num_heads=8)

    # Query from decoder (target sequence)
    query = cx.Tensor.random([4, 15, 256])  # target: 15 positions

    # Key and Value from encoder (source sequence)
    key = cx.Tensor.random([4, 30, 256])    # source: 30 positions
    value = key  # Usually K and V come from same source

    # Cross-attention
    output = cross_attn.forward_qkv(query, key, value)
    print(f"   Query shape: {query.shape()}")
    print(f"   Key/Value shape: {key.shape()}")
    print(f"   Output shape: {output.shape()}")

    # Example 3: Transformer Encoder Layer
    print("\n3. Transformer Encoder Layer")
    print("-" * 40)

    d_model = 512
    nhead = 8
    dim_feedforward = 2048

    encoder_layer = cx.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        norm_first=False  # Post-LN (original transformer)
    )

    print(f"   d_model: {d_model}")
    print(f"   nhead: {nhead}")
    print(f"   feedforward dim: {dim_feedforward}")

    src = cx.Tensor.random([8, 50, d_model])
    encoded = encoder_layer.forward(src)
    print(f"   Input shape: {src.shape()}")
    print(f"   Output shape: {encoded.shape()}")

    # Example 4: Transformer Encoder Stack
    print("\n4. Stacked Transformer Encoder")
    print("-" * 40)

    num_layers = 6
    d_model = 256
    nhead = 4

    # Create multiple encoder layers
    encoder_layers = [
        cx.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1)
        for _ in range(num_layers)
    ]

    print(f"   Number of layers: {num_layers}")

    # Pass input through all layers
    x = cx.Tensor.random([4, 30, d_model])
    for i, layer in enumerate(encoder_layers):
        x = layer.forward(x)
        if i == 0:
            print(f"   After layer 0: {x.shape()}")

    print(f"   Final output: {x.shape()}")

    # Example 5: Transformer Decoder Layer
    print("\n5. Transformer Decoder Layer")
    print("-" * 40)

    decoder_layer = cx.TransformerDecoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1
    )

    # Target sequence (what we're generating)
    tgt = cx.Tensor.random([4, 20, 512])

    # Memory from encoder
    memory = cx.Tensor.random([4, 30, 512])

    # Generate causal mask for autoregressive decoding
    causal_mask = cx.TransformerDecoderLayer.generate_causal_mask(20)
    print(f"   Causal mask shape: {causal_mask.shape()}")

    # Decode with cross-attention to encoder memory
    decoded = decoder_layer.forward_with_memory(tgt, memory, causal_mask, None)
    print(f"   Target shape: {tgt.shape()}")
    print(f"   Memory shape: {memory.shape()}")
    print(f"   Decoded shape: {decoded.shape()}")

    # Example 6: Full Transformer Model Architecture
    print("\n6. Full Transformer (Encoder-Decoder)")
    print("-" * 40)

    # Hyperparameters
    vocab_size = 10000
    d_model = 256
    nhead = 4
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_ff = 1024

    # Embeddings
    src_embedding = cx.Embedding(vocab_size, d_model)
    tgt_embedding = cx.Embedding(vocab_size, d_model)

    # Encoder stack
    encoders = [
        cx.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.1)
        for _ in range(num_encoder_layers)
    ]

    # Decoder stack
    decoders = [
        cx.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout=0.1)
        for _ in range(num_decoder_layers)
    ]

    # Output projection
    output_proj = cx.Dense(d_model, vocab_size)

    print(f"   Vocab size: {vocab_size}")
    print(f"   d_model: {d_model}")
    print(f"   Encoder layers: {num_encoder_layers}")
    print(f"   Decoder layers: {num_decoder_layers}")

    # Simulated translation: source tokens -> target tokens
    src_tokens = cx.Tensor.random([8, 50], cx.DataType.Int32)
    tgt_tokens = cx.Tensor.random([8, 40], cx.DataType.Int32)

    # Encode
    src = src_embedding.forward(src_tokens)
    print(f"   Source embedded: {src.shape()}")

    for enc in encoders:
        src = enc.forward(src)
    memory = src
    print(f"   Encoder output: {memory.shape()}")

    # Decode
    tgt = tgt_embedding.forward(tgt_tokens)
    causal_mask = cx.TransformerDecoderLayer.generate_causal_mask(40)

    for dec in decoders:
        tgt = dec.forward_with_memory(tgt, memory, causal_mask, None)
    print(f"   Decoder output: {tgt.shape()}")

    # Output projection (would apply softmax for probabilities)
    # logits = output_proj.forward(tgt)
    print("   (Would project to vocabulary logits)")

    # Example 7: BERT-style Encoder Only
    print("\n7. BERT-style Encoder Only")
    print("-" * 40)

    bert_layers = 12
    d_model = 768
    nhead = 12

    # BERT uses Pre-LN (norm_first=True in modern implementations)
    bert_encoders = [
        cx.TransformerEncoderLayer(d_model, nhead, dim_feedforward=3072,
                                   dropout=0.1, norm_first=True)
        for _ in range(bert_layers)
    ]

    embedding = cx.Embedding(30000, d_model)  # BERT vocab
    cls_head = cx.Dense(d_model, 2)  # Binary classification

    print(f"   BERT layers: {bert_layers}")
    print(f"   d_model: {d_model}")
    print(f"   nhead: {nhead}")

    # Simulated BERT input
    tokens = cx.Tensor.random([16, 128], cx.DataType.Int32)

    x = embedding.forward(tokens)
    for encoder in bert_encoders:
        x = encoder.forward(x)

    print(f"   BERT output: {x.shape()}")
    # Would take [CLS] token (x[:, 0, :]) for classification
    print("   (Would use [CLS] token for classification)")

    # Example 8: GPT-style Decoder Only
    print("\n8. GPT-style Decoder Only")
    print("-" * 40)

    gpt_layers = 6
    d_model = 512
    nhead = 8

    # GPT uses decoder layers with causal masking
    gpt_decoders = [
        cx.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048,
                                   dropout=0.1, norm_first=True)
        for _ in range(gpt_layers)
    ]

    embedding = cx.Embedding(50000, d_model)
    lm_head = cx.Dense(d_model, 50000)

    print(f"   GPT layers: {gpt_layers}")

    # GPT input (autoregressive)
    tokens = cx.Tensor.random([8, 256], cx.DataType.Int32)
    causal_mask = cx.TransformerDecoderLayer.generate_causal_mask(256)

    x = embedding.forward(tokens)
    for decoder in gpt_decoders:
        # GPT uses self-attention only (no cross-attention)
        x = decoder.forward(x)  # Uses causal masking internally

    print(f"   GPT output: {x.shape()}")
    print("   (Would project to vocabulary for next token prediction)")

    # Example 9: Training Mode Toggle
    print("\n9. Training vs Inference Mode")
    print("-" * 40)

    encoder = cx.TransformerEncoderLayer(256, 4, 512, dropout=0.3)
    decoder = cx.TransformerDecoderLayer(256, 4, 512, dropout=0.3)

    # Training mode (dropout active)
    encoder.set_training(True)
    decoder.set_training(True)
    print("   Training mode: dropout active")

    x = cx.Tensor.random([4, 20, 256])
    train_output = encoder.forward(x)

    # Inference mode (dropout disabled)
    encoder.set_training(False)
    decoder.set_training(False)
    print("   Inference mode: dropout disabled")

    infer_output = encoder.forward(x)

    print("\n" + "=" * 60)
    print("Transformer layers example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
