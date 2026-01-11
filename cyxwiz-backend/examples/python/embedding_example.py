"""
Embedding Layer Example

Demonstrates how to use the Embedding layer for NLP tasks like
word embeddings, token embeddings, and positional encodings.
"""

import pycyxwiz as cx
import numpy as np

# Initialize CyxWiz
cx.initialize()

def main():
    print("=" * 60)
    print("CyxWiz Embedding Layer Example")
    print("=" * 60)

    # Example 1: Basic word embeddings
    print("\n1. Basic Word Embeddings")
    print("-" * 40)

    vocab_size = 10000  # Size of vocabulary
    embedding_dim = 128  # Dimension of each embedding vector

    # Create embedding layer
    embedding = cx.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=0  # Index 0 is padding, always returns zeros
    )

    print(f"   Vocabulary size: {embedding.num_embeddings}")
    print(f"   Embedding dimension: {embedding.embedding_dim}")
    print(f"   Padding index: {embedding.padding_idx}")

    # Create input indices (simulating tokenized text)
    # Shape: [batch_size, sequence_length]
    batch_size = 4
    seq_length = 10
    input_indices = cx.Tensor.random([batch_size, seq_length], cx.DataType.Int32)

    # Forward pass
    embeddings = embedding.forward(input_indices)
    print(f"   Input shape: [{batch_size}, {seq_length}]")
    print(f"   Output shape: {embeddings.shape()}")  # [batch_size, seq_length, embedding_dim]

    # Example 2: Loading pretrained embeddings
    print("\n2. Loading Pretrained Embeddings (e.g., GloVe)")
    print("-" * 40)

    # Simulate pretrained GloVe-style embeddings
    pretrained_weights = cx.Tensor.random([vocab_size, embedding_dim])

    # Create embedding layer and load pretrained weights
    pretrained_embedding = cx.Embedding(vocab_size, embedding_dim)
    pretrained_embedding.load_pretrained_weights(pretrained_weights, freeze=True)

    print(f"   Pretrained weights loaded")
    print(f"   Frozen: {pretrained_embedding.frozen}")  # True = won't update during training

    # You can unfreeze later for fine-tuning
    pretrained_embedding.frozen = False
    print(f"   Unfrozen for fine-tuning: {not pretrained_embedding.frozen}")

    # Example 3: Position embeddings (like in Transformers)
    print("\n3. Positional Embeddings")
    print("-" * 40)

    max_seq_length = 512
    d_model = 256

    # Create position embedding layer
    pos_embedding = cx.Embedding(max_seq_length, d_model)

    # Create position indices [0, 1, 2, ..., seq_length-1]
    seq_len = 20
    positions = np.arange(seq_len).reshape(1, seq_len).astype(np.int32)
    pos_tensor = cx.Tensor.from_numpy(positions)

    pos_encodings = pos_embedding.forward(pos_tensor)
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Model dimension: {d_model}")
    print(f"   Position encodings shape: {pos_encodings.shape()}")

    # Example 4: Get/Set individual embeddings
    print("\n4. Accessing Individual Embeddings")
    print("-" * 40)

    small_embedding = cx.Embedding(100, 32)

    # Get embedding for token index 5
    token_5_embedding = small_embedding.get_embedding(5)
    print(f"   Token 5 embedding shape: {token_5_embedding.shape()}")

    # Set a custom embedding for token index 10
    custom_vec = cx.Tensor.ones([32])
    small_embedding.set_embedding(10, custom_vec)
    print(f"   Set custom embedding for token 10")

    # Example 5: Training with embeddings
    print("\n5. Training Example - Sentiment Classification")
    print("-" * 40)

    # Small vocabulary for demo
    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 32
    num_classes = 2  # positive/negative

    # Create layers
    embed = cx.Embedding(vocab_size, embed_dim)
    fc1 = cx.Dense(embed_dim, hidden_dim)
    fc2 = cx.Dense(hidden_dim, num_classes)

    # Simulated input: batch of 8 sequences, length 15
    input_tokens = cx.Tensor.random([8, 15], cx.DataType.Int32)

    # Forward pass (simplified - averaging embeddings)
    x = embed.forward(input_tokens)  # [8, 15, 64]
    print(f"   Embeddings shape: {x.shape()}")

    # In a real scenario, you would:
    # 1. Pass through LSTM/GRU/Transformer
    # 2. Take final hidden state or pooled output
    # 3. Pass through classifier

    print("\n" + "=" * 60)
    print("Embedding layer example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
