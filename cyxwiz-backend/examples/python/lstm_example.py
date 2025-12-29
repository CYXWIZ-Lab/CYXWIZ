"""
LSTM Layer Example

Demonstrates how to use the LSTM layer for sequence modeling tasks
like time series prediction, language modeling, and sequence classification.
"""

import pycyxwiz as cx
import numpy as np

# Initialize CyxWiz
cx.initialize()

def main():
    print("=" * 60)
    print("CyxWiz LSTM Layer Example")
    print("=" * 60)

    # Example 1: Basic LSTM
    print("\n1. Basic LSTM")
    print("-" * 40)

    input_size = 64    # Features per time step
    hidden_size = 128  # LSTM hidden units
    batch_size = 16
    seq_length = 20

    # Create single-layer LSTM
    lstm = cx.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True  # Input shape: [batch, seq, features]
    )

    print(f"   Input size: {lstm.input_size}")
    print(f"   Hidden size: {lstm.hidden_size}")
    print(f"   Num layers: {lstm.num_layers}")
    print(f"   Batch first: {lstm.batch_first}")

    # Create random input sequence
    x = cx.Tensor.random([batch_size, seq_length, input_size])

    # Forward pass
    output = lstm.forward(x)
    print(f"   Input shape: {x.shape()}")
    print(f"   Output shape: {output.shape()}")  # [batch, seq, hidden]

    # Get final hidden and cell states
    h_n = lstm.get_hidden_state()
    c_n = lstm.get_cell_state()
    print(f"   Final hidden state shape: {h_n.shape()}")
    print(f"   Final cell state shape: {c_n.shape()}")

    # Example 2: Multi-layer LSTM
    print("\n2. Multi-layer LSTM (Stacked)")
    print("-" * 40)

    stacked_lstm = cx.LSTM(
        input_size=64,
        hidden_size=128,
        num_layers=3,      # 3 stacked LSTM layers
        batch_first=True,
        dropout=0.1        # Dropout between layers
    )

    print(f"   Layers: {stacked_lstm.num_layers}")

    x = cx.Tensor.random([8, 30, 64])
    output = stacked_lstm.forward(x)
    print(f"   Output shape: {output.shape()}")

    # Example 3: Bidirectional LSTM
    print("\n3. Bidirectional LSTM")
    print("-" * 40)

    bi_lstm = cx.LSTM(
        input_size=64,
        hidden_size=128,
        num_layers=2,
        batch_first=True,
        bidirectional=True
    )

    print(f"   Bidirectional: {bi_lstm.bidirectional}")
    print(f"   Num directions: {bi_lstm.num_directions}")

    x = cx.Tensor.random([8, 25, 64])
    output = bi_lstm.forward(x)
    # Bidirectional concatenates forward and backward outputs
    print(f"   Output shape: {output.shape()}")  # [8, 25, 256] (128 * 2)

    # Example 4: Sequence-to-Sequence (Many-to-Many)
    print("\n4. Sequence-to-Sequence Model")
    print("-" * 40)

    # Encoder
    encoder = cx.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
    # Decoder
    decoder = cx.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

    # Encoder input
    src_seq = cx.Tensor.random([4, 15, 64])

    # Encode source sequence
    _ = encoder.forward(src_seq)
    encoder_h = encoder.get_hidden_state()
    encoder_c = encoder.get_cell_state()
    print(f"   Encoder hidden shape: {encoder_h.shape()}")

    # Initialize decoder with encoder state
    decoder.set_hidden_state(encoder_h)
    decoder.set_cell_state(encoder_c)

    # Decode target sequence
    tgt_seq = cx.Tensor.random([4, 20, 64])
    decoder_output = decoder.forward(tgt_seq)
    print(f"   Decoder output shape: {decoder_output.shape()}")

    # Example 5: Language Model Architecture
    print("\n5. Language Model Architecture")
    print("-" * 40)

    vocab_size = 10000
    embed_dim = 256
    hidden_dim = 512

    # Layers
    embedding = cx.Embedding(vocab_size, embed_dim)
    lstm_lm = cx.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
    output_proj = cx.Dense(hidden_dim, vocab_size)

    print(f"   Vocab size: {vocab_size}")
    print(f"   Embedding dim: {embed_dim}")
    print(f"   Hidden dim: {hidden_dim}")

    # Simulated token indices
    tokens = cx.Tensor.random([8, 50], cx.DataType.Int32)

    # Forward pass
    embeds = embedding.forward(tokens)
    print(f"   After embedding: {embeds.shape()}")

    lstm_out = lstm_lm.forward(embeds)
    print(f"   After LSTM: {lstm_out.shape()}")

    # For each position, project to vocabulary (in practice, use softmax)
    # logits = output_proj.forward(lstm_out)
    print("   (Would project to vocabulary logits)")

    # Example 6: Resetting State for New Sequences
    print("\n6. State Management")
    print("-" * 40)

    lstm = cx.LSTM(input_size=32, hidden_size=64, batch_first=True)

    # Process first sequence
    seq1 = cx.Tensor.random([4, 10, 32])
    _ = lstm.forward(seq1)
    h1 = lstm.get_hidden_state()
    print(f"   After seq1, hidden state exists: {h1.shape()}")

    # Reset state for new, independent sequences
    lstm.reset_state()
    print("   State reset for new batch")

    # Process new sequence
    seq2 = cx.Tensor.random([4, 15, 32])
    _ = lstm.forward(seq2)
    print(f"   Processed new sequence independently")

    # Example 7: Stateful LSTM (for long sequences)
    print("\n7. Stateful LSTM for Long Sequences")
    print("-" * 40)

    lstm = cx.LSTM(input_size=32, hidden_size=64, batch_first=True)

    # Process a very long sequence in chunks
    chunk_size = 100
    num_chunks = 5
    batch_size = 4
    input_size = 32

    print(f"   Processing {num_chunks} chunks of size {chunk_size}")

    for i in range(num_chunks):
        chunk = cx.Tensor.random([batch_size, chunk_size, input_size])
        output = lstm.forward(chunk)
        # State is automatically maintained between chunks
        print(f"   Chunk {i+1}: output shape {output.shape()}")

    print("\n" + "=" * 60)
    print("LSTM layer example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
