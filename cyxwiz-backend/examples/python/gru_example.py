"""
GRU Layer Example

Demonstrates how to use the GRU (Gated Recurrent Unit) layer for
sequence modeling. GRU is similar to LSTM but with fewer parameters
and often trains faster while achieving comparable results.
"""

import pycyxwiz as cx
import numpy as np

# Initialize CyxWiz
cx.initialize()

def main():
    print("=" * 60)
    print("CyxWiz GRU Layer Example")
    print("=" * 60)

    # Example 1: Basic GRU
    print("\n1. Basic GRU")
    print("-" * 40)

    input_size = 64
    hidden_size = 128
    batch_size = 16
    seq_length = 20

    gru = cx.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True
    )

    print(f"   Input size: {gru.input_size}")
    print(f"   Hidden size: {gru.hidden_size}")
    print(f"   Num layers: {gru.num_layers}")

    x = cx.Tensor.random([batch_size, seq_length, input_size])
    output = gru.forward(x)

    print(f"   Input shape: {x.shape()}")
    print(f"   Output shape: {output.shape()}")

    # GRU only has hidden state (no cell state like LSTM)
    h_n = gru.get_hidden_state()
    print(f"   Final hidden state shape: {h_n.shape()}")

    # Example 2: Multi-layer GRU
    print("\n2. Multi-layer GRU")
    print("-" * 40)

    stacked_gru = cx.GRU(
        input_size=64,
        hidden_size=128,
        num_layers=3,
        batch_first=True,
        dropout=0.1
    )

    print(f"   Layers: {stacked_gru.num_layers}")

    x = cx.Tensor.random([8, 30, 64])
    output = stacked_gru.forward(x)
    print(f"   Output shape: {output.shape()}")

    # Example 3: Bidirectional GRU
    print("\n3. Bidirectional GRU")
    print("-" * 40)

    bi_gru = cx.GRU(
        input_size=64,
        hidden_size=128,
        num_layers=2,
        batch_first=True,
        bidirectional=True
    )

    print(f"   Bidirectional: {bi_gru.bidirectional}")

    x = cx.Tensor.random([8, 25, 64])
    output = bi_gru.forward(x)
    print(f"   Output shape: {output.shape()}")  # [8, 25, 256]

    # Example 4: GRU vs LSTM Comparison
    print("\n4. GRU vs LSTM Comparison")
    print("-" * 40)

    # Same configuration for both
    input_size = 64
    hidden_size = 128
    num_layers = 2

    lstm = cx.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    gru = cx.GRU(input_size, hidden_size, num_layers, batch_first=True)

    lstm_params = lstm.get_parameters()
    gru_params = gru.get_parameters()

    print(f"   LSTM parameters: {len(lstm_params)} tensors")
    print(f"   GRU parameters: {len(gru_params)} tensors")
    print("   (GRU has ~25% fewer parameters than LSTM)")

    # Example 5: Sentiment Classification with GRU
    print("\n5. Sentiment Classification")
    print("-" * 40)

    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    num_classes = 3  # positive/neutral/negative

    # Build model layers
    embedding = cx.Embedding(vocab_size, embed_dim, padding_idx=0)
    gru = cx.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
    classifier = cx.Dense(hidden_dim, num_classes)

    print(f"   Vocabulary: {vocab_size}")
    print(f"   Embedding dim: {embed_dim}")
    print(f"   GRU hidden: {hidden_dim}")
    print(f"   Classes: {num_classes}")

    # Simulated tokenized reviews (batch of 16, max length 100)
    reviews = cx.Tensor.random([16, 100], cx.DataType.Int32)

    # Forward pass
    x = embedding.forward(reviews)
    print(f"   After embedding: {x.shape()}")

    x = gru.forward(x)
    print(f"   After GRU: {x.shape()}")

    # Take last hidden state for classification
    h_n = gru.get_hidden_state()
    print(f"   Final hidden: {h_n.shape()}")

    # Would apply classifier to hidden state
    # logits = classifier.forward(h_n[-1])  # Last layer's hidden
    print("   (Would classify using final hidden state)")

    # Example 6: Time Series Forecasting
    print("\n6. Time Series Forecasting")
    print("-" * 40)

    # Multi-variate time series: 5 features, predict 1 target
    input_features = 5
    hidden_size = 64
    output_size = 1  # Predict single value

    encoder_gru = cx.GRU(input_features, hidden_size, batch_first=True)
    output_layer = cx.Dense(hidden_size, output_size)

    # Historical data: batch of 32 samples, 50 time steps, 5 features
    historical_data = cx.Tensor.random([32, 50, 5])

    # Encode historical sequence
    encoded = encoder_gru.forward(historical_data)
    print(f"   Encoded sequence shape: {encoded.shape()}")

    # Use last time step's hidden state for prediction
    h_n = encoder_gru.get_hidden_state()
    print(f"   Hidden state shape: {h_n.shape()}")

    # prediction = output_layer.forward(h_n)
    print("   (Would predict next value from hidden state)")

    # Example 7: Custom Initial Hidden State
    print("\n7. Custom Initial State")
    print("-" * 40)

    gru = cx.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
    batch_size = 8

    # Create custom initial hidden state
    # Shape: [num_layers, batch_size, hidden_size]
    h0 = cx.Tensor.zeros([2, batch_size, 64])
    gru.set_hidden_state(h0)
    print(f"   Set custom initial state: {h0.shape()}")

    # Process sequence with custom initial state
    x = cx.Tensor.random([batch_size, 20, 32])
    output = gru.forward(x)
    print(f"   Output shape: {output.shape()}")

    # Reset to zero state
    gru.reset_state()
    print("   Reset state to zeros")

    print("\n" + "=" * 60)
    print("GRU layer example complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
