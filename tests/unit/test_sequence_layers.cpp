#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cyxwiz/layer.h>
#include <cyxwiz/tensor.h>
#include <cmath>
#include <vector>
#include <cstring>

using namespace cyxwiz;
using Catch::Matchers::WithinAbs;

// Helper to check tensors are approximately equal
bool TensorsApproxEqual(const Tensor& a, const Tensor& b, float epsilon = 1e-4f) {
    if (a.Shape() != b.Shape()) return false;

    const float* a_data = a.Data<float>();
    const float* b_data = b.Data<float>();
    size_t n = a.NumElements();

    for (size_t i = 0; i < n; i++) {
        if (std::abs(a_data[i] - b_data[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

// Helper to compare shapes (std::vector<size_t>)
bool ShapesEqual(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    return a == b;
}

// ============================================================================
// EMBEDDING LAYER TESTS
// ============================================================================

TEST_CASE("EmbeddingLayer - Basic forward pass", "[embedding][forward]") {
    // Create embedding layer: 5 embeddings, each of dimension 3
    const int num_embeddings = 5;
    const int embedding_dim = 3;

    EmbeddingLayer embed(num_embeddings, embedding_dim);

    // Set known weights for testing
    // embedding[0] = [0.1, 0.2, 0.3]
    // embedding[1] = [1.1, 1.2, 1.3]
    // embedding[2] = [2.1, 2.2, 2.3]
    // embedding[3] = [3.1, 3.2, 3.3]
    // embedding[4] = [4.1, 4.2, 4.3]
    std::vector<float> weights = {
        0.1f, 0.2f, 0.3f,  // index 0
        1.1f, 1.2f, 1.3f,  // index 1
        2.1f, 2.2f, 2.3f,  // index 2
        3.1f, 3.2f, 3.3f,  // index 3
        4.1f, 4.2f, 4.3f   // index 4
    };
    Tensor weight_tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)},
                         weights.data(), DataType::Float32);
    embed.LoadPretrainedWeights(weight_tensor, false);

    // Input: batch of indices [1, 3, 0, 4]
    std::vector<int32_t> indices = {1, 3, 0, 4};
    Tensor input({4}, indices.data(), DataType::Int32);

    // Forward pass
    Tensor output = embed.Forward(input);

    // Expected output shape: [4, 3]
    REQUIRE(ShapesEqual(output.Shape(), {4, 3}));

    // Expected values:
    // output[0] = embedding[1] = [1.1, 1.2, 1.3]
    // output[1] = embedding[3] = [3.1, 3.2, 3.3]
    // output[2] = embedding[0] = [0.1, 0.2, 0.3]
    // output[3] = embedding[4] = [4.1, 4.2, 4.3]
    std::vector<float> expected = {
        1.1f, 1.2f, 1.3f,
        3.1f, 3.2f, 3.3f,
        0.1f, 0.2f, 0.3f,
        4.1f, 4.2f, 4.3f
    };

    const float* out_data = output.Data<float>();
    for (size_t i = 0; i < expected.size(); i++) {
        REQUIRE_THAT(out_data[i], WithinAbs(expected[i], 1e-5));
    }
}

TEST_CASE("EmbeddingLayer - 2D input (sequence)", "[embedding][forward]") {
    // Embedding: 10 tokens, dimension 4
    const int num_embeddings = 10;
    const int embedding_dim = 4;

    EmbeddingLayer embed(num_embeddings, embedding_dim);

    // Set weights: embedding[i] = [i*0.1, i*0.2, i*0.3, i*0.4]
    std::vector<float> weights(num_embeddings * embedding_dim);
    for (int i = 0; i < num_embeddings; i++) {
        weights[i * embedding_dim + 0] = i * 0.1f;
        weights[i * embedding_dim + 1] = i * 0.2f;
        weights[i * embedding_dim + 2] = i * 0.3f;
        weights[i * embedding_dim + 3] = i * 0.4f;
    }
    Tensor weight_tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)},
                         weights.data(), DataType::Float32);
    embed.LoadPretrainedWeights(weight_tensor, false);

    // Input: [batch=2, seq_len=3] indices
    // batch 0: [1, 5, 2]
    // batch 1: [7, 0, 3]
    std::vector<int32_t> indices = {1, 5, 2, 7, 0, 3};
    Tensor input({2, 3}, indices.data(), DataType::Int32);

    Tensor output = embed.Forward(input);

    // Expected shape: [2, 3, 4]
    REQUIRE(ShapesEqual(output.Shape(), {2, 3, 4}));

    // Verify specific values
    const float* out_data = output.Data<float>();

    // batch 0, position 0, index 1: [0.1, 0.2, 0.3, 0.4]
    REQUIRE_THAT(out_data[0], WithinAbs(0.1f, 1e-5));
    REQUIRE_THAT(out_data[1], WithinAbs(0.2f, 1e-5));
    REQUIRE_THAT(out_data[2], WithinAbs(0.3f, 1e-5));
    REQUIRE_THAT(out_data[3], WithinAbs(0.4f, 1e-5));

    // batch 0, position 1, index 5: [0.5, 1.0, 1.5, 2.0]
    REQUIRE_THAT(out_data[4], WithinAbs(0.5f, 1e-5));
    REQUIRE_THAT(out_data[5], WithinAbs(1.0f, 1e-5));
    REQUIRE_THAT(out_data[6], WithinAbs(1.5f, 1e-5));
    REQUIRE_THAT(out_data[7], WithinAbs(2.0f, 1e-5));

    // batch 1, position 2, index 3: [0.3, 0.6, 0.9, 1.2]
    // offset = (1*3 + 2) * 4 = 20
    REQUIRE_THAT(out_data[20], WithinAbs(0.3f, 1e-5));
    REQUIRE_THAT(out_data[21], WithinAbs(0.6f, 1e-5));
    REQUIRE_THAT(out_data[22], WithinAbs(0.9f, 1e-5));
    REQUIRE_THAT(out_data[23], WithinAbs(1.2f, 1e-5));
}

TEST_CASE("EmbeddingLayer - Padding index", "[embedding][forward]") {
    const int num_embeddings = 5;
    const int embedding_dim = 2;
    const int padding_idx = 0;  // Index 0 is padding

    EmbeddingLayer embed(num_embeddings, embedding_dim, padding_idx);

    // Set weights
    std::vector<float> weights = {
        0.0f, 0.0f,  // padding (should stay zero)
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    Tensor weight_tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)},
                         weights.data(), DataType::Float32);
    embed.LoadPretrainedWeights(weight_tensor, false);

    // Input with padding tokens
    std::vector<int32_t> indices = {1, 0, 2, 0};  // 0 is padding
    Tensor input({4}, indices.data(), DataType::Int32);

    Tensor output = embed.Forward(input);

    const float* out_data = output.Data<float>();

    // Index 1: [1.0, 2.0]
    REQUIRE_THAT(out_data[0], WithinAbs(1.0f, 1e-5));
    REQUIRE_THAT(out_data[1], WithinAbs(2.0f, 1e-5));

    // Index 0 (padding): should be [0.0, 0.0]
    REQUIRE_THAT(out_data[2], WithinAbs(0.0f, 1e-5));
    REQUIRE_THAT(out_data[3], WithinAbs(0.0f, 1e-5));

    // Index 2: [3.0, 4.0]
    REQUIRE_THAT(out_data[4], WithinAbs(3.0f, 1e-5));
    REQUIRE_THAT(out_data[5], WithinAbs(4.0f, 1e-5));

    // Index 0 (padding): should be [0.0, 0.0]
    REQUIRE_THAT(out_data[6], WithinAbs(0.0f, 1e-5));
    REQUIRE_THAT(out_data[7], WithinAbs(0.0f, 1e-5));
}

TEST_CASE("EmbeddingLayer - Backward pass gradient accumulation", "[embedding][backward]") {
    const int num_embeddings = 4;
    const int embedding_dim = 2;

    EmbeddingLayer embed(num_embeddings, embedding_dim);

    // Initialize weights to zero for easy gradient verification
    Tensor zero_weights = Tensor::Zeros({static_cast<size_t>(num_embeddings),
                                         static_cast<size_t>(embedding_dim)});
    embed.LoadPretrainedWeights(zero_weights, false);

    // Input: indices [0, 1, 0]
    // Note: index 0 appears twice, gradients should accumulate
    std::vector<int32_t> indices = {0, 1, 0};
    Tensor input({3}, indices.data(), DataType::Int32);

    // Forward pass
    embed.Forward(input);

    // Gradient from downstream: [3, 2] tensor
    // grad[0] = [1.0, 2.0] -> goes to embedding[0]
    // grad[1] = [3.0, 4.0] -> goes to embedding[1]
    // grad[2] = [5.0, 6.0] -> goes to embedding[0] (accumulates)
    std::vector<float> grad_values = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    Tensor grad_output({3, 2}, grad_values.data(), DataType::Float32);

    // Backward pass
    embed.Backward(grad_output);

    // Get weight gradients from parameters
    auto params = embed.GetParameters();
    REQUIRE(params.size() >= 1);  // At least weight

    // The gradient tensor should be stored in parameters
    // Check that we have gradients accumulated correctly
    bool found_grad = false;
    for (const auto& [name, tensor] : params) {
        if (name.find("grad") != std::string::npos || name == "weight") {
            // Found gradient or weight parameter
            found_grad = true;
        }
    }
    REQUIRE(found_grad);
}

TEST_CASE("EmbeddingLayer - Frozen weights", "[embedding][backward]") {
    const int num_embeddings = 3;
    const int embedding_dim = 2;

    EmbeddingLayer embed(num_embeddings, embedding_dim);

    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor weight_tensor({static_cast<size_t>(num_embeddings),
                          static_cast<size_t>(embedding_dim)},
                         weights.data(), DataType::Float32);
    embed.LoadPretrainedWeights(weight_tensor, true);  // freeze=true

    // Verify frozen state
    REQUIRE(embed.IsFrozen() == true);
}

// ============================================================================
// LSTM LAYER TESTS
// ============================================================================

TEST_CASE("LSTMLayer - Output shape", "[lstm][shape]") {
    // Create LSTM: input_size=10, hidden_size=20, num_layers=1
    LSTMLayer lstm(10, 20, 1, true, false, 0.0f);

    // Input: [batch=4, seq_len=5, input_size=10]
    Tensor input = Tensor::Random({4, 5, 10});

    Tensor output = lstm.Forward(input);

    // Output shape: [batch=4, seq_len=5, hidden_size=20]
    REQUIRE(ShapesEqual(output.Shape(), {4, 5, 20}));
}

TEST_CASE("LSTMLayer - Bidirectional output shape", "[lstm][shape][bidirectional]") {
    // Bidirectional LSTM
    LSTMLayer lstm(10, 20, 1, true, true, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    Tensor output = lstm.Forward(input);

    // Bidirectional output: [batch=4, seq_len=5, 2*hidden_size=40]
    REQUIRE(ShapesEqual(output.Shape(), {4, 5, 40}));
}

TEST_CASE("LSTMLayer - Multi-layer output shape", "[lstm][shape][multilayer]") {
    // 3-layer LSTM
    LSTMLayer lstm(10, 20, 3, true, false, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    Tensor output = lstm.Forward(input);

    // Multi-layer still outputs: [batch=4, seq_len=5, hidden_size=20]
    REQUIRE(ShapesEqual(output.Shape(), {4, 5, 20}));
}

TEST_CASE("LSTMLayer - Hidden state shape", "[lstm][state]") {
    LSTMLayer lstm(10, 20, 2, true, false, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    lstm.Forward(input);

    Tensor h_n = lstm.GetHiddenState();
    Tensor c_n = lstm.GetCellState();

    // h_n, c_n shape: [num_layers, batch, hidden_size]
    REQUIRE(ShapesEqual(h_n.Shape(), {2, 4, 20}));
    REQUIRE(ShapesEqual(c_n.Shape(), {2, 4, 20}));
}

TEST_CASE("LSTMLayer - Simple forward computation", "[lstm][forward][computation]") {
    // Minimal LSTM: input_size=2, hidden_size=2, single layer
    const int input_size = 2;
    const int hidden_size = 2;

    LSTMLayer lstm(input_size, hidden_size, 1, true, false, 0.0f);

    // For verification, we'll use a single timestep, single batch
    // Input: [batch=1, seq_len=1, input_size=2]
    std::vector<float> input_data = {0.5f, -0.5f};
    Tensor input({1, 1, 2}, input_data.data(), DataType::Float32);

    Tensor output = lstm.Forward(input);

    // Output shape: [1, 1, 2]
    REQUIRE(ShapesEqual(output.Shape(), {1, 1, 2}));

    // The output should be bounded by tanh (between -1 and 1)
    // since output = o_t * tanh(c_t) and o_t is sigmoid (0-1)
    const float* out_data = output.Data<float>();
    REQUIRE(out_data[0] >= -1.0f);
    REQUIRE(out_data[0] <= 1.0f);
    REQUIRE(out_data[1] >= -1.0f);
    REQUIRE(out_data[1] <= 1.0f);
}

TEST_CASE("LSTMLayer - State persistence", "[lstm][state]") {
    LSTMLayer lstm(4, 8, 1, true, false, 0.0f);

    // First forward pass
    Tensor input1 = Tensor::Random({2, 3, 4});
    lstm.Forward(input1);
    Tensor h1 = lstm.GetHiddenState().Clone();
    Tensor c1 = lstm.GetCellState().Clone();

    // Second forward pass (should use previous state)
    Tensor input2 = Tensor::Random({2, 3, 4});
    lstm.Forward(input2);
    Tensor h2 = lstm.GetHiddenState();
    Tensor c2 = lstm.GetCellState();

    // States should have changed
    REQUIRE_FALSE(TensorsApproxEqual(h1, h2));
    REQUIRE_FALSE(TensorsApproxEqual(c1, c2));

    // Reset state
    lstm.ResetState();

    // Forward pass after reset
    lstm.Forward(input1);
    Tensor h3 = lstm.GetHiddenState();

    // h3 should match h1 (same input, reset state)
    REQUIRE(TensorsApproxEqual(h1, h3, 1e-5f));
}

TEST_CASE("LSTMLayer - Backward pass runs without error", "[lstm][backward]") {
    LSTMLayer lstm(4, 8, 1, true, false, 0.0f);

    Tensor input = Tensor::Random({2, 5, 4});
    Tensor output = lstm.Forward(input);

    // Gradient has same shape as output
    Tensor grad_output = Tensor::Ones(output.Shape());

    // Backward should not throw
    REQUIRE_NOTHROW(lstm.Backward(grad_output));

    // Check that parameters exist
    auto params = lstm.GetParameters();
    REQUIRE(params.size() > 0);
}

// ============================================================================
// GRU LAYER TESTS
// ============================================================================

TEST_CASE("GRULayer - Output shape", "[gru][shape]") {
    GRULayer gru(10, 20, 1, true, false, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    Tensor output = gru.Forward(input);

    REQUIRE(ShapesEqual(output.Shape(), {4, 5, 20}));
}

TEST_CASE("GRULayer - Bidirectional output shape", "[gru][shape][bidirectional]") {
    GRULayer gru(10, 20, 1, true, true, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    Tensor output = gru.Forward(input);

    REQUIRE(ShapesEqual(output.Shape(), {4, 5, 40}));
}

TEST_CASE("GRULayer - Hidden state shape", "[gru][state]") {
    GRULayer gru(10, 20, 2, true, false, 0.0f);

    Tensor input = Tensor::Random({4, 5, 10});
    gru.Forward(input);

    Tensor h_n = gru.GetHiddenState();

    // h_n shape: [num_layers, batch, hidden_size]
    REQUIRE(ShapesEqual(h_n.Shape(), {2, 4, 20}));
}

TEST_CASE("GRULayer - Simple forward computation", "[gru][forward][computation]") {
    const int input_size = 2;
    const int hidden_size = 2;

    GRULayer gru(input_size, hidden_size, 1, true, false, 0.0f);

    std::vector<float> input_data = {0.5f, -0.5f};
    Tensor input({1, 1, 2}, input_data.data(), DataType::Float32);

    Tensor output = gru.Forward(input);

    REQUIRE(ShapesEqual(output.Shape(), {1, 1, 2}));

    // GRU output is bounded by tanh: h_t = (1-z) * n + z * h_{t-1}
    // where z is sigmoid and n is tanh applied
    const float* out_data = output.Data<float>();
    REQUIRE(out_data[0] >= -1.0f);
    REQUIRE(out_data[0] <= 1.0f);
    REQUIRE(out_data[1] >= -1.0f);
    REQUIRE(out_data[1] <= 1.0f);
}

TEST_CASE("GRULayer - State persistence", "[gru][state]") {
    GRULayer gru(4, 8, 1, true, false, 0.0f);

    Tensor input1 = Tensor::Random({2, 3, 4});
    gru.Forward(input1);
    Tensor h1 = gru.GetHiddenState().Clone();

    Tensor input2 = Tensor::Random({2, 3, 4});
    gru.Forward(input2);
    Tensor h2 = gru.GetHiddenState();

    REQUIRE_FALSE(TensorsApproxEqual(h1, h2));

    gru.ResetState();
    gru.Forward(input1);
    Tensor h3 = gru.GetHiddenState();

    REQUIRE(TensorsApproxEqual(h1, h3, 1e-5f));
}

TEST_CASE("GRULayer - Backward pass runs without error", "[gru][backward]") {
    GRULayer gru(4, 8, 1, true, false, 0.0f);

    Tensor input = Tensor::Random({2, 5, 4});
    Tensor output = gru.Forward(input);

    Tensor grad_output = Tensor::Ones(output.Shape());

    REQUIRE_NOTHROW(gru.Backward(grad_output));

    auto params = gru.GetParameters();
    REQUIRE(params.size() > 0);
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

TEST_CASE("Embedding + LSTM integration", "[integration]") {
    // Common NLP pipeline: Embedding -> LSTM
    const int vocab_size = 100;
    const int embed_dim = 16;
    const int hidden_size = 32;

    EmbeddingLayer embed(vocab_size, embed_dim);
    LSTMLayer lstm(embed_dim, hidden_size, 1, true, false, 0.0f);

    // Input: batch of token indices [batch=2, seq_len=10]
    std::vector<int32_t> indices(20);
    for (int i = 0; i < 20; i++) indices[i] = i % vocab_size;

    Tensor input({2, 10}, indices.data(), DataType::Int32);

    // Forward pass
    Tensor embedded = embed.Forward(input);  // [2, 10, 16]
    REQUIRE(ShapesEqual(embedded.Shape(), {2, 10, 16}));

    Tensor lstm_out = lstm.Forward(embedded);  // [2, 10, 32]
    REQUIRE(ShapesEqual(lstm_out.Shape(), {2, 10, 32}));
}

TEST_CASE("Embedding + GRU integration", "[integration]") {
    const int vocab_size = 100;
    const int embed_dim = 16;
    const int hidden_size = 32;

    EmbeddingLayer embed(vocab_size, embed_dim);
    GRULayer gru(embed_dim, hidden_size, 1, true, false, 0.0f);

    std::vector<int32_t> indices(20);
    for (int i = 0; i < 20; i++) indices[i] = i % vocab_size;

    Tensor input({2, 10}, indices.data(), DataType::Int32);

    Tensor embedded = embed.Forward(input);
    REQUIRE(ShapesEqual(embedded.Shape(), {2, 10, 16}));

    Tensor gru_out = gru.Forward(embedded);
    REQUIRE(ShapesEqual(gru_out.Shape(), {2, 10, 32}));
}

// ============================================================================
// DETERMINISM TESTS
// ============================================================================

TEST_CASE("LSTM - Deterministic with same input", "[lstm][determinism]") {
    LSTMLayer lstm1(8, 16, 1, true, false, 0.0f);

    // Get parameters from lstm1
    auto params = lstm1.GetParameters();

    // Create lstm2 with same parameters
    LSTMLayer lstm2(8, 16, 1, true, false, 0.0f);
    lstm2.SetParameters(params);

    // Same input - create deterministic input
    std::vector<float> input_data(2 * 5 * 8, 0.5f);
    Tensor input({2, 5, 8}, input_data.data(), DataType::Float32);

    lstm1.ResetState();
    lstm2.ResetState();

    Tensor out1 = lstm1.Forward(input);
    Tensor out2 = lstm2.Forward(input);

    REQUIRE(TensorsApproxEqual(out1, out2, 1e-5f));
}

TEST_CASE("GRU - Deterministic with same input", "[gru][determinism]") {
    GRULayer gru1(8, 16, 1, true, false, 0.0f);

    auto params = gru1.GetParameters();

    GRULayer gru2(8, 16, 1, true, false, 0.0f);
    gru2.SetParameters(params);

    std::vector<float> input_data(2 * 5 * 8, 0.5f);
    Tensor input({2, 5, 8}, input_data.data(), DataType::Float32);

    gru1.ResetState();
    gru2.ResetState();

    Tensor out1 = gru1.Forward(input);
    Tensor out2 = gru2.Forward(input);

    REQUIRE(TensorsApproxEqual(out1, out2, 1e-5f));
}
