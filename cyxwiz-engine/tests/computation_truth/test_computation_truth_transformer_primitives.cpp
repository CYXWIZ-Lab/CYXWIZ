#include <cyxwiz/sequential.h>
#include <cyxwiz/layers/attention.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/tensor.h>
#include "core/language_model_training.h"
#include "core/language_model_generation.h"
#include "core/transformer_primitive_contracts.h"

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
#include <torch/torch.h>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual, float expected, float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream ss;
        ss << message << " expected=" << expected << " actual=" << actual
           << " tolerance=" << tolerance;
        Check(false, ss.str());
    }
}

void CheckShape(const cyxwiz::Tensor& tensor,
                const std::vector<size_t>& expected,
                const std::string& label) {
    Check(tensor.Shape() == expected, label + " shape mismatch");
}

void CheckCandidatesNear(
    const std::vector<cyxwiz::NextTokenCandidate>& actual,
    const std::vector<int64_t>& expected_ids,
    const std::vector<float>& expected_probabilities,
    float tolerance,
    const std::string& label) {
    Check(actual.size() == expected_ids.size(), label + " candidate count");
    Check(expected_ids.size() == expected_probabilities.size(),
          label + " expected fixture size");
    for (size_t i = 0; i < actual.size(); ++i) {
        Check(actual[i].token_id == expected_ids[i], label + " token order");
        CheckNear(actual[i].probability,
                  expected_probabilities[i],
                  tolerance,
                  label + " probability");
    }
}

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
std::vector<float> TensorToVector(const torch::Tensor& tensor) {
    torch::Tensor contiguous =
        tensor.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
    const float* data = contiguous.data_ptr<float>();
    return std::vector<float>(data, data + contiguous.numel());
}

std::vector<int64_t> TensorToInt64Vector(const torch::Tensor& tensor) {
    torch::Tensor contiguous =
        tensor.detach().to(torch::kCPU).contiguous().to(torch::kInt64);
    const int64_t* data = contiguous.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + contiguous.numel());
}
#endif

void TestEmbeddingForwardAndGradientParity() {
    cyxwiz::EmbeddingModule embedding(5, 3, -1);

    const float weight_values[] = {
        0.1f, 0.2f, 0.3f,
        1.0f, 1.1f, 1.2f,
        -0.5f, -0.4f, -0.3f,
        2.0f, 2.1f, 2.2f,
        4.0f, 4.1f, 4.2f,
    };
    cyxwiz::Tensor weights({5, 3}, weight_values, cyxwiz::DataType::Float32);
    embedding.SetParameters({{"weight", weights}});

    const int64_t token_values[] = {1, 2, 1, 4};
    cyxwiz::Tensor tokens({2, 2}, token_values, cyxwiz::DataType::Int64);
    cyxwiz::Tensor output = embedding.Forward(tokens);
    CheckShape(output, {2, 2, 3}, "Embedding output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_weight = torch::from_blob(
        const_cast<float*>(weight_values), {5, 3}, torch::kFloat32).clone();
    auto torch_embedding = torch::nn::Embedding(
        torch::nn::EmbeddingOptions(5, 3));
    {
        torch::NoGradGuard no_grad;
        torch_embedding->weight.copy_(torch_weight);
    }
    auto torch_tokens = torch::tensor({{1, 2}, {1, 4}}, torch::kInt64);
    auto torch_output = torch_embedding->forward(torch_tokens);
    const std::vector<float> expected_output =
        TensorToVector(torch_output);
#else
    const float expected_output[] = {
        1.0f, 1.1f, 1.2f,
        -0.5f, -0.4f, -0.3f,
        1.0f, 1.1f, 1.2f,
        4.0f, 4.1f, 4.2f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-6f,
                  "Embedding forward matches PyTorch nn.Embedding lookup");
    }

    const float grad_values[] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f,
    };
    cyxwiz::Tensor grad_output({2, 2, 3}, grad_values,
                               cyxwiz::DataType::Float32);
    (void)embedding.Backward(grad_output);

    const auto grads = embedding.GetGradients();
    auto it = grads.find("weight");
    Check(it != grads.end(), "Embedding weight gradient exists");
    CheckShape(it->second, {5, 3}, "Embedding weight gradient");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_grad_output = torch::from_blob(
        const_cast<float*>(grad_values), {2, 2, 3}, torch::kFloat32).clone();
    torch_output.backward(torch_grad_output);
    const std::vector<float> expected_grad_weight =
        TensorToVector(torch_embedding->weight.grad());
#else
    const float expected_grad_weight[] = {
        0.0f, 0.0f, 0.0f,
        0.8f, 1.0f, 1.2f,
        0.4f, 0.5f, 0.6f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.1f, 1.2f,
    };
#endif
    const float* grad = it->second.Data<float>();
    for (size_t i = 0; i < it->second.NumElements(); ++i) {
        CheckNear(grad[i], expected_grad_weight[i], 1e-6f,
                  "Embedding backward matches PyTorch sparse accumulation semantics");
    }
}

void TestPositionalEncodingParity() {
    cyxwiz::PositionalEncodingModule positional(4, 3);
    cyxwiz::Tensor input = cyxwiz::Tensor::Zeros({1, 2, 4});
    cyxwiz::Tensor output = positional.Forward(input);
    CheckShape(output, {1, 2, 4}, "PositionalEncoding output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto position = torch::arange(0, 2, torch::kFloat32).unsqueeze(1);
    auto div_term = torch::exp(
        torch::tensor({0.0f, 2.0f}) *
        static_cast<float>(-std::log(10000.0) / 4.0));
    auto torch_pe = torch::zeros({1, 2, 4}, torch::kFloat32);
    torch_pe.index_put_({0, torch::indexing::Slice(), 0},
                        torch::sin(position.index({torch::indexing::Slice(), 0}) *
                                   div_term.index({0})));
    torch_pe.index_put_({0, torch::indexing::Slice(), 1},
                        torch::cos(position.index({torch::indexing::Slice(), 0}) *
                                   div_term.index({0})));
    torch_pe.index_put_({0, torch::indexing::Slice(), 2},
                        torch::sin(position.index({torch::indexing::Slice(), 0}) *
                                   div_term.index({1})));
    torch_pe.index_put_({0, torch::indexing::Slice(), 3},
                        torch::cos(position.index({torch::indexing::Slice(), 0}) *
                                   div_term.index({1})));
    const std::vector<float> expected = TensorToVector(torch_pe);
#else
    const float expected[] = {
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.8414709848f,
        0.5403023059f,
        0.0099998333f,
        0.9999500004f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected[i], 1e-6f,
                  "PositionalEncoding matches PyTorch sinusoidal fixture");
    }

    cyxwiz::Tensor grad = positional.Backward(output);
    CheckShape(grad, {1, 2, 4}, "PositionalEncoding backward");
    const float* grad_data = grad.Data<float>();
    for (size_t i = 0; i < grad.NumElements(); ++i) {
        CheckNear(grad_data[i], out[i], 1e-6f,
                  "PositionalEncoding backward is identity");
    }
}

void TestScaledDotProductAttentionParity() {
    const size_t batch_size = 1;
    const size_t num_heads = 1;
    const size_t sequence_length = 2;
    const size_t head_dim = 2;

    const std::vector<float> query = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const std::vector<float> key = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const std::vector<float> value = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };

    const auto output = cyxwiz::ScaledDotProductAttentionMultiHeadCpu(
        query,
        key,
        value,
        {},
        batch_size,
        num_heads,
        sequence_length,
        head_dim);

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_query = torch::from_blob(
        const_cast<float*>(query.data()), {1, 1, 2, 2}, torch::kFloat32).clone();
    auto torch_key = torch::from_blob(
        const_cast<float*>(key.data()), {1, 1, 2, 2}, torch::kFloat32).clone();
    auto torch_value = torch::from_blob(
        const_cast<float*>(value.data()), {1, 1, 2, 2}, torch::kFloat32).clone();
    auto scores = torch::matmul(torch_query, torch_key.transpose(-2, -1)) /
                  std::sqrt(static_cast<float>(head_dim));
    auto weights = torch::softmax(scores, -1);
    const std::vector<float> expected = TensorToVector(
        torch::matmul(weights, torch_value));
#else
    const float expected[] = {
        1.6604769f, 2.6604769f,
        2.3395231f, 3.3395231f,
    };
#endif
    Check(output.size() == 4, "Scaled dot-product attention output size");
    for (size_t i = 0; i < output.size(); ++i) {
        CheckNear(output[i], expected[i], 1e-5f,
                  "Scaled dot-product attention matches PyTorch");
    }

    const std::vector<float> causal_mask =
        cyxwiz::BuildCausalAttentionMask(sequence_length);
    const auto masked_output = cyxwiz::ScaledDotProductAttentionMultiHeadCpu(
        query,
        key,
        value,
        causal_mask,
        batch_size,
        num_heads,
        sequence_length,
        head_dim);

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_mask = torch::from_blob(
        const_cast<float*>(causal_mask.data()),
        {1, 1, 2, 2},
        torch::kFloat32).clone();
    auto masked_scores = scores + torch_mask;
    auto masked_weights = torch::softmax(masked_scores, -1);
    const std::vector<float> expected_masked = TensorToVector(
        torch::matmul(masked_weights, torch_value));
#else
    const float expected_masked[] = {
        1.0f, 2.0f,
        2.3395231f, 3.3395231f,
    };
#endif
    Check(masked_output.size() == 4,
          "Masked scaled dot-product attention output size");
    for (size_t i = 0; i < masked_output.size(); ++i) {
        CheckNear(masked_output[i], expected_masked[i], 1e-5f,
                  "Causal masked scaled dot-product attention matches PyTorch");
    }
}

void TestMultiHeadAttentionForwardParity() {
    const int embed_dim = 4;
    const int num_heads = 2;
    const std::vector<float> input_values = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
    };
    const std::vector<float> W_q = {
        0.10f, 0.20f, -0.10f, 0.00f,
        0.00f, 0.15f, 0.25f, -0.05f,
        -0.20f, 0.05f, 0.30f, 0.10f,
        0.05f, -0.10f, 0.20f, 0.25f,
    };
    const std::vector<float> W_k = {
        0.05f, -0.15f, 0.10f, 0.20f,
        0.20f, 0.00f, -0.10f, 0.05f,
        0.10f, 0.25f, 0.05f, -0.20f,
        -0.05f, 0.10f, 0.15f, 0.30f,
    };
    const std::vector<float> W_v = {
        0.30f, -0.10f, 0.05f, 0.00f,
        -0.20f, 0.25f, 0.10f, 0.15f,
        0.05f, 0.00f, 0.20f, -0.10f,
        0.10f, 0.30f, -0.15f, 0.05f,
    };
    const std::vector<float> W_o = {
        0.20f, 0.10f, -0.05f, 0.30f,
        -0.10f, 0.25f, 0.15f, 0.05f,
        0.05f, -0.20f, 0.35f, 0.10f,
        0.30f, 0.00f, -0.10f, 0.20f,
    };
    const std::vector<float> b_q = {0.01f, -0.02f, 0.03f, 0.04f};
    const std::vector<float> b_k = {-0.03f, 0.02f, 0.01f, -0.01f};
    const std::vector<float> b_v = {0.05f, -0.04f, 0.02f, 0.03f};
    const std::vector<float> b_o = {0.01f, 0.02f, -0.03f, 0.04f};

    cyxwiz::MultiHeadAttentionLayer attention(embed_dim, num_heads, 0.0f, true);
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({4, 4}, W_q.data(), cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({4, 4}, W_k.data(), cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({4, 4}, W_v.data(), cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({4, 4}, W_o.data(), cyxwiz::DataType::Float32)},
        {"b_q", cyxwiz::Tensor({4}, b_q.data(), cyxwiz::DataType::Float32)},
        {"b_k", cyxwiz::Tensor({4}, b_k.data(), cyxwiz::DataType::Float32)},
        {"b_v", cyxwiz::Tensor({4}, b_v.data(), cyxwiz::DataType::Float32)},
        {"b_o", cyxwiz::Tensor({4}, b_o.data(), cyxwiz::DataType::Float32)},
    });

    const cyxwiz::Tensor input({1, 2, 4}, input_values.data(),
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = attention.Forward(input);
    CheckShape(output, {1, 2, 4}, "MultiHeadAttention output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_input = torch::from_blob(
        const_cast<float*>(input_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    auto torch_wq = torch::from_blob(
        const_cast<float*>(W_q.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wk = torch::from_blob(
        const_cast<float*>(W_k.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wv = torch::from_blob(
        const_cast<float*>(W_v.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wo = torch::from_blob(
        const_cast<float*>(W_o.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_bq = torch::from_blob(
        const_cast<float*>(b_q.data()), {4}, torch::kFloat32).clone();
    auto torch_bk = torch::from_blob(
        const_cast<float*>(b_k.data()), {4}, torch::kFloat32).clone();
    auto torch_bv = torch::from_blob(
        const_cast<float*>(b_v.data()), {4}, torch::kFloat32).clone();
    auto torch_bo = torch::from_blob(
        const_cast<float*>(b_o.data()), {4}, torch::kFloat32).clone();

    auto q = torch::linear(torch_input, torch_wq, torch_bq)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto k = torch::linear(torch_input, torch_wk, torch_bk)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto v = torch::linear(torch_input, torch_wv, torch_bv)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) /
                  std::sqrt(2.0f);
    auto torch_weights = torch::softmax(scores, -1);
    auto context = torch::matmul(torch_weights, v)
                       .transpose(1, 2)
                       .contiguous()
                       .view({1, 2, 4});
    const std::vector<float> expected_output =
        TensorToVector(torch::linear(context, torch_wo, torch_bo));
    const std::vector<float> expected_weights =
        TensorToVector(torch_weights.permute({2, 3, 0, 1}).contiguous());
#else
    const float expected_output[] = {
        0.0394057f, 0.0472653f, -0.0243494f, 0.0558073f,
        0.0401521f, 0.0470527f, -0.0239654f, 0.0566013f,
    };
    const float expected_weights[] = {
        0.4988863f, 0.5046093f, 0.5011137f, 0.4953907f,
        0.5058510f, 0.4919088f, 0.4941490f, 0.5080912f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-5f,
                  "MultiHeadAttention projected output matches PyTorch");
    }

    const cyxwiz::Tensor weights = attention.GetAttentionWeights();
    CheckShape(weights, {2, 2, 1, 2}, "MultiHeadAttention weights");
    const float* weight_data = weights.Data<float>();
    for (size_t i = 0; i < weights.NumElements(); ++i) {
        CheckNear(weight_data[i], expected_weights[i], 1e-5f,
                  "MultiHeadAttention per-head weights match PyTorch");
    }

    const std::vector<float> grad_output_values = {
        0.3f, -0.2f, 0.1f, 0.4f,
        -0.5f, 0.2f, 0.6f, -0.1f,
    };
    const cyxwiz::Tensor grad_output({1, 2, 4}, grad_output_values.data(),
                                     cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_input = attention.Backward(grad_output);
    CheckShape(grad_input, {1, 2, 4}, "MultiHeadAttention grad_input");
    const auto attention_grads = attention.GetParameters();

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto bwd_input = torch::from_blob(
        const_cast<float*>(input_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    bwd_input.set_requires_grad(true);
    auto bwd_wq = torch::from_blob(
        const_cast<float*>(W_q.data()), {4, 4}, torch::kFloat32).clone();
    bwd_wq.set_requires_grad(true);
    auto bwd_wk = torch::from_blob(
        const_cast<float*>(W_k.data()), {4, 4}, torch::kFloat32).clone();
    bwd_wk.set_requires_grad(true);
    auto bwd_wv = torch::from_blob(
        const_cast<float*>(W_v.data()), {4, 4}, torch::kFloat32).clone();
    bwd_wv.set_requires_grad(true);
    auto bwd_wo = torch::from_blob(
        const_cast<float*>(W_o.data()), {4, 4}, torch::kFloat32).clone();
    bwd_wo.set_requires_grad(true);
    auto bwd_bq = torch::from_blob(
        const_cast<float*>(b_q.data()), {4}, torch::kFloat32).clone();
    bwd_bq.set_requires_grad(true);
    auto bwd_bk = torch::from_blob(
        const_cast<float*>(b_k.data()), {4}, torch::kFloat32).clone();
    bwd_bk.set_requires_grad(true);
    auto bwd_bv = torch::from_blob(
        const_cast<float*>(b_v.data()), {4}, torch::kFloat32).clone();
    bwd_bv.set_requires_grad(true);
    auto bwd_bo = torch::from_blob(
        const_cast<float*>(b_o.data()), {4}, torch::kFloat32).clone();
    bwd_bo.set_requires_grad(true);
    auto bwd_q = torch::linear(bwd_input, bwd_wq, bwd_bq)
                     .view({1, 2, 2, 2})
                     .transpose(1, 2);
    auto bwd_k = torch::linear(bwd_input, bwd_wk, bwd_bk)
                     .view({1, 2, 2, 2})
                     .transpose(1, 2);
    auto bwd_v = torch::linear(bwd_input, bwd_wv, bwd_bv)
                     .view({1, 2, 2, 2})
                     .transpose(1, 2);
    auto bwd_scores = torch::matmul(bwd_q, bwd_k.transpose(-2, -1)) /
                      std::sqrt(2.0f);
    auto bwd_weights = torch::softmax(bwd_scores, -1);
    auto bwd_context = torch::matmul(bwd_weights, bwd_v)
                           .transpose(1, 2)
                           .contiguous()
                           .view({1, 2, 4});
    auto bwd_output = torch::linear(bwd_context, bwd_wo, bwd_bo);
    auto torch_grad_output = torch::from_blob(
        const_cast<float*>(grad_output_values.data()), {1, 2, 4},
        torch::kFloat32).clone();
    (bwd_output * torch_grad_output).sum().backward();
    const std::vector<float> expected_grad_input =
        TensorToVector(bwd_input.grad());
    const std::vector<float> expected_grad_W_q =
        TensorToVector(bwd_wq.grad());
    const std::vector<float> expected_grad_W_k =
        TensorToVector(bwd_wk.grad());
    const std::vector<float> expected_grad_W_v =
        TensorToVector(bwd_wv.grad());
    const std::vector<float> expected_grad_W_o =
        TensorToVector(bwd_wo.grad());
    const std::vector<float> expected_grad_b_q =
        TensorToVector(bwd_bq.grad());
    const std::vector<float> expected_grad_b_k =
        TensorToVector(bwd_bk.grad());
    const std::vector<float> expected_grad_b_v =
        TensorToVector(bwd_bv.grad());
    const std::vector<float> expected_grad_b_o =
        TensorToVector(bwd_bo.grad());
#else
    const float expected_grad_input[] = {
        0.0377994f, -0.0126506f, 0.0103623f, -0.0218529f,
        0.0380220f, -0.0143080f, 0.0124266f, -0.0212364f,
    };
    const float expected_grad_W_q[] = {
        0.000858584f, -0.000767066f, 0.001041620f, 0.002184736f,
        0.000303807f, -0.000271423f, 0.000368573f, 0.000773061f,
        0.000368031f, -0.000374570f, 0.000354954f, 0.000825334f,
        -0.000319423f, 0.000325098f, -0.000308073f, -0.000716328f,
    };
    const float expected_grad_W_k[] = {
        -0.000235253f, 0.000282303f, -0.000141152f, -0.000423455f,
        -0.000002303f, 0.000002763f, -0.000001382f, -0.000004145f,
        -0.000209541f, 0.000251449f, -0.000125724f, -0.000377173f,
        -0.000664802f, 0.000797762f, -0.000398881f, -0.001196643f,
    };
    const float expected_grad_W_v[] = {
        -0.004715217f, 0.017558262f, 0.020970875f, 0.020412616f,
        0.007671213f, -0.031605452f, -0.040197276f, -0.040591821f,
        -0.012477780f, 0.046473339f, 0.055513337f, 0.054040000f,
        -0.002703646f, 0.013044374f, 0.017977811f, 0.018933438f,
    };
    const float expected_grad_W_o[] = {
        -0.006233417f, -0.016222928f, -0.008527144f, -0.012705697f,
        0.000313413f, -0.000118403f, 0.000012702f, 0.000330216f,
        0.020014834f, 0.057461064f, 0.029771972f, 0.042571198f,
        0.008018119f, 0.024837602f, 0.012736734f, 0.017655127f,
    };
    const float expected_grad_b_q[] = {
        0.001880249f, 0.000665319f, 0.000479054f, -0.000415783f,
    };
    const float expected_grad_b_k[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    const float expected_grad_b_v[] = {
        0.085000023f, -0.159999996f, 0.225000024f, 0.069999993f,
    };
    const float expected_grad_b_o[] = {
        -0.199999988f, 0.0f, 0.700000048f, 0.300000012f,
    };
#endif

    const auto check_tensor_values =
        [](const cyxwiz::Tensor& tensor,
           const auto& expected,
           float tolerance,
           const std::string& message) {
            const float* data = tensor.Data<float>();
            for (size_t i = 0; i < tensor.NumElements(); ++i) {
                CheckNear(data[i], expected[i], tolerance, message);
            }
        };

    check_tensor_values(grad_input, expected_grad_input, 1e-5f,
                        "MultiHeadAttention grad_input matches PyTorch");
    check_tensor_values(attention_grads.at("grad_W_q"), expected_grad_W_q,
                        1e-5f, "MultiHeadAttention grad_W_q matches PyTorch");
    check_tensor_values(attention_grads.at("grad_W_k"), expected_grad_W_k,
                        1e-5f, "MultiHeadAttention grad_W_k matches PyTorch");
    check_tensor_values(attention_grads.at("grad_W_v"), expected_grad_W_v,
                        1e-5f, "MultiHeadAttention grad_W_v matches PyTorch");
    check_tensor_values(attention_grads.at("grad_W_o"), expected_grad_W_o,
                        1e-5f, "MultiHeadAttention grad_W_o matches PyTorch");
    check_tensor_values(attention_grads.at("grad_b_q"), expected_grad_b_q,
                        1e-5f, "MultiHeadAttention grad_b_q matches PyTorch");
    check_tensor_values(attention_grads.at("grad_b_k"), expected_grad_b_k,
                        1e-5f, "MultiHeadAttention grad_b_k matches PyTorch");
    check_tensor_values(attention_grads.at("grad_b_v"), expected_grad_b_v,
                        1e-5f, "MultiHeadAttention grad_b_v matches PyTorch");
    check_tensor_values(attention_grads.at("grad_b_o"), expected_grad_b_o,
                        1e-5f, "MultiHeadAttention grad_b_o matches PyTorch");

    const std::vector<float> mask_values = {
        0.0f, -1.0e9f,
        0.0f, 0.0f,
    };
    const cyxwiz::Tensor mask({2, 2}, mask_values.data(),
                              cyxwiz::DataType::Float32);
    const cyxwiz::Tensor masked_output =
        attention.Forward(input, input, input, &mask);
    CheckShape(masked_output, {1, 2, 4}, "Masked MultiHeadAttention output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_mask = torch::from_blob(
        const_cast<float*>(mask_values.data()), {1, 1, 2, 2},
        torch::kFloat32).clone();
    auto masked_weights = torch::softmax(scores + torch_mask, -1);
    auto masked_context = torch::matmul(masked_weights, v)
                              .transpose(1, 2)
                              .contiguous()
                              .view({1, 2, 4});
    const std::vector<float> expected_masked_output =
        TensorToVector(torch::linear(masked_context, torch_wo, torch_bo));
    const std::vector<float> expected_masked_weights =
        TensorToVector(masked_weights.permute({2, 3, 0, 1}).contiguous());
#else
    const float expected_masked_output[] = {
        0.0385000f, 0.0217500f, -0.0175000f, 0.0770000f,
        0.0401521f, 0.0470527f, -0.0239654f, 0.0566013f,
    };
    const float expected_masked_weights[] = {
        1.0f, 1.0f, 0.0f, 0.0f,
        0.5058510f, 0.4919088f, 0.4941490f, 0.5080912f,
    };
#endif
    const float* masked_out = masked_output.Data<float>();
    for (size_t i = 0; i < masked_output.NumElements(); ++i) {
        CheckNear(masked_out[i], expected_masked_output[i], 1e-5f,
                  "Masked MultiHeadAttention output matches PyTorch");
    }

    const cyxwiz::Tensor masked_weights_tensor =
        attention.GetAttentionWeights();
    CheckShape(masked_weights_tensor, {2, 2, 1, 2},
               "Masked MultiHeadAttention weights");
    const float* masked_weight_data = masked_weights_tensor.Data<float>();
    for (size_t i = 0; i < masked_weights_tensor.NumElements(); ++i) {
        CheckNear(masked_weight_data[i], expected_masked_weights[i], 1e-5f,
                  "Masked MultiHeadAttention weights match PyTorch");
    }
}

void TestMultiHeadAttentionCrossAttentionParity() {
    const int embed_dim = 4;
    const int num_heads = 2;
    const std::vector<float> query_values = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
    };
    const std::vector<float> key_value_values = {
        0.1f, 0.0f, -0.2f, 0.3f,
        0.5f, -0.4f, 0.2f, 0.1f,
        -0.1f, 0.3f, 0.6f, -0.5f,
    };
    const std::vector<float> W_q = {
        0.10f, 0.20f, -0.10f, 0.00f,
        0.00f, 0.15f, 0.25f, -0.05f,
        -0.20f, 0.05f, 0.30f, 0.10f,
        0.05f, -0.10f, 0.20f, 0.25f,
    };
    const std::vector<float> W_k = {
        0.05f, -0.15f, 0.10f, 0.20f,
        0.20f, 0.00f, -0.10f, 0.05f,
        0.10f, 0.25f, 0.05f, -0.20f,
        -0.05f, 0.10f, 0.15f, 0.30f,
    };
    const std::vector<float> W_v = {
        0.30f, -0.10f, 0.05f, 0.00f,
        -0.20f, 0.25f, 0.10f, 0.15f,
        0.05f, 0.00f, 0.20f, -0.10f,
        0.10f, 0.30f, -0.15f, 0.05f,
    };
    const std::vector<float> W_o = {
        0.20f, 0.10f, -0.05f, 0.30f,
        -0.10f, 0.25f, 0.15f, 0.05f,
        0.05f, -0.20f, 0.35f, 0.10f,
        0.30f, 0.00f, -0.10f, 0.20f,
    };
    const std::vector<float> b_q = {0.01f, -0.02f, 0.03f, 0.04f};
    const std::vector<float> b_k = {-0.03f, 0.02f, 0.01f, -0.01f};
    const std::vector<float> b_v = {0.05f, -0.04f, 0.02f, 0.03f};
    const std::vector<float> b_o = {0.01f, 0.02f, -0.03f, 0.04f};

    cyxwiz::MultiHeadAttentionLayer attention(embed_dim, num_heads, 0.0f, true);
    attention.SetParameters({
        {"W_q", cyxwiz::Tensor({4, 4}, W_q.data(), cyxwiz::DataType::Float32)},
        {"W_k", cyxwiz::Tensor({4, 4}, W_k.data(), cyxwiz::DataType::Float32)},
        {"W_v", cyxwiz::Tensor({4, 4}, W_v.data(), cyxwiz::DataType::Float32)},
        {"W_o", cyxwiz::Tensor({4, 4}, W_o.data(), cyxwiz::DataType::Float32)},
        {"b_q", cyxwiz::Tensor({4}, b_q.data(), cyxwiz::DataType::Float32)},
        {"b_k", cyxwiz::Tensor({4}, b_k.data(), cyxwiz::DataType::Float32)},
        {"b_v", cyxwiz::Tensor({4}, b_v.data(), cyxwiz::DataType::Float32)},
        {"b_o", cyxwiz::Tensor({4}, b_o.data(), cyxwiz::DataType::Float32)},
    });

    const cyxwiz::Tensor query({1, 2, 4}, query_values.data(),
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor key({1, 3, 4}, key_value_values.data(),
                             cyxwiz::DataType::Float32);
    const cyxwiz::Tensor value({1, 3, 4}, key_value_values.data(),
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = attention.Forward(query, key, value, nullptr);
    CheckShape(output, {1, 2, 4}, "Cross-attention output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_query = torch::from_blob(
        const_cast<float*>(query_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    auto torch_key = torch::from_blob(
        const_cast<float*>(key_value_values.data()), {1, 3, 4}, torch::kFloat32).clone();
    auto torch_value = torch::from_blob(
        const_cast<float*>(key_value_values.data()), {1, 3, 4}, torch::kFloat32).clone();
    auto torch_wq = torch::from_blob(
        const_cast<float*>(W_q.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wk = torch::from_blob(
        const_cast<float*>(W_k.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wv = torch::from_blob(
        const_cast<float*>(W_v.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wo = torch::from_blob(
        const_cast<float*>(W_o.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_bq = torch::from_blob(
        const_cast<float*>(b_q.data()), {4}, torch::kFloat32).clone();
    auto torch_bk = torch::from_blob(
        const_cast<float*>(b_k.data()), {4}, torch::kFloat32).clone();
    auto torch_bv = torch::from_blob(
        const_cast<float*>(b_v.data()), {4}, torch::kFloat32).clone();
    auto torch_bo = torch::from_blob(
        const_cast<float*>(b_o.data()), {4}, torch::kFloat32).clone();
    auto q = torch::linear(torch_query, torch_wq, torch_bq)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto k = torch::linear(torch_key, torch_wk, torch_bk)
                 .view({1, 3, 2, 2})
                 .transpose(1, 2);
    auto v = torch::linear(torch_value, torch_wv, torch_bv)
                 .view({1, 3, 2, 2})
                 .transpose(1, 2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(2.0f);
    auto weights = torch::softmax(scores, -1);
    auto context = torch::matmul(weights, v)
                       .transpose(1, 2)
                       .contiguous()
                       .view({1, 2, 4});
    const std::vector<float> expected_output =
        TensorToVector(torch::linear(context, torch_wo, torch_bo));
    const std::vector<float> expected_weights =
        TensorToVector(weights.permute({2, 3, 0, 1}).contiguous());
#else
    const float expected_output[] = {
        0.0239670f, 0.0031000f, 0.0147798f, 0.0678257f,
        0.0238973f, 0.0028428f, 0.0151240f, 0.0679453f,
    };
    const float expected_weights[] = {
        0.3335101f, 0.3332717f,
        0.3331566f, 0.3288474f,
        0.3333333f, 0.3378809f,
        0.3345418f, 0.3302163f,
        0.3365110f, 0.3309877f,
        0.3289472f, 0.3387960f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-5f,
                  "Cross-attention output matches PyTorch");
    }
    const cyxwiz::Tensor weights_tensor = attention.GetAttentionWeights();
    CheckShape(weights_tensor, {2, 3, 1, 2}, "Cross-attention weights");
    const float* weight_data = weights_tensor.Data<float>();
    for (size_t i = 0; i < weights_tensor.NumElements(); ++i) {
        CheckNear(weight_data[i], expected_weights[i], 1e-5f,
                  "Cross-attention weights match PyTorch");
    }

    const std::vector<float> padding_mask_values = {
        0.0f, 0.0f, -1.0e9f,
        0.0f, 0.0f, -1.0e9f,
    };
    const cyxwiz::Tensor padding_mask({2, 3}, padding_mask_values.data(),
                                      cyxwiz::DataType::Float32);
    const cyxwiz::Tensor masked_output =
        attention.Forward(query, key, value, &padding_mask);
    CheckShape(masked_output, {1, 2, 4},
               "Cross-attention padding-mask output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_padding_mask = torch::from_blob(
        const_cast<float*>(padding_mask_values.data()), {1, 1, 2, 3},
        torch::kFloat32).clone();
    auto masked_weights = torch::softmax(scores + torch_padding_mask, -1);
    auto masked_context = torch::matmul(masked_weights, v)
                              .transpose(1, 2)
                              .contiguous()
                              .view({1, 2, 4});
    const std::vector<float> expected_masked_output =
        TensorToVector(torch::linear(masked_context, torch_wo, torch_bo));
    const std::vector<float> expected_masked_weights =
        TensorToVector(masked_weights.permute({2, 3, 0, 1}).contiguous());
#else
    const float expected_masked_output[] = {
        0.0324154f, -0.0232690f, 0.0081484f, 0.0886260f,
        0.0322481f, -0.0233326f, 0.0083288f, 0.0885547f,
    };
    const float expected_masked_weights[] = {
        0.5002651f, 0.5033410f,
        0.4997348f, 0.4966590f,
        0.0f, 0.0f,
        0.4985327f, 0.4994166f,
        0.5014672f, 0.5005834f,
        0.0f, 0.0f,
    };
#endif
    const float* masked_out = masked_output.Data<float>();
    for (size_t i = 0; i < masked_output.NumElements(); ++i) {
        CheckNear(masked_out[i], expected_masked_output[i], 1e-5f,
                  "Cross-attention padding-mask output matches PyTorch");
    }
    const cyxwiz::Tensor masked_weights_tensor =
        attention.GetAttentionWeights();
    CheckShape(masked_weights_tensor, {2, 3, 1, 2},
               "Cross-attention padding-mask weights");
    const float* masked_weight_data = masked_weights_tensor.Data<float>();
    for (size_t i = 0; i < masked_weights_tensor.NumElements(); ++i) {
        CheckNear(masked_weight_data[i], expected_masked_weights[i], 1e-5f,
                  "Cross-attention padding-mask weights match PyTorch");
    }

    (void)attention.Forward(query, key, value, nullptr);

    const std::vector<float> grad_output_values = {
        0.25f, -0.15f, 0.05f, 0.35f,
        -0.45f, 0.30f, 0.55f, -0.05f,
    };
    const cyxwiz::Tensor grad_output({1, 2, 4}, grad_output_values.data(),
                                     cyxwiz::DataType::Float32);
    const cyxwiz::Tensor grad_query = attention.Backward(grad_output);
    const cyxwiz::Tensor grad_key = attention.GetLastKeyGradient();
    const cyxwiz::Tensor grad_value = attention.GetLastValueGradient();
    CheckShape(grad_query, {1, 2, 4}, "Cross-attention grad_query");
    CheckShape(grad_key, {1, 3, 4}, "Cross-attention grad_key");
    CheckShape(grad_value, {1, 3, 4}, "Cross-attention grad_value");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto bwd_query = torch::from_blob(
        const_cast<float*>(query_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    bwd_query.set_requires_grad(true);
    auto bwd_key = torch::from_blob(
        const_cast<float*>(key_value_values.data()), {1, 3, 4}, torch::kFloat32).clone();
    bwd_key.set_requires_grad(true);
    auto bwd_value = torch::from_blob(
        const_cast<float*>(key_value_values.data()), {1, 3, 4}, torch::kFloat32).clone();
    bwd_value.set_requires_grad(true);
    auto bwd_wq = torch_wq.clone();
    bwd_wq.set_requires_grad(true);
    auto bwd_wk = torch_wk.clone();
    bwd_wk.set_requires_grad(true);
    auto bwd_wv = torch_wv.clone();
    bwd_wv.set_requires_grad(true);
    auto bwd_wo = torch_wo.clone();
    bwd_wo.set_requires_grad(true);
    auto bwd_bq = torch_bq.clone();
    bwd_bq.set_requires_grad(true);
    auto bwd_bk = torch_bk.clone();
    bwd_bk.set_requires_grad(true);
    auto bwd_bv = torch_bv.clone();
    bwd_bv.set_requires_grad(true);
    auto bwd_bo = torch_bo.clone();
    bwd_bo.set_requires_grad(true);
    auto bwd_q = torch::linear(bwd_query, bwd_wq, bwd_bq)
                     .view({1, 2, 2, 2})
                     .transpose(1, 2);
    auto bwd_k = torch::linear(bwd_key, bwd_wk, bwd_bk)
                     .view({1, 3, 2, 2})
                     .transpose(1, 2);
    auto bwd_v = torch::linear(bwd_value, bwd_wv, bwd_bv)
                     .view({1, 3, 2, 2})
                     .transpose(1, 2);
    auto bwd_weights = torch::softmax(
        torch::matmul(bwd_q, bwd_k.transpose(-2, -1)) / std::sqrt(2.0f),
        -1);
    auto bwd_context = torch::matmul(bwd_weights, bwd_v)
                           .transpose(1, 2)
                           .contiguous()
                           .view({1, 2, 4});
    auto bwd_output = torch::linear(bwd_context, bwd_wo, bwd_bo);
    auto torch_grad_output = torch::from_blob(
        const_cast<float*>(grad_output_values.data()), {1, 2, 4},
        torch::kFloat32).clone();
    (bwd_output * torch_grad_output).sum().backward();
    const std::vector<float> expected_grad_query =
        TensorToVector(bwd_query.grad());
    const std::vector<float> expected_grad_key =
        TensorToVector(bwd_key.grad());
    const std::vector<float> expected_grad_value =
        TensorToVector(bwd_value.grad());
#else
    const float expected_grad_query[] = {
        0.000213297f, 0.000300431f, 0.000033963f, -0.000018570f,
        -0.000417809f, 0.000124337f, 0.000416885f, 0.000023726f,
    };
    const float expected_grad_key[] = {
        -0.000122985f, 0.000073003f, 0.000259061f, 0.000623961f,
        0.000026629f, -0.000105136f, -0.000171070f, -0.000241730f,
        0.000096356f, 0.000032132f, -0.000087990f, -0.000382232f,
    };
    const float expected_grad_value[] = {
        0.019096673f, -0.003906057f, 0.008244991f, -0.011004903f,
        0.018997787f, -0.004122680f, 0.008406989f, -0.011105428f,
        0.019280553f, -0.003846259f, 0.008723017f, -0.011139668f,
    };
#endif
    const auto check_tensor_values =
        [](const cyxwiz::Tensor& tensor,
           const auto& expected,
           float tolerance,
           const std::string& message) {
            const float* data = tensor.Data<float>();
            for (size_t i = 0; i < tensor.NumElements(); ++i) {
                CheckNear(data[i], expected[i], tolerance, message);
            }
        };
    check_tensor_values(grad_query, expected_grad_query, 1e-5f,
                        "Cross-attention grad_query matches PyTorch");
    check_tensor_values(grad_key, expected_grad_key, 1e-5f,
                        "Cross-attention grad_key matches PyTorch");
    check_tensor_values(grad_value, expected_grad_value, 1e-5f,
                        "Cross-attention grad_value matches PyTorch");
}

void TestLayerNormParity() {
    const size_t outer_size = 2;
    const size_t normalized_size = 4;
    const float epsilon = 1.0e-5f;
    const std::vector<float> input = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f,
    };
    const std::vector<float> gamma = {
        1.0f, 1.5f, -0.5f, 2.0f,
    };
    const std::vector<float> beta = {
        0.1f, -0.2f, 0.3f, 0.4f,
    };

    const auto output = cyxwiz::LayerNormForwardCpu(
        input,
        outer_size,
        normalized_size,
        gamma,
        beta,
        epsilon);

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_input = torch::from_blob(
        const_cast<float*>(input.data()), {2, 4}, torch::kFloat32).clone();
    auto torch_gamma = torch::from_blob(
        const_cast<float*>(gamma.data()), {4}, torch::kFloat32).clone();
    auto torch_beta = torch::from_blob(
        const_cast<float*>(beta.data()), {4}, torch::kFloat32).clone();
    const std::vector<float> expected = TensorToVector(
        torch::layer_norm(
            torch_input,
            {static_cast<int64_t>(normalized_size)},
            torch_gamma,
            torch_beta,
            epsilon));
#else
    const float expected[] = {
        -1.2416356f, -0.8708178f, 0.0763941f, 3.0832713f,
        -1.2416356f, -0.8708178f, 0.0763941f, 3.0832713f,
    };
#endif
    Check(output.size() == input.size(), "LayerNorm output size");
    for (size_t i = 0; i < output.size(); ++i) {
        CheckNear(output[i], expected[i], 1e-5f,
                  "LayerNorm forward matches PyTorch");
    }

    const std::vector<float> grad_output = {
        0.2f, -0.1f, 0.3f, 0.4f,
        -0.5f, 0.6f, -0.2f, 0.1f,
    };
    const auto backward = cyxwiz::LayerNormBackwardCpu(
        input,
        grad_output,
        outer_size,
        normalized_size,
        gamma,
        epsilon);

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_input_for_backward = torch::from_blob(
        const_cast<float*>(input.data()), {2, 4}, torch::kFloat32).clone();
    torch_input_for_backward.set_requires_grad(true);
    auto torch_gamma_for_backward = torch::from_blob(
        const_cast<float*>(gamma.data()), {4}, torch::kFloat32).clone();
    torch_gamma_for_backward.set_requires_grad(true);
    auto torch_beta_for_backward = torch::from_blob(
        const_cast<float*>(beta.data()), {4}, torch::kFloat32).clone();
    torch_beta_for_backward.set_requires_grad(true);
    auto torch_grad_output = torch::from_blob(
        const_cast<float*>(grad_output.data()), {2, 4}, torch::kFloat32).clone();
    auto torch_backward_output = torch::layer_norm(
        torch_input_for_backward,
        {static_cast<int64_t>(normalized_size)},
        torch_gamma_for_backward,
        torch_beta_for_backward,
        epsilon);
    (torch_backward_output * torch_grad_output).sum().backward();
    const std::vector<float> expected_grad_input =
        TensorToVector(torch_input_for_backward.grad());
    const std::vector<float> expected_grad_gamma =
        TensorToVector(torch_gamma_for_backward.grad());
    const std::vector<float> expected_grad_beta =
        TensorToVector(torch_beta_for_backward.grad());
#else
    const float expected_grad_input[] = {
        0.2638530f, -0.2101902f, -0.3711852f, 0.3175223f,
        -0.4293247f, 0.7065942f, -0.1252188f, -0.1520506f,
    };
    const float expected_grad_gamma[] = {
        0.4024906f, -0.2236059f, 0.0447212f, 0.6708177f,
    };
    const float expected_grad_beta[] = {
        -0.3f, 0.5f, 0.1f, 0.5f,
    };
#endif
    Check(backward.grad_input.size() == input.size(),
          "LayerNorm grad_input size");
    for (size_t i = 0; i < backward.grad_input.size(); ++i) {
        CheckNear(backward.grad_input[i], expected_grad_input[i], 1e-5f,
                  "LayerNorm grad_input matches PyTorch");
    }
    Check(backward.grad_gamma.size() == normalized_size,
          "LayerNorm grad_gamma size");
    Check(backward.grad_beta.size() == normalized_size,
          "LayerNorm grad_beta size");
    for (size_t i = 0; i < normalized_size; ++i) {
        CheckNear(backward.grad_gamma[i], expected_grad_gamma[i], 1e-5f,
                  "LayerNorm grad_gamma matches PyTorch");
        CheckNear(backward.grad_beta[i], expected_grad_beta[i], 1e-5f,
                  "LayerNorm grad_beta matches PyTorch");
    }
}

void TestTransformerEncoderForwardParity() {
    const std::vector<float> input_values = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
    };
    const std::vector<float> W_q = {
        0.10f, 0.20f, -0.10f, 0.00f,
        0.00f, 0.15f, 0.25f, -0.05f,
        -0.20f, 0.05f, 0.30f, 0.10f,
        0.05f, -0.10f, 0.20f, 0.25f,
    };
    const std::vector<float> W_k = {
        0.05f, -0.15f, 0.10f, 0.20f,
        0.20f, 0.00f, -0.10f, 0.05f,
        0.10f, 0.25f, 0.05f, -0.20f,
        -0.05f, 0.10f, 0.15f, 0.30f,
    };
    const std::vector<float> W_v = {
        0.30f, -0.10f, 0.05f, 0.00f,
        -0.20f, 0.25f, 0.10f, 0.15f,
        0.05f, 0.00f, 0.20f, -0.10f,
        0.10f, 0.30f, -0.15f, 0.05f,
    };
    const std::vector<float> W_o = {
        0.20f, 0.10f, -0.05f, 0.30f,
        -0.10f, 0.25f, 0.15f, 0.05f,
        0.05f, -0.20f, 0.35f, 0.10f,
        0.30f, 0.00f, -0.10f, 0.20f,
    };
    const std::vector<float> b_q = {0.01f, -0.02f, 0.03f, 0.04f};
    const std::vector<float> b_k = {-0.03f, 0.02f, 0.01f, -0.01f};
    const std::vector<float> b_v = {0.05f, -0.04f, 0.02f, 0.03f};
    const std::vector<float> b_o = {0.01f, 0.02f, -0.03f, 0.04f};
    const std::vector<float> norm1_gamma = {1.0f, 1.1f, 0.9f, 1.2f};
    const std::vector<float> norm1_beta = {0.01f, -0.02f, 0.03f, -0.04f};
    const std::vector<float> norm2_gamma = {0.8f, 1.0f, 1.2f, 0.7f};
    const std::vector<float> norm2_beta = {-0.03f, 0.04f, -0.01f, 0.02f};
    const std::vector<float> linear1_weights = {
        0.10f, -0.20f, 0.30f, 0.05f,
        -0.15f, 0.25f, 0.10f, -0.05f,
        0.20f, 0.05f, -0.10f, 0.15f,
    };
    const std::vector<float> linear1_bias = {0.01f, -0.02f, 0.03f};
    const std::vector<float> linear2_weights = {
        0.30f, -0.10f, 0.20f,
        -0.05f, 0.25f, 0.10f,
        0.15f, 0.05f, -0.20f,
        0.10f, 0.30f, -0.15f,
    };
    const std::vector<float> linear2_bias = {0.02f, -0.01f, 0.04f, -0.03f};

    cyxwiz::TransformerEncoderModule encoder(4, 2, 3, 0.0f, false);
    encoder.SetTraining(false);
    encoder.SetParameters({
        {"self_attn.W_q", cyxwiz::Tensor({4, 4}, W_q.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_k", cyxwiz::Tensor({4, 4}, W_k.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_v", cyxwiz::Tensor({4, 4}, W_v.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_o", cyxwiz::Tensor({4, 4}, W_o.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_q", cyxwiz::Tensor({4}, b_q.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_k", cyxwiz::Tensor({4}, b_k.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_v", cyxwiz::Tensor({4}, b_v.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_o", cyxwiz::Tensor({4}, b_o.data(), cyxwiz::DataType::Float32)},
        {"norm1.gamma", cyxwiz::Tensor({4}, norm1_gamma.data(), cyxwiz::DataType::Float32)},
        {"norm1.beta", cyxwiz::Tensor({4}, norm1_beta.data(), cyxwiz::DataType::Float32)},
        {"norm2.gamma", cyxwiz::Tensor({4}, norm2_gamma.data(), cyxwiz::DataType::Float32)},
        {"norm2.beta", cyxwiz::Tensor({4}, norm2_beta.data(), cyxwiz::DataType::Float32)},
        {"linear1.weights", cyxwiz::Tensor({3, 4}, linear1_weights.data(), cyxwiz::DataType::Float32)},
        {"linear1.bias", cyxwiz::Tensor({3}, linear1_bias.data(), cyxwiz::DataType::Float32)},
        {"linear2.weights", cyxwiz::Tensor({4, 3}, linear2_weights.data(), cyxwiz::DataType::Float32)},
        {"linear2.bias", cyxwiz::Tensor({4}, linear2_bias.data(), cyxwiz::DataType::Float32)},
    });

    const cyxwiz::Tensor input({1, 2, 4}, input_values.data(),
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = encoder.Forward(input);
    CheckShape(output, {1, 2, 4}, "TransformerEncoder output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_input = torch::from_blob(
        const_cast<float*>(input_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    auto torch_wq = torch::from_blob(
        const_cast<float*>(W_q.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wk = torch::from_blob(
        const_cast<float*>(W_k.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wv = torch::from_blob(
        const_cast<float*>(W_v.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wo = torch::from_blob(
        const_cast<float*>(W_o.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_bq = torch::from_blob(
        const_cast<float*>(b_q.data()), {4}, torch::kFloat32).clone();
    auto torch_bk = torch::from_blob(
        const_cast<float*>(b_k.data()), {4}, torch::kFloat32).clone();
    auto torch_bv = torch::from_blob(
        const_cast<float*>(b_v.data()), {4}, torch::kFloat32).clone();
    auto torch_bo = torch::from_blob(
        const_cast<float*>(b_o.data()), {4}, torch::kFloat32).clone();
    auto torch_norm1_gamma = torch::from_blob(
        const_cast<float*>(norm1_gamma.data()), {4}, torch::kFloat32).clone();
    auto torch_norm1_beta = torch::from_blob(
        const_cast<float*>(norm1_beta.data()), {4}, torch::kFloat32).clone();
    auto torch_norm2_gamma = torch::from_blob(
        const_cast<float*>(norm2_gamma.data()), {4}, torch::kFloat32).clone();
    auto torch_norm2_beta = torch::from_blob(
        const_cast<float*>(norm2_beta.data()), {4}, torch::kFloat32).clone();
    auto torch_linear1_weights = torch::from_blob(
        const_cast<float*>(linear1_weights.data()), {3, 4}, torch::kFloat32).clone();
    auto torch_linear1_bias = torch::from_blob(
        const_cast<float*>(linear1_bias.data()), {3}, torch::kFloat32).clone();
    auto torch_linear2_weights = torch::from_blob(
        const_cast<float*>(linear2_weights.data()), {4, 3}, torch::kFloat32).clone();
    auto torch_linear2_bias = torch::from_blob(
        const_cast<float*>(linear2_bias.data()), {4}, torch::kFloat32).clone();

    auto q = torch::linear(torch_input, torch_wq, torch_bq)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto k = torch::linear(torch_input, torch_wk, torch_bk)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto v = torch::linear(torch_input, torch_wv, torch_bv)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(2.0f);
    auto weights = torch::softmax(scores, -1);
    auto context = torch::matmul(weights, v)
                       .transpose(1, 2)
                       .contiguous()
                       .view({1, 2, 4});
    auto attn_output = torch::linear(context, torch_wo, torch_bo);
    auto norm1_output = torch::layer_norm(
        torch_input + attn_output,
        {4},
        torch_norm1_gamma,
        torch_norm1_beta,
        1.0e-5);
    auto ffn_hidden = torch::relu(torch::linear(
        norm1_output.view({2, 4}),
        torch_linear1_weights,
        torch_linear1_bias));
    auto ffn_output = torch::linear(
        ffn_hidden,
        torch_linear2_weights,
        torch_linear2_bias).view({1, 2, 4});
    const std::vector<float> expected_output = TensorToVector(
        torch::layer_norm(
            norm1_output + ffn_output,
            {4},
            torch_norm2_gamma,
            torch_norm2_beta,
            1.0e-5));
#else
    const float expected_output[] = {
        -0.1815133f, -1.3401134f, 0.1597225f, 1.0196487f,
        -0.8367455f, 1.6354986f, 0.0746440f, -0.4403224f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-5f,
                  "TransformerEncoder forward matches PyTorch primitive composition");
    }
}

void TestBertEncoderHeadLogitParity() {
    const std::vector<float> bert_hidden_values = {
        -0.1815133f, -1.3401134f, 0.1597225f, 1.0196487f,
        -0.8367455f, 1.6354986f, 0.0746440f, -0.4403224f,
    };
    const std::vector<float> cls_head_weights = {
        0.25f, -0.10f, 0.05f, 0.20f,
        -0.15f, 0.30f, 0.10f, -0.05f,
        0.05f, 0.15f, -0.20f, 0.25f,
    };
    const std::vector<float> cls_head_bias = {0.01f, -0.02f, 0.03f};
    const std::vector<float> token_head_weights = {
        0.20f, -0.05f, 0.10f, 0.15f,
        -0.10f, 0.25f, 0.05f, -0.20f,
        0.05f, 0.10f, -0.15f, 0.30f,
        0.30f, -0.20f, 0.25f, 0.05f,
    };
    const std::vector<float> token_head_bias = {0.02f, -0.01f, 0.03f, -0.04f};

    const cyxwiz::Tensor hidden({1, 2, 4}, bert_hidden_values.data(),
                                cyxwiz::DataType::Float32);
    const cyxwiz::Tensor cls_hidden({1, 4}, bert_hidden_values.data(),
                                    cyxwiz::DataType::Float32);

    cyxwiz::LinearModule cls_head(4, 3, true);
    cls_head.SetParameters({
        {"weight", cyxwiz::Tensor({3, 4}, cls_head_weights.data(),
                                   cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({3}, cls_head_bias.data(),
                                 cyxwiz::DataType::Float32)},
    });
    const cyxwiz::Tensor cls_logits = cls_head.Forward(cls_hidden);
    CheckShape(cls_logits, {1, 3}, "BERT CLS classifier logits");

    cyxwiz::TimeDistributedDenseModule token_head(4, 4, true);
    token_head.SetParameters({
        {"weight", cyxwiz::Tensor({4, 4}, token_head_weights.data(),
                                   cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({4}, token_head_bias.data(),
                                 cyxwiz::DataType::Float32)},
    });
    const cyxwiz::Tensor token_logits = token_head.Forward(hidden);
    CheckShape(token_logits, {1, 2, 4}, "BERT token classifier logits");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_hidden = torch::from_blob(
        const_cast<float*>(bert_hidden_values.data()), {1, 2, 4},
        torch::kFloat32).clone();
    auto torch_cls_weights = torch::from_blob(
        const_cast<float*>(cls_head_weights.data()), {3, 4},
        torch::kFloat32).clone();
    auto torch_cls_bias = torch::from_blob(
        const_cast<float*>(cls_head_bias.data()), {3},
        torch::kFloat32).clone();
    auto torch_token_weights = torch::from_blob(
        const_cast<float*>(token_head_weights.data()), {4, 4},
        torch::kFloat32).clone();
    auto torch_token_bias = torch::from_blob(
        const_cast<float*>(token_head_bias.data()), {4},
        torch::kFloat32).clone();

    auto torch_cls_hidden = torch_hidden.index(
        {0, 0, torch::indexing::Slice()}).view({1, 4});
    const std::vector<float> expected_cls_hidden =
        TensorToVector(torch_cls_hidden);
    const std::vector<float> expected_cls_logits = TensorToVector(
        torch::linear(torch_cls_hidden, torch_cls_weights, torch_cls_bias));
    const std::vector<float> expected_token_logits = TensorToVector(
        torch::linear(torch_hidden.view({2, 4}),
                      torch_token_weights,
                      torch_token_bias).view({1, 2, 4}));
#else
    const float expected_cls_hidden[] = {
        -0.1815133f, -1.3401134f, 0.1597225f, 1.0196487f,
    };
    const float expected_cls_logits[] = {
        0.3105489f, -0.4298172f, 0.0428750f,
    };
    const float expected_token_logits[] = {
        0.2196226f, -0.5228206f, 0.1688492f, 0.2644818f,
        -0.2877080f, 0.5743459f, 0.0084193f, -0.6214785f,
    };
#endif

    const float* cls_hidden_data = cls_hidden.Data<float>();
    for (size_t i = 0; i < cls_hidden.NumElements(); ++i) {
        CheckNear(cls_hidden_data[i], expected_cls_hidden[i], 1e-6f,
                  "BERT CLS extraction matches PyTorch hidden[:, 0, :]");
    }

    const float* cls_logit_data = cls_logits.Data<float>();
    for (size_t i = 0; i < cls_logits.NumElements(); ++i) {
        CheckNear(cls_logit_data[i], expected_cls_logits[i], 1e-5f,
                  "BERT CLS classifier logits match PyTorch linear head");
    }

    const float* token_logit_data = token_logits.Data<float>();
    for (size_t i = 0; i < token_logits.NumElements(); ++i) {
        CheckNear(token_logit_data[i], expected_token_logits[i], 1e-5f,
                  "BERT token classifier logits match PyTorch per-token linear head");
    }
}
void TestTransformerDecoderCausalForwardParity() {
    const std::vector<float> input_values = {
        0.2f, -0.1f, 0.4f, 0.7f,
        -0.3f, 0.5f, 0.1f, -0.2f,
    };
    const std::vector<float> W_q = {
        0.10f, 0.20f, -0.10f, 0.00f,
        0.00f, 0.15f, 0.25f, -0.05f,
        -0.20f, 0.05f, 0.30f, 0.10f,
        0.05f, -0.10f, 0.20f, 0.25f,
    };
    const std::vector<float> W_k = {
        0.05f, -0.15f, 0.10f, 0.20f,
        0.20f, 0.00f, -0.10f, 0.05f,
        0.10f, 0.25f, 0.05f, -0.20f,
        -0.05f, 0.10f, 0.15f, 0.30f,
    };
    const std::vector<float> W_v = {
        0.30f, -0.10f, 0.05f, 0.00f,
        -0.20f, 0.25f, 0.10f, 0.15f,
        0.05f, 0.00f, 0.20f, -0.10f,
        0.10f, 0.30f, -0.15f, 0.05f,
    };
    const std::vector<float> W_o = {
        0.20f, 0.10f, -0.05f, 0.30f,
        -0.10f, 0.25f, 0.15f, 0.05f,
        0.05f, -0.20f, 0.35f, 0.10f,
        0.30f, 0.00f, -0.10f, 0.20f,
    };
    const std::vector<float> b_q = {0.01f, -0.02f, 0.03f, 0.04f};
    const std::vector<float> b_k = {-0.03f, 0.02f, 0.01f, -0.01f};
    const std::vector<float> b_v = {0.05f, -0.04f, 0.02f, 0.03f};
    const std::vector<float> b_o = {0.01f, 0.02f, -0.03f, 0.04f};
    const std::vector<float> norm1_gamma = {1.0f, 1.1f, 0.9f, 1.2f};
    const std::vector<float> norm1_beta = {0.01f, -0.02f, 0.03f, -0.04f};
    const std::vector<float> norm2_gamma = {0.8f, 1.0f, 1.2f, 0.7f};
    const std::vector<float> norm2_beta = {-0.03f, 0.04f, -0.01f, 0.02f};
    const std::vector<float> linear1_weights = {
        0.10f, -0.20f, 0.30f, 0.05f,
        -0.15f, 0.25f, 0.10f, -0.05f,
        0.20f, 0.05f, -0.10f, 0.15f,
    };
    const std::vector<float> linear1_bias = {0.01f, -0.02f, 0.03f};
    const std::vector<float> linear2_weights = {
        0.30f, -0.10f, 0.20f,
        -0.05f, 0.25f, 0.10f,
        0.15f, 0.05f, -0.20f,
        0.10f, 0.30f, -0.15f,
    };
    const std::vector<float> linear2_bias = {0.02f, -0.01f, 0.04f, -0.03f};

    cyxwiz::TransformerDecoderModule decoder(4, 2, 3, 0.0f, false);
    decoder.SetTraining(false);
    decoder.SetParameters({
        {"self_attn.W_q", cyxwiz::Tensor({4, 4}, W_q.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_k", cyxwiz::Tensor({4, 4}, W_k.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_v", cyxwiz::Tensor({4, 4}, W_v.data(), cyxwiz::DataType::Float32)},
        {"self_attn.W_o", cyxwiz::Tensor({4, 4}, W_o.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_q", cyxwiz::Tensor({4}, b_q.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_k", cyxwiz::Tensor({4}, b_k.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_v", cyxwiz::Tensor({4}, b_v.data(), cyxwiz::DataType::Float32)},
        {"self_attn.b_o", cyxwiz::Tensor({4}, b_o.data(), cyxwiz::DataType::Float32)},
        {"norm1.gamma", cyxwiz::Tensor({4}, norm1_gamma.data(), cyxwiz::DataType::Float32)},
        {"norm1.beta", cyxwiz::Tensor({4}, norm1_beta.data(), cyxwiz::DataType::Float32)},
        {"norm2.gamma", cyxwiz::Tensor({4}, norm2_gamma.data(), cyxwiz::DataType::Float32)},
        {"norm2.beta", cyxwiz::Tensor({4}, norm2_beta.data(), cyxwiz::DataType::Float32)},
        {"linear1.weights", cyxwiz::Tensor({3, 4}, linear1_weights.data(), cyxwiz::DataType::Float32)},
        {"linear1.bias", cyxwiz::Tensor({3}, linear1_bias.data(), cyxwiz::DataType::Float32)},
        {"linear2.weights", cyxwiz::Tensor({4, 3}, linear2_weights.data(), cyxwiz::DataType::Float32)},
        {"linear2.bias", cyxwiz::Tensor({4}, linear2_bias.data(), cyxwiz::DataType::Float32)},
    });

    const cyxwiz::Tensor input({1, 2, 4}, input_values.data(),
                               cyxwiz::DataType::Float32);
    const cyxwiz::Tensor output = decoder.Forward(input);
    CheckShape(output, {1, 2, 4}, "TransformerDecoder causal output");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_input = torch::from_blob(
        const_cast<float*>(input_values.data()), {1, 2, 4}, torch::kFloat32).clone();
    auto torch_wq = torch::from_blob(
        const_cast<float*>(W_q.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wk = torch::from_blob(
        const_cast<float*>(W_k.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wv = torch::from_blob(
        const_cast<float*>(W_v.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_wo = torch::from_blob(
        const_cast<float*>(W_o.data()), {4, 4}, torch::kFloat32).clone();
    auto torch_bq = torch::from_blob(
        const_cast<float*>(b_q.data()), {4}, torch::kFloat32).clone();
    auto torch_bk = torch::from_blob(
        const_cast<float*>(b_k.data()), {4}, torch::kFloat32).clone();
    auto torch_bv = torch::from_blob(
        const_cast<float*>(b_v.data()), {4}, torch::kFloat32).clone();
    auto torch_bo = torch::from_blob(
        const_cast<float*>(b_o.data()), {4}, torch::kFloat32).clone();
    auto torch_norm1_gamma = torch::from_blob(
        const_cast<float*>(norm1_gamma.data()), {4}, torch::kFloat32).clone();
    auto torch_norm1_beta = torch::from_blob(
        const_cast<float*>(norm1_beta.data()), {4}, torch::kFloat32).clone();
    auto torch_norm2_gamma = torch::from_blob(
        const_cast<float*>(norm2_gamma.data()), {4}, torch::kFloat32).clone();
    auto torch_norm2_beta = torch::from_blob(
        const_cast<float*>(norm2_beta.data()), {4}, torch::kFloat32).clone();
    auto torch_linear1_weights = torch::from_blob(
        const_cast<float*>(linear1_weights.data()), {3, 4}, torch::kFloat32).clone();
    auto torch_linear1_bias = torch::from_blob(
        const_cast<float*>(linear1_bias.data()), {3}, torch::kFloat32).clone();
    auto torch_linear2_weights = torch::from_blob(
        const_cast<float*>(linear2_weights.data()), {4, 3}, torch::kFloat32).clone();
    auto torch_linear2_bias = torch::from_blob(
        const_cast<float*>(linear2_bias.data()), {4}, torch::kFloat32).clone();

    auto q = torch::linear(torch_input, torch_wq, torch_bq)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto k = torch::linear(torch_input, torch_wk, torch_bk)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto v = torch::linear(torch_input, torch_wv, torch_bv)
                 .view({1, 2, 2, 2})
                 .transpose(1, 2);
    auto causal_mask = torch::tensor(
        {{{{0.0f, -1.0e9f}, {0.0f, 0.0f}}}},
        torch::kFloat32);
    auto scores =
        torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(2.0f) +
        causal_mask;
    auto weights = torch::softmax(scores, -1);
    auto context = torch::matmul(weights, v)
                       .transpose(1, 2)
                       .contiguous()
                       .view({1, 2, 4});
    auto attn_output = torch::linear(context, torch_wo, torch_bo);
    auto norm1_output = torch::layer_norm(
        torch_input + attn_output,
        {4},
        torch_norm1_gamma,
        torch_norm1_beta,
        1.0e-5);
    auto ffn_hidden = torch::relu(torch::linear(
        norm1_output.view({2, 4}),
        torch_linear1_weights,
        torch_linear1_bias));
    auto ffn_output = torch::linear(
        ffn_hidden,
        torch_linear2_weights,
        torch_linear2_bias).view({1, 2, 4});
    const std::vector<float> expected_output = TensorToVector(
        torch::layer_norm(
            norm1_output + ffn_output,
            {4},
            torch_norm2_gamma,
            torch_norm2_beta,
            1.0e-5));
#else
    const float expected_output[] = {
        -0.1711998f, -1.3525668f, 0.1731623f, 1.0115018f,
        -0.8367455f, 1.6354986f, 0.0746440f, -0.4403224f,
    };
#endif
    const float* out = output.Data<float>();
    for (size_t i = 0; i < output.NumElements(); ++i) {
        CheckNear(out[i], expected_output[i], 1e-5f,
                  "TransformerDecoder causal forward matches PyTorch primitive composition");
    }
}

void TestGenerationSamplingDistributionParity() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.temperature = 0.75f;
    config.top_k = 4;
    config.top_p = 0.85f;

    const std::vector<float> logits = {
        9.0f, 8.0f, 7.0f, 6.0f, 5.0f,
        0.25f, 1.25f, -0.75f, 2.0f, 0.5f,
    };

    const auto candidates = cyxwiz::BuildNextTokenDistribution(
        logits,
        1,
        2,
        5,
        config);

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_logits = torch::tensor(
        {0.25f, 1.25f, -0.75f, 2.0f, 0.5f}, torch::kFloat32) /
        config.temperature;
    auto torch_probabilities = torch::softmax(torch_logits, -1);
    auto topk = torch::topk(torch_probabilities,
                            static_cast<int64_t>(config.top_k));
    std::vector<float> topk_probabilities = TensorToVector(std::get<0>(topk));
    std::vector<int64_t> topk_ids = TensorToInt64Vector(std::get<1>(topk));

    std::vector<int64_t> expected_ids;
    std::vector<float> expected_probabilities;
    float cumulative = 0.0f;
    for (size_t i = 0; i < topk_ids.size(); ++i) {
        expected_ids.push_back(topk_ids[i]);
        expected_probabilities.push_back(topk_probabilities[i]);
        cumulative += topk_probabilities[i];
        if (cumulative >= config.top_p) {
            break;
        }
    }
    float filtered_sum = 0.0f;
    for (const float probability : expected_probabilities) {
        filtered_sum += probability;
    }
    for (float& probability : expected_probabilities) {
        probability /= filtered_sum;
    }
#else
    const std::vector<int64_t> expected_ids = {3, 1, 4};
    const std::vector<float> expected_probabilities = {
        0.6652409f,
        0.2447285f,
        0.0900306f,
    };
#endif

    CheckCandidatesNear(candidates,
                        expected_ids,
                        expected_probabilities,
                        1e-5f,
                        "Generation top-k/top-p distribution matches PyTorch");

    config.sampling_mode = cyxwiz::LanguageModelSamplingMode::Greedy;
    std::mt19937 rng(11);
    const auto selection = cyxwiz::SelectNextTokenFromDistribution(
        candidates,
        config,
        rng);
    Check(selection.token_id == expected_ids.front(),
          "Generation greedy selection should match PyTorch argmax candidate");
    CheckNear(selection.probability,
              expected_probabilities.front(),
              1e-5f,
              "Generation greedy selection probability matches PyTorch fixture");

    // PyTorch is the oracle for candidate probabilities. CyxWiz owns the
    // C++ RNG stream, so deterministic multinomial parity means seeded replay
    // over that PyTorch-verified distribution, not RNG identity with torch.multinomial.
    config.sampling_mode = cyxwiz::LanguageModelSamplingMode::Multinomial;
    std::mt19937 replay_rng_a(2026);
    std::mt19937 replay_rng_b(2026);
    for (size_t i = 0; i < 8; ++i) {
        const auto selected_a = cyxwiz::SelectNextTokenFromDistribution(
            candidates,
            config,
            replay_rng_a);
        const auto selected_b = cyxwiz::SelectNextTokenFromDistribution(
            candidates,
            config,
            replay_rng_b);
        Check(selected_a.token_id == selected_b.token_id,
              "Generation multinomial seeded replay token ID should be stable");
        CheckNear(selected_a.probability,
                  selected_b.probability,
                  0.0f,
                  "Generation multinomial seeded replay probability should be stable");
        bool in_candidate_set = false;
        for (const int64_t token_id : expected_ids) {
            if (selected_a.token_id == token_id) {
                in_candidate_set = true;
                break;
            }
        }
        Check(in_candidate_set,
              "Generation multinomial selection should stay within PyTorch candidate set");
    }
}

void TestTinyCausalLanguageModelLogitsAndLossParity() {
    const std::vector<float> decoder_hidden_values = {
        -0.1711998f, -1.3525668f, 0.1731623f, 1.0115018f,
        -0.8367455f, 1.6354986f, 0.0746440f, -0.4403224f,
    };
    const std::vector<float> lm_head_weights = {
        0.20f, -0.10f, 0.05f, 0.30f,
        -0.15f, 0.25f, 0.10f, -0.05f,
        0.05f, 0.15f, -0.20f, 0.10f,
        0.30f, -0.05f, 0.25f, -0.10f,
        -0.10f, 0.20f, 0.15f, 0.05f,
    };
    const std::vector<float> lm_head_bias = {
        0.01f, -0.02f, 0.03f, 0.04f, -0.01f,
    };

    cyxwiz::LinearModule lm_head(4, 5, true);
    lm_head.SetParameters({
        {"weight", cyxwiz::Tensor({5, 4}, lm_head_weights.data(),
                                   cyxwiz::DataType::Float32)},
        {"bias", cyxwiz::Tensor({5}, lm_head_bias.data(),
                                 cyxwiz::DataType::Float32)},
    });

    const cyxwiz::Tensor hidden({2, 4}, decoder_hidden_values.data(),
                                cyxwiz::DataType::Float32);
    const cyxwiz::Tensor logits = lm_head.Forward(hidden);
    CheckShape(logits, {2, 5}, "Tiny causal LM logits");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_hidden = torch::from_blob(
        const_cast<float*>(decoder_hidden_values.data()), {2, 4},
        torch::kFloat32).clone();
    auto torch_weights = torch::from_blob(
        const_cast<float*>(lm_head_weights.data()), {5, 4},
        torch::kFloat32).clone();
    auto torch_bias = torch::from_blob(
        const_cast<float*>(lm_head_bias.data()), {5},
        torch::kFloat32).clone();
    auto torch_logits = torch::linear(torch_hidden, torch_weights, torch_bias);
    const std::vector<float> expected_logits = TensorToVector(torch_logits);
#else
    const float expected_logits[] = {
        0.4231254f, -0.3657206f, -0.1149273f, -0.0015912f,
        -0.1868439f, -0.4492635f, 0.5438670f, 0.1745265f,
        -0.2301053f, 0.3899548f,
    };
#endif
    const float* logit_data = logits.Data<float>();
    for (size_t i = 0; i < logits.NumElements(); ++i) {
        CheckNear(logit_data[i], expected_logits[i], 1e-5f,
                  "Tiny causal LM vocabulary logits match PyTorch linear head");
    }

    const int64_t target_values[] = {1, 3};
    const cyxwiz::Tensor targets({2}, target_values, cyxwiz::DataType::Int64);
    cyxwiz::CrossEntropyLoss loss(cyxwiz::Reduction::Mean);
    const cyxwiz::Tensor loss_value = loss.Forward(logits, targets);
    CheckShape(loss_value, {1}, "Tiny causal LM loss");

#if defined(CYXWIZ_HAS_PYTORCH) && !defined(_DEBUG)
    auto torch_targets = torch::tensor({1, 3}, torch::kInt64);
    const std::vector<float> expected_loss =
        TensorToVector(torch::nn::functional::cross_entropy(
            torch_logits,
            torch_targets,
            torch::nn::functional::CrossEntropyFuncOptions()
                .reduction(torch::kMean)));
#else
    const float expected_loss[] = {1.9774697f};
#endif
    CheckNear(loss_value.Data<float>()[0], expected_loss[0], 1e-5f,
              "Tiny causal LM cross entropy matches PyTorch");
}

} // namespace

int main() {
    try {
        TestEmbeddingForwardAndGradientParity();
        TestPositionalEncodingParity();
        TestScaledDotProductAttentionParity();
        TestMultiHeadAttentionForwardParity();
        TestMultiHeadAttentionCrossAttentionParity();
        TestLayerNormParity();
        TestTransformerEncoderForwardParity();
        TestBertEncoderHeadLogitParity();
        TestTransformerDecoderCausalForwardParity();
        TestGenerationSamplingDistributionParity();
        TestTinyCausalLanguageModelLogitsAndLossParity();
        std::cout << "Computation truth transformer primitive checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unhandled exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "FAIL: unknown unhandled exception\n";
        return 1;
    }
}
