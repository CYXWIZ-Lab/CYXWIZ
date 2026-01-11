#include <cyxwiz/tensor.h>
#include <cyxwiz/layer.h>
#include <iostream>
#include <vector>
#include <cstring>

using namespace cyxwiz;

int main() {
    // Create weight tensor
    const int num_embeddings = 10;
    const int embedding_dim = 4;
    
    std::vector<float> weights(num_embeddings * embedding_dim);
    for (int i = 0; i < num_embeddings; i++) {
        weights[i * embedding_dim + 0] = i * 0.1f;
        weights[i * embedding_dim + 1] = i * 0.2f;
        weights[i * embedding_dim + 2] = i * 0.3f;
        weights[i * embedding_dim + 3] = i * 0.4f;
    }
    
    Tensor weight_tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)},
                         weights.data(), DataType::Float32);
    
    std::cout << "Weight tensor values:\n";
    const float* wdata = weight_tensor.Data<float>();
    for (int i = 0; i < 10; i++) {
        std::cout << "  weights[" << i << "] = " << wdata[i] << "\n";
    }
    
    // Create embedding layer
    EmbeddingLayer embed(num_embeddings, embedding_dim);
    embed.LoadPretrainedWeights(weight_tensor, false);
    
    // Check internal weights
    auto params = embed.GetParameters();
    std::cout << "\nEmbedding params count: " << params.size() << "\n";
    for (const auto& [name, tensor] : params) {
        std::cout << "  " << name << " shape: ";
        for (auto s : tensor.Shape()) std::cout << s << " ";
        std::cout << "\n";
        if (name == "weight") {
            const float* d = tensor.Data<float>();
            std::cout << "  First 10 weight values: ";
            for (int i = 0; i < 10; i++) std::cout << d[i] << " ";
            std::cout << "\n";
        }
    }
    
    // Create input
    std::vector<int32_t> indices = {1, 5, 2, 7, 0, 3};
    Tensor input({2, 3}, indices.data(), DataType::Int32);
    
    std::cout << "\nInput indices: ";
    const int32_t* idata = input.Data<int32_t>();
    for (int i = 0; i < 6; i++) std::cout << idata[i] << " ";
    std::cout << "\n";
    
    // Forward
    Tensor output = embed.Forward(input);
    
    std::cout << "\nOutput shape: ";
    for (auto s : output.Shape()) std::cout << s << " ";
    std::cout << "\n";
    
    std::cout << "Output values:\n";
    const float* odata = output.Data<float>();
    for (int i = 0; i < 24; i++) {
        std::cout << "  out[" << i << "] = " << odata[i];
        if ((i + 1) % 4 == 0) std::cout << "\n";
        else std::cout << " | ";
    }
    
    std::cout << "\nExpected for batch 0, pos 0, idx 1: 0.1, 0.2, 0.3, 0.4\n";
    std::cout << "Expected for batch 0, pos 1, idx 5: 0.5, 1.0, 1.5, 2.0\n";
    
    return 0;
}
