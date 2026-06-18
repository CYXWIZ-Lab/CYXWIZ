#pragma once

#include <cstddef>
#include <string>

namespace cyxwiz {

struct TrainingRunComparisonRecord {
    std::string run_id;
    std::string run_status;
    std::string dataset_name;
    std::string model_family;
    std::string primary_layer_type;
    std::string architecture_summary;
    int model_layer_count = 0;
    int epochs = 0;
    int batch_size = 0;
    float learning_rate = 0.0f;
    float train_ratio = 0.0f;
    float val_ratio = 0.0f;
    float test_ratio = 0.0f;
    size_t train_sample_count = 0;
    size_t val_sample_count = 0;
    size_t test_sample_count = 0;
    bool bidirectional = false;
    int hidden_size = 0;
    int num_layers = 0;
    bool save_best_checkpoint = false;
    int early_stopping_patience = 0;
    std::string checkpoint_dir;
    std::string checkpoint_used;
    bool has_validation_metrics = false;
    bool has_test_metrics = false;
    float best_val_loss = 0.0f;
    float best_val_accuracy = 0.0f;
    float final_train_loss = 0.0f;
    float final_train_accuracy = 0.0f;
    float final_test_loss = 0.0f;
    float final_test_accuracy = 0.0f;
    float elapsed_seconds = 0.0f;
};

} // namespace cyxwiz
