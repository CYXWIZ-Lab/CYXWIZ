#include "core/test_manager.h"
#include <chrono>
#include <thread>
#include "core/test_executor.h"
#include "core/arrow_dataset.h"
#include "core/sequence_arrow_batcher.h"
#include "core/checkpoint_manager.h"
#include "core/model_builder.h"
#include "core/execution_device_context.h"
#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <iostream>
#include <stdexcept>

void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
class FixedTokenLogits : public cyxwiz::Module {
public:
    explicit FixedTokenLogits(std::vector<float> row) : row_(std::move(row)) {}
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        std::vector<float> data;
        for (size_t i=0;i<input.NumElements();++i) data.insert(data.end(),row_.begin(),row_.end());
        return cyxwiz::Tensor({input.Shape()[0],input.Shape()[1],row_.size()},data.data(),cyxwiz::DataType::Float32);
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor&) override { throw std::runtime_error("Evaluation must not backpropagate"); }
    std::string GetName() const override { return "FixedTokenLogits"; }
private:
    std::vector<float> row_;
};

int main(int argc, char** argv) {
    try {
        if (argc == 3) {
            using Json = nlohmann::json;
            std::ifstream source(argv[1]);
            auto fixture = Json::parse(source);
            const auto activated=cyxwiz::Device(cyxwiz::DeviceType::CUDA,0).ActivateExact(true);
            Check(activated.success && activated.execution_validated,"CUDA activation");
            cyxwiz::ScopedArrayFireFallbackPolicy policy(cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
            arrow::StringBuilder tokens, groups;
            for(const auto& row:fixture.at("rows")) {
                Check(tokens.Append(row.at("tokens").get<std::string>()).ok(),"tokens");
                Check(groups.Append(row.at("group").get<std::string>()).ok(),"group");
            }
            std::shared_ptr<arrow::Array> t,g;
            Check(tokens.Finish(&t).ok() && groups.Finish(&g).ok(),"finish");
            auto dataset=std::make_shared<cyxwiz::ArrowDataset>(arrow::Table::Make(
                arrow::schema({arrow::field("tokens",arrow::utf8()),arrow::field("group",arrow::utf8())}),{t,g}),"checkpoint_test");
            cyxwiz::TrainingConfiguration config;
            config.input_size=fixture.at("context"); config.input_shape={config.input_size};
            config.sequence_batch.enabled=true;
            config.sequence_batch.create_causal_lm_targets=true;
            config.sequence_batch.create_attention_mask=true;
            config.sequence_batch.token_column="tokens"; config.sequence_batch.sentence_id_column="group";
            config.sequence_batch.max_sequence_length=static_cast<int>(config.input_size);
            config.sequence_batch.expected_token_vocabulary=fixture.at("vocabulary").get<std::vector<std::string>>();
            config.output_size=config.sequence_batch.expected_token_vocabulary.size();
            config.has_data_split=true; config.train_ratio=fixture.at("train_ratio");
            config.val_ratio=fixture.at("val_ratio"); config.test_ratio=fixture.at("test_ratio");
            config.shuffle=false; config.drop_last=false; config.forbid_native_cpu_fallback=true;
            config.loss_type=gui::NodeType::CrossEntropyLoss;
            config.loss_params["ignore_index"]="-100"; config.loss_params["reduction"]="mean";
            for(const auto& entry:fixture.at("layers")) {
                cyxwiz::CompiledLayer layer;
                layer.type=static_cast<gui::NodeType>(entry.at("type").get<int>());
                layer.parameters=entry.at("parameters").get<std::map<std::string,std::string>>();
                if(layer.parameters.count("units")) layer.units=std::stoi(layer.parameters.at("units"));
                config.layers.push_back(layer);
            }
            auto built=cyxwiz::BuildSequentialFromConfig(config);
            Check(built.ok(),"checkpoint model build");
            auto model=std::shared_ptr<cyxwiz::SequentialModel>(std::move(built.model));
            const std::filesystem::path checkpoint=fixture.at("checkpoint").get<std::string>();
            cyxwiz::CheckpointManager manager(checkpoint.parent_path().string());
            Check(manager.LoadCheckpoint(*model,nullptr,checkpoint.filename().string()).has_value(),"checkpoint load");
            Json results=Json::array();
            for(int batch_size:{2,7}) {
                cyxwiz::TestExecutor test(config,dataset,"target_ids",cyxwiz::TestDatasetScope::ConfiguredTestSplit);
                test.SetModel(model); test.Test(batch_size);
                const auto m=test.GetMetrics();
                Check(m.is_complete && m.total_samples==fixture.at("expected_windows").get<int>(),"window parity");
                Check(m.total_target_values==fixture.at("expected_tokens").get<size_t>(),"target parity");
                Check(std::abs(m.test_loss-fixture.at("expected_loss").get<float>())<1e-4,"automatic-test loss parity");
                Check(std::abs(m.test_accuracy-fixture.at("expected_accuracy").get<float>())<1e-5,"automatic-test accuracy parity");
                results.push_back({{"batch_size",batch_size},{"windows",m.total_samples},{"valid_tokens",m.total_target_values},
                    {"loss",m.test_loss},{"accuracy",m.test_accuracy},{"complete",m.is_complete}});
            }
            std::ofstream output(argv[2]); output<<results.dump(2)<<'\n';
            Check(bool(output),"report write");
            std::cout<<"PASS: saved-checkpoint Run Test agrees with automatic held-out results\n";
            return 0;
        }
        Check(argc==1,"Usage: test_causal_lm_testing [fixture.json new_report.json]");
        arrow::StringBuilder tokens, groups;
        for (auto text : {"x x y", "x y", "x y", "x y", "x y x", "x y"})
            Check(tokens.Append(text).ok(),"tokens");
        for (auto group : {"a","b","c","d","e","e"}) Check(groups.Append(group).ok(),"groups");
        std::shared_ptr<arrow::Array> t,g;
        Check(tokens.Finish(&t).ok() && groups.Finish(&g).ok(),"finish");
        auto dataset=std::make_shared<cyxwiz::ArrowDataset>(arrow::Table::Make(
            arrow::schema({arrow::field("tokens",arrow::utf8()),arrow::field("group",arrow::utf8())}),{t,g}),"causal_test");
        cyxwiz::TrainingConfiguration config;
        config.sequence_batch.enabled=true;
        config.sequence_batch.create_causal_lm_targets=true;
        config.sequence_batch.create_attention_mask=true;
        config.sequence_batch.token_column="tokens";
        config.sequence_batch.sentence_id_column="group";
        config.sequence_batch.max_sequence_length=4;
        config.has_data_split=true;
        config.train_ratio=.6f; config.val_ratio=.2f; config.test_ratio=.2f;
        config.shuffle=false; config.drop_last=false;
        config.loss_type=gui::NodeType::CrossEntropyLoss;
        config.loss_params["ignore_index"]="-100";
        config.loss_params["reduction"]="mean";
        auto built=cyxwiz::BuildSequenceBatcherFromArrowDataset(dataset,config,2);
        Check(built.success(),"sequence fixture build");
        cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(built,config);
        Check(built.id_to_label.size()==4,"fixture vocabulary");
        std::vector<float> row;
        for (auto word : built.id_to_label) row.push_back(std::log(word=="x"?.6f:word=="y"?.2f:.1f));
        auto model=std::make_shared<cyxwiz::SequentialModel>();
        model->Add<FixedTokenLogits>(row);
        for (int batch_size : {1,2,3}) {
            cyxwiz::TestExecutor executor(config,dataset,"target_ids",cyxwiz::TestDatasetScope::ConfiguredTestSplit);
            executor.SetModel(model);
            executor.Test(batch_size);
            const auto m=executor.GetMetrics();
            Check(m.is_complete && !executor.IsTesting() && m.causal_lm_mode,"completed causal mode");
            Check(m.total_samples==2 && m.total_target_values==3,"group partition and padding counts");
            Check(std::abs(m.test_accuracy-1.0/3)<1e-6,"token accuracy");
            Check(std::abs(m.test_loss-(-std::log(.6)-2*std::log(.2))/3)<1e-5,"token-weighted mean loss");
            Check(m.confusion_matrix.matrix.empty() && m.per_class_metrics.empty(),"no class metrics");
        }
        auto sum_config=config;
        sum_config.loss_params["reduction"]="sum";
        cyxwiz::TestExecutor summed(sum_config,dataset,"",cyxwiz::TestDatasetScope::ConfiguredTestSplit);
        summed.SetModel(model); summed.Test(1);
        Check(std::abs(summed.GetMetrics().test_loss-(-std::log(.6)-2*std::log(.2)))<1e-5,"sum loss");
        auto changed=config;
        std::swap(changed.sequence_batch.expected_token_vocabulary[2],changed.sequence_batch.expected_token_vocabulary[3]);
        // Loading a checkpoint for testing: a compiled config has no frozen
        // vocabulary; it is prepared from the training dataset exactly as training does.
        {
            auto compiled=config; compiled.sequence_batch.expected_token_vocabulary.clear();
            std::string error;
            Check(cyxwiz::PrepareSequenceEvaluationVocabulary(compiled,dataset,error),"vocabulary prepared");
            Check(compiled.sequence_batch.expected_token_vocabulary==config.sequence_batch.expected_token_vocabulary,
                  "prepared vocabulary equals the training vocabulary");
            auto missing=config; missing.sequence_batch.expected_token_vocabulary.clear(); missing.dataset_name="absent";
            Check(!cyxwiz::PrepareSequenceEvaluationVocabulary(missing,nullptr,error) &&
                  error.find("Apply its Data Input")!=std::string::npos,"missing dataset explains the fix");
            auto ready=config; std::string unused;
            Check(cyxwiz::PrepareSequenceEvaluationVocabulary(ready,nullptr,unused) &&
                  ready.sequence_batch.expected_token_vocabulary==config.sequence_batch.expected_token_vocabulary,
                  "an existing vocabulary is kept");
        }
        // A supplied Test dataset is scored whole (every row), not re-split.
        {
            cyxwiz::TestExecutor whole(config,dataset,"",cyxwiz::TestDatasetScope::EntireProvidedDataset);
            whole.SetModel(model); whole.Test(2);
            const auto w=whole.GetMetrics();
            Check(w.is_complete && w.causal_lm_mode && !whole.IsTesting(),"supplied Test dataset completes");
            // All six rows, 8 next-token targets (x-targets: 2, y-targets: 6); the fixture
            // always predicts x, so accuracy 2/8 and loss (2*-ln .6 + 6*-ln .2)/8.
            Check(w.total_samples==6 && w.total_target_values==8 && w.correct_predictions==2,"supplied Test dataset scores every row");
            Check(std::abs(w.test_accuracy-0.25)<1e-6,"whole-dataset accuracy");
            Check(std::abs(w.test_loss-(-2*std::log(.6)-6*std::log(.2))/8)<1e-5,"whole-dataset token-weighted loss");
        }
        for(int failure=0;failure<4;++failure) {
            auto bad_config=failure==2?config:changed;
            if(failure==2) bad_config.sequence_batch.expected_token_vocabulary.clear();
            // 0: split, different vocabulary; 1 and 3: supplied dataset with a
            // different vocabulary; 2: no frozen vocabulary - all fail closed.
            cyxwiz::TestExecutor bad(bad_config,dataset,"",failure==1||failure==3?
                cyxwiz::TestDatasetScope::EntireProvidedDataset:cyxwiz::TestDatasetScope::ConfiguredTestSplit);
            bad.SetModel(model);
            bool rejected=false;
            try { bad.Test(failure==3?1:2); } catch(const std::runtime_error&) { rejected=true; }
            Check(rejected && !bad.IsTesting() && !bad.GetMetrics().is_complete,"fail closed and clean lifecycle");
        }
        cyxwiz::TestExecutor cancelled(config,dataset,"",cyxwiz::TestDatasetScope::ConfiguredTestSplit);
        cancelled.SetModel(model);
        cancelled.Test(1,[&](int,int,float){cancelled.Stop();});
        Check(!cancelled.GetMetrics().is_complete && !cancelled.IsTesting(),"cancel is not completion");

        // Exercise the actual manager worker and main-thread completion queue.
        auto& tasks=cyxwiz::AsyncTaskManager::Instance();tasks.Initialize(2);
        auto& manager=cyxwiz::TestManager::Instance();
        const auto ui=std::this_thread::get_id();int callbacks=0;
        manager.SetOnTestingEnd([&](bool ok,const cyxwiz::TestingMetrics& metrics) {
            Check(std::this_thread::get_id()==ui,"manager end callback must be on UI thread");
            Check(ok&&metrics.causal_lm_mode,"manager causal result");++callbacks;
        });
        for(int run=0;run<12;++run) {
            Check(manager.StartTestingArrow(config,dataset,"target_ids",
                cyxwiz::TestDatasetScope::ConfiguredTestSplit,1,model,
                [&](const cyxwiz::TestingMetrics& metrics) {
                    Check(std::this_thread::get_id()==ui,"manager completion must be on UI thread");
                    Check(std::abs(metrics.test_accuracy-1.0/3)<1e-6,"manager parity");
                    ++callbacks;
                }),"repeated test must start");
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(manager.IsTestingActive()||callbacks<(run+1)*2) {
                Check(std::chrono::steady_clock::now()<deadline,"manager completion timeout/deadlock");
                tasks.ProcessCompletedCallbacks();std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        manager.SetOnTestingEnd(nullptr);manager.WaitForTestingStop();
        tasks.Shutdown();tasks.ProcessCompletedCallbacks();
        Check(manager.HasResults(),"last results published on UI");
        std::cout<<"PASS: 12 repeated manager runs, UI-only callbacks, unchanged causal metrics\n";
        std::cout<<"PASS: causal Run Test group splits, shifted targets, padding, mean/sum loss, vocabulary identity, scope and cancellation\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
