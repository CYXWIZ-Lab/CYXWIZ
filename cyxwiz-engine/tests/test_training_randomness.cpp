#include "core/training_executor.h"
#include "core/training_randomness.h"
#include "core/checkpoint_manager.h"
#include "core/debug_run_paths.h"
#include "core/ner_sequence_builder.h"
#include "core/execution_device_preferences.h"
#include "core/training_trace_collector.h"
#include "plugin/registries/plugin_training_hook_manager.h"
#include "route_qualification_test_fixture.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <cyxwiz/tokenizer.h>
#include <sstream>

namespace {
using namespace cyxwiz;
void Require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::vector<float> Values(const Tensor& t){
    const auto*p=t.ReadData<float>();std::vector<float> v(p,p+t.NumElements());
    for(float f:v)Require(std::isfinite(f),"nonfinite value");return v;
}
std::vector<float> Weights(SequentialModel& model){
    std::vector<float> result;
    for(const auto&[name,t]:model.GetParameters()){
        if(name.find("grad_")!=std::string::npos)continue;
        const auto values=Values(t);result.insert(result.end(),values.begin(),values.end());
    }return result;
}
struct Result {
    std::vector<float> initial,output1,output2,losses,final;
    std::vector<std::vector<float>> updates;
    TrainingMetrics metrics;
};
struct InitialSnapshot : plugin::ITrainingHook {
    TrainingExecutor* executor=nullptr;Result* result=nullptr;
    std::thread::id caller;
    void OnTrainingStart(plugin::TrainingContext&) override {
        Require(std::this_thread::get_id()!=caller,"seeding/construction must execute on worker");
        auto* model=executor->GetModel();Require(model!=nullptr,"initial model");
        const ScopedArrayFireHostSyncAttribution output(ArrayFireHostSyncCategory::DebugSampleDump,"ModelSeedTest::InitialSnapshot");
        result->initial=Weights(*model);
        const int64_t ids[]={2,3,4,5,5,4,3,2};Tensor x({2,4},ids,DataType::Int64);
        model->SetTraining(true);
        result->output1=Values(model->Forward(x));result->output2=Values(model->Forward(x));
    }
};
Result Run(const std::filesystem::path& path,DeviceType device,int seed,int data_seed,bool previews=false){
    std::filesystem::create_directories(path);
    const std::vector<SequenceSample> samples={
        {{2,3,4,5},{},{}}, {{5,4,3,2},{},{}},
        {{3,3,4,2},{},{}}, {{4,2,5,3},{},{}}};
    const std::vector<std::string> vocabulary={"[PAD]","[UNK]","[BOS]","[EOS]","one","two"};
    SequenceBatcherConfig builder;builder.batch_size=2;builder.max_sequence_length=4;
    builder.create_causal_lm_targets=true;builder.shuffle=true;
    builder.seed=static_cast<uint32_t>(data_seed);
    TrainingConfiguration config;config.model_seed=seed;config.dataloader_seed=data_seed;
    config.input_size=4;config.input_shape={4};config.output_size=vocabulary.size();
    config.loss_type=gui::NodeType::CrossEntropyLoss;config.loss_params["ignore_index"]="-100";
    config.optimizer_type=gui::NodeType::Adam;config.learning_rate=.003f;
    config.sequence_batch.enabled=true;config.sequence_batch.ignore_index=-100;
    config.sequence_batch.create_causal_lm_targets=true;config.sequence_batch.target_ignore_index=-100;
    config.save_best_checkpoint=false;config.early_stopping_patience=0;
    config.checkpoint_dir=path.string();config.log_interval=1;
    config.forbid_native_cpu_fallback=true;
    if(previews){
        Vocabulary vocabulary_asset; vocabulary_asset.SetVocabulary(vocabulary);
        std::ostringstream artifact;Require(vocabulary_asset.SaveToStream(artifact),"preview vocabulary");
        config.generation_preview.enabled=true;config.generation_preview.every_epochs=1;
        config.generation_preview.max_new_tokens=1;config.generation_preview.prompts="one";
        config.generation_preview.tokenizer_type=0;config.generation_preview.context=4;
        config.generation_preview.vocabulary_artifact=artifact.str();
    }
    CompiledLayer embedding;embedding.type=gui::NodeType::Embedding;
    embedding.parameters={{"num_embeddings",std::to_string(vocabulary.size())},{"embedding_dim","8"}};
    config.layers.push_back(embedding);
    for(int i=0;i<2;++i){CompiledLayer decoder;decoder.type=gui::NodeType::TransformerDecoder;
        decoder.parameters={{"d_model","8"},{"num_heads","2"},{"dim_feedforward","16"},{"dropout","0.25"},{"ffn_dropout","0.25"}};
        config.layers.push_back(decoder);}
    CompiledLayer head;head.type=gui::NodeType::TimeDistributed;head.units=static_cast<int>(vocabulary.size());config.layers.push_back(head);
    TrainingExecutor executor(config,std::make_unique<SequenceBatcher>(samples,builder),vocabulary);
    Result result;InitialSnapshot hook;hook.executor=&executor;hook.result=&result;hook.caller=std::this_thread::get_id();
    plugin::PluginTrainingHookManager::Instance().RegisterHook("model-seed-test",&hook);
    SetPendingExecutionDeviceSelection(device,0);
    std::exception_ptr error;
    std::thread worker([&]{try{
        if(seed==-1){
            const auto activated=Device(device,0).ActivateExact(true);
            Require(activated.success&&activated.execution_validated,"unset probe activation");
            SeedCurrentArrayFireRandomEngine(52);
        }
        executor.Train(2,2,[&](int,int,int,float loss,float){
            Require(std::isfinite(loss),"finite training loss");result.losses.push_back(loss);
            const ScopedArrayFireHostSyncAttribution output(ArrayFireHostSyncCategory::DebugSampleDump,"ModelSeedTest::OptimizerSnapshot");
            result.updates.push_back(Weights(*executor.GetModel()));
        },nullptr,[&](const TrainingMetrics& metrics){result.metrics=metrics;});
        Require(result.metrics.optimizer_step_count==4,"two epochs must execute four optimizer updates");
        result.final=Weights(*executor.GetModel());
        auto& trace=TrainingTraceCollector::Instance();
        for(int i=0;i<250;++i)trace.RecordRuntimeEvent("SeedRetentionProbe","bounded event queue");
        trace.FinishRun("completed");
        const auto snapshot=trace.Snapshot();
        const auto persisted=TrainingTraceCollector::LoadLastTrace();
        Require(snapshot.native_cpu_fallback_count==0,"strict training must have zero native fallback");
        Require(snapshot.randomness.model_seed==seed&&persisted&&persisted->randomness.model_seed==seed,
                "seed provenance must survive event eviction and trace reload");
        std::filesystem::copy_file(GetDebugRunRoot()/"current_training_trace.json",path/"training_trace.json",
                                   std::filesystem::copy_options::overwrite_existing);
        CheckpointManager checkpoints(path.string());
        const auto saved=checkpoints.SaveCheckpoint(*executor.GetModel(),nullptr,result.metrics,"provenance");
        Require(!saved.empty(),"checkpoint write");
        const auto loaded=checkpoints.LoadCheckpoint(*executor.GetModel(),nullptr,"provenance");
        Require(loaded&&loaded->randomness.model_seed==seed&&loaded->randomness.dataloader_seed==data_seed,"checkpoint seed roundtrip");
        Require(!checkpoints.InspectCheckpoint("provenance").can_exact_resume,"seed must not advertise exact resume");
        const auto metadata_path=std::filesystem::path(saved)/"metadata.json";
        nlohmann::json metadata;
        {std::ifstream file(metadata_path);file>>metadata;}
        auto legacy=metadata;legacy.erase("randomness");
        {std::ofstream file(metadata_path);file<<legacy.dump(2);Require(file.good(),"legacy fixture write");}
        const auto historical=checkpoints.LoadCheckpoint(*executor.GetModel(),nullptr,"provenance");
        Require(historical&&historical->randomness.model_seed==-1&&historical->randomness.dataloader_seed==-1,
                "historical checkpoints must load with unknown/unset seed provenance");
        {std::ofstream file(metadata_path);file<<metadata.dump(2);Require(file.good(),"restore fixture provenance");}
    }catch(...){error=std::current_exception();}});
    worker.join();plugin::PluginTrainingHookManager::Instance().RemoveByPlugin("model-seed-test");
    if(error)std::rethrow_exception(error);
    Require(result.losses.size()==4&&result.updates.size()==4,"batch reporting/update cadence");
    Require(result.initial!=result.final,"optimizer must update model weights");
    Require(result.output1!=result.output2,"dropout stream must advance");
    Require(result.metrics.randomness.model_seed==seed,"effective seed in metrics");
    return result;
}
}

int RunTrainingRandomnessTests(const std::string& backend){try{
    using namespace cyxwiz;
    Require(backend=="cpu"||backend=="cuda"||backend=="opencl","backend");
    const auto device=backend=="cpu"?DeviceType::CPU:backend=="cuda"?DeviceType::CUDA:DeviceType::OPENCL;
    const auto directory=std::filesystem::current_path()/("model_seed_"+backend);
    std::filesystem::create_directories(directory);
    const ScopedDebugRunRootOverrideForTesting debug_root(directory/"debug_runs");
    test::InstallQualifiedRouteSnapshot();
    for(const auto& value:{"-2","2147483648","1x","","1.5"," 52"}){
        bool rejected=false;try{ParseModelRandomSeed(value);}catch(const std::invalid_argument&){rejected=true;}
        Require(rejected,"invalid model seed must be rejected");}
    Require(ParseModelRandomSeed("-1")==-1&&ParseModelRandomSeed("0")==0&&ParseModelRandomSeed("2147483647")==2147483647,"valid seed boundaries");
    for(const auto& bad:std::vector<nlohmann::json>{
        {{"model_seed",1.5}},{{"model_seed",(std::numeric_limits<uint64_t>::max)()}},
        {{"model_seed",-2}},{{"dataloader_seed","42"}}}) {
        bool rejected=false;try{(void)bad.get<TrainingRandomness>();}catch(const std::exception&){rejected=true;}
        Require(rejected,"malformed checkpoint seed provenance must be rejected");
    }
    const auto a=Run(directory/"a",device,52,42),b=Run(directory/"b",device,52,42);
    Require(a.initial==b.initial&&a.output1==b.output1&&a.output2==b.output2&&a.losses==b.losses&&a.updates==b.updates&&a.final==b.final,"worker TrainingExecutor replay must be exact");
    const auto preview=Run(directory/"preview",device,52,42,true);
    Require(preview.initial==a.initial&&preview.losses==a.losses&&preview.updates==a.updates&&preview.final==a.final,
            "epoch previews must preserve exact dropout RNG, losses and optimizer updates");
    int preview_files=0;
    for(const auto& file:std::filesystem::recursive_directory_iterator(directory/"preview"))
        if(file.path().parent_path().filename()=="generation_previews" && file.path().extension()==".json") ++preview_files;
    Require(preview_files==2,"actual TrainingExecutor must emit previews after both epochs");
    const auto different=Run(directory/"different",device,53,42);
    Require(different.initial!=a.initial,"model seed must change initialization");
    const auto shuffled=Run(directory/"shuffle",device,52,97);
    Require(shuffled.initial==a.initial&&shuffled.output1==a.output1,"DataLoader seed must not change model initialization/dropout start");
    Require(shuffled.losses!=a.losses,"changed DataLoader seed must alter shuffled training");
    const auto unset=Run(directory/"unset",device,-1,42);
    Require(unset.initial==a.initial&&unset.losses==a.losses&&unset.metrics.randomness.generator.empty(),"unset seed must leave the worker RNG unchanged");
    nlohmann::json result={{"backend",backend},{"runs",6},{"epoch_previews_preserve_training",true},{"initial_parameter_values",a.initial.size()},
        {"optimizer_updates_per_run",a.updates.size()},{"exact_worker_replay",true},
        {"dropout_advances",true},{"model_and_data_seed_independent",true},
        {"checkpoint_provenance_roundtrip",true},{"exact_resume_supported",false},
        {"qualification","injected test snapshot; real device execution"}};
    std::ofstream report(directory/"result.json");report<<result.dump(2);Require(report.good(),"report write");
    std::cout<<result.dump(2)<<std::endl;return 0;
}catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<std::endl;return 1;}}
