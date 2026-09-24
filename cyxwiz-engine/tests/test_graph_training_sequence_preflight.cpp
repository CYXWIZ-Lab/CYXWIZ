#include "../src/core/sequence_arrow_batcher.h"
#include "../src/core/node_executors/text_tokenizer_operator.h"
#include <arrow/util/key_value_metadata.h>
#include "../src/gui/graph_training_launcher.h"

#include "../src/core/arrow_dataset.h"
#include "../src/core/async_task_manager.h"
#include "../src/core/data_registry.h"

#include <arrow/api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kDatasetName = "gui_sequence_preflight_runtime";
constexpr const char* kRoleTrainDataset = "gui_role_schema_train";
constexpr const char* kRoleMissingLabelDataset = "gui_role_schema_missing_label";
constexpr const char* kRoleMismatchedFeatureDataset = "gui_role_schema_mismatched_feature";
constexpr const char* kRoleOverlappingIdDataset = "gui_role_schema_overlapping_id";
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishDoubleArray(
    const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    for (double value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Array> FinishInt64Array(
    const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    for (int64_t value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeRoleTable(
    const std::string& label_name,
    bool mismatched_second_feature_type = false) {
    auto second_feature = mismatched_second_feature_type
        ? arrow::field("sensor_b", arrow::int64())
        : arrow::field("sensor_b", arrow::float64());
    auto schema = arrow::schema({
        arrow::field("sample_id", arrow::int64()),
        arrow::field("sensor_a", arrow::float64()),
        second_feature,
        arrow::field(label_name, arrow::int64()),
    });

    std::vector<std::shared_ptr<arrow::Array>> arrays = {
        FinishInt64Array({100, 200, 300}),
        FinishDoubleArray({1.0, 2.0, 3.0}),
        mismatched_second_feature_type
            ? FinishInt64Array({10, 20, 30})
            : FinishDoubleArray({10.0, 20.0, 30.0}),
        FinishInt64Array({0, 1, 0}),
    };
    return arrow::Table::Make(schema, arrays, 3);
}
std::shared_ptr<arrow::Table> MakeRoleTableWithSampleIds(
    const std::vector<int64_t>& sample_ids) {
    auto schema = arrow::schema({
        arrow::field("sample_id", arrow::int64()),
        arrow::field("sensor_a", arrow::float64()),
        arrow::field("sensor_b", arrow::float64()),
        arrow::field("label", arrow::int64()),
    });

    return arrow::Table::Make(
        schema,
        {FinishInt64Array(sample_ids),
         FinishDoubleArray({1.0, 2.0, 3.0}),
         FinishDoubleArray({10.0, 20.0, 30.0}),
         FinishInt64Array({0, 1, 0})},
        3);
}

std::shared_ptr<arrow::Table> MakeSequenceTable() {
    auto schema = arrow::schema({
        arrow::field("tokens", arrow::utf8()),
        arrow::field("pos_tags", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema,
                              {FinishStringArray({"London", "wins"}),
                               FinishStringArray({"NNP", "VBZ"}),
                               FinishStringArray({"B-geo", "O"})},
                              2);
}

gui::MLNode MakeDataInputNode() {
    gui::MLNode node;
    node.id = 1;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = "Sequence Data";
    node.parameters = {
        {"dataset_name", kDatasetName},
        {"data_loaded", "true"},
        {"file_category", "ner"},
    };
    return node;
}

gui::MLNode MakeRoleDataInputNode(
    int id,
    const std::string& name,
    const std::string& dataset_name,
    const std::string& role = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = gui::NodeType::DataInput;
    node.category = gui::NodeCategory::DataPipeline;
    node.name = name;
    node.parameters = {
        {"dataset_name", dataset_name},
        {"data_loaded", "true"},
        {"file_category", "tabular"},
        {"label_column", "label"},
    };
    if (!role.empty()) {
        node.parameters["dataset_role"] = role;
    }
    return node;
}

cyxwiz::TrainingConfiguration MakeRoleConfig(
    const std::string& train_dataset,
    const std::string& test_dataset) {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = train_dataset;
    config.data_source_node_id = 10;
    config.batch_size = 2;
    config.epochs = 1;
    config.dataset_roles.train.dataset_name = train_dataset;
    config.dataset_roles.train.source_node_id = 10;
    config.dataset_roles.test.dataset_name = test_dataset;
    config.dataset_roles.test.source_node_id = 11;
    config.dataset_roles.test.externally_supplied = true;
    return config;
}

bool WaitFor(const std::function<bool()>& predicate,
             std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

bool HasActiveTaskNamed(const std::string& name) {
    const auto tasks = cyxwiz::AsyncTaskManager::Instance().GetActiveTasks();
    return std::any_of(
        tasks.begin(), tasks.end(), [&name](const cyxwiz::TaskInfo& task) {
            return task.name == name;
        });
}

std::string LatestFailedTaskError() {
    for (const auto& task :
         cyxwiz::AsyncTaskManager::Instance().GetRecentTasks(10)) {
        if (task.state == cyxwiz::TaskState::Failed) {
            return task.error_message;
        }
    }
    return {};
}
cyxwiz::TrainingConfiguration MakeSequenceConfig() {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = kDatasetName;
    config.batch_size = 2;
    config.epochs = 1;
    config.sequence_batch.enabled = true;
    config.sequence_batch.token_column = "tokens";
    config.sequence_batch.pos_column = "pos_tags";
    config.sequence_batch.tag_column = "ner_tags";
    config.sequence_batch.create_attention_mask = true;
    config.sequence_batch.max_sequence_length = 8;
    return config;
}


void CheckCausalExternalRoleLaunch(cyxwiz::DataRegistry& registry) {
    auto documents=arrow::Table::Make(arrow::schema({arrow::field("document_id",arrow::utf8()),
        arrow::field("text",arrow::utf8()),arrow::field("split",arrow::utf8())}),
        {FinishStringArray({"A","B","C"}),FinishStringArray({"ab","cd","xy"}),
         FinishStringArray({"train","train","train"})});
    cyxwiz::TextTokenizerOperator tokenizer;std::string error;
    Check(tokenizer.Configure({{"text_col","text"},{"document_id_col","document_id"},
        {"split_col","split"},{"output_mode","causal_windows"},{"tokenizer_type","3"},{"lowercase","false"},
        {"max_length","3"},{"max_vocab_size","260"}},error),error);
    auto tokenized=tokenizer.Apply(documents);Check(tokenized.ok(),tokenized.status().ToString());
    auto windows=*tokenized;Check(windows->num_rows()==3,"three document windows expected");
    const std::string train="causal_role_train",dev="causal_role_dev";
    Check(registry.RegisterArrowTable(windows->Slice(0,2),train)!=nullptr,"register token-window Train");
    Check(registry.RegisterArrowTable(windows->Slice(2,1),dev)!=nullptr,"register token-window Dev");
    auto config=MakeRoleConfig(train,"");config.dataset_roles.test={};
    config.dataset_roles.dev.dataset_name=dev;config.dataset_roles.dev.source_node_id=11;
    config.dataset_roles.dev.externally_supplied=true;
    config.has_data_split=true;config.train_ratio=1;config.val_ratio=0;config.test_ratio=0;
    config.sequence_batch.enabled=true;config.sequence_batch.create_causal_lm_targets=true;
    config.sequence_batch.token_column="token_ids";config.sequence_batch.sentence_id_column="document_id";
    config.sequence_batch.max_sequence_length=3;
    std::vector<gui::MLNode> nodes={MakeRoleDataInputNode(10,"Train",train),MakeRoleDataInputNode(11,"Dev",dev)};
    for(auto& node:nodes) node.parameters["label_column"]="";
    std::atomic<bool> prepared{false};
    auto dispatch=[&](cyxwiz::TrainingConfiguration actual,const std::string& name,const std::string&,
        int,int batch,std::weak_ptr<cyxwiz::TrainingPlotPanel>,std::function<void(bool)>) {
        auto built=cyxwiz::BuildSequenceBatcherFromArrowDataset(registry.GetArrowDataset(name),actual,batch,
            registry.GetArrowDataset(actual.dataset_roles.dev.dataset_name));
        if(!built.success()) throw std::runtime_error(built.error_message);
        Check(built.batcher->GetNumSamples()==2,"two training windows reach the batcher");
        built.batcher->SetPhase(cyxwiz::BatcherPhase::Val);built.batcher->Reset();
        Check(built.batcher->GetNumSamples()==1,"one supplied validation window is preserved");
        auto payload=built.batcher->GetNextSequenceBatch();
        const auto* target=static_cast<const cyxwiz::Tensor&>(payload.target_ids).Data<int64_t>();
        Check(target[0]==125 && target[1]==3 && target[2]==-100,"generated targets use the validation token IDs and EOS");
        prepared.store(true);return true;
    };
    auto result=gui::StartGraphTrainingFromCompiledConfig(nodes,{},config,registry,{},[](bool){},dispatch);
    Check(result.started,"label-free causal role should queue: "+result.error_message);
    Check(WaitFor([&]{return prepared.load();},std::chrono::seconds(10)),"causal role must pass BOTH launcher preflights and construct its batcher");
    Check(WaitFor([]{return !HasActiveTaskNamed("Prepare graph training");},std::chrono::seconds(5)),"causal preparation completes");
    // A sequence-specific preflight must still reject missing token columns.
    registry.UnregisterTabularDataset(dev);
    Check(registry.RegisterArrowTable(MakeRoleTable("label"),dev)!=nullptr,"register invalid Dev");
    prepared.store(false);
    result=gui::StartGraphTrainingFromCompiledConfig(nodes,{},config,registry,{},[](bool){},dispatch);
    Check(!result.started && !prepared.load(),"missing Dev tokens must block before dispatch");
    Check(result.error_message.find("token_ids")!=std::string::npos,"sequence error must name missing tokens");
    registry.UnregisterTabularDataset(train);registry.UnregisterTabularDataset(dev);
}

void CheckLaunchReadinessAtCompile(cyxwiz::DataRegistry& registry) {
    // Compile must report what Train would reject, before a run starts.
    auto documents=arrow::Table::Make(arrow::schema({arrow::field("document_id",arrow::utf8()),
        arrow::field("text",arrow::utf8()),arrow::field("split",arrow::utf8())}),
        {FinishStringArray({"A","B","C"}),FinishStringArray({"ab","cd","xy"}),
         FinishStringArray({"train","train","train"})});
    cyxwiz::TextTokenizerOperator tokenizer;std::string error;
    Check(tokenizer.Configure({{"text_col","text"},{"document_id_col","document_id"},
        {"split_col","split"},{"output_mode","causal_windows"},{"tokenizer_type","3"},{"lowercase","false"},
        {"max_length","3"},{"max_vocab_size","260"}},error),error);
    auto windows=*tokenizer.Apply(documents);
    const std::string train="readiness_train",dev="readiness_dev";
    Check(registry.RegisterArrowTable(windows->Slice(0,2),train)!=nullptr,"register readiness Train");
    Check(registry.RegisterArrowTable(windows->Slice(2,1),dev)!=nullptr,"register readiness Dev");
    auto config=MakeRoleConfig(train,"");config.dataset_roles.test={};
    config.dataset_roles.train.source_node_id=20;
    config.dataset_roles.dev.dataset_name=dev;config.dataset_roles.dev.source_node_id=21;
    config.has_data_split=true;
    config.sequence_batch.enabled=true;config.sequence_batch.create_causal_lm_targets=true;
    config.sequence_batch.token_column="token_ids";config.sequence_batch.sentence_id_column="document_id";
    config.layers.emplace_back();
    std::vector<gui::MLNode> nodes={MakeRoleDataInputNode(20,"Schedule",train),MakeRoleDataInputNode(21,"Validation",dev)};
    cyxwiz::RequestedRouteTrainingReadiness qualified;
    qualified.route_available=true;qualified.authorized=true;
    const auto count=[](const std::vector<cyxwiz::ValidationIssue>& issues,cyxwiz::IssueLevel level,const std::string& text){
        return std::count_if(issues.begin(),issues.end(),[&](const auto& i){
            return i.level==level && i.message.find(text)!=std::string::npos;});
    };

    auto issues=gui::CheckGraphLaunchReadiness(nodes,config,registry,&qualified);
    Check(issues.empty(),"matching columns, a qualified route and no row limit report nothing");

    // The Berean case: the training windows name their ID column differently.
    registry.UnregisterTabularDataset(train);
    auto renamed=*windows->Slice(0,2)->RenameColumns([&]{
        auto names=windows->ColumnNames();
        for(auto& name:names) if(name=="document_id") name="schedule_document_id";
        return names;}());
    Check(registry.RegisterArrowTable(renamed,train)!=nullptr,"register renamed Train");
    issues=gui::CheckGraphLaunchReadiness(nodes,config,registry,&qualified);
    Check(count(issues,cyxwiz::IssueLevel::Error,"sentence id column 'document_id'")==1 &&
          issues.front().node_id==20 && issues.front().message.rfind("Training data:",0)==0,
          "Compile names the training input missing the sentence id column");
    registry.UnregisterTabularDataset(train);
    Check(registry.RegisterArrowTable(windows->Slice(0,2),train)!=nullptr,"re-register Train");

    // Device route not qualified: error when fallback cannot help, warning otherwise.
    cyxwiz::RequestedRouteTrainingReadiness unqualified;
    unqualified.route_available=true;unqualified.authorized=false;unqualified.route_name="GTX 1050 Ti";
    unqualified.message="No isolated route qualification snapshot is installed";
    config.forbid_native_cpu_fallback=true;
    issues=gui::CheckGraphLaunchReadiness(nodes,config,registry,&unqualified);
    Check(count(issues,cyxwiz::IssueLevel::Error,"Verify Selected")==1 &&
          count(issues,cyxwiz::IssueLevel::Error,"No isolated route qualification")==1,
          "an unqualified device with fallback forbidden is a Compile error with the action");
    config.forbid_native_cpu_fallback=false;unqualified.cpu_recovery_qualified=true;
    issues=gui::CheckGraphLaunchReadiness(nodes,config,registry,&unqualified);
    Check(count(issues,cyxwiz::IssueLevel::Warning,"fall back to ArrayFire CPU")==1 &&
          count(issues,cyxwiz::IssueLevel::Error,"")==0,"a qualified CPU fallback is a warning");
    auto data_only=config;data_only.layers.clear();
    Check(gui::CheckGraphLaunchReadiness(nodes,data_only,registry,&unqualified).empty(),
          "graphs without a model are not judged on the training device");

    // A row limit the training path does not apply.
    nodes[0].parameters["max_rows"]="1";
    issues=gui::CheckGraphLaunchReadiness(nodes,config,registry,&qualified);
    Check(count(issues,cyxwiz::IssueLevel::Warning,"max_rows=1 is not applied by training")==1,
          "a max_rows limit smaller than the loaded data is reported");
    nodes[0].parameters["max_rows"]="5";
    Check(gui::CheckGraphLaunchReadiness(nodes,config,registry,&qualified).empty(),
          "a limit the loaded data already satisfies is not reported");
    registry.UnregisterTabularDataset(train);registry.UnregisterTabularDataset(dev);
    std::cout << "Launch readiness at Compile passed\n";
}

} // namespace

int main() {
    const auto project_root =
        std::filesystem::temp_directory_path() / "cyxwiz_project_cache_root";
    const auto project_cache =
        gui::GraphMaterializationCacheConfig(project_root);
    Check(project_cache.cache_root == project_root.lexically_normal(),
          "graph materialization cache should use the active project root");
    Check(cyxwiz::MaterializationCacheEntryDirectory(project_cache, "key") ==
              project_root / "cache" / "materialized" / "key",
          "active-project cache entries should live under cache/materialized");
    const auto standalone_cache = gui::GraphMaterializationCacheConfig();
    Check(standalone_cache.cache_root.filename() == ".cyxwiz",
          "standalone graph cache should retain the runtime-local fallback");

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kRoleTrainDataset);
    registry.UnregisterTabularDataset(kRoleMissingLabelDataset);
    registry.UnregisterTabularDataset(kRoleMismatchedFeatureDataset);
    registry.UnregisterTabularDataset(kRoleOverlappingIdDataset);
    Check(registry.RegisterArrowTable(MakeSequenceTable(), kDatasetName) !=
              nullptr,
          "sequence preflight dataset should register");
    Check(registry.RegisterArrowTable(MakeRoleTable("label"), kRoleTrainDataset) !=
              nullptr,
          "role schema training dataset should register");
    Check(registry.RegisterArrowTable(MakeRoleTable("target"),
                                      kRoleMissingLabelDataset) != nullptr,
          "missing-label role dataset should register");
    Check(registry.RegisterArrowTable(
              MakeRoleTable("label", true),
              kRoleMismatchedFeatureDataset) != nullptr,
          "mismatched-feature role dataset should register");
    Check(registry.RegisterArrowTable(
              MakeRoleTableWithSampleIds({400, 200, 500}),
              kRoleOverlappingIdDataset) != nullptr,
          "overlapping-id external role dataset should register");


    const std::vector<gui::MLNode> nodes = {MakeDataInputNode()};
    const std::vector<gui::NodeLink> links;

    std::atomic<bool> dispatch_called{false};
    auto valid_config = MakeSequenceConfig();
    auto valid_result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(valid_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [](bool) {},
        [&](cyxwiz::TrainingConfiguration,
            const std::string&,
            const std::string&,
            int,
            int,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>,
            std::function<void(bool)>) {
            dispatch_called.store(true);
            return true;
        });
    Check(valid_result.started, valid_result.error_message);
    Check(WaitFor([&] { return dispatch_called.load(); },
                  std::chrono::seconds(5)),
          "valid sequence preflight should call dispatch");
    Check(WaitFor(
              [] { return !HasActiveTaskNamed("Prepare graph training"); },
              std::chrono::seconds(2)),
          "valid sequence preparation should reach a terminal state");

    {
        const std::vector<gui::MLNode> role_nodes = {
            MakeRoleDataInputNode(10, "Training Data", kRoleTrainDataset),
            MakeRoleDataInputNode(11, "External Test Data",
                                  kRoleMissingLabelDataset, "test"),
        };
        bool role_dispatch_called = false;
        auto role_result = gui::StartGraphTrainingFromCompiledConfig(
            role_nodes,
            links,
            MakeRoleConfig(kRoleTrainDataset, kRoleMissingLabelDataset),
            registry,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
            [](bool) {},
            [&](cyxwiz::TrainingConfiguration,
                const std::string&,
                const std::string&,
                int,
                int,
                std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                std::function<void(bool)>) {
                role_dispatch_called = true;
                return true;
            });
        Check(!role_result.started,
              "missing external role label should block synchronously");
        Check(!role_dispatch_called,
              "missing external role label should not call dispatch");
        Check(role_result.status_title == "Test dataset label unavailable",
              "missing external role label should name Test role: " +
                  role_result.status_title);
        Check(role_result.error_message.find(kRoleMissingLabelDataset) !=
                  std::string::npos,
              "missing external role label should name dataset: " +
                  role_result.error_message);
        Check(role_result.error_message.find("label") != std::string::npos,
              "missing external role label should name label column: " +
                  role_result.error_message);
    }

    {
        const std::vector<gui::MLNode> role_nodes = {
            MakeRoleDataInputNode(10, "Training Data", kRoleTrainDataset),
            MakeRoleDataInputNode(11, "External Test Data",
                                  kRoleMismatchedFeatureDataset, "test"),
        };
        bool role_dispatch_called = false;
        auto role_result = gui::StartGraphTrainingFromCompiledConfig(
            role_nodes,
            links,
            MakeRoleConfig(kRoleTrainDataset, kRoleMismatchedFeatureDataset),
            registry,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
            [](bool) {},
            [&](cyxwiz::TrainingConfiguration,
                const std::string&,
                const std::string&,
                int,
                int,
                std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                std::function<void(bool)>) {
                role_dispatch_called = true;
                return true;
            });
        Check(!role_result.started,
              "mismatched external role feature should block synchronously");
        Check(!role_dispatch_called,
              "mismatched external role feature should not call dispatch");
        Check(role_result.status_title == "Test dataset schema mismatch",
              "mismatched external role feature should name schema mismatch: " +
                  role_result.status_title);
        Check(role_result.error_message.find("sensor_b") != std::string::npos,
              "mismatched external role feature should name column: " +
                  role_result.error_message);
        Check(role_result.error_message.find(kRoleMismatchedFeatureDataset) !=
                  std::string::npos,
              "mismatched external role feature should name dataset: " +
                  role_result.error_message);
    }
    {
        const std::vector<gui::MLNode> role_nodes = {
            MakeRoleDataInputNode(10, "Training Data", kRoleTrainDataset),
            MakeRoleDataInputNode(11, "External Test Data",
                                  kRoleOverlappingIdDataset, "test"),
        };
        bool role_dispatch_called = false;
        auto role_result = gui::StartGraphTrainingFromCompiledConfig(
            role_nodes,
            links,
            MakeRoleConfig(kRoleTrainDataset, kRoleOverlappingIdDataset),
            registry,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
            [](bool) {},
            [&](cyxwiz::TrainingConfiguration,
                const std::string&,
                const std::string&,
                int,
                int,
                std::weak_ptr<cyxwiz::TrainingPlotPanel>,
                std::function<void(bool)>) {
                role_dispatch_called = true;
                return true;
            });
        Check(!role_result.started,
              "overlapping external role id should block synchronously");
        Check(!role_dispatch_called,
              "overlapping external role id should not call dispatch");
        Check(role_result.status_title == "Test dataset leakage detected",
              "overlapping external role id should name leakage: " +
                  role_result.status_title);
        Check(role_result.error_message.find("sample_id") != std::string::npos,
              "overlapping external role id should name id column: " +
                  role_result.error_message);
        Check(role_result.error_message.find("200") != std::string::npos,
              "overlapping external role id should name overlapping value: " +
                  role_result.error_message);
        Check(role_result.error_message.find(kRoleOverlappingIdDataset) !=
                  std::string::npos,
              "overlapping external role id should name dataset: " +
                  role_result.error_message);
    }
    std::atomic<bool> missing_dispatch_called{false};
    std::atomic<bool> missing_preparation_failed{false};
    auto missing_config = MakeSequenceConfig();
    missing_config.sequence_batch.tag_column = "missing_ner_tags";
    auto missing_result = gui::StartGraphTrainingFromCompiledConfig(
        nodes,
        links,
        std::move(missing_config),
        registry,
        std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
        [&](bool preparing) {
            if (!preparing) {
                missing_preparation_failed.store(true);
            }
        },
        [&](cyxwiz::TrainingConfiguration,
            const std::string&,
            const std::string&,
            int,
            int,
            std::weak_ptr<cyxwiz::TrainingPlotPanel>,
            std::function<void(bool)>) {
            missing_dispatch_called.store(true);
            return true;
        });
    Check(missing_result.started, missing_result.error_message);
    Check(WaitFor([&] { return missing_preparation_failed.load(); },
                  std::chrono::seconds(5)),
          "missing sequence column should fail async preparation");
    Check(!missing_dispatch_called.load(),
          "missing sequence column should not call dispatch");

    const std::string missing_error = LatestFailedTaskError();
    Check(missing_error.find("tag column 'missing_ner_tags'") !=
              std::string::npos,
          "missing sequence column error should name the missing tag column: " +
              missing_error);
    Check(missing_error.find(kDatasetName) != std::string::npos,
          "missing sequence column error should name the dataset: " +
              missing_error);
    Check(missing_result.status_title == "Training launch queued",
          "missing sequence column should initially expose queued status: " +
              missing_result.status_title);

    registry.UnregisterTabularDataset(kDatasetName);
    registry.UnregisterTabularDataset(kRoleTrainDataset);
    registry.UnregisterTabularDataset(kRoleMissingLabelDataset);
    registry.UnregisterTabularDataset(kRoleMismatchedFeatureDataset);
    registry.UnregisterTabularDataset(kRoleOverlappingIdDataset);
    CheckCausalExternalRoleLaunch(registry);
    CheckLaunchReadinessAtCompile(registry);
    cyxwiz::AsyncTaskManager::Instance().Shutdown();
    std::cout << "Graph training sequence preflight test passed\n";
    return 0;
}
