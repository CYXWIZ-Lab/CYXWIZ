#include <iostream>
#include <algorithm>
#include <stdexcept>
static int checks=0;
static void Require(bool value,const char* expression) { ++checks; if(!value) throw std::runtime_error(expression); }
#define REQUIRE(...) Require((__VA_ARGS__), #__VA_ARGS__)
#define REQUIRE_FALSE(...) REQUIRE(!(__VA_ARGS__))
#define INFO(message) std::cout << message << '\n'
#include "core/node_executors/text_tokenizer_operator.h"
#include "core/sequence_arrow_batcher.h"
#include "core/arrow_dataset.h"
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <filesystem>
#include <map>

using namespace cyxwiz;
static std::shared_ptr<arrow::Array> Strings(const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for(const auto& value:values) REQUIRE(builder.Append(value).ok());
    return builder.Finish().ValueOrDie();
}
static std::shared_ptr<arrow::Table> Input() {
    return arrow::Table::Make(arrow::schema({arrow::field("document_id",arrow::utf8()),
        arrow::field("split",arrow::utf8()),arrow::field("text",arrow::utf8()),arrow::field("reference",arrow::utf8())}),
        {Strings({"A","B","empty"}),Strings({"train","train","train"}),Strings({"abcdefg","XY",""}),Strings({"source:1","source:2","source:3"})});
}
static std::map<std::string,std::string> Params() {
    return {{"text_col","text"},{"document_id_col","document_id"},{"split_col","split"},
        {"output_mode","causal_windows"},{"tokenizer_type","3"},{"lowercase","false"},
        {"max_length","3"},{"min_word_freq","1"},{"max_vocab_size","260"}};
}
static std::shared_ptr<arrow::Table> Windows() {
    TextTokenizerOperator op; std::string error;
    REQUIRE(op.Configure(Params(),error));
    auto result=op.Apply(Input()); INFO(result.status().ToString()); REQUIRE(result.ok()); return *result;
}
static TrainingConfiguration Config() {
    TrainingConfiguration cfg;
    cfg.sequence_batch.enabled=true; cfg.sequence_batch.create_causal_lm_targets=true;
    cfg.sequence_batch.create_attention_mask=true; cfg.sequence_batch.max_sequence_length=3;
    cfg.sequence_batch.token_column="token_ids"; cfg.shuffle=false;
    return cfg;
}

void CheckCoverage() {
    auto table=Windows(); REQUIRE(table->num_rows()==4); REQUIRE(table->GetColumnByName("text")==nullptr);
    auto tokens=std::static_pointer_cast<arrow::ListArray>(table->GetColumnByName("token_ids")->chunk(0));
    REQUIRE(tokens->value_type()->id()==arrow::Type::INT64);
    REQUIRE(tokens->value_length(0)==4); REQUIRE(tokens->value_length(1)==4);
    REQUIRE(tokens->value_length(2)==2); REQUIRE(tokens->value_length(3)==3);
    auto ids=std::static_pointer_cast<arrow::Int64Array>(tokens->values());
    REQUIRE(ids->Value(0)==101); REQUIRE(ids->Value(3)==104);
    REQUIRE(ids->Value(4)==104); REQUIRE(ids->Value(7)==107);
    REQUIRE(ids->Value(8)==107); REQUIRE(ids->Value(9)==3);
    REQUIRE(ids->Value(10)==92); REQUIRE(ids->Value(12)==3);
    REQUIRE(table->GetColumnByName("document_id")->GetScalar(2).ValueOrDie()->ToString()=="A");
    REQUIRE(table->GetColumnByName("reference")->GetScalar(3).ValueOrDie()->ToString()=="source:2");
    REQUIRE(table->GetColumnByName("__valid_targets")->GetScalar(2).ValueOrDie()->ToString()=="1");
}

void CheckReloadAndTargets() {
    auto table=Windows(); ArrowDataset data(table,"windows");
    auto path=std::filesystem::temp_directory_path()/"cyxwiz_document_windows_test.parquet";
    REQUIRE(data.ExportParquet(path.string(),true));
    auto loaded=ArrowDataset::FromParquet(path.string(),"reloaded"); REQUIRE(loaded!=nullptr);
    INFO(table->schema()->ToString()); INFO(loaded->GetArrowTable()->schema()->ToString());
    REQUIRE(table->Equals(*loaded->GetArrowTable(),true));
    auto cfg=Config(); auto result=BuildSequenceBatcherFromArrowDataset(loaded,cfg,4);
    INFO(result.error_message); REQUIRE(result.success()); REQUIRE(result.token_vocabulary_size==260);
    auto batch=result.batcher->GetNextSequenceBatch(); REQUIRE(batch.size==4);
    REQUIRE(batch.word_ids.GetDataType()==DataType::Int64); REQUIRE(batch.target_ids.GetDataType()==DataType::Int64);
    const auto* target=static_cast<const Tensor&>(batch.target_ids).Data<int64_t>();
    REQUIRE(target[0]==102); REQUIRE(target[2]==104); REQUIRE(target[3]==105);
    REQUIRE(target[5]==107); REQUIRE(target[6]==3);
    REQUIRE(target[7]==-100); REQUIRE(target[9]==93); REQUIRE(target[10]==3); REQUIRE(target[11]==-100);
    const auto* mask=static_cast<const Tensor&>(batch.attention_mask).Data<int64_t>();
    REQUIRE(mask[8]==0); REQUIRE(mask[11]==1);
    cfg.sequence_batch.max_sequence_length=2;
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(loaded,cfg,4).success());
    cfg=Config(); cfg.sequence_batch.expected_token_vocabulary={"different"};
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(loaded,cfg,4).success());
    auto no_metadata=std::make_shared<ArrowDataset>(table->ReplaceSchemaMetadata(nullptr),"bad");
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(no_metadata,Config(),4).success());
    cfg=Config(); cfg.has_data_split=true; cfg.train_ratio=0.5f; cfg.val_ratio=0.0f;
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(loaded,cfg,4).success());
    cfg.sequence_batch.sentence_id_column="document_id";
    auto grouped=BuildSequenceBatcherFromArrowDataset(loaded,cfg,4);
    REQUIRE(grouped.success()); REQUIRE(grouped.batcher->GetNumSamples()==3);
    grouped.batcher->SetPhase(BatcherPhase::Test); grouped.batcher->Reset();
    REQUIRE(grouped.batcher->GetNumSamples()==1);
    auto value_builder=std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder invalid_ids(arrow::default_memory_pool(),value_builder,
        arrow::list(arrow::field("element",arrow::int64())));
    for(int64_t row=0;row<table->num_rows();++row) {
        auto scalar=std::static_pointer_cast<arrow::ListScalar>(table->GetColumnByName("token_ids")->GetScalar(row).ValueOrDie());
        auto original=std::static_pointer_cast<arrow::Int64Array>(scalar->value);
        REQUIRE(invalid_ids.Append().ok());
        for(int64_t i=0;i<original->length();++i)
            REQUIRE(value_builder->Append(row==0 && i==0 ? 260 : original->Value(i)).ok());
    }
    auto invalid_table=table->SetColumn(table->schema()->GetFieldIndex("token_ids"),
        table->schema()->GetFieldByName("token_ids"),
        std::make_shared<arrow::ChunkedArray>(invalid_ids.Finish().ValueOrDie())).ValueOrDie();
    auto invalid_data=std::make_shared<ArrowDataset>(invalid_table,"out_of_range");
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(invalid_data,Config(),4).success());

    auto legacy_table=arrow::Table::Make(arrow::schema({arrow::field("tokens",arrow::utf8())}),
        {Strings({"alpha beta","beta gamma"})});
    auto legacy_data=std::make_shared<ArrowDataset>(legacy_table,"legacy_strings");
    auto legacy_cfg=Config(); legacy_cfg.sequence_batch.token_column="tokens";
    auto legacy=BuildSequenceBatcherFromArrowDataset(legacy_data,legacy_cfg,2);
    REQUIRE(legacy.success());
    auto beta=std::find(legacy.id_to_label.begin(),legacy.id_to_label.end(),"beta");
    auto gamma=std::find(legacy.id_to_label.begin(),legacy.id_to_label.end(),"gamma");
    REQUIRE(beta!=legacy.id_to_label.end()); REQUIRE(gamma!=legacy.id_to_label.end());
    auto legacy_batch=legacy.batcher->GetNextSequenceBatch(); REQUIRE(legacy_batch.size==2);
    const auto* legacy_targets=static_cast<const Tensor&>(legacy_batch.target_ids).Data<int64_t>();
    REQUIRE(legacy_targets[0]==std::distance(legacy.id_to_label.begin(),beta));
    REQUIRE(legacy_targets[3]==std::distance(legacy.id_to_label.begin(),gamma));
    std::filesystem::remove(path);
}

void CheckInvalidInput() {
    auto p=Params(); TextTokenizerOperator op; std::string error;
    p["document_id_col"]=""; REQUIRE_FALSE(op.Configure(p,error));
    p=Params(); REQUIRE(op.Configure(p,error));
    auto duplicate=Input()->SetColumn(0,arrow::field("document_id",arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(Strings({"same","same","empty"}))).ValueOrDie();
    REQUIRE_FALSE(op.Apply(duplicate).ok());
    auto mixed=Input()->SetColumn(1,arrow::field("split",arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(Strings({"train","test","train"}))).ValueOrDie();
    REQUIRE_FALSE(op.Apply(mixed).ok());
    auto vocabulary=std::filesystem::temp_directory_path()/"cyxwiz_document_window_vocab.txt";
    p["vocab_file"]=vocabulary.string(); p["vocab_build_if_missing"]="true";
    TextTokenizerOperator fit; REQUIRE(fit.Configure(p,error)); REQUIRE(fit.Apply(Input()).ok());
    p["vocab_build_if_missing"]="false";
    TextTokenizerOperator load; REQUIRE(load.Configure(p,error));
    auto validation=load.Apply(mixed); REQUIRE(validation.ok());
    REQUIRE((*validation)->GetColumnByName("split")->GetScalar(3).ValueOrDie()->ToString()=="test");
    std::filesystem::remove(vocabulary);
    // A named vocabulary that is missing must fail as missing, not fall back to
    // fitting and then reject the non-train rows with a misleading message.
    TextTokenizerOperator missing; REQUIRE(missing.Configure(p,error));
    auto missing_result=missing.Apply(mixed); REQUIRE_FALSE(missing_result.ok());
    REQUIRE(missing_result.status().ToString().find("does not exist")!=std::string::npos);
    REQUIRE(missing_result.status().ToString().find("Fit vocabulary")==std::string::npos);
    auto null_text=Input()->SetColumn(2,arrow::field("text",arrow::utf8()),
        std::make_shared<arrow::ChunkedArray>(arrow::MakeArrayOfNull(arrow::utf8(),3).ValueOrDie())).ValueOrDie();
    REQUIRE_FALSE(op.Apply(null_text).ok());
    PipelineOperatorExecutionContext context;
    context.cancellation_requested=[] { return true; }; op.SetExecutionContext(context);
    REQUIRE(op.Apply(Input()).status().IsCancelled());
    context.cancellation_requested={}; context.memory.policy.hard_limit_bytes=1;
    op.SetExecutionContext(context); REQUIRE(op.Apply(Input()).status().IsCapacityError());
}


void CheckExternalRolesAndMetadata() {
    auto all = Windows();
    auto train = std::make_shared<ArrowDataset>(all->Slice(0,3), "train");
    auto dev = std::make_shared<ArrowDataset>(all->Slice(3,1), "dev");
    auto test_table = all->Slice(3,1)->SetColumn(0, all->field(0),
        std::make_shared<arrow::ChunkedArray>(Strings({"C"}))).ValueOrDie();
    auto test = std::make_shared<ArrowDataset>(test_table, "test");
    auto cfg = Config(); cfg.has_data_split=true; cfg.sequence_batch.sentence_id_column="document_id";
    cfg.train_ratio=1; cfg.val_ratio=0; cfg.test_ratio=0; cfg.shuffle=true; cfg.drop_last=true;
    auto built = BuildSequenceBatcherFromArrowDataset(train,cfg,2,dev,test);
    INFO(built.error_message); REQUIRE(built.success()); REQUIRE(built.sample_count==5);
    REQUIRE(built.batcher->GetNumSamples()==3); REQUIRE(built.batcher->GetNumBatches()==1);
    ApplySequenceBatcherBuildResultToTrainingConfig(built,cfg);
    REQUIRE(cfg.sequence_batch.tokenizer_vocabulary_artifact==*all->schema()->metadata()->Get("cyxwiz.token_windows.vocabulary"));
    REQUIRE(!cfg.sequence_batch.tokenizer_config_json.empty());
    for(auto phase : {BatcherPhase::Val, BatcherPhase::Test}) {
        built.batcher->SetPhase(phase); built.batcher->Reset();
        REQUIRE(built.batcher->GetNumSamples()==1); REQUIRE(built.batcher->GetNumBatches()==1);
        auto batch=built.batcher->GetNextSequenceBatch(); REQUIRE(batch.size==1);
        const auto* target=static_cast<const Tensor&>(batch.target_ids).Data<int64_t>();
        REQUIRE(target[0]==93 && target[1]==3 && target[2]==-100);
        REQUIRE(built.batcher->IsEpochComplete());
    }
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(train,cfg,2,train,test).success());
    auto missing=std::make_shared<ArrowDataset>(test_table->ReplaceSchemaMetadata(nullptr),"missing");
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(train,cfg,2,missing).success());
    auto metadata=test_table->schema()->metadata()->Copy(); REQUIRE(metadata->Set("cyxwiz.token_windows.context","99").ok());
    auto mismatch=std::make_shared<ArrowDataset>(test_table->ReplaceSchemaMetadata(metadata),"mismatch");
    REQUIRE_FALSE(BuildSequenceBatcherFromArrowDataset(train,cfg,2,nullptr,mismatch).success());
    // An external Test does not suppress a configured, derived Dev role from Train.
    cfg=Config();cfg.has_data_split=true;cfg.sequence_batch.sentence_id_column="document_id";
    cfg.train_ratio=.5f;cfg.val_ratio=.5f;cfg.test_ratio=0;
    auto both_train=std::make_shared<ArrowDataset>(all,"two_groups");
    auto external_test=BuildSequenceBatcherFromArrowDataset(both_train,cfg,2,nullptr,test);
    REQUIRE(external_test.success());
    REQUIRE(external_test.batcher->GetNumSamples()==3);
    external_test.batcher->SetPhase(BatcherPhase::Val);REQUIRE(external_test.batcher->GetNumSamples()==1);
    external_test.batcher->SetPhase(BatcherPhase::Test);REQUIRE(external_test.batcher->GetNumSamples()==1);
}

int main() {
    try {
        CheckCoverage(); CheckReloadAndTargets(); CheckInvalidInput(); CheckExternalRolesAndMetadata();
        std::cout << "Document token windows passed: " << checks << " checks\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n"; return 1;
    }
}
