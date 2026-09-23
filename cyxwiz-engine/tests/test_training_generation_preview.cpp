#include "core/training_generation_preview.h"
#include "core/language_model_generation.h"
#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace {
void Require(bool value,const char* why) { if(!value) throw std::runtime_error(why); }
class PreviewLogits : public cyxwiz::Module {
public:
    bool fail=false;
    int calls=0;
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        ++calls;
        if(fail) throw std::runtime_error("injected generation failure");
        Require(!IsTraining(),"preview must execute in eval mode");
        size_t length=input.Shape()[1];
        std::vector<float> logits(length*6,-10.f);
        logits[(length-1)*6+4]=10.f;
        return cyxwiz::Tensor({1,length,6},logits.data());
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& g) override {return g;}
    std::string GetName() const override {return "PreviewLogits";}
};
}

void TestTrainingGenerationPreview() {
    using namespace cyxwiz;
    Require(!ParseTrainingGenerationPreview({}).enabled,"legacy preview default off");
    Require(!ParseTrainingGenerationPreview({{"generation_preview_enabled","false"},{"generation_preview_every_epochs","bad"}}).enabled,"disabled preserves old behavior");
    for(const auto& value:{"0","-1","2x","10001"}) {
        bool rejected=false;
        try {ParseTrainingGenerationPreview({{"generation_preview_enabled","true"},{"generation_preview_prompts","one"},{"generation_preview_every_epochs",value}});}
        catch(const std::invalid_argument&) {rejected=true;}
        Require(rejected,"invalid cadence must fail");
    }
    auto settings=ParseTrainingGenerationPreview({{"generation_preview_enabled","true"},{"generation_preview_prompts","one\r\ntwo"},{"generation_preview_every_epochs","2"},{"generation_preview_max_new_tokens","2"}});
    Require(!ShouldRunTrainingPreview(settings,0)&&!ShouldRunTrainingPreview(settings,1)&&ShouldRunTrainingPreview(settings,2)&&!ShouldRunTrainingPreview(settings,3),"completed epoch cadence");
    Vocabulary vocab;vocab.SetVocabulary({"[PAD]","[UNK]","[BOS]","[EOS]","one","two"});
    std::ostringstream serialized;Require(vocab.SaveToStream(serialized),"vocabulary serialize");
    settings.vocabulary_artifact=serialized.str();settings.tokenizer_type=0;settings.context=8;
    TrainingGenerationPreview preview(settings,6);
    bool rejected=false;
    try {TrainingGenerationPreview wrong(settings,7);} catch(const std::invalid_argument&){rejected=true;}
    Require(rejected,"mismatched vocabulary rejected before training");
    auto long_prompt=settings;long_prompt.context=2;long_prompt.prompts="one two";
    rejected=false;try {TrainingGenerationPreview wrong(long_prompt,6);} catch(const std::invalid_argument&){rejected=true;}
    Require(rejected,"prompt overflow rejected before training");
    const auto directory=std::filesystem::current_path()/("preview_unit_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    SequentialModel model;model.Add<PreviewLogits>();model.SetTraining(true);
    auto* module=static_cast<PreviewLogits*>(model.GetModule(0));
    Require(preview.Run(model,1,"test",directory.string(),{}),"noncadence returns");
    Require(module->calls==0&&!std::filesystem::exists(directory),"noncadence must not generate/write");
    Require(preview.Run(model,2,"test",directory.string(),{}),"cadence generation");
    Require(module->calls==4&&module->IsTraining(),"bounded generation and training-mode restoration");
    nlohmann::json report;std::ifstream(directory/"epoch_2.json")>>report;
    Require(report["samples"].size()==2&&report["completed_epoch"]==2&&report["weights"]=="current_epoch","artifact provenance");
    Require(report["tokenizer_sha256"].get<std::string>().size()==64,"tokenizer hash");
    module->SetTraining(false);
    int checks=0;
    Require(!preview.Run(model,4,"test",directory.string(),[&]{return ++checks>=3;}),"between-token cancellation");
    Require(!module->IsTraining(),"restore previously eval module after cancellation");
    module->SetTraining(true);module->fail=true;
    rejected=false;try {preview.Run(model,6,"test",directory.string(),{});} catch(const std::runtime_error&){rejected=true;}
    Require(rejected&&module->IsTraining(),"generation exception propagates with restored mode");
    module->fail=false;
    LanguageModelGenerationConfig config;config.should_cancel=[] {return true;};
    const int before=module->calls;
    const auto cancelled=GenerateTokenIdsWithReport(model,{4},config);
    Require(cancelled.stop_reason==LanguageModelGenerationStopReason::UserCancelled&&module->calls==before,"cancel before first forward");
}
