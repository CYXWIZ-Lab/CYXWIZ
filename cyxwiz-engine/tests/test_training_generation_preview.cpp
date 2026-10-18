#include "core/generation_preview_presentation.h"
#include "core/training_generation_preview.h"
#include "core/language_model_generation.h"
#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace {
void Require(bool value,const char* why) {
    if(value) return;
    std::cerr<<"Generation preview test failed: "<<why<<std::endl;
    throw std::runtime_error(why);
}

template <typename F>
bool Rejects(F&& f) {
    try { f(); } catch(const std::invalid_argument&) { return true; }
    return false;
}

// Always predicts token 4 ("one"), or the scripted sequence when set.
class PreviewLogits : public cyxwiz::Module {
public:
    bool fail=false;
    int calls=0;
    std::vector<int> script;  // Next tokens in order; empty = always 4.
    bool soft=false;          // Token 4 at 2.0, token 5 at 1.5: close enough for a penalty to flip.
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        ++calls;
        if(fail) throw std::runtime_error("injected generation failure");
        Require(!IsTraining(),"preview must execute in eval mode");
        size_t length=input.Shape()[1];
        std::vector<float> logits(length*6,-10.f);
        const int next=script.empty()?4:script[std::min(script.size()-1,static_cast<size_t>(produced))];
        ++produced;
        if(soft) { logits[(length-1)*6+4]=2.f; logits[(length-1)*6+5]=1.5f; }
        else logits[(length-1)*6+next]=10.f;
        return cyxwiz::Tensor({1,length,6},logits.data());
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& g) override {return g;}
    std::string GetName() const override {return "PreviewLogits";}
    int produced=0;
};

std::map<std::string,std::string> Params(std::map<std::string,std::string> extra) {
    std::map<std::string,std::string> params={{"generation_preview_enabled","true"},
        {"generation_preview_prompt_set",R"([{"text":"one"},{"text":"two"}])"}};
    for(auto& [k,v]:extra) params[k]=v;
    return params;
}
}

void TestGenerationPreviewSettings() {
    using namespace cyxwiz;
    Require(!ParseTrainingGenerationPreview({}).enabled,"default off");
    // A disabled preview ignores everything, including keys from older Engines.
    Require(!ParseTrainingGenerationPreview({{"generation_preview_enabled","false"},
        {"generation_preview_every_epochs","bad"},{"generation_preview_prompts","x"}}).enabled,"disabled ignores the rest");
    Require(Rejects([]{ParseTrainingGenerationPreview(Params({{"generation_preview_prompts","one"}}));}),
            "an enabled preview with the old prompt key is refused");
    for(const auto& value:{"0","-1","2x","1000001"})
        Require(Rejects([&]{ParseTrainingGenerationPreview(Params({{"generation_preview_every",value}}));}),"invalid cadence");
    Require(Rejects([]{ParseTrainingGenerationPreview(Params({{"generation_preview_max_new_tokens","1025"}}));}),"token cap");
    Require(Rejects([]{ParseTrainingGenerationPreview(Params({{"generation_preview_prompt_set","[]"}}));}),"no prompts");
    Require(Rejects([]{ParseTrainingGenerationPreview(Params({{"generation_preview_prompt_set",
        R"([{"text":"a","check":{"kind":"regex","value":"("}}])"}}));}),"bad regex");
    Require(Rejects([]{ParseTrainingGenerationPreview(Params({{"generation_preview_decoding","greedy"},
        {"generation_preview_samples_per_prompt","2"}}));}),"greedy gives one sample");

    // Round trip through the stored parameters.
    TrainingGenerationPreviewSettings s;
    s.enabled=true; s.mode=GenerationPreviewMode::Chat; s.chat_template=GenerationPreviewTemplate::RoleLines;
    s.system_prompt="Answer using the supplied evidence."; s.label_assistant="Berean:";
    s.stop_texts={"</answer>"}; s.max_new_tokens=128; s.decoding=GenerationPreviewDecoding::Sampling;
    s.temperature=0.7f; s.top_k=20; s.top_p=0.9f; s.seed=7; s.samples_per_prompt=2; s.repetition_penalty=1.1f;
    s.every=500; s.cadence=GenerationPreviewCadence::Steps; s.baseline=true; s.after_last=false;
    s.weights=GenerationPreviewWeights::Best;
    s.prompts={{"<question>Q</question>",{GenerationPreviewCheckKind::Contains,"Psalm 34:13"}},{"Second\nline",{}}};
    const auto back=ParseTrainingGenerationPreview(SerializeTrainingGenerationPreview(s));
    Require(back.mode==s.mode&&back.chat_template==s.chat_template&&back.system_prompt==s.system_prompt&&
            back.label_assistant=="Berean:"&&back.stop_texts==s.stop_texts&&back.max_new_tokens==128&&
            back.decoding==s.decoding&&back.temperature==0.7f&&back.top_k==20&&back.top_p==0.9f&&back.seed==7&&
            back.samples_per_prompt==2&&back.repetition_penalty==1.1f&&back.every==500&&back.cadence==s.cadence&&
            back.baseline&&!back.after_last&&back.weights==s.weights&&back.prompts.size()==2&&
            back.prompts[0].check.kind==GenerationPreviewCheckKind::Contains&&
            back.prompts[0].check.value=="Psalm 34:13"&&back.prompts[1].text=="Second\nline","round trip");

    // Role lines render the format the SFT data was trained on.
    const auto rendered=RenderGenerationPreviewPrompt(s,"<question>Q</question>");
    Require(rendered.text=="System: Answer using the supplied evidence.\nUser: <question>Q</question>\nBerean:\n",
            "role lines render");
    size_t from_template=0;
    for(const auto& segment:rendered.segments) from_template+=segment.from_template?segment.text.size():0;
    Require(from_template==std::string("System: \nUser: \nBerean:\n").size(),"template segments are marked");
    auto chatml=s; chatml.chat_template=GenerationPreviewTemplate::ChatML; chatml.system_prompt.clear();
    Require(RenderGenerationPreviewPrompt(chatml,"hi").text==
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n","chatml render");
    Require(GenerationPreviewStopTexts(chatml)==std::vector<std::string>({"<|im_end|>","</answer>"}),"chatml stops");
    Require(GenerationPreviewStopTexts(s)==std::vector<std::string>({"\nUser:","</answer>"}),"role-line stops");
    auto base=s; base.mode=GenerationPreviewMode::Continuation;
    Require(RenderGenerationPreviewPrompt(base,"In the beginning").text=="In the beginning","continuation is raw");
    Require(GenerationPreviewTemplateStopTexts(base).empty(),"continuation has no template stops");

    // Cadence.
    TrainingGenerationPreviewSettings c; c.enabled=true; c.every=2; c.after_last=true;
    Require(!ShouldRunPreviewAfterEpoch(c,1,5)&&ShouldRunPreviewAfterEpoch(c,2,5)&&!ShouldRunPreviewAfterEpoch(c,3,5)&&
            ShouldRunPreviewAfterEpoch(c,5,5),"every 2 epochs and the last");
    c.after_last=false;
    Require(!ShouldRunPreviewAfterEpoch(c,5,5),"last epoch off");
    c.cadence=GenerationPreviewCadence::Steps; c.every=100;
    Require(!ShouldRunPreviewAfterEpoch(c,2,5)&&ShouldRunPreviewAfterStep(c,200)&&!ShouldRunPreviewAfterStep(c,150),
            "step cadence");
    Require(SuggestedGenerationPreviewMode("response_only")==GenerationPreviewMode::Chat&&
            SuggestedGenerationPreviewMode("all_tokens")==GenerationPreviewMode::Continuation,"suggested mode");
}

void TestGenerationPreviewPresentation() {
    using namespace cyxwiz;
    // Real SFT output (Berean run train-1791984271862, epoch 4) as sample data.
    const std::string looping="<explanation>Psalm 34:1 says to keep your heart from being a burden to your mouth."
        "</explanation>\n<answer>You can be a burden to your heart, a burden to your mouth.";
    Require(DetectRepeatedPhrase(looping)=="a burden to your mouth","repeated phrase");
    Require(DetectRepeatedPhrase("Deuteronomy 25:2 says to be a shepherd, and the shepherds").empty(),"no repeat");
    GenerationPreviewCheck contains{GenerationPreviewCheckKind::Contains,"Psalm 34:13"};
    Require(EvaluateGenerationPreviewCheck(contains,looping,"max_tokens")==GenerationPreviewCheckResult::Failed,
            "34:1 is not 34:13");
    Require(EvaluateGenerationPreviewCheck({GenerationPreviewCheckKind::Regex,"Psalm \\d+:\\d+"},looping,"eos")==
            GenerationPreviewCheckResult::Passed,"regex");
    Require(EvaluateGenerationPreviewCheck({GenerationPreviewCheckKind::EndsWithStop,""},looping,"stop_text")==
            GenerationPreviewCheckResult::Passed,"ends with stop");
    Require(GenerationPreviewExcerpt("a\n\nb  c",10)=="a b c","excerpt joins lines");
    Require(GenerationPreviewExcerpt("abcdefghijkl",5)=="abcde...","excerpt cuts");

    const auto directory=std::filesystem::temp_directory_path()/
        ("preview_presentation_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    nlohmann::json report={{"schema",2},{"run_id","train-1"},{"point","epoch"},{"epoch",4},{"total_epochs",4},
        {"weights","current"},{"mode","chat"},{"template","role_lines"},{"decoding","greedy"},{"max_new_tokens",128},
        {"elapsed_ms",13700.0},{"cancelled",false},{"samples",nlohmann::json::array({
            {{"prompt_index",0},{"sample_index",0},{"prompt","q1"},{"text",looping},{"stop_reason","max_tokens"},
             {"new_token_ids",std::vector<int>(128,4)},{"check",{{"kind","contains"},{"value","Psalm 34:13"}}}},
            {{"prompt_index",1},{"sample_index",0},{"prompt","q2"},
             {"text","Deuteronomy 25:2 says to be a shepherd, and the shepherds"},{"stop_reason","eos"},
             {"new_token_ids",std::vector<int>(114,4)}}})}};
    const auto path=GenerationPreviewReportPath(directory,"epoch",4,0);
    Require(path.filename()=="epoch_4.json","report name");
    std::ofstream(path)<<report.dump();
    Require(GenerationPreviewReportPath(directory,"epoch",4,0).filename()=="epoch_4_2.json","resumed run adds _2");
    std::ofstream(directory/"epoch_3.json")<<R"({"run_id":"old","completed_epoch":3,"samples":[]})";
    std::vector<std::string> skipped;
    const auto reports=LoadGenerationPreviewReports(directory,&skipped);
    Require(reports.size()==1&&skipped.size()==1&&skipped[0].find("older Engine")!=std::string::npos,
            "older reports are skipped and named");
    const auto& loaded=reports[0];
    Require(loaded.samples[0].repeated_phrase=="a burden to your mouth"&&
            loaded.samples[0].check_result==GenerationPreviewCheckResult::Failed&&loaded.samples[1].new_tokens==114,
            "loaded samples");
    const auto summary=SummarizeGenerationPreview(loaded);
    Require(summary.checks==1&&summary.checks_passed==0&&summary.stopped_by_itself==1&&summary.repeating==1,"summary");
    Require(GenerationPreviewPointLabel(loaded)=="Epoch 4 of 4","point label");
    Require(GenerationPreviewStopLabel(loaded.samples[1],128)=="Stopped by itself · 114 tokens","eos label");
    Require(GenerationPreviewStopLabel(loaded.samples[0],128)=="Hit the 128-token limit","limit label");
    Require(GenerationPreviewCheckLabel(loaded.samples[0])=="No \"Psalm 34:13\"","check label");
    const auto lines=GenerationPreviewLogLines(loaded);
    Require(lines.size()==4,"header, two samples, summary");
    Require(lines[0]=="Generation · epoch 4/4 · chat (role lines) · greedy · up to 128 new tokens · current weights · "
                      "2 prompts · 13.7 s","log header");
    Require(lines[1].rfind("  1  limit 128 · repeats · check failed  \"<explanation>",0)==0,"log sample 1");
    Require(lines[2].rfind("  2  stopped 114  \"Deuteronomy 25:2",0)==0,"log sample 2");
    Require(lines[3].find("checks 0/1 · stopped by itself 1/2 · repeating 1/2 · saved ")==2,"log summary");

    // Runs are found under the checkpoints folder; the stamp moves when a report is added.
    const auto root=directory.parent_path()/(directory.filename().string()+"_root");
    const auto run_folder=root/"sft_v2_768"/"train-7"/"generation_previews";
    std::filesystem::create_directories(run_folder);
    std::filesystem::copy_file(path,run_folder/"epoch_4.json");
    const auto before=GenerationPreviewFolderStamp(run_folder);
    std::ofstream(run_folder/"baseline.json")<<report.dump();
    Require(GenerationPreviewFolderStamp(run_folder)!=before,"stamp changes with a new report");
    const auto runs=FindGenerationPreviewRuns(root);
    Require(runs.size()==1&&runs[0].run_id=="train-7"&&runs[0].graph=="sft_v2_768"&&runs[0].files==2,"run found");
    Require(FindGenerationPreviewRuns(root/"missing").empty(),"no folder, no runs");
    std::string export_error;
    Require(ExportGenerationPreviewSamples(reports,run_folder/"samples.jsonl",export_error),"export");
    std::ifstream exported(run_folder/"samples.jsonl");
    std::string first,second,extra;
    std::getline(exported,first); std::getline(exported,second);
    Require(!std::getline(exported,extra)&&nlohmann::json::parse(first)["check"]["passed"]==false&&
            nlohmann::json::parse(second)["stop_reason"]=="eos","one JSON line per sample");
    exported.close();  // Windows keeps an open file's folder.
    std::filesystem::remove_all(root);
    std::filesystem::remove_all(directory);
}

void TestTrainingGenerationPreview() {
    using namespace cyxwiz;
    TestGenerationPreviewSettings();
    TestGenerationPreviewPresentation();

    auto settings=ParseTrainingGenerationPreview(Params({{"generation_preview_max_new_tokens","2"}}));
    Vocabulary vocab;vocab.SetVocabulary({"[PAD]","[UNK]","[BOS]","[EOS]","one","two"});
    std::ostringstream serialized;Require(vocab.SaveToStream(serialized),"vocabulary serialize");
    settings.vocabulary_artifact=serialized.str();settings.tokenizer_type=0;settings.context=8;
    TrainingGenerationPreview preview(settings,6);
    Require(Rejects([&]{TrainingGenerationPreview wrong(settings,7);}),"mismatched vocabulary rejected before training");
    auto long_prompt=settings;long_prompt.context=2;long_prompt.prompts={{"one two",{}}};
    Require(Rejects([&]{TrainingGenerationPreview wrong(long_prompt,6);}),"prompt overflow rejected before training");

    const auto directory=std::filesystem::temp_directory_path()/
        ("preview_unit_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    SequentialModel model;model.Add<PreviewLogits>();model.SetTraining(true);
    auto* module=static_cast<PreviewLogits*>(model.GetModule(0));
    GenerationPreviewPoint point; point.epoch=2; point.total_epochs=3; point.validation_loss=3.5;
    Require(preview.Run(model,point,"test",directory,{}),"generation");
    Require(module->calls==4&&module->IsTraining(),"bounded generation and training-mode restoration");
    nlohmann::json report;std::ifstream(directory/"epoch_2.json")>>report;
    Require(report["schema"]==2&&report["samples"].size()==2&&report["epoch"]==2&&report["weights"]=="current"&&
            report["validation_loss"]==3.5,"report provenance");
    Require(report["tokenizer_sha256"].get<std::string>().size()==64&&report["settings_sha256"].get<std::string>().size()==64,
            "hashes");
    Require(report["samples"][0]["stop_reason"]=="max_tokens"&&report["samples"][0]["text"]=="one one","greedy text");

    // A stop text ends generation and is cut from the output.
    auto stopping=settings; stopping.stop_texts={"two"}; stopping.max_new_tokens=8;
    TrainingGenerationPreview stopper(stopping,6);
    module->script={4,5,4}; module->produced=0;
    GenerationPreviewPoint step_point; step_point.kind="step"; step_point.step=100;
    Require(stopper.Run(model,step_point,"test",directory,{}),"stop text run");
    nlohmann::json stopped;std::ifstream(directory/"step_100.json")>>stopped;
    Require(stopped["samples"][0]["stop_reason"]=="stop_text"&&stopped["samples"][0]["stop_text"]=="two"&&
            stopped["samples"][0]["text"]=="one ","stop text cut");
    module->script.clear();

    module->SetTraining(false);
    int checks=0;
    point.epoch=4;
    Require(!preview.Run(model,point,"test",directory,[&]{return ++checks>=3;}),"between-token cancellation");
    Require(!module->IsTraining(),"restore previously eval module after cancellation");
    module->SetTraining(true);module->fail=true;
    bool rejected=false;
    point.epoch=6;
    try {preview.Run(model,point,"test",directory,{});} catch(const std::runtime_error&){rejected=true;}
    Require(rejected&&module->IsTraining(),"generation exception propagates with restored mode");
    module->fail=false;
    std::filesystem::remove_all(directory);

    LanguageModelGenerationConfig config;config.should_cancel=[] {return true;};
    const int before=module->calls;
    const auto cancelled=GenerateTokenIdsWithReport(model,{4},config);
    Require(cancelled.stop_reason==LanguageModelGenerationStopReason::UserCancelled&&module->calls==before,"cancel before first forward");

    // Repetition penalty: a token already in the prompt loses to the next best.
    module->soft=true; model.SetTraining(false);
    LanguageModelGenerationConfig plain; plain.max_new_tokens=1;
    Require(GenerateTokenIdsWithReport(model,{4},plain).new_token_ids==std::vector<int64_t>{4},"no penalty keeps 4");
    LanguageModelGenerationConfig penalized=plain; penalized.repetition_penalty=2.0f;
    Require(GenerateTokenIdsWithReport(model,{4},penalized).new_token_ids==std::vector<int64_t>{5},"penalty picks 5");
    module->soft=false;
    Require(Rejects([]{LanguageModelGenerationConfig bad; bad.repetition_penalty=0.5f;
                       RequireValidLanguageModelGenerationConfig(bad);}),"penalty below 1 refused");
}
