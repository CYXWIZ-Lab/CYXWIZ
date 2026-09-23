#include <cyxwiz/layers/transformer.h>
#include <cyxwiz/sequential.h>
#include "core/execution_device_context.h"
#include "core/language_model_generation.h"
#include <arrayfire.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>

namespace {
using cyxwiz::Tensor;using json=nlohmann::json;
uint64_t host_count=0,host_bytes=0,fallbacks=0;
void Host(const cyxwiz::ArrayFireHostSyncEvent& e){++host_count;host_bytes+=e.bytes;}
void Fallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&){++fallbacks;}
void Check(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
Tensor Fixture(const std::vector<size_t>& shape,float offset=0){
    Tensor t(shape);auto*p=t.MutableData<float>();
    for(size_t i=0;i<t.NumElements();++i)p[i]=0.05f*std::sin(0.2f*static_cast<float>(i)+offset);
    t.GetSemanticArray().eval();return t;
}
template<class Model> void FixedParameters(Model& model){
    auto p=model.GetParameters();float offset=0;
    for(auto i=p.begin();i!=p.end();){
        if(i->first.find("grad_")!=std::string::npos){i=p.erase(i);continue;}
        i->second=Fixture(i->second.Shape(),offset+=0.2f);
        if(i->first.find("gamma")!=std::string::npos){
            auto*data=i->second.MutableData<float>();for(size_t j=0;j<i->second.NumElements();++j)data[j]+=1;
            i->second.GetSemanticArray().eval();
        }++i;
    }model.SetParameters(p);
}
json Measure(const std::string& name,const std::function<void()>& work,size_t tokens,int iterations){
    for(int i=0;i<5;++i){work();af::sync();}
    std::vector<double> samples;host_count=host_bytes=fallbacks=0;
    {
        cyxwiz::ScopedArrayFireHostSyncObserver h(Host);
        cyxwiz::ScopedArrayFireNativeCpuFallbackObserver f(Fallback);
        for(int i=0;i<iterations;++i){
            af::sync();const auto start=std::chrono::steady_clock::now();
            work();af::sync();
            samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
    }
    Check(fallbacks==0,"strict benchmark encountered native fallback");
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    const double median=sorted[sorted.size()/2];
    const double p95=sorted[static_cast<size_t>(std::ceil(.95*sorted.size()))-1];
    std::cout<<name<<": median="<<median<<" ms; host bytes/call="<<double(host_bytes)/iterations<<std::endl;
    return {{"name",name},{"samples_ms",samples},{"median_ms",median},{"p95_ms",p95},
        {"tokens_per_second",tokens*1000./median},{"tokens_per_call",tokens},{"iterations",iterations},
        {"host_syncs",host_count},{"host_sync_bytes",host_bytes},{"native_fallbacks",fallbacks},
        {"explicit_measurement_fences",iterations*2},{"warmup_calls",5}};
}
void Blocks(json& results,size_t sequence,int width,int iterations){
    const std::string prefix="B2_S"+std::to_string(sequence)+"_D"+std::to_string(width);
    const auto x=Fixture({2,sequence,static_cast<size_t>(width)},.3f);
    const auto g=Fixture(x.Shape(),.8f);Tensor sink;
    cyxwiz::TransformerDecoderLayer raw(width,4,width*4,0.0f,false);FixedParameters(raw);raw.SetTraining(false);
    cyxwiz::TransformerDecoderModule wrapper(width,4,width*4,0.0f,false);
    wrapper.SetParameters(raw.GetParameters());wrapper.SetTraining(false);
    results.push_back(Measure(prefix+"_causal_mask",[&]{sink=cyxwiz::TransformerDecoderLayer::GenerateCausalMask(static_cast<int>(sequence));sink.GetSemanticArray().eval();},0,iterations));
    results.push_back(Measure(prefix+"_raw_eval",[&]{sink=raw.Forward(x);sink.GetSemanticArray().eval();},2*sequence,iterations));
    const auto reference=sink.Clone();
    results.push_back(Measure(prefix+"_wrapper_eval",[&]{sink=wrapper.Forward(x);sink.GetSemanticArray().eval();},2*sequence,iterations));
    const auto*a=sink.ReadData<float>();const auto*b=reference.ReadData<float>();
    for(size_t i=0;i<sink.NumElements();++i)Check(std::isfinite(a[i])&&std::abs(a[i]-b[i])<1e-6f,"raw/wrapper numerical mismatch");
    cyxwiz::SequentialModel stack;
    stack.Add<cyxwiz::TransformerDecoderModule>(width,4,width*4,0.0f,false);
    stack.Add<cyxwiz::TransformerDecoderModule>(width,4,width*4,0.0f,false);
    FixedParameters(stack);stack.SetTraining(true);
    results.push_back(Measure(prefix+"_two_blocks_forward_backward",[&]{
        stack.Forward(x);sink=stack.Backward(g);sink.GetSemanticArray().eval();
        for(const auto&[name,gradient]:stack.GetGradients())if(gradient.NumElements())gradient.GetSemanticArray().eval();
    },2*sequence,iterations));
}
void Generation(json& results,int iterations){
    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::EmbeddingModule>(2540,16);
    model.Add<cyxwiz::PositionalEncodingModule>(16,32);
    model.Add<cyxwiz::TransformerDecoderModule>(16,4,64,0.0f,false);
    model.Add<cyxwiz::TransformerDecoderModule>(16,4,64,0.0f,false);
    model.Add<cyxwiz::LayerNormModule>(std::vector<int>{16});
    model.Add<cyxwiz::TimeDistributedDenseModule>(16,2540);
    FixedParameters(model);model.SetTraining(false);
    cyxwiz::LanguageModelGenerationConfig config;config.max_new_tokens=8;config.max_context_tokens=32;
    std::vector<int64_t> prompt(24);for(size_t i=0;i<prompt.size();++i)prompt[i]=1+i;
    cyxwiz::LanguageModelGenerationResult sink;
    auto measured=Measure("generation_D16_V2540_prompt24_new8",[&]{sink=cyxwiz::GenerateTokenIdsWithReport(model,prompt,config,52);},8,iterations);
    Check(sink.new_token_ids.size()==8,"generation length");
    measured["generated_ids"]=sink.new_token_ids;
    measured["full_prefix_logits_bytes_expected_per_call"]=uint64_t(24+25+26+27+28+29+30+31)*2540*4;
    measured["last_position_logits_bytes_per_call"]=8*2540*4;
    results.push_back(measured);
}
} // namespace
int main(int argc,char**argv){try{
    Check(argc==4||(argc==5&&std::string(argv[4])=="large"),"Usage: test_transformer_benchmark cpu|cuda|opencl output.json iterations [large]");
    const bool large=argc==5;  // tofix68 attention evidence: adds S512/D256 blocks
    std::string backend=argv[1];int iterations=std::stoi(argv[3]);Check(iterations>=5&&iterations<=100,"iterations must be 5..100");
    Check(backend=="cpu"||backend=="cuda"||backend=="opencl","backend");
    auto activation=cyxwiz::Device(backend=="cpu"?cyxwiz::DeviceType::CPU:backend=="cuda"?cyxwiz::DeviceType::CUDA:cyxwiz::DeviceType::OPENCL,0).ActivateExact(true);
    Check(activation.success&&activation.execution_validated,activation.message);
    auto policy=cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
    const auto context=cyxwiz::CaptureCurrentExecutionDeviceContext(policy);
    cyxwiz::ScopedActiveExecutionDeviceContext active;
    cyxwiz::ScopedExecutionDeviceContext binding(context);
    cyxwiz::ScopedArrayFireFallbackPolicy strict(policy);
    af::setSeed(52);int major,minor,patch;Check(af_get_version(&major,&minor,&patch)==AF_SUCCESS,"ArrayFire version");
    json report={{"device",context.Describe()},{"backend",backend},
       {"exact_activation",{{"success",activation.success},{"execution_validated",activation.execution_validated},{"message",activation.message}}},
       {"qualification_note","Standalone capture does not load Engine route-qualification evidence; see exact_activation for the executed device probe."},
       {"arrayfire_version",std::to_string(major)+"."+std::to_string(minor)+"."+std::to_string(patch)},
       {"clock","steady_clock, device synchronized"},{"fixture","deterministic sine weights/inputs; no corpus"},
       {"full_training_step",false},{"cases",json::array()}};
    Blocks(report["cases"],32,16,iterations);Blocks(report["cases"],128,64,iterations);
    if(large){Blocks(report["cases"],512,256,iterations);}
    Generation(report["cases"],iterations);
    std::ofstream out(argv[2]);Check(out.good(),"output file");out<<report.dump(2)<<'\n';Check(out.good(),"output write");
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
