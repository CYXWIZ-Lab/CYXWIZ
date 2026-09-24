#include <cyxwiz/layers/attention.h>
#include <cyxwiz/layers/transformer.h>
#include <cyxwiz/sequential.h>
#include "core/execution_device_context.h"
#include <arrayfire.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <limits>
#include <stdexcept>

namespace {
using cyxwiz::Tensor;
using json = nlohmann::json;
size_t hosts = 0, fallbacks = 0;
void Host(const cyxwiz::ArrayFireHostSyncEvent&) { ++hosts; }
void Fallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) { ++fallbacks; }
void Check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void Near(float a, float b, const std::string& message) {
    Check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b) < 7e-4f + 3e-3f * std::abs(b), message);
}
Tensor Fixture(const std::vector<size_t>& shape, float offset) {
    Tensor x(shape);
    auto* p=x.MutableData<float>();
    for(size_t i=0;i<x.NumElements();++i) p[i]=0.25f*std::sin(0.7f*static_cast<float>(i)+offset);
    x.GetSemanticArray().eval(); return x;
}
json Pack(const Tensor& x) {
    if (x.Shape().empty() || x.NumElements()==0) return {{"shape", x.Shape()}, {"data", json::array()}};
    const auto* p=x.ReadData<float>();
    return { {"shape", x.Shape()}, {"data",std::vector<float>(p,p+x.NumElements())} };
}
float Objective(const Tensor& y, const Tensor& upstream) {
    const auto* a=y.ReadData<float>(); const auto* b=upstream.ReadData<float>();
    float value=0; for(size_t i=0;i<y.NumElements();++i) value+=a[i]*b[i]; return value;
}
template<class Layer>
auto Configure(Layer& layer) {
    auto params=layer.GetParameters(); float offset=0;
    for(auto it=params.begin();it!=params.end();) {
        if(it->first.find("grad_")!=std::string::npos) {it=params.erase(it);continue;}
        it->second=Fixture(it->second.Shape(),offset+=0.43f);
        if(it->first.find("gamma")!=std::string::npos) {
            auto* p=it->second.template MutableData<float>();
            for(size_t i=0;i<it->second.NumElements();++i) p[i]+=1.0f;
            it->second.GetSemanticArray().eval();
        }
        ++it;
    }
    layer.SetParameters(params); return params;
}
void Difference(Tensor& variable,const Tensor& derivative,const std::function<Tensor()>& forward,
                const Tensor& upstream,const std::function<void()>& apply,const std::string& label) {
    const Tensor snapshot=derivative.Clone();
    const auto* expected=snapshot.ReadData<float>();
    for(size_t i=0;i<variable.NumElements();++i) {
        const float original=variable.ReadData<float>()[i];
        variable.MutableData<float>()[i]=original+1e-3f; apply();
        af::setSeed(81); const float plus=Objective(forward(),upstream);
        variable.MutableData<float>()[i]=original-1e-3f; apply();
        af::setSeed(81); const float minus=Objective(forward(),upstream);
        variable.MutableData<float>()[i]=original; apply();
        Near((plus-minus)/2e-3f,expected[i],label+"["+std::to_string(i)+"]");
    }
}
template<class Layer>
json Verify(Layer& layer,std::map<std::string,Tensor>& params,Tensor& x,
            const std::function<Tensor()>& forward) {
    Tensor upstream=Fixture(x.Shape(),1.7f),y,dx;
    layer.SetTraining(true); af::setSeed(81); hosts=fallbacks=0;
    {
        cyxwiz::ScopedArrayFireHostSyncObserver h(Host);
        cyxwiz::ScopedArrayFireNativeCpuFallbackObserver f(Fallback);
        y=forward(); dx=layer.Backward(upstream); af::sync();
        Check(hosts==0 && fallbacks==0,"dropout forward/backward residency");
    }
    const auto gradients=layer.GetParameters();
    json result={{"input",Pack(x)},{"upstream",Pack(upstream)},
                 {"output",Pack(y)},{"dx",Pack(dx)}, {"parameters",json::object()},
                 {"gradients",json::object()}};
    for(const auto& [name,t]:params) result["parameters"][name]=Pack(t);
    for(const auto& [name,t]:gradients) if(name.find("grad_")!=std::string::npos)
        result["gradients"][name]=Pack(t);
    af::setSeed(81); const Tensor replay=forward();
    for(size_t i=0;i<y.NumElements();++i) Check(y.ReadData<float>()[i]==replay.ReadData<float>()[i],"seed replay");
    const af::array after_training=af::randu(8);after_training.eval();
    af::setSeed(81);const af::array fresh=af::randu(8);fresh.eval();
    Check(!af::allTrue<bool>(after_training==fresh),"training must advance RNG");
    Difference(x,dx,forward,upstream,[]{},"input gradient");
    for(auto& [name,t]:params) {
        const size_t dot=name.rfind('.');
        const std::string gradient=dot==std::string::npos ? "grad_"+name :
            name.substr(0,dot+1)+"grad_"+name.substr(dot+1);
        const auto it=gradients.find(gradient);
        // Decoder-only owns unused cross-attention/norm3 parameters; no gradient.
        if(it==gradients.end() || it->second.Shape().empty()) continue;
        Difference(t,it->second,forward,upstream,[&]{layer.SetParameters(params);},name);
    }
    layer.SetTraining(false); af::setSeed(99);
    const Tensor eval1=forward(),eval2=forward();
    for(size_t i=0;i<y.NumElements();++i) Check(eval1.ReadData<float>()[i]==eval2.ReadData<float>()[i],"evaluation deterministic");
    const af::array after_eval=af::randu(8); af::setSeed(99); const af::array expected=af::randu(8);
    Check(af::allTrue<bool>(after_eval==expected),"evaluation must not consume RNG");
    return result;
}
json AttentionCase(bool self) {
    cyxwiz::MultiHeadAttentionLayer layer(4,2,0.25f,true);
    auto params=Configure(layer);
    Tensor q=Fixture({1,3,4},0.2f), k=Fixture({1,4,4},0.5f),v=Fixture({1,4,4},0.9f);
    const size_t keys=self?3:4;
    Tensor mask=Tensor::Zeros({3,keys}); auto* m=mask.MutableData<float>();
    for(size_t i=0;i<keys;++i) m[i]=-INFINITY;
    mask.GetSemanticArray().eval();
    const auto forward=[&]{return self?layer.Forward(q,q,q,&mask):layer.Forward(q,k,v,&mask);};
    af::setSeed(81);
    const auto keep=(af::randu(af::dim4(3,keys,1,2),f32)>.25f).as(f32); keep.eval();
    const Tensor replay_mask=Tensor::FromSemanticArray(keep,{3,keys,1,2});
    json result=Verify(layer,params,q,forward);
    result["kind"]=self?"attention_self":"attention_cross";
    result["key"]=Pack(self?q:k); result["value"]=Pack(self?q:v);
    result["mask"]=Pack(replay_mask); result["p"]=.25;
    if(!self) {
        layer.SetTraining(true); af::setSeed(81); forward();
        Tensor upstream=Fixture(q.Shape(),1.7f); layer.Backward(upstream);
        const auto dk=layer.GetLastKeyGradient(),dv=layer.GetLastValueGradient();
        result["dk"]=Pack(dk);result["dv"]=Pack(dv);
        Difference(k,dk,forward,upstream,[]{},"key gradient");
        Difference(v,dv,forward,upstream,[]{},"value gradient");
    }
    std::cout<<"PASS "<<result["kind"]<<" mask replay/all gradients/residency\n";
    return result;
}
template<class Layer>
json TransformerCase(bool decoder,bool pre) {
    Layer layer(4,2,5,0.0f,pre,0.25f);
    auto params=Configure(layer); Tensor x=Fixture({1,2,4},0.2f);
    af::setSeed(81); const auto keep=(af::randu(af::dim4(2,5),f32)>.25f).as(f32);keep.eval();
    const Tensor replay_mask=Tensor::FromSemanticArray(keep,{2,5}).Reshape({1,2,5});
    auto result=Verify(layer,params,x,[&]{return layer.Forward(x);});
    result["kind"]=decoder?"decoder":"encoder";result["norm_first"]=pre;
    result["mask"]=Pack(replay_mask);result["p"]=.25;
    std::cout<<"PASS "<<result["kind"]<<" pre="<<pre<<" FFN dropout all gradients/residency\n";
    return result;
}

template<class Layer>
void CompatibilityAndCombinedDropout() {
    for(float invalid:{-0.1f,1.0f,std::numeric_limits<float>::quiet_NaN()}) {
        bool rejected=false;
        try { Layer bad(4,2,5,0.1f,false,invalid); }
        catch(const std::invalid_argument&) { rejected=true; }
        Check(rejected,"invalid FFN probability rejected");
    }
    Layer legacy(4,2,5,0.0f,false),explicit_zero(4,2,5,0.0f,false,0.0f);
    const auto params=Configure(legacy);explicit_zero.SetParameters(params);
    Tensor x=Fixture({1,2,4},0.2f), upstream=Fixture(x.Shape(),1.7f);
    af::setSeed(99);
    const auto a=legacy.Forward(x),b=explicit_zero.Forward(x);
    const af::array after=af::randu(8);af::setSeed(99);const af::array expected=af::randu(8);
    Check(af::allTrue<bool>(after==expected),"zero dropout must not consume RNG");
    for(size_t i=0;i<a.NumElements();++i) Check(a.template ReadData<float>()[i]==b.template ReadData<float>()[i],"legacy zero dropout compatibility");
    for(bool pre:{false,true}) {
        Layer combined(4,2,5,0.1f,pre,0.25f);Configure(combined);
        combined.SetTraining(true);af::setSeed(81);hosts=fallbacks=0;
        {
            cyxwiz::ScopedArrayFireHostSyncObserver h(Host);
            cyxwiz::ScopedArrayFireNativeCpuFallbackObserver f(Fallback);
            combined.Forward(x);const auto dx=combined.Backward(upstream);af::sync();
            Check(hosts==0 && fallbacks==0,"combined dropout residency");
        }
        af::setSeed(81);combined.Forward(x);
        const Tensor train_dx=combined.Backward(upstream).Clone();
        af::setSeed(81);combined.Forward(x);combined.SetTraining(false);
        const Tensor switched_dx=combined.Backward(upstream);
        for(size_t i=0;i<x.NumElements();++i) Near(switched_dx.ReadData<float>()[i],train_dx.ReadData<float>()[i],"backward reuses forward masks after mode switch");
    }
    std::cout<<"PASS legacy/invalid FFN/combined dropout/mode switch\n";
}

json MemoryCase(bool pre) {
    cyxwiz::TransformerDecoderLayer layer(4,2,5,0.0f,pre,0.25f);
    auto params=Configure(layer);
    Tensor x=Fixture({1,2,4},0.2f),memory=Fixture({1,3,4},0.6f);
    Tensor mask=cyxwiz::TransformerDecoderLayer::GenerateCausalMask(2);
    memory.GetSemanticArray().eval();mask.GetSemanticArray().eval();
    af::setSeed(81);const auto keep=(af::randu(af::dim4(2,5),f32)>.25f).as(f32);keep.eval();
    const Tensor replay_mask=Tensor::FromSemanticArray(keep,{2,5}).Reshape({1,2,5});
    const auto forward=[&]{return layer.Forward(x,memory,&mask);};
    auto result=Verify(layer,params,x,forward);
    layer.SetTraining(true);af::setSeed(81);forward();
    Tensor upstream=Fixture(x.Shape(),1.7f);layer.Backward(upstream);
    const Tensor dm=layer.GetLastMemoryGradient();
    result["kind"]="memory";result["norm_first"]=pre;result["p"]=.25;
    result["memory"]=Pack(memory);result["dm"]=Pack(dm);result["mask"]=Pack(replay_mask);
    Difference(memory,dm,forward,upstream,[]{},"memory gradient");
    std::cout<<"PASS memory decoder pre="<<pre<<" FFN dropout all gradients/residency\n";
    return result;
}
} // namespace
int main(int argc,char**argv) {
    try {
        Check(argc==3,"Usage: test_transformer_dropout cpu|cuda|opencl output.json");
        std::string backend=argv[1];
        Check(backend=="cpu"||backend=="cuda"||backend=="opencl","backend");
        auto activation=cyxwiz::Device(backend=="cpu"?cyxwiz::DeviceType::CPU:
            backend=="cuda"?cyxwiz::DeviceType::CUDA:cyxwiz::DeviceType::OPENCL,0).ActivateExact(true);
        Check(activation.success && activation.execution_validated,activation.message);
        auto policy=cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
        auto context=cyxwiz::CaptureCurrentExecutionDeviceContext(policy);
        cyxwiz::ScopedActiveExecutionDeviceContext active;
        cyxwiz::ScopedExecutionDeviceContext binding(context);
        cyxwiz::ScopedArrayFireFallbackPolicy strict(policy);
        json cases=json::array();
        cases.push_back(AttentionCase(true)); cases.push_back(AttentionCase(false));
        for(bool pre:{false,true}) {
            cases.push_back(TransformerCase<cyxwiz::TransformerEncoderLayer>(false,pre));
            cases.push_back(TransformerCase<cyxwiz::TransformerDecoderLayer>(true,pre));
            cases.push_back(MemoryCase(pre));
        }
        CompatibilityAndCombinedDropout<cyxwiz::TransformerEncoderLayer>();
        CompatibilityAndCombinedDropout<cyxwiz::TransformerDecoderLayer>();
        std::ofstream out(argv[2]); Check(out.good(),"output file");out<<cases.dump(2)<<'\n';
        Check(out.good(),"write output");
        std::cout<<"PASS dropout backend="<<backend<<'\n';
    } catch(const std::exception&e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
