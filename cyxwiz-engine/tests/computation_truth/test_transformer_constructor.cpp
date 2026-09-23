#include <cyxwiz/layers/attention.h>
#include <cyxwiz/layers/transformer.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/device.h>
#include "algorithms/arrayfire_backend_utils.h"
#include <arrayfire.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void Check(bool ok,const std::string& message) {if(!ok) throw std::runtime_error(message);}
int rejected=0;
void Reject(const std::function<void()>& construct,const std::string& label) {
    af::setSeed(67);
    bool invalid=false;
    try {construct();}
    catch(const std::invalid_argument& e) {invalid=std::string(e.what()).size()>0;}
    Check(invalid,label+" must throw invalid_argument");
    const af::array after=af::randu(16);after.eval();
    af::setSeed(67);const af::array untouched=af::randu(16);untouched.eval();
    Check(af::allTrue<bool>(after==untouched),label+" must reject before weight RNG");
    ++rejected;
}
template<class Module>
void InvalidModule(const std::string& name) {
    const size_t large=static_cast<size_t>((std::numeric_limits<int>::max)())+1;
    for(size_t bad:{size_t{0},large,(std::numeric_limits<size_t>::max)()}) {
        Reject([&]{Module m(bad,2,8,0.0f,false);},name+" width");
        Reject([&]{Module m(4,bad,8,0.0f,false);},name+" heads");
        Reject([&]{Module m(4,2,bad,0.0f,false);},name+" FFN width");
    }
    Reject([]{Module m(4,3,8,0.0f,false);},name+" divisibility");
    for(float p:{-0.1f,1.0f,2.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        Reject([&]{Module m(4,2,8,p,false);},name+" dropout");
        Reject([&]{Module m(4,2,8,0.0f,false,p);},name+" FFN dropout");
    }
}
template<class Layer>
void InvalidLayer(const std::string& name) {
    for(int bad:{0,-1,(std::numeric_limits<int>::min)()}) {
        Reject([&]{Layer m(bad,2,8,0.0f,false);},name+" width");
        Reject([&]{Layer m(4,bad,8,0.0f,false);},name+" heads");
        Reject([&]{Layer m(4,2,bad,0.0f,false);},name+" FFN width");
    }
    Reject([]{Layer m(4,3,8,0.0f,false);},name+" divisibility");
    for(float p:{-0.1f,1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        Reject([&]{Layer m(4,2,8,p,false);},name+" dropout");
        Reject([&]{Layer m(4,2,8,0.0f,false,p);},name+" FFN dropout");
    }
}
cyxwiz::Tensor Sample() {
    cyxwiz::Tensor x(std::vector<size_t>{1,3,4});
    for(size_t i=0;i<x.NumElements();++i)x.MutableData<float>()[i]=std::sin(static_cast<float>(i));
    x.GetSemanticArray().eval();return x;
}
void Equal(const cyxwiz::Tensor& actual,const cyxwiz::Tensor& expected) {
    Check(actual.Shape()==expected.Shape(),"valid configuration shape");
    const auto* a=actual.ReadData<float>();const auto* b=expected.ReadData<float>();
    for(size_t i=0;i<actual.NumElements();++i)
        Check(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i]-b[i])<1e-6f,"valid configuration numerical parity");
}
template<class Module,class Layer>
void ValidBlocks() {
    for(bool pre:{false,true}) for(float p:{0.0f,0.25f}) {
        Module module(4,2,7,p,pre,p);Layer layer(4,2,7,p,pre,p);
        module.SetParameters(layer.GetParameters());module.SetTraining(true);layer.SetTraining(true);
        auto x=Sample();
        af::setSeed(52);const auto reference=layer.Forward(x);
        af::setSeed(52);Equal(module.Forward(x),reference);
        Equal(module.Backward(x),layer.Backward(x));
        Check(module.GetName().find("heads=2")!=std::string::npos,"valid requested head count");
    }
}
} // namespace
int main(int argc,char**argv) {
    try {
        Check(argc==2,"Usage: test_transformer_constructor cpu|cuda|opencl");
        const std::string backend=argv[1];
        Check(backend=="cpu"||backend=="cuda"||backend=="opencl","backend name");
        auto activation=cyxwiz::Device(backend=="cpu"?cyxwiz::DeviceType::CPU:
            backend=="cuda"?cyxwiz::DeviceType::CUDA:cyxwiz::DeviceType::OPENCL,0).ActivateExact(true);
        Check(activation.success && activation.execution_validated,activation.message);
        cyxwiz::ScopedArrayFireFallbackPolicy strict(cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
        InvalidModule<cyxwiz::TransformerEncoderModule>("encoder module");
        InvalidModule<cyxwiz::TransformerDecoderModule>("decoder module");
        InvalidLayer<cyxwiz::TransformerEncoderLayer>("encoder layer");
        InvalidLayer<cyxwiz::TransformerDecoderLayer>("decoder layer");
        const size_t large=static_cast<size_t>((std::numeric_limits<int>::max)())+1;
        for(size_t bad:{size_t{0},large,(std::numeric_limits<size_t>::max)()}) {
            Reject([&]{cyxwiz::MultiHeadAttentionModule m(bad,2);},"attention width");
            Reject([&]{cyxwiz::MultiHeadAttentionModule m(4,bad);},"attention heads");
        }
        Reject([]{cyxwiz::MultiHeadAttentionModule m(4,3);},"attention divisibility");
        for(float p:{-0.1f,1.0f,2.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
            Reject([&]{cyxwiz::MultiHeadAttentionModule m(4,2,p);},"attention dropout");
        ValidBlocks<cyxwiz::TransformerEncoderModule,cyxwiz::TransformerEncoderLayer>();
        ValidBlocks<cyxwiz::TransformerDecoderModule,cyxwiz::TransformerDecoderLayer>();
        for(float p:{0.0f,0.25f}) {
            cyxwiz::MultiHeadAttentionModule module(4,2,p);cyxwiz::MultiHeadAttentionLayer layer(4,2,p);
            module.SetParameters(layer.GetParameters());module.SetTraining(true);layer.SetTraining(true);
            auto x=Sample();af::setSeed(52);const auto reference=layer.Forward(x);
            af::setSeed(52);Equal(module.Forward(x),reference);Equal(module.Backward(x),layer.Backward(x));
        }
        std::cout<<"PASS "<<backend<<": "<<rejected<<" invalid constructors rejected before RNG; 10 valid module/layer forward/backward cases\n";
    }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
