#include "../src/core/formats/cyxmodel_archive.h"
#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/model_converter.h"
#include <cyxwiz/utilities.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs=std::filesystem;
using namespace cyxwiz;
using namespace cyxwiz::formats;
void Check(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class F> void Reject(F action,const std::string& message) {
    bool rejected=false;
    try { action(); } catch(const std::exception&) { rejected=true; }
    Check(rejected,message);
}
std::vector<uint8_t> Bytes(const fs::path& path) {
    std::ifstream input(path,std::ios::binary);
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
void Put(const fs::path& path,const std::vector<uint8_t>& bytes) {
    std::ofstream output(path,std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    Check(static_cast<bool>(output),"fixture write failed");
}
void ConvertRoundtrip(const fs::path& source,const fs::path& root) {
    const auto binary=root/"converted.cyxmodel";
    Check(ModelConverter::DirectoryToBinary(source.string(),binary.string()),ModelConverter::GetLastError());
    Check(CyxModelArchive::IsV3(binary),"converter must write v3");
    const auto directory=root/"unpacked.cyxmodel";
    Check(ModelConverter::BinaryToDirectory(binary.string(),directory.string()),ModelConverter::GetLastError());
    const auto original=CyxModelArchive::Read(source);
    Check(original==CyxModelArchive::Read(binary),"directory->binary payload identity");
    Check(original==CyxModelArchive::Read(directory),"binary->directory payload identity");
    const auto again=root/"repacked.cyxmodel";
    Check(ModelConverter::DirectoryToBinary(directory.string(),again.string()),ModelConverter::GetLastError());
    Check(Bytes(binary)==Bytes(again),"deterministic v3 repacking");
    Check(!ModelConverter::BinaryToDirectory(binary.string(),directory.string()),"existing directory must be preserved");
    Check(original==CyxModelArchive::Read(directory),"failed conversion changed directory");
}
int main(int argc,char** argv) {
    try {
        if(argc==3) {
            const fs::path root(argv[2]);
            Check(!fs::exists(root),"actual conversion output already exists");
            fs::create_directories(root);
            ConvertRoundtrip(argv[1],root);
            std::cout<<"Actual package conversion roundtrip passed\n";
            return 0;
        }
        Check(argc==1,"usage: test [source_directory new_output_directory]");
        const auto root=fs::temp_directory_path()/("cyxmodel-v3-"+Utilities::GenerateUUIDs(1).front());
        fs::create_directory(root);
        const auto path=root/"model.cyxmodel";
        const CyxModelAssets assets={{"manifest.json",{'{','}'}},{"sub/bytes.bin",{0,1,0,255}}};
        CyxModelArchive::WriteBinary(path,assets);
        Check(fs::is_regular_file(path) && CyxModelArchive::IsV3(path),"binary header and layout");
        Check(CyxModelArchive::Read(path)==assets,"binary payload identity including NUL");
        const auto original=Bytes(path);
        for (const auto& name : {"../escape","/absolute","C:/drive","sub\\escape","a/./b","NUL.bin"})
            Reject([&]{CyxModelArchive::WriteBinary(path,{{name,{1}}});},"unsafe path accepted");
        Reject([&]{CyxModelArchive::WriteBinary(path,{{"A",{1}},{"a",{2}}});},"case collision accepted");
        Reject([&]{CyxModelArchive::WriteBinary(path,{{"a",{1}},{"a/b",{2}}});},"file/directory collision accepted");
        Check(Bytes(path)==original,"rejected export changed destination");
        const auto bad=root/"bad.cyxmodel";
        auto modified=original; modified.back()^=1; Put(bad,modified);
        Reject([&]{CyxModelArchive::Read(bad);},"corrupt digest accepted");
        modified=original; modified.pop_back(); Put(bad,modified);
        Reject([&]{CyxModelArchive::Read(bad);},"truncated payload accepted");
        modified=original; modified.push_back(0); Put(bad,modified);
        Reject([&]{CyxModelArchive::Read(bad);},"trailing bytes accepted");
        modified=original; modified[4]=99; Put(bad,modified);
        Reject([&]{CyxModelArchive::Read(bad);},"unknown version accepted");
        modified=original; modified[8]=1; Put(bad,modified);
        Reject([&]{CyxModelArchive::Read(bad);},"unsupported compression accepted");
        CyxModelArchiveLimits tiny; tiny.payload_bytes=1;
        Reject([&]{CyxModelArchive::Read(path,tiny);},"payload budget ignored");
        tiny={}; tiny.entries=1;
        Reject([&]{CyxModelArchive::Read(path,tiny);},"entry budget ignored");
        // Independent malformed archive fixture exercises reader path validation.
        auto raw_archive = [&](const std::string& name) {
            std::vector<uint8_t> bytes;
            auto integer=[&](uint64_t value,size_t count) {
                for(size_t i=0;i<count;++i) bytes.push_back(static_cast<uint8_t>(value>>(8*i)));
            };
            integer(0x43595857,4); integer(3,4); integer(0,4); integer(1,4);
            integer(name.size(),4); integer(1,8);
            const auto hash=Utilities::HashText(name+std::string(1,'\0')+"x","sha256").sha256_hash;
            bytes.insert(bytes.end(),hash.begin(),hash.end());
            bytes.insert(bytes.end(),name.begin(),name.end()); bytes.push_back('x');
            return bytes;
        };
        Put(bad,raw_archive("../escape"));
        Reject([&]{CyxModelArchive::Read(bad);},"reader accepted traversal with valid digest");
        Put(bad,raw_archive(std::string(1,static_cast<char>(255))));
        Reject([&]{CyxModelArchive::Read(bad);},"reader accepted invalid UTF-8");
        Put(root/"blocker",{1});
        Reject([&]{CyxModelArchive::WriteBinary(root/"blocker"/"out",assets);},"invalid output parent accepted");
        Check(Bytes(path)==original,"failed writes changed existing output");
        fs::create_directory(root/"occupied");
        Reject([&]{CyxModelArchive::WriteBinary(root/"occupied",assets);},"binary replaced directory");

        // Complete native export preserves five supported tensor types.
        CyxModelFormat format;
        ModelManifest manifest; manifest.has_graph=true;
        TrainingConfig config;
        ExportOptions options;
        std::map<std::string,std::vector<uint8_t>> weights={
            {"float32",{1,2,3,4}},{"float64",{1,2,3,4,5,6,7,8}},
            {"int32",{4,3,2,1}},{"int64",{8,7,6,5,4,3,2,1}},{"uint8",{255}}};
        std::map<std::string,std::vector<int64_t>> shapes;
        for(const auto& pair:weights) shapes[pair.first]={1};
        const std::map<std::string,TensorDType> dtypes={{"float32",TensorDType::Float32},
            {"float64",TensorDType::Float64},{"int32",TensorDType::Int32},
            {"int64",TensorDType::Int64},{"uint8",TensorDType::UInt8}};
        const auto typed=root/"typed.cyxmodel";
        Check(format.Create(typed.string(),manifest,"{\"nodes\":[]}",config,nullptr,weights,shapes,nullptr,options,&dtypes),format.GetLastError());
        Check(fs::is_regular_file(typed),"default export is not a single file");
        std::map<std::string,std::vector<uint8_t>> loaded;
        std::map<std::string,std::vector<int64_t>> loaded_shapes;
        std::map<std::string,TensorDType> loaded_dtypes;
        std::string graph;
        Check(format.Extract(typed.string(),manifest,graph,config,nullptr,loaded,loaded_shapes,nullptr,{},&loaded_dtypes),format.GetLastError());
        Check(weights==loaded && shapes==loaded_shapes && dtypes==loaded_dtypes,"mixed dtype roundtrip");
        const auto before_compress=Bytes(typed);
        options.compress=true;
        Check(!format.Create(typed.string(),manifest,graph,config,nullptr,weights,shapes,nullptr,options,&dtypes),"unimplemented compression must reject");
        Check(Bytes(typed)==before_compress,"failed compression changed existing model");
        const auto folder=root/"source.cyxmodel";
        CyxModelArchive::WriteDirectory(folder,CyxModelArchive::Read(typed));
        Check(!ModelConverter::DirectoryToBinary(folder.string(),(root/"cancelled").string(),
            [](int,int,const std::string&){throw std::runtime_error("cancelled before publication");}),
            "cancelled conversion should fail");
        Check(!fs::exists(root/"cancelled"),"cancelled conversion published output");
        ConvertRoundtrip(folder,root);
        auto invalid=CyxModelArchive::Read(typed);
        invalid.at("weights/float32.bin").pop_back();
        CyxModelArchive::WriteBinary(bad,invalid);
        Check(!format.Extract(bad.string(),manifest,graph,config,nullptr,loaded,loaded_shapes,nullptr,{}),"invalid tensor byte count accepted");
        modified=original; modified[4]=2; Put(bad,modified);
        Check(!ModelConverter::BinaryToDirectory(bad.string(),(root/"legacy").string()),"lossy v2 conversion accepted");
        Check(!fs::exists(root/"legacy"),"failed v2 conversion created output");
        fs::remove_all(root);
        std::cout<<"CyxModel archive, typed export and conversion tests passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
