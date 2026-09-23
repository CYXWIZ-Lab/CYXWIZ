#include "gui/subgraph_presentation.h"
#include <imgui.h>
#include <imnodes.h>
#include <iostream>
#include <stdexcept>
using namespace gui;
using namespace gui::detail;
namespace {
int checks = 0;
void Check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
MLNode Node(int id, int in, int out, NodeType type = NodeType::TextCleanNode) {
    MLNode node{}; node.id=id; node.type=type; node.name="Member "+std::to_string(id);
    if (in) { NodePin pin{}; pin.id=in; pin.type=PinType::Dataset; pin.is_input=true; node.inputs.push_back(pin); }
    if (out) { NodePin pin{}; pin.id=out; pin.type=PinType::Dataset; node.outputs.push_back(pin); }
    return node;
}
struct Fixture {
    std::vector<MLNode> nodes{Node(1,0,11),Node(2,21,22),Node(3,31,32),Node(4,41,0),Node(5,51,52,NodeType::Subgraph)};
    std::vector<SubgraphData> groups{{5,{nodes[1],nodes[2]},{{2,2,22,3,31}},{21},{32},true}};
};
ImVec2 button_center;
int Frame(Fixture& f, bool busy, float zoom=1.0f) {
    auto& io=ImGui::GetIO(); io.DisplaySize=ImVec2(1000,700); io.DeltaTime=1.0f/60.0f;
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0,0)); ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("Canvas",nullptr,ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove);
    ImNodes::BeginNodeEditor();
    for (const auto& n:f.nodes) {
        if (IsExpandedSubgraph(n.id,f.groups)) continue;
        ImNodes::SetNodeGridSpacePos(n.id,ImVec2(80.0f+static_cast<float>(n.id)*130.0f,240.0f));
        ImNodes::BeginNode(n.id); ImGui::Dummy(ImVec2(64,64)); ImNodes::EndNode();
    }
    const auto cursor=ImGui::GetCursorScreenPos();
    const int channel=ImGui::GetWindowDrawList()->_Splitter._Current;
    const int result=DrawExpandedSubgraphFrames(f.nodes,f.groups,zoom,busy ? "Training is active" : "");
    const auto a=ImGui::GetItemRectMin(), b=ImGui::GetItemRectMax();
    button_center=ImVec2((a.x+b.x)*0.5f,(a.y+b.y)*0.5f);
    Check(ImGui::GetWindowDrawList()->_Splitter._Current==channel,"draw channel restored");
    Check(ImGui::GetCursorScreenPos().x==cursor.x && ImGui::GetCursorScreenPos().y==cursor.y,"cursor restored");
    ImNodes::EndNodeEditor(); ImGui::End(); ImGui::Render();
    Check(ImGui::GetDrawData()->TotalVtxCount>0,"frame produced draw data");
    return result;
}
}
int main() {
    try {
        Fixture f;
        Check(IsExpandedSubgraph(5,f.groups),"expanded wrapper is hidden");
        Check(!IsExpandedSubgraph(2,f.groups),"member remains visible");
        NodeLink input{10,1,11,5,51}, output{11,5,52,4,41}, internal{12,2,22,3,31};
        auto in=DisplaySubgraphLink(input,f.nodes,f.groups), out=DisplaySubgraphLink(output,f.nodes,f.groups);
        Check(in && in->to_node==2 && in->to_pin==21 && in->id==10,"input renders at member pin with canonical ID");
        Check(out && out->from_node==3 && out->from_pin==32 && out->to_node==4,"output renders at member pin");
        Check(input.to_node==5 && output.from_node==5,"projection does not mutate canonical links");
        Check(DisplaySubgraphLink(internal,f.nodes,f.groups)->from_pin==22,"internal connection unchanged");
        Check(CrossesExpandedSubgraphBoundary(1,2,f.groups),"external rewiring while expanded is blocked");
        Check(!CrossesExpandedSubgraphBoundary(2,3,f.groups),"internal rewiring allowed");
        Check(!CrossesExpandedSubgraphBoundary(1,4,f.groups),"ordinary rewiring allowed");
        f.groups[0].expanded=false;
        Check(!IsExpandedSubgraph(5,f.groups),"collapsed wrapper visible");
        Check(DisplaySubgraphLink(input,f.nodes,f.groups)->to_pin==51,"collapsed edge uses wrapper pin");
        Check(!CrossesExpandedSubgraphBoundary(1,5,f.groups),"collapsed boundary editable");
        f.groups[0].expanded=true;
        f.groups[0].input_pin_mappings[0]=999;
        Check(!DisplaySubgraphLink(input,f.nodes,f.groups),"invalid mapping fails without dereferencing missing pin");
        f.groups[0].input_pin_mappings[0]=21;
        auto second=Node(6,61,62,NodeType::Subgraph); f.nodes.push_back(second);
        auto child=Node(7,71,72); f.nodes.push_back(child);
        f.groups.push_back({6,{child},{},{71},{72},true});
        auto cross=DisplaySubgraphLink({20,5,52,6,61},f.nodes,f.groups);
        Check(cross && cross->from_pin==32 && cross->to_pin==71,"two expanded groups project both endpoints");
        Check(CrossesExpandedSubgraphBoundary(3,7,f.groups),"cross-group edits blocked");
        f=Fixture{};
        ImGui::CreateContext(); ImNodes::CreateContext();
        auto& io=ImGui::GetIO(); io.IniFilename=nullptr;
        unsigned char* pixels; int width,height; io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
        Check(Frame(f,false)==-1,"no action without click");
        const ImVec2 button=button_center;
        io.AddMousePosEvent(button.x,button.y); Frame(f,false);
        io.AddMouseButtonEvent(0,true); Check(Frame(f,false)==-1,"button press defers mutation");
        io.AddMouseButtonEvent(0,false); Check(Frame(f,false)==5,"button release requests collapse for correct wrapper");
        io.AddMouseButtonEvent(0,true); Frame(f,true);
        io.AddMouseButtonEvent(0,false); Check(Frame(f,true)==-1,"busy frame cannot request collapse");
        Check(Frame(f,false,0.5f)==-1,"zoomed frame renders");
        Check(Frame(f,false,1.5f)==-1,"larger frame renders");
        ImNodes::DestroyContext(); ImGui::DestroyContext();
        std::cout<<"PASS: "<<checks<<" subgraph presentation checks\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
