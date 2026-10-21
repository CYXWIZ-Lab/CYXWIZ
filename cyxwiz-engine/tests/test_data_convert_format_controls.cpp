#include "../src/gui/data_convert_format_controls.h"
#include <iostream>
#include <stdexcept>

int main() try {
    namespace dc = cyxwiz::data_convert;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(800, 600);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    int checks = 0;
    for (bool xlsx : {false, true}) for (bool hdf5 : {false, true})
        for (auto direction : {dc::Direction::Input, dc::Direction::Output})
            for (const auto original : {"auto", "", "csv", "excel", "xlsx", "hdf", "hdf5", "arrow", "pq", "unsupported", " XLSX "}) {
                std::string value = original;
                ImGui::NewFrame();
                ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_Always);
                ImGui::Begin("DataConvert format controls");
                const bool changed = gui::RenderDataConvertFormatCombo("format", value, direction, {xlsx, hdf5});
                ImGui::End();
                ImGui::Render();
                if (changed || value != original) throw std::runtime_error("render changed a persisted format without input");
                ++checks;
            }
    ImGui::DestroyContext();
    std::cout << "DataConvert format controls: " << checks << " saved-selection checks passed\n";
    return 0;
} catch (const std::exception& error) {
    if (ImGui::GetCurrentContext()) ImGui::DestroyContext();
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
