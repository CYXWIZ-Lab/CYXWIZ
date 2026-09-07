#include "../src/gui/data_preview_table_renderer.h"

#include <cstdlib>
#include <iostream>

namespace {
void Check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}

int main() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(900, 700);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    gui::DataPreviewViewState state;
    const std::vector<cyxwiz::DataPreviewColumn> columns = {{"statement"}, {"status"}};
    const gui::DataPreviewRow row = {"Quoted, multiline\nsentiment text", "positive"};
    ImVec2 next_button;
    ImVec2 previous_button;
    auto frame = [&](int64_t total, float window_width = 640.0f) {
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(window_width, 520), ImGuiCond_Always);
        ImGui::Begin("Preview controls test");
        const auto origin = ImGui::GetCursorScreenPos();
        const auto& style = ImGui::GetStyle();
        const float first_width = ImGui::CalcTextSize("First").x + style.FramePadding.x * 2;
        const float previous_width = ImGui::CalcTextSize("Previous").x + style.FramePadding.x * 2;
        const float nav_y = origin.y + ImGui::GetFrameHeightWithSpacing() + ImGui::GetFrameHeight() / 2;
        previous_button = ImVec2(origin.x + first_width + style.ItemSpacing.x + previous_width / 2, nav_y);
        next_button = ImVec2(origin.x + first_width + previous_width + style.ItemSpacing.x * 2 +
                            (ImGui::CalcTextSize("Next").x + style.FramePadding.x * 2) / 2, nav_y);
        const auto count = gui::RenderDataPreviewControls(state, total);
        const auto start = state.offset;
        int calls = 0;
        const auto result = gui::RenderDataPreviewTable("preview", columns, start, count,
            [&](int64_t index) -> const gui::DataPreviewRow* {
                Check(index >= start && index < start + count, "lookup must stay in current page");
                ++calls;
                return &row;
            }, ImVec2(0, 200), true, "statement", &state);
        Check(calls <= count, "render must not scan rows outside its bounded page");
        gui::RenderDataPreviewCellDetails(state);
        ImGui::End();
        ImGui::Render();
        return std::make_pair(count, result);
    };
    frame(24); // settle first-frame table geometry
    Check(frame(24).first == 10 && state.offset == 0, "default preview page has ten rows");
    const auto click = [&](ImVec2 position) {
        io.AddMousePosEvent(position.x, position.y);
        frame(24);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        frame(24);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        frame(24);
    };
    click(next_button);
    Check(state.offset == 10, "Next click must navigate to the second preview page");
    click(previous_button);
    Check(state.offset == 0, "Previous click must navigate back to the first page");
    click(previous_button);
    Check(state.offset == 0, "Previous must stay disabled at the first page");
    state.offset = 20;
    Check(frame(24).first == 4 && state.offset == 20, "last partial sample page must be bounded");
    state.offset = 500;
    Check(frame(24).first == 4 && state.offset == 20, "refresh to smaller data must clamp old offset");
    Check(frame(0).first == 0 && state.offset == 0, "empty preview has no rows and no stale offset");
    state.rows_per_page = 25;
    Check(frame(24, 340).first == 24, "larger page must remain bounded to available sample");
    state.rows_per_page = 0;
    frame(24);
    Check(state.rows_per_page == 1, "invalid page size must not divide by zero");

    state.selected_cell = row[0];
    state.selected_column = "statement";
    state.selected_row = 3;
    state.inspect_cell = true;
    frame(24);
    frame(24);
    io.AddKeyEvent(ImGuiKey_Escape, true);
    frame(24);
    Check(!state.inspect_cell && state.selected_cell.empty(), "Escape must close and clear the cell reader");
    io.AddKeyEvent(ImGuiKey_Escape, false);
    frame(24);
    Check(!state.inspect_cell, "closed cell reader must not reopen next frame");
    ImGui::DestroyContext();
    std::cout << "Data preview controls passed\n";
}
