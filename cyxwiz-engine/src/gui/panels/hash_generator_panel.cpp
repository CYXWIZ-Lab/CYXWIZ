#include "hash_generator_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

HashGeneratorPanel::HashGeneratorPanel() {
    strcpy(text_buffer_, "Hello, World!");
    spdlog::info("HashGeneratorPanel initialized");
}

HashGeneratorPanel::~HashGeneratorPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void HashGeneratorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(650, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_FINGERPRINT " Hash Generator###HashGeneratorPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderInputSection();
            ImGui::Separator();
            RenderResults();
            ImGui::Separator();
            RenderVerification();
        }
    }
    ImGui::End();
}

void HashGeneratorPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Hash")) {
        HashAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        ClearAll();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Algorithm selection
    const char* algorithms[] = { "MD5", "SHA-1", "SHA-256", "SHA-512", "All" };
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("Algorithm", &algorithm_idx_, algorithms, IM_ARRAYSIZE(algorithms));

    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("| %.2f ms", result_.compute_time_ms);
    }
}

void HashGeneratorPanel::RenderInputSection() {
    ImGui::Text(ICON_FA_KEYBOARD " Input");

    // Input mode tabs
    if (ImGui::BeginTabBar("InputModeTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_FONT " Text")) {
            input_mode_ = 0;

            float available_height = ImGui::GetContentRegionAvail().y * 0.3f;
            ImGui::InputTextMultiline("##TextInput", text_buffer_, sizeof(text_buffer_),
                                      ImVec2(-1, available_height),
                                      ImGuiInputTextFlags_AllowTabInput);

            // Show input size
            size_t len = strlen(text_buffer_);
            ImGui::TextDisabled("%zu bytes", len);

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_FILE " File")) {
            input_mode_ = 1;

            ImGui::SetNextItemWidth(-80);
            ImGui::InputText("##FilePath", file_path_buffer_, sizeof(file_path_buffer_));
            ImGui::SameLine();
            if (ImGui::Button("Browse")) {
                BrowseFile();
            }

            // Show file info if exists
            if (strlen(file_path_buffer_) > 0) {
                std::ifstream file(file_path_buffer_, std::ios::binary | std::ios::ate);
                if (file.is_open()) {
                    auto size = file.tellg();
                    ImGui::TextDisabled("File size: %lld bytes", static_cast<long long>(size));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui::Text("File not found or cannot be opened");
                    ImGui::PopStyleColor();
                }
            }

            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void HashGeneratorPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter text or select a file and click 'Hash'");
        return;
    }

    ImGui::Text(ICON_FA_FINGERPRINT " Hash Results");
    ImGui::Separator();

    // Helper lambda to render hash row
    auto render_hash_row = [this](const char* name, const std::string& hash, bool highlight = false) {
        if (hash.empty()) return;

        ImGui::PushID(name);

        if (ImGui::SmallButton(ICON_FA_COPY)) {
            CopyHash(hash);
        }
        ImGui::SameLine();

        if (highlight) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
        }

        ImGui::Text("%s:", name);
        ImGui::SameLine();

        // Wrap long hashes
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
        ImGui::TextUnformatted(hash.c_str());
        ImGui::PopTextWrapPos();

        if (highlight) {
            ImGui::PopStyleColor();
        }

        ImGui::PopID();
    };

    // Show selected or all algorithms
    const char* algorithms[] = { "md5", "sha1", "sha256", "sha512" };
    std::string selected_algo = algorithm_idx_ < 4 ? algorithms[algorithm_idx_] : "";

    if (!result_.md5_hash.empty()) {
        render_hash_row("MD5", result_.md5_hash, result_.algorithm == "md5" || result_.algorithm == "all");
    }
    if (!result_.sha1_hash.empty()) {
        render_hash_row("SHA-1", result_.sha1_hash, result_.algorithm == "sha1" || result_.algorithm == "all");
    }
    if (!result_.sha256_hash.empty()) {
        render_hash_row("SHA-256", result_.sha256_hash, result_.algorithm == "sha256" || result_.algorithm == "all");
    }
    if (!result_.sha512_hash.empty()) {
        render_hash_row("SHA-512", result_.sha512_hash, result_.algorithm == "sha512" || result_.algorithm == "all");
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Input size: %zu bytes | Compute time: %.2f ms",
                        result_.input_size, result_.compute_time_ms);
}

void HashGeneratorPanel::RenderVerification() {
    ImGui::Text(ICON_FA_CHECK_DOUBLE " Hash Verification");
    ImGui::Separator();

    ImGui::SetNextItemWidth(-100);
    ImGui::InputText("Expected Hash", expected_hash_buffer_, sizeof(expected_hash_buffer_));
    ImGui::SameLine();
    if (ImGui::Button("Verify")) {
        VerifyAsync();
    }

    if (has_verification_result_) {
        ImGui::Spacing();
        if (verification_result_) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
            ImGui::Text(ICON_FA_CHECK " Hashes match!");
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
            ImGui::Text(ICON_FA_XMARK " Hashes do not match!");
            ImGui::PopStyleColor();
        }
    }
}

void HashGeneratorPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Computing hash...");
}

void HashGeneratorPanel::HashAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();
    has_verification_result_ = false;

    const char* algorithms[] = { "md5", "sha1", "sha256", "sha512", "all" };
    std::string algorithm = algorithms[algorithm_idx_];

    std::string input_text;
    std::string file_path;

    if (input_mode_ == 0) {
        input_text = text_buffer_;
        if (input_text.empty()) {
            error_message_ = "No input text";
            is_computing_ = false;
            return;
        }
    } else {
        file_path = file_path_buffer_;
        if (file_path.empty()) {
            error_message_ = "No file selected";
            is_computing_ = false;
            return;
        }
    }

    compute_thread_ = std::make_unique<std::thread>([this, algorithm, input_text, file_path]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            if (!file_path.empty()) {
                result_ = Utilities::HashFile(file_path, algorithm);
            } else {
                result_ = Utilities::HashText(input_text, algorithm);
            }

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Hash computed: {} algorithm, {} bytes",
                            algorithm, result_.input_size);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void HashGeneratorPanel::VerifyAsync() {
    std::string expected = expected_hash_buffer_;
    if (expected.empty()) {
        has_verification_result_ = false;
        return;
    }

    // Convert to lowercase for comparison
    std::transform(expected.begin(), expected.end(), expected.begin(), ::tolower);

    // Determine algorithm from expected hash length
    std::string algorithm;
    if (expected.length() == 32) {
        algorithm = "md5";
    } else if (expected.length() == 40) {
        algorithm = "sha1";
    } else if (expected.length() == 64) {
        algorithm = "sha256";
    } else if (expected.length() == 128) {
        algorithm = "sha512";
    } else {
        error_message_ = "Invalid hash length. Expected 32 (MD5), 40 (SHA-1), 64 (SHA-256), or 128 (SHA-512) characters.";
        has_verification_result_ = false;
        return;
    }

    std::string input_text;
    if (input_mode_ == 0) {
        input_text = text_buffer_;
    }

    if (input_mode_ == 0) {
        verification_result_ = Utilities::VerifyHash(input_text, expected, algorithm);
    } else {
        // For file verification, compute the hash and compare
        HashResult file_hash = Utilities::HashFile(file_path_buffer_, algorithm);
        if (file_hash.success) {
            std::string computed;
            if (algorithm == "md5") computed = file_hash.md5_hash;
            else if (algorithm == "sha1") computed = file_hash.sha1_hash;
            else if (algorithm == "sha256") computed = file_hash.sha256_hash;
            else if (algorithm == "sha512") computed = file_hash.sha512_hash;

            std::transform(computed.begin(), computed.end(), computed.begin(), ::tolower);
            verification_result_ = (computed == expected);
        } else {
            error_message_ = file_hash.error_message;
            has_verification_result_ = false;
            return;
        }
    }

    has_verification_result_ = true;
    spdlog::info("Hash verification: {}", verification_result_ ? "MATCH" : "MISMATCH");
}

void HashGeneratorPanel::CopyHash(const std::string& hash) {
    ImGui::SetClipboardText(hash.c_str());
    spdlog::info("Hash copied to clipboard");
}

void HashGeneratorPanel::ClearAll() {
    text_buffer_[0] = '\0';
    file_path_buffer_[0] = '\0';
    expected_hash_buffer_[0] = '\0';
    has_result_ = false;
    has_verification_result_ = false;
    error_message_.clear();
    result_ = HashResult();
}

void HashGeneratorPanel::BrowseFile() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char szFile[512] = {0};

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        strcpy(file_path_buffer_, ofn.lpstrFile);
    }
#else
    // For non-Windows, just show a message
    spdlog::info("File browser not implemented for this platform. Enter file path manually.");
#endif
}

} // namespace cyxwiz
