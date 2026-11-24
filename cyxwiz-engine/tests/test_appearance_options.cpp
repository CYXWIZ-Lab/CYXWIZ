#include "../src/core/appearance_options.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace cyxwiz::appearance;
    Check(NormalizeUiTextPx(15) == 15, "default interface size kept");
    Check(NormalizeUiTextPx(16) == 15 && NormalizeUiTextPx(18) == 17, "nearest interface size");
    Check(NormalizeUiTextPx(0) == 13 && NormalizeUiTextPx(99) == 20, "out of range snaps to ends");
    Check(UiTextIndex(kDefaultUiTextPx) == 1, "default index");
    Check(NormalizeCodeScale(1.3f) == 1.3f && NormalizeCodeScale(1.4f) == 1.3f,
          "nearest code scale");
    Check(NormalizeCodeScale(std::numeric_limits<float>::quiet_NaN()) == kDefaultCodeScale,
          "bad saved code scale falls back to the default");
    Check(kCodeTextSizes[CodeScaleIndex(2.0f)].pixels == 24, "code pixels");
    Check(NormalizeThemePreset(3, 19) == 3 && NormalizeThemePreset(-1, 19) == 0 &&
              NormalizeThemePreset(40, 19) == 0,
          "theme preset range");
    Check(kNodeIconPacks.size() == 6 && kDefaultNodeIconPack == 0 &&
              std::string(kNodeIconPacks[kDefaultNodeIconPack]) == "Font Awesome (default)",
          "node icon packs, Font Awesome first and the default");
    Check(NormalizeNodeIconPack(1) == 1 && NormalizeNodeIconPack(5) == 5, "saved pack kept");
    Check(NormalizeNodeIconPack(-1) == 0 && NormalizeNodeIconPack(6) == 0 &&
              NormalizeNodeIconPack(99) == 0,
          "a saved pack out of range falls back to Font Awesome");
    Check(!IsLightBackground(0.059f, 0.075f, 0.098f), "CyxWiz Dark window is dark");
    Check(IsLightBackground(0.96f, 0.96f, 0.97f), "light window");
    Check(Luminance(1, 1, 1) > 0.99f && Luminance(0, 0, 0) < 0.01f, "luminance ends");
    if (failures) {
        std::cerr << failures << " appearance option check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Appearance option tests passed\n";
    return EXIT_SUCCESS;
}
