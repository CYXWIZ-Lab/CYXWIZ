// Quality Analyzer measurements (TOFIX140) against OpenCV.
//
// fixtures/image_quality_opencv.json (generate_image_quality_fixtures.py)
// holds images and their OpenCV blur / brightness / contrast / difference
// hash; the backend measures the same rows on the ArrayFire device and must
// match on every backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND). Then the
// near-duplicate search, the judgement and the refusals.
#include "computation_truth/test_device_selection.h"

#include <cyxwiz/image_quality.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
namespace img = cyxwiz::image;

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Near(double actual, double expected, double relative, double absolute) {
    return std::abs(actual - expected) <= absolute + relative * std::abs(expected);
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto local = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                       "image_quality_opencv.json";
    return std::filesystem::exists(local) ? local : std::filesystem::path(CYXWIZ_IMAGE_QUALITY_FIXTURE);
}

void CheckFixtures(const std::filesystem::path& path) {
    std::ifstream in(path);
    Check(in.good(), "fixture opens: " + path.string());
    const json fixture = json::parse(in);
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name");
        const auto shape = c.at("rows").at("shape").get<std::vector<size_t>>();  // [N, H, W, C]
        const auto values = c.at("rows").at("values").get<std::vector<float>>();
        const img::ImageShape image{shape[1], shape[2], shape[3]};
        const cyxwiz::Tensor rows({shape[0], image.Size()}, values.data(), cyxwiz::DataType::Float32);
        const auto metrics = img::MeasureImageQuality(rows, image);
        Check(metrics.size() == shape[0], name + ": one result per image");
        for (size_t i = 0; i < metrics.size(); ++i) {
            const json& want = c.at("metrics")[i];
            const std::string at = name + " image " + std::to_string(i);
            Check(Near(metrics[i].blur, want.at("blur").get<double>(), 2e-4, 1e-3),
                  at + " blur " + std::to_string(metrics[i].blur) + " vs " + want.at("blur").dump());
            Check(Near(metrics[i].brightness, want.at("brightness").get<double>(), 0.0, 1e-3), at + " brightness");
            Check(Near(metrics[i].contrast, want.at("contrast").get<double>(), 0.0, 1e-5), at + " contrast");
            Check(metrics[i].hash == std::stoull(want.at("hash").get<std::string>()), at + " hash");
        }
        // Image 4 is image 0 with faint noise: a near-duplicate.
        Check(img::HashDistance(metrics[0].hash, metrics[4].hash) <= 4, name + ": near-duplicate found");
        Check(img::HashDistance(metrics[0].hash, metrics[5].hash) > 4, name + ": a different scene is not");
    }
}

void CheckNearDuplicates() {
    // 0 and 2 differ in 3 bits, 3 in 5 bits from 0, 4 equals 2: with 4 bits,
    // 2 and 4 are duplicates of 0 (the first kept one), 1 and 3 are kept.
    const uint64_t a = 0x0123456789abcdefull;
    const std::vector<uint64_t> hashes = {a, ~a, a ^ 0x7ull, a ^ 0x1full, a ^ 0x7ull};
    const auto dup = img::FindNearDuplicates(hashes, 4);
    Check(dup == std::vector<int64_t>({-1, -1, 0, -1, 0}), "near-duplicates keep the first of each group");
    Check(img::FindNearDuplicates(hashes, 0) == std::vector<int64_t>({-1, -1, -1, -1, 2}),
          "distance 0 only matches equal hashes");

    // A chain: 1 is 3 bits from 0, 2 is 3 bits from 1 but 6 from 0. 1 is
    // left out, so 2 compares only against kept images and stays.
    const std::vector<uint64_t> chain = {a, a ^ 0x7ull, a ^ 0x3full};
    Check(img::FindNearDuplicates(chain, 4) == std::vector<int64_t>({-1, 0, -1}), "only kept images absorb later ones");

    // Many images: enough for several device tiles, every 1000th a copy of image 0.
    std::vector<uint64_t> many(5000);
    uint64_t state = 0x9e3779b97f4a7c15ull;
    for (auto& h : many) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        h = state;
    }
    for (size_t i = 1000; i < many.size(); i += 1000) many[i] = many[0];
    const auto found = img::FindNearDuplicates(many, 2);
    size_t total = 0;
    for (size_t i = 0; i < found.size(); ++i) {
        if (found[i] >= 0) ++total;
        if (i % 1000 == 0 && i > 0) Check(found[i] == 0, "copy " + std::to_string(i) + " points at image 0");
    }
    Check(total == 4, "random hashes are not near each other: " + std::to_string(total));
}

void CheckJudgement() {
    std::vector<img::ImageQualityMetrics> m(5);
    m[0] = {1000.0f, 120.0f, 0.2f, 0};
    m[1] = {100.0f, 120.0f, 0.2f, 0};   // blurry
    m[2] = {1000.0f, 30.0f, 0.05f, 0};  // dark and low contrast
    m[3] = {1000.0f, 240.0f, 0.2f, 0};  // bright
    m[4] = {1000.0f, 120.0f, 0.2f, 0};  // duplicate of 0
    const std::vector<int64_t> dup = {-1, -1, -1, -1, 0};
    img::ImageQualityChecks settings;
    const auto r = img::JudgeImageQuality(m, dup, settings);
    Check(r[0] == 0, "a good image passes");
    Check(r[1] == img::kQualityBlurry, "blurry");
    Check(r[2] == (img::kQualityDark | img::kQualityLowContrast), "dark and low contrast");
    Check(r[3] == img::kQualityBright, "bright");
    Check(r[4] == img::kQualityDuplicate, "duplicate");

    settings.blur = false;
    settings.duplicates = false;
    const auto off = img::JudgeImageQuality(m, dup, settings);
    Check(off[1] == 0 && off[4] == 0, "a check that is off rejects nothing");
}

void CheckRefusals() {
    Check(!img::ValidateImageQualityShape({7, 32, 3}).empty(), "fewer than 8 rows refused");
    Check(!img::ValidateImageQualityShape({32, 8, 3}).empty(), "fewer than 9 columns refused");
    Check(!img::ValidateImageQualityShape({32, 32, 4}).empty(), "4 channels refused");
    Check(img::ValidateImageQualityShape({8, 9, 1}).empty(), "8 x 9 grey accepted");

    img::ImageQualityChecks settings;
    Check(img::ValidateImageQualityChecks(settings).empty(), "defaults are valid");
    settings.brightness_min = 230.0f;
    Check(!img::ValidateImageQualityChecks(settings).empty(), "darkest above brightest refused");
    settings.brightness = false;
    Check(img::ValidateImageQualityChecks(settings).empty(), "an off check is not validated");
    settings.contrast_min = 1.5f;
    Check(!img::ValidateImageQualityChecks(settings).empty(), "contrast above 1 refused");
    settings.contrast_min = 0.1f;
    settings.duplicate_bits = 40;
    Check(!img::ValidateImageQualityChecks(settings).empty(), "duplicate distance above 32 refused");

    bool threw = false;
    try {
        const float pixel[3] = {0.0f, 0.0f, 0.0f};
        img::MeasureImageQuality(cyxwiz::Tensor({1, 3}, pixel, cyxwiz::DataType::Float32), {1, 1, 3});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "measuring a 1 x 1 image is refused");
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested device");
    CheckFixtures(FixturePath(argv[0]));
    CheckNearDuplicates();
    CheckJudgement();
    CheckRefusals();
    std::cout << "image quality matches OpenCV (" << checks << " checks)\n";
    return 0;
}
