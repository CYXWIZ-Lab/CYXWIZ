// The dataset picker's wording (TOFIX140 A): origin, size, the tabular filter
// and the closed-picker preview, over catalog entries.
#include "core/dataset_picker_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

cyxwiz::DatasetEntry Entry(std::string name, unsigned backings, size_t rows, size_t columns, std::string label = {}) {
    cyxwiz::DatasetEntry e;
    e.name = std::move(name);
    e.backings = backings;
    e.rows = rows;
    e.columns = columns;
    e.label = std::move(label);
    return e;
}

}  // namespace

int main() {
    using namespace cyxwiz;
    const DatasetEntry input = Entry("ds_datainput_1", kBackingArrow, 8582, 15, "Spotify");
    const DatasetEntry kmeans = Entry("ds_operator_KMeansCluster_5", kBackingArrow, 248, 4);
    // A prepared step result read back from the prepared data cache (TOFIX178 S0).
    const DatasetEntry prepared = Entry("prepared_StandardScaler_3", kBackingParquet, 8582, 15);
    const DatasetEntry legacy = Entry("old", kBackingLegacy, 10, 2);
    DatasetEntry images = Entry("catdog", kBackingImage, 2000, 0, "Cats and dogs");
    images.modality = DatasetModality::Image;
    images.classes = 2;

    Check(std::string(DatasetOriginText(input)) == "Data Input", "Data Input origin");
    Check(std::string(DatasetOriginText(kmeans)) == "node result", "a pipeline node output is a node result");
    Check(std::string(DatasetOriginText(prepared)) == "node result", "a prepared result read from disk is a node result");
    Check(std::string(DatasetOriginText(legacy)) == "older dataset", "legacy origin");

    Check(DatasetSizeText(input) == "8,582 rows \xC2\xB7 15 cols", "size text: " + DatasetSizeText(input));
    Check(DatasetSizeText(images) == "2,000 files \xC2\xB7 2 classes", "image size text: " + DatasetSizeText(images));

    Check(IsTabularDataset(input) && IsTabularDataset(kmeans) && IsTabularDataset(prepared), "tables are tabular");
    Check(!IsTabularDataset(images) && !IsTabularDataset(legacy), "images and legacy entries are not tables");

    const std::vector<DatasetEntry> entries = {input, kmeans};
    Check(DatasetPickerPreview(entries, "") == "Select a dataset...", "nothing picked");
    Check(DatasetPickerPreview(entries, "ds_datainput_1") ==
              "Spotify  \xC2\xB7  Data Input  \xC2\xB7  8,582 rows \xC2\xB7 15 cols",
          "picked preview: " + DatasetPickerPreview(entries, "ds_datainput_1"));
    Check(DatasetPickerPreview(entries, "gone") == "gone  (no longer loaded)", "a dataset that went away");
    std::cout << "dataset picker presentation: ok" << std::endl;
    return 0;
}
