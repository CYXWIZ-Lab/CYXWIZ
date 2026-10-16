// The Data Input dialog's data strip (TOFIX129 B-2), on the sentiment
// project's real numbers.
#include "core/data_input_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cyxwiz::data_input_view;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

StripInput SentimentCsv() {
    StripInput in;
    in.source = "D:/demo/mrcj/datasets/sentiment analysis/sentiment_mental_health.csv";
    in.kind = "CSV table";
    in.file_size = 31457280;
    return in;
}

void CheckFormatting() {
    Check(FormatCount(52681) == "52,681" && FormatCount(7) == "7" && FormatCount(1000000) == "1,000,000",
          "counts group thousands");
    Check(FormatSize(31457280) == "30.0 MB" && FormatSize(512) == "512 B", "sizes: " + FormatSize(31457280));
    Check(FormatSeconds(4.567) == "4.57 s" && FormatSeconds(0.42) == "420 ms", "seconds");
}

void CheckNotLoaded() {
    const Strip s = BuildStrip(SentimentCsv());
    Check(s.title == "sentiment_mental_health.csv", "title: " + s.title);
    Check(s.tone == LoadTone::NotLoaded && s.state == "Not loaded" && s.detail == "CSV table", "not loaded");
    Check(s.facts.size() == 1 && s.facts[0].value == "30.0 MB" && s.facts[0].label == "on disk", "size on disk");
    Check(s.note == "Apply loads it.", "note: " + s.note);
    const Strip none = BuildStrip(StripInput{});
    Check(none.title == "No file or folder chosen" && none.note == "Choose a file or folder in Source.", "no source");
}

void CheckLoadingAndFailed() {
    StripInput in = SentimentCsv();
    in.loading = true;
    in.loading_message = "Reading sentiment_mental_health.csv";
    in.loading_seconds = 2.4;
    Strip s = BuildStrip(in);
    Check(s.tone == LoadTone::Loading && s.state == "Loading: Reading sentiment_mental_health.csv" &&
              s.detail == "2.40 s",
          "loading: " + s.state);
    in = SentimentCsv();
    in.failed = true;
    in.failure = "Error loading text: file not found";
    s = BuildStrip(in);
    Check(s.tone == LoadTone::Failed && s.state == "Not loaded" && s.detail == in.failure &&
              s.note == "Fix the source in Source, then Apply.",
          "failed");
}

void CheckLoaded() {
    StripInput in = SentimentCsv();
    in.loaded = true;
    in.dataset_name = "sentiment_mental_health";
    in.rows = 52681;
    in.cols = 1;
    in.classes = 7;
    in.backend = Backend::Text;
    in.bytes = 31876710;
    in.load_seconds = 2.92;
    Strip s = BuildStrip(in);
    Check(s.tone == LoadTone::Loaded && s.state == "Loaded in 2.92 s", "state: " + s.state);
    Check(s.detail == "as sentiment_mental_health (text)", "detail: " + s.detail);
    Check(s.facts.size() == 3 && s.facts[0].value == "52,681" && s.facts[0].label == "samples" &&
              s.facts[1].value == "7" && s.facts[1].label == "classes" && s.facts[2].label == "on disk",
          "facts: samples, classes, size");
    Check(s.note == "30.4 MB in memory", "held: " + s.note);

    StripInput images;
    images.source = "D:/demo/mrcj/datasets/Cats_Dogs/cat_dog";
    images.loaded = true;
    images.dataset_name = "cat_dog";
    images.rows = 25000;
    images.cols = 1;
    images.classes = 2;
    images.backend = Backend::ImageFolder;
    images.load_seconds = 1.9;
    s = BuildStrip(images);
    Check(s.title == "cat_dog" && s.facts[0].value == "25,000" && s.facts[0].label == "images" &&
              s.note == "decoded per batch at the Resize size; nothing is held",
          "images: " + s.note);

    StripInput disk = SentimentCsv();
    disk.loaded = true;
    disk.rows = 5111808;
    disk.cols = 12;
    disk.backend = Backend::ParquetDisk;
    disk.bytes = 268435456;
    disk.audit_warnings = 2;
    s = BuildStrip(disk);
    Check(s.facts.size() == 4 && s.facts[1].value == "12" && s.facts[1].label == "columns" &&
              s.facts[3].value == "0 / 2",
          "a table on disk: rows, columns, size, audit");
    Check(s.note == "kept on disk (Parquet, 256.0 MB), read a row group at a time", "on disk: " + s.note);

    disk.failed = true;
    disk.failure = "Apply refused by dataset audit.";
    s = BuildStrip(disk);
    Check(s.tone == LoadTone::Loaded &&
              s.note == "The last Apply failed: Apply refused by dataset audit.",
          "a failed Apply over an earlier load keeps it and says so");
}

}  // namespace

int main() {
    CheckFormatting();
    CheckNotLoaded();
    CheckLoadingAndFailed();
    CheckLoaded();
    std::cout << "data input presentation: strip states, facts and holding. OK\n";
    return 0;
}
