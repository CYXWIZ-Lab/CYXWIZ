#include "audio_dataset.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace cyxwiz {

namespace {

// Detect the most likely CSV delimiter by counting candidates in the header line.
// Picks whichever character (',', ';', '\t') appears most often.
char DetectDelimiter(const std::string& sample_line) {
    int counts[3] = {0, 0, 0};  // comma, semicolon, tab
    for (char c : sample_line) {
        if (c == ',') counts[0]++;
        else if (c == ';') counts[1]++;
        else if (c == '\t') counts[2]++;
    }
    int best = 0;
    if (counts[1] > counts[best]) best = 1;
    if (counts[2] > counts[best]) best = 2;
    static const char delims[3] = {',', ';', '\t'};
    return delims[best];
}

// Minimal CSV row parser. Handles double-quoted fields with embedded delimiter
// and escaped "" inside quoted fields. Reads one *physical* line at a time;
// rows whose quoted fields span multiple lines (e.g. multi-line transcripts)
// will deserialize as several broken rows — those get skipped by the caller's
// "row.size() must match header count" check. Lossy on those rows but doesn't
// crash the whole load, which is the right tradeoff for noisy real-world CSVs.
std::vector<std::string> ParseCSVLine(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"';
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == delim && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

// Trim ASCII whitespace and BOM from both ends.
void Trim(std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' ||
                           s.front() == '\n' || (unsigned char)s.front() == 0xEF ||
                           (unsigned char)s.front() == 0xBB || (unsigned char)s.front() == 0xBF)) {
        s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
}

bool IsAudioExt(const std::string& ext_in) {
    std::string e = ext_in;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".wav" || e == ".flac" || e == ".ogg" ||
           e == ".aiff" || e == ".aif" || e == ".mp3";
}

} // namespace

AudioDataset::AudioDataset(const std::string& directory, const AudioDatasetConfig& config)
    : config_(config) {
    if (!config_.csv_path.empty()) {
        LoadFromCSV(directory);
    } else {
        ScanDirectory(directory);
    }

    // Seed feature_rows_/feature_cols_ with the closed-form prediction in
    // case the probe below fails. The probe is the source of truth when
    // it succeeds — these values exist only as a graceful fallback so
    // GetInfo() returns *something* sensible.
    {
        int max_samples = static_cast<int>(config_.target_sr * config_.max_duration);
        feature_cols_ = (max_samples + config_.n_fft) / config_.hop_length;
        switch (config_.feature_type) {
            case AudioDatasetConfig::FeatureType::Spectrogram:
                feature_rows_ = config_.n_fft / 2 + 1; break;
            case AudioDatasetConfig::FeatureType::MelSpectrogram:
                feature_rows_ = config_.n_mels; break;
            case AudioDatasetConfig::FeatureType::MFCC:
                feature_rows_ = config_.n_mfcc; break;
        }
    }

    // Probe samples until we find a non-silent one, to discover the
    // *actual* feature shape produced by ComputeMelSpectrogram / ComputeMFCC.
    // The closed-form prediction below is wrong by 1-3 frames depending on
    // the framing convention inside the FFTW spectrogram code, and a
    // mismatch here causes AudioDatasetBatcher to zero-fill every sample.
    //
    // We need to skip silent files (they're rejected by ExtractFeatures'
    // silent-file guard) because real-world datasets like Kaggle's
    // "Binary Drone Audio" have silent files near the top of the
    // alphabetical scan. Try up to 32 samples before giving up — that
    // tolerates datasets with up to ~97% silent files at the front.
    int probed_rows = 0;
    int probed_cols = 0;
    size_t silent_probes = 0;
    const size_t kMaxProbeAttempts = 32;
    for (size_t i = 0; i < file_paths_.size() && i < kMaxProbeAttempts; ++i) {
        AudioFeatures probe = ExtractFeatures(file_paths_[i]);
        if (probe.valid && probe.rows > 0 && probe.cols > 0) {
            probed_rows = probe.rows;
            probed_cols = probe.cols;
            if (silent_probes > 0) {
                spdlog::info("AudioDataset: probe found non-silent sample at index {} "
                             "after {} silent/invalid files", i, silent_probes);
            }
            break;
        } else if (probe.valid && !probe.data.empty()) {
            // Some feature types may not populate rows/cols on the struct;
            // fall back to dividing by the expected row count from config.
            int rows_guess = 0;
            switch (config_.feature_type) {
                case AudioDatasetConfig::FeatureType::Spectrogram:
                    rows_guess = config_.n_fft / 2 + 1; break;
                case AudioDatasetConfig::FeatureType::MelSpectrogram:
                    rows_guess = config_.n_mels; break;
                case AudioDatasetConfig::FeatureType::MFCC:
                    rows_guess = config_.n_mfcc; break;
            }
            if (rows_guess > 0 && probe.data.size() % rows_guess == 0) {
                probed_rows = rows_guess;
                probed_cols = static_cast<int>(probe.data.size()) / rows_guess;
                break;
            }
        } else {
            // Invalid probe (silent file, decode error, etc.) — try the next one.
            ++silent_probes;
        }
    }

    if (probed_rows > 0 && probed_cols > 0) {
        feature_rows_ = probed_rows;
        feature_cols_ = probed_cols;
        spdlog::info("AudioDataset: probed feature shape [{} x {}] ({} elements)",
                     feature_rows_, feature_cols_,
                     feature_rows_ * feature_cols_);
    } else {
        spdlog::warn("AudioDataset: probe failed, using predicted shape [{} x {}] — "
                     "training may fail with size mismatch",
                     feature_rows_, feature_cols_);
    }

}

void AudioDataset::ScanDirectory(const std::string& directory) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("AudioDataset: directory not found: " + directory);
    }

    // Supported extensions
    auto is_audio = [](const std::string& ext) {
        return ext == ".wav" || ext == ".flac" || ext == ".ogg" || ext == ".aiff" || ext == ".aif";
    };

    if (config_.labeled_subdirs) {
        // Each subdirectory is a class
        std::vector<std::string> subdirs;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_directory()) {
                subdirs.push_back(entry.path().filename().string());
            }
        }
        std::sort(subdirs.begin(), subdirs.end());
        class_names_ = subdirs;

        for (int cls = 0; cls < static_cast<int>(subdirs.size()); cls++) {
            fs::path cls_dir = fs::path(directory) / subdirs[cls];
            for (const auto& entry : fs::recursive_directory_iterator(cls_dir)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (is_audio(ext)) {
                        file_paths_.push_back(entry.path().string());
                        labels_.push_back(cls);
                    }
                }
            }
        }
    } else {
        // Flat directory, no labels
        class_names_ = {"audio"};
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (is_audio(ext)) {
                    file_paths_.push_back(entry.path().string());
                    labels_.push_back(0);
                }
            }
        }
    }

    spdlog::info("AudioDataset: found {} audio files in {} classes from '{}'",
                 file_paths_.size(), class_names_.size(), directory);

    if (file_paths_.empty()) {
        throw std::runtime_error("AudioDataset: no audio files found in: " + directory);
    }

    // Initialize split indices
    all_indices_.resize(file_paths_.size());
    for (size_t i = 0; i < file_paths_.size(); i++) all_indices_[i] = i;
}

void AudioDataset::LoadFromCSV(const std::string& directory) {
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("AudioDataset: audio folder not found: " + directory);
    }
    if (!fs::exists(config_.csv_path)) {
        throw std::runtime_error("AudioDataset: labels CSV not found: " + config_.csv_path);
    }

    std::ifstream file(config_.csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("AudioDataset: failed to open CSV: " + config_.csv_path);
    }

    // Read first non-empty line to detect delimiter and headers.
    std::string header_line;
    while (std::getline(file, header_line)) {
        Trim(header_line);
        if (!header_line.empty()) break;
    }
    if (header_line.empty()) {
        throw std::runtime_error("AudioDataset: CSV is empty: " + config_.csv_path);
    }

    char delim = DetectDelimiter(header_line);
    std::vector<std::string> headers = ParseCSVLine(header_line, delim);
    for (auto& h : headers) Trim(h);

    spdlog::info("AudioDataset: CSV delimiter='{}', {} columns", delim, headers.size());

    // Find filename + label column indices. User-supplied col names take
    // priority over auto-detect. Auto-detect uses ranked priority lists
    // covering the most common conventions (filename/file/path/id, then
    // label/class/category/target/diagnosis).
    auto find_col = [&](const std::string& want, const std::vector<std::string>& priorities,
                        int default_idx) -> int {
        if (!want.empty()) {
            for (size_t i = 0; i < headers.size(); ++i) {
                if (headers[i] == want) return static_cast<int>(i);
            }
        }
        for (const auto& p : priorities) {
            for (size_t i = 0; i < headers.size(); ++i) {
                std::string lower = headers[i];
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower == p) return static_cast<int>(i);
            }
        }
        return default_idx;
    };

    int filename_idx = find_col(config_.filename_col,
        {"filename", "file_name", "file", "path", "audio_path", "audio_file",
         "audio", "wav", "id", "name", "call_id", "clip_id", "utterance_id"},
        0);
    int label_idx = find_col(config_.label_col,
        {"label", "labels", "class", "class_name", "category", "categories",
         "target", "y", "emotion", "sentiment", "intent", "diagnosis"},
        std::min(1, static_cast<int>(headers.size()) - 1));

    spdlog::info("AudioDataset: filename column = '{}' (idx {}), label column = '{}' (idx {})",
                 filename_idx < (int)headers.size() ? headers[filename_idx] : "?", filename_idx,
                 label_idx < (int)headers.size() ? headers[label_idx] : "?", label_idx);

    if (filename_idx < 0 || label_idx < 0 ||
        filename_idx >= (int)headers.size() || label_idx >= (int)headers.size()) {
        throw std::runtime_error("AudioDataset: could not resolve filename/label columns");
    }

    // Build a stem→full-path index of every audio file in the folder tree
    // so the CSV's filename values can be resolved whether or not they
    // include an extension. Mirrors the image FlatWithCSV pattern.
    std::map<std::string, std::string> stem_to_path;
    std::map<std::string, std::string> name_to_path;
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (!IsAudioExt(p.extension().string())) continue;
        stem_to_path[p.stem().string()] = p.string();
        name_to_path[p.filename().string()] = p.string();
    }
    spdlog::info("AudioDataset: scanned {} audio files in folder '{}'",
                 stem_to_path.size(), directory);

    // Stream the rest of the CSV row by row. Build a label_str → int map
    // on the fly (first-seen order) so the user doesn't have to encode labels.
    std::map<std::string, int> label_map;
    std::string line;
    size_t total_rows = 0;
    size_t skipped_rows = 0;
    size_t missing_files = 0;
    int header_cols = static_cast<int>(headers.size());

    while (std::getline(file, line)) {
        Trim(line);
        if (line.empty()) continue;
        ++total_rows;

        std::vector<std::string> row = ParseCSVLine(line, delim);
        // Skip rows that don't have enough columns (typically multi-line
        // quoted fields that broke the physical-line assumption).
        if ((int)row.size() < std::max(filename_idx, label_idx) + 1) {
            ++skipped_rows;
            continue;
        }

        std::string fname = row[filename_idx];
        std::string lbl_str = row[label_idx];
        Trim(fname);
        Trim(lbl_str);
        if (fname.empty() || lbl_str.empty()) {
            ++skipped_rows;
            continue;
        }

        // Resolve filename to a real path. Try (in order):
        //   1. exact filename (with extension) lookup
        //   2. stem (no extension) lookup
        //   3. join with directory and exists() check
        std::string resolved;
        auto it1 = name_to_path.find(fname);
        if (it1 != name_to_path.end()) {
            resolved = it1->second;
        } else {
            auto it2 = stem_to_path.find(fname);
            if (it2 != stem_to_path.end()) {
                resolved = it2->second;
            } else {
                fs::path candidate = fs::path(directory) / fname;
                if (fs::exists(candidate)) {
                    resolved = candidate.string();
                }
            }
        }

        if (resolved.empty()) {
            ++missing_files;
            continue;
        }

        if (label_map.find(lbl_str) == label_map.end()) {
            label_map[lbl_str] = static_cast<int>(label_map.size());
        }
        file_paths_.push_back(resolved);
        labels_.push_back(label_map[lbl_str]);
    }
    file.close();

    // Class names sorted by assigned id (first-seen order).
    class_names_.assign(label_map.size(), "");
    for (const auto& [name, id] : label_map) {
        if (id >= 0 && id < (int)class_names_.size()) class_names_[id] = name;
    }

    spdlog::info("AudioDataset: CSV loaded {} samples in {} classes "
                 "({} skipped rows, {} unresolved filenames out of {} total rows)",
                 file_paths_.size(), class_names_.size(),
                 skipped_rows, missing_files, total_rows);

    if (file_paths_.empty()) {
        throw std::runtime_error("AudioDataset: CSV produced 0 usable samples — "
                                 "check filename column matches actual audio files");
    }

    all_indices_.resize(file_paths_.size());
    for (size_t i = 0; i < file_paths_.size(); ++i) all_indices_[i] = i;
}

AudioFeatures AudioDataset::ExtractFeatures(const std::string& filepath) const {
    // Load audio
    AudioData audio = AudioProcessing::LoadAudio(filepath, config_.target_sr);
    if (!audio.valid) {
        AudioFeatures empty;
        empty.error_message = audio.error_message;
        return empty;
    }

    // Silent-file detection. Some public audio datasets (the Kaggle
    // "Binary Drone Audio" set is one example) contain files with valid
    // headers but all-zero sample data — the mel spectrogram of silence
    // is constant log(1e-10) = -23.0259, which looks identical to a bug
    // downstream. Detect and reject here so the failure mode is obvious
    // in the log instead of appearing as garbage mel features.
    //
    // Only checks a cheap prefix of samples; a real ML pipeline would
    // run RMS energy or voice-activity detection, but that's beyond
    // scope for now.
    bool any_nonzero = false;
    const size_t probe_limit = std::min<size_t>(audio.samples.size(), 2048);
    for (size_t i = 0; i < probe_limit; ++i) {
        if (audio.samples[i] != 0.0f) { any_nonzero = true; break; }
    }
    if (!any_nonzero) {
        // Double-check by scanning the rest in case the silent prefix
        // is misleading (some files have intro silence).
        for (size_t i = probe_limit; i < audio.samples.size(); ++i) {
            if (audio.samples[i] != 0.0f) { any_nonzero = true; break; }
        }
    }
    if (!any_nonzero) {
        AudioFeatures silent;
        silent.error_message = "audio file contains only silence (all-zero samples)";
        return silent;
    }

    // Pad or truncate to max_duration
    size_t max_samples = static_cast<size_t>(config_.target_sr * config_.max_duration);
    if (audio.num_samples > max_samples) {
        audio.samples.resize(max_samples);
        audio.num_samples = max_samples;
    } else if (audio.num_samples < max_samples) {
        audio.samples.resize(max_samples, 0.0f);  // Zero-pad
        audio.num_samples = max_samples;
    }
    audio.duration_seconds = config_.max_duration;

    // Extract features based on config
    switch (config_.feature_type) {
        case AudioDatasetConfig::FeatureType::Spectrogram: {
            SpectrogramConfig sc;
            sc.n_fft = config_.n_fft;
            sc.hop_length = config_.hop_length;
            return AudioProcessing::ComputeSpectrogram(audio, sc);
        }
        case AudioDatasetConfig::FeatureType::MelSpectrogram: {
            MelConfig mc;
            mc.n_fft = config_.n_fft;
            mc.hop_length = config_.hop_length;
            mc.n_mels = config_.n_mels;
            mc.fmin = config_.fmin;
            mc.fmax = config_.fmax;
            return AudioProcessing::ComputeMelSpectrogram(audio, mc);
        }
        case AudioDatasetConfig::FeatureType::MFCC: {
            MFCCConfig cc;
            cc.n_fft = config_.n_fft;
            cc.hop_length = config_.hop_length;
            cc.n_mels = config_.n_mels;
            cc.fmin = config_.fmin;
            cc.fmax = config_.fmax;
            cc.n_mfcc = config_.n_mfcc;
            return AudioProcessing::ComputeMFCC(audio, cc);
        }
    }

    AudioFeatures empty;
    empty.error_message = "Unknown feature type";
    return empty;
}

size_t AudioDataset::Size() const {
    return file_paths_.size();
}

std::pair<std::vector<float>, int> AudioDataset::GetItem(size_t index) const {
    if (index >= file_paths_.size()) {
        return {{}, -1};
    }

    AudioFeatures features = ExtractFeatures(file_paths_[index]);
    if (!features.valid) {
        // Rate-limit the warning spam. Real-world datasets (e.g. the
        // Kaggle Binary Drone Audio set) can have hundreds of silent
        // files — logging each one per epoch floods the console. Print
        // the first 10 offending files at full detail, then fall back
        // to a count-every-100 summary.
        static std::atomic<size_t> bad_count{0};
        size_t n = bad_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 10) {
            spdlog::warn("AudioDataset: skipping '{}' ({}) [{} total so far]",
                         file_paths_[index], features.error_message, n);
        } else if (n % 100 == 0) {
            spdlog::warn("AudioDataset: {} unusable files encountered so far "
                         "(most recent: '{}' - {})",
                         n, file_paths_[index], features.error_message);
        }
        // Return zeros as fallback so the batch still has uniform shape.
        // Zero samples contribute nothing to gradient when normalized
        // — effectively "ignored" but not removed from the batch count.
        return {std::vector<float>(feature_rows_ * feature_cols_, 0.0f), labels_[index]};
    }

    return {features.data, labels_[index]};
}

DatasetInfo AudioDataset::GetInfo() const {
    DatasetInfo info;
    info.name = "AudioDataset";
    info.num_samples = file_paths_.size();
    info.shape = {static_cast<size_t>(feature_rows_), static_cast<size_t>(feature_cols_)};
    info.num_classes = class_names_.size();
    info.class_names = class_names_;
    return info;
}

} // namespace cyxwiz
