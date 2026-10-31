#include "data_studio_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace cyxwiz::studio {

using gui::NodeType;

const char* JobName(Job job) {
    switch (job) {
        case Job::BringIn: return "Bring in";
        case Job::Explore: return "Explore";
        case Job::Prepare: return "Prepare";
        case Job::Validate: return "Validate";
        case Job::Ready: return "ML readiness";
        case Job::Publish: return "Publish";
        case Job::Results: return "Results";
    }
    return "";
}

bool JobOf(NodeType type, Job& job) {
    switch (type) {
        // Data enters through a Data Input; older file nodes are sources too.
        case NodeType::DataInput: case NodeType::CSVFile: case NodeType::ParquetFile: case NodeType::JSONFile:
        case NodeType::ExcelFile: case NodeType::TSVFile: case NodeType::TXTFile: case NodeType::HDF5Dataset:
        case NodeType::FeatherFile: case NodeType::ArrowIPCFile: case NodeType::NumPyFile: case NodeType::TimeSeriesCSV:
        case NodeType::TextCorpusDataset: case NodeType::DataConvert:
            job = Job::BringIn; return true;
        case NodeType::DataProfiler: case NodeType::Plot: case NodeType::Dashboard: case NodeType::DescribeStats:
        case NodeType::CorrelationMatrix: case NodeType::ValueCounts: case NodeType::VisualizeData:
        case NodeType::CrossTabulation: case NodeType::QualityAnalyzer:
        // Time series analysis kept as nodes (Explore > Series and Forecast, TOFIX129 D6).
        case NodeType::TimeSeriesDecomposition: case NodeType::ACFNode: case NodeType::PACFNode:
        case NodeType::StationarityTest: case NodeType::SeasonalityDetector: case NodeType::ARIMAForecaster:
        case NodeType::ExponentialSmoothing:
            job = Job::Explore; return true;
        case NodeType::Subgraph: case NodeType::SQLQuery: case NodeType::FilterRows: case NodeType::SelectColumns:
        case NodeType::JoinTables: case NodeType::GroupByAggregate: case NodeType::SortRows:
        case NodeType::FillMissingValues: case NodeType::RemoveDuplicateRows: case NodeType::PivotTable:
        case NodeType::UnionTables: case NodeType::RenameColumns: case NodeType::SampleRows:
        case NodeType::RowToColumnNames: case NodeType::TableSplitter: case NodeType::CellExtractor:
        case NodeType::CellUpdater: case NodeType::TableCropper: case NodeType::ColumnAppender:
        case NodeType::RowAppender: case NodeType::Unpivot: case NodeType::StringManipulation:
        case NodeType::MathFormula: case NodeType::RuleEngine: case NodeType::TextCleanNode:
        case NodeType::BinningNode: case NodeType::PolynomialFeaturesNode: case NodeType::CalculatorNode:
        case NodeType::UnitConverter: case NodeType::JSONPathExtractor: case NodeType::RegexTester:
        case NodeType::StandardScaler: case NodeType::MinMaxScaler: case NodeType::RobustScaler:
        case NodeType::MaxAbsScaler: case NodeType::QuantileTransform: case NodeType::PowerTransform:
        case NodeType::LogTransform:
        case NodeType::LabelEncoder: case NodeType::OrdinalEncoder: case NodeType::TargetEncoder:
        case NodeType::OutlierDetector:
        // Groups and projections added as steps (Explore > Groups / Projection, TOFIX129 D5).
        case NodeType::KMeansCluster: case NodeType::DBSCANCluster: case NodeType::HierarchicalCluster:
        case NodeType::GMMCluster: case NodeType::PCANode: case NodeType::TSNENode: case NodeType::UMAPNode:
            job = Job::Prepare; return true;
        case NodeType::RowCountCheck: case NodeType::DataValidator:
            job = Job::Validate; return true;
        case NodeType::DataSplit: case NodeType::DataLoader: case NodeType::TimeSeriesSplit:
            job = Job::Ready; return true;
        case NodeType::DataOutput: case NodeType::ExportCSV: case NodeType::ExportParquet: case NodeType::ExportJSON:
        case NodeType::ExportExcel: case NodeType::ExportSQL:
            job = Job::Publish; return true;
        case NodeType::ConfusionMatrixNode: case NodeType::ROCCurveNode: case NodeType::PRCurveNode:
        case NodeType::ClassificationMetricsNode: case NodeType::RegressionMetricsNode:
        case NodeType::HypothesisTest: case NodeType::TestEvaluationNode:
        case NodeType::LearningCurvesNode: case NodeType::FeatureImportanceNode: case NodeType::CrossValidationNode:
            job = Job::Results; return true;
        default:
            return false;
    }
}

std::string Thousands(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + digits : digits;
}

namespace {

const char* TypeLabel(NodeType type) {
    switch (type) {
        case NodeType::DataInput: return "Data Input";
        case NodeType::DataConvert: return "Data Convert";
        case NodeType::DataProfiler: return "Data Profiler";
        case NodeType::Plot: return "Plot";
        case NodeType::Dashboard: return "Dashboard";
        case NodeType::Subgraph: return "Preparation Recipe";
        case NodeType::SQLQuery: return "SQL Query";
        case NodeType::RowCountCheck: return "Check";
        case NodeType::DataValidator: return "Data Validator";
        case NodeType::DataSplit: return "Data Split";
        case NodeType::DataLoader: return "Data Loader";
        case NodeType::TimeSeriesSplit: return "Time Series Split";
        case NodeType::DataOutput: return "Data Output";
        case NodeType::ExportCSV: return "Export CSV";
        case NodeType::ExportParquet: return "Export Parquet";
        case NodeType::ExportJSON: return "Export JSON";
        case NodeType::ExportExcel: return "Export Excel";
        case NodeType::ExportSQL: return "Export SQL";
        case NodeType::ConfusionMatrixNode: return "Confusion Matrix";
        case NodeType::ROCCurveNode: return "ROC Curve";
        case NodeType::PRCurveNode: return "PR Curve";
        case NodeType::ClassificationMetricsNode: return "Classification Metrics";
        case NodeType::TestEvaluationNode: return "Test Evaluation";
        case NodeType::RegressionMetricsNode: return "Regression Metrics";
        default: return nullptr;  // grouped by the caller
    }
}

bool IsDataInput(const GraphNode& n) { return n.type == NodeType::DataInput; }
bool IsCheck(const GraphNode& n) { return n.type == NodeType::RowCountCheck; }
bool IsExport(const GraphNode& n) {
    Job job;
    return JobOf(n.type, job) && job == Job::Publish;
}
bool CheckPassed(const GraphNode& n) { return n.ran && n.error.empty() && n.outcome.rfind("passed", 0) == 0; }

std::string Plural(size_t n, const char* one, const char* many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

std::string FileName(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string Seconds(double s) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), s < 10.0 ? "%.1f s" : "%.0f s", s);
    return buf;
}

const Dataset* DatasetNamed(const Input& in, const std::string& name) {
    for (const auto& d : in.datasets)
        if (d.name == name) return &d;
    return nullptr;
}

// A Data Input's dataset: the one its last run registered, else its
// dataset_name, else the loaded dataset the catalog labels with the node's name.
const Dataset* DatasetOfInput(const Input& in, const GraphNode& n) {
    if (const Dataset* d = DatasetNamed(in, n.output_dataset)) return d;
    if (const Dataset* d = DatasetNamed(in, n.dataset_name)) return d;
    if (const Dataset* d = DatasetNamed(in, "ds_datainput_" + std::to_string(n.id))) return d;
    // Only a label no other dataset has (two Data Inputs may share a name).
    const Dataset* found = nullptr;
    for (const auto& d : in.datasets)
        if (!d.label.empty() && d.label == n.name) {
            if (found) return nullptr;
            found = &d;
        }
    return found;
}

const GraphNode* NodeWithId(const Input& in, int id) {
    for (const auto& n : in.nodes)
        if (n.id == id) return &n;
    return nullptr;
}

// The node on a Data Split's first pin (Training Dataset), else on a Data
// Loader's first pin: the same rule as TrainingInputFeeder (the launcher's).
TrainingReader TrainingReaderOf(const Input& in) {
    TrainingReader reader;
    const auto feeder_of = [&](NodeType type) -> const GraphNode* {
        for (const auto& n : in.nodes) {
            if (n.type != type || n.input_from.empty()) continue;
            return NodeWithId(in, n.input_from[0]);
        }
        return nullptr;
    };
    const GraphNode* feeder = feeder_of(NodeType::DataSplit);
    reader.split = feeder != nullptr;
    if (!feeder) feeder = feeder_of(NodeType::DataLoader);
    if (!feeder) return reader;
    reader.node_id = feeder->id;
    reader.name = feeder->name;
    if (IsDataInput(*feeder)) {
        if (const Dataset* d = DatasetOfInput(in, *feeder)) reader.dataset = d->name;
    } else if (feeder->ran && feeder->error.empty() && DatasetNamed(in, feeder->output_dataset)) {
        reader.dataset = feeder->output_dataset;
    }
    return reader;
}

void AddChips(JobView& view, const std::vector<const GraphNode*>& nodes) {
    // One chip per named type ("Check × 4"); Prepare's other step
    // nodes share one chip naming them.
    std::map<std::string, NodeChip> by_type;
    std::vector<std::string> order;
    NodeChip steps{"step nodes", "", {}};
    std::vector<std::string> step_names;
    std::vector<std::string> recipes;
    for (const GraphNode* n : nodes) {
        const char* label = TypeLabel(n->type);
        if (!label) {
            steps.ids.push_back(n->id);
            step_names.push_back(n->name);
            continue;
        }
        if (!by_type.count(label)) order.push_back(label);
        NodeChip& chip = by_type[label];
        chip.name = label;
        chip.ids.push_back(n->id);
        if (n->type == NodeType::Subgraph) recipes.push_back(n->name);
    }
    for (const auto& label : order) {
        NodeChip chip = by_type[label];
        if (label == "Preparation Recipe") {
            for (size_t i = 0; i < recipes.size(); ++i) chip.note += (i ? ", " : "") + recipes[i];
        } else if (chip.ids.size() > 1) {
            chip.note = "\xC3\x97 " + std::to_string(chip.ids.size());
        } else {
            for (const GraphNode* n : nodes)
                if (n->id == chip.ids.front() && n->name != label) chip.note = n->name;
        }
        view.chips.push_back(std::move(chip));
    }
    if (!steps.ids.empty()) {
        for (size_t i = 0; i < step_names.size() && i < 3; ++i) steps.note += (i ? ", " : "") + step_names[i];
        if (step_names.size() > 3) steps.note += ", \xE2\x80\xA6";
        view.chips.push_back(std::move(steps));
    }
}

}  // namespace

std::vector<StepRow> StepRows(const std::vector<GraphNode>& steps) {
    std::vector<StepRow> rows;
    for (const auto& n : steps) {
        StepRow r;
        r.id = n.id;
        r.name = n.name;
        const char* kind = TypeLabel(n.type);
        r.kind = kind ? kind : "step node";
        r.sql = n.type == NodeType::SQLQuery;
        if (!n.error.empty()) {
            r.meta = "failed";
            r.tip = n.error;
            r.tone = Tone::Error;
        } else if (IsCheck(n) && n.ran) {
            r.meta = CheckPassed(n) ? "pass" : "fail";
            r.tip = n.outcome;
            r.tone = CheckPassed(n) ? Tone::Ok : Tone::Error;
        } else if (n.ran && n.rows >= 0) {
            r.meta = Thousands(n.rows) + " rows \xC2\xB7 " + Seconds(n.seconds);
        } else {
            r.meta = "not run";
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

std::string SplitOfName(const std::string& name, std::string* stem) {
    static const std::pair<const char*, const char*> kSuffixes[] = {
        {"train", "train"},           {"training", "train"}, {"validation", "validation"}, {"valid", "validation"},
        {"val", "validation"},        {"dev", "validation"}, {"test", "test"},             {"testing", "test"}};
    std::string lower = name;
    for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    for (const auto& [suffix, split] : kSuffixes) {
        const size_t n = std::strlen(suffix);
        if (lower.size() < n + 2) continue;
        const size_t at = lower.size() - n;
        const char sep = lower[at - 1];
        if (lower.compare(at, n, suffix) != 0 || (sep != '_' && sep != '-' && sep != ' ' && sep != '.')) continue;
        if (stem) *stem = name.substr(0, at - 1);
        return split;
    }
    return "";
}

View Build(const Input& in) {
    View v;

    // Jobs and their nodes.
    std::array<std::vector<const GraphNode*>, kJobCount> per_job;
    for (const auto& n : in.nodes) {
        Job job;
        if (JobOf(n.type, job) && !(n.type == NodeType::Subgraph && !n.recipe)) per_job[static_cast<size_t>(job)].push_back(&n);
    }
    size_t recipes = 0, checks = 0, checks_ran = 0, checks_passed = 0, inputs = 0;
    std::string recipe_name;
    for (const auto& n : in.nodes) {
        if (n.recipe) {
            ++recipes;
            recipe_name = n.name;
        }
        if (IsDataInput(n)) ++inputs;
        if (IsCheck(n)) {
            ++checks;
            if (n.ran) ++checks_ran;
            if (CheckPassed(n)) ++checks_passed;
        }
    }
    for (size_t j = 0; j < kJobCount; ++j) {
        JobView& jv = v.jobs[j];
        jv.job = static_cast<Job>(j);
        jv.name = JobName(jv.job);
        for (const GraphNode* n : per_job[j]) jv.node_ids.push_back(n->id);
        AddChips(jv, per_job[j]);
        const size_t count = per_job[j].size();
        switch (jv.job) {
            case Job::BringIn:
                jv.state = inputs ? Plural(inputs, "Data Input", "Data Inputs") : count ? Plural(count, "source", "sources") : "none yet";
                break;
            case Job::Explore: jv.state = "query, profile, missing, outliers, plot"; break;
            case Job::Prepare:
                jv.state = recipes ? Plural(recipes, "recipe", "recipes") : count ? Plural(count, "step", "steps") : "none yet";
                break;
            case Job::Validate:
                jv.state = !checks ? (count ? Plural(count, "validator", "validators") : "no checks")
                           : !checks_ran ? Plural(checks, "check", "checks") + ", not run"
                                         : std::to_string(checks_passed) + " of " + std::to_string(checks_ran) + " pass";
                break;
            case Job::Ready: break;  // after the split sets, below
            case Job::Publish: jv.state = count ? Plural(count, "output", "outputs") : "no output"; break;
            case Job::Results: {
                std::vector<std::string> parts;
                if (in.test_rows > 0) parts.push_back("test: " + Thousands(in.test_rows) + (in.test_rows == 1 ? " row" : " rows"));
                if (in.preview_runs) parts.push_back(Plural(in.preview_runs, "preview run", "preview runs"));
                if (count) parts.push_back(Plural(count, "evaluation node", "evaluation nodes"));
                for (size_t i = 0; i < parts.size(); ++i) jv.state += (i ? " \xC2\xB7 " : "") + parts[i];
                if (parts.empty()) jv.state = "nothing yet";
                break;
            }
        }
    }

    // The graph line.
    v.graph_label = in.graph_name.empty() ? "unsaved graph" : in.graph_name;
    if (recipes == 1) v.graph_label += " \xC2\xB7 recipe " + recipe_name;
    else if (recipes > 1) v.graph_label += " \xC2\xB7 " + Plural(recipes, "recipe", "recipes");

    // Bring in: the Data Input nodes.
    for (const auto& n : in.nodes) {
        if (!IsDataInput(n)) continue;
        SourceRow row;
        row.node_id = n.id;
        row.name = n.name;
        row.file = n.file_path;
        if (row.file.empty() && !n.folder_path.empty()) {
            row.file = n.folder_path;
            if (!n.labels_csv.empty())
                row.source_note = "image folder, labels from " + FileName(n.labels_csv);
        }
        if (!n.error.empty()) {
            row.state = "failed";
            row.detail = n.error;
            row.tone = Tone::Error;
        } else if (const Dataset* d = DatasetOfInput(in, n)) {
            row.state = "loaded";
            row.size = d->images ? Thousands(static_cast<int64_t>(d->rows)) + " images"
                                 : Thousands(static_cast<int64_t>(d->rows)) + " \xC3\x97 " + std::to_string(d->columns);
            if (d->images && row.source_note.empty() && !n.folder_path.empty()) row.source_note = "image folder, class folders";
            row.dataset = d->name;
            row.tone = Tone::Ok;
        } else {
            row.state = "not loaded";
            row.tone = Tone::Warn;
        }
        v.sources.push_back(std::move(row));
    }

    // The rail: Data Inputs, node results, published files, other datasets.
    v.training = TrainingReaderOf(in);
    std::set<std::string> shown;
    RailSection sources{"Data Inputs", {}};
    for (const auto& n : in.nodes) {
        if (!IsDataInput(n)) continue;
        RailItem item;
        item.kind = RailItem::Kind::DataInput;
        item.label = n.name;
        item.node_id = n.id;
        item.training = n.id == v.training.node_id;
        item.tip = n.file_path;
        if (item.training) item.tip += (item.tip.empty() ? "" : "\n") + std::string("Training reads this table as it is loaded.");
        if (const Dataset* d = DatasetOfInput(in, n)) {
            item.dataset = d->name;
            item.meta = Thousands(static_cast<int64_t>(d->rows));
            // Also the input's other copies (a pipeline run reads the file
            // again as ds_datainput_<id>): one row per Data Input, not a
            // second "Sentiment CSV" under Other datasets.
            for (const std::string& name :
                 {d->name, n.output_dataset, n.dataset_name, "ds_datainput_" + std::to_string(n.id)})
                if (!name.empty()) shown.insert(name);
        } else {
            item.meta = "not loaded";
            item.tone = Tone::Warn;
            item.tip += (item.tip.empty() ? "" : "\n") + std::string("Open its dialog and Apply, or run the graph, to load it.");
        }
        const Dataset* images = DatasetOfInput(in, n);
        sources.items.push_back(std::move(item));
        if (!images || !images->images) continue;
        // Its index table, one row per image, so the table tools work on images.
        RailItem index;
        index.kind = RailItem::Kind::ImageIndex;
        index.label = images->name + "_index";
        index.tip = "One row per image: path, label, size, bytes (and quality once measured).";
        if (const Dataset* table = DatasetNamed(in, index.label)) {
            index.dataset = table->name;
            index.meta = Thousands(static_cast<int64_t>(table->rows)) + " \xC3\x97 " + std::to_string(table->columns);
            shown.insert(table->name);
        } else if (const auto p = in.image_index_progress.find(images->name); p != in.image_index_progress.end()) {
            index.meta = "indexing";
            index.tone = Tone::Warn;
            index.tip += "\nIndexing: " + p->second + " files read.";
        } else {
            continue;
        }
        sources.items.push_back(std::move(index));
    }
    RailSection results{"Node results", {}};
    for (const auto& n : in.nodes) {
        if (IsDataInput(n) || IsExport(n) || n.type == NodeType::Subgraph || !n.ran) continue;
        RailItem item;
        item.kind = RailItem::Kind::NodeResult;
        item.label = n.name;
        item.node_id = n.id;
        item.training = n.id == v.training.node_id;
        if (!n.error.empty()) {
            item.meta = "failed";
            item.tone = Tone::Error;
            item.tip = n.error;
        } else if (IsCheck(n)) {
            item.meta = CheckPassed(n) ? "pass" : "fail";
            item.tone = CheckPassed(n) ? Tone::Ok : Tone::Error;
            item.tip = n.outcome;
        } else if (n.rows >= 0) {
            item.meta = Thousands(n.rows);
        } else {
            continue;  // ran, but made no table
        }
        if (DatasetNamed(in, n.output_dataset)) {
            item.dataset = n.output_dataset;
            shown.insert(n.output_dataset);
        }
        if (item.training)
            item.tip += (item.tip.empty() ? "" : "\n") + std::string("Training reads this table: the steps before it have run on the data.");
        results.items.push_back(std::move(item));
    }
    RailSection published{"Published", {}};
    for (const auto& n : in.nodes) {
        if (!IsExport(n)) continue;
        RailItem item;
        item.kind = RailItem::Kind::Published;
        item.label = n.file_path.empty() ? n.name : FileName(n.file_path);
        item.node_id = n.id;
        item.tip = n.file_path.empty() ? n.name : n.file_path;
        if (!n.error.empty()) {
            item.meta = "failed";
            item.tone = Tone::Error;
            item.tip += "\n" + n.error;
        } else if (n.ran && n.rows >= 0) {
            item.meta = Thousands(n.rows);
            item.tone = Tone::Ok;
        } else {
            item.meta = "not written";
        }
        if (DatasetNamed(in, n.output_dataset)) item.dataset = n.output_dataset;
        published.items.push_back(std::move(item));
    }
    RailSection other{"Other datasets", {}};
    for (const auto& d : in.datasets) {
        if (shown.count(d.name) || !d.tabular) continue;
        // The pipeline run's partition-marked copy of a Data Input (<name>__split,
        // TOFIX178 S0-g2) is the run's own; listing it would look like a dataset.
        if (d.name.size() > 7 && d.name.compare(d.name.size() - 7, 7, "__split") == 0) continue;
        RailItem item;
        item.kind = RailItem::Kind::Other;
        item.label = d.label.empty() ? d.name : d.label;
        item.meta = Thousands(static_cast<int64_t>(d.rows));
        item.dataset = d.name;
        item.tip = d.name;
        other.items.push_back(std::move(item));
    }
    v.rail = {std::move(sources), std::move(results), std::move(published)};
    if (!other.items.empty()) v.rail.push_back(std::move(other));

    // ML readiness: each Data Split's inputs, then the catalog's <stem>_train / ... names.
    static const char* kSplits[] = {"train", "validation", "test"};
    std::set<std::string> in_sets;
    for (const auto& n : in.nodes) {
        if (n.type == NodeType::DataLoader && v.balance.empty()) v.balance = n.balance.empty() ? "none" : n.balance;
        if (n.type != NodeType::DataSplit) continue;
        SplitSet set;
        set.name = n.name;
        set.node_id = n.id;
        for (size_t pin = 0; pin < n.input_from.size() && pin < 3; ++pin) {
            const int from = n.input_from[pin];
            if (from < 0) continue;
            const GraphNode* source = nullptr;
            for (const auto& m : in.nodes)
                if (m.id == from) source = &m;
            const Dataset* d = source && IsDataInput(*source) ? DatasetOfInput(in, *source) : nullptr;
            if (!d) {
                set.note += std::string(set.note.empty() ? "" : "; ") + kSplits[pin] + ": " +
                            (source && IsDataInput(*source) ? "\"" + source->name + "\" is not loaded" : "not read from a Data Input");
                continue;
            }
            set.parts.push_back({kSplits[pin], d->name, source->name});
            in_sets.insert(d->name);
        }
        v.split_sets.push_back(std::move(set));
    }
    std::map<std::string, SplitSet> by_stem;
    for (const auto& d : in.datasets) {
        if (!d.tabular || in_sets.count(d.name)) continue;
        std::string stem;
        const std::string display = d.label.empty() ? d.name : d.label;
        std::string split = SplitOfName(display, &stem);
        if (split.empty()) split = SplitOfName(d.name, &stem);
        if (split.empty()) continue;
        SplitSet& set = by_stem[stem];
        set.name = stem;
        if (std::none_of(set.parts.begin(), set.parts.end(), [&](const SplitPart& p) { return p.split == split; }))
            set.parts.push_back({split, d.name, display});
    }
    for (auto& [stem, set] : by_stem) {
        const bool has_train = std::any_of(set.parts.begin(), set.parts.end(), [](const SplitPart& p) { return p.split == "train"; });
        if (!has_train || set.parts.size() < 2) continue;
        std::sort(set.parts.begin(), set.parts.end(), [](const SplitPart& a, const SplitPart& b) {
            const auto rank = [](const std::string& s) { return s == "train" ? 0 : s == "validation" ? 1 : 2; };
            return rank(a.split) < rank(b.split);
        });
        v.split_sets.push_back(std::move(set));
    }
    {
        JobView& ready = v.jobs[static_cast<size_t>(Job::Ready)];
        if (v.split_sets.size() == 1) ready.state = Plural(v.split_sets[0].parts.size(), "split", "splits");
        else if (!v.split_sets.empty()) ready.state = Plural(v.split_sets.size(), "split set", "split sets");
        else ready.state = ready.node_ids.empty() ? "no split" : Plural(ready.node_ids.size(), "node", "nodes");
    }

    // The last run: the latest node that finished with a table (checks count
    // separately, exports are in the rail), or the failure.
    const GraphNode* last = nullptr;
    for (const auto& n : in.nodes) {
        if (!n.ran || n.finished_at.empty() || IsCheck(n) || (IsExport(n) && n.error.empty())) continue;
        if (!n.error.empty()) {
            if (!last || last->error.empty() || n.finished_at >= last->finished_at) last = &n;
            continue;
        }
        if (n.rows < 0 || (last && !last->error.empty())) continue;
        if (!last || n.finished_at > last->finished_at || (n.finished_at == last->finished_at && n.type == NodeType::SQLQuery))
            last = &n;
    }
    if (last && !last->error.empty()) {
        v.last_run = last->name + " failed: " + last->error;
        v.last_run_tone = Tone::Error;
    } else if (last) {
        v.last_run = last->name + " \xC2\xB7 " + Thousands(last->rows) + " rows";
        if (checks_ran) v.last_run += " \xC2\xB7 " + std::to_string(checks_passed) + "/" + std::to_string(checks_ran) + " checks";
        v.last_run += " \xC2\xB7 " + Seconds(last->seconds) + " \xC2\xB7 at " + last->finished_at;
        v.last_run_tone = checks_ran && checks_passed < checks_ran ? Tone::Warn : Tone::Ok;
    }
    return v;
}

ReadsNote TrainingReads(const View& view, const std::string& dataset) {
    ReadsNote note;
    const TrainingReader& t = view.training;
    if (t.node_id < 0) {
        note.kind = ReadsNote::Kind::None;
        note.text = "Nothing feeds a Data Split or a Data Loader yet, so training reads no table. "
                    "Steps added here feed it once the graph trains on this data.";
        return note;
    }
    if (!dataset.empty() && dataset == t.dataset) {
        note.kind = ReadsNote::Kind::This;
        note.text = std::string("Training reads this table") + (t.split ? " through the Data Split." : " through the Data Loader.");
        return note;
    }
    if (t.dataset.empty()) {
        note.kind = ReadsNote::Kind::NotRun;
        note.text = "Training reads \"" + t.name + "\", which has no table yet (run its steps in Prepare); this is not it.";
        note.link = "Show the node";
        return note;
    }
    note.kind = ReadsNote::Kind::Other;
    note.text = "Training reads \"" + t.name + "\", not this table.";
    note.link = "Show what training reads";
    return note;
}

}  // namespace cyxwiz::studio
