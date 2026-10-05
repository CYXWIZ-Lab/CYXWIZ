#include "dashboard_session.h"

#include "../async_task_manager.h"
#include "../column_role_store.h"
#include "../dataset_catalog.h"
#include "../plot/plot_arrow_source.h"
#include "../plot/plot_prepare.h"
#include "../session_query_service.h"

#include <algorithm>
#include <functional>

namespace cyxwiz::dashboard {

namespace {

std::string ParamsText(const std::vector<QueryParam>& params) {
    std::string out;
    for (const auto& p : params) out += std::to_string(static_cast<int>(p.type)) + ":" + p.s + ":" + std::to_string(p.d) + ":" + std::to_string(p.i) + "|";
    return out;
}

// What a widget's result depends on: the data version, the widget's
// content (not its place) and the filters that apply to it.
std::string Fingerprint(const std::string& dataset, uint64_t generation, uint64_t epoch, const WidgetSpec& w, const FilterState& filters) {
    std::vector<QueryParam> params;
    const std::string where = filters.WhereFor(w.id, params);
    std::string content = std::to_string(static_cast<int>(w.type)) + "|" + MeasureId(w.measure) + "|" + w.field + "|" + std::to_string(w.rows) + "|";
    for (const auto& c : w.columns) content += c + ",";
    if (w.type == WidgetType::Plot) content += plot::SpecToJson(w.plot);
    return dataset + "#" + std::to_string(generation) + "#" + std::to_string(epoch) + "#" + content + "#" + where + "#" + ParamsText(params);
}

double Number(const std::shared_ptr<arrow::Table>& t, const std::string& col) {
    if (!t || t->num_rows() == 0) return NAN;
    auto c = t->GetColumnByName(col);
    if (!c) return NAN;
    auto s = c->GetScalar(0);
    if (!s.ok() || !(*s)->is_valid) return NAN;
    auto d = (*s)->CastTo(arrow::float64());
    return d.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*d)->value : NAN;
}

std::string Text(const std::shared_ptr<arrow::Table>& t, const std::string& col) {
    if (!t || t->num_rows() == 0) return {};
    auto c = t->GetColumnByName(col);
    if (!c) return {};
    auto s = c->GetScalar(0);
    return s.ok() && (*s)->is_valid ? (*s)->ToString() : std::string();
}

}  // namespace

DashboardSession::DashboardSession() = default;

DashboardSession::~DashboardSession() {
    for (const auto& [id, r] : running_) AsyncTaskManager::Instance().Cancel(r.task);
}

void DashboardSession::SetDataset(const std::string& catalog_name) {
    if (catalog_name == dataset_) return;
    for (const auto& [id, r] : running_) AsyncTaskManager::Instance().Cancel(r.task);
    running_.clear();
    results_.clear();
    cache_.clear();
    strip_ = {};
    words_tables_.clear();
    dataset_ = catalog_name;
    generation_ = 0;
}

void DashboardSession::RefreshAll() {
    ++epoch_;
}

const WidgetResult& DashboardSession::ResultOf(const std::string& widget_id) const {
    auto it = results_.find(widget_id);
    return it == results_.end() ? empty_ : it->second;
}

void DashboardSession::Remember(const WidgetResult& r) {
    cache_.remove_if([&](const WidgetResult& c) { return c.fingerprint == r.fingerprint; });
    cache_.push_front(r);
    if (cache_.size() > kCacheSize) cache_.pop_back();
}

void DashboardSession::Poll(DashboardSpec& spec, const DatasetContract& contract, const DatasetProfile& profile) {
    if (dataset_.empty()) return;
    if (const auto entry = DatasetCatalog::Instance().Resolve(dataset_)) generation_ = entry->generation;
    // A filter change waits for the next click (150 ms).
    const auto now = std::chrono::steady_clock::now();
    const std::string filter_text = spec.filters.Text();
    if (filter_text != filter_text_) {
        filter_text_ = filter_text;
        filter_changed_ = now;
    }
    const bool settled = now - filter_changed_ >= std::chrono::milliseconds(150);
    bool all_bound = true;
    for (const auto& w : spec.widgets) {
        WidgetResult& r = results_[w.id];
        const Binding b = CheckBinding(w, contract, spec.known_types);
        if (b.state != Binding::State::Ok) {
            all_bound = false;
            if (r.state != WidgetResult::State::Unbound || r.message != b.message) {
                auto run = running_.find(w.id);
                if (run != running_.end()) {
                    AsyncTaskManager::Instance().Cancel(run->second.task);
                    running_.erase(run);
                }
                r = WidgetResult{};
                r.state = WidgetResult::State::Unbound;
                r.binding = b;
                r.message = b.message;
                ++r.version;
            }
            continue;
        }
        // Its text column's words are being split once: wait for them (not split here too).
        if (WordsPending(w)) continue;
        const std::string fp = Fingerprint(dataset_ + "|" + WordsFor(w), generation_, epoch_, w, spec.filters);
        if (r.fingerprint == fp && r.state != WidgetResult::State::Unbound) continue;
        auto run = running_.find(w.id);
        if (run != running_.end() && run->second.fingerprint == fp) continue;
        // Seen before (an earlier selection): show it at once.
        auto cached = std::find_if(cache_.begin(), cache_.end(), [&](const WidgetResult& c) { return c.fingerprint == fp; });
        if (cached != cache_.end()) {
            const uint64_t v = r.version;
            r = *cached;
            r.version = v + 1;
            if (run != running_.end()) {
                AsyncTaskManager::Instance().Cancel(run->second.task);
                running_.erase(run);
            }
            continue;
        }
        if (!settled && r.prepared) continue;  // keep the old picture until the clicks settle
        if (run != running_.end()) AsyncTaskManager::Instance().Cancel(run->second.task);
        Start(w, profile, fp, spec.filters);
    }
    // Results of widgets that were removed.
    for (auto it = results_.begin(); it != results_.end();) {
        if (!spec.Find(it->first)) it = results_.erase(it);
        else ++it;
    }
    if (all_bound) {
        std::map<std::string, std::string> types;
        for (const auto& c : contract.columns) types[c.name] = c.type;
        if (types != spec.known_types) spec.known_types = std::move(types);
    }
    // The summary strip.
    std::vector<QueryParam> params;
    const std::string where = spec.filters.WhereFor("", params);
    const std::string strip_fp = dataset_ + "#" + std::to_string(generation_) + "#" + std::to_string(epoch_) + "#" + where + "#" +
                                 ParamsText(params) + "#" + contract.Target().value_or("");
    auto srun = running_.find("#strip");
    if (strip_.fingerprint != strip_fp && (srun == running_.end() || srun->second.fingerprint != strip_fp) && (settled || !strip_.ready)) {
        if (srun != running_.end()) AsyncTaskManager::Instance().Cancel(srun->second.task);
        StartStrip(spec, contract, profile, strip_fp);
    }
}

std::string DashboardSession::WordsFor(const WidgetSpec& w) const {
    const bool text_kpi = w.type == WidgetType::Kpi &&
                          (w.measure == Measure::MedianWords || w.measure == Measure::Vocabulary || w.measure == Measure::EmptyTexts);
    const std::string field = w.IsText() ? w.text_field : text_kpi ? w.field : std::string();
    auto it = field.empty() ? words_tables_.end() : words_tables_.find(field);
    return it == words_tables_.end() ? std::string() : it->second;
}

void DashboardSession::SetWordsTable(const std::string& text_field, const std::string& table_name) { words_tables_[text_field] = table_name; }

bool DashboardSession::WordsPending(const WidgetSpec& w) const {
    const bool text_kpi = w.type == WidgetType::Kpi &&
                          (w.measure == Measure::MedianWords || w.measure == Measure::Vocabulary || w.measure == Measure::EmptyTexts);
    const std::string field = w.IsText() ? w.text_field : text_kpi ? w.field : std::string();
    auto it = field.empty() ? words_tables_.end() : words_tables_.find(field);
    return it != words_tables_.end() && it->second.empty();
}

std::vector<std::string> DashboardSession::TextFields(const DashboardSpec& spec) {
    std::vector<std::string> out;
    for (const auto& w : spec.widgets) {
        const bool text_kpi = w.type == WidgetType::Kpi &&
                              (w.measure == Measure::MedianWords || w.measure == Measure::Vocabulary || w.measure == Measure::EmptyTexts);
        const std::string f = w.IsText() ? w.text_field : text_kpi ? w.field : std::string();
        if (!f.empty() && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    }
    return out;
}

QueryRequest DashboardSession::RequestFor(const WidgetSpec& w, const DatasetProfile& profile, const FilterState& filters,
                                         const std::string& table) const {
    if (w.type == WidgetType::Kpi) return KpiQuery(w, table, filters, WordsFor(w));
    if (w.type == WidgetType::Missing) {
        std::vector<std::string> cols;
        for (const auto& c : profile.columns) cols.push_back(c.facts.name);
        std::map<std::string, std::vector<std::string>> texts;
        if (const auto entry = DatasetCatalog::Instance().Resolve(dataset_))
            if (const auto* s = ProjectColumnRoles().Find(RoleSourceKey(entry->source_path, entry->Shown()))) texts = s->missing_text;
        return MissingQuery(w, table, filters, cols, texts);
    }
    // Plots read a reproducible sample only when the table is bigger than the cap.
    return WidgetQuery(w, table, filters, w.type == WidgetType::Plot && profile.rows > kRowCap ? kRowCap : 0, WordsFor(w));
}

void DashboardSession::Start(const WidgetSpec& in, const DatasetProfile& profile, const std::string& fingerprint, const FilterState& filters) {
    WidgetSpec w = in;
    // Histogram bins over the column's whole range, so filtered and all rows line up.
    if (w.type == WidgetType::Plot && w.plot.kind == plot::Kind::Histogram && !std::isfinite(w.plot.range_lo))
        if (const ProfiledColumn* c = profile.Find(w.plot.x_column))
            if (c->numeric && c->max > c->min) {
                w.plot.range_lo = c->min;
                w.plot.range_hi = c->max;
            }
    QueryRequest request = RequestFor(w, profile, filters, dataset_);
    // The same over all rows (grey behind), when filters apply to this bar or histogram.
    std::vector<QueryParam> unused;
    const bool with_all = w.type == WidgetType::Plot && (w.plot.kind == plot::Kind::Bar || w.plot.kind == plot::Kind::Histogram) &&
                          !filters.WhereFor(w.id, unused).empty();
    const QueryRequest all_request =
        with_all ? WidgetQuery(w, dataset_, FilterState{}, profile.rows > kRowCap ? kRowCap : 0, WordsFor(w)) : QueryRequest{};
    // The row count decides whether the plot is drawn from a sample.
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_);
    const bool sampled = w.type == WidgetType::Plot && entry && entry->rows > kRowCap;
    auto result = std::make_shared<WidgetResult>();
    result->fingerprint = fingerprint;
    result->sampled = sampled;
    WidgetResult& r = results_[w.id];
    if (r.state != WidgetResult::State::Ready) r.state = WidgetResult::State::Running;
    const std::string id = w.id;
    const WidgetSpec widget = w;  // with the aligned range
    std::weak_ptr<int> alive = alive_;
    const uint64_t task = AsyncTaskManager::Instance().RunAsync(
        "Dashboard: " + (w.title.empty() ? KindOf(w).label : w.title),
        [request, all_request, with_all, widget, result](LambdaTask& t) {
            QueryResult q = SessionQueryService::Instance().RunNow(request);
            if (!q.ok) {
                result->state = q.cancelled ? WidgetResult::State::Waiting : WidgetResult::State::Failed;
                result->message = q.error;
                if (!q.cancelled) t.MarkFailed(q.error);
                return;
            }
            if (widget.type == WidgetType::Kpi) {
                result->value = Number(q.table, "value");
                result->all = Number(q.table, "all_rows");
            } else if (widget.type == WidgetType::Table || widget.type == WidgetType::Missing) {
                result->table = q.table;
            } else {
                // Prepared off the UI thread, exactly as the Plot window does.
                const plot::Source src = plot::SourceFromArrow(*q.table, plot::ColumnsNeeded(widget.plot));
                result->prepared = std::make_shared<const plot::Prepared>(plot::Prepare(widget.plot, src));
                if (with_all) {
                    QueryResult a = SessionQueryService::Instance().RunNow(all_request);
                    if (a.ok)
                        result->all_rows = std::make_shared<const plot::Prepared>(
                            plot::Prepare(widget.plot, plot::SourceFromArrow(*a.table, plot::ColumnsNeeded(widget.plot))));
                }
            }
            result->state = WidgetResult::State::Ready;
        },
        nullptr,
        [this, alive, id, result](bool, const std::string&) {
            if (alive.expired()) return;
            auto run = running_.find(id);
            if (run == running_.end() || run->second.fingerprint != result->fingerprint) return;  // superseded
            running_.erase(run);
            if (result->state == WidgetResult::State::Waiting) return;  // cancelled
            WidgetResult& r = results_[id];
            const uint64_t v = r.version;
            r = *result;
            r.version = v + 1;
            if (r.state == WidgetResult::State::Ready) Remember(r);
        },
        alive_);
    running_[id] = Running{task, fingerprint};
}

void DashboardSession::StartStrip(const DashboardSpec& spec, const DatasetContract& contract, const DatasetProfile& profile, const std::string& fp) {
    const auto target = contract.Target();
    bool numeric = false;
    if (target)
        if (const ColumnContract* c = contract.Find(*target)) numeric = c->type == "int" || c->type == "float";
    std::map<std::string, std::vector<std::string>> texts;
    if (const auto entry = DatasetCatalog::Instance().Resolve(dataset_))
        if (const auto* s = ProjectColumnRoles().Find(RoleSourceKey(entry->source_path, entry->Shown()))) texts = s->missing_text;
    QueryRequest request = StripQuery(dataset_, spec.filters, profile, target.value_or(""), numeric, texts);
    auto result = std::make_shared<StripResult>();
    result->fingerprint = fp;
    std::weak_ptr<int> alive = alive_;
    const uint64_t task = AsyncTaskManager::Instance().RunAsync(
        "Dashboard summary",
        [request, result, numeric](LambdaTask&) {
            QueryResult q = SessionQueryService::Instance().RunNow(request);
            if (!q.ok) return;
            result->rows_now = Number(q.table, "rows_now");
            result->rows_all = Number(q.table, "rows_all");
            result->missing_now = Number(q.table, "missing_now");
            if (numeric) {
                result->target_now = Number(q.table, "target_now");
                result->target_all = Number(q.table, "target_all");
            } else {
                result->target_text_now = Text(q.table, "target_now");
                result->target_text_all = Text(q.table, "target_all");
            }
            result->ready = true;
        },
        nullptr,
        [this, alive, result](bool, const std::string&) {
            if (alive.expired()) return;
            auto run = running_.find("#strip");
            if (run == running_.end() || run->second.fingerprint != result->fingerprint) return;
            running_.erase(run);
            if (result->ready) strip_ = *result;
        },
        alive_);
    running_["#strip"] = Running{task, fp};
}

}  // namespace cyxwiz::dashboard
