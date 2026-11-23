#include "text_view.h"

#include "explore_plots.h"

#include "../../core/dataset_contract.h"
#include "../../core/explore_presentation.h"
#include "../../core/file_dialogs.h"
#include "../../core/session_query_service.h"
#include "../../core/text_presentation.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <cyxwiz/tokenizer.h>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace cyxwiz {

namespace {

namespace tv = text_view;

QueryResult RunNow(const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); }

constexpr size_t kRowTextShown = 1200;  // characters of a long row shown (the rest said)
constexpr size_t kGroupDistinct = 200;  // a group column has at most this many values

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Faint(const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void FaintAfter(const std::string& text) {
    if (text.empty()) return;
    ImGui::SameLine(0.0f, 6.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    if (text.empty()) return;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

void Heading(const char* text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::PopTextWrapPos();
}

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

void WrappedCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    ImGui::PushTextWrapPos(0.0f);
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
}

// The next setting on the line, or on a new line when it would not fit.
void Next(float width) {
    ImGui::SameLine(0.0f, 16.0f);
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

void Setting(const char* label, float width) {
    Next(ImGui::CalcTextSize(label).x + width);
    Label(label);
}

std::string Fixed(double v, int digits) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

std::string Percent(double share) {
    char buf[32];
    std::snprintf(buf, sizeof buf, share > 0.9 || share < 0.1 ? "%.1f%%" : "%.0f%%", share * 100.0);
    return buf;
}

float PlotHeight(float w) { return std::clamp(w * 0.3f, 140.0f, 220.0f); }

// A long row shown in part: its first characters (at a UTF-8 boundary) and how much is left.
std::string Clipped(const std::string& text, size_t limit = kRowTextShown) {
    if (text.size() <= limit) return text;
    size_t cut = limit;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
    return text.substr(0, cut) + " ... (" + explore_view::Number(text.size() - cut) + " more characters)";
}

// The two-column split of a look (plots left, tables right) when there is room;
// true when the table began (EndTable must run only then).
bool BeginSplit(const char* id, float left, float right) {
    if (ImGui::GetContentRegionAvail().x < 820.0f) return false;
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) return false;
    ImGui::TableSetupColumn("left", ImGuiTableColumnFlags_WidthStretch, left);
    ImGui::TableSetupColumn("right", ImGuiTableColumnFlags_WidthStretch, right);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    return true;
}

void SplitNext(bool split) {
    if (split) ImGui::TableNextColumn();
    else ImGui::Spacing();
}

bool BeginTable(const char* id, int columns) {
    return ImGui::BeginTable(id, columns, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings);
}

const char* const kTokenizers[] = {"Whitespace", "Word", "Character", "Byte BPE", "WordPiece", "SentencePiece BPE", "SentencePiece Unigram"};
constexpr int kTokenizerCount = 7;

void RowField(const char* id, int* row) {
    Setting("Row", 120.0f);
    ui::NumberField(id, row, 0, 1 << 30, 1, 72.0f, "The row looked at, as the table numbers it (1 is the first); 0: the first row with text");
}

}  // namespace

void TextView::Message(const std::string& text, bool error) {
    message_ = text;
    message_error_ = error;
}

template <typename T>
bool TextView::Waiting(ExploreJob<T>& job, const std::string& key, const char* text) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (key == stopped_key_) {
        Faint("Stopped. Change a setting to run again.");
        return false;
    }
    if (!job.Error().empty()) {
        Wrapped(t.error, job.Error());
        return false;
    }
    std::string message;
    const float done = job.Progress(&message);
    Faint(message.empty() ? std::string(text) : std::string(text) + " " + message);
    if (done > 0.0f) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, t.accent);
        ImGui::ProgressBar(std::clamp(done, 0.0f, 1.0f), ImVec2(std::min(420.0f, ImGui::GetContentRegionAvail().x), 6.0f), "");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "%d%%", static_cast<int>(std::lround(done * 100.0f)));
    }
    if (ui::LinkButton("Stop")) {
        job.Cancel();
        stopped_key_ = key;
    }
    return true;
}

std::string TextView::BaseKey(const std::string& dataset) const { return dataset + "|" + base_.text + "|" + base_.group; }

void TextView::Export(const std::string& name, const std::string& csv) {
    if (const auto path = FileDialogs::SaveFile("Export text", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(*path, std::ios::binary);
        out << csv;
        Message(out ? "Saved " + *path : "Could not write " + *path, !out);
    }
}

void TextView::Render(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!ExploreProfileReady(context, "Choose a table in the list to read one of its text columns.")) return;
    const DatasetProfile& p = *context.profile;
    if (seen_ != context.profile) {
        seen_ = context.profile;
        words_job_.Reset();
        terms_job_.Reset();
        sentiment_job_.Reset();
        tokens_job_.Reset();
        embeddings_job_.Reset();
        learn_key_.clear();
        nearest_for_.clear();
        map_source_.clear();
        message_.clear();
        stopped_key_.clear();
        if (!base_.text.empty() && !p.Find(base_.text)) base_.text.clear();
        if (!base_.group.empty() && !p.Find(base_.group)) base_.group.clear();
        // The text column: the one inferred as Text (the longest), else the longest text.
        if (base_.text.empty()) {
            double best = -1.0;
            bool best_role = false;
            for (const auto& c : p.columns) {
                if (c.facts.type != ColumnFacts::Type::Text) continue;
                const bool role = InferRole(c.facts) == ColumnRole::Text;
                if ((role && !best_role) || (role == best_role && c.facts.avg_length > best)) {
                    best = c.facts.avg_length;
                    best_role = role;
                    base_.text = c.facts.name;
                }
            }
            // The group: a category column of 2 .. 50 values, the fewest first.
            if (!base_.text.empty() && base_.group.empty()) {
                size_t fewest = 51;
                for (const auto& c : p.columns) {
                    if (c.facts.name == base_.text || c.facts.distinct < 2 || c.facts.distinct >= fewest) continue;
                    const ColumnRole role = InferRole(c.facts);
                    if (role == ColumnRole::Category || role == ColumnRole::Target) {
                        fewest = c.facts.distinct;
                        base_.group = c.facts.name;
                    }
                }
            }
        }
    }
    RenderControls(p);
    if (!message_.empty()) Wrapped(message_error_ ? t.error : t.text_dim, message_);
    ImGui::Spacing();
    if (base_.text.empty()) {
        Faint("Pick the text column to read.");
        return;
    }
    switch (look_) {
        case Look::Words: RenderWords(context); break;
        case Look::Terms: RenderTerms(context); break;
        case Look::Sentiment: RenderSentiment(context); break;
        case Look::Tokens: RenderTokens(context); break;
        case Look::Embeddings: RenderEmbeddings(context); break;
    }
}

void TextView::RenderControls(const DatasetProfile& p) {
    Label("Text");
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::BeginCombo("##text_column", base_.text.empty() ? "pick a column" : base_.text.c_str())) {
        for (const auto& c : p.columns) {
            if (c.facts.type != ColumnFacts::Type::Text) continue;
            const std::string item = c.facts.name + "##text_" + c.facts.name;
            if (ImGui::Selectable(item.c_str(), c.facts.name == base_.text)) {
                base_.text = c.facts.name;
                if (base_.group == base_.text) base_.group.clear();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s characters on average, %s distinct", explore_view::Figure(c.facts.avg_length).c_str(),
                                  explore_view::Number(c.facts.distinct).c_str());
        }
        ImGui::EndCombo();
    }
    Setting("Group by", 200.0f);
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::BeginCombo("##text_group", base_.group.empty() ? "none" : base_.group.c_str())) {
        if (ImGui::Selectable("none", base_.group.empty())) base_.group.clear();
        for (const auto& c : p.columns) {
            if (c.facts.name == base_.text || c.facts.distinct < 2 || c.facts.distinct > kGroupDistinct ||
                (c.facts.type != ColumnFacts::Type::Text && c.facts.type != ColumnFacts::Type::Integer && c.facts.type != ColumnFacts::Type::Boolean))
                continue;
            const std::string item = c.facts.name + "##group_" + c.facts.name;
            if (ImGui::Selectable(item.c_str(), c.facts.name == base_.group)) base_.group = c.facts.name;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s values", explore_view::Number(c.facts.distinct).c_str());
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A category or the label: the words, terms and sentiment are also shown per group");
    static const char* const kLooks[] = {"Words", "Terms", "Sentiment", "Tokens", "Embeddings"};
    int look = static_cast<int>(look_);
    ui::FlowRow looks;
    looks.Next(ImGui::CalcTextSize("Look atWordsTermsSentimentTokensEmbeddings").x + 130.0f);
    Label("Look at");
    if (ExploreChoices("##text_look", kLooks, 5, &look)) look_ = static_cast<Look>(look);
    RenderLookSettings();
}

void TextView::RenderLookSettings() {
    switch (look_) {
        case Look::Words: {
            static const char* const kCount[] = {"words", "word pairs"};
            int count = words_.pairs ? 1 : 0;
            Setting("Count", 180.0f);
            if (ExploreChoices("##text_count", kCount, 2, &count)) words_.pairs = count == 1;
            Next(260.0f);
            ImGui::Checkbox("leave out common English words##text_stop", &words_.stop_words);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("scikit-learn's English stop words (318) and their contractions: the list the TF-IDF node uses");
            ImGui::SameLine(0.0f, 10.0f);
            ImGui::Checkbox("by stem (Porter)##text_stems", &words_.stems);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("\"feeling\" and \"feels\" count as \"feel\" (the Porter stemmer)");
            Setting("Show the top", 120.0f);
            ui::NumberField("##text_top", &words_.top, 5, 200, 5, 56.0f, "How many words the chart and the export list");
            break;
        }
        case Look::Terms: {
            static const char* const kTerms[] = {"words", "words and pairs"};
            int terms = terms_.ngram_max >= 2 ? 1 : 0;
            Setting("Terms", 200.0f);
            if (ExploreChoices("##text_terms", kTerms, 2, &terms)) terms_.ngram_max = terms + 1;
            Setting("in at least", 150.0f);
            ui::NumberField("##text_min_df", &terms_.min_df, 1, 1 << 30, 1, 56.0f, "A term counts when it is in this many rows or more (min_df)");
            FaintAfter("rows");
            static const char* const kNorm[] = {"L2", "L1", "none"};
            static const char* const kNormNames[] = {"l2", "l1", "none"};
            int norm = terms_.norm == "l1" ? 1 : terms_.norm == "none" ? 2 : 0;
            Setting("Normalise", 180.0f);
            if (ExploreChoices("##text_norm", kNorm, 3, &norm)) terms_.norm = kNormNames[norm];
            Next(240.0f);
            ImGui::Checkbox("leave out common English words##text_terms_stop", &terms_.stop_words);
            RowField("##text_terms_row", &row_);
            break;
        }
        case Look::Sentiment: {
            ui::NumberFieldOptions o;
            o.min = -1.0;
            o.max = 1.0;
            o.format = "%g";
            o.width = 56.0f;
            Setting("Positive from", 120.0f);
            o.hint = "A row reads positive at this compound score and above (VADER: 0.05)";
            ui::NumberField("##text_pos", &sentiment_.positive_from, o);
            Setting("negative at", 120.0f);
            o.hint = "A row reads negative at this compound score and below (VADER: -0.05)";
            ui::NumberField("##text_neg", &sentiment_.negative_below, o);
            FaintAfter("VADER's compound, -1 .. 1");
            RowField("##text_sentiment_row", &row_);
            break;
        }
        case Look::Tokens: {
            Setting("Tokenizer", 200.0f);
            ImGui::SetNextItemWidth(170.0f);
            if (ImGui::BeginCombo("##text_tokenizer", kTokenizers[std::clamp(tokens_.tokenizer, 0, kTokenizerCount - 1)])) {
                for (int i = 0; i < kTokenizerCount; ++i) {
                    const bool available = !IsSentencePieceTokenizerType(static_cast<TokenizerType>(i)) || IsSentencePieceTokenizerAvailable();
                    if (!available) ImGui::BeginDisabled();
                    if (ImGui::Selectable(kTokenizers[i], tokens_.tokenizer == i)) tokens_.tokenizer = i;
                    if (!available) {
                        ImGui::EndDisabled();
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", SentencePieceTokenizerUnavailableMessage().c_str());
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("The Tokenizer node's tokenizers, trained on this column as the node trains them");
            Setting("Vocabulary", 120.0f);
            ui::NumberField("##text_vocab", &tokens_.vocabulary, 1, 1 << 24, 1000, 72.0f, "The largest vocabulary (max_vocab_size)");
            Setting("seen at least", 120.0f);
            ui::NumberField("##text_min_freq", &tokens_.min_frequency, 1, 1 << 20, 1, 48.0f, "A token joins the vocabulary when seen this many times (min_word_freq)");
            Setting("Length limit", 120.0f);
            ui::NumberField("##text_max_length", &tokens_.max_length, 1, 1 << 20, 32, 64.0f, "Tokens per row the node keeps (max_length); longer rows are cut");
            FaintAfter("tokens");
            RowField("##text_tokens_row", &row_);
            break;
        }
        case Look::Embeddings: {
            static const char* const kMethod[] = {"word2vec", "fastText", "GloVe"};
            static const char* const kMethodNames[] = {"word2vec", "fasttext", "glove"};
            int method = embeddings_.method == "fasttext" ? 1 : embeddings_.method == "glove" ? 2 : 0;
            Setting("Learn", 230.0f);
            if (ExploreChoices("##text_method", kMethod, 3, &method)) {
                embeddings_.method = kMethodNames[method];
                if (embeddings_.method == "glove") {
                    embeddings_.window = 10;
                    embeddings_.epochs = 25;
                } else if (embeddings_.window == 10 && embeddings_.epochs == 25) {
                    embeddings_.window = 5;
                    embeddings_.epochs = 5;
                }
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("The Word Embeddings node's trainers on the device: word2vec, fastText (with character n-grams) or GloVe");
            if (embeddings_.method != "glove") {
                static const char* const kArchitecture[] = {"skip-gram", "CBOW"};
                int architecture = embeddings_.cbow ? 1 : 0;
                Setting("Architecture", 190.0f);
                if (ExploreChoices("##text_architecture", kArchitecture, 2, &architecture)) embeddings_.cbow = architecture == 1;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("skip-gram: each word predicts its context (better for rare words); CBOW: the context predicts the word (faster)");
            }
            Setting("Size", 100.0f);
            ui::NumberField("##text_size", &embeddings_.size, 2, 1024, 10, 56.0f, "Numbers per word vector");
            Setting("Window", 100.0f);
            ui::NumberField("##text_window", &embeddings_.window, 1, 50, 1, 48.0f, "The largest distance between a word and its context");
            Setting("Words seen at least", 100.0f);
            ui::NumberField("##text_min_count", &embeddings_.min_count, 1, 1 << 20, 1, 48.0f, "Words seen fewer times get no vector");
            Setting("Epochs", 100.0f);
            ui::NumberField("##text_epochs", &embeddings_.epochs, 1, 200, 1, 48.0f, "Passes over the column");
            static const char* const kRows[] = {"a sample (500,000 words)", "every row"};
            int rows = all_rows_ ? 1 : 0;
            Setting("Learn from", 260.0f);
            if (ExploreChoices("##text_rows", kRows, 2, &rows)) all_rows_ = rows == 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("A sample: rows drawn at random until 500,000 words, so the look takes seconds; every row can take minutes. The node learns from every row.");
            break;
        }
    }
}

void TextView::RenderRowsLine(const text::TextData& d) {
    std::string line = tv::RowsWords(d);
    const std::string groups = tv::GroupsWords(d, base_.group);
    if (!groups.empty()) line += "; " + groups;
    Faint(line + ".");
}

// ---- Words

void TextView::RenderWords(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    words_.base = base_;
    const std::string key = BaseKey(context.dataset) + "|words|" + (words_.pairs ? "p" : "w") + (words_.stop_words ? "s" : "") + (words_.stems ? "t" : "") + "|" +
                            std::to_string(words_.top);
    const std::string dataset = context.dataset;
    const text::WordsRequest req = words_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_)
        words_job_.Ask("Text: words", key, [dataset, req, texts](LambdaTask& task) {
            return text::RunWords(dataset, req, RunNow, texts, [&task] { return task.ShouldStop(); });
        });
    const text::WordsAnswer* a = words_job_.Answer(key);
    if (!a) {
        Waiting(words_job_, key, "Counting the words...");
        return;
    }
    ExploreFigures("##text_words_figures", {{"words", explore_view::Number(static_cast<size_t>(a->words))},
                                            {a->request.pairs ? "distinct pairs" : "distinct", explore_view::Number(static_cast<size_t>(a->distinct))},
                                            {"words per row", tv::LengthWords(a->per_row, "words")},
                                            {"rows", explore_view::Number(a->data.texts.size())}});
    Wrapped(t.text, tv::WordsSentence(*a));
    RenderRowsLine(a->data);
    ImGui::Spacing();
    const bool split = BeginSplit("##text_words_split", 1.3f, 1.0f);
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading(a->request.pairs ? "THE COMMONEST WORD PAIRS" : (a->request.stop_words ? "THE COMMONEST WORDS (COMMON ENGLISH WORDS LEFT OUT)" : "THE COMMONEST WORDS"));
        std::vector<std::string> names;
        std::vector<double> values;
        std::vector<std::string> shown;
        for (const auto& c : a->top) {
            names.push_back(c.term);
            values.push_back(static_cast<double>(c.count));
            shown.push_back(explore_view::Number(static_cast<size_t>(c.count)));
        }
        explore_plot::NamedBars(names, values, w, &shown);
        ImGui::Spacing();
        Heading("WORDS PER ROW");
        explore_plot::Bars(a->per_row.bins, 0.0, static_cast<double>(a->per_row.cap), w, PlotHeight(w));
        if (a->per_row.over > 0) Faint("The last bar also holds the " + explore_view::Number(static_cast<size_t>(a->per_row.over)) + " rows longer than " +
                                       explore_view::Number(static_cast<size_t>(a->per_row.cap)) + " words.");
    }
    SplitNext(split);
    {
        if (!a->by_group.empty()) {
            Heading(a->request.pairs ? "THE COMMONEST PAIRS PER GROUP" : "THE COMMONEST WORDS PER GROUP");
            if (BeginTable("##text_words_groups", 3)) {
                ImGui::TableSetupColumn("group", ImGuiTableColumnFlags_WidthStretch, 0.8f);
                ImGui::TableSetupColumn("rows", ImGuiTableColumnFlags_WidthStretch, 0.4f);
                ImGui::TableSetupColumn("top", ImGuiTableColumnFlags_WidthStretch, 2.0f);
                for (size_t g = 0; g < a->by_group.size() && g < a->data.groups.size(); ++g) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(a->data.groups[g].c_str());
                    NumberCell(explore_view::Number(static_cast<size_t>(a->data.group_rows[g])), &t.text_dim);
                    std::string top;
                    for (size_t k = 0; k < a->by_group[g].size() && k < 6; ++k) top += (k ? ", " : "") + a->by_group[g][k].term;
                    WrappedCell(top);
                }
                ImGui::EndTable();
            }
            ImGui::Spacing();
        }
        if (!a->request.pairs && !a->pairs.empty()) {
            Heading("WORD PAIRS");
            if (BeginTable("##text_words_pairs", 2)) {
                ImGui::TableSetupColumn("pair", ImGuiTableColumnFlags_WidthStretch, 1.5f);
                ImGui::TableSetupColumn("count", ImGuiTableColumnFlags_WidthStretch, 0.6f);
                for (size_t k = 0; k < a->pairs.size() && k < 8; ++k) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(a->pairs[k].term.c_str());
                    NumberCell(explore_view::Number(static_cast<size_t>(a->pairs[k].count)));
                }
                ImGui::EndTable();
            }
        }
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (ui::SecondaryButton("Export CSV...")) Export("text_words.csv", tv::WordsCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("rank, word, count, share of the words counted");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((tv::WordsSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Terms

void TextView::RenderTerms(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    terms_.base = base_;
    terms_.row = TableRow();
    const std::string key = BaseKey(context.dataset) + "|terms|" + std::to_string(terms_.ngram_max) + "|" + std::to_string(terms_.min_df) + "|" + terms_.norm +
                            (terms_.stop_words ? "|s" : "") + "|" + std::to_string(terms_.row);
    const std::string dataset = context.dataset;
    const text::TermsRequest req = terms_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_)
        terms_job_.Ask("Text: terms", key, [dataset, req, texts](LambdaTask& task) {
            return text::RunTerms(dataset, req, RunNow, texts, [&task] { return task.ShouldStop(); });
        });
    const text::TermsAnswer* a = terms_job_.Answer(key);
    if (!a) {
        Waiting(terms_job_, key, "Weighing the terms (TF-IDF)...");
        return;
    }
    std::vector<std::pair<std::string, std::string>> figures = {
        {"terms", explore_view::Number(a->terms) + " in " + explore_view::Number(static_cast<size_t>(a->request.min_df)) + " rows or more"},
        {"weights", std::string("count x idf (smoothed), ") + (a->request.norm == "none" ? "not normalised" : a->request.norm == "l1" ? "L1" : "L2")}};
    if (a->html_entities > 0) figures.push_back({"HTML entities", explore_view::Number(static_cast<size_t>(a->html_entities)) + " rows"});
    if (a->links > 0) figures.push_back({"links", explore_view::Number(static_cast<size_t>(a->links)) + " rows"});
    if (a->repeated > 0) figures.push_back({"repeated rows", explore_view::Number(static_cast<size_t>(a->repeated))});
    if (a->broken > 0) figures.push_back({"broken characters", explore_view::Number(static_cast<size_t>(a->broken)) + " rows"});
    ExploreFigures("##text_terms_figures", figures);
    Wrapped(t.text, tv::TermsSentence(*a));
    RenderRowsLine(a->data);
    ImGui::Spacing();
    const bool split = BeginSplit("##text_terms_split", 1.0f, 1.1f);
    {
        if (!a->distinctive.empty()) {
            Heading("WHAT SETS EACH GROUP APART (ITS MEAN TF-IDF, ABOVE THE TABLE'S)");
            if (BeginTable("##text_terms_groups", 2)) {
                ImGui::TableSetupColumn("group", ImGuiTableColumnFlags_WidthStretch, 0.7f);
                ImGui::TableSetupColumn("terms", ImGuiTableColumnFlags_WidthStretch, 2.4f);
                for (size_t g = 0; g < a->distinctive.size() && g < a->data.groups.size(); ++g) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(a->data.groups[g].c_str());
                    std::string terms, scores;
                    for (size_t k = 0; k < a->distinctive[g].size() && k < 6; ++k) {
                        terms += (k ? ", " : "") + a->distinctive[g][k].term;
                        scores += a->distinctive[g][k].term + ": mean " + Fixed(a->distinctive[g][k].score, 4) + ", above the table's by " +
                                  Fixed(a->distinctive[g][k].lift, 4) + "\n";
                    }
                    WrappedCell(terms);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", scores.c_str());
                }
                ImGui::EndTable();
            }
        } else {
            Heading("THE TERMS THAT WEIGH MOST (MEAN TF-IDF)");
            if (BeginTable("##text_terms_overall", 2)) {
                ImGui::TableSetupColumn("term", ImGuiTableColumnFlags_WidthStretch, 1.5f);
                ImGui::TableSetupColumn("mean", ImGuiTableColumnFlags_WidthStretch, 0.6f);
                for (const auto& term : a->overall) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(term.term.c_str());
                    NumberCell(Fixed(term.score, 4));
                }
                ImGui::EndTable();
            }
        }
    }
    SplitNext(split);
    {
        Heading(("ROW " + explore_view::Number(static_cast<size_t>(a->request.row + 1)) + (a->row_group.empty() ? "" : " (" + a->row_group + ")") +
                 ": ITS TERMS, AND THE ROWS THAT READ MOST LIKE IT")
                    .c_str());
        Wrapped(t.text_dim, Clipped(a->row_text, 600));
        if (!a->row_terms.empty() && BeginTable("##text_row_terms", 2)) {
            ImGui::TableSetupColumn("term", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableSetupColumn("tf-idf", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            for (const auto& term : a->row_terms) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(term.term.c_str());
                NumberCell(Fixed(term.score, 3));
            }
            ImGui::EndTable();
        }
        if (a->row_terms.empty()) Faint("This row has no term in " + explore_view::Number(static_cast<size_t>(a->request.min_df)) + " rows or more.");
        if (!a->nearest.empty() && BeginTable("##text_nearest_rows", 4)) {
            ImGui::TableSetupColumn("row", ImGuiTableColumnFlags_WidthStretch, 0.4f);
            ImGui::TableSetupColumn("alike", ImGuiTableColumnFlags_WidthStretch, 0.4f);
            ImGui::TableSetupColumn("group", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("text", ImGuiTableColumnFlags_WidthStretch, 2.4f);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Heading("row");
            NumberCell("alike", &t.text_dim);
            ImGui::TableNextColumn();
            Heading(a->nearest.front().group.empty() ? "" : "group");
            ImGui::TableNextColumn();
            Heading("text");
            for (const auto& r : a->nearest) {
                ImGui::TableNextRow();
                NumberCell(explore_view::Number(static_cast<size_t>(r.row + 1)));
                NumberCell(Fixed(r.similarity, 2));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.group.c_str());
                WrappedCell(Clipped(r.text, 160), &t.text_dim);
            }
            ImGui::EndTable();
        }
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph("TFIDFVectorizer", "TF-IDF of " + base_.text, text::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", explore_view::StepTip("TF-IDF", "a table of tfidf_0 .. (the 2,000 commonest of these terms, max_features) with the label and the split").c_str());
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("text_terms.csv", tv::TermsCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("group, term, mean TF-IDF, above the table's");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((tv::TermsSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Sentiment

void TextView::RenderSentiment(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    sentiment_.base = base_;
    sentiment_.row = TableRow();
    char thresholds[64];
    std::snprintf(thresholds, sizeof thresholds, "%.17g|%.17g", sentiment_.positive_from, sentiment_.negative_below);
    const std::string key = BaseKey(context.dataset) + "|sentiment|" + thresholds + "|" + std::to_string(sentiment_.row);
    const std::string dataset = context.dataset;
    const text::SentimentRequest req = sentiment_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_)
        sentiment_job_.Ask("Text: sentiment", key, [dataset, req, texts](LambdaTask& task) {
            return text::RunSentiment(dataset, req, RunNow, texts, [&task] { return task.ShouldStop(); });
        });
    const text::SentimentAnswer* a = sentiment_job_.Answer(key);
    if (!a) {
        Waiting(sentiment_job_, key, "Scoring every row with VADER...");
        return;
    }
    const double n = static_cast<double>(std::max<size_t>(1, a->data.texts.size()));
    ExploreFigures("##text_sentiment_figures",
                   {{"negative", explore_view::Number(static_cast<size_t>(a->negative)) + " (" + Percent(static_cast<double>(a->negative) / n) + ")"},
                    {"positive", explore_view::Number(static_cast<size_t>(a->positive)) + " (" + Percent(static_cast<double>(a->positive) / n) + ")"},
                    {"neutral", explore_view::Number(static_cast<size_t>(a->neutral)) + " (" + Percent(static_cast<double>(a->neutral) / n) + ")"},
                    {"mean compound", Fixed(a->mean, 2)}});
    Wrapped(t.text, tv::SentimentSentence(*a));
    RenderRowsLine(a->data);
    ImGui::Spacing();
    const bool split = BeginSplit("##text_sentiment_split", 1.3f, 1.0f);
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading("COMPOUND SCORE OF THE ROWS (NEGATIVE .. POSITIVE)");
        explore_plot::Bars(a->histogram, -1.0, 1.0, w, PlotHeight(w));
        ImGui::Spacing();
        Heading(("ROW " + explore_view::Number(static_cast<size_t>(a->request.row + 1)) + ", WORD BY WORD").c_str());
        // The words as VADER reads them; a word with a valence after the rules is coloured and scored.
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        size_t chars = 0;
        bool first = true;
        for (const auto& [word, valence] : a->row_scores.words) {
            if (chars > kRowTextShown) {
                ImGui::SameLine(0.0f, 4.0f);
                ImGui::TextColored(t.text_faint, "...");
                break;
            }
            chars += word.size() + 1;
            const std::string shown = valence != 0.0 ? word + " " + (valence > 0 ? "+" : "") + Fixed(valence, 1) : word;
            const float wide = ImGui::CalcTextSize(shown.c_str()).x;
            if (!first) {
                ImGui::SameLine(0.0f, 5.0f);
                if (ImGui::GetCursorScreenPos().x + wide > right) ImGui::NewLine();
            }
            first = false;
            if (valence > 0) ImGui::TextColored(t.success, "%s", shown.c_str());
            else if (valence < 0) ImGui::TextColored(t.error, "%s", shown.c_str());
            else ImGui::TextColored(t.text_dim, "%s", word.c_str());
        }
        const VaderScores& s = a->row_scores;
        Faint("compound " + Fixed(s.compound, 2) + " (" + tv::SentimentLabel(s.compound, a->request) + "): negative " + Percent(s.negative) + ", neutral " +
              Percent(s.neutral) + ", positive " + Percent(s.positive) + " of the words' weight.");
    }
    SplitNext(split);
    {
        if (!a->by_group.empty()) {
            Heading("BY GROUP");
            if (BeginTable("##text_sentiment_groups", 5)) {
                ImGui::TableSetupColumn("group", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                ImGui::TableSetupColumn("rows", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                ImGui::TableSetupColumn("median", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                ImGui::TableSetupColumn("negative", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                ImGui::TableSetupColumn("positive", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                Heading("group");
                NumberCell("rows", &t.text_dim);
                NumberCell("median", &t.text_dim);
                NumberCell("negative", &t.text_dim);
                NumberCell("positive", &t.text_dim);
                for (const auto& g : a->by_group) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(g.group.c_str());
                    NumberCell(explore_view::Number(static_cast<size_t>(g.rows)));
                    NumberCell(Fixed(g.median, 2));
                    NumberCell(Percent(g.negative_share));
                    NumberCell(Percent(g.positive_share));
                }
                ImGui::EndTable();
            }
        }
        Faint("VADER (Hutto and Gilbert, 2014): a rated lexicon with rules for negation (\"not good\" reads negative), boosters (\"very\"), "
              "capitals, \"but\", emoticons and emojis. It was made for short social-media text; long rows add up many words.");
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph("SentimentAnalyzer", "Sentiment of " + base_.text, text::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", explore_view::StepTip("Sentiment Analysis", "the columns " + base_.text + "_compound, _positive, _neutral, _negative and " +
                                                                                       base_.text + "_sentiment beside the others")
                                        .c_str());
        ExploreNextButton("Export CSV...");
    }
    if (ui::SecondaryButton("Export CSV...")) Export("text_sentiment.csv", tv::SentimentCsv(*a));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("One row per row with text: row, group, compound, sentiment");
    ExploreNextButton("Copy");
    if (ui::SecondaryButton("Copy")) {
        ImGui::SetClipboardText((tv::SentimentSentence(*a) + "\n").c_str());
        Message("Copied the sentence.", false);
    }
}

// ---- Tokens

void TextView::RenderTokens(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    tokens_.base = base_;
    tokens_.row = TableRow();
    const std::string key = BaseKey(context.dataset) + "|tokens|" + std::to_string(tokens_.tokenizer) + "|" + std::to_string(tokens_.vocabulary) + "|" +
                            std::to_string(tokens_.min_frequency) + "|" + std::to_string(tokens_.max_length) + "|" + std::to_string(tokens_.row);
    const std::string dataset = context.dataset;
    const text::TokensRequest req = tokens_;
    const explore::MissingTexts texts = context.missing_texts;
    if (key != stopped_key_)
        tokens_job_.Ask("Text: tokens", key, [dataset, req, texts](LambdaTask& task) {
            return text::RunTokens(dataset, req, RunNow, texts, [&task] { return task.ShouldStop(); });
        });
    const text::TokensAnswer* a = tokens_job_.Answer(key);
    if (!a) {
        Waiting(tokens_job_, key, "Training the tokenizer on the column and reading the rows...");
        return;
    }
    ExploreFigures("##text_tokens_figures", {{"tokens per row", tv::LengthWords(a->per_row, "tokens")},
                                             {"cut at the limit", explore_view::Number(static_cast<size_t>(a->cut_rows)) + " rows (" + Percent(a->cut_share) + ")"},
                                             {"vocabulary", explore_view::Number(a->vocabulary_size) + " tokens, covers " + Percent(a->coverage)},
                                             {"tokenizer", kTokenizers[std::clamp(a->request.tokenizer, 0, kTokenizerCount - 1)]}});
    Wrapped(t.text, tv::TokensSentence(*a));
    RenderRowsLine(a->data);
    ImGui::Spacing();
    const bool split = BeginSplit("##text_tokens_split", 1.2f, 1.0f);
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading("TOKENS PER ROW, WITH THE LENGTH LIMIT");
        explore_plot::Bars(a->per_row.bins, 0.0, static_cast<double>(a->per_row.cap), w, PlotHeight(w), static_cast<double>(a->request.max_length),
                           "limit " + explore_view::Number(static_cast<size_t>(a->request.max_length)));
        if (a->per_row.over > 0)
            Faint("The last bar also holds the " + explore_view::Number(static_cast<size_t>(a->per_row.over)) + " rows longer than " +
                  explore_view::Number(static_cast<size_t>(a->per_row.cap)) + " tokens.");
    }
    SplitNext(split);
    {
        Heading(("ROW " + explore_view::Number(static_cast<size_t>(a->request.row + 1)) + " AS THE TOKENIZER READS IT").c_str());
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        bool first = true;
        for (size_t i = 0; i < a->row_tokens.size(); ++i) {
            const auto& tk = a->row_tokens[i];
            const std::string shown = tk.token + " " + std::to_string(tk.id);
            const float wide = ImGui::CalcTextSize(shown.c_str()).x + 8.0f;
            if (!first) {
                ImGui::SameLine(0.0f, 4.0f);
                if (ImGui::GetCursorScreenPos().x + wide > right) ImGui::NewLine();
            }
            first = false;
            ImGui::PushID(static_cast<int>(i));
            ImGui::TextColored(tk.unknown ? t.caution : t.text, "%s", tk.token.c_str());
            ImGui::SameLine(0.0f, 2.0f);
            ImGui::TextColored(t.text_faint, "%d", tk.id);
            if (tk.unknown && ImGui::IsItemHovered()) ImGui::SetTooltip("Not in the vocabulary: the unknown token");
            ImGui::PopID();
        }
        if (a->row_length > a->row_tokens.size())
            Faint("... " + explore_view::Number(a->row_length) + " tokens in all" +
                  (static_cast<int64_t>(a->row_length) > a->request.max_length ? "; the limit keeps the first " + explore_view::Number(static_cast<size_t>(a->request.max_length)) : "") +
                  ".");
        Faint("Each token with its id in the vocabulary; an unknown token is marked.");
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::PrimaryButton("Add as a step", true, nullptr, ui::ButtonSize::Small))
            Message(add_to_graph("TextTokenizer", "Tokens of " + base_.text, text::NodeParameters(a->request), true), false);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", explore_view::StepTip("Tokenizer", "token id columns (" + explore_view::Number(static_cast<size_t>(a->request.max_length)) +
                                                                            " per row) with the label and the split, for an Embedding, LSTM or Transformer model")
                                        .c_str());
        ExploreNextButton("Copy tokens");
    }
    if (ui::SecondaryButton("Copy tokens")) {
        std::string tokens;
        for (const auto& tk : a->row_tokens) tokens += tk.token + "\t" + std::to_string(tk.id) + "\n";
        ImGui::SetClipboardText(tokens.c_str());
        Message("Copied the row's tokens and ids.", false);
    }
}

// ---- Embeddings

void TextView::RenderEmbeddings(const ExploreContext& context) {
    const ui::Tokens& t = ui::CurrentTokens();
    embeddings_.base = base_;
    embeddings_.sample_words = all_rows_ ? 0 : 500000;
    const std::string key = BaseKey(context.dataset) + "|embeddings|" + embeddings_.method + (embeddings_.cbow ? "|cbow" : "|sg") + "|" +
                            std::to_string(embeddings_.size) + "|" + std::to_string(embeddings_.window) + "|" + std::to_string(embeddings_.min_count) + "|" +
                            std::to_string(embeddings_.epochs) + "|" + std::to_string(embeddings_.sample_words);
    const bool asked = learn_key_ == key;
    // Learn: always offered; off while these settings are learned or learning (the reason on hover).
    const bool learned = asked && key != stopped_key_ && (embeddings_job_.Answer(key) != nullptr || embeddings_job_.Running());
    const char* reason = embeddings_job_.Running() && asked ? "Learning these settings now (Stop below)" : "These settings are learned; change one to learn again";
    if (ui::PrimaryButton(embeddings_job_.Answer(learn_key_) == nullptr ? "Learn" : "Learn again", !learned, reason, ui::ButtonSize::Small)) {
        learn_key_ = key;
        stopped_key_.clear();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "%s", all_rows_ ? "from every row: a large column takes minutes" : "from a sample of the rows: seconds");
    if (!asked) {
        Faint("Learns word vectors from this column with the Word Embeddings node's trainers on the device: each word placed near the words used in the same "
              "places.");
        if (embeddings_job_.Answer(learn_key_) == nullptr) return;
    }
    if (asked && key != stopped_key_) {
        const std::string dataset = context.dataset;
        const text::EmbeddingsRequest req = embeddings_;
        const explore::MissingTexts texts = context.missing_texts;
        embeddings_job_.Ask("Text: word vectors", key, [dataset, req, texts](LambdaTask& task) {
            return text::RunEmbeddings(dataset, req, RunNow, texts, [&task] { return task.ShouldStop(); },
                                       [&task](double done) { task.ReportProgress(static_cast<float>(done), "Learning word vectors"); });
        });
    }
    const text::EmbeddingsAnswer* a = embeddings_job_.Answer(asked ? key : learn_key_);
    if (!a) {
        if (asked) Waiting(embeddings_job_, key, "Learning word vectors on the device...");
        return;
    }
    if (!asked) Faint("Shown: the vectors learned with the settings before; Learn again for the new ones.");
    const std::string method = a->request.method == "glove" ? std::string("GloVe")
                             : (a->request.method == "fasttext" ? std::string("fastText ") : std::string("word2vec ")) + (a->request.cbow ? "CBOW" : "skip-gram");
    // The word map shows these vectors.
    if (map_source_ != learn_key_) {
        map_source_ = learn_key_;
        WordMapView::Source source;
        source.vectors = std::make_shared<const WordVectors>(a->vectors);
        source.title = base_.text + ": " + method + ", " + explore_view::Number(a->vectors.words.size()) + " words";
        source.key = learn_key_;
        if (!a->group_rows.empty()) {
            source.groups = a->data.groups;
            source.word_rows = a->word_group_rows;
            source.group_rows = a->group_rows;
        }
        map_view_.SetSource(std::move(source));
        nearest_for_.clear();
    }
    const std::string nearest_key = learn_key_ + "|" + query_ + "|" + std::to_string(nearest_count_);
    if (nearest_key != nearest_for_) {
        nearest_for_ = nearest_key;
        nearest_error_.clear();
        nearest_ = query_[0] ? text::NearestWords(a->vectors, query_, nearest_count_, &nearest_error_) : std::vector<std::pair<std::string, double>>{};
    }
    std::vector<std::pair<std::string, std::string>> figures = {
        {"words learned", explore_view::Number(a->vectors.words.size()) + " (seen " + std::to_string(a->request.min_count) + " times or more)"},
        {"vectors", std::to_string(a->vectors.dim) + " numbers per word, " + method},
        {"learned from", a->sampled ? explore_view::Number(static_cast<size_t>(a->rows_used)) + " of " + explore_view::Number(a->data.texts.size()) + " rows (a sample)"
                                    : "every row"},
        {"words read", explore_view::Number(static_cast<size_t>(a->words_read))}};
    ExploreFigures("##text_embeddings_figures", figures);
    const std::string query_words = query_[0] ? word_map::QueryWord(query_) : std::string();
    Wrapped(t.text, tv::EmbeddingsSentence(*a, query_words, nearest_));
    RenderRowsLine(a->data);
    ImGui::Spacing();
    const bool split = BeginSplit("##text_embeddings_split", 1.0f, 1.3f);
    {
        Heading("NEAREST WORDS");
        Label("Word");
        ImGui::SetNextItemWidth(160.0f);
        // The table follows the typing; the map looks the word up on Enter or when the field is left
        // (its neighbours may need a new layout).
        ImGui::InputTextWithHint("##text_query", "type a word, then Enter", query_, sizeof query_);
        if (ImGui::IsItemDeactivated() && query_[0]) map_view_.LookUp(query_);
        Setting("show", 80.0f);
        ui::NumberField("##text_nearest", &nearest_count_, 1, 50, 1, 48.0f, "How many nearest words");
        if (query_words.empty()) Faint("Type a word to see the words nearest to it, here and on the map.");
        else if (!nearest_error_.empty()) Wrapped(t.error, nearest_error_);
        else if (nearest_.empty()) Faint("\"" + query_words + "\" has no vector: it is not seen " + std::to_string(a->request.min_count) + " times or more.");
        else if (BeginTable("##text_nearest_words", 2)) {
            ImGui::TableSetupColumn("word", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableSetupColumn("cosine", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Heading("word");
            NumberCell("cosine", &t.text_dim);
            for (const auto& [word, cosine] : nearest_) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(word.c_str());
                NumberCell(Fixed(cosine, 2));
            }
            ImGui::EndTable();
        }
        Faint("Words used in the same places sit together, opposites included.");
    }
    SplitNext(split);
    {
        const float w = ImGui::GetContentRegionAvail().x;
        Heading("THE WORD MAP (PREVIEW)");
        map_view_.DrawPreview(ImVec2(w, std::clamp(w * 0.55f, 200.0f, 320.0f)));
        if (ui::PrimaryButton("Open the word map", true, nullptr, ui::ButtonSize::Small)) map_view_.Open();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The words in 3D in a window of its own (dock it, float it, or fill the screen): the word looked up and its nearest words marked");
    }
    if (split) ImGui::EndTable();
    ImGui::Spacing();
    if (add_to_graph) {
        if (ui::SecondaryButton("Keep as a node"))
            Message(add_to_graph("WordEmbeddings", "Word vectors of " + base_.text, text::NodeParameters(a->request), false), false);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", (explore_view::BranchTip("Word Embeddings") +
                                     " It learns these vectors (the same settings) from every row when the graph runs, and saves them for an Embedding layer.")
                                        .c_str());
        ExploreNextButton("Save vectors (.vec)...");
    }
    if (ui::SecondaryButton("Save vectors (.vec)...")) {
        if (const auto path = FileDialogs::SaveFile("Save word vectors", {{"Word vectors", "vec"}}, nullptr, "word_vectors.vec")) {
            std::string error;
            const bool ok = WordEmbeddings::SaveText(a->vectors, *path, &error);
            Message(ok ? "Saved " + *path : "Could not save: " + error, !ok);
        }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("word2vec text format: a word and its numbers per line; the Word Embeddings node can load it");
}

}  // namespace cyxwiz
