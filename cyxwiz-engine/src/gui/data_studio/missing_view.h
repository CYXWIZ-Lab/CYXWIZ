#pragma once

// Data Studio > Explore > Missing (TOFIX129 D1, approved mockup board 1;
// replaces the Missing Values window): missing cells, complete and incomplete
// rows, every column's count, the columns that go missing together, and for
// the picked column its advice and rows (50 a page), with Drop these rows in
// Prepare and Open in Query. Counts come from Profile's profile; the rest
// from core/data_explore.

#include "explore_tab.h"
#include "../../core/data_explore.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class MissingView {
public:
    void Render(const ExploreContext& context);
    // Picks the column (Profile's missing count, Data > Explore).
    void Select(const std::string& column) { column_ = column; offset_ = 0; }
    bool IsRunning() const { return summary_.Running() || rows_.Running(); }

    std::function<void(const std::string& sql)> on_open_query;
    std::function<void(const std::string& column)> on_open_profile;
    // Adds the Keep rows step after the dataset's node; returns what happened
    // (or why not). Null, or `drop_blocker` non-empty: the button is disabled.
    std::function<std::string(const std::string& column, const std::vector<std::string>& texts)> on_drop_rows;
    std::function<std::string()> drop_blocker;

private:
    void RenderColumns(const ExploreContext& context, float height);
    void RenderPatterns(const ExploreContext& context);
    void RenderColumn(const ExploreContext& context);

    std::shared_ptr<const DatasetProfile> seen_;  // the profile the answers are for
    std::string column_;
    size_t offset_ = 0;
    int sort_ = 1;          // 0 column name, 1 missing count
    bool descending_ = true;
    std::string message_;
    bool message_error_ = false;
    ExploreJob<explore::MissingSummary> summary_;
    ExploreJob<explore::RowPage> rows_;
};

}  // namespace cyxwiz
