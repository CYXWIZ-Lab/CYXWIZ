#pragma once

// One group of the missing-values fix card (feature columns or the label),
// drawn the same in the "Cannot start training" report and on the Training
// Dashboard (TOFIX223 0.3). The words come from
// core/missing_values_presentation; this draws them and says what was clicked.

#include "../core/missing_values_presentation.h"

#include <string>

namespace cyxwiz::ui {

struct MissingValuesGroupEvents {
    bool fill_changed = false;  // the numbers' value (Mean / Median / 0) changed
    bool fill = false;          // Fill these columns
    bool drop = false;          // Drop these rows / Drop rows without a label
    bool undo = false;          // Undo the steps that were added
    std::string leave_out;      // Actions > Leave <column> out of the features
};

// A card (BeginCard / EndCard) with the group's title, its columns, then its
// actions (open) or what was added with Undo (resolved). `id` is stable.
MissingValuesGroupEvents MissingValuesGroup(const char* id, const missing_values_fix::Group& group, bool label,
                                            missing_values_fix::NumberFill& fill);

}  // namespace cyxwiz::ui
