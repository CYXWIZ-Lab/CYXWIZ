#pragma once

#include "node_editor.h"

#include <string>

namespace gui::detail {

inline bool IsDenseEncodedSequencePlaceholder(NodeType node_type,
                                              const std::string& node_name) {
    if (node_type != NodeType::Dense) {
        return false;
    }

    const char* placeholder_names[] = {
        "NERSequenceBuilder",
        "TokenVocabulary",
        "POSVocabulary",
        "NERTagVocabulary",
        "SequencePadding",
        "FeatureConcat",
        "TimeDistributedDense",
        "TokenCrossEntropyLoss",
        "SequenceTagMetrics",
        "SequenceTagOutput"
    };

    for (const char* name : placeholder_names) {
        if (node_name == name) {
            return true;
        }
    }
    return false;
}

} // namespace gui::detail
