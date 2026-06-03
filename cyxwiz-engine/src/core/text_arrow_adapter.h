#pragma once

#include "formats/text_dataset.h"

#include <arrow/api.h>

#include <memory>
#include <string>

namespace cyxwiz {

/**
 * Build a raw Arrow table from an already-loaded TextDataset.
 *
 * Schema:
 *   - text_column: utf8, one raw text sample per row
 *   - label_column: int32, only emitted when TextDataset reports classes
 *
 * CSV/TSV text files still use Arrow's native CSV reader in TextLoader so
 * original columns are preserved. This adapter covers JSON/JSONL, TXT, and
 * folder corpora that TextDataset already parses.
 */
arrow::Result<std::shared_ptr<arrow::Table>> BuildRawTextArrowTable(
    const TextDataset& dataset,
    const std::string& text_column,
    const std::string& label_column);

} // namespace cyxwiz
