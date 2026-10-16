#pragma once

// Test Results exports (TOFIX129 B5 E-3): what the Test Results window shows,
// written as CSV or JSON. The caller passes the results it shows, so an export
// never writes an older or cleared run.

#include "test_executor.h"

#include <string>

namespace cyxwiz {

// Overview, per-class metrics (with found / false alarms / missed), the
// confusion matrix with class names, and each prediction.
bool WriteTestResultsCsv(const TestingMetrics& results, const std::string& path, std::string* error = nullptr);

// Everything the CSV has, plus pair / retrieval metrics and the ROC and
// precision-recall curves.
bool WriteTestResultsJson(const TestingMetrics& results, const std::string& path, std::string* error = nullptr);

}  // namespace cyxwiz
