/**
 * Arrow Integration Test (Phase 0 Day 5)
 *
 * Tests the complete Data Studio foundation:
 * 1. ArrowDataset: Load CSV/Parquet files
 * 2. DataRegistry: Register and manage Arrow tables
 * 3. ArrowConverters: Bidirectional Arrow <-> Tensor conversion
 * 4. DuckDBConnector: SQL queries on Arrow tables
 *
 * This validates the zero-copy pipeline that Data Studio nodes will use.
 */

#include "../src/core/arrow_dataset.h"
#include "../src/core/data_registry.h"
#include "../src/core/arrow_converters.h"
#include "../src/core/duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <cassert>

// Include ArrayFire header for tests
#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

using namespace cyxwiz;

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a test CSV file
 */
void CreateTestCSV(const std::string& path) {
    std::ofstream file(path);
    file << "id,name,age,salary,department\n";
    file << "1,Alice,30,75000.50,Engineering\n";
    file << "2,Bob,25,60000.00,Sales\n";
    file << "3,Charlie,35,85000.75,Engineering\n";
    file << "4,Diana,28,70000.00,Marketing\n";
    file << "5,Eve,32,80000.25,Engineering\n";
    file << "6,Frank,29,65000.00,Sales\n";
    file.close();

    spdlog::info("Created test CSV: {}", path);
}

/**
 * Test result validation
 */
#define TEST_ASSERT(condition, message) \
    if (!(condition)) { \
        spdlog::error("TEST FAILED: {} ({}:{})", message, __FILE__, __LINE__); \
        return false; \
    } else { \
        spdlog::info("✓ {}", message); \
    }

// =============================================================================
// Test Cases
// =============================================================================

/**
 * Test 1: ArrowDataset Loading
 */
bool Test_ArrowDataset_Load() {
    spdlog::info("\n=== Test 1: ArrowDataset Loading ===");

    // Create test data
    std::string csv_path = "test_arrow_load.csv";
    CreateTestCSV(csv_path);

    // Load CSV
    auto dataset = ArrowDataset::FromCSV(csv_path, "employees");
    TEST_ASSERT(dataset != nullptr, "Load CSV file");
    TEST_ASSERT(dataset->GetNumRows() == 6, "Correct row count");
    TEST_ASSERT(dataset->GetNumColumns() == 5, "Correct column count");

    // Check column names
    auto columns = dataset->GetColumnNames();
    TEST_ASSERT(columns.size() == 5, "Column names retrieved");
    TEST_ASSERT(columns[0] == "id", "First column is 'id'");
    TEST_ASSERT(columns[4] == "department", "Last column is 'department'");

    // Check memory usage
    size_t memory = dataset->GetMemoryUsage();
    TEST_ASSERT(memory > 0, "Memory usage calculated");

    spdlog::info("ArrowDataset: {} bytes memory", memory);

    // Cleanup
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 2: ArrowDataset Operations
 */
bool Test_ArrowDataset_Operations() {
    spdlog::info("\n=== Test 2: ArrowDataset Operations ===");

    std::string csv_path = "test_arrow_ops.csv";
    CreateTestCSV(csv_path);

    auto dataset = ArrowDataset::FromCSV(csv_path, "employees");
    TEST_ASSERT(dataset != nullptr, "Dataset loaded");

    // Select columns
    auto subset = dataset->SelectColumns({"name", "age", "salary"});
    TEST_ASSERT(subset != nullptr, "Column selection");
    TEST_ASSERT(subset->GetNumColumns() == 3, "Correct subset column count");

    // Slice rows
    auto slice = dataset->Slice(1, 3);  // Rows 1-3
    TEST_ASSERT(slice != nullptr, "Row slicing");
    TEST_ASSERT(slice->GetNumRows() == 3, "Correct slice row count");

    // Filter rows by indices
    std::vector<int64_t> indices = {0, 2, 4};  // Rows 0, 2, 4
    auto filtered = dataset->FilterRows(indices);
    TEST_ASSERT(filtered != nullptr, "Row filtering");
    TEST_ASSERT(filtered->GetNumRows() == 3, "Correct filtered row count");

    // Cleanup
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 3: DataRegistry Integration
 */
bool Test_DataRegistry_Arrow() {
    spdlog::info("\n=== Test 3: DataRegistry Integration ===");

    std::string csv_path = "test_registry.csv";
    CreateTestCSV(csv_path);

    auto& registry = DataRegistry::Instance();

    // Load via registry
    auto dataset = registry.LoadArrowTable(csv_path, "employees");
    TEST_ASSERT(dataset != nullptr, "Load Arrow table via registry");

    // Retrieve from registry
    auto retrieved = registry.GetArrowDataset("employees");
    TEST_ASSERT(retrieved != nullptr, "Retrieve Arrow dataset");
    TEST_ASSERT(retrieved->GetName() == "employees", "Correct dataset name");

    // Check if registered
    TEST_ASSERT(registry.IsArrowDataset("employees"), "Dataset is registered");
    TEST_ASSERT(!registry.IsArrowDataset("nonexistent"), "Non-existent check works");

    // Cleanup
    registry.UnloadDataset("employees");
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 4: Arrow to Tensor Conversion
 */
bool Test_ArrowToTensor() {
    spdlog::info("\n=== Test 4: Arrow to Tensor Conversion ===");

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::warn("ArrayFire not available - skipping tensor conversion test");
    return true;
#endif

    std::string csv_path = "test_tensor.csv";
    CreateTestCSV(csv_path);

    auto dataset = ArrowDataset::FromCSV(csv_path, "data");
    TEST_ASSERT(dataset != nullptr, "Dataset loaded");

    auto table = dataset->GetArrowTable();

    // Get numeric columns
    auto numeric_cols = arrow_converters::GetNumericColumns(table);
    TEST_ASSERT(!numeric_cols.empty(), "Numeric columns detected");
    spdlog::info("Numeric columns: {}", numeric_cols.size());

    // Convert to tensor
    arrow_converters::ConversionOptions opts;
    opts.selected_columns = {"age", "salary"};  // Only numeric columns
    auto tensor = arrow_converters::ArrowToTensor(table, opts);
    TEST_ASSERT(tensor != nullptr, "Arrow to Tensor conversion");

    // Check tensor dimensions
    ::af::dim4 dims = tensor->dims();
    TEST_ASSERT(dims[0] == 6, "Correct tensor rows");
    TEST_ASSERT(dims[1] == 2, "Correct tensor columns");

    spdlog::info("Tensor shape: [{}, {}]", dims[0], dims[1]);

    // Cleanup
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 5: Tensor to Arrow Conversion
 */
bool Test_TensorToArrow() {
    spdlog::info("\n=== Test 5: Tensor to Arrow Conversion ===");

#ifndef CYXWIZ_HAS_ARRAYFIRE
    spdlog::warn("ArrayFire not available - skipping");
    return true;
#endif

    // Create a simple tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ::af::array tensor(3, 2, data.data());  // 3 rows, 2 cols

    // Convert to Arrow
    std::vector<std::string> col_names = {"feature1", "feature2"};
    auto table = arrow_converters::TensorToArrow(tensor, col_names, "test_data");
    TEST_ASSERT(table != nullptr, "Tensor to Arrow conversion");
    TEST_ASSERT(table->num_rows() == 3, "Correct row count");
    TEST_ASSERT(table->num_columns() == 2, "Correct column count");

    spdlog::info("Arrow table: {} rows, {} columns", table->num_rows(), table->num_columns());

    return true;
}

/**
 * Test 6: DuckDB SQL Queries
 */
bool Test_DuckDB_Queries() {
    spdlog::info("\n=== Test 6: DuckDB SQL Queries ===");

    std::string csv_path = "test_duckdb.csv";
    CreateTestCSV(csv_path);

    auto dataset = ArrowDataset::FromCSV(csv_path, "employees");
    TEST_ASSERT(dataset != nullptr, "Dataset loaded");

    // Create DuckDB connector
    DuckDBConnector db;

    // Register table
    bool registered = db.RegisterTable("employees", dataset->GetArrowTable());
    TEST_ASSERT(registered, "Register Arrow table in DuckDB");

    // Test 1: Simple SELECT
    auto result = db.Query("SELECT * FROM employees WHERE age > 28");
    TEST_ASSERT(result != nullptr, "Simple SELECT query");
    TEST_ASSERT(result->num_rows() == 4, "Correct filtered result count");

    // Test 2: Aggregation
    auto agg_result = db.Query("SELECT department, COUNT(*) as count, AVG(salary) as avg_salary FROM employees GROUP BY department");
    TEST_ASSERT(agg_result != nullptr, "Aggregation query");
    spdlog::info("Aggregation result: {} rows", agg_result->num_rows());

    // Test 3: Sorting
    auto sorted = db.Query("SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3");
    TEST_ASSERT(sorted != nullptr, "ORDER BY query");
    TEST_ASSERT(sorted->num_rows() == 3, "Top 3 salaries");

    // Test 4: Convenience methods
    auto filtered = db.FilterRows("employees", "department = 'Engineering'");
    TEST_ASSERT(filtered != nullptr, "FilterRows convenience method");
    TEST_ASSERT(filtered->num_rows() == 3, "Engineering department has 3 employees");

    auto selected = db.SelectColumns("employees", {"name", "age"});
    TEST_ASSERT(selected != nullptr, "SelectColumns convenience method");
    TEST_ASSERT(selected->num_columns() == 2, "2 columns selected");

    // Cleanup
    db.UnregisterTable("employees");
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 7: End-to-End Data Studio Pipeline
 */
bool Test_EndToEnd_Pipeline() {
    spdlog::info("\n=== Test 7: End-to-End Pipeline ===");

    // This simulates a typical Data Studio workflow:
    // 1. Load CSV -> Arrow
    // 2. SQL query for data cleaning
    // 3. Convert to Tensor for training
    // 4. Convert predictions back to Arrow
    // 5. Export to Parquet

    std::string csv_path = "test_e2e.csv";
    CreateTestCSV(csv_path);

    // Step 1: Load data
    auto& registry = DataRegistry::Instance();
    auto dataset = registry.LoadArrowTable(csv_path, "raw_data");
    TEST_ASSERT(dataset != nullptr, "Load raw data");

    // Step 2: SQL data cleaning
    DuckDBConnector db;
    db.RegisterTable("raw_data", dataset->GetArrowTable());

    auto cleaned = db.Query(R"(
        SELECT age, salary
        FROM raw_data
        WHERE age BETWEEN 25 AND 35
          AND salary > 60000
    )");
    TEST_ASSERT(cleaned != nullptr, "Data cleaning query");
    spdlog::info("Cleaned data: {} rows", cleaned->num_rows());

    // Step 3: Convert to tensor for training
#ifdef CYXWIZ_HAS_ARRAYFIRE
    auto tensor = arrow_converters::ArrowToTensor(cleaned);
    TEST_ASSERT(tensor != nullptr, "Convert cleaned data to tensor");
    ::af::dim4 tensor_dims = tensor->dims();
    spdlog::info("Training tensor: [{}, {}]", tensor_dims[0], tensor_dims[1]);

    // Step 4: Simulate predictions and convert back
    // (In real workflow, this would be model.predict(tensor))
    ::af::array predictions = ::af::randu(tensor_dims[0], 1);  // Random predictions
    auto results = arrow_converters::TensorToArrow(predictions, {"prediction"}, "results");
    TEST_ASSERT(results != nullptr, "Convert predictions to Arrow");

    // Step 5: Register results for export
    auto results_dataset = registry.RegisterArrowTable(results, "predictions");
    TEST_ASSERT(results_dataset != nullptr, "Register prediction results");
#endif

    // Export to Parquet
    std::string parquet_path = "test_results.parquet";
    bool exported = dataset->ExportParquet(parquet_path);
    TEST_ASSERT(exported, "Export to Parquet");

    // Cleanup
    registry.UnloadDataset("raw_data");
    std::remove(csv_path.c_str());
    std::remove(parquet_path.c_str());

    return true;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    spdlog::set_level(spdlog::level::info);
    spdlog::info("\n╔══════════════════════════════════════════════════════════╗");
    spdlog::info("║   CyxWiz Engine - Arrow Integration Test Suite         ║");
    spdlog::info("║   Phase 0: Data Studio Foundation                       ║");
    spdlog::info("╚══════════════════════════════════════════════════════════╝\n");

    int passed = 0;
    int failed = 0;

    struct TestCase {
        const char* name;
        bool (*func)();
    };

    TestCase tests[] = {
        {"ArrowDataset Loading", Test_ArrowDataset_Load},
        {"ArrowDataset Operations", Test_ArrowDataset_Operations},
        {"DataRegistry Integration", Test_DataRegistry_Arrow},
        {"Arrow to Tensor Conversion", Test_ArrowToTensor},
        {"Tensor to Arrow Conversion", Test_TensorToArrow},
        {"DuckDB SQL Queries", Test_DuckDB_Queries},
        {"End-to-End Pipeline", Test_EndToEnd_Pipeline},
    };

    for (const auto& test : tests) {
        spdlog::info("\n▶ Running: {}", test.name);
        try {
            if (test.func()) {
                passed++;
                spdlog::info("✓ PASSED: {}", test.name);
            } else {
                failed++;
                spdlog::error("✗ FAILED: {}", test.name);
            }
        } catch (const std::exception& e) {
            failed++;
            spdlog::error("✗ EXCEPTION in {}: {}", test.name, e.what());
        }
    }

    spdlog::info("\n" + std::string(60, '='));
    spdlog::info("Test Results: {} passed, {} failed", passed, failed);
    spdlog::info(std::string(60, '=') + "\n");

    return (failed == 0) ? 0 : 1;
}
