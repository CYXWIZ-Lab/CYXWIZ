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
#include <cmath>

// Include ArrayFire header for tests
#include <arrayfire.h>

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
 * Test 6: Categorical encoding preserves values and nulls
 */
bool Test_CategoricalEncoding() {
    spdlog::info("\n=== Test 6: Categorical Encoding ===");

    arrow::StringBuilder source_builder;
    TEST_ASSERT(source_builder.Append("red").ok(), "Append first category");
    TEST_ASSERT(source_builder.AppendNull().ok(), "Append null category");
    TEST_ASSERT(source_builder.Append("blue").ok(), "Append second category");
    TEST_ASSERT(source_builder.Append("red").ok(), "Append repeated category");

    std::shared_ptr<arrow::Array> source_array;
    TEST_ASSERT(source_builder.Finish(&source_array).ok(),
                "Finish categorical source array");

    auto source_table = arrow::Table::Make(
        arrow::schema({arrow::field("color", arrow::utf8())}),
        {source_array});

    std::vector<std::string> vocabulary;
    auto encoded = arrow_converters::EncodeCategorical(
        source_table, "color", vocabulary);
    TEST_ASSERT(encoded != nullptr, "Encode categorical column");
    TEST_ASSERT(vocabulary == std::vector<std::string>({"red", "blue"}),
                "Vocabulary preserves first-seen order");

    auto encoded_values = std::static_pointer_cast<arrow::Int32Array>(
        encoded->GetColumnByName("color_encoded")->chunk(0));
    TEST_ASSERT(encoded_values->Value(0) == 0, "Encode first category as zero");
    TEST_ASSERT(encoded_values->IsNull(1), "Preserve null encoded value");
    TEST_ASSERT(encoded_values->Value(2) == 1, "Encode second category as one");
    TEST_ASSERT(encoded_values->Value(3) == 0, "Reuse existing category code");

    auto one_hot = arrow_converters::OneHotEncode(source_table, "color");
    TEST_ASSERT(one_hot != nullptr, "One-hot encode categorical column");
    TEST_ASSERT(one_hot->GetColumnByName("color_red") != nullptr,
                "Create first one-hot column");
    TEST_ASSERT(one_hot->GetColumnByName("color_blue") != nullptr,
                "Create second one-hot column");

    auto red_values = std::static_pointer_cast<arrow::BooleanArray>(
        one_hot->GetColumnByName("color_red")->chunk(0));
    auto blue_values = std::static_pointer_cast<arrow::BooleanArray>(
        one_hot->GetColumnByName("color_blue")->chunk(0));
    TEST_ASSERT(red_values->Value(0) && !blue_values->Value(0),
                "Set first one-hot category");
    TEST_ASSERT(red_values->IsNull(1) && blue_values->IsNull(1),
                "Preserve nulls in one-hot columns");
    TEST_ASSERT(!red_values->Value(2) && blue_values->Value(2),
                "Set second one-hot category");

    return true;
}

/**
 * Test 7: DuckDB SQL Queries
 */
bool Test_DuckDB_Queries() {
    spdlog::info("\n=== Test 7: DuckDB SQL Queries ===");

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

    // Test 4: Filter and project
    auto filtered = db.Query("SELECT name, age FROM employees WHERE department = 'Engineering'");
    TEST_ASSERT(filtered != nullptr, "Filter and project");
    TEST_ASSERT(filtered->num_rows() == 3 && filtered->num_columns() == 2, "Engineering: 3 employees, 2 columns");

    // Every connection is restricted: SQL cannot read files (the CSV is on disk right here).
    TEST_ASSERT(db.Query("SELECT * FROM read_csv('" + csv_path + "')") == nullptr, "no file access from SQL");
    TEST_ASSERT(!db.Execute("SET enable_external_access = true"), "the configuration is locked");

    // DECIMAL results (DuckDB's type for 0.1) are numbers, not text.
    auto decimals = db.Query("SELECT 0.1 AS tenth, 1234567.5 AS big, CAST(NULL AS DECIMAL(4,1)) AS none");
    TEST_ASSERT(decimals != nullptr, "DECIMAL query");
    TEST_ASSERT(decimals->schema()->field(0)->type()->id() == arrow::Type::DOUBLE &&
                    decimals->schema()->field(1)->type()->id() == arrow::Type::DOUBLE,
                "DECIMAL columns are double");
    auto tenth = decimals->column(0)->GetScalar(0);
    TEST_ASSERT(tenth.ok() && std::static_pointer_cast<arrow::DoubleScalar>(*tenth)->value == 0.1, "0.1 stays 0.1");
    TEST_ASSERT(decimals->column(2)->null_count() == 1, "a NULL decimal stays null");

    // The result type table: the per-value path (TINYINT, UINTEGER, DATE) and
    // the native export give the same Arrow types; the SQL step's exactness
    // check reads the same table.
    auto typed = db.Query("SELECT CAST(1 AS TINYINT) AS t, CAST(2 AS UINTEGER) AS u, DATE '2024-01-02' AS d");
    TEST_ASSERT(typed != nullptr, "typed query");
    TEST_ASSERT(typed->schema()->field(0)->type()->id() == arrow::Type::INT32 &&
                    typed->schema()->field(1)->type()->id() == arrow::Type::UINT32 &&
                    typed->schema()->field(2)->type()->id() == arrow::Type::DATE32,
                "TINYINT int32, UINTEGER uint32, DATE date32");
    auto day = typed->column(2)->GetScalar(0);
    TEST_ASSERT(day.ok() && (*day)->ToString() == "2024-01-02", "the date itself");
    // Natively exported dates and timestamps; a decimal beside them takes the per-value path.
    auto when = db.Query("SELECT DATE '2024-01-02' AS d, TIMESTAMP '2024-01-02 03:04:05' AS ts");
    TEST_ASSERT(when != nullptr && when->schema()->field(0)->type()->id() == arrow::Type::DATE32 &&
                    when->schema()->field(1)->type()->id() == arrow::Type::TIMESTAMP,
                "native export: DATE date32, TIMESTAMP timestamp");
    auto when_slow = db.Query("SELECT TIMESTAMP '2024-01-02 03:04:05' AS ts, 0.5 AS w");
    TEST_ASSERT(when_slow != nullptr && when->column(1)->GetScalar(0).ValueOrDie()->Equals(
                                            *when_slow->column(0)->GetScalar(0).ValueOrDie()),
                "a timestamp is the same on both paths");
    // Dates, timestamps and decimals going in stay dates, timestamps and numbers.
    {
        arrow::Date32Builder db32;
        arrow::TimestampBuilder tsb(arrow::timestamp(arrow::TimeUnit::MILLI), arrow::default_memory_pool());
        arrow::Decimal128Builder decb(arrow::decimal128(10, 2));
        TEST_ASSERT(db32.Append(19724).ok() && tsb.Append(1704164645000).ok() && decb.Append(arrow::Decimal128(1234)).ok(),
                    "typed input builders");
        std::shared_ptr<arrow::Array> d_arr, t_arr, m_arr;
        TEST_ASSERT(db32.Finish(&d_arr).ok() && tsb.Finish(&t_arr).ok() && decb.Finish(&m_arr).ok(), "typed input arrays");
        auto typed_in = arrow::Table::Make(arrow::schema({arrow::field("d", arrow::date32()),
                                                          arrow::field("t", arrow::timestamp(arrow::TimeUnit::MILLI)),
                                                          arrow::field("m", arrow::decimal128(10, 2))}),
                                           {d_arr, t_arr, m_arr});
        TEST_ASSERT(db.RegisterTable("typed_in", typed_in), "register dates, timestamps, decimals");
        auto back = db.Query("SELECT d + 1 AS next_day, year(t) AS y, m * 2 AS doubled FROM typed_in");
        TEST_ASSERT(back != nullptr, "SQL on typed input");
        TEST_ASSERT(back->column(0)->GetScalar(0).ValueOrDie()->ToString() == "2024-01-03", "date arithmetic");
        TEST_ASSERT(back->column(1)->GetScalar(0).ValueOrDie()->ToString() == "2024", "timestamp functions");
        TEST_ASSERT(std::abs(std::static_pointer_cast<arrow::DoubleScalar>(back->column(2)->GetScalar(0).ValueOrDie())->value - 24.68) < 1e-9,
                    "a decimal input is a number");
        db.UnregisterTable("typed_in");
    }
    auto native = db.Query("SELECT CAST(1 AS INTEGER) AS i, 'x' AS s");
    TEST_ASSERT(native != nullptr && native->schema()->field(0)->type()->id() == arrow::Type::INT32 &&
                    native->schema()->field(1)->type()->id() == arrow::Type::STRING,
                "native export: INTEGER int32, VARCHAR text");
    TEST_ASSERT(db.UnsupportedResultColumns("SELECT 1 AS a, 'x' AS b").empty(), "exact columns pass the check");
    TEST_ASSERT(db.UnsupportedResultColumns("SELECT DATE '2024-01-02' AS d, TIMESTAMP '2024-01-02 03:04' AS t, 0.5 AS w").empty(),
                "DATE, TIMESTAMP and a short DECIMAL pass the check");
    const std::string why = db.UnsupportedResultColumns(
        "SELECT CAST(1 AS DECIMAL(18,2)) AS wide, CAST(1 AS HUGEINT) AS h, TIMESTAMPTZ '2024-01-02 03:04+00' AS tz, 1 AS a");
    TEST_ASSERT(why.find("'wide'") != std::string::npos && why.find("'h'") != std::string::npos &&
                    why.find("'tz'") != std::string::npos && why.find("'a'") == std::string::npos,
                "a DECIMAL past 15 digits, HUGEINT and a time-zone timestamp are named as not exact");

    // Cleanup
    db.UnregisterTable("employees");
    std::remove(csv_path.c_str());

    return true;
}

/**
 * Test 8: End-to-End Data Studio Pipeline
 */
bool Test_EndToEnd_Pipeline() {
    spdlog::info("\n=== Test 8: End-to-End Pipeline ===");

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
        {"Categorical Encoding", Test_CategoricalEncoding},
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
