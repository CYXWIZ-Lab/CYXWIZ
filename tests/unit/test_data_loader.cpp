#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/data_loader.h>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// Helper to create a test CSV file
static std::string CreateTestCSV(const std::string& content) {
    fs::path temp = fs::temp_directory_path() / "test_data_loader.csv";
    std::ofstream file(temp);
    file << content;
    file.close();
    return temp.string();
}

TEST_CASE("DataLoader availability", "[data_loader]") {
    // This test always passes - just checks compile-time availability
    bool available = cyxwiz::DataLoader::IsAvailable();
    INFO("DuckDB available: " << (available ? "yes" : "no"));

    if (available) {
        std::string version = cyxwiz::DataLoader::GetVersion();
        INFO("DuckDB version: " << version);
        REQUIRE(!version.empty());
    }
}

TEST_CASE("DataLoader construction", "[data_loader]") {
    SECTION("Default construction") {
        if (!cyxwiz::DataLoader::IsAvailable()) {
            SKIP("DuckDB not available");
        }

        cyxwiz::DataLoader loader;
        REQUIRE(loader.GetConfig().batch_size == 1024);
    }

    SECTION("Custom configuration") {
        if (!cyxwiz::DataLoader::IsAvailable()) {
            SKIP("DuckDB not available");
        }

        cyxwiz::DataLoaderConfig config;
        config.batch_size = 512;
        config.verbose = true;

        cyxwiz::DataLoader loader(config);
        REQUIRE(loader.GetConfig().batch_size == 512);
        REQUIRE(loader.GetConfig().verbose == true);
    }
}

TEST_CASE("DataLoader LoadCSV", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    // Create test CSV
    std::string csv_path = CreateTestCSV(
        "a,b,c\n"
        "1.0,2.0,3.0\n"
        "4.0,5.0,6.0\n"
        "7.0,8.0,9.0\n"
    );

    cyxwiz::DataLoader loader;

    SECTION("Load all columns") {
        cyxwiz::Tensor data = loader.LoadCSV(csv_path);

        REQUIRE(data.NumDimensions() == 2);
        REQUIRE(data.Shape()[0] == 3);  // 3 rows
        REQUIRE(data.Shape()[1] == 3);  // 3 columns
    }

    SECTION("Load specific columns") {
        cyxwiz::Tensor data = loader.LoadCSV(csv_path, {"a", "c"});

        REQUIRE(data.NumDimensions() == 2);
        REQUIRE(data.Shape()[0] == 3);  // 3 rows
        REQUIRE(data.Shape()[1] == 2);  // 2 columns
    }

    // Cleanup
    fs::remove(csv_path);
}

TEST_CASE("DataLoader Query", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    // Create test CSV
    std::string csv_path = CreateTestCSV(
        "x,y,z\n"
        "1.0,10.0,100.0\n"
        "2.0,20.0,200.0\n"
        "3.0,30.0,300.0\n"
        "4.0,40.0,400.0\n"
        "5.0,50.0,500.0\n"
    );

    cyxwiz::DataLoader loader;

    SECTION("Simple SELECT") {
        std::string sql = "SELECT * FROM '" + csv_path + "'";
        cyxwiz::Tensor result = loader.Query(sql);

        REQUIRE(result.Shape()[0] == 5);  // 5 rows
        REQUIRE(result.Shape()[1] == 3);  // 3 columns
    }

    SECTION("SELECT with WHERE") {
        std::string sql = "SELECT x, y FROM '" + csv_path + "' WHERE x > 2";
        cyxwiz::Tensor result = loader.Query(sql);

        REQUIRE(result.Shape()[0] == 3);  // 3 rows (x=3,4,5)
        REQUIRE(result.Shape()[1] == 2);  // 2 columns
    }

    SECTION("SELECT with aggregation") {
        std::string sql = "SELECT SUM(x), AVG(y) FROM '" + csv_path + "'";
        cyxwiz::Tensor result = loader.Query(sql);

        REQUIRE(result.Shape()[0] == 1);  // 1 row (aggregation result)
        REQUIRE(result.Shape()[1] == 2);  // 2 columns
    }

    // Cleanup
    fs::remove(csv_path);
}

TEST_CASE("DataLoader BatchIterator", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    // Create larger test CSV
    std::stringstream ss;
    ss << "value\n";
    for (int i = 0; i < 100; i++) {
        ss << i << ".0\n";
    }
    std::string csv_path = CreateTestCSV(ss.str());

    cyxwiz::DataLoader loader;
    std::string sql = "SELECT * FROM '" + csv_path + "'";

    SECTION("Iterate with batch size 25") {
        auto iter = loader.CreateBatchIterator(sql, 25);

        int batch_count = 0;
        size_t total_rows = 0;

        while (iter.HasNext()) {
            cyxwiz::Tensor batch = iter.Next();
            total_rows += batch.Shape()[0];
            batch_count++;
        }

        REQUIRE(batch_count == 4);  // 100 rows / 25 batch = 4 batches
        REQUIRE(total_rows == 100);
    }

    SECTION("Reset iterator") {
        auto iter = loader.CreateBatchIterator(sql, 50);

        // Consume all batches
        while (iter.HasNext()) {
            iter.Next();
        }
        REQUIRE(!iter.HasNext());

        // Reset and iterate again
        iter.Reset();
        REQUIRE(iter.HasNext());

        cyxwiz::Tensor first_batch = iter.Next();
        REQUIRE(first_batch.Shape()[0] == 50);
    }

    // Cleanup
    fs::remove(csv_path);
}

TEST_CASE("DataLoader GetSchema", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    std::string csv_path = CreateTestCSV(
        "id,name,value\n"
        "1,foo,10.5\n"
        "2,bar,20.5\n"
    );

    cyxwiz::DataLoader loader;
    auto schema = loader.GetSchema(csv_path);

    REQUIRE(schema.size() == 3);
    REQUIRE(schema[0].name == "id");
    REQUIRE(schema[1].name == "name");
    REQUIRE(schema[2].name == "value");

    // Cleanup
    fs::remove(csv_path);
}

TEST_CASE("DataLoader GetColumns", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    std::string csv_path = CreateTestCSV(
        "col_a,col_b,col_c\n"
        "1,2,3\n"
    );

    cyxwiz::DataLoader loader;
    auto columns = loader.GetColumns(csv_path);

    REQUIRE(columns.size() == 3);
    REQUIRE(columns[0] == "col_a");
    REQUIRE(columns[1] == "col_b");
    REQUIRE(columns[2] == "col_c");

    // Cleanup
    fs::remove(csv_path);
}

TEST_CASE("DataLoader GetRowCount", "[data_loader]") {
    if (!cyxwiz::DataLoader::IsAvailable()) {
        SKIP("DuckDB not available");
    }

    std::string csv_path = CreateTestCSV(
        "x\n"
        "1\n"
        "2\n"
        "3\n"
        "4\n"
        "5\n"
    );

    cyxwiz::DataLoader loader;
    size_t count = loader.GetRowCount(csv_path);

    REQUIRE(count == 5);

    // Cleanup
    fs::remove(csv_path);
}
