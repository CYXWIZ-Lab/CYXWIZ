"""
DataLoader Example

Demonstrates how to use the DataLoader for high-performance data loading
with DuckDB integration for SQL queries, batch iteration, and schema inspection.
"""

import pycyxwiz as cx
import tempfile
import os

# Initialize CyxWiz
cx.initialize()


def create_test_csv():
    """Create a temporary CSV file for testing."""
    csv_content = """id,name,value,category
1,apple,10.5,fruit
2,banana,5.25,fruit
3,carrot,3.0,vegetable
4,orange,8.75,fruit
5,broccoli,4.5,vegetable
6,grape,12.0,fruit
7,spinach,2.25,vegetable
8,mango,15.5,fruit
9,potato,1.75,vegetable
10,strawberry,20.0,fruit
"""
    fd, path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w') as f:
        f.write(csv_content)
    return path


def main():
    print("=" * 60)
    print("CyxWiz DataLoader Example")
    print("=" * 60)

    # Check if DuckDB is available
    print("\n1. Check DuckDB Availability")
    print("-" * 40)

    if not cx.DataLoader.is_available():
        print("   DuckDB not available - DataLoader features disabled")
        print("   Build with CYXWIZ_HAS_DUCKDB=ON to enable")
        return

    print(f"   DuckDB available: {cx.DataLoader.is_available()}")
    print(f"   DuckDB version: {cx.DataLoader.get_version()}")

    # Create test data
    csv_path = create_test_csv()
    print(f"\n   Created test CSV: {csv_path}")

    try:
        # Example 2: Basic DataLoader usage
        print("\n2. Basic DataLoader Usage")
        print("-" * 40)

        # Create loader with default config
        loader = cx.DataLoader()
        print("   Created DataLoader with default config")

        # Load entire CSV
        data = loader.load_csv(csv_path)
        print(f"   Loaded CSV data shape: {data.shape()}")

        # Example 3: Load specific columns
        print("\n3. Load Specific Columns")
        print("-" * 40)

        # Only load numeric columns
        data = loader.load_csv(csv_path, columns=["id", "value"])
        print(f"   Loaded columns [id, value], shape: {data.shape()}")

        # Example 4: SQL Queries
        print("\n4. SQL Queries")
        print("-" * 40)

        # Simple SELECT
        result = loader.query(f"SELECT * FROM '{csv_path}'")
        print(f"   SELECT * result shape: {result.shape()}")

        # SELECT with WHERE clause
        result = loader.query(f"SELECT id, value FROM '{csv_path}' WHERE value > 5")
        print(f"   SELECT WHERE value > 5 shape: {result.shape()}")

        # Aggregation query
        result = loader.query(f"SELECT AVG(value), MAX(value), MIN(value) FROM '{csv_path}'")
        print(f"   Aggregation result shape: {result.shape()}")

        # Example 5: Schema Inspection
        print("\n5. Schema Inspection")
        print("-" * 40)

        schema = loader.get_schema(csv_path)
        print(f"   Number of columns: {len(schema)}")
        for col in schema:
            print(f"     - {col.name}: {col.type} (nullable: {col.nullable})")

        columns = loader.get_columns(csv_path)
        print(f"   Column names: {columns}")

        row_count = loader.get_row_count(csv_path)
        print(f"   Row count: {row_count}")

        # Example 6: Batch Iterator (for large datasets)
        print("\n6. Batch Iterator")
        print("-" * 40)

        sql = f"SELECT id, value FROM '{csv_path}'"
        batch_size = 3

        print(f"   Iterating with batch size: {batch_size}")

        iterator = loader.create_batch_iterator(sql, batch_size)
        batch_num = 0
        total_rows = 0

        while iterator.has_next():
            batch = iterator.next()
            rows = batch.shape()[0]
            total_rows += rows
            print(f"     Batch {batch_num}: {rows} rows")
            batch_num += 1

        print(f"   Total batches: {batch_num}")
        print(f"   Total rows processed: {total_rows}")

        # Reset and iterate again
        iterator.reset()
        first_batch = iterator.next()
        print(f"   After reset, first batch shape: {first_batch.shape()}")

        # Example 7: Custom Configuration
        print("\n7. Custom Configuration")
        print("-" * 40)

        config = cx.DataLoaderConfig()
        config.batch_size = 512
        config.verbose = True
        config.num_threads = 8

        loader2 = cx.DataLoader(config)
        print(f"   Created loader with batch_size={config.batch_size}")
        print(f"   Verbose: {config.verbose}")
        print(f"   Num threads: {config.num_threads}")

        # Example 8: Using with Polars (if available)
        print("\n8. Integration with Polars")
        print("-" * 40)

        try:
            import polars as pl

            # Load with Polars for comparison
            df = pl.read_csv(csv_path)
            print(f"   Polars DataFrame shape: {df.shape}")

            # Use Polars SQL on the same data
            lazy_df = pl.scan_csv(csv_path)
            result_df = lazy_df.filter(pl.col("value") > 5).collect()
            print(f"   Polars filtered result: {result_df.shape}")

            # Both CyxWiz DataLoader and Polars can be used together
            print("   Polars integration verified!")
        except ImportError:
            print("   Polars not installed - pip install polars")

        print("\n" + "=" * 60)
        print("DataLoader Example Complete!")
        print("=" * 60)

    finally:
        # Cleanup
        os.unlink(csv_path)
        print(f"\n   Cleaned up: {csv_path}")


if __name__ == "__main__":
    main()
