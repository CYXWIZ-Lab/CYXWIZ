"""
Quick Start Snippets for Command Window

Copy-paste these snippets into CyxWiz Command Window for quick data analysis.
Each section is self-contained and can be run independently.

Usage in Command Window:
  1. Open Command Window (View -> Command Window or Ctrl+Shift+P)
  2. Copy a snippet below
  3. Paste and press Ctrl+Enter to run
"""

# =============================================================================
# SNIPPET 1: Load and inspect a CSV file
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

# Check if DuckDB is available
print(f"DuckDB available: {cx.DataLoader.is_available()}")
print(f"Version: {cx.DataLoader.get_version()}")

# Create loader and inspect a file
loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Get schema
schema = loader.get_schema(path)
for col in schema:
    print(f"  {col.name}: {col.type}")

# Get row count
print(f"Rows: {loader.get_row_count(path)}")
"""

# =============================================================================
# SNIPPET 2: Run SQL query and get results
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Simple SELECT
result = loader.query(f"SELECT * FROM '{path}' LIMIT 10")
print(f"Shape: {result.shape()}")

# Aggregation
stats = loader.query(f'''
    SELECT COUNT(*), AVG(column_name), MAX(column_name)
    FROM '{path}'
''')
print(f"Stats: {stats.shape()}")
"""

# =============================================================================
# SNIPPET 3: Filter and group data
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Filter rows
filtered = loader.query(f'''
    SELECT *
    FROM '{path}'
    WHERE column_name > 100
    ORDER BY column_name DESC
''')
print(f"Filtered: {filtered.shape()}")

# Group by
grouped = loader.query(f'''
    SELECT category, COUNT(*) as cnt, AVG(value) as avg_val
    FROM '{path}'
    GROUP BY category
    ORDER BY cnt DESC
''')
print(f"Grouped: {grouped.shape()}")
"""

# =============================================================================
# SNIPPET 4: Load specific columns for ML
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Load only numeric features
features = loader.load_csv(path, columns=["col1", "col2", "col3"])
print(f"Features: {features.shape()}")

# Use the tensor
# features is a Tensor object ready for ML operations
"""

# =============================================================================
# SNIPPET 5: Batch iteration for large datasets
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Create batch iterator
sql = f"SELECT * FROM '{path}'"
batch_iter = loader.create_batch_iterator(sql, batch_size=32)

# Process batches
batch_count = 0
total_rows = 0
while batch_iter.has_next():
    batch = batch_iter.next()
    batch_count += 1
    total_rows += batch.shape()[0]
    print(f"Batch {batch_count}: {batch.shape()}")

print(f"Total: {total_rows} rows in {batch_count} batches")
"""

# =============================================================================
# SNIPPET 6: Join two CSV files
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
file1 = "path/to/features.csv"
file2 = "path/to/labels.csv"

# Join on common column
joined = loader.query(f'''
    SELECT a.*, b.label
    FROM '{file1}' a
    JOIN '{file2}' b ON a.id = b.id
''')
print(f"Joined: {joined.shape()}")
"""

# =============================================================================
# SNIPPET 7: Quick statistics summary
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Get comprehensive stats
stats = loader.query(f'''
    SELECT
        COUNT(*) as count,
        AVG(value) as mean,
        STDDEV(value) as std,
        MIN(value) as min,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q3,
        MAX(value) as max
    FROM '{path}'
''')
print(f"Stats: {stats.shape()}")
"""

# =============================================================================
# SNIPPET 8: Working with dates
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Filter by date range
filtered = loader.query(f'''
    SELECT *
    FROM '{path}'
    WHERE date >= '2024-01-01' AND date < '2024-07-01'
''')
print(f"Date filtered: {filtered.shape()}")

# Group by month
monthly = loader.query(f'''
    SELECT
        STRFTIME(date, '%Y-%m') as month,
        COUNT(*) as count,
        SUM(amount) as total
    FROM '{path}'
    GROUP BY month
    ORDER BY month
''')
print(f"Monthly: {monthly.shape()}")
"""

# =============================================================================
# SNIPPET 9: Random sampling
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Sample 10% of data
sample = loader.query(f'''
    SELECT * FROM '{path}'
    USING SAMPLE 10%
''')
print(f"Sample: {sample.shape()}")

# Sample fixed number
sample_n = loader.query(f'''
    SELECT * FROM '{path}'
    USING SAMPLE 1000 ROWS
''')
print(f"Sample N: {sample_n.shape()}")
"""

# =============================================================================
# SNIPPET 10: Create ML dataset with encoding
# =============================================================================
"""
import pycyxwiz as cx
cx.initialize()

loader = cx.DataLoader()
path = "path/to/your/data.csv"  # Change this!

# Create ML-ready dataset with encoded categoricals
ml_data = loader.query(f'''
    SELECT
        feature1,
        feature2,
        feature3,
        CASE category
            WHEN 'A' THEN 0
            WHEN 'B' THEN 1
            WHEN 'C' THEN 2
        END as category_encoded,
        CASE WHEN flag = 'yes' THEN 1 ELSE 0 END as flag_binary
    FROM '{path}'
''')
print(f"ML Dataset: {ml_data.shape()}")
"""

print(__doc__)
print("\nSnippets ready! Copy any section to Command Window.")
