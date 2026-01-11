# DuckDB & Polars in CyxWiz Command Window

CyxWiz provides built-in support for **DuckDB** (SQL analytics) and **Polars** (fast DataFrames) in the Python Command Window.

## Quick Start

Open the Command Window from **View > Command Window** or the sidebar. On first command, you'll see:
```
[CyxWiz] DuckDB loaded - use sql(), read_csv(), read_parquet(), read_json()
[CyxWiz] Polars loaded - use pl, df(), col(), scan_csv(), etc.
```

---

## DuckDB Examples

### Basic SQL Queries
```python
# Simple calculation
sql("SELECT 1 + 1 AS result")

# Create table and query
sql("CREATE TABLE users (id INT, name VARCHAR, age INT)")
sql("INSERT INTO users VALUES (1, 'Alice', 25), (2, 'Bob', 30), (3, 'Carol', 28)")
sql("SELECT * FROM users WHERE age > 25")
```

### Query Files Directly (No Import Needed)
```python
# Query CSV file with SQL
sql("SELECT * FROM 'data/sales.csv' LIMIT 10")

# Aggregate data from Parquet
sql("SELECT category, SUM(amount) as total FROM 'data/transactions.parquet' GROUP BY category")

# Join multiple files
sql("""
    SELECT a.*, b.category_name
    FROM 'products.csv' a
    JOIN 'categories.csv' b ON a.cat_id = b.id
""")
```

### Load Files into DuckDB
```python
# Read CSV
data = read_csv('data/sample.csv')
data.show()

# Read Parquet (fast columnar format)
data = read_parquet('data/large_dataset.parquet')

# Read JSON
data = read_json('data/config.json')
```

### Advanced DuckDB Features
```python
# Window functions
sql("""
    SELECT name, salary,
           RANK() OVER (ORDER BY salary DESC) as rank
    FROM 'employees.csv'
""")

# CTEs (Common Table Expressions)
sql("""
    WITH top_customers AS (
        SELECT customer_id, SUM(amount) as total
        FROM 'orders.csv'
        GROUP BY customer_id
        ORDER BY total DESC
        LIMIT 10
    )
    SELECT * FROM top_customers
""")

# Export results
sql("COPY (SELECT * FROM 'data.csv') TO 'output.parquet' (FORMAT PARQUET)")
```

---

## Polars Examples

### Create DataFrames
```python
# From dictionary
data = df({'name': ['Alice', 'Bob', 'Carol'],
           'age': [25, 30, 28],
           'city': ['NYC', 'LA', 'Chicago']})

# From list of dicts
data = df([
    {'name': 'Alice', 'score': 95},
    {'name': 'Bob', 'score': 87},
    {'name': 'Carol', 'score': 92}
])
```

### Read Files
```python
# CSV
data = pl_csv('data/sales.csv')

# Parquet (recommended for large files)
data = pl_parquet('data/large_dataset.parquet')

# JSON
data = pl_json('data/records.json')

# Excel
data = pl_excel('data/report.xlsx')
```

### Filter and Select
```python
data = df({'name': ['Alice', 'Bob', 'Carol'], 'age': [25, 30, 28]})

# Filter rows
data.filter(col('age') > 25)

# Select columns
data.select(['name', 'age'])

# Multiple conditions
data.filter((col('age') > 25) & (col('name') != 'Bob'))
```

### Transform Data
```python
# Add new column
data.with_columns([
    (col('age') + 1).alias('next_year_age'),
    col('name').str.to_uppercase().alias('NAME')
])

# Rename columns
data.rename({'name': 'full_name'})

# Sort
data.sort('age', descending=True)
```

### Aggregations
```python
# Group by and aggregate
sales = pl_csv('sales.csv')
sales.group_by('region').agg([
    col('amount').sum().alias('total_sales'),
    col('amount').mean().alias('avg_sale'),
    col('order_id').count().alias('num_orders')
])

# Multiple aggregations
data.select([
    col('price').mean().alias('avg_price'),
    col('price').min().alias('min_price'),
    col('price').max().alias('max_price'),
    col('price').std().alias('std_price')
])
```

### Lazy Evaluation (For Large Files)
```python
# Lazy read - doesn't load until .collect()
lazy_df = scan_csv('huge_file.csv')

# Build query plan
result = (lazy_df
    .filter(col('status') == 'active')
    .group_by('category')
    .agg(col('value').sum())
    .sort('value', descending=True)
    .limit(10)
    .collect())  # Execute here
```

### Join DataFrames
```python
orders = df({'order_id': [1, 2, 3], 'customer_id': [101, 102, 101]})
customers = df({'customer_id': [101, 102], 'name': ['Alice', 'Bob']})

# Inner join
orders.join(customers, on='customer_id')

# Left join
orders.join(customers, on='customer_id', how='left')
```

---

## Combined DuckDB + Polars Workflow

```python
# 1. Use DuckDB for complex SQL transformations
result = sql("""
    SELECT
        date_trunc('month', order_date) as month,
        category,
        SUM(amount) as total
    FROM 'orders.parquet'
    WHERE order_date >= '2024-01-01'
    GROUP BY 1, 2
""")

# 2. Convert to Polars for further processing
df_result = result.pl()  # DuckDB to Polars

# 3. Continue with Polars operations
df_result.pivot(
    values='total',
    index='month',
    columns='category'
)
```

---

## Available Functions Reference

### DuckDB
| Function | Description |
|----------|-------------|
| `sql(query)` | Execute SQL query |
| `read_csv(path)` | Load CSV file |
| `read_parquet(path)` | Load Parquet file |
| `read_json(path)` | Load JSON file |
| `db` | DuckDB connection object |

### Polars
| Function | Description |
|----------|-------------|
| `pl` | Polars module |
| `df(data)` | Create DataFrame |
| `lf(data)` | Create LazyFrame |
| `col('name')` | Column expression |
| `lit(value)` | Literal value |
| `when(cond)` | Conditional expression |
| `pl_csv(path)` | Read CSV |
| `pl_parquet(path)` | Read Parquet |
| `pl_json(path)` | Read JSON |
| `pl_excel(path)` | Read Excel |
| `scan_csv(path)` | Lazy CSV read |
| `scan_parquet(path)` | Lazy Parquet read |

---

## Tips

1. **Large files**: Use `scan_csv()`/`scan_parquet()` for lazy evaluation
2. **SQL on any file**: DuckDB can query CSV/Parquet/JSON directly without loading
3. **Performance**: Polars is typically 10-100x faster than pandas
4. **Memory**: Both libraries use Apache Arrow for efficient memory usage
5. **Type `help()`** in Command Window for quick reference
