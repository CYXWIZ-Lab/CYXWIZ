"""
Data Analysis & EDA Example

Real-world example showing exploratory data analysis workflows:
- Sales data analysis
- Time-based aggregations
- Top-N queries
- Statistical summaries
- Data quality checks

Run this in CyxWiz Command Window or as a standalone script.
"""

import pycyxwiz as cx
import tempfile
import os
from datetime import datetime, timedelta
import random

# Initialize CyxWiz
cx.initialize()


def create_sales_csv():
    """Create a realistic sales dataset for analysis."""
    # Generate 100 sales records
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse", "Headphones"]
    regions = ["North", "South", "East", "West"]
    categories = {
        "Laptop": "Electronics",
        "Phone": "Electronics",
        "Tablet": "Electronics",
        "Monitor": "Electronics",
        "Keyboard": "Accessories",
        "Mouse": "Accessories",
        "Headphones": "Audio"
    }
    prices = {
        "Laptop": 999.99,
        "Phone": 699.99,
        "Tablet": 449.99,
        "Monitor": 349.99,
        "Keyboard": 79.99,
        "Mouse": 29.99,
        "Headphones": 149.99
    }

    lines = ["order_id,date,product,category,region,quantity,unit_price,total_amount"]

    base_date = datetime(2024, 1, 1)
    for i in range(1, 101):
        product = random.choice(products)
        region = random.choice(regions)
        category = categories[product]
        quantity = random.randint(1, 10)
        unit_price = prices[product]
        total = quantity * unit_price
        date = base_date + timedelta(days=random.randint(0, 180))

        lines.append(
            f"{i},{date.strftime('%Y-%m-%d')},{product},{category},"
            f"{region},{quantity},{unit_price:.2f},{total:.2f}"
        )

    csv_content = "\n".join(lines)
    fd, path = tempfile.mkstemp(suffix='_sales.csv')
    with os.fdopen(fd, 'w') as f:
        f.write(csv_content)
    return path


def main():
    print("=" * 70)
    print("   Data Analysis & EDA with CyxWiz DataLoader")
    print("=" * 70)

    if not cx.DataLoader.is_available():
        print("\nDuckDB not available. Build with CYXWIZ_HAS_DUCKDB=ON")
        return

    # Create sales dataset
    data_path = create_sales_csv()
    print(f"\nCreated sales dataset: {data_path}")

    try:
        loader = cx.DataLoader()

        # =================================================================
        # 1. Dataset Overview
        # =================================================================
        print("\n" + "-" * 70)
        print("   1. DATASET OVERVIEW")
        print("-" * 70)

        schema = loader.get_schema(data_path)
        row_count = loader.get_row_count(data_path)

        print(f"\nDataset: {row_count} sales records")
        print("\nColumns:")
        for col in schema:
            print(f"  {col.name:15s} | {col.type}")

        # =================================================================
        # 2. Sales Summary Statistics
        # =================================================================
        print("\n" + "-" * 70)
        print("   2. SALES SUMMARY STATISTICS")
        print("-" * 70)

        summary = loader.query(f"""
            SELECT
                COUNT(*) as total_orders,
                SUM(quantity) as total_units,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as avg_order_value,
                MIN(total_amount) as min_order,
                MAX(total_amount) as max_order
            FROM '{data_path}'
        """)
        print(f"\nSummary statistics: {summary.shape()}")

        # =================================================================
        # 3. Sales by Product (Top-N Analysis)
        # =================================================================
        print("\n" + "-" * 70)
        print("   3. TOP PRODUCTS BY REVENUE")
        print("-" * 70)

        top_products = loader.query(f"""
            SELECT
                product,
                COUNT(*) as orders,
                SUM(quantity) as units_sold,
                SUM(total_amount) as revenue
            FROM '{data_path}'
            GROUP BY product
            ORDER BY revenue DESC
        """)
        print(f"\nProduct performance: {top_products.shape()}")

        # =================================================================
        # 4. Sales by Region
        # =================================================================
        print("\n" + "-" * 70)
        print("   4. REGIONAL ANALYSIS")
        print("-" * 70)

        by_region = loader.query(f"""
            SELECT
                region,
                COUNT(*) as orders,
                SUM(total_amount) as revenue,
                AVG(total_amount) as avg_order
            FROM '{data_path}'
            GROUP BY region
            ORDER BY revenue DESC
        """)
        print(f"\nRegional breakdown: {by_region.shape()}")

        # =================================================================
        # 5. Category Analysis
        # =================================================================
        print("\n" + "-" * 70)
        print("   5. CATEGORY BREAKDOWN")
        print("-" * 70)

        by_category = loader.query(f"""
            SELECT
                category,
                COUNT(DISTINCT product) as unique_products,
                SUM(quantity) as units_sold,
                SUM(total_amount) as revenue
            FROM '{data_path}'
            GROUP BY category
            ORDER BY revenue DESC
        """)
        print(f"\nCategory analysis: {by_category.shape()}")

        # =================================================================
        # 6. Monthly Trends
        # =================================================================
        print("\n" + "-" * 70)
        print("   6. MONTHLY SALES TRENDS")
        print("-" * 70)

        monthly = loader.query(f"""
            SELECT
                STRFTIME(date, '%Y-%m') as month,
                COUNT(*) as orders,
                SUM(total_amount) as revenue
            FROM '{data_path}'
            GROUP BY month
            ORDER BY month
        """)
        print(f"\nMonthly trends: {monthly.shape()}")

        # =================================================================
        # 7. High-Value Orders
        # =================================================================
        print("\n" + "-" * 70)
        print("   7. HIGH-VALUE ORDERS (Top 10)")
        print("-" * 70)

        top_orders = loader.query(f"""
            SELECT
                order_id,
                date,
                product,
                quantity,
                total_amount
            FROM '{data_path}'
            ORDER BY total_amount DESC
            LIMIT 10
        """)
        print(f"\nTop 10 orders: {top_orders.shape()}")

        # =================================================================
        # 8. Data Quality Check
        # =================================================================
        print("\n" + "-" * 70)
        print("   8. DATA QUALITY CHECKS")
        print("-" * 70)

        # Check for anomalies
        quality = loader.query(f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(CASE WHEN quantity <= 0 THEN 1 END) as invalid_qty,
                COUNT(CASE WHEN total_amount < 0 THEN 1 END) as negative_amounts,
                COUNT(DISTINCT product) as unique_products,
                COUNT(DISTINCT region) as unique_regions
            FROM '{data_path}'
        """)
        print(f"\nQuality metrics: {quality.shape()}")

        # =================================================================
        # 9. Cross-Tabulation (Product x Region)
        # =================================================================
        print("\n" + "-" * 70)
        print("   9. PRODUCT x REGION MATRIX")
        print("-" * 70)

        cross_tab = loader.query(f"""
            SELECT
                product,
                region,
                SUM(total_amount) as revenue
            FROM '{data_path}'
            GROUP BY product, region
            ORDER BY product, region
        """)
        print(f"\nCross-tabulation: {cross_tab.shape()}")

        # =================================================================
        # 10. Filtering & Export Pattern
        # =================================================================
        print("\n" + "-" * 70)
        print("   10. FILTERED DATA FOR EXPORT")
        print("-" * 70)

        # Filter large orders in specific region for further analysis
        filtered = loader.query(f"""
            SELECT *
            FROM '{data_path}'
            WHERE total_amount > 500
              AND region = 'North'
            ORDER BY date
        """)
        print(f"\nFiltered dataset (North, >$500): {filtered.shape()}")

        # =================================================================
        # Summary
        # =================================================================
        print("\n" + "=" * 70)
        print("   EDA COMPLETE - KEY QUERIES DEMONSTRATED")
        print("=" * 70)
        print("""
  Queries demonstrated:
    1. COUNT, SUM, AVG, MIN, MAX        - Aggregations
    2. GROUP BY product                  - Top products
    3. GROUP BY region                   - Regional analysis
    4. COUNT(DISTINCT)                   - Category breakdown
    5. STRFTIME for dates                - Monthly trends
    6. ORDER BY ... LIMIT                - Top-N queries
    7. CASE WHEN                         - Data quality flags
    8. GROUP BY col1, col2               - Cross-tabulation
    9. WHERE with multiple conditions    - Filtering

  All queries return Tensor objects that can be:
    - Used directly in ML models
    - Converted to numpy arrays
    - Processed further with CyxWiz operations
        """)

    finally:
        os.unlink(data_path)
        print(f"\nCleaned up: {data_path}")


if __name__ == "__main__":
    main()
