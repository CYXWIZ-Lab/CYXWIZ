"""
Multi-File Data Pipeline Example

Real-world example showing how to work with multiple data files:
- Join tables (customers + orders + products)
- Aggregate across files
- Feature engineering from multiple sources
- Create unified ML datasets

This pattern is common in:
- E-commerce analytics
- Financial data processing
- Healthcare record linkage
- Manufacturing quality analysis
"""

import pycyxwiz as cx
import tempfile
import os
import random

# Initialize CyxWiz
cx.initialize()


def create_test_files():
    """Create related test files for join demonstrations."""

    # 1. Customers table
    customers_csv = """customer_id,name,segment,country
1,Alice,Premium,USA
2,Bob,Standard,USA
3,Charlie,Premium,UK
4,Diana,Standard,Germany
5,Eve,Premium,France
6,Frank,Standard,USA
7,Grace,Premium,UK
8,Henry,Standard,Germany
"""

    # 2. Products table
    products_csv = """product_id,product_name,category,price
101,Laptop Pro,Electronics,1299.99
102,Wireless Mouse,Accessories,49.99
103,USB-C Hub,Accessories,79.99
104,Monitor 27,Electronics,399.99
105,Mechanical Keyboard,Accessories,129.99
106,Webcam HD,Electronics,89.99
107,Headphones,Audio,199.99
108,Speakers,Audio,149.99
"""

    # 3. Orders table (links customers and products)
    orders_lines = ["order_id,customer_id,product_id,quantity,order_date,status"]
    order_id = 1000
    statuses = ["completed", "completed", "completed", "shipped", "pending"]
    for _ in range(50):
        customer_id = random.randint(1, 8)
        product_id = random.choice([101, 102, 103, 104, 105, 106, 107, 108])
        quantity = random.randint(1, 5)
        month = random.randint(1, 6)
        day = random.randint(1, 28)
        status = random.choice(statuses)
        orders_lines.append(
            f"{order_id},{customer_id},{product_id},{quantity},2024-{month:02d}-{day:02d},{status}"
        )
        order_id += 1
    orders_csv = "\n".join(orders_lines)

    # 4. Customer reviews/ratings
    reviews_lines = ["review_id,customer_id,product_id,rating,review_date"]
    review_id = 5000
    for _ in range(30):
        customer_id = random.randint(1, 8)
        product_id = random.choice([101, 102, 103, 104, 105, 106, 107, 108])
        rating = random.randint(1, 5)
        month = random.randint(1, 6)
        day = random.randint(1, 28)
        reviews_lines.append(
            f"{review_id},{customer_id},{product_id},{rating},2024-{month:02d}-{day:02d}"
        )
        review_id += 1
    reviews_csv = "\n".join(reviews_lines)

    # Create temp files
    files = {}
    for name, content in [
        ("customers", customers_csv),
        ("products", products_csv),
        ("orders", orders_csv),
        ("reviews", reviews_csv)
    ]:
        fd, path = tempfile.mkstemp(suffix=f'_{name}.csv')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        files[name] = path

    return files


def main():
    print("=" * 70)
    print("   Multi-File Data Pipeline with CyxWiz DataLoader")
    print("=" * 70)

    if not cx.DataLoader.is_available():
        print("\nDuckDB not available. Build with CYXWIZ_HAS_DUCKDB=ON")
        return

    # Create test files
    files = create_test_files()
    print("\nCreated test files:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    try:
        loader = cx.DataLoader()

        # =================================================================
        # 1. Inspect All Tables
        # =================================================================
        print("\n" + "-" * 70)
        print("   1. TABLE SCHEMAS")
        print("-" * 70)

        for name, path in files.items():
            schema = loader.get_schema(path)
            rows = loader.get_row_count(path)
            cols = ", ".join([c.name for c in schema])
            print(f"\n  {name.upper()} ({rows} rows)")
            print(f"    Columns: {cols}")

        # =================================================================
        # 2. Simple Two-Table Join
        # =================================================================
        print("\n" + "-" * 70)
        print("   2. ORDERS + PRODUCTS JOIN")
        print("-" * 70)

        orders_products = loader.query(f"""
            SELECT
                o.order_id,
                o.customer_id,
                p.product_name,
                p.category,
                o.quantity,
                p.price,
                o.quantity * p.price as order_total
            FROM '{files["orders"]}' o
            JOIN '{files["products"]}' p ON o.product_id = p.product_id
            LIMIT 10
        """)
        print(f"\nOrders with product details: {orders_products.shape()}")

        # =================================================================
        # 3. Three-Table Join (Full Order Details)
        # =================================================================
        print("\n" + "-" * 70)
        print("   3. FULL ORDER DETAILS (3-way join)")
        print("-" * 70)

        full_orders = loader.query(f"""
            SELECT
                o.order_id,
                c.name as customer_name,
                c.segment,
                c.country,
                p.product_name,
                p.category,
                o.quantity,
                p.price * o.quantity as total_amount,
                o.order_date,
                o.status
            FROM '{files["orders"]}' o
            JOIN '{files["customers"]}' c ON o.customer_id = c.customer_id
            JOIN '{files["products"]}' p ON o.product_id = p.product_id
            ORDER BY o.order_date DESC
            LIMIT 15
        """)
        print(f"\nFull order details: {full_orders.shape()}")

        # =================================================================
        # 4. Customer Spending Analysis
        # =================================================================
        print("\n" + "-" * 70)
        print("   4. CUSTOMER SPENDING ANALYSIS")
        print("-" * 70)

        customer_spending = loader.query(f"""
            SELECT
                c.customer_id,
                c.name,
                c.segment,
                COUNT(o.order_id) as total_orders,
                SUM(o.quantity) as total_items,
                SUM(p.price * o.quantity) as total_spent
            FROM '{files["customers"]}' c
            LEFT JOIN '{files["orders"]}' o ON c.customer_id = o.customer_id
            LEFT JOIN '{files["products"]}' p ON o.product_id = p.product_id
            GROUP BY c.customer_id, c.name, c.segment
            ORDER BY total_spent DESC
        """)
        print(f"\nCustomer spending summary: {customer_spending.shape()}")

        # =================================================================
        # 5. Product Performance with Reviews
        # =================================================================
        print("\n" + "-" * 70)
        print("   5. PRODUCT PERFORMANCE + RATINGS")
        print("-" * 70)

        product_perf = loader.query(f"""
            SELECT
                p.product_id,
                p.product_name,
                p.category,
                COALESCE(sales.total_sold, 0) as units_sold,
                COALESCE(sales.revenue, 0) as revenue,
                COALESCE(ratings.avg_rating, 0) as avg_rating,
                COALESCE(ratings.num_reviews, 0) as num_reviews
            FROM '{files["products"]}' p
            LEFT JOIN (
                SELECT
                    product_id,
                    SUM(quantity) as total_sold,
                    SUM(quantity * (SELECT price FROM '{files["products"]}' WHERE product_id = o.product_id)) as revenue
                FROM '{files["orders"]}' o
                WHERE status = 'completed'
                GROUP BY product_id
            ) sales ON p.product_id = sales.product_id
            LEFT JOIN (
                SELECT
                    product_id,
                    AVG(rating) as avg_rating,
                    COUNT(*) as num_reviews
                FROM '{files["reviews"]}'
                GROUP BY product_id
            ) ratings ON p.product_id = ratings.product_id
            ORDER BY revenue DESC
        """)
        print(f"\nProduct performance: {product_perf.shape()}")

        # =================================================================
        # 6. Segment Analysis
        # =================================================================
        print("\n" + "-" * 70)
        print("   6. CUSTOMER SEGMENT ANALYSIS")
        print("-" * 70)

        segment_analysis = loader.query(f"""
            SELECT
                c.segment,
                COUNT(DISTINCT c.customer_id) as customers,
                COUNT(o.order_id) as orders,
                SUM(p.price * o.quantity) as total_revenue,
                AVG(p.price * o.quantity) as avg_order_value
            FROM '{files["customers"]}' c
            LEFT JOIN '{files["orders"]}' o ON c.customer_id = o.customer_id
            LEFT JOIN '{files["products"]}' p ON o.product_id = p.product_id
            GROUP BY c.segment
        """)
        print(f"\nSegment analysis: {segment_analysis.shape()}")

        # =================================================================
        # 7. Country-Category Cross Analysis
        # =================================================================
        print("\n" + "-" * 70)
        print("   7. COUNTRY x CATEGORY MATRIX")
        print("-" * 70)

        cross_analysis = loader.query(f"""
            SELECT
                c.country,
                p.category,
                SUM(o.quantity) as units,
                SUM(p.price * o.quantity) as revenue
            FROM '{files["orders"]}' o
            JOIN '{files["customers"]}' c ON o.customer_id = c.customer_id
            JOIN '{files["products"]}' p ON o.product_id = p.product_id
            GROUP BY c.country, p.category
            ORDER BY c.country, p.category
        """)
        print(f"\nCountry-Category matrix: {cross_analysis.shape()}")

        # =================================================================
        # 8. Feature Engineering for ML
        # =================================================================
        print("\n" + "-" * 70)
        print("   8. ML FEATURE ENGINEERING")
        print("-" * 70)

        # Create customer features for churn prediction or RFM analysis
        ml_features = loader.query(f"""
            SELECT
                c.customer_id,
                CASE c.segment WHEN 'Premium' THEN 1 ELSE 0 END as is_premium,
                CASE c.country
                    WHEN 'USA' THEN 0
                    WHEN 'UK' THEN 1
                    WHEN 'Germany' THEN 2
                    WHEN 'France' THEN 3
                END as country_code,
                COALESCE(stats.order_count, 0) as order_count,
                COALESCE(stats.total_spent, 0) as total_spent,
                COALESCE(stats.avg_order, 0) as avg_order,
                COALESCE(stats.distinct_products, 0) as product_variety,
                COALESCE(ratings.avg_rating, 0) as avg_rating_given
            FROM '{files["customers"]}' c
            LEFT JOIN (
                SELECT
                    o.customer_id,
                    COUNT(*) as order_count,
                    SUM(p.price * o.quantity) as total_spent,
                    AVG(p.price * o.quantity) as avg_order,
                    COUNT(DISTINCT o.product_id) as distinct_products
                FROM '{files["orders"]}' o
                JOIN '{files["products"]}' p ON o.product_id = p.product_id
                GROUP BY o.customer_id
            ) stats ON c.customer_id = stats.customer_id
            LEFT JOIN (
                SELECT customer_id, AVG(rating) as avg_rating
                FROM '{files["reviews"]}'
                GROUP BY customer_id
            ) ratings ON c.customer_id = ratings.customer_id
        """)
        print(f"\nML feature matrix: {ml_features.shape()}")
        print("  Features: is_premium, country_code, order_count, total_spent,")
        print("            avg_order, product_variety, avg_rating_given")

        # =================================================================
        # 9. Batch Processing Across Files
        # =================================================================
        print("\n" + "-" * 70)
        print("   9. BATCH PROCESSING FOR TRAINING")
        print("-" * 70)

        batch_sql = f"""
            SELECT
                c.customer_id,
                CASE c.segment WHEN 'Premium' THEN 1 ELSE 0 END as label,
                COALESCE(SUM(p.price * o.quantity), 0) as total_spent,
                COALESCE(COUNT(o.order_id), 0) as order_count
            FROM '{files["customers"]}' c
            LEFT JOIN '{files["orders"]}' o ON c.customer_id = o.customer_id
            LEFT JOIN '{files["products"]}' p ON o.product_id = p.product_id
            GROUP BY c.customer_id, c.segment
        """

        batch_iter = loader.create_batch_iterator(batch_sql, batch_size=3)
        total_batches = 0
        while batch_iter.has_next():
            batch = batch_iter.next()
            total_batches += 1

        print(f"\nTotal batches for training: {total_batches}")
        print(f"  (batch_size=3, used for classification: Premium vs Standard)")

        # =================================================================
        # Summary
        # =================================================================
        print("\n" + "=" * 70)
        print("   MULTI-FILE PIPELINE SUMMARY")
        print("=" * 70)
        print("""
  Tables used:
    - customers (8 rows): customer_id, name, segment, country
    - products (8 rows): product_id, product_name, category, price
    - orders (50 rows): order_id, customer_id, product_id, quantity, date
    - reviews (30 rows): review_id, customer_id, product_id, rating

  Join patterns demonstrated:
    1. Two-table JOIN (orders + products)
    2. Three-table JOIN (orders + customers + products)
    3. LEFT JOIN for complete customer list
    4. Subquery JOINs for aggregated stats
    5. Cross-table aggregations (country x category)

  ML feature engineering:
    - Encoded categorical variables (segment, country)
    - Aggregated numeric features (order_count, total_spent)
    - Derived features (avg_order, product_variety)

  Key SQL patterns:
    - COALESCE for null handling
    - CASE WHEN for encoding
    - Subqueries for pre-aggregation
    - GROUP BY with multiple columns
        """)

    finally:
        for name, path in files.items():
            os.unlink(path)
        print("\nCleaned up all temporary files")


if __name__ == "__main__":
    main()
