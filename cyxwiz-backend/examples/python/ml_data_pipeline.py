"""
Machine Learning Data Pipeline Example

Real-world example showing how to use DataLoader for ML workflows:
- Load and explore datasets
- Prepare training/validation/test splits
- Create batch iterators for training
- Work with multiple data formats

This example uses the Iris dataset pattern as a template.
"""

import pycyxwiz as cx
import tempfile
import os
import random

# Initialize CyxWiz
cx.initialize()


def create_iris_csv():
    """Create a synthetic Iris-like dataset for ML demonstration."""
    csv_content = """sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
"""
    fd, path = tempfile.mkstemp(suffix='_iris.csv')
    with os.fdopen(fd, 'w') as f:
        f.write(csv_content)
    return path


def main():
    print("=" * 70)
    print("   Machine Learning Data Pipeline with CyxWiz DataLoader")
    print("=" * 70)

    if not cx.DataLoader.is_available():
        print("\nDuckDB not available. Build with CYXWIZ_HAS_DUCKDB=ON")
        return

    print(f"\nUsing DuckDB {cx.DataLoader.get_version()}")

    # Create test dataset
    data_path = create_iris_csv()
    print(f"Created Iris dataset: {data_path}")

    try:
        loader = cx.DataLoader()

        # =================================================================
        # Step 1: Explore the Dataset
        # =================================================================
        print("\n" + "=" * 70)
        print("   STEP 1: Exploratory Data Analysis")
        print("=" * 70)

        # Get schema information
        schema = loader.get_schema(data_path)
        print("\nDataset Schema:")
        print("-" * 50)
        for col in schema:
            print(f"  {col.name:20s} | {col.type:15s} | nullable={col.nullable}")

        # Get basic statistics
        row_count = loader.get_row_count(data_path)
        print(f"\nTotal samples: {row_count}")

        # Class distribution
        print("\nClass Distribution (SQL aggregation):")
        class_dist = loader.query(f"""
            SELECT COUNT(*) as count
            FROM '{data_path}'
            GROUP BY species
        """)
        print(f"  Query returned shape: {class_dist.shape()}")

        # Summary statistics for features
        print("\nFeature Statistics:")
        stats = loader.query(f"""
            SELECT
                AVG(sepal_length) as avg_sepal_l,
                AVG(sepal_width) as avg_sepal_w,
                AVG(petal_length) as avg_petal_l,
                AVG(petal_width) as avg_petal_w,
                MIN(sepal_length) as min_sepal_l,
                MAX(sepal_length) as max_sepal_l
            FROM '{data_path}'
        """)
        print(f"  Statistics tensor shape: {stats.shape()}")

        # =================================================================
        # Step 2: Prepare Features and Labels
        # =================================================================
        print("\n" + "=" * 70)
        print("   STEP 2: Feature Extraction")
        print("=" * 70)

        # Load only numeric features (4 columns)
        features = loader.load_csv(
            data_path,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
        print(f"\nFeature matrix shape: {features.shape()}")
        print(f"  Expected: (30, 4) = 30 samples x 4 features")

        # For labels, we'd need to encode species to numeric
        # Using SQL CASE expression
        labels_query = f"""
            SELECT
                CASE species
                    WHEN 'setosa' THEN 0
                    WHEN 'versicolor' THEN 1
                    WHEN 'virginica' THEN 2
                END as label
            FROM '{data_path}'
        """
        labels = loader.query(labels_query)
        print(f"Labels shape: {labels.shape()}")

        # =================================================================
        # Step 3: Create Train/Validation/Test Splits
        # =================================================================
        print("\n" + "=" * 70)
        print("   STEP 3: Data Splitting (80/10/10)")
        print("=" * 70)

        # Use SQL with row numbers for splitting
        # Training set (first 80%)
        train_data = loader.query(f"""
            SELECT sepal_length, sepal_width, petal_length, petal_width,
                   CASE species
                       WHEN 'setosa' THEN 0
                       WHEN 'versicolor' THEN 1
                       WHEN 'virginica' THEN 2
                   END as label
            FROM '{data_path}'
            LIMIT 24
        """)
        print(f"\nTraining set: {train_data.shape()}")

        # Validation set (next 10%)
        val_data = loader.query(f"""
            SELECT sepal_length, sepal_width, petal_length, petal_width,
                   CASE species
                       WHEN 'setosa' THEN 0
                       WHEN 'versicolor' THEN 1
                       WHEN 'virginica' THEN 2
                   END as label
            FROM '{data_path}'
            LIMIT 3 OFFSET 24
        """)
        print(f"Validation set: {val_data.shape()}")

        # Test set (final 10%)
        test_data = loader.query(f"""
            SELECT sepal_length, sepal_width, petal_length, petal_width,
                   CASE species
                       WHEN 'setosa' THEN 0
                       WHEN 'versicolor' THEN 1
                       WHEN 'virginica' THEN 2
                   END as label
            FROM '{data_path}'
            LIMIT 3 OFFSET 27
        """)
        print(f"Test set: {test_data.shape()}")

        # =================================================================
        # Step 4: Batch Iterator for Training
        # =================================================================
        print("\n" + "=" * 70)
        print("   STEP 4: Batch Training Simulation")
        print("=" * 70)

        batch_size = 8
        epochs = 3

        train_sql = f"""
            SELECT sepal_length, sepal_width, petal_length, petal_width,
                   CASE species
                       WHEN 'setosa' THEN 0
                       WHEN 'versicolor' THEN 1
                       WHEN 'virginica' THEN 2
                   END as label
            FROM '{data_path}'
            LIMIT 24
        """

        print(f"\nBatch size: {batch_size}")
        print(f"Training epochs: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            iterator = loader.create_batch_iterator(train_sql, batch_size)
            batch_num = 0
            epoch_samples = 0

            while iterator.has_next():
                batch = iterator.next()
                batch_samples = batch.shape()[0]
                epoch_samples += batch_samples

                # Simulate training step
                # In real training: forward pass, loss, backward pass
                fake_loss = 1.0 / (epoch + 1) + random.random() * 0.1

                batch_num += 1

            print(f"  Epoch {epoch + 1}/{epochs}: {batch_num} batches, "
                  f"{epoch_samples} samples, loss={fake_loss:.4f}")

        # =================================================================
        # Step 5: Feature Normalization via SQL
        # =================================================================
        print("\n" + "=" * 70)
        print("   STEP 5: Feature Normalization")
        print("=" * 70)

        # Z-score normalization using SQL subqueries
        normalized_query = f"""
            WITH stats AS (
                SELECT
                    AVG(sepal_length) as mean_sl, STDDEV(sepal_length) as std_sl,
                    AVG(sepal_width) as mean_sw, STDDEV(sepal_width) as std_sw,
                    AVG(petal_length) as mean_pl, STDDEV(petal_length) as std_pl,
                    AVG(petal_width) as mean_pw, STDDEV(petal_width) as std_pw
                FROM '{data_path}'
            )
            SELECT
                (sepal_length - stats.mean_sl) / stats.std_sl as norm_sl,
                (sepal_width - stats.mean_sw) / stats.std_sw as norm_sw,
                (petal_length - stats.mean_pl) / stats.std_pl as norm_pl,
                (petal_width - stats.mean_pw) / stats.std_pw as norm_pw
            FROM '{data_path}', stats
            LIMIT 5
        """

        try:
            normalized = loader.query(normalized_query)
            print(f"\nNormalized features (first 5): {normalized.shape()}")
        except Exception as e:
            print(f"\nAdvanced SQL may not be supported: {e}")
            print("  (This is expected - shows SQL capability boundaries)")

        # =================================================================
        # Summary
        # =================================================================
        print("\n" + "=" * 70)
        print("   PIPELINE SUMMARY")
        print("=" * 70)
        print(f"""
  Dataset: {row_count} samples, {len(schema)} columns
  Features: 4 numeric (sepal/petal length/width)
  Classes: 3 (setosa, versicolor, virginica)

  Training set: 24 samples (80%)
  Validation set: 3 samples (10%)
  Test set: 3 samples (10%)

  Batch training: {epochs} epochs, batch_size={batch_size}

  Key APIs used:
    - loader.get_schema(path)      # Inspect columns
    - loader.get_row_count(path)   # Count rows
    - loader.load_csv(path, cols)  # Load specific columns
    - loader.query(sql)            # Execute SQL
    - loader.create_batch_iterator(sql, size)  # Mini-batches
        """)

    finally:
        os.unlink(data_path)
        print(f"\nCleaned up: {data_path}")


if __name__ == "__main__":
    main()
