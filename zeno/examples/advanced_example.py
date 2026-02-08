"""
Phase 2: Advanced Features Example
examples/phase2_advanced_example.py
"""

import zeno as zn
from zeno.advanced import (
    ArrowWindow,
    PolarsWindow,
    AdvancedLeakageDetector,
    PolarsTemporalValidator,
    ExpandingWindowValidator,
)
import polars as pl
import pyarrow as pa
from datetime import datetime, timedelta
import numpy as np


def demo_arrow_pipeline():
    """Demonstrate zero-copy Arrow operations"""
    print("\n" + "="*60)
    print("🔷 Arrow Pipeline Demo (Zero-Copy Operations)")
    print("="*60)
    
    # Create sample data as Arrow Table
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(1000)]
    values = np.random.randn(1000).cumsum()
    
    table = pa.table({
        'timestamp': dates,
        'value': values,
    })
    
    print(f"\n📊 Original table: {table.num_rows} rows, {table.num_columns} columns")
    print(f"   Memory usage: {table.nbytes / 1024:.2f} KB")
    
    # Create Arrow pipeline
    arrow_window = ArrowWindow()
    
    # Add lag features (zero-copy)
    print("\n🔧 Adding lag features [1, 7, 30]...")
    table_with_lags = arrow_window.create_lags(table, 'value', [1, 7, 30])
    print(f"   Result: {table_with_lags.num_columns} columns")
    print(f"   Memory usage: {table_with_lags.nbytes / 1024:.2f} KB")
    print(f"   Memory increase: {(table_with_lags.nbytes - table.nbytes) / 1024:.2f} KB")
    
    # Add rolling mean
    print("\n🔧 Adding rolling mean (window=7)...")
    table_final = arrow_window.rolling_mean(table_with_lags, 'value', 7)
    print(f"   Final columns: {table_final.num_columns}")
    
    # Add EMA
    print("\n🔧 Adding EMA (alpha=0.3)...")
    table_final = arrow_window.ema(table_final, 'value', 0.3)
    print(f"   Final columns: {table_final.num_columns}")
    print(f"   Column names: {table_final.column_names}")


def demo_advanced_leakage_detection():
    """Demonstrate feature fingerprinting and leakage detection"""
    print("\n" + "="*60)
    print("🔍 Advanced Leakage Detection Demo")
    print("="*60)
    
    detector = AdvancedLeakageDetector(threshold=0.1)
    
    # Register training features
    print("\n📝 Registering training features...")
    train_dates = list(range(0, 100))
    train_values = np.random.randn(100).cumsum().tolist()
    
    detector.register_training_feature(
        train_dates,
        train_values,
        "feature_1"
    )
    print("   ✓ Registered 'feature_1' with 100 samples")
    
    # Test 1: Clean test feature (should pass)
    print("\n✅ Test 1: Clean test feature (different time period)...")
    test_dates_clean = list(range(100, 200))
    test_values_clean = np.random.randn(100).cumsum().tolist()
    
    try:
        result = detector.check_test_feature(
            test_dates_clean,
            test_values_clean,
            "test_feature_clean"
        )
        print(f"   ✓ No leakage detected: {result}")
    except ValueError as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test 2: Overlapping feature (should fail)
    print("\n❌ Test 2: Overlapping feature (same time period)...")
    test_dates_leak = list(range(50, 150))  # Overlaps with training
    test_values_leak = train_values[50:] + np.random.randn(50).tolist()
    
    try:
        result = detector.check_test_feature(
            test_dates_leak,
            test_values_leak,
            "test_feature_leak"
        )
        print(f"   ✗ Should have detected leakage!")
    except ValueError as e:
        print(f"   ✓ Leakage detected: {str(e)[:100]}...")
    
    # Test 3: Exact duplicate (should fail immediately)
    print("\n❌ Test 3: Exact duplicate...")
    try:
        result = detector.check_test_feature(
            train_dates,
            train_values,
            "test_feature_duplicate"
        )
        print(f"   ✗ Should have detected exact match!")
    except ValueError as e:
        print(f"   ✓ Exact match detected: {str(e)[:100]}...")
    
    # Get report
    print("\n📊 Leakage Report:")
    report = detector.get_report()
    for test_feature, overlaps in report.items():
        print(f"   {test_feature}:")
        for train_feature, similarity in overlaps:
            print(f"      - {train_feature}: {similarity:.2%} similar")


def demo_polars_pipeline():
    """Demonstrate Polars-native operations"""
    print("\n" + "="*60)
    print("⚡ Polars-Native Pipeline Demo (Fastest)")
    print("="*60)
    
    # Create Polars DataFrame
    df = pl.DataFrame({
        'timestamp': [datetime(2024, 1, 1) + timedelta(days=i) for i in range(1000)],
        'value': np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 1000, 1000),
    })
    
    print(f"\n📊 Original DataFrame: {df.shape}")
    print(df.head())
    
    # Create Polars pipeline
    window = PolarsWindow(lags=[1, 7, 30], rolling=[7, 30])
    
    # Transform single column
    print("\n🔧 Adding features to 'value' column...")
    df_with_features = window.transform(df, 'value')
    print(f"   Result: {df_with_features.shape}")
    print(f"   New columns: {[c for c in df_with_features.columns if c not in df.columns]}")
    
    # Transform multiple columns in parallel
    print("\n🔧 Adding features to multiple columns (parallel)...")
    df_final = window.transform_parallel(df, ['value', 'volume'])
    print(f"   Final shape: {df_final.shape}")
    print(f"   Total columns: {len(df_final.columns)}")


def demo_polars_validation():
    """Demonstrate Polars temporal validation"""
    print("\n" + "="*60)
    print("✅ Polars Temporal Validation Demo")
    print("="*60)
    
    # Create time series DataFrame
    df = pl.DataFrame({
        'timestamp': [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)],
        'value': np.random.randn(365).cumsum(),
    })
    
    validator = PolarsTemporalValidator()
    
    # Validate split
    train_end = datetime(2024, 9, 1)
    test_start = datetime(2024, 9, 2)
    
    print(f"\n📅 Validating split:")
    print(f"   Train end: {train_end.date()}")
    print(f"   Test start: {test_start.date()}")
    
    is_valid = validator.validate_split(df, 'timestamp', train_end, test_start)
    print(f"   ✓ Split is valid: {is_valid}")
    
    # Perform split
    print("\n✂️  Splitting DataFrame...")
    train, test = validator.split(df, 'timestamp', train_end)
    print(f"   Train: {len(train)} samples ({train['timestamp'].min().date()} to {train['timestamp'].max().date()})")
    print(f"   Test:  {len(test)} samples ({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")
    
    # Test invalid split (should fail)
    print("\n❌ Testing invalid split (test before train)...")
    try:
        validator.validate_split(df, 'timestamp', test_start, train_end)
        print("   ✗ Should have failed!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")


def demo_expanding_window_validation():
    """Demonstrate expanding window cross-validation"""
    print("\n" + "="*60)
    print("📈 Expanding Window Cross-Validation Demo")
    print("="*60)
    
    # Create time series DataFrame
    df = pl.DataFrame({
        'timestamp': [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)],
        'value': np.random.randn(365).cumsum(),
    })
    
    # Create expanding window validator
    validator = ExpandingWindowValidator(
        min_train_size=180,  # 6 months minimum training
        test_size=30,        # 1 month test
        step_size=30,        # Move forward 1 month each fold
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Min train size: {validator.min_train_size} days")
    print(f"   Test size: {validator.test_size} days")
    print(f"   Step size: {validator.step_size} days")
    
    # Generate splits
    print("\n📊 Generating splits...")
    splits = validator.split(df, 'timestamp')
    print(f"   Generated {len(splits)} folds")
    
    # Validate and show results
    results = validator.validate(df, 'timestamp', ['value'])
    
    print("\n📈 Fold Statistics:")
    for i, (train_size, test_size, gap) in enumerate(zip(
        results['train_sizes'],
        results['test_sizes'],
        results['gaps']
    )):
        print(f"   Fold {i+1}: Train={train_size}, Test={test_size}, Gap={gap}")


def demo_full_pipeline():
    """Demonstrate complete Phase 2 pipeline"""
    print("\n" + "="*60)
    print("🚀 Complete Phase 2 Pipeline Demo")
    print("="*60)
    
    # 1. Load data with Polars
    print("\n1️⃣  Loading data with Polars...")
    df = pl.DataFrame({
        'timestamp': [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10000)],
        'price': np.random.randn(10000).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 10000),
    })
    print(f"   ✓ Loaded {len(df)} rows")
    
    # 2. Create features with Polars pipeline
    print("\n2️⃣  Creating features...")
    window = PolarsWindow(lags=[1, 24, 168], rolling=[24, 168])
    df = window.transform_parallel(df, ['price', 'volume'])
    print(f"   ✓ Created {len(df.columns) - 3} features")
    
    # 3. Validate temporal split
    print("\n3️⃣  Validating temporal split...")
    validator = PolarsTemporalValidator()
    train_end = datetime(2024, 12, 1)
    test_start = datetime(2024, 12, 2)
    
    validator.validate_split(df, 'timestamp', train_end, test_start)
    train, test = validator.split(df, 'timestamp', train_end)
    print(f"   ✓ Train: {len(train)} samples")
    print(f"   ✓ Test:  {len(test)} samples")
    
    # 4. Check for leakage
    print("\n4️⃣  Checking for feature leakage...")
    detector = AdvancedLeakageDetector(threshold=0.05)
    
    # Register training features
    train_timestamps = train['timestamp'].to_list()
    train_prices = train['price'].to_list()
    detector.register_training_feature(
        [int(ts.timestamp()) for ts in train_timestamps[:100]],
        train_prices[:100],
        'train_price_window'
    )
    
    # Check test features
    test_timestamps = test['timestamp'].to_list()
    test_prices = test['price'].to_list()
    try:
        detector.check_test_feature(
            [int(ts.timestamp()) for ts in test_timestamps[:100]],
            test_prices[:100],
            'test_price_window'
        )
        print("   ✓ No leakage detected")
    except ValueError as e:
        print(f"   ✗ Leakage detected: {e}")
    
    # 5. Expanding window validation
    print("\n5️⃣  Expanding window cross-validation...")
    exp_validator = ExpandingWindowValidator(
        min_train_size=5000,
        test_size=1000,
        step_size=1000,
    )
    splits = exp_validator.split(df, 'timestamp')
    print(f"   ✓ Generated {len(splits)} cross-validation folds")
    
    print("\n" + "="*60)
    print("✨ Phase 2 Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    print("\n🌀 ZENO PHASE 2: ADVANCED FEATURES DEMO")
    print("━" * 60)
    
    # Run all demos
    demo_arrow_pipeline()
    demo_advanced_leakage_detection()
    demo_polars_pipeline()
    demo_polars_validation()
    demo_expanding_window_validation()
    demo_full_pipeline()
    
    print("\n✅ All Phase 2 demos completed successfully!")
    print("\n📚 Key Features Demonstrated:")
    print("   • Zero-copy Arrow pipelines")
    print("   • Feature fingerprinting & leakage detection")
    print("   • Polars-native operations (fastest path)")
    print("   • Advanced temporal validation")
    print("   • Expanding window cross-validation")
    print("\n🚀 Ready for production use!")