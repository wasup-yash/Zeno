"""
Quickstart example for Zeno
"""
import zeno as zn
from datetime import datetime, timedelta

# Example time series data
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
values = [10.0 + i * 0.5 for i in range(100)]

print("Zeno Quickstart Example")
print("=" * 50)

# 1. Create lag features
print("\n1️⃣  Creating lag features...")
window = zn.Window(lags=[1, 7, 14])
lag_features = window.transform(values[:20])
print(f"   Created {len(lag_features)} lag features")
print(f"   Lag-1 (first 5): {[f for f in lag_features[0][:5]]}")

# 2. Create rolling features
print("\n2️⃣  Creating rolling mean...")
rolling_mean = window.rolling_mean(values, window=7)
print(f"   Rolling mean (days 7-12): {rolling_mean[6:12]}")

# 3. Temporal validation
print("\n3️⃣  Temporal split validation...")
splitter = zn.TemporalSplitter()

train_end = datetime(2024, 3, 1)
test_start = datetime(2024, 3, 2)

train_mask, test_mask = splitter.split(dates, train_end, test_start)
print(f"   Train samples: {sum(train_mask)}")
print(f"   Test samples: {sum(test_mask)}")

# 4. Check for leakage
print("\n4️⃣  Checking for temporal leakage...")
try:
    # This should pass
    splitter.validate_feature(datetime(2024, 2, 15))
    print("   ✓ Feature from 2024-02-15 is valid (before train cutoff)")
    
    # This should fail
    splitter.validate_feature(datetime(2024, 3, 5))
except ValueError as e:
    print(f"   ✗ Leakage detected: {e}")

# 5. Pipeline composition
print("\n5️  Building a pipeline...")
pipeline = zn.Molecule([
    zn.Window(lags=[1, 7]),
    zn.Scale(method="robust")
])
print(f"   Pipeline: {pipeline}")

print("\n" + "=" * 50)
print("Zeno is ready! Check benchmarks/ for performance tests.")
