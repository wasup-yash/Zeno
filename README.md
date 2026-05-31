# Zeno: The Zero-Copy Time Series Engine

**High-performance, audit-ready time series library built for Foundation Models and billion-row datasets.**

## The Problem

Traditional time series libraries (sktime, Prophet, Darts) suffer from:

1. **Memory Wall**: Copying data 4-5x between CSV → Pandas → NumPy → GPU
2. **Temporal Leakage**: 80% of ML practitioners unknowingly leak future data into features
3. **Slow Windowing**: Creating lags/rolling features is the bottleneck in pipelines
4. **Heavy Dependencies**: 500MB+ install size, 400+ package dependencies


## Install and Run

```powershell
cd D:\Zeno\zeno\zeno-py
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e . maturin pytest
maturin develop --release
python ..\examples\quickstart.py
pytest ..\tests -v
```

The default engine path is Arrow/Polars-native. Plain Python lists still work
for quick experiments, but they are treated as a compatibility path because
lists must be copied across the Python/Rust boundary.

## Zero-Copy Contract

- Arrow lag features are built from a null prefix plus slices of the original
  Arrow buffers.
- Polars feature generation uses expressions instead of `to_dict`,
  `to_list`, or Python row loops.
- Temporal DataFrame splits require a sorted time column and return contiguous
  `slice` views instead of boolean-filtered copies.
- New derived feature columns allocate their own output buffers, but source
  columns are never materialized through Python lists or dictionaries.

## 🗺️ Roadmap

### Phase 1: Zero-Copy Windowing + Validation (Current)
- Fast lag/rolling feature creation
- Temporal split enforcement
- Basic leakage detection

### Phase 2: Advanced Validation + Arrow Integration
- Rolling hash comparisons for feature fingerprinting
- Full Apache Arrow zero-copy pipeline
- Polars native integration
- Advanced leakage detection algorithms

### Phase 3: Foundation Model Integration 
- HuggingFace Chronos/Lag-Llama wrappers
- GPU-accelerated inference
- Batch prediction APIs

### Phase 4: Zeno Cloud
- Serverless backtesting platform
- Managed validation pipelines
- Audit reports for compliance

---
### Temporal Validation

**The Problem:**
```python
# WRONG: This creates leakage!
df['lag_1'] = df['value'].shift(1)
train = df[df['date'] < '2024-03-01']  # lag_1 includes Feb 28!
```

**Zeno's Solution:**
```python
# RIGHT: Validation happens at feature creation time
splitter = zn.TemporalSplitter()
splitter.set_split(train_end='2024-03-01', test_start='2024-03-02')

# This will FAIL if feature uses data after train_end
window.transform(values, validator=splitter)
```

---

## 📄 License

MIT License - See LICENSE file

---

**Built with ❤️ for the time series community**
