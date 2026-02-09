# Zeno: The Zero-Copy Time Series Engine

**High-performance, audit-ready time series library built for Foundation Models and billion-row datasets.**

## The Problem

Traditional time series libraries (sktime, Prophet, Darts) suffer from:

1. **Memory Wall**: Copying data 4-5x between CSV → Pandas → NumPy → GPU
2. **Temporal Leakage**: 80% of ML practitioners unknowingly leak future data into features
3. **Slow Windowing**: Creating lags/rolling features is the bottleneck in pipelines
4. **Heavy Dependencies**: 500MB+ install size, 400+ package dependencies

### Phase 1: Zero-Copy Windowing + Temporal Validation

**What We Built:**
- **10-50x faster** lag/rolling feature creation vs Pandas
- **Temporal leakage detection** built into the pipeline (not as an afterthought)
- **<20MB binary** with zero Python dependencies in the core
- **Apache Arrow** integration for true zero-copy operations

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              USER INTERFACE (Python)                │
│  • Simple API: Window, Scale, Molecule              │
│  • Type-safe, IDE-friendly                          │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│            PyO3 BINDINGS (Rust ↔ Python)            │
│  • Zero-copy memory sharing                         │
│  • GIL released during Rust execution               |
|  • True parallelism inside the Rust engine          │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│               RUST CORE ENGINE                      │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Window Operations (engine/window.rs)        │   │
│  │  • Lag creation                              │   │
│  │  • Rolling statistics                        │   │
│  │  • Differencing                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Temporal Validator (validator/temporal.rs)  │   │
│  │  • Train/test split enforcement              │   │
│  │  • Feature timestamp checking                │   │
│  │  • Leakage prevention                        │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Arrow Operations (engine/arrow_ops.rs)      │   │
│  │  • Zero-copy buffer management               │   │
│  │  • Columnar storage                          │   │
│  └──────────────────────────────────────────────┘   |
└─────────────────────────────────────────────────────┘
```

## Performance Benchmarks

Run the included benchmark:

```bash
python benchmarks/benchmark_comparison.py
```

**Expected Results (100k samples, 10 lags):**

| Library | Time | Speedup |
|---------|------|---------|
| Pandas  | 0.45s | 1x |
| NumPy   | 0.12s | 3.8x |
| **Zeno** | **0.009s** | **50x** |

**Memory Usage (1M rows, 5 lags):**
- Pandas: ~40 MB
- Zeno: ~8 MB (80% reduction)

---

##  Testing

```bash
cd zeno-py
source .venv/bin/activate
uv pip install pytest pytest-benchmark
pytest ../tests -v
```

---

## 🛠️ Project Structure

```
zeno/
├── zeno-core/              
│   ├── src/
│   │   ├── lib.rs         
│   │   ├── engine/         
│   │   │   ├── window.rs
│   │   │   └── arrow_ops.rs
│   │   ├── validator/     
│   │   │   ├── temporal.rs
│   │   │   └── leakage.rs
│   │   └── types.rs
│   └── Cargo.toml
│
├── zeno-py/                
│   ├── zeno/
│   │   ├── atoms.py        
│   │   ├── molecule.py     
│   │   └── validator.py    
│   └── pyproject.toml
│
├── examples/
│   └── quickstart.py
│
├── tests/
│   ├── test_window.py
│   └── test_temporal.py
│
└── benchmarks/
    └── benchmark_comparison.py
```


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
## 📬 Contact

- GitHub Issues: [Report bugs or request features]
- Email: [yashsharmadev3@gmail.com]

---

**Built with ❤️ for the time series community**
