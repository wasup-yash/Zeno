import polars as pl
from zeno.advanced import PolarsWindow

# Test the Polars integration
df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
pw = PolarsWindow(lags=[1, 2])
result = pw.transform(df, "val")
print(result)