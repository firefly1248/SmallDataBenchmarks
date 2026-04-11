# Thin re-export shim — kept for backward compatibility with any existing scripts.
# New code should import directly from benchmark.data.
from benchmark.data import load_data, load_data_df  # noqa: F401
