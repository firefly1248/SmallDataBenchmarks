"""
Run all benchmark scripts sequentially.

Order:
  1. compare_baseline_models.py  — required by all others
  2. optuna_models.py
  3. benchmark_autogluon.py
  4. benchmark_mljar.py
"""

import subprocess
import sys
import time
from pathlib import Path

from config import AUTOML_SEC

PYTHON = sys.executable
RESULTS = Path("results")

# Maps each script to the output file it produces (used for skip logic).
SCRIPT_OUTPUTS = {
    "compare_baseline_models.py": RESULTS / "compare_baseline_models.joblib",
    "optuna_models.py":           RESULTS / "optuna_models.joblib",
    "benchmark_autogluon.py":     RESULTS / f"autogluon_sec_{AUTOML_SEC}.joblib",
    "benchmark_mljar.py":         RESULTS / f"mljar_sec_{AUTOML_SEC}.joblib",
}

SCRIPTS = list(SCRIPT_OUTPUTS.keys())

# optuna_models.py keeps its own (dataset, model) checkpoint and skips finished
# pairs itself, so an existing aggregate output must not stop it — otherwise a
# newly added model silently never runs while the script reports success.
SELF_RESUMING = {"optuna_models.py"}


def run(script: str) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print(f"  Starting: {script}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run([PYTHON, script])
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  {script} finished in {elapsed/3600:.2f}h — {status}")
    return result


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    total_start = time.time()

    for script in SCRIPTS:
        output = SCRIPT_OUTPUTS[script]
        if script not in SELF_RESUMING and output.exists():
            print(f"\n  Skipping {script} — output already exists: {output}")
            continue
        result = run(script)
        if result.returncode != 0:
            print(f"\n{script} failed — aborting.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All done in {total/3600:.2f}h")
    print(f"  Results saved in: {RESULTS}/")
