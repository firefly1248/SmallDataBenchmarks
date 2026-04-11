#!/bin/zsh
# Run MLJAR then AutoGluon sequentially.
# Safe to re-run: both benchmarks resume from checkpoint.
# Uses a pidfile to prevent duplicate instances.

PIDFILE=/tmp/smalldata_automl.pid
cd /Users/ilia.ekhlakov/SmallDataBenchmarks

# Prevent duplicate runs
if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$(date): already running (PID $OLD_PID), exiting."
        exit 1
    fi
fi
echo $$ > "$PIDFILE"

# Wait a bit for the system to settle after boot
sleep 30

echo "$(date): Starting MLJAR..." >> results/mljar_sec_1000_run.log
/Users/ilia.ekhlakov/SmallDataBenchmarks/.venv/bin/python benchmark_mljar.py >> results/mljar_sec_1000_run.log 2>&1
echo "$(date): MLJAR done." >> results/mljar_sec_1000_run.log

echo "$(date): Starting AutoGluon..." >> results/autogluon_sec_1000_run.log
/Users/ilia.ekhlakov/SmallDataBenchmarks/.venv/bin/python benchmark_autogluon.py >> results/autogluon_sec_1000_run.log 2>&1
echo "$(date): AutoGluon done." >> results/autogluon_sec_1000_run.log

rm -f "$PIDFILE"
