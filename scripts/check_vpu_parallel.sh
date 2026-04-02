#!/bin/bash
#
# Check VPU Parallel Status
#
# Shows current status of parallel VPU training
#

PIDFILE="/tmp/vpu_parallel_pids.txt"
STATUSFILE="logs/vpu_rerun/parallel/status.txt"

echo "========================================="
echo "VPU Parallel Status Check"
echo "========================================="
echo ""

# Check for running processes
RUNNING_COUNT=0
DEAD_COUNT=0

if [ -f "$PIDFILE" ]; then
    echo "Tracked processes (from PID file):"
    echo ""

    while IFS= read -r pid; do
        if ps -p "$pid" > /dev/null 2>&1; then
            # Get process info
            PROC_INFO=$(ps -p "$pid" -o pid,etime,pcpu,pmem,command | tail -1)
            echo "  ✅ Running: $PROC_INFO"
            RUNNING_COUNT=$((RUNNING_COUNT + 1))
        else
            echo "  ❌ Dead:    PID $pid (no longer running)"
            DEAD_COUNT=$((DEAD_COUNT + 1))
        fi
    done < "$PIDFILE"

    echo ""
    echo "Summary: $RUNNING_COUNT running, $DEAD_COUNT dead"
else
    echo "⚠️  No PID file found at: $PIDFILE"
    echo "   (Parallel run may not be active)"
fi

echo ""

# Check for any run_train.py processes
echo "All python training processes:"
echo ""

PYTHON_PROCS=$(ps aux | grep "[r]un_train.py" | grep -v grep)
if [ -z "$PYTHON_PROCS" ]; then
    echo "  (none running)"
else
    echo "$PYTHON_PROCS" | awk '{printf "  PID %-6s | CPU %4s%% | Mem %4s%% | Runtime %s\n", $2, $3, $4, $10}'
fi

echo ""

# Show completed experiments
COMPLETED=$(find results/ -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
echo "Experiments completed: $COMPLETED / 1,260 ($(echo "scale=1; $COMPLETED * 100 / 1260" | bc)%)"

echo ""

# Show recent status if available
if [ -f "$STATUSFILE" ]; then
    echo "Recent activity (last 10 lines from status log):"
    echo ""
    tail -10 "$STATUSFILE" | sed 's/^/  /'
    echo ""
fi

# Show recent completions
echo "Most recent completions (by file time):"
echo ""
find results/ -name "*.json" -type f -exec ls -lt {} + 2>/dev/null | head -6 | tail -5 | \
    awk '{print "  " $8 " " $9 " - " $10}' || echo "  (none found)"

echo ""
echo "========================================="
echo ""
echo "Commands:"
echo "  Kill all:  bash scripts/kill_vpu_parallel.sh"
echo "  Watch:     watch -n 10 'bash scripts/check_vpu_parallel.sh'"
echo ""
