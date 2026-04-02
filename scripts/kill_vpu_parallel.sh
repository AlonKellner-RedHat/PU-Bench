#!/bin/bash
#
# Kill VPU Parallel Workers
#
# Safely stops all parallel VPU training processes
#

PIDFILE="/tmp/vpu_parallel_pids.txt"

echo "========================================="
echo "Stopping VPU Parallel Workers"
echo "========================================="
echo ""

# Check for PID file
if [ ! -f "$PIDFILE" ]; then
    echo "⚠️  No PID file found at: $PIDFILE"
    echo ""
    echo "Checking for any running python processes..."
    echo ""

    # Look for any running run_train.py processes
    RUNNING_PROCS=$(ps aux | grep "[r]un_train.py" | grep -v grep)

    if [ -z "$RUNNING_PROCS" ]; then
        echo "✅ No VPU training processes found."
    else
        echo "Found running processes:"
        echo "$RUNNING_PROCS"
        echo ""
        read -p "Kill these processes? (y/n): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ps aux | grep "[r]un_train.py" | awk '{print $2}' | xargs kill 2>/dev/null || true
            sleep 2
            # Force kill if needed
            ps aux | grep "[r]un_train.py" | awk '{print $2}' | xargs kill -9 2>/dev/null || true
            echo "✅ Processes killed."
        else
            echo "Aborted."
        fi
    fi
    exit 0
fi

# Kill processes from PID file
echo "Found PID file. Killing tracked processes..."
echo ""

killed_count=0
already_dead=0
force_killed=0

while IFS= read -r pid; do
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "  Killing PID $pid..."
        kill "$pid" 2>/dev/null && killed_count=$((killed_count + 1)) || true
    else
        already_dead=$((already_dead + 1))
    fi
done < "$PIDFILE"

# Wait for graceful shutdown
if [ $killed_count -gt 0 ]; then
    echo ""
    echo "Waiting for graceful shutdown (5 seconds)..."
    sleep 5

    # Force kill any remaining processes
    echo ""
    echo "Checking for remaining processes..."
    while IFS= read -r pid; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "  Force killing PID $pid..."
            kill -9 "$pid" 2>/dev/null && force_killed=$((force_killed + 1)) || true
        fi
    done < "$PIDFILE"
fi

# Clean up PID file
rm -f "$PIDFILE"

echo ""
echo "========================================="
echo "Summary:"
echo "  Gracefully killed: $killed_count"
echo "  Already stopped:   $already_dead"
echo "  Force killed:      $force_killed"
echo "========================================="
echo ""

# Check for any remaining python processes
REMAINING=$(ps aux | grep "[r]un_train.py" | grep -v grep | wc -l | tr -d ' ')
if [ "$REMAINING" != "0" ]; then
    echo "⚠️  Warning: $REMAINING python processes still running"
    echo "   Run this script again or use: pkill -9 -f run_train.py"
else
    echo "✅ All processes stopped successfully."
fi
echo ""
