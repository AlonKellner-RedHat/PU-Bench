#!/bin/bash
# Monitor progress of extended training

OUTPUT_FILE="/private/tmp/claude-501/-Users-akellner-MyDir-Code-Other-PU-Bench/tasks/b3iikml8l.output"

echo "==============================================="
echo "Extended Training Progress Monitor"
echo "==============================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "train_extended_pool.py" > /dev/null; then
    echo "✓ Training process is RUNNING"
    CPU_TIME=$(ps aux | grep -v grep | grep "train_extended_pool.py" | awk '{print $10}')
    MEM=$(ps aux | grep -v grep | grep "train_extended_pool.py" | awk '{print $4}')
    echo "  CPU time: $CPU_TIME"
    echo "  Memory: $MEM%"
else
    echo "✗ Training process is NOT running"
fi
echo ""

# Check output file
if [ -f "$OUTPUT_FILE" ]; then
    echo "Last 40 lines of output:"
    echo "-----------------------------------------------"
    tail -40 "$OUTPUT_FILE"
    echo "-----------------------------------------------"
    echo ""

    # Try to extract progress info
    echo "Progress Summary:"
    if grep -q "CREATING CHECKPOINT POOL" "$OUTPUT_FILE"; then
        TOTAL=$(grep "Total checkpoints:" "$OUTPUT_FILE" | tail -1 | awk '{print $3}')
        CREATED=$(grep -c "Created.*checkpoints at epochs" "$OUTPUT_FILE")
        echo "  Checkpoints created: $CREATED / $TOTAL (estimated)"
    fi

    if grep -q "Starting extended PU meta-training" "$OUTPUT_FILE"; then
        echo "  ✓ Checkpoint creation COMPLETE"
        LAST_ITER=$(grep "Iteration [0-9]*/[0-9]*" "$OUTPUT_FILE" | tail -1)
        if [ -n "$LAST_ITER" ]; then
            echo "  Meta-training: $LAST_ITER"
        fi
    fi
else
    echo "Output file not found: $OUTPUT_FILE"
fi

echo ""
echo "To watch live: tail -f $OUTPUT_FILE"
