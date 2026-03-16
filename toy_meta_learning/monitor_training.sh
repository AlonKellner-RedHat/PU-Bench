#!/bin/bash
# Live monitoring of extended training

LOG_FILE="extended_training.log"

clear
echo "=============================================="
echo "     Extended Training Live Monitor"
echo "=============================================="
echo ""

while true; do
    # Move cursor to home position
    tput cup 4 0

    # Check if process is running
    if ps aux | grep -v grep | grep "train_extended_pool.py" > /dev/null; then
        echo "✓ Status: RUNNING                          "
        CPU_TIME=$(ps aux | grep -v grep | grep "train_extended_pool.py" | awk 'NR==1{print $10}')
        MEM=$(ps aux | grep -v grep | grep "train_extended_pool.py" | awk 'NR==1{print $4}')
        echo "  CPU time: $CPU_TIME                      "
        echo "  Memory: $MEM%                            "
    else
        echo "✗ Status: NOT RUNNING                      "
        echo "                                             "
        echo "Training has completed or stopped.          "
        break
    fi
    echo ""

    # Extract progress information
    if [ -f "$LOG_FILE" ]; then
        # Checkpoint creation progress
        if grep -q "CREATING CHECKPOINT POOL" "$LOG_FILE"; then
            CURRENT_TASK=$(grep -oE "\[[0-9]+/80\]" "$LOG_FILE" | tail -1 | tr -d '[]')
            if [ -n "$CURRENT_TASK" ]; then
                TASK_NUM=$(echo $CURRENT_TASK | cut -d'/' -f1)
                TASK_TOTAL=$(echo $CURRENT_TASK | cut -d'/' -f2)
                TASK_PCT=$((TASK_NUM * 100 / TASK_TOTAL))
                echo "Checkpoint Creation Progress:               "
                echo "  Task: $CURRENT_TASK ($TASK_PCT%)          "

                # Show current task details
                CURRENT_TASK_LINE=$(grep "\[$CURRENT_TASK\]" "$LOG_FILE" | tail -1)
                if [ -n "$CURRENT_TASK_LINE" ]; then
                    echo "  $CURRENT_TASK_LINE                        "
                fi
            fi
        fi
        echo ""

        # Meta-training progress
        if grep -q "Starting extended PU meta-training" "$LOG_FILE"; then
            echo "✓ Checkpoint Pool Complete!                 "
            LAST_ITER=$(grep -oE "Iteration [0-9]+/[0-9]+" "$LOG_FILE" | tail -1)
            if [ -n "$LAST_ITER" ]; then
                ITER_NUM=$(echo $LAST_ITER | awk '{print $2}' | cut -d'/' -f1)
                ITER_TOTAL=$(echo $LAST_ITER | awk '{print $2}' | cut -d'/' -f2)
                ITER_PCT=$((ITER_NUM * 100 / ITER_TOTAL))
                echo "Meta-Training Progress:                     "
                echo "  $LAST_ITER ($ITER_PCT%)                   "

                # Show recent performance
                META_LOSS=$(grep "Meta-loss:" "$LOG_FILE" | tail -1 | awk '{print $2}')
                ORACLE_BCE=$(grep "Oracle checkpoints BCE:" "$LOG_FILE" | tail -1 | awk '{print $4}')
                NAIVE_BCE=$(grep "Naive checkpoints BCE:" "$LOG_FILE" | tail -1 | awk '{print $4}')

                if [ -n "$META_LOSS" ]; then
                    echo "  Meta-loss: $META_LOSS                      "
                fi
                if [ -n "$ORACLE_BCE" ]; then
                    echo "  Oracle BCE: $ORACLE_BCE                    "
                fi
                if [ -n "$NAIVE_BCE" ]; then
                    echo "  Naive BCE: $NAIVE_BCE                      "
                fi
            fi
        fi
        echo ""

        echo "Last 10 lines:                              "
        echo "----------------------------------------------"
        tail -10 "$LOG_FILE" | cut -c 1-80
        echo "----------------------------------------------"
    else
        echo "Log file not found: $LOG_FILE               "
    fi

    echo ""
    echo "Press Ctrl+C to exit monitor                 "

    # Update every 2 seconds
    sleep 2
done

echo ""
echo "Training has finished. Check $LOG_FILE for full results."
