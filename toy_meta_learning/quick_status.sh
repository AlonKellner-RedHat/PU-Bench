#!/bin/bash
echo "Extended Training Status"
echo "========================"
if ps aux | grep -v grep | grep "train_extended_pool.py" > /dev/null; then
    echo "Status: RUNNING"
    CPU=$(ps aux | grep -v grep | grep "train_extended_pool.py" | awk 'NR==1{print $10}')
    echo "CPU Time: $CPU"
    
    # Show progress
    LAST_TASK=$(grep -oE "\[[0-9]+/80\]" extended_training.log 2>/dev/null | tail -1)
    if [ -n "$LAST_TASK" ]; then
        TASK_NUM=$(echo $LAST_TASK | cut -d'/' -f1 | tr -d '[')
        TASK_PCT=$((TASK_NUM * 100 / 80))
        echo "Checkpoint Creation: $LAST_TASK ($TASK_PCT%)"
    fi
    
    # Check if meta-training started
    if grep -q "Starting extended PU meta-training" extended_training.log 2>/dev/null; then
        echo "Phase: META-TRAINING"
        LAST_ITER=$(grep -oE "Iteration [0-9]+/500" extended_training.log 2>/dev/null | tail -1)
        if [ -n "$LAST_ITER" ]; then
            echo "Progress: $LAST_ITER"
        fi
    else
        echo "Phase: Creating checkpoints..."
    fi
else
    echo "Status: COMPLETED or STOPPED"
fi
