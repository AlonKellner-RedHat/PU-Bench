#!/bin/bash
# Monitor Phase 2 (CIFAR10) progress

echo "=== Phase 2 Monitor (CIFAR10 Only) ==="
echo ""

running_workers=0
for worker_id in 0 1; do
    if [ -f "logs/comprehensive/phase2_worker_${worker_id}.pid" ]; then
        pid=$(cat "logs/comprehensive/phase2_worker_${worker_id}.pid")
        if ps -p $pid > /dev/null 2>&1; then
            echo "✓ Worker $worker_id running (PID: $pid)"
            running_workers=$((running_workers + 1))
        else
            echo "✗ Worker $worker_id stopped"
        fi
    fi
done

echo ""
echo "Workers running: $running_workers/2"

echo ""
echo "=== Progress per Worker ==="
total_completed=0
total_failed=0

for i in 0 1; do
    if [ -f "logs/comprehensive/phase2_worker_$i.log" ]; then
        completed=$(grep -c "✔ DONE" logs/comprehensive/phase2_worker_$i.log 2>/dev/null || echo "0")
        failed=$(grep -c "✗ FAILED" logs/comprehensive/phase2_worker_$i.log 2>/dev/null || echo "0")
        total_assigned=$(grep "Worker $i/" logs/comprehensive/phase2_worker_$i.log 2>/dev/null | head -n 1 | grep -oE "[0-9]+ experiments" | grep -oE "[0-9]+" || echo "0")

        echo "Worker $i: $completed completed, $failed failed (out of $total_assigned)"
        total_completed=$((total_completed + completed))
        total_failed=$((total_failed + failed))
    fi
done

echo ""
cifar_count=$(find results_comprehensive -name "*CIFAR10*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "CIFAR10 results: $cifar_count"
echo "Phase 2: $total_completed completed, $total_failed failed"

echo ""
echo "=== Recent Activity ==="
for i in 0 1; do
    if [ -f "logs/comprehensive/phase2_worker_$i.log" ]; then
        last=$(tail -n 50 logs/comprehensive/phase2_worker_$i.log 2>/dev/null | grep -E "(▶ RUN|✔ DONE)" | tail -n 1)
        [ -n "$last" ] && echo "Worker $i: $last"
    fi
done
