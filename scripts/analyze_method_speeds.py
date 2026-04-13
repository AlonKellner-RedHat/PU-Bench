#!/usr/bin/env python3
"""Analyze epoch speeds from worker logs to identify slow methods."""

import re
from pathlib import Path
from collections import defaultdict
import statistics

def extract_epoch_times(log_file):
    """Extract epoch times from progress bars like '12/40 [01:31<01:22, 4.60s/it]'"""
    method_times = defaultdict(list)
    
    with open(log_file) as f:
        content = f.read()
    
    # Pattern: (METHOD): 45%|████▌     | 18/40 [03:14<03:21, 5.76s/it]
    # Captures: method name, current epoch, total epochs, elapsed time, remaining time, seconds per iteration
    pattern = r'\((.*?)\):\s+\d+%.*?\|\s+(\d+)/(\d+)\s+\[[\d:]+<[\d:]+,\s+([\d.]+)s/it\]'
    
    for match in re.finditer(pattern, content):
        method = match.group(1)
        current = int(match.group(2))
        total = int(match.group(3))
        sec_per_epoch = float(match.group(4))
        
        # Only include if we're past the first few epochs (initialization overhead)
        if current > 3:
            method_times[method].append(sec_per_epoch)
    
    return method_times

def main():
    logs_dir = Path("logs/phase1_extended")
    all_method_times = defaultdict(list)
    
    # Process all worker logs
    for log_file in sorted(logs_dir.glob("worker_*.log")):
        method_times = extract_epoch_times(log_file)
        for method, times in method_times.items():
            all_method_times[method].extend(times)
    
    # Calculate statistics
    method_stats = {}
    for method, times in all_method_times.items():
        if times:
            method_stats[method] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'min': min(times),
                'max': max(times),
                'samples': len(times),
            }
    
    # Sort by mean time
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print("=" * 80)
    print("Method Epoch Speeds (seconds per epoch)")
    print("=" * 80)
    print(f"{'Method':<30} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Samples':<8}")
    print("-" * 80)
    
    for method, stats in sorted_methods:
        print(f"{method:<30} {stats['mean']:>7.2f}  {stats['median']:>7.2f}  "
              f"{stats['min']:>7.2f}  {stats['max']:>7.2f}  {stats['samples']:>7}")
    
    print("=" * 80)
    
    # Identify slow outliers (> 20s/epoch)
    slow_methods = [(m, s) for m, s in sorted_methods if s['mean'] > 20.0]
    if slow_methods:
        print("\n🐌 SLOW METHODS (>20s/epoch):")
        for method, stats in slow_methods:
            total_epochs = 40  # Typical for most methods
            if method == 'CGENPU':
                total_epochs = 200
            elif method == 'PAN':
                total_epochs = 120
            elif method == 'PULCPBF':
                total_epochs = 20
            
            total_time = stats['mean'] * total_epochs
            print(f"  {method:<30} {stats['mean']:>6.2f}s/epoch × {total_epochs} epochs = "
                  f"{total_time/60:>6.1f} min/experiment")
    
    # Identify fast methods (< 5s/epoch)
    fast_methods = [(m, s) for m, s in sorted_methods if s['mean'] < 5.0]
    if fast_methods:
        print("\n⚡ FAST METHODS (<5s/epoch):")
        for method, stats in fast_methods:
            print(f"  {method:<30} {stats['mean']:>6.2f}s/epoch")

if __name__ == "__main__":
    main()
