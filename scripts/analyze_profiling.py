#!/usr/bin/env python3
"""Analyze cProfile output to validate performance hypotheses."""

import pstats
from pstats import SortKey
import sys

def analyze_profile(prof_file, output_file=None):
    """Analyze cProfile output and validate hypotheses."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {prof_file}")
    print(f"{'='*80}\n")
    
    stats = pstats.Stats(prof_file)
    
    # === TOP FUNCTIONS BY CUMULATIVE TIME ===
    print("\n=== TOP 30 FUNCTIONS BY CUMULATIVE TIME ===\n")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    
    # === HYPOTHESIS 1: Validation overhead ===
    print("\n" + "="*80)
    print("HYPOTHESIS 1: Validation every epoch consumes 50-70% of time")
    print("="*80 + "\n")
    
    stats_eval = stats.stats
    total_time = stats.total_tt
    
    # Look for evaluate_metrics function
    eval_time = 0
    for func_key, func_stats in stats_eval.items():
        filename, line, func_name = func_key
        if 'evaluate_metrics' in func_name or 'evaluate_model' in func_name:
            cumtime = func_stats[3]  # cumulative time
            eval_time += cumtime
            print(f"  {func_name}: {cumtime:.2f}s ({cumtime/total_time*100:.1f}%)")
    
    print(f"\nTotal evaluation time: {eval_time:.2f}s ({eval_time/total_time*100:.1f}% of total)")
    if eval_time / total_time > 0.5:
        print("✅ CONFIRMED: Evaluation is a major bottleneck (>50%)")
    elif eval_time / total_time > 0.3:
        print("⚠️  PARTIAL: Evaluation is significant (30-50%)")
    else:
        print("❌ NOT CONFIRMED: Evaluation is not the main bottleneck (<30%)")
    
    # === HYPOTHESIS 2: Expensive metrics ===
    print("\n" + "="*80)
    print("HYPOTHESIS 2: Calibration/PR curves add 20-30% overhead")
    print("="*80 + "\n")
    
    expensive_metrics_time = 0
    for func_key, func_stats in stats_eval.items():
        filename, line, func_name = func_key
        if any(x in func_name for x in ['calibration', 'precision_recall_curve', 'isotonic', 'roc_auc']):
            cumtime = func_stats[3]
            expensive_metrics_time += cumtime
            print(f"  {func_name}: {cumtime:.2f}s ({cumtime/eval_time*100 if eval_time > 0 else 0:.1f}% of eval)")
    
    if eval_time > 0:
        print(f"\nExpensive metrics time: {expensive_metrics_time:.2f}s ({expensive_metrics_time/eval_time*100:.1f}% of evaluation)")
        if expensive_metrics_time / eval_time > 0.2:
            print("✅ CONFIRMED: Expensive metrics are significant (>20% of evaluation)")
        else:
            print("❌ NOT CONFIRMED: Expensive metrics are minor (<20% of evaluation)")
    
    # === HYPOTHESIS 3: DataLoader blocking ===
    print("\n" + "="*80)
    print("HYPOTHESIS 3: DataLoader blocking causes 10-20% slowdown")
    print("="*80 + "\n")
    
    dataloader_time = 0
    for func_key, func_stats in stats_eval.items():
        filename, line, func_name = func_key
        if 'DataLoader' in func_name or '__next__' in func_name and 'loader' in filename.lower():
            cumtime = func_stats[3]
            dataloader_time += cumtime
            print(f"  {func_name}: {cumtime:.2f}s ({cumtime/total_time*100:.1f}%)")
    
    print(f"\nDataLoader time: {dataloader_time:.2f}s ({dataloader_time/total_time*100:.1f}% of total)")
    if dataloader_time / total_time > 0.1:
        print("✅ CONFIRMED: DataLoader overhead is significant (>10%)")
    else:
        print("❌ NOT CONFIRMED: DataLoader overhead is minor (<10%)")
    
    # === HYPOTHESIS 4: Image loading (AlzheimerMRI) ===
    print("\n" + "="*80)
    print("HYPOTHESIS 4: AlzheimerMRI image loading is slow")
    print("="*80 + "\n")
    
    image_load_time = 0
    for func_key, func_stats in stats_eval.items():
        filename, line, func_name = func_key
        if 'load_images' in func_name or 'PIL' in filename or 'Image' in func_name:
            cumtime = func_stats[3]
            image_load_time += cumtime
            if cumtime > 1.0:  # Only show significant functions
                print(f"  {func_name}: {cumtime:.2f}s ({cumtime/total_time*100:.1f}%)")
    
    print(f"\nImage loading time: {image_load_time:.2f}s ({image_load_time/total_time*100:.1f}% of total)")
    if image_load_time / total_time > 0.1:
        print("✅ CONFIRMED: Image loading is significant (>10%)")
    else:
        print("❌ NOT CONFIRMED: Image loading is minor (<10%)")
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("OPTIMIZATION PRIORITY SUMMARY")
    print("="*80 + "\n")
    
    priorities = []
    
    if eval_time / total_time > 0.5:
        priorities.append(("HIGH", "Reduce validation frequency (H1)", f"{eval_time/total_time*100:.1f}% of time"))
    
    if expensive_metrics_time / eval_time > 0.2 and eval_time > 0:
        priorities.append(("MEDIUM", "Defer expensive metrics (H2)", f"{expensive_metrics_time/eval_time*100:.1f}% of eval"))
    
    if dataloader_time / total_time > 0.1:
        priorities.append(("MEDIUM", "Enable parallel DataLoader (H3)", f"{dataloader_time/total_time*100:.1f}% of time"))
    
    if image_load_time / total_time > 0.1:
        priorities.append(("HIGH", "Optimize image loading (H4)", f"{image_load_time/total_time*100:.1f}% of time"))
    
    if priorities:
        for priority, action, impact in sorted(priorities, key=lambda x: x[0]):
            print(f"  [{priority}] {action}: {impact}")
    else:
        print("  No major bottlenecks identified. Performance may be acceptable.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_profiling.py <profile_file>")
        print("Example: python analyze_profiling.py profiling_output/cifar10.prof")
        sys.exit(1)
    
    analyze_profile(sys.argv[1])
