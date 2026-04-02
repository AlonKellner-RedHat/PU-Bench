# VPU Parallel Execution Guide

## Overview

The parallel execution scripts allow running 2-3 experiments simultaneously to utilize available CPU/GPU resources more efficiently.

**Expected speedup:** 2-3x faster than sequential execution

## Scripts

### 1. `run_vpu_parallel.sh` - Start parallel execution

Runs experiments in parallel with proper process tracking and cleanup.

**Features:**
- Tracks all child process PIDs in `/tmp/vpu_parallel_pids.txt`
- Handles Ctrl+C gracefully (cleans up all workers)
- Logs each dataset separately in `logs/vpu_rerun/parallel/`
- Status tracking in `logs/vpu_rerun/parallel/status.txt`
- Auto-resumes from completed experiments (`--resume` flag)

**Usage:**
```bash
bash scripts/run_vpu_parallel.sh
```

**To stop:**
- Press `Ctrl+C` (graceful cleanup)
- Or run: `bash scripts/kill_vpu_parallel.sh`

### 2. `kill_vpu_parallel.sh` - Stop all workers

Safely stops all parallel training processes.

**Usage:**
```bash
bash scripts/kill_vpu_parallel.sh
```

**What it does:**
1. Reads PIDs from tracking file
2. Sends graceful SIGTERM
3. Waits 5 seconds
4. Force kills (SIGKILL) any remaining processes
5. Cleans up PID file

### 3. `check_vpu_parallel.sh` - Check status

Shows current status of parallel execution.

**Usage:**
```bash
# One-time check
bash scripts/check_vpu_parallel.sh

# Continuous monitoring (updates every 10 seconds)
watch -n 10 'bash scripts/check_vpu_parallel.sh'
```

**Shows:**
- Running/dead processes from PID file
- All Python training processes (CPU, memory, runtime)
- Total experiments completed
- Recent activity from status log
- Most recent completions

## Process Flow

### Starting Parallel Execution

1. **Check current status:**
   ```bash
   bash scripts/check_vpu_parallel.sh
   ```

2. **Stop existing sequential run (if needed):**
   ```bash
   # Find PID of current run
   ps aux | grep run_train.py

   # Kill it
   kill <PID>
   ```

3. **Start parallel execution:**
   ```bash
   bash scripts/run_vpu_parallel.sh
   ```

4. **Monitor progress:**
   ```bash
   watch -n 10 'bash scripts/check_vpu_parallel.sh'
   ```

### Stopping Parallel Execution

**Option 1: Graceful stop (Ctrl+C)**
- Press `Ctrl+C` in terminal where script is running
- Automatically cleans up all workers

**Option 2: Kill script**
```bash
bash scripts/kill_vpu_parallel.sh
```

**Option 3: Manual cleanup (if needed)**
```bash
# Find all training processes
ps aux | grep run_train.py

# Kill them
pkill -f run_train.py

# Force kill if needed
pkill -9 -f run_train.py

# Clean up PID file
rm -f /tmp/vpu_parallel_pids.txt
```

## Parallelization Details

### With GNU Parallel (Recommended)

Install: `brew install parallel`

**Runs:** 3 jobs simultaneously
**Expected speedup:** ~3x

### Without GNU Parallel

**Runs:** 2 jobs simultaneously
**Expected speedup:** ~2x

## File Locations

- **PID tracking:** `/tmp/vpu_parallel_pids.txt`
- **Status log:** `logs/vpu_rerun/parallel/status.txt`
- **Individual dataset logs:** `logs/vpu_rerun/parallel/<dataset>.log`
- **Results:** `results/` (same as sequential)

## Safety Features

1. **Process tracking:** All worker PIDs tracked in file
2. **Signal handling:** Ctrl+C triggers cleanup
3. **Graceful shutdown:** SIGTERM first, then SIGKILL
4. **Resume support:** `--resume` flag skips completed experiments
5. **Isolated logs:** Each dataset logs separately
6. **No ghost processes:** Kill script ensures all workers stopped

## Example Session

```bash
# 1. Start parallel execution
bash scripts/run_vpu_parallel.sh &

# 2. Monitor in separate terminal
watch -n 10 'bash scripts/check_vpu_parallel.sh'

# 3. To stop (in another terminal or Ctrl+C main process)
bash scripts/kill_vpu_parallel.sh

# 4. Verify cleanup
bash scripts/check_vpu_parallel.sh
# Should show: (none running)
```

## Expected Performance

**Current (sequential):**
- Pace: 11.6 min/experiment
- Remaining: 1,104 experiments
- Time: 213 hours (8.9 days)

**With 2x parallel:**
- Pace: 11.6 min/experiment per worker
- Effective: 5.8 min/experiment
- Time: **106 hours (4.4 days)**

**With 3x parallel (GNU parallel):**
- Pace: 11.6 min/experiment per worker
- Effective: 3.9 min/experiment
- Time: **71 hours (3 days)**

## Troubleshooting

### Ghost processes remain after kill

```bash
# Find all training processes
ps aux | grep run_train.py | grep -v grep

# Force kill all
pkill -9 -f run_train.py

# Clean up PID file
rm -f /tmp/vpu_parallel_pids.txt
```

### Check for zombie processes

```bash
# List all Python processes
ps aux | grep python | grep -v grep

# Kill specific PID
kill -9 <PID>
```

### Logs not appearing

Check: `logs/vpu_rerun/parallel/`

If missing, create: `mkdir -p logs/vpu_rerun/parallel`
