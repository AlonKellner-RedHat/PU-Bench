# Early Stopping Implementation in PU-Bench

## Overview

Early stopping is implemented through the `CheckpointHandler` class (also known as `ModelCheckpoint`) in `train/train_utils.py`. It monitors a specified metric during training and stops when that metric stops improving.

---

## How It Works

### 1. Configuration (Method Config Files)

Early stopping is configured in method YAML files (e.g., `config/methods/vpu_nomixup_mean_prior.yaml`):

```yaml
checkpoint:
  enabled: true              # Enable checkpointing
  save_model: false          # Whether to save model weights
  monitor: "val_f1"          # Metric to monitor for improvement
  mode: "max"                # "max" for metrics to maximize, "min" for metrics to minimize
  early_stopping:
    enabled: true            # Enable early stopping
    patience: 10             # Number of epochs to wait without improvement
    min_delta: 0.0001        # Minimum change to qualify as improvement
```

**Default Configuration** (used in Phase 1 & 2 experiments):
- **Monitor**: `val_f1` (validation F1 score)
- **Mode**: `max` (stop when F1 stops increasing)
- **Patience**: 10 epochs
- **Min delta**: 0.0001 (improvement must be > 0.0001 to count)

---

### 2. Initialization (base_trainer.py)

The `CheckpointHandler` is created in `_init_checkpoint_handler()`:

```python
self.checkpoint_handler = ModelCheckpoint(
    save_dir=save_dir,
    filename=filename,
    monitor="val_f1",              # From config
    mode="max",                    # From config
    save_model=True/False,         # From config
    verbose=True,
    file_console=self.file_console,
    early_stopping_params={
        "enabled": True,
        "patience": 10,
        "min_delta": 0.0001
    }
)
```

**Early stopping attributes** (train_utils.py, lines 935-940):
```python
self.early_stopping_enabled = False
self.patience = float("inf")    # Epochs to wait without improvement
self.min_delta = 0.0            # Minimum improvement threshold
self.wait = 0                   # Counter for epochs without improvement
self.should_stop = False        # Flag to trigger stopping
```

---

### 3. Training Loop (base_trainer.py)

**Every epoch** (lines 396-469):

1. **Train one epoch**: `self.train_one_epoch(epoch_idx)`

2. **Evaluate all splits**:
   ```python
   train_metrics = evaluate_metrics(model, train_loader, device, prior)
   test_metrics = evaluate_metrics(model, test_loader, device, prior)
   val_metrics = evaluate_metrics(model, validation_loader, device, prior)
   ```

3. **Call checkpoint handler**:
   ```python
   all_metrics = {
       "train_f1": ..., "train_ap": ...,
       "val_f1": ..., "val_ap": ...,
       "test_f1": ..., "test_ap": ...
   }
   
   checkpoint_handler(
       epoch=epoch_idx,
       all_metrics=all_metrics,
       model=model,
       elapsed_seconds=time_since_start
   )
   ```

4. **Check if should stop**:
   ```python
   if checkpoint_handler.should_stop:
       console.log("Early stopping triggered.")
       break
   ```

---

### 4. CheckpointHandler Logic (train_utils.py)

**On each call** (lines 963-1038):

```python
def __call__(self, epoch, all_metrics, model, elapsed_seconds):
    current_score = all_metrics.get(self.monitor)  # e.g., "val_f1"
    
    # Check if improved
    if self.mode == "max":
        improved = (current_score > self.best_score + self.min_delta)
    else:  # mode == "min"
        improved = (current_score < self.best_score - self.min_delta)
    
    if improved:
        # Save new best
        self.best_score = current_score
        self.best_epoch = epoch
        self.best_metrics = all_metrics
        
        if self.save_model:
            torch.save(model.state_dict(), self.save_path)
        
        # Reset early stopping counter
        self.wait = 0
        
    elif self.early_stopping_enabled:
        # No improvement
        self.wait += 1
        
        if self.wait >= self.patience:
            self.should_stop = True  # Trigger early stopping
```

---

## Decision Flow

```
Epoch N completes
    ├─ Evaluate: train, val, test metrics
    ├─ CheckpointHandler called with val_f1
    │
    ├─ val_f1 > best_val_f1 + 0.0001?
    │   ├─ YES: Improvement detected
    │   │   ├─ Save new best_score = val_f1
    │   │   ├─ Save best_epoch = N
    │   │   ├─ Save best_metrics (all metrics from this epoch)
    │   │   ├─ Reset wait = 0
    │   │   └─ Save model (if enabled)
    │   │
    │   └─ NO: No improvement
    │       ├─ Increment wait += 1
    │       └─ wait >= patience (10)?
    │           ├─ YES: Set should_stop = True → STOP TRAINING
    │           └─ NO: Continue training
    │
    └─ Check checkpoint_handler.should_stop
        ├─ True: Break training loop
        └─ False: Continue to next epoch
```

---

## Example: Early Stopping in Action

**Epoch progression**:

| Epoch | val_f1 | best_f1 | Improved? | wait | Action |
|-------|--------|---------|-----------|------|--------|
| 1 | 0.7500 | -inf | ✓ | 0 | Save as best |
| 2 | 0.8000 | 0.7500 | ✓ | 0 | Save as best |
| 3 | 0.8200 | 0.8000 | ✓ | 0 | Save as best |
| 4 | 0.8100 | 0.8200 | ✗ | 1 | Wait |
| 5 | 0.8150 | 0.8200 | ✗ | 2 | Wait (0.815 < 0.82 + 0.0001) |
| 6 | 0.8050 | 0.8200 | ✗ | 3 | Wait |
| 7 | 0.8300 | 0.8200 | ✓ | 0 | Save as best, reset wait |
| 8 | 0.8280 | 0.8300 | ✗ | 1 | Wait |
| 9 | 0.8290 | 0.8300 | ✗ | 2 | Wait |
| 10 | 0.8250 | 0.8300 | ✗ | 3 | Wait |
| 11 | 0.8200 | 0.8300 | ✗ | 4 | Wait |
| 12 | 0.8150 | 0.8300 | ✗ | 5 | Wait |
| 13 | 0.8100 | 0.8300 | ✗ | 6 | Wait |
| 14 | 0.8050 | 0.8300 | ✗ | 7 | Wait |
| 15 | 0.8000 | 0.8300 | ✗ | 8 | Wait |
| 16 | 0.7950 | 0.8300 | ✗ | 9 | Wait |
| 17 | 0.7900 | 0.8300 | ✗ | 10 | **STOP** (patience=10 reached) |

**Result**: Training stops at epoch 17, but **best model is from epoch 7** with val_f1 = 0.8300

---

## Key Features

### 1. Best Model Selection

The handler **always tracks the best model** based on the monitored metric, not the final epoch:

```python
self.best_epoch = 7        # Epoch where best validation F1 occurred
self.best_score = 0.8300   # Best validation F1 value
self.best_metrics = {...}  # All metrics from epoch 7
```

**This is crucial**: Even though training stopped at epoch 17, the reported metrics are from **epoch 7**.

### 2. Metric Monitoring

**Monitored metric**: `val_f1` (validation F1 score)
- Uses **validation set** for early stopping (not test set)
- Prevents overfitting to test set
- Test metrics are still evaluated every epoch but not used for stopping

**Fallback behavior** (lines 434-451):
If `val_f1` doesn't exist (no validation set), it falls back to:
1. `test_f1` (if available)
2. `train_f1` (if available)
3. Warn and skip checkpoint logic

### 3. Min Delta Threshold

`min_delta = 0.0001` means improvement must be **> 0.0001** to reset the patience counter.

**Example**:
- Current best: 0.8200
- New score: 0.82009
- Improvement: 0.00009 < 0.0001 → **Not considered improvement**
- `wait` counter increments

This prevents noisy fluctuations from resetting early stopping.

### 4. Model Saving

**In Phase 1 & 2 experiments**: `save_model: false`
- Models are **not saved to disk**
- Only best metrics are tracked
- Saves storage space (1,500 experiments × multiple methods)

**If enabled**: Model weights saved to `checkpoints/{method}_{experiment}.pth` whenever improvement occurs.

---

## Phase 1 & 2 Configuration

**All comprehensive experiments use**:
- Monitor: `val_f1`
- Patience: 10 epochs
- Min delta: 0.0001
- Max epochs: 40

**Typical behavior**:
- Training runs for 10-20 epochs (median: 8-10)
- Early stopping triggers before max epochs (40) most of the time
- Best model usually found in first 10-15 epochs

**From convergence analysis** (comprehensive_auto_vs_05_results):
```
Epochs until early stopping:
  auto: 11.01 ± 8.76 (median: 8)
  0.5:  9.45 ± 7.97 (median: 7)
```

Most experiments stop early, saving compute time vs running full 40 epochs.

---

## Advantages

1. **Prevents overfitting**: Stops when validation performance plateaus
2. **Saves compute**: Doesn't waste time on unnecessary epochs
3. **Robust model selection**: Always returns best model, not final model
4. **Configurable**: Can adjust patience, min_delta, monitored metric per method
5. **Works with validation set**: Uses proper held-out data for stopping decisions

---

## Limitations

1. **Requires validation set**: Must have enough data to split train/val
   - Phase 1 & 2 use `val_ratio: 0.01` (1% of training data for validation)
   
2. **May stop too early**: If patience is too low, might miss later improvements

3. **Validation set noise**: Small validation sets (1%) can be noisy, causing premature stopping

4. **Metric-dependent**: Only monitors one metric (val_f1)
   - Doesn't consider other metrics like AP or calibration
   - Could theoretically stop when val_f1 plateaus but test_ap still improving

---

## Summary

**Early stopping in PU-Bench**:
- Monitors **validation F1 score** every epoch
- Stops training after **10 epochs without improvement** (patience=10)
- Improvement must be **> 0.0001** (min_delta)
- Always returns **best model** (not final model)
- Median stopping epoch: **8-10** (vs max 40 epochs)
- Saves **50-75% of training time** vs running full 40 epochs

This is why in the Phase 2 logs you see experiments stopping early (e.g., epoch 14/40) but the final metrics come from an earlier epoch where validation F1 was highest.
