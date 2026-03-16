#!/usr/bin/env python3
"""Deep diagnostic script to pinpoint exact source of NaN in meta-training.

Reproduces the exact conditions from training:
- L0.5 lambda = 0.005
- L1 lambda = 0.01
- Max weight norm = 10.0
- Meta LR = 0.0005
- Cosine similarity loss

Monitors every operation to find where NaN first appears.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/Users/akellner/MyDir/Code/Other/PU-Bench/toy_meta_learning')

from loss.neural_pu_loss import NeuralPULoss
from tasks.gaussian_task import GaussianBlobTask
from models.simple_mlp import SimpleMLP
import torch.func


def check_tensor(name, tensor, iteration, step_desc):
    """Check tensor for NaN/Inf and print diagnostics."""
    if tensor is None:
        print(f"  [{step_desc}] {name}: None")
        return False

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"  ⚠️  [{step_desc}] {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"      Shape: {tensor.shape}, Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
        print(f"      Sample values: {tensor.flatten()[:5]}")
        return True
    else:
        print(f"  ✓  [{step_desc}] {name}: OK (range: [{tensor.min():.6f}, {tensor.max():.6f}])")
        return False


def diagnose_forward_pass(loss_fn, outputs, labels, iteration):
    """Detailed monitoring of forward pass."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - FORWARD PASS DIAGNOSIS")
    print(f"{'='*70}")

    # Hook into intermediate activations
    intermediate_values = {}

    def capture_intermediate():
        nonlocal intermediate_values
        p = torch.sigmoid(outputs.view(-1))
        labels_flat = labels.view(-1)

        M_p = (labels_flat == 1).float()
        M_u = (labels_flat == -1).float()

        B = float(len(M_p))
        M_p_norm = M_p / B
        M_u_norm = M_u / B

        V = p
        V_p = V * M_p
        V_u = V * M_u

        log_V = loss_fn.safe_log(V)
        log_V_p = loss_fn.safe_log(V_p)
        log_V_u = loss_fn.safe_log(V_u)
        log_1mV = loss_fn.safe_log(1 - V)
        log_1mV_p = loss_fn.safe_log(1 - V_p)
        log_1mV_u = loss_fn.safe_log(1 - V_u)

        intermediate_values['p'] = p
        intermediate_values['M_p'] = M_p
        intermediate_values['M_u'] = M_u
        intermediate_values['V_p'] = V_p
        intermediate_values['V_u'] = V_u
        intermediate_values['log_V'] = log_V
        intermediate_values['log_V_p'] = log_V_p
        intermediate_values['log_V_u'] = log_V_u
        intermediate_values['log_1mV'] = log_1mV
        intermediate_values['log_1mV_p'] = log_1mV_p
        intermediate_values['log_1mV_u'] = log_1mV_u

        input_tensor = torch.stack([
            M_p, M_u, M_p_norm, M_u_norm, V, V_p, V_u,
            log_V, log_V_p, log_V_u,
            log_1mV, log_1mV_p, log_1mV_u
        ], dim=1)

        intermediate_values['input_tensor'] = input_tensor

        T = loss_fn.linear(input_tensor)
        intermediate_values['T'] = T

        half = loss_fn.hidden_dim // 2
        X1 = T[:, :half]
        X2 = T[:, half:]
        intermediate_values['X1'] = X1
        intermediate_values['X2'] = X2

        P = X1 * X2
        intermediate_values['P'] = P

        A = P.sum(dim=0)
        intermediate_values['A'] = A

        quarter = half // 2
        A1 = A[:quarter]
        A2 = A[quarter:]
        intermediate_values['A1'] = A1
        intermediate_values['A2'] = A2

        log_abs_A2 = loss_fn.safe_log(torch.abs(A2))
        intermediate_values['log_abs_A2'] = log_abs_A2

        loss_main = A1.sum() + log_abs_A2.sum()
        intermediate_values['loss_main'] = loss_main

        return loss_main

    # Capture intermediate values
    loss_main = capture_intermediate()

    # Check each intermediate value
    print("\n1. Input Processing:")
    check_tensor("outputs (logits)", outputs, iteration, "input")
    check_tensor("labels", labels, iteration, "input")
    check_tensor("p (sigmoid)", intermediate_values['p'], iteration, "sigmoid")

    print("\n2. Masks and Derived Features:")
    check_tensor("M_p", intermediate_values['M_p'], iteration, "masks")
    check_tensor("M_u", intermediate_values['M_u'], iteration, "masks")
    check_tensor("V_p", intermediate_values['V_p'], iteration, "derived")
    check_tensor("V_u", intermediate_values['V_u'], iteration, "derived")

    print("\n3. Log Features:")
    check_tensor("log_V", intermediate_values['log_V'], iteration, "logs")
    check_tensor("log_V_p", intermediate_values['log_V_p'], iteration, "logs")
    check_tensor("log_V_u", intermediate_values['log_V_u'], iteration, "logs")
    check_tensor("log_1mV", intermediate_values['log_1mV'], iteration, "logs")
    check_tensor("log_1mV_p", intermediate_values['log_1mV_p'], iteration, "logs")
    check_tensor("log_1mV_u", intermediate_values['log_1mV_u'], iteration, "logs")

    print("\n4. Input Tensor:")
    check_tensor("input_tensor [B, 13]", intermediate_values['input_tensor'], iteration, "stacked")

    print("\n5. Linear Transformation:")
    check_tensor("weights", loss_fn.linear.weight, iteration, "params")
    check_tensor("bias", loss_fn.linear.bias, iteration, "params")
    check_tensor("T [B, hidden_dim]", intermediate_values['T'], iteration, "linear")

    print("\n6. Split and Multiply:")
    check_tensor("X1 [B, 64]", intermediate_values['X1'], iteration, "split")
    check_tensor("X2 [B, 64]", intermediate_values['X2'], iteration, "split")
    check_tensor("P = X1*X2 [B, 64]", intermediate_values['P'], iteration, "multiply")

    print("\n7. Aggregation:")
    check_tensor("A = sum(P) [64]", intermediate_values['A'], iteration, "aggregate")
    check_tensor("A1 [32]", intermediate_values['A1'], iteration, "split_A")
    check_tensor("A2 [32]", intermediate_values['A2'], iteration, "split_A")

    print("\n8. Final Loss Computation:")
    check_tensor("log(abs(A2))", intermediate_values['log_abs_A2'], iteration, "log_A2")
    check_tensor("loss_main", loss_main, iteration, "main_loss")

    # Compute regularization separately
    print("\n9. Regularization:")
    if loss_fn.l1_lambda > 0 or loss_fn.l05_lambda > 0:
        reg = loss_fn.compute_regularization()
        check_tensor("regularization", reg, iteration, "reg")

        # Break down regularization components
        if loss_fn.l1_lambda > 0:
            l1_reg = loss_fn.l1_lambda * torch.sum(torch.abs(loss_fn.linear.weight))
            l1_reg += loss_fn.l1_lambda * torch.sum(torch.abs(loss_fn.linear.bias))
            check_tensor("L1 regularization", l1_reg, iteration, "L1")

        if loss_fn.l05_lambda > 0:
            # Without eps
            l05_reg_no_eps = loss_fn.l05_lambda * torch.sum(torch.abs(loss_fn.linear.weight) ** 0.5)
            l05_reg_no_eps += loss_fn.l05_lambda * torch.sum(torch.abs(loss_fn.linear.bias) ** 0.5)
            check_tensor("L0.5 reg (NO eps)", l05_reg_no_eps, iteration, "L0.5_no_eps")

            # With eps (current implementation)
            l05_reg_with_eps = loss_fn.l05_lambda * torch.sum((torch.abs(loss_fn.linear.weight) + loss_fn.eps) ** 0.5)
            l05_reg_with_eps += loss_fn.l05_lambda * torch.sum((torch.abs(loss_fn.linear.bias) + loss_fn.eps) ** 0.5)
            check_tensor("L0.5 reg (WITH eps)", l05_reg_with_eps, iteration, "L0.5_with_eps")

    # Final loss
    loss = loss_fn(outputs, labels, mode='pu')
    check_tensor("TOTAL LOSS", loss, iteration, "final")

    return loss, intermediate_values


def diagnose_backward_pass(loss, learned_loss, meta_optimizer, iteration):
    """Monitor backward pass and gradient updates."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - BACKWARD PASS DIAGNOSIS")
    print(f"{'='*70}")

    # Check loss before backward
    print("\n1. Pre-backward:")
    check_tensor("loss (requires_grad)", loss, iteration, "pre_backward")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    # Backward
    print("\n2. Computing gradients...")
    meta_optimizer.zero_grad()
    loss.backward()

    # Check gradients
    print("\n3. Post-backward gradients:")
    check_tensor("weight.grad", learned_loss.linear.weight.grad, iteration, "gradients")
    check_tensor("bias.grad", learned_loss.linear.bias.grad, iteration, "gradients")

    # Check gradient norms
    weight_grad_norm = learned_loss.linear.weight.grad.norm().item()
    bias_grad_norm = learned_loss.linear.bias.grad.norm().item()
    print(f"  Weight grad norm: {weight_grad_norm:.6f}")
    print(f"  Bias grad norm: {bias_grad_norm:.6f}")

    # Clip gradients
    print("\n4. Gradient clipping (max_norm=1.0):")
    total_norm_before = torch.nn.utils.clip_grad_norm_(learned_loss.parameters(), max_norm=1.0)
    print(f"  Total norm before clip: {total_norm_before:.6f}")

    check_tensor("weight.grad (after clip)", learned_loss.linear.weight.grad, iteration, "clipped")
    check_tensor("bias.grad (after clip)", learned_loss.linear.bias.grad, iteration, "clipped")

    # Optimizer step
    print("\n5. Optimizer step:")
    print(f"  LR: {meta_optimizer.param_groups[0]['lr']:.6f}")
    meta_optimizer.step()

    # Check weights after update
    print("\n6. Post-update weights:")
    check_tensor("weight (after step)", learned_loss.linear.weight, iteration, "post_update")
    check_tensor("bias (after step)", learned_loss.linear.bias, iteration, "post_update")

    weight_norm = learned_loss.linear.weight.norm().item()
    print(f"  Weight norm: {weight_norm:.6f}")


def main():
    print("="*70)
    print("NaN DIAGNOSTIC - DEEP INVESTIGATION")
    print("="*70)
    print()
    print("Configuration:")
    print("  L0.5 lambda: 0.005")
    print("  L1 lambda: 0.01")
    print("  Max weight norm: 10.0")
    print("  Meta LR: 0.0005")
    print("  Hidden dim: 128")
    print("="*70)

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Create loss function (exact config from training)
    learned_loss = NeuralPULoss(
        hidden_dim=128,
        eps=1e-7,
        l1_lambda=0.01,
        l05_lambda=0.005,
        init_mode='xavier_uniform',
        init_scale=1.0,
        max_weight_norm=10.0,
    )

    # Create optimizer (exact config from training)
    meta_optimizer = torch.optim.AdamW(
        learned_loss.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=0.00001,
    )

    # Create a simple task
    task = GaussianBlobTask(
        num_dimensions=2,
        num_samples=1000,
        mean_separation=2.0,
        std=1.0,
        labeling_freq=0.3,
        prior=0.5,
        seed=42,
    )

    # Create model
    model = SimpleMLP(2, [32, 32])
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Run 25 iterations (NaN appears around iteration 20)
    for iteration in range(25):
        print(f"\n\n{'#'*70}")
        print(f"# META-ITERATION {iteration}")
        print(f"{'#'*70}")

        # Sample batch
        dataloaders = task.get_dataloaders(batch_size=64, num_train=1000, num_val=200)
        batch = next(iter(dataloaders['train']))
        x, y_true, y_pu = batch

        # Forward through model
        outputs = model(x).squeeze(-1)

        # Diagnose forward pass through loss
        loss, intermediates = diagnose_forward_pass(learned_loss, outputs, y_pu, iteration)

        # Check if NaN appeared
        if torch.isnan(loss).any():
            print("\n" + "!"*70)
            print(f"!!! NaN DETECTED AT ITERATION {iteration} !!!")
            print("!"*70)
            print("\nStopping diagnostics. Review output above to find source.")
            break

        # Diagnose backward pass
        diagnose_backward_pass(loss, learned_loss, meta_optimizer, iteration)

        # Check if NaN appeared after backward
        if torch.isnan(learned_loss.linear.weight).any() or torch.isnan(learned_loss.linear.bias).any():
            print("\n" + "!"*70)
            print(f"!!! NaN IN WEIGHTS AFTER ITERATION {iteration} !!!")
            print("!"*70)
            print("\nStopping diagnostics. Review output above to find source.")
            break

        # Take one training step with the model (to change outputs for next iteration)
        model_optimizer.zero_grad()
        model_loss = learned_loss(model(x).squeeze(-1), y_pu, mode='pu')
        model_loss.backward()
        model_optimizer.step()

        print(f"\n{'='*70}")
        print(f"END OF ITERATION {iteration}")
        print(f"{'='*70}")

    print("\n\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
