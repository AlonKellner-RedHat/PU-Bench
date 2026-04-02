#!/usr/bin/env python3
"""Validate theoretically derived positive_weight formula against empirical data

Tests the formula: α = π · [1 + κ·(1-c)]
against optimal values found in robustness experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error


def compute_positive_weight(prior: float, label_frequency: float,
                           scarcity_factor: float = 0.5) -> float:
    """
    Compute positive weight with theoretical justification.

    Formula: α = π · [1 + κ·(1-c)]

    Args:
        prior: True class prior π
        label_frequency: Fraction of positives labeled (c)
        scarcity_factor: Boost parameter κ (default 0.5)

    Returns:
        Positive weight α
    """
    weight_multiplier = 1 + scarcity_factor * (1 - label_frequency)
    return prior * weight_multiplier


def load_optimal_data():
    """Load empirical optimal prior data from robustness analysis"""
    data_file = Path("results_robustness/optimal_prior_analysis.csv")

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Filter for cases where we have a numeric optimal prior (not NaN, not 'auto')
    df_numeric = df[df['optimal_prior'].notna()].copy()

    # Ensure optimal_prior is numeric
    df_numeric = df_numeric[pd.to_numeric(df_numeric['optimal_prior'], errors='coerce').notna()].copy()
    df_numeric['optimal_prior'] = df_numeric['optimal_prior'].astype(float)

    # Also filter out cases where optimal is same as baseline (auto was best)
    # These would have direction 'auto'
    if 'direction' in df_numeric.columns:
        df_numeric = df_numeric[df_numeric['direction'] != 'auto'].copy()

    return df_numeric


def fit_scarcity_factor(df):
    """Fit optimal κ parameter to data"""

    def model(X, kappa):
        """Model: α = π · [1 + κ·(1-c)]"""
        prior, c = X
        return prior * (1 + kappa * (1 - c))

    # Prepare data
    X = np.vstack([df['true_prior'].values, df['c'].values])
    y = df['optimal_prior'].values

    # Fit
    popt, pcov = curve_fit(model, X, y, p0=[0.5])
    kappa_optimal = popt[0]
    kappa_std = np.sqrt(pcov[0, 0])

    return kappa_optimal, kappa_std


def evaluate_formula(df, kappa):
    """Evaluate formula performance"""

    # Compute predicted values
    df['predicted_alpha'] = df.apply(
        lambda row: compute_positive_weight(row['true_prior'], row['c'], kappa),
        axis=1
    )

    # Compute metrics
    mae = mean_absolute_error(df['optimal_prior'], df['predicted_alpha'])
    r2 = r2_score(df['optimal_prior'], df['predicted_alpha'])

    # Relative error
    df['abs_error'] = np.abs(df['optimal_prior'] - df['predicted_alpha'])
    df['rel_error_pct'] = (df['abs_error'] / df['true_prior']) * 100

    return mae, r2, df


def plot_validation(df, kappa, output_dir):
    """Create validation plots"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by c value
    c_values = sorted(df['c'].unique())
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for c, color in zip(c_values, colors):
        subset = df[df['c'] == c]
        ax.scatter(subset['optimal_prior'], subset['predicted_alpha'],
                  c=color, label=f'c={c}', alpha=0.6, s=60)

    # Perfect prediction line
    min_val = min(df['optimal_prior'].min(), df['predicted_alpha'].min())
    max_val = max(df['optimal_prior'].max(), df['predicted_alpha'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect fit')

    ax.set_xlabel('Empirical Optimal α')
    ax.set_ylabel('Predicted α (formula)')
    ax.set_title(f'Positive Weight Formula Validation\nκ={kappa:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / "formula_validation_scatter.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot")
    plt.close()

    # Plot 2: Error by c
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error
    ax = axes[0]
    summary = df.groupby('c').agg({
        'abs_error': ['mean', 'std', 'count']
    })['abs_error']

    x = np.arange(len(c_values))
    ax.bar(x, summary['mean'], yerr=summary['std'], capsize=5,
           color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in c_values])
    ax.set_xlabel('Label Frequency (c)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error by Label Frequency')
    ax.grid(True, axis='y', alpha=0.3)

    # Add count labels
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.01, f"n={int(row['count'])}",
               ha='center', va='bottom', fontsize=9)

    # Relative error
    ax = axes[1]
    summary_rel = df.groupby('c').agg({
        'rel_error_pct': ['mean', 'std']
    })['rel_error_pct']

    ax.bar(x, summary_rel['mean'], yerr=summary_rel['std'], capsize=5,
           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in c_values])
    ax.set_xlabel('Label Frequency (c)')
    ax.set_ylabel('Mean Relative Error (%)')
    ax.set_title('Prediction Error (Relative to Prior)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "formula_validation_error_by_c.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved error analysis plot")
    plt.close()

    # Plot 3: Residuals
    fig, ax = plt.subplots(figsize=(10, 6))

    df['residual'] = df['optimal_prior'] - df['predicted_alpha']

    for c, color in zip(c_values, colors):
        subset = df[df['c'] == c]
        ax.scatter(subset['true_prior'], subset['residual'],
                  c=color, label=f'c={c}', alpha=0.6, s=60)

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('True Prior π')
    ax.set_ylabel('Residual (Empirical - Predicted)')
    ax.set_title('Residual Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "formula_validation_residuals.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved residuals plot")
    plt.close()


def create_comparison_table(df, kappa):
    """Create detailed comparison table"""

    # Group by (c, dataset) and show mean predictions
    summary = df.groupby(['c', 'dataset']).agg({
        'true_prior': 'mean',
        'optimal_prior': 'mean',
        'predicted_alpha': 'mean',
        'abs_error': 'mean',
        'rel_error_pct': 'mean'
    }).round(3)

    return summary


def test_alternative_formulas(df):
    """Test alternative formula forms"""

    formulas = {
        'Linear boost': lambda pi, c, k: pi + k * (1 - c),
        'Multiplicative (current)': lambda pi, c, k: pi * (1 + k * (1 - c)),
        'Inverse c': lambda pi, c, k: pi * (1 + k / c),
        'Sqrt boost': lambda pi, c, k: pi * (1 + k * np.sqrt(1 - c)),
        'Sampling correction': lambda pi, c, k: pi / (1 - k * c * pi),
    }

    results = []

    for name, formula in formulas.items():
        try:
            # Fit kappa for this formula
            def model(X, kappa):
                prior, c = X
                return formula(prior, c, kappa)

            X = np.vstack([df['true_prior'].values, df['c'].values])
            y = df['optimal_prior'].values

            popt, _ = curve_fit(model, X, y, p0=[0.5], maxfev=10000)
            kappa_fit = popt[0]

            # Compute predictions
            predictions = formula(df['true_prior'].values, df['c'].values, kappa_fit)

            # Metrics
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)

            results.append({
                'Formula': name,
                'κ (fitted)': f'{kappa_fit:.3f}',
                'MAE': f'{mae:.4f}',
                'R²': f'{r2:.4f}'
            })
        except Exception as e:
            print(f"Failed to fit {name}: {e}")

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("Positive Weight Formula Validation")
    print("=" * 80)
    print()

    # Load data
    print("Loading empirical optimal prior data...")
    df = load_optimal_data()
    print(f"✓ Loaded {len(df)} experiments with numeric optimal priors")
    print()

    # Fit scarcity factor
    print("Fitting scarcity factor κ...")
    kappa_opt, kappa_std = fit_scarcity_factor(df)
    print(f"✓ Optimal κ = {kappa_opt:.3f} ± {kappa_std:.3f}")
    print()

    # Evaluate formula
    print("Evaluating formula performance...")
    mae, r2, df_eval = evaluate_formula(df, kappa_opt)
    print(f"✓ Mean Absolute Error: {mae:.4f}")
    print(f"✓ R² Score: {r2:.4f}")
    print(f"✓ Mean Relative Error: {df_eval['rel_error_pct'].mean():.2f}%")
    print()

    # Create plots
    print("Creating validation plots...")
    output_dir = Path("results_robustness/formula_validation")
    plot_validation(df_eval, kappa_opt, output_dir)
    print()

    # Comparison table
    print("Summary by c and dataset:")
    summary = create_comparison_table(df_eval, kappa_opt)
    print(summary.to_string())
    print()

    # Save detailed results
    output_file = output_dir / "formula_validation_results.csv"
    df_eval.to_csv(output_file, index=False)
    print(f"✓ Saved detailed results to {output_file}")
    print()

    # Test alternative formulas
    print("Testing alternative formula forms...")
    alt_results = test_alternative_formulas(df)
    print(alt_results.to_string(index=False))
    print()

    alt_file = output_dir / "alternative_formulas.csv"
    alt_results.to_csv(alt_file, index=False)
    print(f"✓ Saved comparison to {alt_file}")
    print()

    # Final recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print(f"Use formula: α = π · [1 + {kappa_opt:.3f}·(1-c)]")
    print()
    print("Example values:")
    for c in [0.1, 0.5, 0.9]:
        for pi in [0.3, 0.5, 0.7]:
            alpha = compute_positive_weight(pi, c, kappa_opt)
            print(f"  π={pi:.1f}, c={c:.1f} → α={alpha:.3f} (boost: {(alpha/pi - 1)*100:+.1f}%)")
    print()

    print(f"Quality: MAE={mae:.4f}, R²={r2:.4f}")
    print(f"All plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
