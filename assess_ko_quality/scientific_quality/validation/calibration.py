# scientific_quality/validation/calibration.py
"""
Model calibration analysis.
"""

import numpy as np
from typing import Dict, Tuple


def compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration metrics.
    
    A well-calibrated model will have predicted 4.0 actually be 4.0 on average.
    
    Args:
        y_true: True human judgments
        y_pred: Model predictions
        n_bins: Number of calibration bins
    
    Returns:
        Dict with bin_centers, bin_accuracies, bin_counts
    """
    # Create bins
    bin_edges = np.linspace(1, 5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Find predictions in this bin
        mask = (y_pred >= lower) & (y_pred < upper)
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (y_pred >= lower) & (y_pred <= upper)
        
        if np.sum(mask) > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            
            # Accuracy: how close are predictions to true values?
            accuracy = 1 - np.abs(bin_true - bin_pred).mean() / 4  # Normalize to 0-1
            
            bin_accuracies.append(accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": np.array(bin_accuracies),
        "bin_counts": np.array(bin_counts),
        "expected_calibration_error": np.mean(np.abs(np.array(bin_accuracies) - 0.5)),
    }


def plot_calibration_curve(
    calibration_results: Dict,
    dimension: str = "Overall",
    output_path: str = None,
):
    """
    Plot calibration curve.
    
    Args:
        calibration_results: Output from compute_calibration
        dimension: Quality dimension name
        output_path: Where to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    bin_centers = calibration_results["bin_centers"]
    bin_accuracies = calibration_results["bin_accuracies"]
    bin_counts = calibration_results["bin_counts"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(bin_centers, bin_accuracies, 'o-', label='Model', linewidth=2)
    ax.plot([1, 5], [1, 5], 'k--', label='Perfect calibration')
    
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Actual Score')
    ax.set_title(f'Calibration Curve - {dimension}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add count annotations
    for i, (x, y, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        ax.annotate(f'n={count}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Calibration plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
