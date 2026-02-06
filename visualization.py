"""
visualization.py - Visualization module for supernova anomaly detection results.

Creates publication-quality plots showing:
1. Training loss curves (is the model learning properly?)
2. Anomaly score distribution (what's the spread of scores?)
3. Ranked watchlist (which SNe are most anomalous?)
4. Type comparison (do IIn types rank higher as expected?)
5. Feature importance (which features drive anomaly scores?)
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Use non-interactive backend for environments without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color scheme by supernova type
TYPE_COLORS = {
    "IIn": "#e74c3c",   # Red - the interesting ones (potential precursors)
    "II": "#3498db",     # Blue - normal core-collapse
    "Ia": "#2ecc71",     # Green - thermonuclear (control group)
    "unknown": "#95a5a6", # Gray
}


def plot_training_curves(all_loss_histories, save_dir="plots"):
    """
    Plot training loss curves for all models in the ensemble.

    What to look for:
    - All curves should converge to similar final values
    - If one curve is very different, that model may have a bad initialization
    - The curves should level off (plateau), not keep decreasing forever
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: All loss curves
    ax = axes[0]
    for i, history in enumerate(all_loss_histories):
        ax.plot(history, alpha=0.7, label=f"Model {i+1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss - All Ensemble Models")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Loss curves on log scale (shows convergence better)
    ax = axes[1]
    for i, history in enumerate(all_loss_histories):
        ax.semilogy(history, alpha=0.7, label=f"Model {i+1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training Loss - Log Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {path}")


def plot_anomaly_scores(results, save_dir="plots"):
    """
    Plot the ranked anomaly scores as a bar chart.

    This is the main output - the WATCHLIST.
    Each bar is a supernova, colored by type.
    SN2009ip is highlighted with a star marker.
    """
    os.makedirs(save_dir, exist_ok=True)

    names = [r["name"] for r in results]
    scores = [r["anomaly_score"] for r in results]
    types = [r["type"] for r in results]
    score_stds = [r["score_std"] for r in results]

    colors = [TYPE_COLORS.get(t, TYPE_COLORS["unknown"]) for t in types]

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.8,
                   xerr=score_stds, capsize=3, ecolor='gray')

    # Highlight SN2009ip
    for i, name in enumerate(names):
        if "2009ip" in name.lower():
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)
            ax.annotate('KEY VALIDATION',
                       xy=(scores[i], i),
                       xytext=(scores[i] + max(scores) * 0.05, i),
                       fontsize=9, fontweight='bold', color='darkred',
                       arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Anomaly Score (Reconstruction Error)")
    ax.set_title("Supernova Anomaly Watchlist\n(Higher Score = More Anomalous = Priority Target)")
    ax.invert_yaxis()  # Highest anomaly at top

    # Legend
    legend_patches = [
        mpatches.Patch(color=TYPE_COLORS["IIn"], label="Type IIn (CSM interaction)"),
        mpatches.Patch(color=TYPE_COLORS["II"], label="Type II (normal CC)"),
        mpatches.Patch(color=TYPE_COLORS["Ia"], label="Type Ia (thermonuclear)"),
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "anomaly_watchlist.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved anomaly watchlist to {path}")


def plot_type_comparison(results, save_dir="plots"):
    """
    Compare anomaly score distributions by supernova type.

    Scientific expectation:
    - Type IIn should have HIGHER median anomaly scores
      (circumstellar interaction = mass loss = unusual light curves)
    - Type Ia may also score high (different physics = different features)
    - Type II should be the "normal" baseline
    """
    os.makedirs(save_dir, exist_ok=True)

    type_scores = {}
    for r in results:
        t = r["type"]
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["anomaly_score"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Box plot
    ax = axes[0]
    type_labels = sorted(type_scores.keys())
    box_data = [type_scores[t] for t in type_labels]
    box_colors = [TYPE_COLORS.get(t, TYPE_COLORS["unknown"]) for t in type_labels]

    bp = ax.boxplot(box_data, labels=type_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Anomaly Score")
    ax.set_title("Anomaly Score Distribution by Type")
    ax.grid(True, axis='y', alpha=0.3)

    # Right: Individual points with jitter
    ax = axes[1]
    for i, t in enumerate(type_labels):
        scores = type_scores[t]
        jitter = np.random.normal(0, 0.05, len(scores))
        ax.scatter([i + jitter[j] for j in range(len(scores))],
                  scores,
                  c=TYPE_COLORS.get(t, TYPE_COLORS["unknown"]),
                  s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Label SN2009ip
        for r in results:
            if r["type"] == t and "2009ip" in r["name"].lower():
                idx = [s for s in type_scores[t]].index(r["anomaly_score"])
                ax.annotate(r["name"],
                           xy=(i, r["anomaly_score"]),
                           xytext=(i + 0.3, r["anomaly_score"]),
                           fontsize=8, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xticks(range(len(type_labels)))
    ax.set_xticklabels(type_labels)
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Individual Anomaly Scores by Type")
    ax.grid(True, axis='y', alpha=0.3)

    # Add median lines
    for i, t in enumerate(type_labels):
        median = np.median(type_scores[t])
        ax.axhline(y=median, xmin=(i)/len(type_labels),
                   xmax=(i+1)/len(type_labels),
                   color=TYPE_COLORS.get(t, 'gray'), linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "type_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved type comparison to {path}")


def plot_score_distribution(results, save_dir="plots"):
    """
    Histogram of anomaly scores with threshold analysis.

    Helps answer: "What score threshold should we use for the watchlist?"
    Objects above the threshold go on the watchlist.
    """
    os.makedirs(save_dir, exist_ok=True)

    scores = [r["anomaly_score"] for r in results]
    types = [r["type"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram
    ax = axes[0]
    ax.hist(scores, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    # Mark thresholds
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    ax.axvline(mean_score, color='orange', linestyle='--', label=f'Mean ({mean_score:.4f})')
    ax.axvline(mean_score + std_score, color='red', linestyle='--',
               label=f'Mean + 1 std ({mean_score + std_score:.4f})')
    ax.axvline(mean_score + 2 * std_score, color='darkred', linestyle='--',
               label=f'Mean + 2 std ({mean_score + 2 * std_score:.4f})')
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Anomaly Scores")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Cumulative distribution
    ax = axes[1]
    sorted_scores = sorted(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cumulative, 'b-', linewidth=2)

    # Mark where SN2009ip falls
    for r in results:
        if "2009ip" in r["name"].lower():
            sn2009ip_score = r["anomaly_score"]
            percentile = np.mean(np.array(scores) <= sn2009ip_score)
            ax.axvline(sn2009ip_score, color='red', linestyle='--',
                      label=f'SN2009ip (top {(1-percentile)*100:.0f}%)')
            ax.scatter([sn2009ip_score], [percentile], color='red', s=100, zorder=5)

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Distribution of Anomaly Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved score distribution to {path}")


def plot_feature_importance(ensemble, data_tensor, feature_names, save_dir="plots"):
    """
    Estimate which features contribute most to anomaly scores.

    Method: For each feature, we shuffle its values across samples and see
    how much the anomaly scores change. Features that cause big changes when
    shuffled are important for the anomaly detection.

    This is called "permutation importance" and gives us scientific insight:
    which aspects of the light curve make an object anomalous?
    """
    import torch
    os.makedirs(save_dir, exist_ok=True)

    # Get baseline scores
    mean_scores_base, _, _ = ensemble.get_ensemble_scores(data_tensor)
    base_mean = float(mean_scores_base.mean())

    importances = []
    for j in range(data_tensor.shape[1]):
        # Shuffle feature j
        data_shuffled = data_tensor.clone()
        perm = torch.randperm(data_tensor.shape[0])
        data_shuffled[:, j] = data_tensor[perm, j]

        # Get new scores
        mean_scores_shuffled, _, _ = ensemble.get_ensemble_scores(data_shuffled)
        shuffled_mean = float(mean_scores_shuffled.mean())

        # Importance = change in mean anomaly score
        importances.append(abs(shuffled_mean - base_mean))

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = [feature_names[i] for i in sorted_idx[:top_n]]
    top_importances = [importances[i] for i in sorted_idx[:top_n]]

    ax.barh(range(top_n), top_importances, color='steelblue', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Permutation Importance (Change in Mean Anomaly Score)")
    ax.set_title("Top Features Contributing to Anomaly Detection")
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance to {path}")

    return list(zip(top_features, top_importances))


def generate_all_plots(results, all_loss_histories, ensemble, data_tensor,
                       feature_names, save_dir="plots"):
    """Generate all visualization plots."""
    logger.info("Generating visualizations...")

    plot_training_curves(all_loss_histories, save_dir)
    plot_anomaly_scores(results, save_dir)
    plot_type_comparison(results, save_dir)
    plot_score_distribution(results, save_dir)
    top_features = plot_feature_importance(ensemble, data_tensor, feature_names, save_dir)

    logger.info(f"All plots saved to {save_dir}/")
    return top_features
