#!/usr/bin/env python3
"""
main.py - Supernova Precursor Detection Pipeline

A complete PyTorch-based anomaly detection system for identifying supernovae
that may exhibit precursor activity before explosion.

This implements the "Watchlist Approach" inspired by the LAISS pipeline:
1. Collect light curve data from the Open Supernova Catalog
2. Extract statistical features from each light curve
3. Train an autoencoder ensemble to learn "normal" behavior
4. Rank objects by reconstruction error (anomaly score)
5. Validate against known cases (SN2009ip)

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-fetch       # Use cached data
    python main.py --n-models 10      # More ensemble models
    python main.py --n-epochs 500     # More training epochs

Author: Research student project - ML for astrophysics
"""

import argparse
import json
import os
import sys
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supernova Precursor Detection via Anomaly Detection"
    )
    parser.add_argument("--skip-fetch", action="store_true",
                       help="Skip data fetch, use cached data")
    parser.add_argument("--n-models", type=int, default=5,
                       help="Number of models in ensemble (default: 5)")
    parser.add_argument("--n-epochs", type=int, default=300,
                       help="Training epochs per model (default: 300)")
    parser.add_argument("--bottleneck-dim", type=int, default=8,
                       help="Autoencoder bottleneck dimension (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory for data files (default: data)")
    parser.add_argument("--plot-dir", type=str, default="plots",
                       help="Directory for plot output (default: plots)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory for results output (default: outputs)")
    return parser.parse_args()


def set_all_seeds(seed):
    """Set random seeds for full reproducibility."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"All random seeds set to {seed}")


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("SUPERNOVA PRECURSOR DETECTION PIPELINE")
    print("ML-based anomaly detection for pre-explosion activity")
    print("=" * 70)

    # Set seeds for reproducibility
    set_all_seeds(args.seed)

    # Create output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    # STEP 1: DATA ACQUISITION
    # Fetch light curves from the Open Supernova Catalog
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 70)

    from data_acquisition import fetch_all_targets

    if args.skip_fetch:
        cache_file = os.path.join(args.data_dir, "supernovae_cache.json")
        if not os.path.exists(cache_file):
            logger.error("No cached data found! Run without --skip-fetch first.")
            sys.exit(1)
        with open(cache_file) as f:
            supernovae_data = json.load(f)
        logger.info(f"Loaded {len(supernovae_data)} supernovae from cache")
    else:
        supernovae_data = fetch_all_targets(cache_dir=args.data_dir)

    print(f"\nLoaded {len(supernovae_data)} supernovae:")
    type_counts = {}
    for sn in supernovae_data:
        t = sn["claimed_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        n_points = len(sn["photometry"])
        marker = " <-- KEY VALIDATION CASE" if "2009ip" in sn["name"].lower() else ""
        print(f"  {sn['name']:15s} Type {t:5s} - {n_points:5d} data points{marker}")

    print(f"\nType distribution: {type_counts}")

    if len(supernovae_data) < 10:
        logger.warning("Very few supernovae loaded. Results may be unreliable.")

    # Check that SN2009ip is in the data
    has_2009ip = any("2009ip" in sn["name"].lower() for sn in supernovae_data)
    if not has_2009ip:
        logger.warning("SN2009ip not found in data! Validation will be limited.")

    # ==================================================================
    # STEP 2: FEATURE EXTRACTION
    # Convert light curves to statistical feature vectors
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE EXTRACTION")
    print("=" * 70)

    from feature_extraction import extract_all_features

    feature_matrix, names, types, feature_names = extract_all_features(supernovae_data)

    print(f"\nExtracted {len(feature_names)} features from {len(names)} supernovae")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"\nFeatures extracted:")
    for i, fn in enumerate(feature_names):
        vals = feature_matrix[:, i]
        print(f"  {fn:30s}: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}")

    # ==================================================================
    # STEP 3: MODEL TRAINING
    # Train ensemble of autoencoders
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)

    import torch
    from training import train_ensemble, compute_anomaly_scores

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Training {args.n_models} models for {args.n_epochs} epochs each")
    logger.info(f"Bottleneck dimension: {args.bottleneck_dim}")

    ensemble, scaler, data_tensor, all_loss_histories = train_ensemble(
        feature_matrix,
        n_models=args.n_models,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
    )

    # Print final losses
    print("\nFinal training losses:")
    for i, history in enumerate(all_loss_histories):
        print(f"  Model {i+1}: {history[-1]:.6f}")

    # ==================================================================
    # STEP 4: ANOMALY SCORING & WATCHLIST
    # Compute reconstruction errors and rank supernovae
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4: ANOMALY SCORING & WATCHLIST")
    print("=" * 70)

    results = compute_anomaly_scores(ensemble, data_tensor, names, types)

    # Print the full watchlist
    from validation import print_watchlist, validate_results, print_report
    print_watchlist(results)

    # ==================================================================
    # STEP 5: VALIDATION
    # Check if results match scientific expectations
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 5: VALIDATION")
    print("=" * 70)

    report = validate_results(results)
    print_report(report)

    # ==================================================================
    # STEP 6: VISUALIZATION
    # Generate plots
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZATION")
    print("=" * 70)

    from visualization import generate_all_plots

    top_features = generate_all_plots(
        results, all_loss_histories, ensemble, data_tensor,
        feature_names, save_dir=args.plot_dir
    )

    print("\nTop 10 most important features for anomaly detection:")
    for fname, importance in top_features[:10]:
        print(f"  {fname:30s}: {importance:.6f}")

    # ==================================================================
    # STEP 7: SAVE RESULTS
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)

    # Save the full results
    output = {
        "config": {
            "n_models": args.n_models,
            "n_epochs": args.n_epochs,
            "bottleneck_dim": args.bottleneck_dim,
            "seed": args.seed,
            "n_features": len(feature_names),
            "n_supernovae": len(results),
        },
        "feature_names": feature_names,
        "watchlist": results,
        "validation": report,
        "top_features": [{"name": f, "importance": imp} for f, imp in top_features],
    }

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {output_file}")

    # Save model
    model_file = os.path.join(args.output_dir, "ensemble_models.pt")
    torch.save({
        "models": [m.state_dict() for m in ensemble.get_models()],
        "n_features": len(feature_names),
        "bottleneck_dim": args.bottleneck_dim,
        "feature_names": feature_names,
        "scaler_center": scaler.center_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, model_file)
    logger.info(f"Saved models to {model_file}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Supernovae analyzed: {len(results)}")
    print(f"Features extracted: {len(feature_names)}")
    print(f"Ensemble models: {args.n_models}")
    print(f"Validation: {report['passed']} passed, {report['warnings']} warnings, "
          f"{report['failed']} failed")

    # Quick summary of top candidates
    print("\nTOP 5 ANOMALOUS SUPERNOVAE (highest priority for follow-up):")
    for r in results[:5]:
        marker = " *** KNOWN PRECURSOR" if "2009ip" in r["name"].lower() else ""
        print(f"  #{r['rank']} {r['name']} (Type {r['type']}) - "
              f"score: {r['anomaly_score']:.6f}{marker}")

    print(f"\nOutput files:")
    print(f"  Results:  {output_file}")
    print(f"  Models:   {model_file}")
    print(f"  Plots:    {args.plot_dir}/")
    print(f"    - training_curves.png")
    print(f"    - anomaly_watchlist.png")
    print(f"    - type_comparison.png")
    print(f"    - score_distribution.png")
    print(f"    - feature_importance.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
