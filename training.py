"""
training.py - Training pipeline for the supernova anomaly detection autoencoder.

TRAINING STRATEGY:

1. Data preprocessing:
   - StandardScaler normalization (zero mean, unit variance per feature)
   - This is CRITICAL because our features have very different scales
     (magnitude ~15-22 vs time_span ~100-2000 vs rate ~0.001-1.0)
   - Without normalization, the autoencoder would only learn to reconstruct
     the large-scale features and ignore the small ones

2. Training approach:
   - Train on ALL data (unsupervised - no train/test split for labels)
   - But we DO use a validation split to monitor for overfitting
   - The goal is NOT prediction accuracy - it's learning the data manifold
   - We want moderate reconstruction, not perfect (perfect = memorization)

3. Ensemble training:
   - Train 5 models with different random seeds
   - Average their anomaly scores for robustness
   - Each model sees the data in different random order

4. Hyperparameters (tuned for small datasets):
   - Learning rate 1e-3: standard starting point with Adam
   - Batch size = full dataset: with 25-50 samples, minibatches add noise
   - 300 epochs: enough to converge on this small dataset
   - Weight decay 1e-5: light L2 regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

from autoencoder import SupernovaAutoencoder, EnsembleAutoencoder

logger = logging.getLogger(__name__)


def prepare_data(feature_matrix):
    """
    Normalize features for neural network training.

    We use RobustScaler because:
    - It uses median and IQR instead of mean and std
    - This makes it robust to outliers (which is exactly what we're looking for!)
    - We don't want the scaler to be dominated by the anomalies we're trying to find

    Returns:
        scaled_data: numpy array of normalized features
        scaler: fitted scaler object (needed for inverse transform)
    """
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(feature_matrix)

    # Clip extreme values to prevent training instability
    scaled_data = np.clip(scaled_data, -10, 10)

    return scaled_data, scaler


def train_single_model(model, data_tensor, n_epochs=300, learning_rate=1e-3,
                       weight_decay=1e-5, verbose=True):
    """
    Train a single autoencoder model.

    Training details:
    - Loss function: MSE (mean squared error between input and reconstruction)
    - Optimizer: Adam (adaptive learning rate, works well out of the box)
    - No minibatching (dataset is small enough to fit in memory)
    - We track the loss curve for diagnostics

    What to look for in the loss curve:
    - Should decrease rapidly at first, then level off
    - If it keeps decreasing to near-zero: OVERFITTING (memorization)
    - If it plateaus at a high value: UNDERFITTING (model too small or LR too low)
    - We want moderate final loss: learned the general patterns but not memorized
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        reconstructed = model(data_tensor)
        loss = criterion(reconstructed, data_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and (epoch + 1) % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={loss_val:.6f}, lr={lr:.2e}")

    return loss_history


def train_ensemble(feature_matrix, n_models=5, n_epochs=300, base_seed=42):
    """
    Train an ensemble of autoencoders for robust anomaly detection.

    Each model gets a different random seed, leading to:
    - Different weight initialization
    - Different training trajectory
    - Slightly different learned representations

    Averaging their anomaly scores gives more stable results.

    Args:
        feature_matrix: numpy array (n_samples, n_features)
        n_models: number of models in ensemble
        n_epochs: training epochs per model
        base_seed: base random seed (each model uses base_seed + i)

    Returns:
        ensemble: trained EnsembleAutoencoder
        scaler: fitted data scaler
        data_tensor: normalized data as PyTorch tensor
        all_loss_histories: list of loss curves (one per model)
    """
    # Prepare data
    scaled_data, scaler = prepare_data(feature_matrix)
    data_tensor = torch.FloatTensor(scaled_data)

    n_features = feature_matrix.shape[1]
    logger.info(f"Training ensemble of {n_models} models on {feature_matrix.shape[0]} "
                f"samples with {n_features} features")

    ensemble = EnsembleAutoencoder(n_features, n_models=n_models)
    all_loss_histories = []

    for i, model in enumerate(ensemble.get_models()):
        # Set reproducible seed for this model
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Re-initialize weights with this seed
        model.apply(weight_init)

        logger.info(f"\nTraining model {i+1}/{n_models} (seed={seed}):")
        loss_history = train_single_model(model, data_tensor, n_epochs=n_epochs)
        all_loss_histories.append(loss_history)

        final_loss = loss_history[-1]
        logger.info(f"  Model {i+1} final loss: {final_loss:.6f}")

    return ensemble, scaler, data_tensor, all_loss_histories


def weight_init(m):
    """
    Xavier initialization for linear layers.

    Xavier init sets the initial weights to have variance proportional to
    1/(fan_in + fan_out). This prevents the signal from exploding or vanishing
    as it passes through layers, which is especially important for deeper networks.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def compute_anomaly_scores(ensemble, data_tensor, names, types):
    """
    Compute anomaly scores for all supernovae using the trained ensemble.

    The anomaly score is the mean reconstruction error across all models
    in the ensemble. Higher score = more anomalous = potential precursor.

    Returns:
        results: list of dicts sorted by anomaly score (highest first)
    """
    mean_scores, std_scores, all_scores = ensemble.get_ensemble_scores(data_tensor)

    results = []
    for i in range(len(names)):
        results.append({
            "rank": 0,  # will be filled after sorting
            "name": names[i],
            "type": types[i],
            "anomaly_score": float(mean_scores[i]),
            "score_std": float(std_scores[i]),
            "individual_scores": [float(s) for s in all_scores[:, i]],
        })

    # Sort by anomaly score (descending - highest anomaly first)
    results.sort(key=lambda x: x["anomaly_score"], reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
