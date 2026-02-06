"""
autoencoder.py - PyTorch autoencoder for supernova anomaly detection.

WHY AN AUTOENCODER FOR THIS PROBLEM?

An autoencoder is a neural network that learns to compress data into a small
representation and then reconstruct it. The key insight for anomaly detection:

1. TRAINING: We train the autoencoder on ALL our data. It learns to reconstruct
   the "typical" patterns in the feature space.

2. INFERENCE: For each object, we measure how well the autoencoder reconstructs
   its features. Objects that are poorly reconstructed = ANOMALOUS.

Why this works for precursor detection:
- Most supernovae follow relatively predictable patterns
- Objects with precursor activity have unusual feature combinations
  (high variability + long time span + rapid brightness changes)
- The autoencoder learns the "normal" manifold and flags deviations

WHY NOT SUPERVISED LEARNING?
We only have ~1-3 confirmed precursor cases in our small dataset. That's not
enough to train a classifier. Unsupervised anomaly detection doesn't need labels -
it just finds the weird ones. This is exactly the LAISS approach.

ARCHITECTURE CHOICES:
- Small network (we only have 25-50 samples!)
- Bottleneck (compress ~35 features to 8 dimensions)
- Dropout for regularization (prevents overfitting to small dataset)
- BatchNorm for training stability
- The reconstruction error IS the anomaly score
"""

import torch
import torch.nn as nn
import numpy as np


class SupernovaAutoencoder(nn.Module):
    """
    Autoencoder for supernova light curve feature anomaly detection.

    Architecture:
        Input (n_features) -> 64 -> 32 -> bottleneck (8) -> 32 -> 64 -> Output (n_features)

    The bottleneck forces the network to learn a compressed representation.
    Objects that don't fit well into this compressed space get high reconstruction
    error = high anomaly score.

    Design choices for small datasets:
    - Moderate bottleneck (8 dims): tight enough to force compression,
      loose enough to capture real variation in ~35 features
    - Dropout (0.1): light regularization - too much kills the signal with few samples
    - BatchNorm: stabilizes training with small batches
    - LeakyReLU: avoids dead neurons (better gradient flow than ReLU)
    """

    def __init__(self, n_features, bottleneck_dim=8, dropout_rate=0.1):
        super().__init__()

        self.n_features = n_features
        self.bottleneck_dim = bottleneck_dim

        # === ENCODER ===
        # Progressively compresses the feature space
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(32, bottleneck_dim),
            # No activation on bottleneck - allow full range of values
        )

        # === DECODER ===
        # Reconstructs the features from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(64, n_features),
            # No activation on output - we want to reconstruct the full range
        )

    def forward(self, x):
        """Forward pass: encode then decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Get the bottleneck representation (useful for visualization)."""
        return self.encoder(x)

    def get_reconstruction_error(self, x):
        """
        Compute per-sample reconstruction error.

        This IS the anomaly score: higher error = more anomalous = potential precursor.

        We use MSE (mean squared error) per sample, not summed - this makes the
        score independent of the number of features.
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Per-sample MSE
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors


class EnsembleAutoencoder:
    """
    Ensemble of autoencoders for more robust anomaly detection.

    WHY AN ENSEMBLE?
    With only 25-50 samples, a single autoencoder's results can be noisy -
    different random initializations lead to different rankings. By training
    multiple autoencoders and averaging their anomaly scores, we get more
    stable and reliable results.

    This is like asking 5 different experts and taking their consensus opinion.
    """

    def __init__(self, n_features, n_models=5, bottleneck_dim=8, dropout_rate=0.1):
        self.n_models = n_models
        self.models = [
            SupernovaAutoencoder(n_features, bottleneck_dim, dropout_rate)
            for _ in range(n_models)
        ]

    def get_models(self):
        return self.models

    def get_ensemble_scores(self, x):
        """
        Get averaged anomaly scores across all models in the ensemble.

        Returns both individual and averaged scores so you can check consistency.
        """
        all_scores = []
        for model in self.models:
            scores = model.get_reconstruction_error(x)
            all_scores.append(scores)

        all_scores = torch.stack(all_scores)  # (n_models, n_samples)
        mean_scores = torch.mean(all_scores, dim=0)
        std_scores = torch.std(all_scores, dim=0)

        return mean_scores, std_scores, all_scores
