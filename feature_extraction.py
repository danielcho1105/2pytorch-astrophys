"""
feature_extraction.py - Extract statistical features from supernova light curves.

WHY FEATURE EXTRACTION INSTEAD OF RAW TIME SERIES?
With only 25-50 objects, we don't have enough data to train a deep sequence model
(like an LSTM or Transformer) that would process raw light curves. Those models
need thousands of examples to learn temporal patterns.

Instead, we compute statistical features that summarize each light curve's behavior.
This is the approach used in the LAISS pipeline and is well-suited to small datasets.

WHAT FEATURES AND WHY:
Each feature captures a different aspect of the light curve that might indicate
unusual behavior:

1. Brightness statistics (mean, median, min, max magnitude):
   - Unusually bright or faint objects stand out
   - Precursor events make objects temporarily brighter

2. Variability measures (std, MAD, IQR, range):
   - Objects with precursors have MORE variability than typical SNe
   - A star erupting before explosion shows large magnitude swings

3. Temporal statistics (observation span, cadence):
   - Objects observed for longer may show pre-explosion activity
   - Irregular cadence can indicate unusual follow-up (astronomers noticed something)

4. Shape statistics (skewness, kurtosis):
   - Skewness: asymmetric light curves (fast rise, slow decline vs. opposite)
   - Kurtosis: heavy-tailed distributions indicate outlier measurements (eruptions!)

5. Trend measures (slope, residual variance):
   - Linear trend captures whether object is brightening or fading over time
   - High residual variance = the light curve is bumpy, not smooth = possible eruptions

6. Rate of change statistics:
   - How fast does brightness change between observations?
   - Precursor eruptions cause rapid brightness changes

7. Multi-band statistics:
   - How many filters was the object observed in?
   - Color information (brightness difference between bands) can indicate unusual physics
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_features_single(sn_data):
    """
    Extract features from a single supernova's light curve data.

    We work primarily in the V/R/visual bands because:
    - Most complete coverage across different supernovae
    - Closest to what the human eye sees
    - Most standardized across different telescopes

    But we also compute aggregate features across ALL bands.

    Args:
        sn_data: dict with 'name', 'claimed_type', 'photometry'
                 photometry is list of {time, magnitude, band, e_magnitude}

    Returns:
        dict of feature_name -> feature_value
    """
    photometry = sn_data["photometry"]

    # Separate into bands
    times_all = []
    mags_all = []
    errors_all = []
    bands_seen = set()

    band_data = {}  # band -> (times, mags)

    for point in photometry:
        t = point["time"]
        m = point["magnitude"]
        b = point.get("band", "unknown")

        if t is None or m is None:
            continue
        if not np.isfinite(t) or not np.isfinite(m):
            continue

        times_all.append(t)
        mags_all.append(m)
        bands_seen.add(b)

        e = point.get("e_magnitude")
        if e is not None and np.isfinite(e):
            errors_all.append(e)

        if b not in band_data:
            band_data[b] = {"times": [], "mags": []}
        band_data[b]["times"].append(t)
        band_data[b]["mags"].append(m)

    if len(times_all) < 5:
        return None

    times_all = np.array(times_all)
    mags_all = np.array(mags_all)

    # Sort by time
    sort_idx = np.argsort(times_all)
    times_all = times_all[sort_idx]
    mags_all = mags_all[sort_idx]

    features = {}

    # ====== 1. BRIGHTNESS STATISTICS ======
    # Remember: in astronomy, LOWER magnitude = BRIGHTER
    features["mag_mean"] = np.mean(mags_all)
    features["mag_median"] = np.median(mags_all)
    features["mag_min"] = np.min(mags_all)       # brightest point
    features["mag_max"] = np.max(mags_all)       # faintest point

    # ====== 2. VARIABILITY MEASURES ======
    features["mag_std"] = np.std(mags_all)
    features["mag_mad"] = np.median(np.abs(mags_all - np.median(mags_all)))  # median absolute deviation
    features["mag_iqr"] = np.percentile(mags_all, 75) - np.percentile(mags_all, 25)
    features["mag_range"] = np.ptp(mags_all)  # peak-to-peak range

    # Coefficient of variation (normalized variability)
    if features["mag_mean"] != 0:
        features["mag_cv"] = features["mag_std"] / abs(features["mag_mean"])
    else:
        features["mag_cv"] = 0.0

    # ====== 3. TEMPORAL STATISTICS ======
    features["time_span"] = times_all[-1] - times_all[0]  # total observation window in days
    features["n_observations"] = len(times_all)

    # Time gaps between observations
    dt = np.diff(times_all)
    dt = dt[dt > 0]  # remove zero gaps (simultaneous observations)
    if len(dt) > 0:
        features["cadence_mean"] = np.mean(dt)
        features["cadence_std"] = np.std(dt)
        features["cadence_median"] = np.median(dt)
        features["max_gap"] = np.max(dt)
    else:
        features["cadence_mean"] = 0.0
        features["cadence_std"] = 0.0
        features["cadence_median"] = 0.0
        features["max_gap"] = 0.0

    # ====== 4. SHAPE STATISTICS ======
    if features["mag_std"] > 0:
        features["mag_skewness"] = float(
            np.mean(((mags_all - np.mean(mags_all)) / np.std(mags_all)) ** 3)
        )
        features["mag_kurtosis"] = float(
            np.mean(((mags_all - np.mean(mags_all)) / np.std(mags_all)) ** 4) - 3
        )
    else:
        features["mag_skewness"] = 0.0
        features["mag_kurtosis"] = 0.0

    # ====== 5. TREND MEASURES ======
    # Fit a linear trend to magnitude vs time
    if features["time_span"] > 0:
        # Normalize time to [0, 1] for numerical stability
        t_norm = (times_all - times_all[0]) / features["time_span"]
        # Linear regression: mag = slope * t_norm + intercept
        coeffs = np.polyfit(t_norm, mags_all, 1)
        features["trend_slope"] = coeffs[0]  # positive = fading, negative = brightening
        # Residual variance (how much the light curve deviates from the trend)
        predicted = np.polyval(coeffs, t_norm)
        residuals = mags_all - predicted
        features["trend_residual_std"] = np.std(residuals)
        features["trend_residual_mad"] = np.median(np.abs(residuals - np.median(residuals)))
    else:
        features["trend_slope"] = 0.0
        features["trend_residual_std"] = 0.0
        features["trend_residual_mad"] = 0.0

    # ====== 6. RATE OF CHANGE STATISTICS ======
    # How fast does magnitude change between consecutive observations?
    dt_raw = np.diff(times_all)
    dm_raw = np.diff(mags_all)
    # Only keep pairs where time actually advanced (remove simultaneous observations)
    valid_mask = dt_raw > 0
    dt_valid = dt_raw[valid_mask]
    dm_valid = dm_raw[valid_mask]
    if len(dt_valid) > 0:
        rates = dm_valid / dt_valid  # magnitudes per day
        features["rate_mean"] = np.mean(np.abs(rates))
        features["rate_max"] = np.max(np.abs(rates))
        features["rate_std"] = np.std(rates)
        features["rate_median"] = np.median(np.abs(rates))
    else:
        features["rate_mean"] = 0.0
        features["rate_max"] = 0.0
        features["rate_std"] = 0.0
        features["rate_median"] = 0.0

    # ====== 7. PERCENTILE-BASED FEATURES ======
    features["mag_p10"] = np.percentile(mags_all, 10)
    features["mag_p90"] = np.percentile(mags_all, 90)
    features["bright_fraction"] = np.mean(mags_all < features["mag_median"] - features["mag_std"])

    # ====== 8. MULTI-BAND STATISTICS ======
    features["n_bands"] = len(bands_seen)

    # Find the "best" band (most observations) for per-band features
    best_band = max(band_data, key=lambda b: len(band_data[b]["times"]))
    best_mags = np.array(band_data[best_band]["mags"])
    best_times = np.array(band_data[best_band]["times"])
    features["best_band_nobs"] = len(best_mags)

    # Per-band variability (if we have enough data in the best band)
    if len(best_mags) >= 5:
        features["best_band_std"] = np.std(best_mags)
        features["best_band_range"] = np.ptp(best_mags)
    else:
        features["best_band_std"] = features["mag_std"]
        features["best_band_range"] = features["mag_range"]

    # ====== 9. ERROR-BASED FEATURES ======
    if len(errors_all) > 0:
        errors_arr = np.array(errors_all)
        features["error_mean"] = np.mean(errors_arr)
        features["error_std"] = np.std(errors_arr)
        # Signal-to-noise proxy
        features["snr_proxy"] = features["mag_std"] / (features["error_mean"] + 1e-10)
    else:
        features["error_mean"] = 0.0
        features["error_std"] = 0.0
        features["snr_proxy"] = 0.0

    # ====== 10. BEYOND-N-SIGMA FEATURES ======
    # What fraction of points are >2 sigma from the mean?
    # High values indicate unusual variability / outlier measurements
    if features["mag_std"] > 0:
        z_scores = np.abs(mags_all - features["mag_mean"]) / features["mag_std"]
        features["frac_beyond_1sigma"] = np.mean(z_scores > 1)
        features["frac_beyond_2sigma"] = np.mean(z_scores > 2)
        features["frac_beyond_3sigma"] = np.mean(z_scores > 3)
    else:
        features["frac_beyond_1sigma"] = 0.0
        features["frac_beyond_2sigma"] = 0.0
        features["frac_beyond_3sigma"] = 0.0

    # ====== 11. EPOCH / OUTBURST DETECTION ======
    # Precursor supernovae show MULTIPLE distinct activity episodes.
    # We detect "epochs" by finding gaps > 30 days in observations,
    # then measure variability across and within epochs.
    if features["time_span"] > 30:
        # Split observations into epochs based on gaps > 30 days
        dt_all = np.diff(times_all)
        gap_threshold = 30.0  # days
        epoch_breaks = np.where(dt_all > gap_threshold)[0]
        n_epochs = len(epoch_breaks) + 1
        features["n_epochs"] = n_epochs

        # Compute median magnitude per epoch and measure inter-epoch variability
        epoch_starts = [0] + list(epoch_breaks + 1)
        epoch_ends = list(epoch_breaks + 1) + [len(mags_all)]
        epoch_medians = []
        epoch_stds = []
        for start, end in zip(epoch_starts, epoch_ends):
            if end - start >= 2:
                epoch_medians.append(np.median(mags_all[start:end]))
                epoch_stds.append(np.std(mags_all[start:end]))

        if len(epoch_medians) >= 2:
            features["inter_epoch_range"] = np.ptp(epoch_medians)
            features["inter_epoch_std"] = np.std(epoch_medians)
            features["mean_intra_epoch_std"] = np.mean(epoch_stds)
        else:
            features["inter_epoch_range"] = 0.0
            features["inter_epoch_std"] = 0.0
            features["mean_intra_epoch_std"] = features["mag_std"]

        # Count "outbursts" - rapid brightening events (mag decrease > 1.0)
        n_outbursts = 0
        for i in range(1, len(mags_all)):
            if (mags_all[i] - mags_all[i-1]) < -1.0 and dt_all[i-1] < 30:
                n_outbursts += 1
        features["n_outbursts"] = n_outbursts
        features["outburst_rate"] = n_outbursts / (features["time_span"] / 365.25)
    else:
        features["n_epochs"] = 1
        features["inter_epoch_range"] = 0.0
        features["inter_epoch_std"] = 0.0
        features["mean_intra_epoch_std"] = features["mag_std"]
        features["n_outbursts"] = 0
        features["outburst_rate"] = 0.0

    # ====== 12. STRUCTURE FUNCTION (variability vs timescale) ======
    # This captures how variability changes with time delay.
    # Precursors show excess variability at intermediate timescales.
    if len(times_all) >= 10:
        # Sample pairs of observations to estimate structure function
        n_pairs = min(5000, len(times_all) * (len(times_all) - 1) // 2)
        rng = np.random.RandomState(42)
        idx1 = rng.randint(0, len(times_all), n_pairs)
        idx2 = rng.randint(0, len(times_all), n_pairs)
        valid = idx1 != idx2
        idx1, idx2 = idx1[valid], idx2[valid]

        dt_pairs = np.abs(times_all[idx1] - times_all[idx2])
        dm_pairs = np.abs(mags_all[idx1] - mags_all[idx2])

        # Structure function at different timescales
        for label, lo, hi in [("short", 0, 10), ("med", 10, 100), ("long", 100, 1000)]:
            mask = (dt_pairs >= lo) & (dt_pairs < hi)
            if np.sum(mask) > 5:
                features[f"sf_{label}"] = np.median(dm_pairs[mask])
            else:
                features[f"sf_{label}"] = 0.0

        # Ratio of long-term to short-term variability
        if features["sf_short"] > 0:
            features["sf_ratio_long_short"] = features["sf_long"] / features["sf_short"]
        else:
            features["sf_ratio_long_short"] = 0.0
    else:
        features["sf_short"] = 0.0
        features["sf_med"] = 0.0
        features["sf_long"] = 0.0
        features["sf_ratio_long_short"] = 0.0

    # ====== 13. NORMALIZED VARIABILITY FEATURES ======
    # These remove dependence on observation cadence
    features["mag_range_per_epoch"] = features["mag_range"] / max(features.get("n_epochs", 1), 1)
    features["variability_per_day"] = features["mag_std"] / max(features["time_span"], 1)
    features["brightness_asymmetry"] = (features["mag_mean"] - features["mag_median"]) / max(features["mag_std"], 0.01)

    return features


def extract_all_features(supernovae_data):
    """
    Extract features from all supernovae.

    Args:
        supernovae_data: list of dicts from data_acquisition.fetch_all_targets()

    Returns:
        features_list: list of dicts (one per supernova)
        names: list of supernova names
        types: list of supernova types
        feature_names: list of feature names (consistent ordering)
    """
    features_list = []
    names = []
    types = []

    for sn in supernovae_data:
        feats = extract_features_single(sn)
        if feats is not None:
            features_list.append(feats)
            names.append(sn["name"])
            types.append(sn["claimed_type"])

    if not features_list:
        raise ValueError("No features could be extracted from any supernova!")

    # Get consistent feature ordering
    feature_names = sorted(features_list[0].keys())

    # Convert to numpy array
    feature_matrix = np.array([
        [feats[fn] for fn in feature_names]
        for feats in features_list
    ])

    # Handle NaN/Inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Extracted {len(feature_names)} features from {len(names)} supernovae")
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")

    return feature_matrix, names, types, feature_names


if __name__ == "__main__":
    import json
    import os
    # Quick test with cached data
    cache_file = "data/supernovae_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        features, names, types, feature_names = extract_all_features(data)
        print(f"Features shape: {features.shape}")
        print(f"Feature names: {feature_names}")
    else:
        print("No cached data found. Run data_acquisition.py first.")
