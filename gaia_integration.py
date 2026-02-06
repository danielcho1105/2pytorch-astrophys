"""
gaia_integration.py - Integrate Gaia Science Alerts data into the supernova pipeline.

WHAT IS GAIA?
Gaia is ESA's space observatory that surveys the entire sky repeatedly with
high-precision photometry. Since ~2014, the Gaia Science Alerts system
(http://gsaweb.ast.cam.ac.uk/alerts) publishes transient events - including
supernovae - with light curves in the Gaia G-band.

WHY ADD GAIA DATA?
1. UNIFORM PHOTOMETRY: Gaia observes every object the same way (same telescope,
   same filter, same calibration). Ground-based data mixes different telescopes,
   filters, and conditions. Gaia gives us an apples-to-apples comparison.

2. PRE-EXPLOSION BASELINE: Gaia scans the whole sky every ~30 days. For objects
   it was already tracking before they exploded, we get a pre-explosion brightness
   history - exactly what we need for precursor detection.

3. PRECISE MAGNITUDES: Gaia achieves ~1% photometry at G=13 and ~3% at G=19.
   This precision can reveal subtle brightness changes before explosion.

THREE WAYS TO ADD GAIA DATA:

Method 1: USE EXISTING GAIA DATA IN THE CATALOG (easiest, done here)
   Some supernovae in the Open Supernova Catalog already have Gaia photometry
   included (tagged with band='G', telescope='Gaia'). We extract Gaia-specific
   features from these measurements.

Method 2: FETCH FROM GAIA ALERTS API (requires network access)
   For supernovae with Gaia alert names (e.g., SN2017eaw = Gaia17bmy), we can
   download light curves from:
     http://gsaweb.ast.cam.ac.uk/alerts/alert/{GaiaName}/lightcurve.csv

Method 3: ADD GAIA-DISCOVERED TRANSIENTS TO THE SAMPLE (expands dataset)
   The catalog contains hundreds of transients discovered by Gaia (names like
   Gaia19asm) with rich G-band light curves. Adding classified ones increases
   our training set.

USAGE:
   This module provides functions for all three methods. Import and call from
   main.py, or run standalone to explore Gaia coverage.
"""

import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Gaia Alerts base URL for light curve downloads
GAIA_ALERTS_BASE = "http://gsaweb.ast.cam.ac.uk/alerts/alert"

# Directory where cloned catalog repos live
REPO_BASE = "/tmp"
REPO_NAMES = [
    "sne-pre-1990", "sne-1990-1999", "sne-2000-2004",
    "sne-2005-2009", "sne-2010-2014", "sne-2015-2019",
]


# ======================================================================
# METHOD 1: Extract Gaia features from existing catalog data
# ======================================================================

def extract_gaia_features(sn_data):
    """
    Extract Gaia-specific features from a supernova's photometry.

    Gaia observations are identified by band='G' or telescope containing 'Gaia'.
    We compute features that specifically leverage the uniform Gaia photometry.

    Returns:
        dict of feature_name -> value, or None if no Gaia data
    """
    gaia_points = []
    for p in sn_data["photometry"]:
        band = p.get("band", "")
        # Identify Gaia data points
        if band == "G":
            try:
                t = float(p["time"])
                m = float(p["magnitude"])
                is_upper = p.get("upperlimit", False)
                gaia_points.append({
                    "time": t, "magnitude": m,
                    "upperlimit": is_upper
                })
            except (ValueError, TypeError, KeyError):
                continue

    features = {}
    features["gaia_n_points"] = len(gaia_points)
    features["has_gaia"] = 1.0 if len(gaia_points) >= 3 else 0.0

    if len(gaia_points) < 3:
        # Not enough Gaia data - fill with zeros
        features["gaia_mag_mean"] = 0.0
        features["gaia_mag_std"] = 0.0
        features["gaia_mag_range"] = 0.0
        features["gaia_time_span"] = 0.0
        features["gaia_rate_mean"] = 0.0
        features["gaia_n_detections"] = 0.0
        features["gaia_n_upper_limits"] = 0.0
        features["gaia_detection_fraction"] = 0.0
        return features

    # Separate detections from upper limits
    detections = [p for p in gaia_points if not p.get("upperlimit", False)]
    upper_limits = [p for p in gaia_points if p.get("upperlimit", False)]

    features["gaia_n_detections"] = len(detections)
    features["gaia_n_upper_limits"] = len(upper_limits)
    features["gaia_detection_fraction"] = len(detections) / len(gaia_points)

    if len(detections) >= 2:
        times = np.array([p["time"] for p in detections])
        mags = np.array([p["magnitude"] for p in detections])
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        mags = mags[sort_idx]

        features["gaia_mag_mean"] = np.mean(mags)
        features["gaia_mag_std"] = np.std(mags)
        features["gaia_mag_range"] = np.ptp(mags)
        features["gaia_time_span"] = times[-1] - times[0]

        # Rate of change in Gaia G-band
        dt = np.diff(times)
        dm = np.diff(mags)
        valid = dt > 0
        if np.any(valid):
            rates = np.abs(dm[valid] / dt[valid])
            features["gaia_rate_mean"] = np.mean(rates)
        else:
            features["gaia_rate_mean"] = 0.0
    else:
        features["gaia_mag_mean"] = 0.0
        features["gaia_mag_std"] = 0.0
        features["gaia_mag_range"] = 0.0
        features["gaia_time_span"] = 0.0
        features["gaia_rate_mean"] = 0.0

    return features


def add_gaia_features_to_matrix(feature_matrix, feature_names, supernovae_data, names):
    """
    Augment the existing feature matrix with Gaia-specific features.

    Args:
        feature_matrix: numpy array (n_samples, n_features)
        feature_names: list of existing feature names
        supernovae_data: raw supernova data list
        names: list of supernova names (matching rows of feature_matrix)

    Returns:
        augmented_matrix: numpy array with Gaia features appended
        augmented_names: updated feature names list
    """
    # Build name -> data lookup
    sn_lookup = {sn["name"]: sn for sn in supernovae_data}

    gaia_feature_rows = []
    gaia_feature_names = None

    for name in names:
        sn = sn_lookup.get(name)
        if sn:
            gf = extract_gaia_features(sn)
        else:
            gf = extract_gaia_features({"photometry": []})

        if gaia_feature_names is None:
            gaia_feature_names = sorted(gf.keys())

        row = [gf[fn] for fn in gaia_feature_names]
        gaia_feature_rows.append(row)

    gaia_matrix = np.array(gaia_feature_rows)
    n_with_gaia = np.sum(gaia_matrix[:, gaia_feature_names.index("has_gaia")] > 0)
    logger.info(f"Gaia features: {n_with_gaia}/{len(names)} supernovae have Gaia data")

    augmented_matrix = np.hstack([feature_matrix, gaia_matrix])
    augmented_names = feature_names + gaia_feature_names

    return augmented_matrix, augmented_names


# ======================================================================
# METHOD 2: Fetch light curves from Gaia Alerts API
# ======================================================================

def get_gaia_alert_name(sn_name):
    """
    Look up the Gaia alert name for a supernova from its catalog JSON.

    Some supernovae have Gaia alert designations stored as aliases,
    e.g., SN2017eaw = Gaia17bmy.

    Returns:
        Gaia alert name (str) or None
    """
    for repo_name in REPO_NAMES:
        path = os.path.join(REPO_BASE, repo_name, f"{sn_name}.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            key = list(data.keys())[0]
            aliases = data[key].get("alias", [])
            for alias in aliases:
                if isinstance(alias, dict):
                    val = alias.get("value", "")
                    if val.startswith("Gaia"):
                        return val
        except (json.JSONDecodeError, OSError):
            pass
    return None


def fetch_gaia_lightcurve(gaia_name, cache_dir="data/gaia"):
    """
    Download a Gaia Alerts light curve CSV for a given alert name.

    The Gaia Alerts system provides light curves at:
      http://gsaweb.ast.cam.ac.uk/alerts/alert/{GaiaName}/lightcurve.csv

    CSV columns: Date(TCB), JD(TCB), averagemag(in Gaia G-band)

    Args:
        gaia_name: e.g., "Gaia17bmy"
        cache_dir: directory to cache downloaded CSVs

    Returns:
        list of {"time": MJD, "magnitude": G-mag} dicts, or None
    """
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{gaia_name}.csv")

    # Check cache
    if os.path.exists(cache_file):
        return _parse_gaia_csv(cache_file)

    url = f"{GAIA_ALERTS_BASE}/{gaia_name}/lightcurve.csv"
    logger.info(f"Fetching Gaia light curve: {url}")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            logger.warning(f"Gaia fetch failed for {gaia_name}: HTTP {response.status_code}")
            return None

        with open(cache_file, "w") as f:
            f.write(response.text)

        return _parse_gaia_csv(cache_file)

    except requests.exceptions.RequestException as e:
        logger.warning(f"Gaia fetch failed for {gaia_name}: {e}")
        return None


def _parse_gaia_csv(filepath):
    """
    Parse a Gaia Alerts lightcurve.csv file.

    Format: Date(TCB), JD(TCB), averagemag
    JD is Julian Date in TCB (Barycentric Coordinate Time).
    We convert to MJD for consistency with the rest of our pipeline.
    """
    points = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Date") or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        jd = float(parts[1].strip())
                        mag_str = parts[2].strip()
                        if mag_str and mag_str.lower() not in ("null", "nan", ""):
                            mag = float(mag_str)
                            # Convert JD to MJD (MJD = JD - 2400000.5)
                            mjd = jd - 2400000.5
                            points.append({"time": mjd, "magnitude": mag, "band": "G"})
                    except ValueError:
                        continue
    except OSError:
        return None

    return points if points else None


def merge_gaia_into_photometry(sn_data, gaia_points):
    """
    Merge downloaded Gaia photometry into an existing supernova's data.

    Avoids duplicates by checking for time overlaps (within 0.1 day).
    """
    existing_times = set()
    for p in sn_data["photometry"]:
        if p.get("band") == "G":
            existing_times.add(round(p["time"], 1))

    added = 0
    for gp in gaia_points:
        if round(gp["time"], 1) not in existing_times:
            sn_data["photometry"].append({
                "time": gp["time"],
                "magnitude": gp["magnitude"],
                "band": "G",
                "e_magnitude": None,
                "source": "gaia_alerts",
            })
            added += 1

    logger.info(f"  Merged {added} new Gaia points (skipped {len(gaia_points) - added} duplicates)")
    return sn_data


# ======================================================================
# METHOD 3: Add Gaia-discovered transients to expand dataset
# ======================================================================

def find_gaia_transients(min_points=30, type_filter=None):
    """
    Search the catalog repos for transients discovered by Gaia that have
    typed classifications and sufficient photometry.

    These can be added to the training set to increase sample size.

    Args:
        min_points: minimum number of photometry points required
        type_filter: if set, only return transients of this type (e.g., "IIn")

    Returns:
        list of dicts: {"name", "gaia_name", "type", "n_photometry", "n_gaia"}
    """
    results = []
    for repo_name in REPO_NAMES:
        repo_path = os.path.join(REPO_BASE, repo_name)
        if not os.path.isdir(repo_path):
            continue

        for fname in os.listdir(repo_path):
            if not fname.endswith(".json"):
                continue
            if not fname.startswith("Gaia"):
                continue

            filepath = os.path.join(repo_path, fname)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                key = list(data.keys())[0]
                obj = data[key]

                # Check type
                ct = obj.get("claimedtype", [])
                if not ct:
                    continue
                sn_type = ct[0].get("value", "?") if isinstance(ct[0], dict) else "?"
                if sn_type == "?" or sn_type == "Candidate":
                    continue

                if type_filter and type_filter not in sn_type:
                    continue

                # Check photometry
                n_phot = len(obj.get("photometry", []))
                if n_phot < min_points:
                    continue

                n_gaia = sum(1 for p in obj.get("photometry", [])
                            if p.get("band") == "G" or "Gaia" in str(p.get("telescope", "")))

                results.append({
                    "name": key,
                    "gaia_name": fname.replace(".json", ""),
                    "type": sn_type,
                    "n_photometry": n_phot,
                    "n_gaia": n_gaia,
                })
            except (json.JSONDecodeError, OSError, IndexError):
                continue

    results.sort(key=lambda x: x["n_photometry"], reverse=True)
    return results


def load_gaia_transient(name):
    """
    Load a Gaia-discovered transient from the catalog repos.

    Returns same format as data_acquisition.load_supernova_from_file().
    """
    from data_acquisition import load_supernova_from_file
    return load_supernova_from_file(name)


# ======================================================================
# HIGH-LEVEL INTEGRATION FUNCTION
# ======================================================================

def integrate_gaia_data(supernovae_data, feature_matrix, feature_names, names,
                        fetch_alerts=False, add_transients=False,
                        transient_types=None):
    """
    One-call function to add Gaia data to the pipeline.

    Three levels of integration (controlled by arguments):

    1. ALWAYS: Extract Gaia-specific features from existing catalog data.
       This adds ~10 new features from G-band photometry already in the catalog.

    2. fetch_alerts=True: Download additional Gaia Alerts light curves for
       supernovae that have Gaia alert names. Requires network access.

    3. add_transients=True: Expand the dataset with Gaia-discovered transients
       that have typed classifications. Increases sample size.

    Args:
        supernovae_data: raw data list from data_acquisition
        feature_matrix: numpy array from feature_extraction
        feature_names: list of feature names
        names: list of supernova names
        fetch_alerts: whether to fetch Gaia Alerts light curves (needs network)
        add_transients: whether to add Gaia transients to the dataset
        transient_types: list of SN types to add (e.g., ["IIn", "Ia"])

    Returns:
        updated (feature_matrix, feature_names, supernovae_data, names, types)
    """
    logger.info("=" * 50)
    logger.info("GAIA DATA INTEGRATION")
    logger.info("=" * 50)

    types = [sn["claimed_type"] for sn in supernovae_data
             if sn["name"] in names]

    # --- Step 1: Extract Gaia features from existing data ---
    logger.info("Step 1: Extracting Gaia features from existing catalog data...")
    feature_matrix, feature_names = add_gaia_features_to_matrix(
        feature_matrix, feature_names, supernovae_data, names
    )

    # --- Step 2: Optionally fetch Gaia Alerts light curves ---
    if fetch_alerts:
        logger.info("Step 2: Fetching Gaia Alerts light curves...")
        for sn in supernovae_data:
            gaia_name = get_gaia_alert_name(sn["name"])
            if gaia_name:
                logger.info(f"  {sn['name']} -> {gaia_name}")
                gaia_points = fetch_gaia_lightcurve(gaia_name)
                if gaia_points:
                    merge_gaia_into_photometry(sn, gaia_points)
    else:
        logger.info("Step 2: Skipping Gaia Alerts fetch (use --gaia-fetch to enable)")

    # --- Step 3: Optionally add Gaia transients ---
    if add_transients:
        logger.info("Step 3: Searching for Gaia transients to add to dataset...")
        if transient_types is None:
            transient_types = ["IIn", "Ia", "II"]

        for sn_type in transient_types:
            transients = find_gaia_transients(min_points=30, type_filter=sn_type)
            logger.info(f"  Found {len(transients)} Gaia transients of type {sn_type}")

            # Add top 5 per type (don't overwhelm the dataset)
            for t in transients[:5]:
                if t["name"] in names:
                    continue
                sn = load_gaia_transient(t["name"])
                if sn:
                    supernovae_data.append(sn)
                    names.append(sn["name"])
                    types.append(sn["claimed_type"])
                    logger.info(f"    Added {t['name']} (Type {t['type']}, "
                              f"{t['n_photometry']} points, {t['n_gaia']} Gaia)")
    else:
        logger.info("Step 3: Skipping Gaia transient expansion "
                    "(use --gaia-transients to enable)")

    logger.info(f"Gaia integration complete. Feature matrix: {feature_matrix.shape}")
    return feature_matrix, feature_names, supernovae_data, names, types


# ======================================================================
# STANDALONE USAGE
# ======================================================================

if __name__ == "__main__":
    """Run standalone to explore Gaia data coverage."""

    # Load cached data
    cache_file = "data/supernovae_cache.json"
    if not os.path.exists(cache_file):
        print("No cached data. Run main.py first.")
        exit(1)

    with open(cache_file) as f:
        data = json.load(f)

    print("=" * 70)
    print("GAIA DATA COVERAGE REPORT")
    print("=" * 70)

    print("\n--- Gaia data in current supernovae ---")
    for sn in data:
        gf = extract_gaia_features(sn)
        gaia_name = get_gaia_alert_name(sn["name"])
        alias_str = f" = {gaia_name}" if gaia_name else ""
        if gf["gaia_n_points"] > 0:
            print(f"  {sn['name']}{alias_str}: {gf['gaia_n_points']} Gaia pts, "
                  f"G-mag range {gf['gaia_mag_range']:.2f}, "
                  f"detections={gf['gaia_n_detections']}")
        else:
            print(f"  {sn['name']}{alias_str}: no Gaia data")

    print("\n--- Available Gaia transients in catalog ---")
    for sn_type in ["IIn", "Ia", "II"]:
        transients = find_gaia_transients(min_points=30, type_filter=sn_type)
        print(f"\n  Type {sn_type}: {len(transients)} classified Gaia transients")
        for t in transients[:5]:
            print(f"    {t['name']:25s} {t['n_photometry']:5d} pts "
                  f"({t['n_gaia']} Gaia)")
