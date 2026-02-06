"""
data_acquisition.py - Load supernova light curves from the Open Supernova Catalog.

Data source: https://github.com/astrocatalogs/supernovae
The catalog stores individual supernova JSON files across year-based repositories:
  - sne-pre-1990, sne-1990-1999, sne-2000-2004, sne-2005-2009,
    sne-2010-2014, sne-2015-2019

Each JSON file contains photometry (light curve data): time series of
(time, magnitude, filter) measurements describing brightness over time.

This module supports two data loading strategies:
1. LOCAL: Read from cloned git repos (fast, works offline)
2. API: Fetch from the REST API at api.astrocats.space (requires network)

WHY THESE TARGETS:
- Type IIn: Known for circumstellar interaction (mass loss before explosion).
  These are the most likely to show precursor activity. SN2009ip is our key case.
- Type II: "Normal" core-collapse supernovae. Good baseline for typical behavior.
- Type Ia: Thermonuclear explosions of white dwarfs. Different physics entirely.
  They SHOULDN'T have precursors (no massive star wind), so they serve as a
  control group.
"""

import json
import os
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where cloned repos are stored
REPO_BASE = "/tmp"

# Mapping of year-range repo names
REPO_NAMES = [
    "sne-pre-1990",
    "sne-1990-1999",
    "sne-2000-2004",
    "sne-2005-2009",
    "sne-2010-2014",
    "sne-2015-2019",
]

GITHUB_BASE = "https://github.com/astrocatalogs"

# Our target supernovae, organized by type.
# These were chosen because they have well-documented light curves in the catalog.
TARGETS = {
    # === TYPE IIn (circumstellar interaction - likely precursor candidates) ===
    # SN2009ip: THE key validation case. Had eruptions in 2009, 2010 before
    # final explosion in 2012 (Mauerhan et al. 2013).
    "SN2009ip": "IIn",
    "SN2010jl": "IIn",     # Very luminous IIn, strong CSM interaction
    "SN2005gl": "IIn",     # Progenitor detected as luminous blue variable
    "SN2006jd": "IIn",     # Long-lasting CSM interaction
    "SN1998S": "IIn",      # Classic IIn with narrow lines
    "SN2005ip": "IIn",     # Persistent CSM interaction
    "SN1988Z": "IIn",      # One of first recognized IIn
    "SN2006gy": "IIn",     # Extremely luminous, possible pair-instability
    "SN2015bh": "IIn",     # Had precursor outbursts like SN2009ip
    "SN2016jbu": "IIn",    # Recent IIn with precursor activity

    # === TYPE II (normal core-collapse - baseline "typical" behavior) ===
    "SN1999em": "II",      # Classic Type II-P (plateau)
    "SN2004et": "II",      # Well-studied II-P
    "SN2005cs": "II",      # Low-luminosity II-P
    "SN2012aw": "II",      # Bright nearby II-P
    "SN2013ej": "II",      # II-L (linear decline)
    "SN2017eaw": "II",     # Recent well-observed II-P
    "SN1987A": "II",       # The famous one! Closest SN in modern era
    "SN2009N": "II",       # Normal II-P
    "SN2012A": "II",       # Normal II-P
    "SN2013ab": "II",      # Normal II-P

    # === TYPE Ia (thermonuclear - control group, different physics) ===
    "SN2011fe": "Ia",      # Closest Ia in decades, extremely well-observed
    "SN2014J": "Ia",       # Nearest Ia in years, in M82
    "SN2005cf": "Ia",      # Well-studied normal Ia
    "SN1994D": "Ia",       # Classic normal Ia
    "SN2003du": "Ia",      # Normal Ia
    "SN2001el": "Ia",      # Normal Ia
    "SN2006X": "Ia",       # Ia with some dust interaction
    "SN2007af": "Ia",      # Normal Ia
    "SN2012fr": "Ia",      # Recent well-observed Ia
    "SN2018oh": "Ia",      # Ia with early excess (interesting case)

    # === ADDITIONAL INTERESTING CASES ===
    "SN2009kn": "IIn",     # IIn with progenitor constraints
    "SN1997cy": "IIn",     # Luminous IIn
    "SN2010mc": "IIn",     # Had documented precursor outburst
    "SN2011ht": "IIn",     # SN impostor / IIn boundary case
    "SN2000ch": "IIn",     # Multiple outbursts observed
}


def ensure_repos_cloned():
    """
    Ensure all year-range repositories are cloned locally.

    The Open Supernova Catalog splits data across multiple repos by discovery year.
    We do shallow clones (--depth 1) to minimize disk usage since we only need
    the current files, not git history.
    """
    for repo_name in REPO_NAMES:
        repo_path = os.path.join(REPO_BASE, repo_name)
        if os.path.isdir(repo_path):
            logger.info(f"Repo {repo_name} already cloned at {repo_path}")
            continue

        url = f"{GITHUB_BASE}/{repo_name}.git"
        logger.info(f"Cloning {url} -> {repo_path} ...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, repo_path],
                check=True, capture_output=True, text=True, timeout=300
            )
            logger.info(f"Successfully cloned {repo_name}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Failed to clone {repo_name}: {e}")


def find_json_file(name):
    """
    Find the JSON file for a supernova across all year-range repos.

    Returns the file path if found, None otherwise.
    """
    for repo_name in REPO_NAMES:
        path = os.path.join(REPO_BASE, repo_name, f"{name}.json")
        if os.path.isfile(path):
            return path
    return None


def load_supernova_from_file(name):
    """
    Load photometry data for a single supernova from its JSON file.

    The JSON structure (from the Open Supernova Catalog) is:
    {
        "SN_NAME": {
            "photometry": [
                {"time": "55000.5", "magnitude": "16.2", "band": "V",
                 "e_magnitude": "0.05", "u_time": "MJD", ...},
                ...
            ],
            "claimedtype": [{"value": "IIn", "source": "..."}],
            ...
        }
    }

    Returns:
        dict with keys: name, claimed_type, photometry (list of dicts)
        None if load fails or not enough data
    """
    filepath = find_json_file(name)
    if filepath is None:
        logger.warning(f"{name}: JSON file not found in any repo")
        return None

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"{name}: failed to read {filepath}: {e}")
        return None

    # The top-level key is the supernova name
    if name not in data:
        # Try first key
        keys = list(data.keys())
        if keys:
            name_key = keys[0]
        else:
            logger.warning(f"{name}: empty JSON file")
            return None
    else:
        name_key = name

    sn_obj = data[name_key]
    photometry_raw = sn_obj.get("photometry", [])

    if not photometry_raw:
        logger.warning(f"{name}: no photometry data in file")
        return None

    # Determine the claimed type from the JSON, fall back to our TARGETS dict
    claimed_type = TARGETS.get(name, "unknown")
    ct_field = sn_obj.get("claimedtype", [])
    if ct_field and isinstance(ct_field, list):
        # Use the first claimed type value
        for ct in ct_field:
            if isinstance(ct, dict) and "value" in ct:
                val = ct["value"]
                # Map to our simplified categories
                if "IIn" in val:
                    claimed_type = "IIn"
                elif "Ia" in val and "II" not in val:
                    claimed_type = "Ia"
                elif "II" in val:
                    claimed_type = "II"
                break

    # Parse photometry into clean format
    photometry = []
    for point in photometry_raw:
        try:
            # Time field
            t_val = point.get("time")
            if t_val is None:
                continue
            t = float(t_val)

            # Magnitude field
            m_val = point.get("magnitude")
            if m_val is None:
                continue
            m = float(m_val)

            # Band field (optional)
            band = point.get("band", "unknown")

            entry = {"time": t, "magnitude": m, "band": band}

            # Error field (optional)
            e_val = point.get("e_magnitude")
            if e_val is not None:
                try:
                    entry["e_magnitude"] = float(e_val)
                except (ValueError, TypeError):
                    entry["e_magnitude"] = None
            else:
                entry["e_magnitude"] = None

            photometry.append(entry)

        except (TypeError, ValueError):
            continue

    if len(photometry) < 5:
        logger.warning(f"{name}: too few valid data points ({len(photometry)})")
        return None

    logger.info(f"{name}: loaded {len(photometry)} photometry points from {os.path.basename(filepath)}")
    return {
        "name": name,
        "claimed_type": claimed_type,
        "photometry": photometry,
    }


def fetch_all_targets(cache_dir="data"):
    """
    Load data for all target supernovae, with local caching.

    Strategy:
    1. Check for a cached JSON file first (fastest)
    2. If no cache, ensure git repos are cloned
    3. Load from the local JSON files
    4. Save to cache for next run

    Returns:
        list of dicts, each with name, claimed_type, photometry
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "supernovae_cache.json")

    # Check cache first
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        with open(cache_file, "r") as f:
            cached = json.load(f)
        if len(cached) > 0:
            logger.info(f"Loaded {len(cached)} supernovae from cache")
            return cached

    # Ensure repos are cloned
    ensure_repos_cloned()

    # Load from local files
    results = []
    for name in TARGETS:
        sn_data = load_supernova_from_file(name)
        if sn_data is not None:
            results.append(sn_data)

    logger.info(f"Successfully loaded {len(results)} / {len(TARGETS)} supernovae")

    # Save cache
    if results:
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Cached data to {cache_file}")

    return results


if __name__ == "__main__":
    data = fetch_all_targets()
    print(f"\nLoaded {len(data)} supernovae:")
    for sn in data:
        print(f"  {sn['name']:15s} Type {sn['claimed_type']:5s} "
              f"- {len(sn['photometry'])} data points")
