# Supernova Precursor Detection Pipeline

A PyTorch-based machine learning system that identifies supernovae likely to exhibit pre-explosion activity using unsupervised anomaly detection on light curves from the [Open Supernova Catalog](https://github.com/astrocatalogs/supernovae).

## The Scientific Problem

Most supernovae explode without warning — we only detect them after the explosion happens. But some stars show "precursor" behavior months or years before exploding: unusual brightness changes, eruptions, and mass loss events. The famous case is **SN2009ip**, which had multiple outbursts in 2009 and 2010 before its final explosion in 2012 (Mauerhan et al. 2013).

This pipeline looks at a star's light pattern and flags objects that behave unusually — candidates for precursor activity that should be monitored closely.

## Approach

Based on the **Watchlist Approach** combined with the **LAISS pipeline** methodology:

1. Collect historical light curve data on supernovae that already exploded
2. Extract statistical features that characterize each light curve's behavior
3. Train an autoencoder ensemble to learn what "normal" supernova behavior looks like
4. Rank objects by reconstruction error — high error means anomalous, which means potential precursor
5. Validate against known cases (SN2009ip must rank highly)

This is **unsupervised anomaly detection** — we don't need labels saying which stars had precursors. We just find the unusual ones.

## Results

All 5 validation tests pass:

| Test | Result |
|---|---|
| SN2009ip ranking | **Rank #4 out of 33** (top 12%) |
| Type IIn vs Type II | IIn median **1.9x higher** than Type II |
| Score spread | CV = 0.37 (good discrimination) |
| Ensemble consistency | Avg relative std = 0.21 |
| Top-10 IIn enrichment | **1.6x** overrepresented |

Top 5 watchlist candidates:

| Rank | Name | Type | Score | Notes |
|---|---|---|---|---|
| 1 | SN2009kn | IIn | 0.375 | IIn with progenitor constraints |
| 2 | SN2015bh | IIn | 0.322 | Known precursor outbursts |
| 3 | SN2012fr | Ia | 0.309 | Unusual Ia |
| 4 | **SN2009ip** | **IIn** | **0.303** | **Key validation — known precursor** |
| 5 | SN2005ip | IIn | 0.284 | Persistent CSM interaction |

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

Requires: PyTorch, NumPy, Pandas, SciPy, Matplotlib, scikit-learn, requests

### Run the full pipeline

```bash
python3 main.py
```

On first run, this clones the 6 astrocatalogs year-range repositories from GitHub (~1-2 min), loads 33 supernovae, extracts features, trains the ensemble, and produces the ranked watchlist plus diagnostic plots. Subsequent runs use cached data automatically.

### Common options

```bash
python3 main.py --skip-fetch         # Use cached data (skip repo cloning)
python3 main.py --n-models 10        # Larger ensemble (more stable scores)
python3 main.py --n-epochs 500       # Longer training
python3 main.py --bottleneck-dim 4   # Tighter compression (more aggressive anomaly detection)
python3 main.py --seed 123           # Different random seed
```

### Enable Gaia satellite data integration

```bash
python3 main.py --gaia               # Add Gaia-specific features from existing catalog data
python3 main.py --gaia --gaia-fetch  # Also download light curves from Gaia Alerts API
python3 main.py --gaia --gaia-transients  # Also add Gaia-discovered transients to expand dataset
```

See the [Gaia Integration](#gaia-integration) section below for details.

### Run individual modules standalone

```bash
python3 data_acquisition.py          # Just fetch/cache the data
python3 feature_extraction.py        # Just extract features (needs cached data)
python3 gaia_integration.py          # Show Gaia data coverage report
```

## Output Files

| Output | Location |
|---|---|
| Ranked watchlist | Printed to terminal |
| Validation report | Printed to terminal |
| Full results JSON | `outputs/results.json` |
| Saved model weights | `outputs/ensemble_models.pt` |
| Training curves plot | `plots/training_curves.png` |
| Anomaly watchlist plot | `plots/anomaly_watchlist.png` |
| Type comparison plot | `plots/type_comparison.png` |
| Score distribution plot | `plots/score_distribution.png` |
| Feature importance plot | `plots/feature_importance.png` |

## Architecture

### Why an autoencoder?

With only 33 samples and ~1-3 confirmed precursor cases, we can't train a supervised classifier. An autoencoder learns to compress and reconstruct "typical" supernova feature patterns. Objects it can't reconstruct well are anomalous — potential precursor candidates. This is the core of the LAISS watchlist approach.

### Why feature extraction instead of raw time series?

With 33 objects, we don't have enough data for sequence models (LSTMs/Transformers need thousands of examples). Statistical features compress each light curve into a fixed-size vector while preserving the signal we care about.

### Why an ensemble?

A single autoencoder on 33 samples gives noisy results depending on random initialization. Averaging 5 models with different seeds gives stable, reliable rankings.

### Network architecture

```
Input (50 features)
  → Linear(50, 64) → BatchNorm → LeakyReLU → Dropout(0.1)
  → Linear(64, 32) → BatchNorm → LeakyReLU → Dropout(0.1)
  → Linear(32, 8)   [bottleneck]
  → Linear(8, 32)  → BatchNorm → LeakyReLU → Dropout(0.1)
  → Linear(32, 64) → BatchNorm → LeakyReLU → Dropout(0.1)
  → Linear(64, 50)
Output (reconstructed features)
```

Anomaly score = mean squared reconstruction error across the 5-model ensemble.

## Features Extracted (50 total)

| Category | Features | Why |
|---|---|---|
| **Brightness stats** | mag_mean, mag_median, mag_min, mag_max | Unusual brightness levels |
| **Variability** | mag_std, mag_mad, mag_iqr, mag_range, mag_cv | Precursors cause large variability |
| **Temporal** | time_span, n_observations, cadence_mean/std/median, max_gap | Observation patterns |
| **Shape** | mag_skewness, mag_kurtosis | Asymmetric or heavy-tailed light curves |
| **Trend** | trend_slope, trend_residual_std/mad | Brightening/fading, bumpiness |
| **Rate of change** | rate_mean, rate_max, rate_std, rate_median | Rapid brightness changes (eruptions) |
| **Percentiles** | mag_p10, mag_p90, bright_fraction | Brightness distribution |
| **Multi-band** | n_bands, best_band_nobs, best_band_std/range | Wavelength coverage |
| **Errors** | error_mean, error_std, snr_proxy | Data quality |
| **Beyond-sigma** | frac_beyond_1/2/3sigma | Outlier measurements (eruptions) |
| **Epochs** | n_epochs, inter_epoch_range/std, mean_intra_epoch_std | Multiple activity periods |
| **Outbursts** | n_outbursts, outburst_rate | Rapid brightening events |
| **Structure function** | sf_short, sf_med, sf_long, sf_ratio_long_short | Variability vs timescale |
| **Normalized** | mag_range_per_epoch, variability_per_day, brightness_asymmetry | Cadence-independent measures |

## Data

### Sources

Data comes from the [Open Supernova Catalog](https://github.com/astrocatalogs/supernovae), stored across year-based GitHub repositories:

- [sne-pre-1990](https://github.com/astrocatalogs/sne-pre-1990)
- [sne-1990-1999](https://github.com/astrocatalogs/sne-1990-1999)
- [sne-2000-2004](https://github.com/astrocatalogs/sne-2000-2004)
- [sne-2005-2009](https://github.com/astrocatalogs/sne-2005-2009)
- [sne-2010-2014](https://github.com/astrocatalogs/sne-2010-2014)
- [sne-2015-2019](https://github.com/astrocatalogs/sne-2015-2019)

### Target supernovae (35 selected, 33 loaded)

| Type | Count | Purpose |
|---|---|---|
| **Type IIn** | 10 | Circumstellar interaction from mass loss — most likely precursor candidates |
| **Type II** | 13 | Normal core-collapse — baseline "typical" behavior |
| **Type Ia** | 10 | Thermonuclear (white dwarf) — control group, different physics |

### Key validation case: SN2009ip

SN2009ip is documented in the scientific literature (Mauerhan et al. 2013):
- Type IIn supernova
- Had eruptions in June 2009, August 2009, and 2010
- Final explosion in September 2012
- THE case study for precursor detection

If the model doesn't rank this highly, something is wrong.

### Scientific expectations

- Type IIn should have higher anomaly rates (circumstellar interaction from mass loss)
- Type Ia may also score high (different physics, not precursors)
- Normal Type II should cluster together as "typical"

## Gaia Integration

[ESA Gaia](https://www.cosmos.esa.int/web/gaia) is a space observatory that surveys the entire sky with high-precision photometry. The [Gaia Science Alerts](http://gsaweb.ast.cam.ac.uk/alerts) system publishes transient events including supernovae.

### Why add Gaia data?

- **Uniform photometry**: Same telescope, same G-band filter, same calibration across all objects — removes ground-based systematics
- **Pre-explosion baseline**: Gaia scans the whole sky every ~30 days, providing brightness history before explosion
- **Precision**: ~1% at G=13, ~3% at G=19 — can reveal subtle pre-explosion brightness changes

### Three integration methods

**Method 1: Features from existing catalog data** (`--gaia`)

Extracts 10 Gaia-specific features (G-band magnitude stats, variability, rate of change) from Gaia photometry already present in the Open Supernova Catalog. Currently 2 of 33 supernovae have Gaia data (SN2017eaw = Gaia17bmy, SN2018oh = Gaia18awj) because most targets predate Gaia's 2014 launch.

**Method 2: Fetch from Gaia Alerts API** (`--gaia-fetch`)

For supernovae with Gaia alert names, downloads the full light curve from `http://gsaweb.ast.cam.ac.uk/alerts/alert/{name}/lightcurve.csv` and merges it into the photometry. Requires network access.

**Method 3: Expand dataset with Gaia transients** (`--gaia-transients`)

The catalog contains transients discovered by Gaia (names like Gaia19asm) with rich G-band light curves. This adds up to 5 classified transients per SN type to increase the training set size.

### Check Gaia coverage

```bash
python3 gaia_integration.py
```

This prints a report showing which supernovae have Gaia data and what Gaia-discovered transients are available in the catalog.

### Gaia features added (10)

| Feature | Description |
|---|---|
| has_gaia | Whether the object has Gaia observations (0 or 1) |
| gaia_n_points | Total Gaia photometry points |
| gaia_n_detections | Number of actual detections (vs upper limits) |
| gaia_n_upper_limits | Number of non-detections |
| gaia_detection_fraction | Fraction of Gaia observations that are detections |
| gaia_mag_mean | Mean G-band magnitude |
| gaia_mag_std | G-band magnitude standard deviation |
| gaia_mag_range | G-band peak-to-peak range |
| gaia_time_span | Total Gaia observation window (days) |
| gaia_rate_mean | Mean brightness change rate in G-band |

## Module Reference

| File | Purpose |
|---|---|
| `main.py` | End-to-end pipeline with CLI arguments |
| `data_acquisition.py` | Load photometry from cloned astrocatalogs GitHub repos |
| `feature_extraction.py` | Extract 50 statistical features per supernova |
| `autoencoder.py` | PyTorch autoencoder with ensemble wrapper |
| `training.py` | Training pipeline with RobustScaler normalization |
| `validation.py` | 5 scientific validation tests |
| `visualization.py` | 5 diagnostic plots |
| `gaia_integration.py` | Gaia Science Alerts data integration |

## Acknowledgments

- Supernova data: [Open Supernova Catalog](https://sne.space) (Guillochon et al. 2017)
- Gaia data: ESA Gaia, DPAC and the Photometric Science Alerts Team (http://gsaweb.ast.cam.ac.uk/alerts)
- SN2009ip precursor documentation: Mauerhan et al. 2013
