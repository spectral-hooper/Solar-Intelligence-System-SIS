# Hinode SP Sigma-V Analyzer — AI-First Architecture

A pipeline for analyzing **Hinode Solar Optical Telescope Spectro-Polarimeter (SOT/SP)** data. Computes line-of-sight magnetic fields from Stokes V profiles using the Zeeman effect, with an AI dispatcher at the core that routes every pixel to the most appropriate physics engine.

---

## How It Works

The **Solar Intelligence System (SIS)** evaluates each Stokes V profile *before* any physics computation and classifies it into one of three signal classes.

Peak detection runs on a Savitzky-Golay smoothed profile using a **noise-aware prominence threshold** — `max(4σ_noise, 0.20 × max_amp)` — where `σ_noise` is estimated exclusively from off-lobe spectral pixels (outside ±`SEARCH_WINDOW_HALF_PIX` around line centre). SNR is computed from the same off-lobe region. This prevents noise ripples from being counted as sigma-component peaks and producing false Anomaly classifications.

| Class | Trigger | Route |
|---|---|---|
| **Clear** | Exactly 1 positive + 1 negative lobe on opposite sides of line centre, SNR > 3 | Sigma-V (Zeeman splitting) |
| **Noisy** | Weak or low-SNR signal, or no resolvable lobe pair | WFA fallback → MC brute-force (500 iter.) if needed |
| **Anomaly** | ≥2 peaks on the same polarity side (total ≥ 3) at SNR > 5, OR asymmetry index > 0.80 at SNR > 6, OR \|skewness\| > 4.0 at SNR > 5 | Archived for manual scientific review |

The SIS also outputs a rapid **B-field initial guess** (`AI_B_guess`) via a Random Forest Regressor — useful as a seed for external Milne-Eddington inversions.

> **Important:** delete `sis_classifier.joblib`, `sis_regressor.joblib`, and `sis_scaler.joblib` before the first run on any new dataset. Models trained on previous data will produce stale classifications.

---

## Requirements

```
Python 3.9+
numpy, scipy, astropy, pandas, matplotlib
scikit-learn   ← mandatory (the AI dispatcher cannot run without it)
joblib
ollama         ← optional, for AI-generated scientific reports
```

Install dependencies:

```bash
pip install numpy scipy astropy pandas scikit-learn matplotlib
pip install ollama  # optional
```

---

## Quick Start

```bash
# Basic run
python ai_enhanced_analyzer_v2.py --fits your_data.fits

# Custom output prefix
python ai_enhanced_analyzer_v2.py --fits your_data.fits --out nov2023

# Higher Monte Carlo accuracy
python ai_enhanced_analyzer_v2.py --fits your_data.fits --mc 300

# Atlas calibration + Milne-Eddington input prep
python ai_enhanced_analyzer_v2.py --fits your_data.fits --atlas --run_me --out results
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--fits` | `SP3D...fits` | Path to input FITS file |
| `--out` | `solar_results` | Output filename prefix |
| `--mc` | `100` | Monte Carlo iterations for error estimation |
| `--atlas` | off | Enable cross-correlation atlas calibration |
| `--run_me` | off | Prepare Milne-Eddington inversion input files |

> **Note:** There is no `--no-ai` flag. The SIS dispatcher is the mandatory core engine.

---

## Output Files

```
{prefix}_results.csv                # Per-slit results with AI routing columns
{prefix}_B_profile.png              # Magnetic field profile
{prefix}_B_sigma_profile.png        # MC uncertainty profile
{prefix}_routing_distribution.png   # Bar chart of SIS dispatch decisions
{prefix}_examples/                  # Per-slit diagnostic plots
{prefix}_anomalies/                 # Archived anomalous Stokes V profiles (.npy)
{prefix}_AI_SCIENTIFIC_REPORT.txt   # LLM report (requires Ollama)
sis_classifier.joblib               # Trained RF classifier (auto-reloaded)
sis_regressor.joblib                # B_guess RF regressor (auto-reloaded)
sis_scaler.joblib                   # Feature scaler (auto-reloaded)
```

Key columns added to the CSV by the AI dispatcher:

| Column | Description |
|---|---|
| `AI_Signal_Class` | `Clear`, `Noisy`, or `Anomaly` |
| `AI_Confidence` | Classifier confidence (0–100%) |
| `AI_B_guess` | Rapid B estimate in Gauss (ME inversion seed) |
| `AI_Route` | `SigmaV` / `WFA_Noisy` / `MC_BruteForce` / `Anomaly_Flagged` |

The training summary printed at startup includes per-class percentages and warns explicitly if any class has zero samples — a reliable indicator that the noise floor or prominence thresholds need adjustment for the current dataset.

---

## Classification Thresholds

| Rule | Threshold | Physical basis |
|---|---|---|
| Prominence | `max(4 × σ_off-lobe, 0.20 × max_amp)` | Clears noise floor; only real sigma-component lobes survive |
| SNR denominator | MAD of off-lobe pixels only | Excludes signal lobes from noise estimate |
| Anomaly — multi-lobe | total ≥ 3 peaks AND (n⁺ ≥ 2 OR n⁻ ≥ 2), SNR > 5 | A simple 1+1 antisymmetric Zeeman pair cannot trigger this |
| Anomaly — asymmetry | asymmetry index > 0.80, SNR > 6 | Genuine single-polarity profiles; ordinary lobe imbalance ≤ 30 % stays below 0.80 |
| Anomaly — skewness | \|skewness\| > 4.0, SNR > 5 | Zeeman profiles with realistic noise reach \|skew\| ≈ 2–2.5 at most |
| Clear | SNR > 3, opposite-side 1+1 lobe pair | Standard antisymmetric Zeeman signature |

---

## Protected Physics Functions

The following functions are mathematically unchanged and must not be modified:

- `analyze_sigma_v_on_spectrum` — Zeeman sigma-component detection
- `estimate_b_error_mc` — Monte Carlo error estimation
- `cross_calibrate_wavelength` — Wavelength calibration
- `robust_find_line_center` — Line centre detection
- `solve_linear_wavelength_scale` — Linear scale solver
- `calculate_parabolic_centroid` — Sub-pixel peak refinement

---

## Physical Parameters

| Parameter | Value |
|---|---|
| Reference line | Fe I 6302.5 Å |
| Calibration lines | Fe I 6301.5 Å, 6302.5 Å |
| Effective Landé g-factor | 2.5 |
| Zeeman constant K | 4.67 × 10⁻¹³ |
| Max realistic B | 5000 G |
| Smoothing window | Savitzky-Golay, 9 px, order 3 |
| Sigma search window | ±14 px around line centre |
