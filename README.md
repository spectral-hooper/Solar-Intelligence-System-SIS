# Hinode SP Sigma-V Analyzer — AI-First Architecture

A pipeline for analyzing **Hinode Solar Optical Telescope Spectro-Polarimeter (SOT/SP)** data. Computes line-of-sight magnetic fields from Stokes V profiles using the Zeeman effect, with an AI dispatcher at the core that routes every pixel to the most appropriate physics engine.

---

## How It Works

The **Solar Intelligence System (SIS)** evaluates each Stokes V profile *before* any physics computation and classifies it into one of three signal classes:

| Class | Trigger | Route |
|---|---|---|
| **Clear** | Standard anti-symmetric Zeeman profile, SNR > 3 | Sigma-V (Zeeman splitting) |
| **Noisy** | Weak or low-SNR signal | WFA fallback → MC brute-force (500 iter.) if needed |
| **Anomaly** | 3+ peaks, extreme asymmetry, or pathological skewness | Archived for manual scientific review |

The SIS also outputs a rapid **B-field initial guess** (`AI_B_guess`) via a Random Forest Regressor — useful as a seed for external Milne-Eddington inversions.

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
```

Key columns added to the CSV by the AI dispatcher:

| Column | Description |
|---|---|
| `AI_Signal_Class` | `Clear`, `Noisy`, or `Anomaly` |
| `AI_Confidence` | Classifier confidence (0–100%) |
| `AI_B_guess` | Rapid B estimate in Gauss (ME inversion seed) |
| `AI_Route` | `SigmaV` / `WFA_Noisy` / `MC_BruteForce` / `Anomaly_Flagged` |

---

## Protected Physics Functions

The following functions are mathematically unchanged from the original implementation and must not be modified:

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
