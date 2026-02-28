#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinode SP Sigma-V Analyzer — AI-First Architecture (v2)

Refactored pipeline where the Solar Intelligence System (SIS) is the mandatory
core dispatcher that drives all analysis logic.  The AI evaluates every Stokes V
profile FIRST and routes it to the most appropriate physics engine:

  • "Clear"   → Standard Sigma-V (Zeeman splitting) analysis
  • "Noisy"   → Weak Field Approximation (WFA) OR brute-force Monte Carlo (500 it.)
  • "Anomaly" → Profile flagged for manual scientific review; archived to disk

SIS also provides a rapid B-field initial guess (B_guess) via a RandomForest
Regressor — useful as a seed for external Milne-Eddington inversions.

Critical constraint: the following physics/math functions are UNCHANGED:
  - analyze_sigma_v_on_spectrum
  - estimate_b_error_mc
  - cross_calibrate_wavelength
  - robust_find_line_center
  - solve_linear_wavelength_scale
  - calculate_parabolic_centroid
"""

import os
import time
import argparse
import warnings
from typing import Tuple, Dict, Optional, Union, Any, List

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter, find_peaks, fftconvolve
from scipy.stats import median_abs_deviation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── AI Dependencies ────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    AI_SKLEARN_AVAILABLE = True
except ImportError:
    AI_SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not available. SolarIntelligenceSystem disabled — "
          "analysis cannot run without it.  Install with: pip install scikit-learn")

try:
    import ollama
    AI_OLLAMA_AVAILABLE = True
except ImportError:
    AI_OLLAMA_AVAILABLE = False
    print("[WARN] ollama not available. AI Scientist (Layer 2) disabled.")


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

DEFAULT_FITS_FILE    = "SP3D20231104_210115.0C.fits"
DEFAULT_OUTPUT_PREFIX = "solar_results"

# Analysis Parameters
SPATIAL_AVERAGING_WINDOW = 1
REFERENCE_WAVELENGTH_0   = 6302.5
LINE_LAB_1               = 6301.5
LINE_LAB_2               = 6302.5
LANDÉ_FACTOR_EFF         = 2.5
ZEEMAN_CONSTANT_K        = 4.67e-13

# Signal Processing Parameters
SMOOTHING_WINDOW_SIZE    = 9
SMOOTHING_POLY_ORDER     = 3
SEARCH_WINDOW_HALF_PIX   = 14
SIGMA_TIGHT_WINDOW_PIX   = 8
MIN_RELATIVE_AMPLITUDE   = 0.01
NOISE_THRESHOLD_FACTOR   = 4.0
EDGE_SAFETY_MARGIN_PIX   = 2
MAX_REALISTIC_B_GAUSS    = 5000.0

# Monte Carlo Parameters
DEFAULT_MC_ITERATIONS       = 100
NOISY_BRUTE_FORCE_MC_ITERS  = 500   # High iteration count for "Noisy" routing

# AI SIS Configuration
SIS_CLASSIFIER_PATH  = "sis_classifier.joblib"
SIS_REGRESSOR_PATH   = "sis_regressor.joblib"
SIS_SCALER_PATH      = "sis_scaler.joblib"

# Signal classes (internal integer codes → human labels)
CLASS_NOISY   = 0
CLASS_CLEAR   = 1
CLASS_ANOMALY = 2
CLASS_LABELS  = {CLASS_NOISY: "Noisy", CLASS_CLEAR: "Clear", CLASS_ANOMALY: "Anomaly"}

# AI Route labels (stored in CSV)
ROUTE_SIGMA_V      = "SigmaV"
ROUTE_WFA_NOISY    = "WFA_Noisy"
ROUTE_MC_NOISY     = "MC_BruteForce"
ROUTE_ANOMALY      = "Anomaly_Flagged"

# AI Layer 2 (LLM) Configuration
AI_REPORT_FILENAME = "AI_SCIENTIFIC_REPORT.txt"
AI_OLLAMA_MODEL    = "llama3"

# External Milne-Eddington Solver Configuration
USE_ATLAS_REF              = False
REF_ATLAS_WAV_PATH         = None
REF_ATLAS_INTENSITY_PATH   = None
ME_SOLVER_CMD_TEMPLATE     = None
ME_TEMP_DIR_SUFFIX         = "_ME_input"


# =============================================================================
# AI ENGINE — LAYER 1: SOLAR INTELLIGENCE SYSTEM (SIS)
# =============================================================================

class SolarIntelligenceSystem:
    """
    Mandatory core dispatcher for the Hinode SP analysis pipeline.

    Responsibilities
    ----------------
    1. Extract a rich feature vector from each Stokes V profile.
    2. Classify the profile into one of three signal classes:
         "Clear", "Noisy", or "Anomaly".
    3. Provide a confidence score for that classification.
    4. Provide a rapid B-field initial guess (B_guess) via a Random-Forest
       Regressor — useful as a seed for external Milne-Eddington inversions.
    5. Route the pixel to the appropriate physics engine in the main pipeline.

    Both the classifier and regressor are persisted to disk after on-the-fly
    training and reloaded automatically on subsequent runs.
    """

    def __init__(self,
                 classifier_path: str = SIS_CLASSIFIER_PATH,
                 regressor_path:  str = SIS_REGRESSOR_PATH,
                 scaler_path:     str = SIS_SCALER_PATH):

        self.classifier_path = classifier_path
        self.regressor_path  = regressor_path
        self.scaler_path     = scaler_path

        self.classifier  = None
        self.regressor   = None
        self.scaler      = None
        self.is_trained  = False

        if not AI_SKLEARN_AVAILABLE:
            raise RuntimeError(
                "[SIS] scikit-learn is required.  "
                "Install with: pip install scikit-learn"
            )

        # Attempt to load pre-trained models from disk.
        all_paths = [self.classifier_path, self.regressor_path, self.scaler_path]
        if all(os.path.exists(p) for p in all_paths):
            self._load_models()
        else:
            print("[SIS] No pre-trained models found.  "
                  "Will train on-the-fly on first dataset.")

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_models(self):
        """Load pre-trained classifier, regressor, and feature scaler from disk."""
        try:
            self.classifier = joblib.load(self.classifier_path)
            self.regressor  = joblib.load(self.regressor_path)
            self.scaler     = joblib.load(self.scaler_path)
            self.is_trained = True
            print(f"[SIS] Loaded pre-trained models from disk "
                  f"({self.classifier_path}, {self.regressor_path})")
        except Exception as exc:
            print(f"[SIS] Failed to load models: {exc}. Will retrain.")
            self.classifier = self.regressor = self.scaler = None
            self.is_trained = False

    def _save_models(self):
        """Persist trained classifier, regressor, and scaler to disk."""
        try:
            joblib.dump(self.classifier, self.classifier_path)
            joblib.dump(self.regressor,  self.regressor_path)
            joblib.dump(self.scaler,     self.scaler_path)
            print(f"[SIS] Saved models → {self.classifier_path}, "
                  f"{self.regressor_path}, {self.scaler_path}")
        except Exception as exc:
            print(f"[SIS] Failed to save models: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_features(self,
                          stokes_v:    np.ndarray,
                          wavelengths: np.ndarray,
                          center_idx:  int) -> np.ndarray:
        """
        Extract a 12-dimensional feature vector from a Stokes V profile.

        Features (in order)
        -------------------
        0  max_abs_amplitude     – Peak absolute value
        1  std_dev               – Profile standard deviation
        2  n_total_peaks         – Number of significant peaks (pos + neg)
        3  skewness              – Third standardised moment
        4  kurtosis              – Fourth standardised moment (excess)
        5  mad                   – Median Absolute Deviation
        6  peak_to_peak_dist     – Max distance between any two significant peaks
        7  symmetry_correlation  – Cross-correlation of left/right halves around center
        8  snr_estimate          – Amplitude / MAD noise estimate
        9  n_pos_peaks           – Number of positive peaks
        10 n_neg_peaks           – Number of negative peaks
        11 asymmetry_index       – |max(V) + min(V)| / (|max(V)| + |min(V)|)
                                   Anomalous profiles often lack anti-symmetry.
        """
        from scipy.stats import skew, kurtosis as scipy_kurtosis

        features = np.zeros(12, dtype=float)

        max_amp = np.max(np.abs(stokes_v))
        std_val = np.std(stokes_v)

        # ── Basic stats ──
        features[0] = max_amp
        features[1] = std_val

        # ── Peak structure ──
        prominence_thresh = max(0.1 * max_amp, 1e-12)
        peaks_pos, _ = find_peaks( stokes_v, prominence=prominence_thresh)
        peaks_neg, _ = find_peaks(-stokes_v, prominence=prominence_thresh)
        n_pos = len(peaks_pos)
        n_neg = len(peaks_neg)
        total_peaks = n_pos + n_neg

        features[2]  = total_peaks
        features[3]  = skew(stokes_v)
        features[4]  = scipy_kurtosis(stokes_v)
        features[5]  = median_abs_deviation(stokes_v)

        # ── Peak-to-peak distance ──
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg))
        features[6] = (all_peaks[-1] - all_peaks[0]) if len(all_peaks) >= 2 else 0.0

        # ── Symmetry (anti-symmetry expected for a Zeeman Stokes V) ──
        try:
            left  = stokes_v[:center_idx]
            right = stokes_v[center_idx:]
            min_len = min(len(left), len(right))
            if min_len > 5:
                corr = np.corrcoef(left[-min_len:], right[:min_len][::-1])[0, 1]
                features[7] = 0.0 if np.isnan(corr) else corr
        except Exception:
            features[7] = 0.0

        # ── SNR estimate ──
        mad_noise = float(median_abs_deviation(stokes_v))
        features[8] = max_amp / (mad_noise if mad_noise > 0 else 1e-12)

        features[9]  = float(n_pos)
        features[10] = float(n_neg)

        # ── Asymmetry index ──
        vmax =  float(np.max(stokes_v))
        vmin =  float(np.min(stokes_v))
        denom = abs(vmax) + abs(vmin)
        features[11] = abs(vmax + vmin) / denom if denom > 0 else 0.0

        return features

    # ──────────────────────────────────────────────────────────────────────────
    # Heuristic labelling (used during on-the-fly training)
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_heuristic_3class(self,
                                   stokes_v:   np.ndarray,
                                   center_idx: int) -> int:
        """
        Assign a 3-class heuristic label to a Stokes V profile.

        Decision rules (in priority order)
        ------------------------------------
        1. ANOMALY  — ≥3 distinct peaks (pos or neg) with good SNR, OR
                      strong extreme asymmetry despite good amplitude.
        2. CLEAR    — Exactly 1 positive AND 1 negative peak on OPPOSITE sides
                      of center with SNR > 3.
        3. NOISY    — Everything else.

        Returns
        -------
        int : CLASS_CLEAR (1), CLASS_NOISY (0), or CLASS_ANOMALY (2)
        """
        from scipy.stats import skew as scipy_skew

        max_amp = np.max(np.abs(stokes_v))
        mad_noise = float(median_abs_deviation(stokes_v))
        snr = max_amp / (mad_noise if mad_noise > 0 else 1e-12)

        prom = max(0.15 * max_amp, 1e-12)
        peaks_pos, _ = find_peaks( stokes_v, prominence=prom)
        peaks_neg, _ = find_peaks(-stokes_v, prominence=prom)
        n_pos = len(peaks_pos)
        n_neg = len(peaks_neg)

        # ── Rule 1: Anomaly detection ──────────────────────────────────────
        # (a) Three or more significant extrema at decent SNR
        has_multi_peaks = (n_pos + n_neg) >= 3 and snr > 3.0
        # (b) Extreme amplitude asymmetry: profile is mostly one-sided
        vmax, vmin = float(np.max(stokes_v)), float(np.min(stokes_v))
        denom = abs(vmax) + abs(vmin)
        asym_idx = abs(vmax + vmin) / denom if denom > 0 else 0.0
        has_strong_asymmetry = asym_idx > 0.70 and snr > 4.0
        # (c) Very high skewness while having a signal
        sk = abs(float(scipy_skew(stokes_v)))
        has_extreme_skew = sk > 2.5 and snr > 3.5

        if has_multi_peaks or has_strong_asymmetry or has_extreme_skew:
            return CLASS_ANOMALY

        # ── Rule 2: Clear signal ───────────────────────────────────────────
        if snr > 3.0 and n_pos >= 1 and n_neg >= 1:
            # Verify the dominant positive and negative peaks are on opposite
            # sides of the spectral line center (Zeeman anti-symmetric signature)
            opposite = any(
                (int(p) - center_idx) * (int(n) - center_idx) < 0
                for p in peaks_pos for n in peaks_neg
            )
            if opposite:
                return CLASS_CLEAR

        # ── Rule 3: Noisy/Weak ─────────────────────────────────────────────
        return CLASS_NOISY

    # ──────────────────────────────────────────────────────────────────────────
    # On-the-fly Training
    # ──────────────────────────────────────────────────────────────────────────

    def train_on_the_fly(self,
                         data_cube:   np.ndarray,
                         wavelengths: np.ndarray,
                         center_idx:  int,
                         n_samples:   int = 250):
        """
        Train both the classifier and the B_guess regressor on a sample of the
        current FITS cube, using heuristic labelling + Sigma-V ground truth.

        Classifier training
        -------------------
        All 3 classes (Clear / Noisy / Anomaly) are labelled using the
        heuristic `_classify_heuristic_3class`.

        Regressor training
        ------------------
        Only "Clear"-class samples where Sigma-V returns a valid B_G are used
        as regression training points (features → B_G).  This keeps the
        B_guess physically meaningful.

        Parameters
        ----------
        data_cube   : shape (4, n_slits, n_lambda)
        wavelengths : calibrated wavelength axis
        center_idx  : pixel index of the reference line centre
        n_samples   : number of randomly selected slits to train on
        """
        _, n_slits, _ = data_cube.shape
        sample_indices = np.linspace(0, n_slits - 1,
                                     min(n_samples, n_slits),
                                     dtype=int)

        X_all   = []   # features for all samples  (classifier)
        y_cls   = []   # 3-class labels             (classifier)
        X_reg   = []   # features for Clear samples (regressor)
        y_reg   = []   # B_G ground truth            (regressor)

        print(f"[SIS] Training on {len(sample_indices)} samples — "
              f"building classifier + B_guess regressor …")

        for s_idx in sample_indices:
            stokes_i = data_cube[0, s_idx, :]
            stokes_v = data_cube[3, s_idx, :]

            features = self._extract_features(stokes_v, wavelengths, center_idx)
            label    = self._classify_heuristic_3class(stokes_v, center_idx)

            X_all.append(features)
            y_cls.append(label)

            # ── Ground truth B for regressor (Clear class only) ──────────
            if label == CLASS_CLEAR:
                sigma_res = analyze_sigma_v_on_spectrum(
                    wavelengths, stokes_i, stokes_v, center_idx)
                if sigma_res.get("found") and sigma_res.get("B_G") is not None:
                    X_reg.append(features)
                    y_reg.append(float(sigma_res["B_G"]))

        X_all  = np.array(X_all)
        y_cls  = np.array(y_cls)

        # ── Fit shared feature scaler ──────────────────────────────────────
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)

        # ── Train 3-class Random Forest Classifier ─────────────────────────
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.classifier.fit(X_scaled, y_cls)

        # ── Train B_guess Random Forest Regressor ─────────────────────────
        if len(X_reg) >= 5:
            X_reg_arr = np.array(X_reg)
            y_reg_arr = np.array(y_reg)
            X_reg_scaled = self.scaler.transform(X_reg_arr)
            self.regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.regressor.fit(X_reg_scaled, y_reg_arr)
            print(f"[SIS]   Regressor trained on {len(X_reg)} Clear-class samples.")
        else:
            # Not enough Clear samples: create a dummy regressor returning median
            print("[SIS]   Warning: too few Clear samples for regressor. "
                  "B_guess will be 0.0.")
            self.regressor = None

        self.is_trained = True
        self._save_models()

        # ── Training summary ───────────────────────────────────────────────
        unique, counts = np.unique(y_cls, return_counts=True)
        print("[SIS] Classifier training complete.  Class distribution:")
        for lbl, cnt in zip(unique, counts):
            print(f"       {CLASS_LABELS[lbl]:8s} ({lbl}): {cnt:4d} samples")

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self,
                stokes_v:    np.ndarray,
                wavelengths: np.ndarray,
                center_idx:  int) -> Tuple[str, float, float]:
        """
        Evaluate a single Stokes V profile and return the SIS routing decision.

        Returns
        -------
        signal_class : str
            "Clear", "Noisy", or "Anomaly"
        confidence : float
            Classifier confidence in [0, 100] (percentage)
        b_guess : float
            Rapid regressor B-field estimate in Gauss.
            Returns 0.0 if the regressor is unavailable or the class is Noisy.
        """
        if not self.is_trained:
            raise RuntimeError(
                "[SIS] Models not trained yet.  "
                "Call train_on_the_fly() before running the pipeline.")

        try:
            features        = self._extract_features(stokes_v, wavelengths, center_idx)
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            cls_int    = int(self.classifier.predict(features_scaled)[0])
            probs      = self.classifier.predict_proba(features_scaled)[0]
            confidence = float(np.max(probs)) * 100.0
            label      = CLASS_LABELS.get(cls_int, "Noisy")

            # ── B_guess (only meaningful for Clear predictions) ────────────
            b_guess = 0.0
            if self.regressor is not None and cls_int == CLASS_CLEAR:
                b_guess = float(self.regressor.predict(features_scaled)[0])

            return label, confidence, b_guess

        except Exception as exc:
            print(f"[SIS] Prediction error: {exc}")
            return "Noisy", 0.0, 0.0


# =============================================================================
# AI ENGINE — LAYER 2: LLM SCIENTIFIC INTERPRETATION
# =============================================================================

class AIScientist:
    """
    LLM-powered scientific interpretation of analysis results.
    Now reports AI routing distribution and anomaly findings.
    """

    def __init__(self, model_name: str = AI_OLLAMA_MODEL):
        self.model_name = model_name
        self.available  = AI_OLLAMA_AVAILABLE
        if not self.available:
            print("[AI Scientist] Ollama not available. Layer 2 disabled.")

    def _ollama_running(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def generate_report(self,
                        df_results:    pd.DataFrame,
                        fits_filename: str,
                        output_path:   str = AI_REPORT_FILENAME) -> bool:
        """
        Generate a scientific report from the full results DataFrame.
        Includes anomaly analysis and AI routing distribution.
        """
        if not self.available:
            return False
        if not self._ollama_running():
            print(f"[AI Scientist] Ollama not running.  "
                  f"Start it with: ollama run {self.model_name}")
            return False

        print(f"[AI Scientist] Generating report using {self.model_name} …")

        total = len(df_results)
        valid_b = df_results['B_G'].notna()
        n_valid = int(valid_b.sum())

        if n_valid > 0:
            mean_b = df_results.loc[valid_b, 'B_G'].mean()
            std_b  = df_results.loc[valid_b, 'B_G'].std()
            min_b  = df_results.loc[valid_b, 'B_G'].min()
            max_b  = df_results.loc[valid_b, 'B_G'].max()
        else:
            mean_b = std_b = min_b = max_b = 0.0

        # ── AI routing distribution ────────────────────────────────────────
        route_counts = df_results['AI_Route'].value_counts() \
            if 'AI_Route' in df_results.columns else pd.Series(dtype=int)
        n_sigma   = int(route_counts.get(ROUTE_SIGMA_V,   0))
        n_wfa     = int(route_counts.get(ROUTE_WFA_NOISY, 0))
        n_mc      = int(route_counts.get(ROUTE_MC_NOISY,  0))
        n_anomaly = int(route_counts.get(ROUTE_ANOMALY,   0))

        # ── Signal class distribution ──────────────────────────────────────
        cls_counts = df_results['AI_Signal_Class'].value_counts() \
            if 'AI_Signal_Class' in df_results.columns else pd.Series(dtype=int)
        n_clear   = int(cls_counts.get("Clear",   0))
        n_noisy   = int(cls_counts.get("Noisy",   0))
        n_anom_c  = int(cls_counts.get("Anomaly", 0))

        prompt = f"""You are an expert solar physicist analysing spectropolarimetric
observations from the Hinode spacecraft.

Dataset: {fits_filename}
Total spatial positions analysed: {total}

──────────────────────────────────────────
MAGNETIC FIELD STATISTICS
──────────────────────────────────────────
Valid measurements   : {n_valid}  ({n_valid/total*100:.1f}%)
Mean field strength  : {mean_b:.1f}  Gauss
Standard deviation   : {std_b:.1f}  Gauss
Range                : {min_b:.1f} – {max_b:.1f}  Gauss

──────────────────────────────────────────
AI ROUTING DISTRIBUTION  (SolarIntelligenceSystem dispatcher)
──────────────────────────────────────────
Signal class — Clear    : {n_clear}  ({n_clear/total*100:.1f}%)
Signal class — Noisy    : {n_noisy}  ({n_noisy/total*100:.1f}%)
Signal class — Anomaly  : {n_anom_c}  ({n_anom_c/total*100:.1f}%)

Route → Standard Sigma-V            : {n_sigma}  pixels
Route → WFA (noisy fallback)        : {n_wfa}   pixels
Route → Monte Carlo brute-force     : {n_mc}    pixels
Route → Anomaly archive (flagged)   : {n_anomaly} pixels

──────────────────────────────────────────
ANOMALY ANALYSIS
──────────────────────────────────────────
Total anomalous Stokes V profiles   : {n_anomaly}  ({n_anomaly/total*100:.1f}%)
These profiles exhibited ≥3 significant peaks, extreme amplitude asymmetry,
or pathological skewness inconsistent with a standard Zeeman anti-symmetric
profile.  They have been saved to a separate archive for manual review.

Please provide a professional scientific interpretation covering:
1. Assessment of the observed magnetic field strengths (active region, quiet Sun, sunspot?).
2. Interpretation of the AI routing distribution — what does the Noisy fraction tell us
   about observation quality or the target region?
3. Scientific significance of the {n_anomaly} anomalous Stokes V profiles. What physical
   mechanisms (e.g. unresolved polarity mixing, complex atmospheric gradients, magneto-
   optical effects, instrumental artefacts) could cause these profiles?
4. Comparison of Sigma-V versus WFA results and methodological implications.
5. Recommendations for follow-up analysis (e.g. full Milne-Eddington inversion on the
   anomalous pixels, re-observation strategy).

Write in a professional academic style (300–500 words).
"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system',
                     'content': 'You are an expert solar physicist specialising in '
                                'spectropolarimetry and Stokes-profile inversion.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            report_text = response['message']['content']

            full_report = (
                f"{'='*80}\n"
                f"AUTOMATED SCIENTIFIC ANALYSIS REPORT\n"
                f"Hinode SP Sigma-V Analyzer — AI-First Architecture (v2)\n"
                f"{'='*80}\n\n"
                f"File Analysed : {fits_filename}\n"
                f"Analysis Date : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"LLM Model     : {self.model_name}\n\n"
                f"{'='*80}\n\n"
                f"{report_text}\n\n"
                f"{'='*80}\n"
                f"END OF REPORT\n\n"
                f"Note: This report was generated automatically by an LLM.\n"
                f"Numerical analysis uses established astrophysical methods.\n"
                f"Interpretations should be reviewed by qualified researchers.\n"
                f"{'='*80}\n"
            )

            with open(output_path, 'w', encoding='utf-8') as fh:
                fh.write(full_report)

            print(f"[AI Scientist] Report saved → {output_path}")
            return True

        except Exception as exc:
            print(f"[AI Scientist] Report generation failed: {exc}")
            return False


# =============================================================================
# HELPER FUNCTIONS  (ORIGINAL — UNCHANGED)
# =============================================================================

def ensure_odd(number: int) -> int:
    """Ensures the given number is odd and at least 3 (useful for filter windows)."""
    n = int(number)
    if n % 2 == 0:
        n -= 1
    if n < 3:
        n = 3
    return n


def build_wavelength_axis(header: fits.Header,
                          n_pixels: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Constructs the wavelength array from FITS header WCS keywords.

    Returns
    -------
    wavelengths : np.ndarray
    dispersion  : float   Delta lambda per pixel
    ref_val     : float   Reference wavelength value
    ref_pix     : float   Reference pixel index
    """
    crval = header.get('CRVAL1', header.get('CRVAL', None))
    crpix = header.get('CRPIX1', header.get('CRPIX', 1.0))
    cdelt = header.get('CDELT1', header.get('CD1_1', header.get('CDELT', None)))

    if crval is None or cdelt is None:
        wavelengths = np.linspace(
            REFERENCE_WAVELENGTH_0 - 0.5,
            REFERENCE_WAVELENGTH_0 + 0.5,
            n_pixels)
        cdelt = wavelengths[1] - wavelengths[0]
        crval = wavelengths[0]
        crpix = 1.0
    else:
        pixel_indices = np.arange(n_pixels, dtype=float)
        wavelengths = float(crval) + (pixel_indices + 1.0 - float(crpix)) * float(cdelt)

    return (np.array(wavelengths),
            float(cdelt),
            float(crval if crval is not None else wavelengths[0]),
            float(crpix))


def calculate_mad_std(data: np.ndarray) -> float:
    """Calculates the standard deviation estimate using Median Absolute Deviation (MAD)."""
    if data is None or len(data) == 0:
        return 0.0
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def compute_cross_correlation_shift(observed: np.ndarray,
                                    reference: np.ndarray) -> int:
    """
    Computes the pixel shift between observed and reference arrays
    using FFT cross-correlation.
    """
    obs_centered = np.asarray(observed)  - np.nanmean(observed)
    ref_centered = np.asarray(reference) - np.nanmean(reference)
    correlation  = fftconvolve(obs_centered, ref_centered[::-1], mode='same')
    return int(np.argmax(correlation) - (len(correlation) // 2))


# ── UNCHANGED PHYSICS FUNCTIONS ───────────────────────────────────────────────

def robust_find_line_center(wavelengths: np.ndarray,
                            intensity: np.ndarray,
                            target_lambda: float,
                            window_angstrom: float = 0.6) -> Optional[int]:
    """
    Finds the pixel index of the minimum intensity (line center) near
    a target wavelength.
    """
    n_pixels = len(wavelengths)
    if n_pixels < 3:
        return None

    delta_lambda    = abs(wavelengths[1] - wavelengths[0])
    half_window_pix = max(2, int(round(window_angstrom / delta_lambda)))
    pixel_guess     = int(np.argmin(np.abs(wavelengths - target_lambda)))

    start_idx = max(0, pixel_guess - half_window_pix)
    end_idx   = min(n_pixels, pixel_guess + half_window_pix + 1)

    if end_idx - start_idx < 3:
        return None

    region_intensity = intensity[start_idx:end_idx]
    local_min_idx    = int(np.argmin(region_intensity))
    return start_idx + local_min_idx


def solve_linear_wavelength_scale(observed_positions: List[float],
                                  lab_positions: List[float]) -> Tuple[float, float]:
    """
    Solves for linear scaling coefficients (slope 'a' and intercept 'b')
    such that: lambda_calibrated = a * lambda_raw + b
    """
    obs_arr = np.asarray(observed_positions)
    lab_arr = np.asarray(lab_positions)

    if len(obs_arr) < 2:
        return 1.0, 0.0

    design_matrix        = np.vstack([obs_arr, np.ones_like(obs_arr)]).T
    solution, _, _, _    = np.linalg.lstsq(design_matrix, lab_arr, rcond=None)
    return float(solution[0]), float(solution[1])


def calculate_parabolic_centroid(y_data: np.ndarray, peak_index: int) -> float:
    """
    Refines the peak position to sub-pixel accuracy using parabolic
    interpolation around the integer peak index.
    """
    idx = int(round(peak_index))
    if idx <= 0 or idx >= len(y_data) - 1:
        return float(idx)

    y_minus  = y_data[idx - 1]
    y_center = y_data[idx]
    y_plus   = y_data[idx + 1]

    denominator = (y_minus - 2 * y_center + y_plus)
    if denominator == 0:
        return float(idx)

    return float(idx) + 0.5 * (y_minus - y_plus) / denominator


# =============================================================================
# CORE ALGORITHM: SIGMA-V DETECTION  (ORIGINAL — UNCHANGED)
# =============================================================================

def analyze_sigma_v_on_spectrum(wavelengths:      np.ndarray,
                                stokes_i:         np.ndarray,
                                stokes_v:         np.ndarray,
                                center_pixel_idx: int,
                                smooth_window:    int   = SMOOTHING_WINDOW_SIZE,
                                smooth_poly:      int   = SMOOTHING_POLY_ORDER,
                                search_half_width:int   = SEARCH_WINDOW_HALF_PIX,
                                noise_region_width:int  = SIGMA_TIGHT_WINDOW_PIX,
                                min_rel_velocity: float = MIN_RELATIVE_AMPLITUDE,
                                noise_factor:     float = NOISE_THRESHOLD_FACTOR
                                ) -> Dict[str, Any]:
    """
    Analyses a single Stokes V spectrum to find sigma-component peaks and
    estimate B using the Zeeman splitting (Sigma-V) method.
    """
    n_pixels = len(wavelengths)
    result = {
        "found": False, "B_G": None, "wa": None, "wb": None,
        "a_sub": None, "b_sub": None, "delta_lambda": None,
        "V_rel": None, "noise": None, "SNR": None,
        "suspect": False, "suspect_reason": None
    }

    if n_pixels < 5:
        result["suspect"] = True
        result["suspect_reason"] = "insufficient_pixels"
        return result

    window_len = ensure_odd(min(smooth_window, max(3, n_pixels - 1)))
    if n_pixels >= window_len:
        try:
            v_smoothed = savgol_filter(stokes_v, window_len, smooth_poly)
        except Exception:
            v_smoothed = stokes_v.copy()
    else:
        v_smoothed = stokes_v.copy()

    continuum_level = 0.5 * (
        np.median(stokes_i[:max(1, n_pixels // 12)]) +
        np.median(stokes_i[-max(1, n_pixels // 12):])
    )
    v_amplitude  = np.nanmax(np.abs(v_smoothed))
    relative_v   = float(v_amplitude / max(abs(continuum_level), 1.0))
    result["V_rel"] = relative_v

    search_start = max(0, center_pixel_idx - search_half_width)
    search_end   = min(n_pixels, center_pixel_idx + search_half_width)
    inner_start  = max(0, center_pixel_idx - noise_region_width)
    inner_end    = min(n_pixels, center_pixel_idx + noise_region_width)

    mask_noise = np.ones_like(v_smoothed, dtype=bool)
    mask_noise[inner_start:inner_end] = False

    noise_samples = v_smoothed[mask_noise] if np.any(mask_noise) else v_smoothed
    noise_level   = (calculate_mad_std(noise_samples) if noise_samples.size > 4
                     else calculate_mad_std(v_smoothed))

    result["noise"] = float(noise_level)
    snr             = v_amplitude / (noise_level if noise_level > 0 else 1e-12)
    result["SNR"]   = float(snr)

    if relative_v < min_rel_velocity or snr < 1.0:
        result["suspect"] = True
        result["suspect_reason"] = "V_too_weak_or_low_SNR"
        return result

    prominence_threshold = max(noise_factor * noise_level, 0.01 * v_amplitude)
    search_region_v      = v_smoothed[search_start:search_end]

    pos_peaks_local, _ = find_peaks( search_region_v, prominence=prominence_threshold)
    neg_peaks_local, _ = find_peaks(-search_region_v, prominence=prominence_threshold)

    pos_peaks_global = (pos_peaks_local + search_start).tolist()
    neg_peaks_global = (neg_peaks_local + search_start).tolist()

    valid_pairs = []
    for p_idx in pos_peaks_global:
        for n_idx in neg_peaks_global:
            if abs(p_idx - n_idx) < 2:
                continue
            if (p_idx - center_pixel_idx) * (n_idx - center_pixel_idx) < 0:
                valid_pairs.append((p_idx, n_idx))

    if not valid_pairs:
        result["suspect"] = True
        result["suspect_reason"] = "no_opposite_V_peaks"
        return result

    best_pair   = max(valid_pairs,
                      key=lambda pair: abs(v_smoothed[pair[0]]) + abs(v_smoothed[pair[1]]))
    idx_a, idx_b = sorted(best_pair)

    pos_a_sub = calculate_parabolic_centroid(
        v_smoothed if v_smoothed[idx_a] >= 0 else -v_smoothed, idx_a)
    pos_b_sub = calculate_parabolic_centroid(
        v_smoothed if v_smoothed[idx_b] >= 0 else -v_smoothed, idx_b)

    wavelength_a  = float(np.interp(pos_a_sub, np.arange(n_pixels), wavelengths))
    wavelength_b  = float(np.interp(pos_b_sub, np.arange(n_pixels), wavelengths))

    delta_lambda  = abs(wavelength_b - wavelength_a) / 2.0
    denominator   = ZEEMAN_CONSTANT_K * LANDÉ_FACTOR_EFF * (REFERENCE_WAVELENGTH_0 ** 2)

    if denominator == 0:
        result["suspect"] = True
        result["suspect_reason"] = "zero_denominator_error"
        return result

    magnetic_field_gauss = delta_lambda / denominator

    is_edge_issue = (
        idx_a <= EDGE_SAFETY_MARGIN_PIX or idx_b <= EDGE_SAFETY_MARGIN_PIX or
        idx_a >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX or
        idx_b >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX
    )

    if abs(pos_b_sub - pos_a_sub) < 1.5:
        result["suspect"] = True
        result["suspect_reason"] = "peaks_too_close"
        return result

    result.update({
        "found":        True,
        "B_G":          float(magnetic_field_gauss),
        "wa":           wavelength_a,
        "wb":           wavelength_b,
        "a_sub":        float(pos_a_sub),
        "b_sub":        float(pos_b_sub),
        "delta_lambda": float(delta_lambda)
    })

    if is_edge_issue:
        result["suspect"] = True
        result["suspect_reason"] = "peak_on_edge"

    if abs(magnetic_field_gauss) > MAX_REALISTIC_B_GAUSS:
        result["suspect"] = True
        result["suspect_reason"] = "B_out_of_range"
        result["B_G"]   = None
        result["found"] = False

    return result


# =============================================================================
# MONTE CARLO ERROR ESTIMATION  (ORIGINAL — UNCHANGED)
# =============================================================================

def estimate_b_error_mc(wavelengths:     np.ndarray,
                        stokes_i:        np.ndarray,
                        stokes_v:        np.ndarray,
                        center_pixel_idx:int,
                        analysis_function: callable,
                        n_iterations:    int   = 200,
                        noise_estimate:  float = None,
                        show_progress:   bool  = False) -> Dict[str, Any]:
    """
    Estimates the uncertainty of the B-field measurement using Monte Carlo
    simulation.  Injects Gaussian noise into the Stokes V profile and
    re-runs the analysis.
    """
    n_pixels = len(wavelengths)

    if noise_estimate is None:
        inner_start = max(0, center_pixel_idx - SIGMA_TIGHT_WINDOW_PIX)
        inner_end   = min(n_pixels, center_pixel_idx + SIGMA_TIGHT_WINDOW_PIX)
        mask_noise  = np.ones_like(stokes_v, dtype=bool)
        mask_noise[inner_start:inner_end] = False
        noise_samples = stokes_v[mask_noise] if np.any(mask_noise) else stokes_v
        sigma_noise   = calculate_mad_std(noise_samples)
    else:
        sigma_noise = float(noise_estimate)

    empty_result = {
        "B_median": None, "B_mean": None, "B_std": None,
        "B_p16": None, "B_p84": None, "N_success": 0
    }
    if sigma_noise <= 0:
        return empty_result

    b_values = []
    rng      = np.random.default_rng()

    for i in range(n_iterations):
        v_perturbed = stokes_v + rng.normal(0.0, sigma_noise, size=stokes_v.shape)
        res = analysis_function(wavelengths, stokes_i, v_perturbed, center_pixel_idx)
        if res.get("found") and res.get("B_G") is not None:
            b_values.append(res["B_G"])
        if show_progress and (i % max(1, n_iterations // 10) == 0):
            print(f"[MC] Iteration {i}/{n_iterations}, successes: {len(b_values)}")

    if not b_values:
        return empty_result

    b_arr = np.array(b_values)
    return {
        "B_median": float(np.median(b_arr)),
        "B_mean":   float(np.mean(b_arr)),
        "B_std":    (float(np.std(b_arr, ddof=1)) if b_arr.size >= 2 else None),
        "B_p16":    (float(np.percentile(b_arr, 16)) if b_arr.size >= 1 else None),
        "B_p84":    (float(np.percentile(b_arr, 84)) if b_arr.size >= 1 else None),
        "N_success": int(b_arr.size)
    }


# =============================================================================
# CALIBRATION AND DATA PREPARATION  (ORIGINAL — UNCHANGED)
# =============================================================================

def cross_calibrate_wavelength(header:             fits.Header,
                               intensity_data_2d:  np.ndarray,
                               wavelengths:        np.ndarray,
                               lab_lines:          Tuple[float, float] = (LINE_LAB_1, LINE_LAB_2),
                               use_atlas:          bool = False,
                               atlas_wav:          np.ndarray = None,
                               atlas_intensity:    np.ndarray = None
                               ) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Performs wavelength calibration using reference atlas matching and/or
    known spectral lines.
    """
    current_wavelengths = wavelengths.copy()
    info = {"method": "none", "shift_pix": 0, "a": 1.0, "b": 0.0,
            "used_atlas": False, "used_lines": False}

    if use_atlas and (atlas_wav is not None) and (atlas_intensity is not None):
        try:
            ref_i_interp = np.interp(
                wavelengths, atlas_wav, atlas_intensity,
                left=np.nan, right=np.nan)
            quiet_sun_profile = np.median(
                intensity_data_2d[:max(1, intensity_data_2d.shape[0] // 6), :],
                axis=0)
            shift_pix      = compute_cross_correlation_shift(quiet_sun_profile, ref_i_interp)
            delta_lambda_s = shift_pix * (wavelengths[1] - wavelengths[0])
            current_wavelengths = current_wavelengths + delta_lambda_s
            info.update({"method": "atlas_shift",
                         "shift_pix": int(shift_pix),
                         "used_atlas": True})
        except Exception as exc:
            print(f"[CALIB] Atlas shift failed: {exc}")

    observed_positions = []
    lab_positions      = []
    median_intensity   = np.median(intensity_data_2d, axis=0)

    for line_lambda in lab_lines:
        idx = robust_find_line_center(
            current_wavelengths, median_intensity,
            line_lambda, window_angstrom=0.6)
        if idx is not None:
            observed_positions.append(current_wavelengths[idx])
            lab_positions.append(line_lambda)

    if len(observed_positions) >= 2:
        scale_a, offset_b = solve_linear_wavelength_scale(
            observed_positions, lab_positions)
        calibrated_wavelengths = scale_a * current_wavelengths + offset_b
        info.update({"method": "lines_scale", "used_lines": True,
                     "a": float(scale_a), "b": float(offset_b),
                     "obs_positions": observed_positions,
                     "lab_positions": lab_positions})
        return calibrated_wavelengths, float(scale_a), float(offset_b), info

    return current_wavelengths, 1.0, 0.0, info


def prepare_me_input_file(wavelengths: np.ndarray,
                          i: np.ndarray, q: np.ndarray,
                          u: np.ndarray, v: np.ndarray,
                          output_dir: str, slit_index: int) -> str:
    """Writes Stokes profiles to a text file for an external ME inversion code."""
    os.makedirs(output_dir, exist_ok=True)
    filename    = os.path.join(output_dir, f"slit_{slit_index:04d}_stokes.txt")
    data_matrix = np.vstack([wavelengths, i, q, u, v]).T
    np.savetxt(filename, data_matrix, header="wav I Q U V")
    return filename


def slit_index_to_arcsec(header: fits.Header,
                         slit_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """Converts slit index to arcseconds using WCS header information."""
    xcen   = header.get('XCEN',   None)
    crpix2 = header.get('CRPIX2', header.get('CRPIX', None))
    xscale = header.get('XSCALE', header.get('CDELT2', header.get('CDELT', None)))

    if xcen is None or crpix2 is None or xscale is None:
        return None, None

    x_arcsec = float(xcen) + (float(slit_idx) - float(crpix2)) * float(xscale)
    ycen     = header.get('YCEN', None)
    y_arcsec = float(ycen) if ycen is not None else None
    return x_arcsec, y_arcsec


# =============================================================================
# WEAK-FIELD APPROXIMATION  (internal helper)
# =============================================================================

def _run_wfa(calibrated_wavelengths: np.ndarray,
             stokes_i:               np.ndarray,
             stokes_v:               np.ndarray,
             idx_center:             int) -> Tuple[Optional[float], Optional[float]]:
    """
    Weak-Field Approximation (WFA) B-field estimator.

    Fits V(λ) ≈ −C · dI/dλ · B_los to extract the line-of-sight field.

    Returns
    -------
    b_wfa : float or None   Line-of-sight B in Gauss
    r_wfa : float or None   Correlation coefficient of the fit
    """
    try:
        n_lambda  = len(calibrated_wavelengths)
        smooth_w  = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
        i_smooth  = (savgol_filter(stokes_i, smooth_w, SMOOTHING_POLY_ORDER)
                     if n_lambda >= smooth_w else stokes_i)

        di_dlambda = np.gradient(i_smooth, calibrated_wavelengths)
        const_c    = ZEEMAN_CONSTANT_K * (REFERENCE_WAVELENGTH_0 ** 2) * LANDÉ_FACTOR_EFF
        reg_x      = -const_c * di_dlambda

        seg_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
        seg_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)

        x_seg = reg_x[seg_lo:seg_hi]
        v_seg = stokes_v[seg_lo:seg_hi]
        mask  = np.isfinite(x_seg) & np.isfinite(v_seg)

        if mask.sum() < 5:
            return None, None

        design  = np.vstack([x_seg[mask], np.ones(mask.sum())]).T
        sol, _, _, _ = np.linalg.lstsq(design, v_seg[mask], rcond=None)
        slope_b = float(sol[0])
        model_v = design @ sol

        residuals  = v_seg[mask] - model_v
        resid_noise = calculate_mad_std(residuals)
        seg_snr    = (np.nanmax(np.abs(v_seg[mask])) /
                      (resid_noise if resid_noise > 0 else 1e-12))
        corr       = (float(np.corrcoef(v_seg[mask], model_v)[0, 1])
                      if mask.sum() >= 3 else 0.0)

        if np.isfinite(slope_b) and abs(corr) >= 0.4 and seg_snr >= 5.0:
            return slope_b, corr

    except Exception:
        pass

    return None, None


# =============================================================================
# MAIN ANALYSIS PIPELINE — AI-FIRST ARCHITECTURE
# =============================================================================

def analyze_fits_file(fits_path:      str,
                      output_prefix:  str  = DEFAULT_OUTPUT_PREFIX,
                      n_mc_iterations: int = DEFAULT_MC_ITERATIONS,
                      run_me_prep:    bool = False,
                      me_cmd_template: Optional[str] = None,
                      use_atlas:      bool = USE_ATLAS_REF,
                      atlas_wav:      Optional[np.ndarray] = None,
                      atlas_intensity: Optional[np.ndarray] = None
                      ) -> pd.DataFrame:
    """
    Main driver function: load FITS, calibrate, then let the SolarIntelligenceSystem
    (SIS) dispatch each pixel to the most appropriate physics engine.

    Architecture
    ------------
    For every slit position the pipeline runs in this strict order:

      ① SIS.predict(stokes_v) → signal_class, confidence, b_guess
      ② Dispatch based on signal_class:
           "Clear"   → analyze_sigma_v_on_spectrum  (standard Zeeman)
           "Noisy"   → _run_wfa first; if WFA fails → estimate_b_error_mc (500 it.)
           "Anomaly" → skip heavy physics; save V profile to anomaly archive;
                       record pixel for manual review
      ③ Monte Carlo error estimation (for Clear/Noisy that yielded a B value)
      ④ Record results with AI routing metadata

    The AI is NOT optional.  There is no --no-ai mode.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    if not AI_SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is required for the AI-first pipeline.  "
            "Install with: pip install scikit-learn")

    # ── Initialise SolarIntelligenceSystem ────────────────────────────────────
    print("[SIS] Initialising SolarIntelligenceSystem …")
    sis = SolarIntelligenceSystem()

    # ── Initialise LLM layer ──────────────────────────────────────────────────
    ai_scientist = AIScientist()

    # ── Load FITS data ────────────────────────────────────────────────────────
    with fits.open(fits_path, memmap=True) as hdul:
        header   = hdul[0].header
        raw_data = hdul[0].data

        if raw_data is None:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    raw_data = hdu.data
                    header   = hdu.header
                    break

        if raw_data is None:
            raise RuntimeError("No array data found in FITS HDUs.")

        data_cube = raw_data.astype(float)

    # ── Validate 3-D shape (4, Nslit, Nlambda) ───────────────────────────────
    if data_cube.ndim != 3:
        raise RuntimeError(
            f"FITS data must be 3D (4, Nslit, Nlambda). Got: {data_cube.shape}")

    if data_cube.shape[0] != 4:
        if data_cube.shape[2] == 4:
            data_cube = np.transpose(data_cube, (2, 0, 1))
        elif data_cube.shape[1] == 4:
            data_cube = np.transpose(data_cube, (1, 0, 2))
        else:
            raise RuntimeError(
                f"Unexpected FITS shape.  Expected first axis = 4 (Stokes). "
                f"Got: {data_cube.shape}")

    _, n_slits, n_lambda = data_cube.shape

    # ── Wavelength construction & calibration ─────────────────────────────────
    raw_wavelengths, cdelt, _, _ = build_wavelength_axis(header, n_lambda)
    print(f"[INFO] {fits_path}  shape: {data_cube.shape}")
    print(f"[INFO] λ[0]={raw_wavelengths[0]:.6f}  "
          f"λ[-1]={raw_wavelengths[-1]:.6f}  dλ={cdelt:.6f} Å")

    intensity_data = data_cube[0, :, :]
    calibrated_wavelengths, _, _, calib_info = cross_calibrate_wavelength(
        header, intensity_data, raw_wavelengths,
        lab_lines=(LINE_LAB_1, LINE_LAB_2),
        use_atlas=use_atlas,
        atlas_wav=atlas_wav,
        atlas_intensity=atlas_intensity
    )
    print(f"[CALIB] {calib_info}")

    # ── Reference line-centre pixel ───────────────────────────────────────────
    idx_center = int(np.argmin(np.abs(calibrated_wavelengths - REFERENCE_WAVELENGTH_0)))
    idx_center = max(0, min(n_lambda - 1, idx_center))

    # ── Train SIS if not already loaded from disk ─────────────────────────────
    if not sis.is_trained:
        sis.train_on_the_fly(
            data_cube, calibrated_wavelengths, idx_center, n_samples=250)

    # ── Prepare output directories ────────────────────────────────────────────
    examples_dir = f"{output_prefix}_examples"
    anomaly_dir  = f"{output_prefix}_anomalies"
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(anomaly_dir,  exist_ok=True)

    results_list      = []
    b_profile_array   = np.full(n_slits, np.nan)
    sigma_b_profile   = np.full(n_slits, np.nan)

    # Route counters (for summary)
    route_counts: Dict[str, int] = {
        ROUTE_SIGMA_V: 0, ROUTE_WFA_NOISY: 0,
        ROUTE_MC_NOISY: 0, ROUTE_ANOMALY: 0
    }

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP — one iteration per slit position
    # ─────────────────────────────────────────────────────────────────────────
    for s_idx in range(n_slits):

        # ── Spatial averaging (optional window) ───────────────────────────
        s_lo = max(0, s_idx - (SPATIAL_AVERAGING_WINDOW // 2))
        s_hi = min(n_slits, s_lo + SPATIAL_AVERAGING_WINDOW)

        if SPATIAL_AVERAGING_WINDOW > 1:
            stokes_i = data_cube[0, s_lo:s_hi, :].mean(axis=0)
            stokes_q = data_cube[1, s_lo:s_hi, :].mean(axis=0)
            stokes_u = data_cube[2, s_lo:s_hi, :].mean(axis=0)
            stokes_v = data_cube[3, s_lo:s_hi, :].mean(axis=0)
        else:
            stokes_i = data_cube[0, s_idx, :]
            stokes_q = data_cube[1, s_idx, :]
            stokes_u = data_cube[2, s_idx, :]
            stokes_v = data_cube[3, s_idx, :]

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  STEP ①  SIS EVALUATION — AI decides what happens next          ║
        # ╚══════════════════════════════════════════════════════════════════╝
        signal_class, ai_confidence, b_guess = sis.predict(
            stokes_v, calibrated_wavelengths, idx_center)

        # ── Initialise per-slit accumulators ──────────────────────────────
        b_final_val     = None
        wfa_correlation = None
        used_method     = "none"
        ai_route        = "Unknown"
        sigma_res       = {"found": False, "B_G": None, "SNR": None,
                           "noise": None, "V_rel": None, "suspect": False,
                           "suspect_reason": None, "wa": None, "wb": None,
                           "a_sub": None, "b_sub": None, "delta_lambda": None}
        mc_stats        = {"B_median": None, "B_mean": None, "B_std": None,
                           "B_p16": None, "B_p84": None, "N_success": 0}

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  STEP ②  AI-DRIVEN DISPATCH                                     ║
        # ╚══════════════════════════════════════════════════════════════════╝

        if signal_class == "Clear":
            # ── Route A: Standard Sigma-V (Zeeman splitting) ───────────────
            sigma_res = analyze_sigma_v_on_spectrum(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center)

            if sigma_res.get("found") and sigma_res.get("B_G") is not None:
                b_final_val = sigma_res["B_G"]
                used_method = "sigma"
                ai_route    = ROUTE_SIGMA_V
            else:
                # Sigma-V failed on a Clear profile (e.g. edge artefact);
                # fall through gracefully to WFA.
                b_wfa, r_wfa = _run_wfa(
                    calibrated_wavelengths, stokes_i, stokes_v, idx_center)
                if b_wfa is not None:
                    b_final_val     = b_wfa
                    wfa_correlation = r_wfa
                    used_method     = "wfa"
                    ai_route        = ROUTE_WFA_NOISY   # same pool, labelled distinctly
                else:
                    ai_route = ROUTE_SIGMA_V   # attempted but no detection

            route_counts[ROUTE_SIGMA_V] += 1

        elif signal_class == "Noisy":
            # ── Route B: WFA first; MC brute-force if WFA fails ───────────
            b_wfa, r_wfa = _run_wfa(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center)

            if b_wfa is not None:
                b_final_val     = b_wfa
                wfa_correlation = r_wfa
                used_method     = "wfa"
                ai_route        = ROUTE_WFA_NOISY
                route_counts[ROUTE_WFA_NOISY] += 1
            else:
                # WFA gave no result — brute-force Monte Carlo at 500 iterations
                # to squeeze a detection out of the noise.
                print(f"[SIS] Slit {s_idx}: Noisy + WFA failed → "
                      f"MC brute-force ({NOISY_BRUTE_FORCE_MC_ITERS} it.)")

                noise_lo  = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
                noise_hi  = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)
                mc_mask   = np.ones_like(stokes_v, dtype=bool)
                mc_mask[noise_lo:noise_hi] = False
                mc_noise  = calculate_mad_std(stokes_v[mc_mask] if mc_mask.any()
                                              else stokes_v)

                mc_stats = estimate_b_error_mc(
                    calibrated_wavelengths, stokes_i, stokes_v, idx_center,
                    analyze_sigma_v_on_spectrum,
                    n_iterations=NOISY_BRUTE_FORCE_MC_ITERS,
                    noise_estimate=mc_noise
                )
                if mc_stats["B_median"] is not None:
                    b_final_val = mc_stats["B_median"]
                    used_method = "mc_brute"

                ai_route = ROUTE_MC_NOISY
                route_counts[ROUTE_MC_NOISY] += 1

        elif signal_class == "Anomaly":
            # ── Route C: Flag pixel; archive profile; skip heavy physics ──
            # Save the anomalous Stokes V vector for manual scientific review.
            anomaly_file = os.path.join(anomaly_dir, f"anomaly_slit_{s_idx:04d}.npy")
            np.save(anomaly_file,
                    np.vstack([calibrated_wavelengths,
                               stokes_i, stokes_q, stokes_u, stokes_v]))

            ai_route    = ROUTE_ANOMALY
            used_method = "anomaly"
            route_counts[ROUTE_ANOMALY] += 1

            # Lightweight diagnostic plot for each anomaly
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            axes[0].plot(calibrated_wavelengths, stokes_i, color='steelblue')
            axes[0].set_ylabel("Stokes I")
            axes[0].grid(True)
            axes[1].plot(calibrated_wavelengths, stokes_v,
                         color='crimson', lw=1.2)
            axes[1].set_ylabel("Stokes V  [ANOMALY]")
            axes[1].set_xlabel("Wavelength (Å)")
            axes[1].grid(True)
            fig.suptitle(
                f"ANOMALY — Slit {s_idx}  "
                f"AI conf={ai_confidence:.1f}%\n"
                f"Profile archived to {anomaly_dir}/",
                fontsize=10, color='darkred')
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            plt.savefig(os.path.join(examples_dir,
                                     f"ANOMALY_slit_{s_idx:04d}.png"), dpi=150)
            plt.close(fig)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  STEP ③  MONTE CARLO ERROR ESTIMATION (Clear & Noisy with B)    ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if b_final_val is not None and signal_class != "Anomaly" \
                and ai_route != ROUTE_MC_NOISY:   # MC was already the dispatch step
            noise_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
            noise_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)
            mc_mask  = np.ones_like(stokes_v, dtype=bool)
            mc_mask[noise_lo:noise_hi] = False
            mc_noise = calculate_mad_std(stokes_v[mc_mask] if mc_mask.any()
                                         else stokes_v)

            mc_stats = estimate_b_error_mc(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center,
                analyze_sigma_v_on_spectrum,
                n_iterations=max(50, min(n_mc_iterations, 500)),
                noise_estimate=mc_noise
            )

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  STEP ④  RECORD RESULTS WITH FULL AI ROUTING METADATA           ║
        # ╚══════════════════════════════════════════════════════════════════╝
        x_arc, y_arc = slit_index_to_arcsec(header, s_idx)

        row = {
            "slit":       int(s_idx),
            "x_pix":      int(s_idx),
            "x_arcsec":   float(x_arc) if x_arc is not None else None,
            "y_arcsec":   float(y_arc) if y_arc is not None else None,
            "idx_center": int(idx_center),
            "used":       used_method,

            # ── SIS decision ──────────────────────────────────────────────
            "AI_Signal_Class": signal_class,
            "AI_Confidence":   float(ai_confidence),
            "AI_B_guess":      float(b_guess),
            "AI_Route":        ai_route,

            # ── Sigma-V fields ────────────────────────────────────────────
            "B_sigma_G":   (float(sigma_res["B_G"])
                            if (sigma_res.get("found") and
                                sigma_res.get("B_G") is not None) else None),
            "B_wfa_G":     (float(b_final_val)
                            if (used_method in ("wfa", "sigma") and
                                b_final_val is not None) else None),
            "wfa_r":       (float(wfa_correlation)
                            if wfa_correlation is not None else None),

            "V_rel":       (float(sigma_res["V_rel"])
                            if sigma_res.get("V_rel") is not None else None),
            "noise":       (float(sigma_res["noise"])
                            if sigma_res.get("noise") is not None else None),
            "SNR":         (float(sigma_res["SNR"])
                            if sigma_res.get("SNR") is not None else None),
            "sigma_found": bool(sigma_res.get("found", False)),

            "wa_A":            (float(sigma_res["wa"])
                                if sigma_res.get("wa") is not None else None),
            "wb_A":            (float(sigma_res["wb"])
                                if sigma_res.get("wb") is not None else None),
            "delta_lambda_A":  (float(sigma_res["delta_lambda"])
                                if sigma_res.get("delta_lambda") is not None
                                else None),

            "suspect":        bool(sigma_res.get("suspect", False)),
            "suspect_reason": sigma_res.get("suspect_reason"),

            # ── MC error stats ────────────────────────────────────────────
            "B_MC_median":    mc_stats["B_median"],
            "B_MC_mean":      mc_stats["B_mean"],
            "B_MC_std":       mc_stats["B_std"],
            "B_p16":          mc_stats["B_p16"],
            "B_p84":          mc_stats["B_p84"],
            "B_MC_n_success": mc_stats["N_success"],
        }

        # Best B estimate (prefer Sigma-V, fall back to WFA / MC)
        b_choice  = row["B_sigma_G"]
        if b_choice is None:
            b_choice = b_final_val if used_method in ("wfa", "mc_brute") else None
        row["B_G"] = float(b_choice) if b_choice is not None else None

        results_list.append(row)
        b_profile_array[s_idx] = row["B_G"] if row["B_G"] is not None else np.nan
        sigma_b_profile[s_idx] = (float(mc_stats["B_std"])
                                   if mc_stats["B_std"] is not None else np.nan)

        # ── Diagnostic plot (non-anomaly slits with signal or suspect flag) ─
        if signal_class != "Anomaly" and \
                (row["sigma_found"] or row["suspect"]):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            ax1.plot(calibrated_wavelengths, stokes_i, label="I",
                     color='steelblue')
            ax1.set_ylabel("Intensity")
            ax1.grid(True)

            win_len = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
            v_smooth_plot = (savgol_filter(stokes_v, win_len, SMOOTHING_POLY_ORDER)
                             if n_lambda >= win_len else stokes_v)
            ax2.plot(calibrated_wavelengths, stokes_v,
                     label="V_raw", alpha=0.55)
            ax2.plot(calibrated_wavelengths, v_smooth_plot,
                     label="V_smooth", color='red', lw=1.2)

            if row["wa_A"] is not None:
                ax2.axvline(row["wa_A"], color='magenta', ls='--', label='σ−')
            if row["wb_A"] is not None:
                ax2.axvline(row["wb_A"], color='cyan', ls='--', label='σ+')

            ax2.set_ylabel("Stokes V")
            ax2.set_xlabel("Wavelength (Å)")
            ax2.grid(True)
            ax2.legend(fontsize=8)

            b_str  = f"{row['B_G']:.0f} G" if row['B_G'] is not None else "—"
            title  = (f"Slit {s_idx}  X={row['x_arcsec']:.1f}\"  "
                      f"B={b_str}  method={row['used']}\n"
                      f"SIS: {signal_class} ({ai_confidence:.1f}%)  "
                      f"B_guess={b_guess:.0f} G  "
                      f"Route → {ai_route}")
            fig.suptitle(title, fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            out_png = os.path.join(examples_dir, f"slit_{s_idx:04d}.png")
            try:
                plt.savefig(out_png, dpi=150)
            except Exception:
                pass
            plt.close(fig)

        if (s_idx % 50) == 0:
            print(f"[PROGRESS] {s_idx}/{n_slits}  "
                  f"routes so far: {route_counts}")

    # ─────────────────────────────────────────────────────────────────────────
    # POST-LOOP OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────

    df_results = pd.DataFrame(results_list)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_file = f"{output_prefix}_results.csv"
    df_results.to_csv(csv_file, index=False, float_format="%.6f")
    print(f"[INFO] Results CSV → {csv_file}")

    np.save(f"{output_prefix}_B_profile.npy", b_profile_array)

    # ── B-field profile plot ──────────────────────────────────────────────────
    x_axis = np.arange(len(b_profile_array))
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, b_profile_array, marker='o', lw=0.8)
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("B (Gauss)")
    plt.title("Magnetic Field Profile")
    plt.savefig(f"{output_prefix}_B_profile.png", dpi=200)
    plt.close()

    # ── Sigma-B (uncertainty) profile plot ────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, sigma_b_profile, marker='o', lw=0.8, color='darkorange')
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("σ_B (Gauss)")
    plt.title("Magnetic Field Uncertainty (Monte Carlo)")
    plt.savefig(f"{output_prefix}_B_sigma_profile.png", dpi=200)
    plt.close()

    # ── AI Routing distribution bar chart ─────────────────────────────────────
    if 'AI_Route' in df_results.columns:
        route_vc = df_results['AI_Route'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(route_vc.index, route_vc.values,
                      color=['#2196F3', '#FF9800', '#F44336', '#9C27B0'])
        ax.bar_label(bars, fmt='%d', padding=3)
        ax.set_xlabel("AI Route")
        ax.set_ylabel("Count")
        ax.set_title("SIS Dispatch Routing Distribution")
        ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_routing_distribution.png", dpi=200)
        plt.close()
        print(f"[PLOT] Routing chart → {output_prefix}_routing_distribution.png")

    # ── 1-D arcsec B-map ──────────────────────────────────────────────────────
    try:
        coords = df_results['x_arcsec'].values
        if np.all(~pd.isna(coords)):
            plt.figure(figsize=(8, 4))
            sc = plt.scatter(
                coords, np.zeros_like(coords),
                c=df_results['B_G'].values, cmap='inferno', s=15)
            plt.colorbar(sc, label='B (Gauss)')
            plt.xlabel('X (arcsec)')
            plt.title('B along scan')
            plt.savefig(f"{output_prefix}_B_map_scan.png", dpi=200)
            plt.close()
    except Exception:
        pass

    # ── 2-D B-map ─────────────────────────────────────────────────────────────
    try:
        print("[PLOT] Building 2D B-map …")
        b_map = np.full((n_slits, 1), np.nan)
        for s_idx in range(n_slits):
            i_p = data_cube[0, s_idx, :]
            v_p = data_cube[3, s_idx, :]
            res2d = analyze_sigma_v_on_spectrum(
                calibrated_wavelengths, i_p, v_p, idx_center)
            if res2d.get("found") and res2d.get("B_G") is not None:
                b_map[s_idx, 0] = res2d["B_G"]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(b_map.T, aspect='auto', cmap='RdBu_r',
                        extent=[0, n_slits, 0, 1], origin='lower')
        plt.colorbar(im, label="B (Gauss)")
        plt.xlabel("Slit Index")
        plt.ylabel("Pixel along slit")
        plt.title("2D Magnetic Field Map (Sigma-V)")
        plt.savefig(f"{output_prefix}_B_map_2D.png", dpi=200)
        plt.close()
        print(f"[PLOT] 2D map → {output_prefix}_B_map_2D.png")
    except Exception as exc:
        print(f"[WARN] 2D B-map failed: {exc}")

    # ── AI Layer 2: LLM Scientific Report ─────────────────────────────────────
    if ai_scientist.available:
        report_path = f"{output_prefix}_{AI_REPORT_FILENAME}"
        ai_scientist.generate_report(df_results,
                                     os.path.basename(fits_path),
                                     report_path)

    # ── ME Solver preparation (optional) ──────────────────────────────────────
    if run_me_prep and (me_cmd_template is not None):
        me_input_dir = f"{output_prefix}_ME_input"
        os.makedirs(me_input_dir, exist_ok=True)
        top_slits = (df_results[df_results['B_G'].notna()]
                     .sort_values('B_G', ascending=False)['slit']
                     .tolist()[:20])
        for s in top_slits:
            s_idx = int(s)
            s_lo  = max(0, s_idx - (SPATIAL_AVERAGING_WINDOW // 2))
            s_hi  = min(n_slits, s_lo + SPATIAL_AVERAGING_WINDOW)
            if SPATIAL_AVERAGING_WINDOW > 1:
                i_s = data_cube[0, s_lo:s_hi, :].mean(axis=0)
                q_s = data_cube[1, s_lo:s_hi, :].mean(axis=0)
                u_s = data_cube[2, s_lo:s_hi, :].mean(axis=0)
                v_s = data_cube[3, s_lo:s_hi, :].mean(axis=0)
            else:
                i_s = data_cube[0, s_idx, :]
                q_s = data_cube[1, s_idx, :]
                u_s = data_cube[2, s_idx, :]
                v_s = data_cube[3, s_idx, :]
            infile = prepare_me_input_file(
                calibrated_wavelengths, i_s, q_s, u_s, v_s,
                me_input_dir, s_idx)
            print(f"[ME] Prepared input → {infile}")

    print(f"[INFO] Anomalous profiles archived → {anomaly_dir}/")
    print(f"[INFO] Diagnostic plots             → {examples_dir}/")

    return df_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main_cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Hinode SP Sigma-V Analyzer — AI-First Architecture (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python ai_enhanced_analyzer_v2.py --fits mydata.fits --out results
  python ai_enhanced_analyzer_v2.py --fits mydata.fits --mc 200
  python ai_enhanced_analyzer_v2.py --fits mydata.fits --atlas --run_me

AI Architecture
---------------
  The SolarIntelligenceSystem (SIS) is the mandatory core dispatcher.
  It classifies every Stokes V profile into:
    "Clear"   → Sigma-V (Zeeman)
    "Noisy"   → WFA fallback or Monte Carlo brute-force (500 iterations)
    "Anomaly" → Flagged; archived for manual scientific review

  There is no --no-ai mode.  The AI IS the pipeline.
        """
    )
    parser.add_argument("--fits",   default=DEFAULT_FITS_FILE,
                        help="Path to input FITS file")
    parser.add_argument("--out",    default=DEFAULT_OUTPUT_PREFIX,
                        help="Output filename prefix")
    parser.add_argument("--mc",     type=int, default=DEFAULT_MC_ITERATIONS,
                        help="Monte Carlo iterations for error estimation")
    parser.add_argument("--atlas",  action='store_true',
                        help="Enable Atlas-based wavelength calibration")
    parser.add_argument("--run_me", action='store_true',
                        help="Prepare Milne-Eddington inversion input files")

    args = parser.parse_args()

    print("=" * 80)
    print("Hinode SP Sigma-V Analyzer — AI-First Architecture (v2)")
    print("=" * 80)
    print(f"Input FITS       : {args.fits}")
    print(f"Output prefix    : {args.out}")
    print(f"MC iterations    : {args.mc}")
    print(f"SIS dispatcher   : ENABLED (mandatory)")
    print("=" * 80)

    start_time = time.time()

    atlas_wav = atlas_int = None
    if args.atlas and REF_ATLAS_WAV_PATH is not None:
        try:
            atlas_data = np.loadtxt(REF_ATLAS_WAV_PATH)
            atlas_wav  = atlas_data[:, 0]
            atlas_int  = atlas_data[:, 1]
            print(f"[ATLAS] Loaded → {REF_ATLAS_WAV_PATH}")
        except Exception as exc:
            print(f"[ATLAS] Failed to load: {exc}")

    try:
        df_result = analyze_fits_file(
            args.fits,
            output_prefix=args.out,
            n_mc_iterations=args.mc,
            run_me_prep=args.run_me,
            me_cmd_template=ME_SOLVER_CMD_TEMPLATE,
            use_atlas=args.atlas,
            atlas_wav=atlas_wav,
            atlas_intensity=atlas_int,
        )

        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"Analysis complete in {elapsed:.1f} s")
        print("=" * 80)

        # ── Console summary ────────────────────────────────────────────────
        if 'AI_Signal_Class' in df_result.columns:
            print("\n── SIS Signal Classification ──")
            print(df_result['AI_Signal_Class'].value_counts(dropna=False).to_string())

        if 'AI_Route' in df_result.columns:
            print("\n── AI Routing Distribution ──")
            print(df_result['AI_Route'].value_counts(dropna=False).to_string())

        if 'used' in df_result.columns:
            print("\n── Physics Method Usage ──")
            print(df_result['used'].value_counts(dropna=False).to_string())

        if 'suspect' in df_result.columns:
            print(f"\nSuspect measurements : {int(df_result['suspect'].sum())}")

        print("\n── Output files ──")
        print(f"  {args.out}_results.csv")
        print(f"  {args.out}_B_profile.png")
        print(f"  {args.out}_B_sigma_profile.png")
        print(f"  {args.out}_routing_distribution.png")
        print(f"  {args.out}_examples/   (diagnostic plots)")
        print(f"  {args.out}_anomalies/  (flagged Stokes V archives)")
        if AI_OLLAMA_AVAILABLE:
            print(f"  {args.out}_{AI_REPORT_FILENAME}")
        print("=" * 80)

    except Exception as error:
        print(f"[ERROR] Analysis failed: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_cli()
