#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinode SP Sigma-V Analyzer — AI-First Architecture (SIS v3.0)
==============================================================

Pipeline architecture
---------------------
The Solar Intelligence System (SIS) is the mandatory core dispatcher that
drives all analysis logic.  The AI evaluates every Stokes V profile *first*
and routes it to the most appropriate physics engine:

  * ``"Clear"``   → Standard Sigma-V (Zeeman splitting) analysis
  * ``"Noisy"``   → Weak Field Approximation (WFA) **or** brute-force
                    Monte Carlo (500 iterations)
  * ``"Anomaly"`` → Profile flagged for manual scientific review;
                    archived to disk

SIS also provides a rapid B-field initial guess (``B_guess``) via a
``RandomForestRegressor`` — useful as a seed for external
Milne-Eddington inversions.

v3.0 additions ("Fantastic Four")
---------------------------------
Four lightweight physical-parameter extraction methods are computed for
every non-anomaly slit and appended to the output DataFrame/CSV:

  1. **Transverse Magnetic Field** (B_perp) — Weak Field Approximation
     for linear polarisation using d²I/dλ² correlated with Q and U.
  2. **3-D Temperature Stratification** — Eddington-Barbier approximation
     converting Stokes I intensities at 5 optical-depth sample points
     to brightness temperatures via the inverse Planck function.
  3. **Plasma LOS Velocity** — Centre-of-Gravity (COG) Doppler shift
     of the Stokes I absorption profile relative to the 6302.5 Å rest
     wavelength.
  4. **Plasma Turbulence** — Non-thermal line broadening derived by
     subtracting the thermal FWHM contribution from the observed total
     FWHM of the Stokes I profile.

Layer structure
---------------
  1. **SolarIntelligenceSystem** — feature extraction, Random Forest
     classification (3-class) and regression (B_guess).
  2. **AIScientist** — LLM-powered scientific interpretation via Ollama
     (optional; gracefully disabled if unavailable).  The v3.0 prompt
     now ingests statistical summaries of all four new parameters.

Critical constraint
-------------------
The following physics/math functions are intentionally **unchanged**::

    analyze_sigma_v_on_spectrum
    estimate_b_error_mc
    cross_calibrate_wavelength
    robust_find_line_center
    solve_linear_wavelength_scale
    calculate_parabolic_centroid
"""

import os
import time
import argparse
import warnings
os.environ["GEMINI_API_KEY"] = "AIzaSyALeOC0G-kdEmEUPW_WC_CBKuPlbba66LA"
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import fftconvolve, find_peaks, savgol_filter
from scipy.stats import median_abs_deviation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# ---------------------------------------------------------------------------
# Optional visualisation dependency
# ---------------------------------------------------------------------------

try:
    import seaborn as sns

    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional AI / ML dependencies
# ---------------------------------------------------------------------------

# XGBoost — fast local classifier (Phase 1 dispatcher)
try:
    import xgboost as xgb
    import joblib

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print(
        "[WARN] xgboost / joblib not available.  XGBoost dispatcher disabled — "
        "all slits will use heuristic classification.  "
        "Install with: pip install xgboost joblib"
    )

# Google GenAI (new SDK) — used ONLY for the final scientific report
try:
    from google import genai as google_genai

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print(
        "[WARN] google-genai not available.  Gemini report generation disabled.  "
        "Install with: pip install google-genai"
    )


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

DEFAULT_FITS_FILE = "SP3D20231104_210115.0C.fits"
DEFAULT_OUTPUT_PREFIX = "solar_results"

# --- Analysis parameters ---------------------------------------------------
SPATIAL_AVERAGING_WINDOW = 1
REFERENCE_WAVELENGTH_0 = 6302.5
LINE_LAB_1 = 6301.5
LINE_LAB_2 = 6302.5
LANDE_FACTOR_EFF = 2.5
ZEEMAN_CONSTANT_K = 4.67e-13

# --- Signal processing parameters ------------------------------------------
SMOOTHING_WINDOW_SIZE = 9
SMOOTHING_POLY_ORDER = 3
SEARCH_WINDOW_HALF_PIX = 14
SIGMA_TIGHT_WINDOW_PIX = 8
MIN_RELATIVE_AMPLITUDE = 0.01
NOISE_THRESHOLD_FACTOR = 4.0
EDGE_SAFETY_MARGIN_PIX = 2
MAX_REALISTIC_B_GAUSS = 5000.0

# --- Monte Carlo parameters ------------------------------------------------
DEFAULT_MC_ITERATIONS = 100
NOISY_BRUTE_FORCE_MC_ITERS = 500  # High iteration count for "Noisy" routing

# --- Signal-class integer codes and human-readable labels ------------------
CLASS_NOISY = 0
CLASS_CLEAR = 1
CLASS_ANOMALY = 2
CLASS_LABELS: Dict[int, str] = {
    CLASS_NOISY: "Noisy",
    CLASS_CLEAR: "Clear",
    CLASS_ANOMALY: "Anomaly",
}

# --- AI routing labels (stored in the output CSV) --------------------------
ROUTE_SIGMA_V = "SigmaV"
ROUTE_WFA_NOISY = "WFA_Noisy"
ROUTE_MC_NOISY = "MC_BruteForce"
ROUTE_ANOMALY = "Anomaly_Flagged"
ROUTE_NO_SIGNAL = "No_Signal"  # v4: Noisy pixel where WFA also returned None

# --- AI Report configuration -----------------------------------------------
AI_REPORT_FILENAME = "AI_SCIENTIFIC_REPORT.txt"
AI_GEMINI_REPORT_MODEL = "gemini-2.5-flash"

# --- XGBoost dispatcher model path ----------------------------------------
XGB_MODEL_PATH = "xgb_model.pkl"

# --- External Milne-Eddington solver configuration (optional) ---------------
USE_ATLAS_REF = False
REF_ATLAS_WAV_PATH = None
REF_ATLAS_INTENSITY_PATH = None
ME_SOLVER_CMD_TEMPLATE = None
ME_TEMP_DIR_SUFFIX = "_ME_input"

# Preserve the Unicode constant name as a module-level alias so that any
# external code importing it by the original name continues to work.
LANDÉ_FACTOR_EFF = LANDE_FACTOR_EFF  # noqa: N816

# =============================================================================
# v3.0 PHYSICAL CONSTANTS — "Fantastic Four" parameter extraction
# =============================================================================

# Planck function constants (CGS)
PLANCK_H = 6.62607015e-27   # erg·s
PLANCK_C = 2.99792458e10    # cm/s
BOLTZMANN_K = 1.380649e-16  # erg/K

# Iron-56 atomic mass (for thermal broadening of Fe I 6302.5 Å)
FE56_ATOMIC_MASS_G = 56.0 * 1.66053906660e-24  # grams

# Speed of light in km/s (for Doppler velocity)
C_KM_S = 2.99792458e5


# =============================================================================
# AI ENGINE — LAYER 1: SOLAR INTELLIGENCE SYSTEM (SIS)
# =============================================================================


class SolarIntelligenceSystem:
    """
    Hybrid AI dispatcher for the Hinode SP analysis pipeline.

    **Phase 1 (XGBoost)**:  A pre-trained XGBoost model loaded from
    ``xgb_model.pkl`` classifies every slit into ``"Clear"``,
    ``"Noisy"``, or ``"Anomaly"`` in a single vectorised pass (~ms).

    **Heuristic fallback**:  If ``xgb_model.pkl`` is not found, or
    XGBoost is not installed, all classification falls back to a
    lightweight rule-based heuristic so the pipeline never crashes.

    **B_guess**: Estimated from Stokes V amplitude via simple Zeeman
    scaling (no ML regressor required).

    Gemini is **not** used here.  It is used only at the very end of
    the pipeline to generate a scientific text report (see
    :func:`generate_gemini_report`).
    """

    # Number of features extracted per slit for XGBoost
    _N_FEATURES = 16

    def __init__(self) -> None:
        self.xgb_model = None
        self.use_xgb = False

        if _XGB_AVAILABLE:
            if os.path.isfile(XGB_MODEL_PATH):
                try:
                    self.xgb_model = joblib.load(XGB_MODEL_PATH)
                    self.use_xgb = True
                    print(f"[SIS] XGBoost model loaded from {XGB_MODEL_PATH}")
                except Exception as exc:
                    print(
                        f"[SIS] Failed to load XGBoost model: {exc}.  "
                        "Using heuristic fallback."
                    )
            else:
                print(
                    f"[SIS] XGBoost model not found at '{XGB_MODEL_PATH}'.  "
                    "Using heuristic fallback classifier."
                )
        else:
            print("[SIS] XGBoost not installed.  Using heuristic fallback.")

        # Cache: {slit_index: "Clear"/"Noisy"/"Anomaly"}
        self._cache: Dict[int, str] = {}

    # -------------------------------------------------------------------------
    # Feature extraction — compact 16-D numerical vector per slit
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_feature_vector(
        data_cube: np.ndarray,
        slit_idx: int,
        center_idx: int,
    ) -> np.ndarray:
        """
        Extract a 16-dimensional feature vector for one slit.

        Layout (per Stokes parameter I, Q, U, V — 4 features each):
            [max_abs, std, mean, snr]

        Parameters
        ----------
        data_cube : np.ndarray
            Shape ``(4, n_slits, n_lambda)``.
        slit_idx : int
            Slit index to extract.
        center_idx : int
            Pixel index of spectral line centre.

        Returns
        -------
        np.ndarray
            Feature vector of shape ``(16,)``.
        """
        features = np.zeros(16, dtype=np.float64)
        lobe_lo = max(0, center_idx - SEARCH_WINDOW_HALF_PIX)
        lobe_hi = min(data_cube.shape[2], center_idx + SEARCH_WINDOW_HALF_PIX + 1)

        for i in range(4):  # I, Q, U, V
            arr = data_cube[i, slit_idx, :]
            off_lobe = np.concatenate([arr[:lobe_lo], arr[lobe_hi:]])
            noise = (
                1.4826 * float(median_abs_deviation(off_lobe))
                if len(off_lobe) >= 5
                else 1.4826 * float(median_abs_deviation(arr))
            )
            noise = max(noise, 1e-12)
            max_abs = float(np.max(np.abs(arr)))

            base = i * 4
            features[base + 0] = max_abs
            features[base + 1] = float(np.std(arr))
            features[base + 2] = float(np.mean(arr))
            features[base + 3] = max_abs / noise  # SNR

        return features

    # -------------------------------------------------------------------------
    # Heuristic fallback classifier  (unchanged physics)
    # -------------------------------------------------------------------------

    @staticmethod
    def _classify_heuristic(
        stokes_v: np.ndarray,
        center_idx: int,
    ) -> str:
        """
        Lightweight heuristic 3-class classifier.  Used as the fallback
        when XGBoost is unavailable or ``xgb_model.pkl`` is missing.
        """
        from scipy.stats import skew as scipy_skew

        max_amp = np.max(np.abs(stokes_v))
        lobe_lo = max(0, center_idx - SEARCH_WINDOW_HALF_PIX)
        lobe_hi = min(len(stokes_v), center_idx + SEARCH_WINDOW_HALF_PIX + 1)
        off_lobe = np.concatenate([stokes_v[:lobe_lo], stokes_v[lobe_hi:]])
        noise_sigma = (
            1.4826 * float(median_abs_deviation(off_lobe))
            if len(off_lobe) >= 5
            else 1.4826 * float(median_abs_deviation(stokes_v))
        )
        noise_sigma = max(noise_sigma, 1e-12)
        snr = max_amp / noise_sigma

        v_smoothed = savgol_filter(
            stokes_v, SMOOTHING_WINDOW_SIZE, SMOOTHING_POLY_ORDER
        )
        prominence = max(3.0 * noise_sigma, 0.15 * max_amp)
        peaks_pos, _ = find_peaks(v_smoothed, prominence=prominence)
        peaks_neg, _ = find_peaks(-v_smoothed, prominence=prominence)
        n_pos, n_neg = len(peaks_pos), len(peaks_neg)

        # Anomaly rules
        has_multi = (n_pos + n_neg) >= 3 and (n_pos >= 2 or n_neg >= 2) and snr > 6.0
        v_max, v_min = float(np.max(stokes_v)), float(np.min(stokes_v))
        denom = abs(v_max) + abs(v_min)
        asym = abs(v_max + v_min) / denom if denom > 0 else 0.0
        has_asym = asym > 0.80 and snr > 7.0
        skewness = abs(float(scipy_skew(stokes_v)))
        has_skew = skewness > 4.0 and snr > 6.0

        if has_multi or has_asym or has_skew:
            return "Anomaly"

        # Clear rules
        if snr > 2.5 and n_pos >= 1 and n_neg >= 1:
            opposite = any(
                (int(p) - center_idx) * (int(n) - center_idx) < 0
                for p in peaks_pos
                for n in peaks_neg
            )
            if opposite:
                return "Clear"

        return "Noisy"

    # -------------------------------------------------------------------------
    # Full-dataset classification (vectorised XGBoost or per-slit heuristic)
    # -------------------------------------------------------------------------

    def classify_all_slits(
        self,
        data_cube: np.ndarray,
        center_idx: int,
    ) -> None:
        """
        Classify every slit in the data cube.  Results stored in
        ``self._cache``.

        If XGBoost is available, builds a feature matrix for all slits
        and classifies them in one ``model.predict()`` call.
        Otherwise falls back to per-slit heuristic.
        """
        from collections import Counter

        _, n_slits, _ = data_cube.shape

        if self.use_xgb and self.xgb_model is not None:
            # ── Vectorised XGBoost prediction ─────────────────────
            print(f"[SIS] Classifying {n_slits} slits via XGBoost ...")
            t0 = time.time()

            X = np.zeros((n_slits, self._N_FEATURES), dtype=np.float64)
            for s in range(n_slits):
                X[s, :] = self._extract_feature_vector(data_cube, s, center_idx)

            # XGBoost prediction — integer class codes
            y_pred = self.xgb_model.predict(X)

            # Map integer predictions → string labels
            _INT_TO_LABEL = {
                CLASS_NOISY: "Noisy",
                CLASS_CLEAR: "Clear",
                CLASS_ANOMALY: "Anomaly",
            }
            for s in range(n_slits):
                code = int(y_pred[s])
                self._cache[s] = _INT_TO_LABEL.get(code, "Noisy")

            elapsed = time.time() - t0
            print(f"[SIS] XGBoost classified {n_slits} slits in {elapsed:.2f}s")

        else:
            # ── Heuristic fallback ────────────────────────────────
            print(f"[SIS] Classifying {n_slits} slits via heuristic fallback ...")
            for s in range(n_slits):
                self._cache[s] = self._classify_heuristic(
                    data_cube[3, s, :], center_idx
                )

        # Summary
        dist = Counter(self._cache.values())
        print("[SIS] Classification complete.  Distribution:")
        for cls_name in ("Clear", "Noisy", "Anomaly"):
            cnt = dist.get(cls_name, 0)
            pct = 100.0 * cnt / n_slits
            print(f"       {cls_name:8s}: {cnt:4d}  ({pct:5.1f} %)")

    # -------------------------------------------------------------------------
    # Per-slit prediction  (returns cached results)
    # -------------------------------------------------------------------------

    def predict(
        self,
        stokes_v: np.ndarray,
        wavelengths: np.ndarray,
        center_idx: int,
        slit_index: int = -1,
    ) -> Tuple[str, float, float]:
        """
        Return the cached classification for a slit, plus a simple B_guess.

        Parameters
        ----------
        stokes_v : np.ndarray
            1-D Stokes V spectrum.
        wavelengths : np.ndarray
            Calibrated wavelength axis.
        center_idx : int
            Pixel index of the spectral line centre.
        slit_index : int
            Slit index to look up in the cache.

        Returns
        -------
        signal_class : str
        confidence : float
            95.0 if from XGBoost, 80.0 if from heuristic fallback.
        b_guess : float
            Simple amplitude-based B estimate in Gauss.
        """
        # Retrieve cached classification
        if slit_index in self._cache:
            label = self._cache[slit_index]
            confidence = 95.0 if self.use_xgb else 80.0
        else:
            label = self._classify_heuristic(stokes_v, center_idx)
            confidence = 80.0

        # Simple B_guess from Stokes V amplitude (Zeeman scaling)
        b_guess = 0.0
        if label == "Clear":
            max_v = float(np.max(np.abs(stokes_v)))
            denominator = (
                ZEEMAN_CONSTANT_K * LANDE_FACTOR_EFF * REFERENCE_WAVELENGTH_0 ** 2
            )
            if denominator > 0:
                b_guess = max_v / (denominator * 1e4)
                b_guess = min(b_guess, MAX_REALISTIC_B_GAUSS)

        return label, confidence, b_guess


# =============================================================================
# AI ENGINE — PHASE 3: GEMINI SCIENTIFIC REPORT  (end-of-pipeline only)
# =============================================================================


def generate_gemini_report(
    df_results: pd.DataFrame,
    fits_filename: str,
    output_path: str,
) -> bool:
    """
    Generate a scientific report using the Gemini 2.5 Flash model.

    This function is called **once** at the very end of the pipeline,
    after all physics computations, CSV output, and dashboard plots are
    complete.  It compiles summary statistics from ``df_results`` and
    sends a single prompt to Gemini asking for a 2-3 paragraph executive
    scientific interpretation.

    Uses the **new** Google GenAI SDK (``from google import genai``).

    Parameters
    ----------
    df_results : pd.DataFrame
        Full per-slit results table.
    fits_filename : str
        Base name of the source FITS file.
    output_path : str
        Destination path for the plain-text report.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if generation was skipped or failed.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[AI Report] GEMINI_API_KEY not set.  Skipping report generation.")
        return False
    if not _GENAI_AVAILABLE:
        print("[AI Report] google-genai not installed.  Skipping report generation.")
        return False

    print(f"[AI Report] Generating scientific report via {AI_GEMINI_REPORT_MODEL} ...")

    # ── Compile summary statistics ────────────────────────────────────────
    total = len(df_results)

    def _safe_stats(col: str) -> str:
        if col not in df_results.columns:
            return "N/A"
        vals = pd.to_numeric(df_results[col], errors="coerce").dropna()
        if len(vals) == 0:
            return "N/A"
        return (
            f"N={len(vals)}, mean={vals.mean():.2f}, "
            f"std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}"
        )

    # Routing distribution
    route_dist = "N/A"
    if "AI_Route" in df_results.columns:
        route_dist = df_results["AI_Route"].value_counts().to_string()

    # Signal class distribution
    cls_dist = "N/A"
    if "AI_Signal_Class" in df_results.columns:
        cls_dist = df_results["AI_Signal_Class"].value_counts().to_string()

    n_anomaly = 0
    if "AI_Route" in df_results.columns:
        n_anomaly = int((df_results["AI_Route"] == ROUTE_ANOMALY).sum())

    summary = (
        f"Dataset: {fits_filename}\n"
        f"Total spatial positions: {total}\n\n"
        f"Signal classification:\n{cls_dist}\n\n"
        f"Routing distribution:\n{route_dist}\n\n"
        f"Anomalous profiles: {n_anomaly} ({100.0 * n_anomaly / max(total, 1):.1f}%)\n\n"
        f"Magnetic field (B_G): {_safe_stats('B_G')}\n"
        f"Transverse field (B_transverse): {_safe_stats('B_transverse')}\n"
        f"LOS velocity (km/s): {_safe_stats('LOS_Velocity_COG')}\n"
        f"Turbulence velocity (km/s): {_safe_stats('V_turb_km_s')}\n"
        f"Temperature core (K): {_safe_stats('Temp_core')}\n"
        f"Temperature wing blue (K): {_safe_stats('Temp_wing_blue')}\n"
        f"FWHM observed (A): {_safe_stats('FWHM_obs_A')}\n"
    )

    prompt = (
        "You are an expert solar physicist specialising in Hinode SOT/SP "
        "spectropolarimetric observations.  Below are the computed summary "
        "statistics from a fully processed dataset.\n\n"
        f"{summary}\n"
        "Based on these results, write a professional 2-3 paragraph executive "
        "scientific report covering:\n"
        "1. Assessment of the magnetic field strengths and topology "
        "(active region, quiet Sun, sunspot, plage?).\n"
        "2. Interpretation of the velocity and turbulence distributions — "
        "evidence of convective flows, Evershed effect, or chromospheric dynamics.\n"
        "3. Temperature stratification — are the wing-to-core gradients "
        "consistent with a standard photospheric model?\n"
        "4. Significance of the anomaly fraction and routing distribution.\n"
        "5. Recommendations for follow-up analysis.\n\n"
        "Write in a professional academic style (300-500 words)."
    )

    # ── Call Gemini via the new SDK ────────────────────────────────────────
    try:
        client = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=AI_GEMINI_REPORT_MODEL,
            contents=prompt,
        )
        report_text = response.text

        separator = "=" * 80
        full_report = (
            f"{separator}\n"
            f"AUTOMATED SCIENTIFIC ANALYSIS REPORT\n"
            f"Hinode SP Sigma-V Analyzer — Hybrid AI Pipeline\n"
            f"{separator}\n\n"
            f"File Analysed : {fits_filename}\n"
            f"Analysis Date : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Classifier    : XGBoost (local) + Heuristic fallback\n"
            f"Report Model  : {AI_GEMINI_REPORT_MODEL}\n\n"
            f"{separator}\n\n"
            f"{report_text}\n\n"
            f"{separator}\n"
            f"END OF REPORT\n\n"
            f"Note: This report was generated automatically by an LLM.\n"
            f"Numerical analysis uses established astrophysical methods.\n"
            f"Interpretations should be reviewed by qualified researchers.\n"
            f"{separator}\n"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(full_report)

        print(f"[AI Report] Saved -> {output_path}")
        return True

    except Exception as exc:
        print(f"[AI Report] Gemini report generation failed: {exc}")
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_odd(number: int) -> int:
    """
    Return the nearest odd integer >= 3.

    Used to guarantee valid Savitzky-Golay filter window sizes.

    Parameters
    ----------
    number : int
        Input integer.

    Returns
    -------
    int
        Odd integer, minimum 3.
    """
    n = int(number)
    if n % 2 == 0:
        n -= 1
    if n < 3:
        n = 3
    return n


def build_wavelength_axis(
    header: fits.Header,
    n_pixels: int,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Construct the wavelength array from FITS header WCS keywords.

    Falls back to a linear axis centred on ``REFERENCE_WAVELENGTH_0`` when
    mandatory keywords (``CRVAL1``, ``CDELT1``) are absent.

    Parameters
    ----------
    header : fits.Header
        Primary FITS header.
    n_pixels : int
        Number of spectral pixels.

    Returns
    -------
    wavelengths : np.ndarray
        Wavelength axis in Angstrom.
    dispersion : float
        Dispersion (delta-lambda per pixel) in Angstrom.
    ref_val : float
        Reference wavelength value (``CRVAL1``).
    ref_pix : float
        Reference pixel index (``CRPIX1``).
    """
    crval = header.get("CRVAL1", header.get("CRVAL", None))
    crpix = header.get("CRPIX1", header.get("CRPIX", 1.0))
    cdelt = header.get("CDELT1", header.get("CD1_1", header.get("CDELT", None)))

    if crval is None or cdelt is None:
        wavelengths = np.linspace(
            REFERENCE_WAVELENGTH_0 - 0.5,
            REFERENCE_WAVELENGTH_0 + 0.5,
            n_pixels,
        )
        cdelt = wavelengths[1] - wavelengths[0]
        crval = wavelengths[0]
        crpix = 1.0
    else:
        pixel_indices = np.arange(n_pixels, dtype=float)
        wavelengths = float(crval) + (pixel_indices + 1.0 - float(crpix)) * float(cdelt)

    return (
        np.array(wavelengths),
        float(cdelt),
        float(crval if crval is not None else wavelengths[0]),
        float(crpix),
    )


def calculate_mad_std(data: np.ndarray) -> float:
    """
    Estimate the standard deviation using the Median Absolute Deviation (MAD).

    Equivalent to ``1.4826 * median(|x - median(x)|)``, which is a robust
    scale estimator consistent with the Gaussian standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input 1-D array.

    Returns
    -------
    float
        MAD-based standard-deviation estimate, or ``0.0`` for empty input.
    """
    if data is None or len(data) == 0:
        return 0.0
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def compute_cross_correlation_shift(
    observed: np.ndarray,
    reference: np.ndarray,
) -> int:
    """
    Compute the integer pixel shift between two arrays via FFT cross-correlation.

    Parameters
    ----------
    observed : np.ndarray
        Observed (mean-subtracted) spectrum.
    reference : np.ndarray
        Reference (mean-subtracted) spectrum.

    Returns
    -------
    int
        Pixel shift such that ``observed`` aligns with ``reference``.
    """
    obs_centered = np.asarray(observed) - np.nanmean(observed)
    ref_centered = np.asarray(reference) - np.nanmean(reference)
    correlation = fftconvolve(obs_centered, ref_centered[::-1], mode="same")
    return int(np.argmax(correlation) - (len(correlation) // 2))


# =============================================================================
# CORE PHYSICS FUNCTIONS  (ORIGINAL — UNCHANGED)
# =============================================================================


def robust_find_line_center(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    target_lambda: float,
    window_angstrom: float = 0.6,
) -> Optional[int]:
    """
    Find the pixel index of minimum intensity (line centre) near a target wavelength.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength axis.
    intensity : np.ndarray
        Stokes I spectrum.
    target_lambda : float
        Approximate wavelength of the spectral line in Angstrom.
    window_angstrom : float, optional
        Half-width of the search window in Angstrom.

    Returns
    -------
    int or None
        Pixel index of the detected line centre, or ``None`` on failure.
    """
    n_pixels = len(wavelengths)
    if n_pixels < 3:
        return None

    delta_lambda = abs(wavelengths[1] - wavelengths[0])
    half_window_pix = max(2, int(round(window_angstrom / delta_lambda)))
    pixel_guess = int(np.argmin(np.abs(wavelengths - target_lambda)))

    start_idx = max(0, pixel_guess - half_window_pix)
    end_idx = min(n_pixels, pixel_guess + half_window_pix + 1)

    if end_idx - start_idx < 3:
        return None

    region_intensity = intensity[start_idx:end_idx]
    local_min_idx = int(np.argmin(region_intensity))
    return start_idx + local_min_idx


def solve_linear_wavelength_scale(
    observed_positions: List[float],
    lab_positions: List[float],
) -> Tuple[float, float]:
    """
    Solve for linear scaling coefficients via least-squares.

    Fits the model ``lambda_calibrated = a * lambda_raw + b``.

    Parameters
    ----------
    observed_positions : list of float
        Measured wavelength positions of reference lines.
    lab_positions : list of float
        Laboratory (rest-frame) wavelengths of the same lines.

    Returns
    -------
    slope : float
        Scale factor ``a``.
    intercept : float
        Offset ``b`` in Angstrom.
    """
    obs_arr = np.asarray(observed_positions)
    lab_arr = np.asarray(lab_positions)

    if len(obs_arr) < 2:
        return 1.0, 0.0

    design_matrix = np.vstack([obs_arr, np.ones_like(obs_arr)]).T
    solution, _, _, _ = np.linalg.lstsq(design_matrix, lab_arr, rcond=None)
    return float(solution[0]), float(solution[1])


def calculate_parabolic_centroid(y_data: np.ndarray, peak_index: int) -> float:
    """
    Refine an integer peak position to sub-pixel accuracy via parabolic interpolation.

    Parameters
    ----------
    y_data : np.ndarray
        1-D array containing the peak.
    peak_index : int
        Integer index of the peak apex.

    Returns
    -------
    float
        Sub-pixel peak position.
    """
    idx = int(round(peak_index))
    if idx <= 0 or idx >= len(y_data) - 1:
        return float(idx)

    y_minus = y_data[idx - 1]
    y_center = y_data[idx]
    y_plus = y_data[idx + 1]

    denominator = y_minus - 2 * y_center + y_plus
    if denominator == 0:
        return float(idx)

    return float(idx) + 0.5 * (y_minus - y_plus) / denominator


# =============================================================================
# SIGMA-V DETECTION  (ORIGINAL — UNCHANGED)
# =============================================================================


def analyze_sigma_v_on_spectrum(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    stokes_v: np.ndarray,
    center_pixel_idx: int,
    smooth_window: int = SMOOTHING_WINDOW_SIZE,
    smooth_poly: int = SMOOTHING_POLY_ORDER,
    search_half_width: int = SEARCH_WINDOW_HALF_PIX,
    noise_region_width: int = SIGMA_TIGHT_WINDOW_PIX,
    min_rel_velocity: float = MIN_RELATIVE_AMPLITUDE,
    noise_factor: float = NOISE_THRESHOLD_FACTOR,
) -> Dict[str, Any]:
    """
    Analyse a single Stokes V spectrum to find sigma-component peaks and
    estimate B using the Zeeman splitting (Sigma-V) method.

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis.
    stokes_i : np.ndarray
        Stokes I spectrum.
    stokes_v : np.ndarray
        Stokes V spectrum.
    center_pixel_idx : int
        Pixel index of the line centre.
    smooth_window : int, optional
        Savitzky-Golay window length.
    smooth_poly : int, optional
        Savitzky-Golay polynomial order.
    search_half_width : int, optional
        Half-width (pixels) of the sigma-component search region.
    noise_region_width : int, optional
        Half-width (pixels) of the inner noise-exclusion zone.
    min_rel_velocity : float, optional
        Minimum Stokes V amplitude relative to continuum intensity.
    noise_factor : float, optional
        SNR threshold factor for peak prominence.

    Returns
    -------
    dict
        Keys: ``found``, ``B_G``, ``wa``, ``wb``, ``a_sub``, ``b_sub``,
        ``delta_lambda``, ``V_rel``, ``noise``, ``SNR``, ``suspect``,
        ``suspect_reason``.
    """
    n_pixels = len(wavelengths)
    result: Dict[str, Any] = {
        "found": False,
        "B_G": None,
        "wa": None,
        "wb": None,
        "a_sub": None,
        "b_sub": None,
        "delta_lambda": None,
        "V_rel": None,
        "noise": None,
        "SNR": None,
        "suspect": False,
        "suspect_reason": None,
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
        np.median(stokes_i[: max(1, n_pixels // 12)])
        + np.median(stokes_i[-max(1, n_pixels // 12) :])
    )
    v_amplitude = np.nanmax(np.abs(v_smoothed))
    relative_v = float(v_amplitude / max(abs(continuum_level), 1.0))
    result["V_rel"] = relative_v

    search_start = max(0, center_pixel_idx - search_half_width)
    search_end = min(n_pixels, center_pixel_idx + search_half_width)
    inner_start = max(0, center_pixel_idx - noise_region_width)
    inner_end = min(n_pixels, center_pixel_idx + noise_region_width)

    mask_noise = np.ones_like(v_smoothed, dtype=bool)
    mask_noise[inner_start:inner_end] = False

    noise_samples = v_smoothed[mask_noise] if np.any(mask_noise) else v_smoothed
    noise_level = (
        calculate_mad_std(noise_samples)
        if noise_samples.size > 4
        else calculate_mad_std(v_smoothed)
    )

    result["noise"] = float(noise_level)
    snr = v_amplitude / (noise_level if noise_level > 0 else 1e-12)
    result["SNR"] = float(snr)

    if relative_v < min_rel_velocity or snr < 1.0:
        result["suspect"] = True
        result["suspect_reason"] = "V_too_weak_or_low_SNR"
        return result

    prominence_threshold = max(noise_factor * noise_level, 0.01 * v_amplitude)
    search_region_v = v_smoothed[search_start:search_end]

    pos_peaks_local, _ = find_peaks(search_region_v, prominence=prominence_threshold)
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

    best_pair = max(
        valid_pairs,
        key=lambda pair: abs(v_smoothed[pair[0]]) + abs(v_smoothed[pair[1]]),
    )
    idx_a, idx_b = sorted(best_pair)

    pos_a_sub = calculate_parabolic_centroid(
        v_smoothed if v_smoothed[idx_a] >= 0 else -v_smoothed, idx_a
    )
    pos_b_sub = calculate_parabolic_centroid(
        v_smoothed if v_smoothed[idx_b] >= 0 else -v_smoothed, idx_b
    )

    wavelength_a = float(np.interp(pos_a_sub, np.arange(n_pixels), wavelengths))
    wavelength_b = float(np.interp(pos_b_sub, np.arange(n_pixels), wavelengths))

    delta_lambda = abs(wavelength_b - wavelength_a) / 2.0
    denominator = ZEEMAN_CONSTANT_K * LANDE_FACTOR_EFF * (REFERENCE_WAVELENGTH_0 ** 2)

    if denominator == 0:
        result["suspect"] = True
        result["suspect_reason"] = "zero_denominator_error"
        return result

    magnetic_field_gauss = delta_lambda / denominator

    is_edge_issue = (
        idx_a <= EDGE_SAFETY_MARGIN_PIX
        or idx_b <= EDGE_SAFETY_MARGIN_PIX
        or idx_a >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX
        or idx_b >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX
    )

    if abs(pos_b_sub - pos_a_sub) < 1.5:
        result["suspect"] = True
        result["suspect_reason"] = "peaks_too_close"
        return result

    result.update(
        {
            "found": True,
            "B_G": float(magnetic_field_gauss),
            "wa": wavelength_a,
            "wb": wavelength_b,
            "a_sub": float(pos_a_sub),
            "b_sub": float(pos_b_sub),
            "delta_lambda": float(delta_lambda),
        }
    )

    if is_edge_issue:
        result["suspect"] = True
        result["suspect_reason"] = "peak_on_edge"

    if abs(magnetic_field_gauss) > MAX_REALISTIC_B_GAUSS:
        result["suspect"] = True
        result["suspect_reason"] = "B_out_of_range"
        result["B_G"] = None
        result["found"] = False

    return result


# =============================================================================
# MONTE CARLO ERROR ESTIMATION  (ORIGINAL — UNCHANGED)
# =============================================================================


def estimate_b_error_mc(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    stokes_v: np.ndarray,
    center_pixel_idx: int,
    analysis_function: Callable,
    n_iterations: int = 200,
    noise_estimate: Optional[float] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Estimate B-field measurement uncertainty via Monte Carlo simulation.

    Injects Gaussian noise realisations into the Stokes V profile and
    re-runs the analysis function for each realisation.

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis.
    stokes_i : np.ndarray
        Stokes I spectrum.
    stokes_v : np.ndarray
        Stokes V spectrum.
    center_pixel_idx : int
        Pixel index of the line centre.
    analysis_function : callable
        Physics analysis function with the same signature as
        :func:`analyze_sigma_v_on_spectrum`.
    n_iterations : int, optional
        Number of Monte Carlo iterations.
    noise_estimate : float or None, optional
        Pre-computed noise level; estimated from data if ``None``.
    show_progress : bool, optional
        Print progress every 10 % of iterations.

    Returns
    -------
    dict
        Keys: ``B_median``, ``B_mean``, ``B_std``, ``B_p16``, ``B_p84``,
        ``N_success``.
    """
    n_pixels = len(wavelengths)

    if noise_estimate is None:
        inner_start = max(0, center_pixel_idx - SIGMA_TIGHT_WINDOW_PIX)
        inner_end = min(n_pixels, center_pixel_idx + SIGMA_TIGHT_WINDOW_PIX)
        mask_noise = np.ones_like(stokes_v, dtype=bool)
        mask_noise[inner_start:inner_end] = False
        noise_samples = stokes_v[mask_noise] if np.any(mask_noise) else stokes_v
        sigma_noise = calculate_mad_std(noise_samples)
    else:
        sigma_noise = float(noise_estimate)

    empty_result: Dict[str, Any] = {
        "B_median": None,
        "B_mean": None,
        "B_std": None,
        "B_p16": None,
        "B_p84": None,
        "N_success": 0,
    }
    if sigma_noise <= 0:
        return empty_result

    b_values = []
    rng = np.random.default_rng()

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
        "B_mean": float(np.mean(b_arr)),
        "B_std": float(np.std(b_arr, ddof=1)) if b_arr.size >= 2 else None,
        "B_p16": float(np.percentile(b_arr, 16)) if b_arr.size >= 1 else None,
        "B_p84": float(np.percentile(b_arr, 84)) if b_arr.size >= 1 else None,
        "N_success": int(b_arr.size),
    }


# =============================================================================
# CALIBRATION AND DATA PREPARATION  (ORIGINAL — UNCHANGED)
# =============================================================================


def cross_calibrate_wavelength(
    header: fits.Header,
    intensity_data_2d: np.ndarray,
    wavelengths: np.ndarray,
    lab_lines: Tuple[float, float] = (LINE_LAB_1, LINE_LAB_2),
    use_atlas: bool = False,
    atlas_wav: Optional[np.ndarray] = None,
    atlas_intensity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Perform wavelength calibration using reference atlas matching and/or known lines.

    Attempts two calibration steps in order:

    1. **Atlas shift** (if ``use_atlas=True`` and atlas data provided) —
       cross-correlates the quiet-Sun intensity profile against the reference
       atlas to derive a global pixel shift.
    2. **Line-scale fit** — finds both reference lines in the (possibly
       shifted) spectrum, then solves for a linear scale correction.

    Parameters
    ----------
    header : fits.Header
        Primary FITS header.
    intensity_data_2d : np.ndarray
        2-D Stokes I data, shape ``(n_slits, n_lambda)``.
    wavelengths : np.ndarray
        Initial wavelength axis from WCS.
    lab_lines : tuple of float, optional
        Laboratory wavelengths of the two reference lines.
    use_atlas : bool, optional
        Enable atlas-based shift correction.
    atlas_wav : np.ndarray or None, optional
        Reference atlas wavelength axis.
    atlas_intensity : np.ndarray or None, optional
        Reference atlas intensity spectrum.

    Returns
    -------
    calibrated_wavelengths : np.ndarray
    scale_a : float
    offset_b : float
    info : dict
        Calibration metadata (method, shift, coefficients, etc.).
    """
    current_wavelengths = wavelengths.copy()
    info: Dict[str, Any] = {
        "method": "none",
        "shift_pix": 0,
        "a": 1.0,
        "b": 0.0,
        "used_atlas": False,
        "used_lines": False,
    }

    if use_atlas and (atlas_wav is not None) and (atlas_intensity is not None):
        try:
            ref_i_interp = np.interp(
                wavelengths, atlas_wav, atlas_intensity, left=np.nan, right=np.nan
            )
            quiet_sun_profile = np.median(
                intensity_data_2d[: max(1, intensity_data_2d.shape[0] // 6), :],
                axis=0,
            )
            shift_pix = compute_cross_correlation_shift(
                quiet_sun_profile, ref_i_interp
            )
            delta_lambda_shift = shift_pix * (wavelengths[1] - wavelengths[0])
            current_wavelengths = current_wavelengths + delta_lambda_shift
            info.update(
                {
                    "method": "atlas_shift",
                    "shift_pix": int(shift_pix),
                    "used_atlas": True,
                }
            )
        except Exception as exc:
            print(f"[CALIB] Atlas shift failed: {exc}")

    observed_positions = []
    lab_positions = []
    median_intensity = np.median(intensity_data_2d, axis=0)

    for line_lambda in lab_lines:
        idx = robust_find_line_center(
            current_wavelengths, median_intensity, line_lambda, window_angstrom=0.6
        )
        if idx is not None:
            observed_positions.append(current_wavelengths[idx])
            lab_positions.append(line_lambda)

    if len(observed_positions) >= 2:
        scale_a, offset_b = solve_linear_wavelength_scale(
            observed_positions, lab_positions
        )
        calibrated_wavelengths = scale_a * current_wavelengths + offset_b
        info.update(
            {
                "method": "lines_scale",
                "used_lines": True,
                "a": float(scale_a),
                "b": float(offset_b),
                "obs_positions": observed_positions,
                "lab_positions": lab_positions,
            }
        )
        return calibrated_wavelengths, float(scale_a), float(offset_b), info

    return current_wavelengths, 1.0, 0.0, info


def prepare_me_input_file(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    stokes_q: np.ndarray,
    stokes_u: np.ndarray,
    stokes_v: np.ndarray,
    output_dir: str,
    slit_index: int,
) -> str:
    """
    Write Stokes profiles to a plain-text file for an external ME inversion code.

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis.
    stokes_i, stokes_q, stokes_u, stokes_v : np.ndarray
        Stokes parameter spectra.
    output_dir : str
        Destination directory (created if absent).
    slit_index : int
        Slit position index used to form the filename.

    Returns
    -------
    str
        Full path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"slit_{slit_index:04d}_stokes.txt")
    data_matrix = np.vstack([wavelengths, stokes_i, stokes_q, stokes_u, stokes_v]).T
    np.savetxt(filename, data_matrix, header="wav I Q U V")
    return filename


def slit_index_to_arcsec(
    header: fits.Header,
    slit_idx: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert a slit index to solar coordinates in arcseconds using WCS keywords.

    Parameters
    ----------
    header : fits.Header
        Primary FITS header containing ``XCEN``, ``CRPIX2``, and ``XSCALE``.
    slit_idx : int
        Zero-based slit position index.

    Returns
    -------
    x_arcsec : float or None
        X coordinate in arcseconds, or ``None`` if WCS keywords are absent.
    y_arcsec : float or None
        Y coordinate in arcseconds (``YCEN``), or ``None`` if absent.
    """
    xcen = header.get("XCEN", None)
    crpix2 = header.get("CRPIX2", header.get("CRPIX", None))
    xscale = header.get("XSCALE", header.get("CDELT2", header.get("CDELT", None)))

    if xcen is None or crpix2 is None or xscale is None:
        return None, None

    x_arcsec = float(xcen) + (float(slit_idx) - float(crpix2)) * float(xscale)
    ycen = header.get("YCEN", None)
    y_arcsec = float(ycen) if ycen is not None else None
    return x_arcsec, y_arcsec


# =============================================================================
# WEAK-FIELD APPROXIMATION  (internal helper)
# =============================================================================


def _run_wfa(
    calibrated_wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    stokes_v: np.ndarray,
    idx_center: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Weak-Field Approximation (WFA) B-field estimator.

    Fits the model ``V(lambda) = -C * dI/dlambda * B_los`` to extract the
    line-of-sight magnetic field strength.

    Parameters
    ----------
    calibrated_wavelengths : np.ndarray
        Calibrated wavelength axis.
    stokes_i : np.ndarray
        Stokes I spectrum.
    stokes_v : np.ndarray
        Stokes V spectrum.
    idx_center : int
        Pixel index of the line centre.

    Returns
    -------
    b_wfa : float or None
        Line-of-sight B in Gauss, or ``None`` if the fit was rejected.
    r_wfa : float or None
        Pearson correlation coefficient of the fit, or ``None``.
    """
    try:
        n_lambda = len(calibrated_wavelengths)
        smooth_w = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
        i_smooth = (
            savgol_filter(stokes_i, smooth_w, SMOOTHING_POLY_ORDER)
            if n_lambda >= smooth_w
            else stokes_i
        )

        di_dlambda = np.gradient(i_smooth, calibrated_wavelengths)
        const_c = ZEEMAN_CONSTANT_K * (REFERENCE_WAVELENGTH_0 ** 2) * LANDE_FACTOR_EFF
        reg_x = -const_c * di_dlambda

        seg_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
        seg_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)

        x_seg = reg_x[seg_lo:seg_hi]
        v_seg = stokes_v[seg_lo:seg_hi]
        mask = np.isfinite(x_seg) & np.isfinite(v_seg)

        if mask.sum() < 5:
            return None, None

        design = np.vstack([x_seg[mask], np.ones(mask.sum())]).T
        sol, _, _, _ = np.linalg.lstsq(design, v_seg[mask], rcond=None)
        slope_b = float(sol[0])
        model_v = design @ sol

        residuals = v_seg[mask] - model_v
        resid_noise = calculate_mad_std(residuals)
        seg_snr = np.nanmax(np.abs(v_seg[mask])) / (
            resid_noise if resid_noise > 0 else 1e-12
        )
        corr = (
            float(np.corrcoef(v_seg[mask], model_v)[0, 1])
            if mask.sum() >= 3
            else 0.0
        )

        if np.isfinite(slope_b) and abs(corr) >= 0.4 and seg_snr >= 5.0:
            return slope_b, corr

    except Exception:
        pass

    return None, None


# =============================================================================
# v3.0 — "FANTASTIC FOUR" PHYSICAL PARAMETER EXTRACTION
# =============================================================================


def compute_transverse_b_wfa(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    stokes_q: np.ndarray,
    stokes_u: np.ndarray,
    idx_center: int,
) -> Optional[float]:
    """
    Estimate the transverse magnetic field B_perp via the Weak Field
    Approximation (WFA) for linear polarisation.

    Method
    ------
    In the WFA regime the linear polarisation signal scales with the
    **second derivative** of the intensity profile::

        L(λ) ≈ C₂ · (d²I/dλ²) · B_perp²

    where ``L = sqrt(Q² + U²)``.  We compute d²I/dλ² using two
    successive calls to ``numpy.gradient``, then correlate it with the
    observed total linear polarisation inside a window around the line
    core.  The proportionality constant ``C₂`` is derived from the same
    Zeeman parameters used elsewhere in the pipeline.

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis (Å).
    stokes_i : np.ndarray
        Stokes I spectrum.
    stokes_q : np.ndarray
        Stokes Q spectrum.
    stokes_u : np.ndarray
        Stokes U spectrum.
    idx_center : int
        Pixel index of the spectral line centre.

    Returns
    -------
    float or None
        Estimated transverse field B_perp in Gauss, or ``None`` on
        failure / unphysical result.
    """
    try:
        n_lam = len(wavelengths)
        smooth_w = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lam - 1)))
        i_smooth = (
            savgol_filter(stokes_i, smooth_w, SMOOTHING_POLY_ORDER)
            if n_lam >= smooth_w
            else stokes_i.copy()
        )

        # Second derivative d²I/dλ²
        di_dlam = np.gradient(i_smooth, wavelengths)
        d2i_dlam2 = np.gradient(di_dlam, wavelengths)

        # Total linear polarisation
        lin_pol = np.sqrt(stokes_q ** 2 + stokes_u ** 2)

        # WFA proportionality constant for the transverse component
        # C₂ ∝ (K · g_eff · λ₀²)² — same Zeeman parameters as the LOS WFA
        const_c2 = (
            ZEEMAN_CONSTANT_K * LANDE_FACTOR_EFF * REFERENCE_WAVELENGTH_0 ** 2
        ) ** 2

        # Work in a window around line core
        seg_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
        seg_hi = min(n_lam, idx_center + SIGMA_TIGHT_WINDOW_PIX)

        d2_seg = d2i_dlam2[seg_lo:seg_hi]
        lp_seg = lin_pol[seg_lo:seg_hi]

        mask = np.isfinite(d2_seg) & np.isfinite(lp_seg) & (np.abs(d2_seg) > 1e-20)
        if mask.sum() < 5:
            return None

        # Least-squares fit:  L_seg ≈ (C₂ · B_perp²) · |d²I/dλ²|
        # Solve for the scalar  α = C₂ · B_perp²
        x_fit = const_c2 * np.abs(d2_seg[mask])
        y_fit = lp_seg[mask]
        denom = np.dot(x_fit, x_fit)
        if denom < 1e-30:
            return None
        alpha = np.dot(x_fit, y_fit) / denom

        if alpha <= 0:
            return None

        b_perp = np.sqrt(alpha)
        if not np.isfinite(b_perp) or b_perp > 2 * MAX_REALISTIC_B_GAUSS:
            return None

        return float(b_perp)

    except Exception:
        return None


def _inverse_planck_temperature(
    intensity: float,
    wavelength_cm: float,
) -> float:
    """
    Convert a monochromatic intensity to brightness temperature via the
    inverse Planck function.

    Parameters
    ----------
    intensity : float
        Specific intensity (arbitrary detector units — result is
        meaningful as a *relative* temperature scale).
    wavelength_cm : float
        Wavelength in centimetres.

    Returns
    -------
    float
        Brightness temperature in Kelvin, or ``NaN`` on failure.
    """
    if intensity <= 0 or wavelength_cm <= 0:
        return np.nan
    try:
        prefactor = (2.0 * PLANCK_H * PLANCK_C ** 2) / (wavelength_cm ** 5)
        ratio = prefactor / intensity
        if ratio <= 1.0:
            return np.nan
        t_b = (PLANCK_H * PLANCK_C) / (
            wavelength_cm * BOLTZMANN_K * np.log(ratio)
        )
        return float(t_b) if np.isfinite(t_b) and t_b > 0 else np.nan
    except Exception:
        return np.nan


def _forward_planck_intensity(
    temperature_k: float,
    wavelength_cm: float,
) -> float:
    """
    Compute the Planck specific intensity for a given temperature and
    wavelength (forward Planck function).

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin.
    wavelength_cm : float
        Wavelength in centimetres.

    Returns
    -------
    float
        Specific intensity in CGS units (erg s⁻¹ cm⁻² sr⁻¹ Hz⁻¹),
        or ``NaN`` on failure.
    """
    if temperature_k <= 0 or wavelength_cm <= 0:
        return np.nan
    try:
        prefactor = (2.0 * PLANCK_H * PLANCK_C ** 2) / (wavelength_cm ** 5)
        exponent = (PLANCK_H * PLANCK_C) / (
            wavelength_cm * BOLTZMANN_K * temperature_k
        )
        if exponent > 500:
            return np.nan  # overflow guard
        intensity = prefactor / (np.exp(exponent) - 1.0)
        return float(intensity) if np.isfinite(intensity) else np.nan
    except Exception:
        return np.nan


# Standard photospheric reference temperature for DN → physical calibration
_PHOTOSPHERE_T_REF_K = 6000.0


def compute_temperature_stratification(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    idx_center: int,
) -> Dict[str, Any]:
    """
    Derive a simplified 3-D temperature stratification using the
    Eddington-Barbier approximation.

    Method
    ------
    The Eddington-Barbier relation states that the emergent intensity at
    any wavelength within a spectral line is approximately equal to the
    source function at optical depth τ ≈ 1 at that wavelength.  By
    sampling the Stokes I profile at 5 positions (from the far wings to
    the line core) we probe progressively higher atmospheric layers.  The
    sampled intensities are converted to brightness temperatures via the
    inverse Planck function, yielding a coarse temperature-vs-depth map.

    Sample points
    ~~~~~~~~~~~~~
    0. Far blue wing  (deep photosphere, τ_c ≈ 1 at continuum)
    1. Near blue wing  (mid-photosphere)
    2. Line core       (upper photosphere / temperature minimum)
    3. Near red wing   (mid-photosphere)
    4. Far red wing    (deep photosphere)

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis (Å).
    stokes_i : np.ndarray
        Stokes I spectrum.
    idx_center : int
        Pixel index of the line centre.

    Returns
    -------
    dict
        Keys: ``Temp_wing_blue``, ``Temp_mid_blue``, ``Temp_core``,
        ``Temp_mid_red``, ``Temp_wing_red`` — brightness temperatures
        in Kelvin (``NaN`` for any failed conversion); and
        ``Temp_core_K`` (alias used later for turbulence subtraction).
    """
    n_lam = len(wavelengths)
    half_search = SEARCH_WINDOW_HALF_PIX

    # Five sample indices spanning from far-wing → core → far-wing
    offsets = [
        max(0, idx_center - int(1.8 * half_search)),          # far blue wing
        max(0, idx_center - int(0.6 * half_search)),          # near blue wing
        idx_center,                                            # core
        min(n_lam - 1, idx_center + int(0.6 * half_search)),  # near red wing
        min(n_lam - 1, idx_center + int(1.8 * half_search)),  # far red wing
    ]

    labels = [
        "Temp_wing_blue",
        "Temp_mid_blue",
        "Temp_core",
        "Temp_mid_red",
        "Temp_wing_red",
    ]

    # Reference wavelength in cm for Planck inversion
    lam_cm = REFERENCE_WAVELENGTH_0 * 1e-8  # Å → cm

    # --- v4: DN → physical intensity calibration ----------------------------
    # Raw Stokes I values are detector counts (DN), not physical intensities.
    # We calibrate by assuming the far-wing continuum corresponds to the
    # standard quiet-Sun photospheric temperature (_PHOTOSPHERE_T_REF_K).
    raw_intensities = [float(stokes_i[idx]) for idx in offsets]
    observed_continuum_dn = 0.5 * (raw_intensities[0] + raw_intensities[4])

    if observed_continuum_dn > 0:
        physical_continuum = _forward_planck_intensity(_PHOTOSPHERE_T_REF_K, lam_cm)
        if np.isfinite(physical_continuum) and physical_continuum > 0:
            calib_factor = physical_continuum / observed_continuum_dn
        else:
            calib_factor = 1.0  # graceful fallback
    else:
        calib_factor = 1.0  # graceful fallback

    result: Dict[str, Any] = {}
    for label, idx in zip(labels, offsets):
        calibrated_intensity = float(stokes_i[idx]) * calib_factor
        result[label] = _inverse_planck_temperature(calibrated_intensity, lam_cm)

    # Convenience alias — the core temperature is used by the turbulence
    # calculation to estimate the thermal broadening component.
    result["Temp_core_K"] = result["Temp_core"]

    return result


def compute_los_velocity_cog(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    idx_center: int,
    rest_wavelength: float = REFERENCE_WAVELENGTH_0,
) -> Optional[float]:
    """
    Compute the plasma line-of-sight (LOS) velocity using the Centre of
    Gravity (COG) method on the Stokes I absorption profile.

    Method
    ------
    The COG method calculates the intensity-weighted mean wavelength
    (first moment) of the absorption profile, then converts the shift
    relative to the laboratory rest wavelength into a Doppler velocity::

        λ_cog = Σ (1 − I/Ic) · λ  /  Σ (1 − I/Ic)
        V_los  = c · (λ_cog − λ₀) / λ₀

    where Ic is the continuum intensity estimated from the line wings.

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis (Å).
    stokes_i : np.ndarray
        Stokes I spectrum.
    idx_center : int
        Pixel index of the spectral line centre.
    rest_wavelength : float, optional
        Laboratory rest wavelength in Å (default 6302.5).

    Returns
    -------
    float or None
        LOS Doppler velocity in km/s (positive = red-shift / recession),
        or ``None`` on failure.
    """
    try:
        n_lam = len(wavelengths)
        seg_lo = max(0, idx_center - SEARCH_WINDOW_HALF_PIX)
        seg_hi = min(n_lam, idx_center + SEARCH_WINDOW_HALF_PIX)

        lam_seg = wavelengths[seg_lo:seg_hi]
        i_seg = stokes_i[seg_lo:seg_hi]

        if len(lam_seg) < 5:
            return None

        # Estimate continuum from the outermost 20% of the segment on each side
        n_edge = max(2, len(lam_seg) // 5)
        continuum = 0.5 * (np.median(i_seg[:n_edge]) + np.median(i_seg[-n_edge:]))
        if continuum <= 0:
            return None

        # Absorption depth (positive in the line)
        depth = 1.0 - (i_seg / continuum)
        depth = np.clip(depth, 0.0, None)
        total_depth = np.sum(depth)
        if total_depth < 1e-12:
            return None

        lambda_cog = np.sum(depth * lam_seg) / total_depth
        v_los = C_KM_S * (lambda_cog - rest_wavelength) / rest_wavelength

        if not np.isfinite(v_los) or abs(v_los) > 100.0:
            # > 100 km/s is unphysical for photospheric Fe I
            return None

        return float(v_los)

    except Exception:
        return None


def compute_turbulence_fwhm(
    wavelengths: np.ndarray,
    stokes_i: np.ndarray,
    idx_center: int,
    temp_core_k: float,
) -> Dict[str, Any]:
    """
    Quantify plasma turbulence via non-thermal line broadening.

    Method
    ------
    1. Measure the **total FWHM** of the Stokes I absorption profile by
       finding where the normalised line depth crosses half-maximum on
       each side of the core.
    2. Compute the **thermal FWHM** expected from the temperature at the
       line-formation height (``temp_core_k`` from the Eddington-Barbier
       method)::

           FWHM_thermal = (λ₀ / c) · √(8 ln2 · k_B T / m_Fe)

    3. Subtract in quadrature to isolate the **non-thermal (turbulent)**
       broadening::

           FWHM_turb² = FWHM_obs² − FWHM_thermal²
           V_turb      = c · FWHM_turb / (λ₀ · 2√(2 ln2))

    Parameters
    ----------
    wavelengths : np.ndarray
        Calibrated wavelength axis (Å).
    stokes_i : np.ndarray
        Stokes I spectrum.
    idx_center : int
        Pixel index of the spectral line centre.
    temp_core_k : float
        Brightness temperature at the line core (K) — used to compute
        the thermal broadening component.

    Returns
    -------
    dict
        Keys: ``FWHM_obs_A`` (observed FWHM in Å),
        ``FWHM_thermal_A`` (thermal FWHM in Å),
        ``FWHM_turb_A`` (non-thermal FWHM in Å),
        ``V_turb_km_s`` (turbulent velocity in km/s).
        Values are ``None`` where computation is impossible.
    """
    empty: Dict[str, Any] = {
        "FWHM_obs_A": None,
        "FWHM_thermal_A": None,
        "FWHM_turb_A": None,
        "V_turb_km_s": None,
    }
    try:
        n_lam = len(wavelengths)
        seg_lo = max(0, idx_center - SEARCH_WINDOW_HALF_PIX)
        seg_hi = min(n_lam, idx_center + SEARCH_WINDOW_HALF_PIX)

        lam_seg = wavelengths[seg_lo:seg_hi]
        i_seg = stokes_i[seg_lo:seg_hi]

        if len(lam_seg) < 7:
            return empty

        # Continuum and normalisation
        n_edge = max(2, len(lam_seg) // 5)
        continuum = 0.5 * (np.median(i_seg[:n_edge]) + np.median(i_seg[-n_edge:]))
        if continuum <= 0:
            return empty

        depth = 1.0 - (i_seg / continuum)
        depth = np.clip(depth, 0.0, None)
        max_depth = np.max(depth)
        if max_depth < 1e-6:
            return empty

        half_max = max_depth / 2.0
        core_local = idx_center - seg_lo

        # Find half-max crossings on each side of the core
        blue_idx = None
        for k in range(core_local - 1, -1, -1):
            if depth[k] <= half_max:
                # Linear interpolation between k and k+1
                frac = (half_max - depth[k]) / (depth[k + 1] - depth[k] + 1e-30)
                blue_idx = lam_seg[k] + frac * (lam_seg[k + 1] - lam_seg[k])
                break

        red_idx = None
        for k in range(core_local + 1, len(depth)):
            if depth[k] <= half_max:
                frac = (half_max - depth[k]) / (depth[k - 1] - depth[k] + 1e-30)
                red_idx = lam_seg[k] + frac * (lam_seg[k - 1] - lam_seg[k])
                break

        if blue_idx is None or red_idx is None:
            return empty

        fwhm_obs = abs(red_idx - blue_idx)
        if fwhm_obs <= 0:
            return empty

        result: Dict[str, Any] = {"FWHM_obs_A": float(fwhm_obs)}

        # Thermal FWHM
        if np.isfinite(temp_core_k) and temp_core_k > 0:
            # Gaussian thermal width:  Δλ_th = (λ₀/c) √(8 ln2 kT / m)
            thermal_arg = (
                8.0 * np.log(2) * BOLTZMANN_K * temp_core_k / FE56_ATOMIC_MASS_G
            )
            if thermal_arg > 0:
                v_thermal = np.sqrt(thermal_arg)  # cm/s
                # λ₀ in cm, then convert result back to Å
                lam0_cm = REFERENCE_WAVELENGTH_0 * 1e-8
                fwhm_thermal = (lam0_cm * v_thermal / PLANCK_C) * 1e8  # Å
                result["FWHM_thermal_A"] = float(fwhm_thermal)
            else:
                result["FWHM_thermal_A"] = None
        else:
            result["FWHM_thermal_A"] = None

        # Non-thermal (turbulent) component via quadrature subtraction
        if result["FWHM_thermal_A"] is not None:
            diff2 = fwhm_obs ** 2 - result["FWHM_thermal_A"] ** 2
            if diff2 > 0:
                fwhm_turb = np.sqrt(diff2)
                result["FWHM_turb_A"] = float(fwhm_turb)
                # Convert turbulent FWHM to velocity:
                #   V = c · FWHM / (λ₀ · 2√(2ln2))
                gauss_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
                v_turb = C_KM_S * fwhm_turb / (
                    REFERENCE_WAVELENGTH_0 * gauss_factor
                )
                result["V_turb_km_s"] = float(v_turb)
            else:
                # Thermal broadening dominates — no detectable turbulence
                result["FWHM_turb_A"] = 0.0
                result["V_turb_km_s"] = 0.0
        else:
            result["FWHM_turb_A"] = None
            result["V_turb_km_s"] = None

        return result

    except Exception:
        return empty


# =============================================================================
# MAIN ANALYSIS PIPELINE — AI-FIRST ARCHITECTURE
# =============================================================================


def analyze_fits_file(
    fits_path: str,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    n_mc_iterations: int = DEFAULT_MC_ITERATIONS,
    run_me_prep: bool = False,
    me_cmd_template: Optional[str] = None,
    use_atlas: bool = USE_ATLAS_REF,
    atlas_wav: Optional[np.ndarray] = None,
    atlas_intensity: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Main driver: load FITS, calibrate, then dispatch each pixel via SIS.

    Pipeline steps (per slit position)
    ------------------------------------
    Step 1 — SIS evaluation: ``signal_class``, ``confidence``, ``b_guess``.

    Step 2 — AI-driven dispatch:

      * ``"Clear"``   -> :func:`analyze_sigma_v_on_spectrum` (Zeeman).
      * ``"Noisy"``   -> :func:`_run_wfa`; if WFA fails ->
                         :func:`estimate_b_error_mc` (500 iterations).
      * ``"Anomaly"`` -> Profile archived; heavy physics skipped.

    Step 3 — Monte Carlo error estimation for Clear/Noisy pixels with valid B.

    Step 4 — Results recorded with full AI routing metadata.

    The SIS is **not optional** — there is no ``--no-ai`` mode.

    Parameters
    ----------
    fits_path : str
        Path to the Hinode SP FITS file.
    output_prefix : str, optional
        Prefix for all output filenames.
    n_mc_iterations : int, optional
        Monte Carlo iterations for error estimation (capped at 500).
    run_me_prep : bool, optional
        If ``True``, write ME input files for the top 20 slits by B strength.
    me_cmd_template : str or None, optional
        External ME solver command template (not invoked here).
    use_atlas : bool, optional
        Enable atlas-based wavelength shift correction.
    atlas_wav : np.ndarray or None, optional
        Reference atlas wavelength axis.
    atlas_intensity : np.ndarray or None, optional
        Reference atlas intensity spectrum.

    Returns
    -------
    pd.DataFrame
        Per-slit results table with physics and AI routing columns.

    Raises
    ------
    FileNotFoundError
        If ``fits_path`` does not exist.
    RuntimeError
        If scikit-learn is unavailable or the FITS shape is unsupported.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    # -------------------------------------------------------------------------
    # Initialise AI engines
    # -------------------------------------------------------------------------
    print("[SIS] Initialising SolarIntelligenceSystem ...")
    sis = SolarIntelligenceSystem()

    # -------------------------------------------------------------------------
    # Load FITS data
    # -------------------------------------------------------------------------
    with fits.open(fits_path, memmap=True) as hdul:
        header = hdul[0].header
        raw_data = hdul[0].data

        if raw_data is None:
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None:
                    raw_data = hdu.data
                    header = hdu.header
                    break

        if raw_data is None:
            raise RuntimeError("No array data found in FITS HDUs.")

        data_cube = raw_data.astype(float)

    # -------------------------------------------------------------------------
    # Validate and normalise 3-D shape: (4, n_slits, n_lambda)
    # -------------------------------------------------------------------------
    if data_cube.ndim != 3:
        raise RuntimeError(
            f"FITS data must be 3D (4, Nslit, Nlambda). Got: {data_cube.shape}"
        )

    if data_cube.shape[0] != 4:
        if data_cube.shape[2] == 4:
            data_cube = np.transpose(data_cube, (2, 0, 1))
        elif data_cube.shape[1] == 4:
            data_cube = np.transpose(data_cube, (1, 0, 2))
        else:
            raise RuntimeError(
                f"Unexpected FITS shape.  Expected first axis = 4 (Stokes).  "
                f"Got: {data_cube.shape}"
            )

    _, n_slits, n_lambda = data_cube.shape

    # -------------------------------------------------------------------------
    # Wavelength construction and calibration
    # -------------------------------------------------------------------------
    raw_wavelengths, cdelt, _, _ = build_wavelength_axis(header, n_lambda)
    print(f"[INFO] {fits_path}  shape: {data_cube.shape}")
    print(
        f"[INFO] lambda[0]={raw_wavelengths[0]:.6f}  "
        f"lambda[-1]={raw_wavelengths[-1]:.6f}  d_lambda={cdelt:.6f} A"
    )

    intensity_data = data_cube[0, :, :]
    calibrated_wavelengths, _, _, calib_info = cross_calibrate_wavelength(
        header,
        intensity_data,
        raw_wavelengths,
        lab_lines=(LINE_LAB_1, LINE_LAB_2),
        use_atlas=use_atlas,
        atlas_wav=atlas_wav,
        atlas_intensity=atlas_intensity,
    )
    print(f"[CALIB] {calib_info}")

    # Reference line-centre pixel
    idx_center = int(np.argmin(np.abs(calibrated_wavelengths - REFERENCE_WAVELENGTH_0)))
    idx_center = max(0, min(n_lambda - 1, idx_center))

    # -------------------------------------------------------------------------
    # Pre-classify all slits via LLM (or heuristic fallback)
    # -------------------------------------------------------------------------
    sis.classify_all_slits(data_cube, idx_center)

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------
    examples_dir = f"{output_prefix}_examples"
    anomaly_dir = f"{output_prefix}_anomalies"
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)

    results_list: List[Dict[str, Any]] = []
    b_profile_array = np.full(n_slits, np.nan)
    sigma_b_profile = np.full(n_slits, np.nan)

    route_counts: Dict[str, int] = {
        ROUTE_SIGMA_V: 0,
        ROUTE_WFA_NOISY: 0,
        ROUTE_MC_NOISY: 0,
        ROUTE_ANOMALY: 0,
        ROUTE_NO_SIGNAL: 0,
    }

    # =========================================================================
    # MAIN LOOP — one iteration per slit position
    # =========================================================================
    for s_idx in range(n_slits):

        # Spatial averaging (optional window)
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

        # --- Step 1: SIS evaluation — AI decides what happens next -----------
        signal_class, ai_confidence, b_guess = sis.predict(
            stokes_v, calibrated_wavelengths, idx_center, slit_index=s_idx
        )

        # Per-slit accumulators
        b_final_val: Optional[float] = None
        wfa_correlation: Optional[float] = None
        used_method = "none"
        ai_route = "Unknown"
        sigma_res: Dict[str, Any] = {
            "found": False,
            "B_G": None,
            "SNR": None,
            "noise": None,
            "V_rel": None,
            "suspect": False,
            "suspect_reason": None,
            "wa": None,
            "wb": None,
            "a_sub": None,
            "b_sub": None,
            "delta_lambda": None,
        }
        mc_stats: Dict[str, Any] = {
            "B_median": None,
            "B_mean": None,
            "B_std": None,
            "B_p16": None,
            "B_p84": None,
            "N_success": 0,
        }

        # --- Step 2: AI-driven dispatch ---------------------------------------

        if signal_class == "Clear":
            # Route A: Standard Sigma-V (Zeeman splitting)
            sigma_res = analyze_sigma_v_on_spectrum(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center
            )

            if sigma_res.get("found") and sigma_res.get("B_G") is not None:
                b_final_val = sigma_res["B_G"]
                used_method = "sigma"
                ai_route = ROUTE_SIGMA_V
            else:
                # Sigma-V failed on a Clear profile (e.g. edge artefact);
                # fall through gracefully to WFA.
                b_wfa, r_wfa = _run_wfa(
                    calibrated_wavelengths, stokes_i, stokes_v, idx_center
                )
                if b_wfa is not None:
                    b_final_val = b_wfa
                    wfa_correlation = r_wfa
                    used_method = "wfa"
                    ai_route = ROUTE_WFA_NOISY  # same pool, labelled distinctly
                else:
                    ai_route = ROUTE_SIGMA_V  # attempted but no detection

            route_counts[ROUTE_SIGMA_V] += 1

        elif signal_class == "Noisy":
            # Route B: WFA first; if WFA fails, accept no detection.
            # v4: MC brute-force on pure noise has been removed — it
            # hallucinated extreme B-fields (2500+ G) from random peaks.
            b_wfa, r_wfa = _run_wfa(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center
            )

            if b_wfa is not None:
                b_final_val = b_wfa
                wfa_correlation = r_wfa
                used_method = "wfa"
                ai_route = ROUTE_WFA_NOISY
                route_counts[ROUTE_WFA_NOISY] += 1
            else:
                # WFA gave no result — do NOT brute-force MC on noise.
                # Leave b_final_val = None; pixel is genuinely below the
                # detection threshold.
                ai_route = ROUTE_NO_SIGNAL
                used_method = "none"
                route_counts[ROUTE_NO_SIGNAL] += 1

        elif signal_class == "Anomaly":
            # Route C: Flag pixel; archive profile; skip heavy physics
            anomaly_file = os.path.join(
                anomaly_dir, f"anomaly_slit_{s_idx:04d}.npy"
            )
            np.save(
                anomaly_file,
                np.vstack(
                    [calibrated_wavelengths, stokes_i, stokes_q, stokes_u, stokes_v]
                ),
            )

            ai_route = ROUTE_ANOMALY
            used_method = "anomaly"
            route_counts[ROUTE_ANOMALY] += 1

            # 4-панельный график для аномалий
            fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
            
            axes[0].plot(calibrated_wavelengths, stokes_i, color="steelblue")
            axes[0].set_ylabel("Stokes I")
            
            axes[1].plot(calibrated_wavelengths, stokes_q, color="darkorange")
            axes[1].set_ylabel("Stokes Q")
            
            axes[2].plot(calibrated_wavelengths, stokes_u, color="forestgreen")
            axes[2].set_ylabel("Stokes U")
            
            axes[3].plot(calibrated_wavelengths, stokes_v, color="crimson", lw=1.2)
            axes[3].set_ylabel("Stokes V [ANOMALY]")
            axes[3].set_xlabel("Wavelength (Å)")
            
            for ax in axes:
                ax.grid(True, alpha=0.4)
                
            fig.suptitle(
                f"ANOMALY - Slit {s_idx}  AI conf={ai_confidence:.1f}%\n"
                f"Profile archived to {anomaly_dir}/",
                fontsize=10,
                color="darkred",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                os.path.join(examples_dir, f"ANOMALY_slit_{s_idx:04d}.png"), dpi=150
            )
            plt.close(fig)

        # --- Step 2b: v3.0 "Fantastic Four" parameter extraction ---------------
        # These lightweight physics methods run for every non-anomaly slit,
        # regardless of the AI routing outcome.  Results are appended to the
        # output row dict below.

        ff_b_transverse: Optional[float] = None
        ff_temp: Dict[str, Any] = {
            "Temp_wing_blue": None, "Temp_mid_blue": None,
            "Temp_core": None, "Temp_mid_red": None,
            "Temp_wing_red": None, "Temp_core_K": np.nan,
        }
        ff_v_los: Optional[float] = None
        ff_turb: Dict[str, Any] = {
            "FWHM_obs_A": None, "FWHM_thermal_A": None,
            "FWHM_turb_A": None, "V_turb_km_s": None,
        }

        if signal_class != "Anomaly":
            # 1. Transverse B from linear polarisation WFA
            ff_b_transverse = compute_transverse_b_wfa(
                calibrated_wavelengths, stokes_i, stokes_q, stokes_u, idx_center
            )

            # 2. Temperature stratification (Eddington-Barbier)
            ff_temp = compute_temperature_stratification(
                calibrated_wavelengths, stokes_i, idx_center
            )

            # 3. LOS velocity (Centre-of-Gravity Doppler)
            ff_v_los = compute_los_velocity_cog(
                calibrated_wavelengths, stokes_i, idx_center
            )

            # 4. Turbulence (non-thermal line broadening)
            temp_for_turb = ff_temp.get("Temp_core_K", np.nan)
            if temp_for_turb is None or not np.isfinite(temp_for_turb):
                temp_for_turb = 5000.0  # fallback: typical photospheric T
            ff_turb = compute_turbulence_fwhm(
                calibrated_wavelengths, stokes_i, idx_center, temp_for_turb
            )

        # --- Step 3: Monte Carlo error estimation (only if B was found) -------
        # v4: MC is now purely for error bars, never for field discovery.
        # It only runs when a valid b_final_val was already determined by
        # Sigma-V or WFA in Steps 1-2.
        if (
            b_final_val is not None
            and signal_class != "Anomaly"
        ):
            noise_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
            noise_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)
            mc_mask = np.ones_like(stokes_v, dtype=bool)
            mc_mask[noise_lo:noise_hi] = False
            mc_noise = calculate_mad_std(
                stokes_v[mc_mask] if mc_mask.any() else stokes_v
            )

            mc_stats = estimate_b_error_mc(
                calibrated_wavelengths,
                stokes_i,
                stokes_v,
                idx_center,
                analyze_sigma_v_on_spectrum,
                n_iterations=max(50, min(n_mc_iterations, 500)),
                noise_estimate=mc_noise,
            )

        # --- Step 4: Record results with full AI routing metadata ------------
        x_arc, y_arc = slit_index_to_arcsec(header, s_idx)

        row: Dict[str, Any] = {
            "slit": int(s_idx),
            "x_pix": int(s_idx),
            "x_arcsec": float(x_arc) if x_arc is not None else None,
            "y_arcsec": float(y_arc) if y_arc is not None else None,
            "idx_center": int(idx_center),
            "used": used_method,
            # SIS decision
            "AI_Signal_Class": signal_class,
            "AI_Confidence": float(ai_confidence),
            "AI_B_guess": float(b_guess),
            "AI_Route": ai_route,
            # Sigma-V fields
            "B_sigma_G": (
                float(sigma_res["B_G"])
                if (sigma_res.get("found") and sigma_res.get("B_G") is not None)
                else None
            ),
            "B_wfa_G": (
                float(b_final_val)
                if (used_method in ("wfa", "sigma") and b_final_val is not None)
                else None
            ),
            "wfa_r": float(wfa_correlation) if wfa_correlation is not None else None,
            "V_rel": (
                float(sigma_res["V_rel"])
                if sigma_res.get("V_rel") is not None
                else None
            ),
            "noise": (
                float(sigma_res["noise"])
                if sigma_res.get("noise") is not None
                else None
            ),
            "SNR": (
                float(sigma_res["SNR"])
                if sigma_res.get("SNR") is not None
                else None
            ),
            "sigma_found": bool(sigma_res.get("found", False)),
            "wa_A": (
                float(sigma_res["wa"]) if sigma_res.get("wa") is not None else None
            ),
            "wb_A": (
                float(sigma_res["wb"]) if sigma_res.get("wb") is not None else None
            ),
            "delta_lambda_A": (
                float(sigma_res["delta_lambda"])
                if sigma_res.get("delta_lambda") is not None
                else None
            ),
            "suspect": bool(sigma_res.get("suspect", False)),
            "suspect_reason": sigma_res.get("suspect_reason"),
            # MC error statistics
            "B_MC_median": mc_stats["B_median"],
            "B_MC_mean": mc_stats["B_mean"],
            "B_MC_std": mc_stats["B_std"],
            "B_p16": mc_stats["B_p16"],
            "B_p84": mc_stats["B_p84"],
            "B_MC_n_success": mc_stats["N_success"],
            # --- v3.0 Fantastic Four columns ---
            "B_transverse": (
                float(ff_b_transverse) if ff_b_transverse is not None else None
            ),
            "Temp_wing_blue": ff_temp.get("Temp_wing_blue"),
            "Temp_mid_blue": ff_temp.get("Temp_mid_blue"),
            "Temp_core": ff_temp.get("Temp_core"),
            "Temp_mid_red": ff_temp.get("Temp_mid_red"),
            "Temp_wing_red": ff_temp.get("Temp_wing_red"),
            "LOS_Velocity_COG": (
                float(ff_v_los) if ff_v_los is not None else None
            ),
            "FWHM_obs_A": ff_turb.get("FWHM_obs_A"),
            "FWHM_thermal_A": ff_turb.get("FWHM_thermal_A"),
            "FWHM_turb_A": ff_turb.get("FWHM_turb_A"),
            "V_turb_km_s": ff_turb.get("V_turb_km_s"),
        }

        # Best B estimate: prefer Sigma-V, fall back to WFA / MC median.
        b_choice = row["B_sigma_G"]
        if b_choice is None:
            b_choice = b_final_val if used_method in ("wfa", "mc_brute") else None
        row["B_G"] = float(b_choice) if b_choice is not None else None

        results_list.append(row)
        b_profile_array[s_idx] = row["B_G"] if row["B_G"] is not None else np.nan
        sigma_b_profile[s_idx] = (
            float(mc_stats["B_std"]) if mc_stats["B_std"] is not None else np.nan
        )

        # Diagnostic plot for non-anomaly slits with a detection or suspect flag
        if signal_class != "Anomaly" and (row["sigma_found"] or row["suspect"]):
            fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
            
            axes[0].plot(calibrated_wavelengths, stokes_i, label="I", color="steelblue")
            axes[0].set_ylabel("Stokes I")
            
            axes[1].plot(calibrated_wavelengths, stokes_q, label="Q", color="darkorange")
            axes[1].set_ylabel("Stokes Q")
            
            axes[2].plot(calibrated_wavelengths, stokes_u, label="U", color="forestgreen")
            axes[2].set_ylabel("Stokes U")

            win_len = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
            v_smooth_plot = (
                savgol_filter(stokes_v, win_len, SMOOTHING_POLY_ORDER)
                if n_lambda >= win_len
                else stokes_v
            )
            axes[3].plot(calibrated_wavelengths, stokes_v, label="V_raw", alpha=0.55)
            axes[3].plot(
                calibrated_wavelengths,
                v_smooth_plot,
                label="V_smooth",
                color="red",
                lw=1.2,
            )

            if row["wa_A"] is not None:
                axes[3].axvline(row["wa_A"], color="magenta", ls="--", label="sigma-")
            if row["wb_A"] is not None:
                axes[3].axvline(row["wb_A"], color="cyan", ls="--", label="sigma+")

            axes[3].set_ylabel("Stokes V")
            axes[3].set_xlabel("Wavelength (Å)")
            
            for ax in axes:
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=8, loc="upper right")

            b_str = f"{row['B_G']:.0f} G" if row["B_G"] is not None else "-"
            b_perp_str = f"{row['B_transverse']:.0f} G" if row.get("B_transverse") is not None else "N/A"
            x_str = f"{row['x_arcsec']:.1f}\"" if row["x_arcsec"] is not None else "N/A"
            
            title = (
                f"Slit {s_idx}  X={x_str} | B_LOS={b_str} | B_perp={b_perp_str} | method={row['used']}\n"
                f"SIS: {signal_class} ({ai_confidence:.1f}%)  Route -> {ai_route}"
            )
            fig.suptitle(title, fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            out_png = os.path.join(examples_dir, f"slit_{s_idx:04d}.png")
            try:
                plt.savefig(out_png, dpi=150)
            except Exception:
                pass
            plt.close(fig)

        if s_idx % 50 == 0:
            print(f"[PROGRESS] {s_idx}/{n_slits}  routes so far: {route_counts}")

    # =========================================================================
    # POST-LOOP OUTPUTS
    # =========================================================================

    df_results = pd.DataFrame(results_list)

    # CSV output
    csv_file = f"{output_prefix}_results.csv"
    df_results.to_csv(csv_file, index=False, float_format="%.6f")
    print(f"[INFO] Results CSV -> {csv_file}")

    np.save(f"{output_prefix}_B_profile.npy", b_profile_array)

    # B-field profile plot
    x_axis = np.arange(len(b_profile_array))
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, b_profile_array, marker="o", lw=0.8)
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("B (Gauss)")
    plt.title("Magnetic Field Profile")
    plt.savefig(f"{output_prefix}_B_profile.png", dpi=200)
    plt.close()

    # Sigma-B (uncertainty) profile plot
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, sigma_b_profile, marker="o", lw=0.8, color="darkorange")
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("sigma_B (Gauss)")
    plt.title("Magnetic Field Uncertainty (Monte Carlo)")
    plt.savefig(f"{output_prefix}_B_sigma_profile.png", dpi=200)
    plt.close()

    # AI routing distribution bar chart
    if "AI_Route" in df_results.columns:
        route_vc = df_results["AI_Route"].value_counts()
        _bar_palette = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#607D8B"]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            route_vc.index,
            route_vc.values,
            color=_bar_palette[: len(route_vc)],
        )
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_xlabel("AI Route")
        ax.set_ylabel("Count")
        ax.set_title("SIS Dispatch Routing Distribution")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_routing_distribution.png", dpi=200)
        plt.close()
        print(f"[PLOT] Routing chart -> {output_prefix}_routing_distribution.png")

    # 1-D arcsec B-map
    try:
        coords = df_results["x_arcsec"].values
        if np.all(~pd.isna(coords)):
            plt.figure(figsize=(8, 4))
            sc = plt.scatter(
                coords,
                np.zeros_like(coords),
                c=df_results["B_G"].values,
                cmap="inferno",
                s=15,
            )
            plt.colorbar(sc, label="B (Gauss)")
            plt.xlabel("X (arcsec)")
            plt.title("B along scan")
            plt.savefig(f"{output_prefix}_B_map_scan.png", dpi=200)
            plt.close()
    except Exception:
        pass

    # 2-D B-map
    try:
        print("[PLOT] Building 2D B-map ...")
        b_map = np.full((n_slits, 1), np.nan)
        for s_idx in range(n_slits):
            i_p = data_cube[0, s_idx, :]
            v_p = data_cube[3, s_idx, :]
            res2d = analyze_sigma_v_on_spectrum(
                calibrated_wavelengths, i_p, v_p, idx_center
            )
            if res2d.get("found") and res2d.get("B_G") is not None:
                b_map[s_idx, 0] = res2d["B_G"]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            b_map.T,
            aspect="auto",
            cmap="RdBu_r",
            extent=[0, n_slits, 0, 1],
            origin="lower",
        )
        plt.colorbar(im, label="B (Gauss)")
        plt.xlabel("Slit Index")
        plt.ylabel("Pixel along slit")
        plt.title("2D Magnetic Field Map (Sigma-V)")
        plt.savefig(f"{output_prefix}_B_map_2D.png", dpi=200)
        plt.close()
        print(f"[PLOT] 2D map -> {output_prefix}_B_map_2D.png")
    except Exception as exc:
        print(f"[WARN] 2D B-map failed: {exc}")

    # =========================================================================
    # v3.0 — ADVANCED SCIENTIFIC VISUALIZATION SUITE
    # =========================================================================
    #
    # All plots use data already present in df_results.  NaN values are
    # handled gracefully via masking.  Each standalone figure is saved at
    # 250 dpi; the master dashboard at 300 dpi.
    # =========================================================================

    print("[VIZ-v3] Generating Advanced Scientific Visualization Suite ...")

    # --- shared style configuration ------------------------------------------
    _VIZ_DPI = 250
    _DASH_DPI = 300
    _FONT_TITLE = 13
    _FONT_LABEL = 11
    _FONT_TICK = 9
    _EDGE_COLOR = "#2c3e50"
    _PALETTE = {
        "b_los": "#1a5276",
        "b_trans": "#c0392b",
        "turb_fill": "#8e44ad",
        "turb_line": "#6c3483",
        "velocity": "#16a085",
    }

    if _SNS_AVAILABLE:
        sns.set_context("notebook", font_scale=0.95)

    slit_axis = df_results["slit"].values.astype(float)

    # -------------------------------------------------------------------------
    # Plot 1 — LOS Doppler Velocity Map
    # -------------------------------------------------------------------------
    try:
        v_los = pd.to_numeric(
            df_results["LOS_Velocity_COG"], errors="coerce"
        ).values.astype(float)
        valid_v = np.isfinite(v_los)

        if valid_v.sum() >= 3:
            v_abs_max = max(np.nanmax(np.abs(v_los[valid_v])), 0.1)

            fig_vel, ax_vel = plt.subplots(figsize=(12, 4.0))
            norm_vel = mcolors.TwoSlopeNorm(
                vmin=-v_abs_max, vcenter=0.0, vmax=v_abs_max
            )
            sc_vel = ax_vel.scatter(
                slit_axis[valid_v],
                v_los[valid_v],
                c=v_los[valid_v],
                cmap="coolwarm",
                norm=norm_vel,
                s=18,
                edgecolors="k",
                linewidths=0.25,
                zorder=3,
            )
            ax_vel.axhline(0, color="grey", ls="--", lw=0.7, zorder=1)

            # Connect dots with a faint line for trend visibility
            ax_vel.plot(
                slit_axis[valid_v],
                v_los[valid_v],
                color="silver",
                lw=0.5,
                zorder=2,
            )

            cb_vel = fig_vel.colorbar(sc_vel, ax=ax_vel, pad=0.015)
            cb_vel.set_label(
                "V$_{\\mathrm{LOS}}$ (km s$^{-1}$)",
                fontsize=_FONT_LABEL,
            )
            cb_vel.ax.tick_params(labelsize=_FONT_TICK)

            ax_vel.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
            ax_vel.set_ylabel(
                "V$_{\\mathrm{LOS}}$ (km s$^{-1}$)",
                fontsize=_FONT_LABEL,
            )
            ax_vel.set_title(
                "Line-of-Sight Doppler Velocity  (Centre-of-Gravity)",
                fontsize=_FONT_TITLE,
                fontweight="bold",
            )
            ax_vel.tick_params(labelsize=_FONT_TICK)
            ax_vel.grid(True, alpha=0.30, ls=":")
            fig_vel.tight_layout()
            vel_path = f"{output_prefix}_LOS_velocity_COG.png"
            fig_vel.savefig(vel_path, dpi=_VIZ_DPI)
            plt.close(fig_vel)
            print(f"[VIZ-v3] LOS velocity map   -> {vel_path}")
        else:
            print("[VIZ-v3] Skipping LOS velocity map (insufficient valid data).")
    except Exception as exc:
        print(f"[VIZ-v3] LOS velocity plot failed: {exc}")

    # -------------------------------------------------------------------------
    # Plot 2 — Plasma Turbulence Profile
    # -------------------------------------------------------------------------
    try:
        v_turb = pd.to_numeric(
            df_results["V_turb_km_s"], errors="coerce"
        ).values.astype(float)
        valid_t = np.isfinite(v_turb)

        if valid_t.sum() >= 3:
            fig_turb, ax_turb = plt.subplots(figsize=(12, 4.0))

            ax_turb.fill_between(
                slit_axis[valid_t],
                0,
                v_turb[valid_t],
                color=_PALETTE["turb_fill"],
                alpha=0.25,
                label="Non-thermal broadening envelope",
            )
            ax_turb.plot(
                slit_axis[valid_t],
                v_turb[valid_t],
                color=_PALETTE["turb_line"],
                lw=1.0,
                marker=".",
                markersize=3,
                label="V$_{\\mathrm{turb}}$",
            )

            ax_turb.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
            ax_turb.set_ylabel(
                "V$_{\\mathrm{turb}}$ (km s$^{-1}$)",
                fontsize=_FONT_LABEL,
            )
            ax_turb.set_title(
                "Plasma Turbulence Profile  (Non-thermal Line Broadening)",
                fontsize=_FONT_TITLE,
                fontweight="bold",
            )
            ax_turb.tick_params(labelsize=_FONT_TICK)
            ax_turb.legend(fontsize=_FONT_TICK, loc="upper right")
            ax_turb.grid(True, alpha=0.30, ls=":")
            ax_turb.set_ylim(bottom=0)
            fig_turb.tight_layout()
            turb_path = f"{output_prefix}_turbulence_profile.png"
            fig_turb.savefig(turb_path, dpi=_VIZ_DPI)
            plt.close(fig_turb)
            print(f"[VIZ-v3] Turbulence profile -> {turb_path}")
        else:
            print("[VIZ-v3] Skipping turbulence plot (insufficient valid data).")
    except Exception as exc:
        print(f"[VIZ-v3] Turbulence plot failed: {exc}")

    # -------------------------------------------------------------------------
    # Plot 3 — Magnetic Vector Overview  (B_LOS + B_transverse)
    # -------------------------------------------------------------------------
    try:
        b_los_arr = pd.to_numeric(
            df_results["B_G"], errors="coerce"
        ).values.astype(float)
        b_trn_arr = pd.to_numeric(
            df_results["B_transverse"], errors="coerce"
        ).values.astype(float)

        v_los_ok = np.isfinite(b_los_arr)
        v_trn_ok = np.isfinite(b_trn_arr)
        has_los = v_los_ok.sum() >= 3
        has_trn = v_trn_ok.sum() >= 3

        if has_los or has_trn:
            fig_mag, ax_mag = plt.subplots(figsize=(12, 5.0))

            if has_los:
                ax_mag.plot(
                    slit_axis[v_los_ok],
                    b_los_arr[v_los_ok],
                    color=_PALETTE["b_los"],
                    lw=1.1,
                    marker="o",
                    markersize=3,
                    label="B$_{\\mathrm{LOS}}$ (Sigma-V / WFA)",
                    zorder=3,
                )

            if has_trn:
                ax_mag.plot(
                    slit_axis[v_trn_ok],
                    b_trn_arr[v_trn_ok],
                    color=_PALETTE["b_trans"],
                    lw=1.1,
                    marker="s",
                    markersize=3,
                    label="B$_{\\perp}$ (Linear-pol WFA)",
                    zorder=3,
                )

            # Total field magnitude where both components exist
            both_ok = v_los_ok & v_trn_ok
            if both_ok.sum() >= 3:
                b_total = np.sqrt(b_los_arr[both_ok] ** 2 + b_trn_arr[both_ok] ** 2)
                ax_mag.fill_between(
                    slit_axis[both_ok],
                    0,
                    b_total,
                    color="#f0b27a",
                    alpha=0.18,
                    label="|B$_{\\mathrm{total}}$| envelope",
                    zorder=1,
                )

            ax_mag.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
            ax_mag.set_ylabel("Magnetic Field (Gauss)", fontsize=_FONT_LABEL)
            ax_mag.set_title(
                "Magnetic Vector Overview — LOS + Transverse Components",
                fontsize=_FONT_TITLE,
                fontweight="bold",
            )
            ax_mag.tick_params(labelsize=_FONT_TICK)
            ax_mag.legend(fontsize=_FONT_TICK, loc="upper right")
            ax_mag.grid(True, alpha=0.30, ls=":")
            fig_mag.tight_layout()
            mag_path = f"{output_prefix}_magnetic_vector_overview.png"
            fig_mag.savefig(mag_path, dpi=_VIZ_DPI)
            plt.close(fig_mag)
            print(f"[VIZ-v3] Magnetic vector    -> {mag_path}")
        else:
            print("[VIZ-v3] Skipping magnetic vector plot (insufficient data).")
    except Exception as exc:
        print(f"[VIZ-v3] Magnetic vector plot failed: {exc}")

    # -------------------------------------------------------------------------
    # Plot 4 — 2D Temperature Stratification Heatmap (Tomography)
    # -------------------------------------------------------------------------
    _TEMP_COLS = [
        "Temp_wing_blue",
        "Temp_mid_blue",
        "Temp_core",
        "Temp_mid_red",
        "Temp_wing_red",
    ]
    _DEPTH_LABELS = [
        "Wing (blue)\nDeep phot.",
        "Mid (blue)",
        "Line core\nUpper phot.",
        "Mid (red)",
        "Wing (red)\nDeep phot.",
    ]

    try:
        temp_cols_present = [c for c in _TEMP_COLS if c in df_results.columns]
        if len(temp_cols_present) == 5:
            temp_matrix = (
                df_results[temp_cols_present]
                .apply(pd.to_numeric, errors="coerce")
                .values
            )  # shape (n_slits, 5)

            # At least some finite values required
            if np.isfinite(temp_matrix).sum() > 10:
                fig_temp, ax_temp = plt.subplots(figsize=(14, 4.5))

                t_min_robust = np.nanpercentile(temp_matrix[np.isfinite(temp_matrix)], 2)
                t_max_robust = np.nanpercentile(temp_matrix[np.isfinite(temp_matrix)], 98)

                im_temp = ax_temp.imshow(
                    temp_matrix.T,
                    aspect="auto",
                    cmap="inferno",
                    origin="lower",
                    extent=[
                        slit_axis[0], slit_axis[-1],
                        -0.5, 4.5,
                    ],
                    vmin=t_min_robust,
                    vmax=t_max_robust,
                    interpolation="bilinear",
                )

                ax_temp.set_yticks(range(5))
                ax_temp.set_yticklabels(_DEPTH_LABELS, fontsize=_FONT_TICK)
                ax_temp.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
                ax_temp.set_ylabel(
                    "Atmospheric Depth  (τ sampling)",
                    fontsize=_FONT_LABEL,
                )
                ax_temp.set_title(
                    "2D Temperature Stratification  (Eddington-Barbier / "
                    "Inverse Planck)",
                    fontsize=_FONT_TITLE,
                    fontweight="bold",
                )
                ax_temp.tick_params(axis="x", labelsize=_FONT_TICK)

                cb_temp = fig_temp.colorbar(
                    im_temp, ax=ax_temp, pad=0.015, fraction=0.03
                )
                cb_temp.set_label(
                    "Brightness Temperature (K)", fontsize=_FONT_LABEL
                )
                cb_temp.ax.tick_params(labelsize=_FONT_TICK)

                fig_temp.tight_layout()
                temp_path = f"{output_prefix}_temperature_stratification.png"
                fig_temp.savefig(temp_path, dpi=_VIZ_DPI)
                plt.close(fig_temp)
                print(f"[VIZ-v3] Temp stratification -> {temp_path}")
            else:
                print("[VIZ-v3] Skipping temp heatmap (too few finite values).")
        else:
            print("[VIZ-v3] Skipping temp heatmap (columns missing).")
    except Exception as exc:
        print(f"[VIZ-v3] Temperature heatmap failed: {exc}")

    # =========================================================================
    # Plot 5 — THE "INVESTOR DASHBOARD"  (Master multi-panel figure)
    # =========================================================================
    try:
        print("[VIZ-v3] Composing Master Dashboard ...")

        fig_dash = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(
            3, 2,
            figure=fig_dash,
            height_ratios=[1.0, 1.0, 1.2],
            hspace=0.38,
            wspace=0.28,
        )

        # ---- Title banner ---------------------------------------------------
        fig_dash.suptitle(
            "SIS v3.0  —  Solar Plasma Diagnostic Dashboard",
            fontsize=18,
            fontweight="bold",
            color=_EDGE_COLOR,
            y=0.975,
        )
        _subtitle = (
            f"FITS: {os.path.basename(fits_path)}   |   "
            f"Slits: {n_slits}   |   "
            f"λ₀ = {REFERENCE_WAVELENGTH_0:.1f} Å   |   "
            f"g_eff = {LANDE_FACTOR_EFF}"
        )
        fig_dash.text(
            0.5, 0.955, _subtitle,
            ha="center", va="top",
            fontsize=10, color="grey", style="italic",
        )

        # ==== Panel A  (top-left) — Magnetic Vector Overview =================
        ax_a = fig_dash.add_subplot(gs[0, 0])
        _blos = pd.to_numeric(df_results["B_G"], errors="coerce").values.astype(float)
        _btrn = pd.to_numeric(
            df_results["B_transverse"], errors="coerce"
        ).values.astype(float)
        _ok_l = np.isfinite(_blos)
        _ok_t = np.isfinite(_btrn)

        if _ok_l.sum() >= 2:
            ax_a.plot(
                slit_axis[_ok_l], _blos[_ok_l],
                color=_PALETTE["b_los"], lw=0.9, marker=".", ms=2,
                label="B$_{\\mathrm{LOS}}$",
            )
        if _ok_t.sum() >= 2:
            ax_a.plot(
                slit_axis[_ok_t], _btrn[_ok_t],
                color=_PALETTE["b_trans"], lw=0.9, marker=".", ms=2,
                label="B$_{\\perp}$",
            )
        _both = _ok_l & _ok_t
        if _both.sum() >= 2:
            _bt = np.sqrt(_blos[_both] ** 2 + _btrn[_both] ** 2)
            ax_a.fill_between(
                slit_axis[_both], 0, _bt,
                color="#f0b27a", alpha=0.15,
                label="|B$_{\\mathrm{tot}}$|",
            )
        ax_a.set_ylabel("B (Gauss)", fontsize=_FONT_LABEL)
        ax_a.set_title(
            "A · Magnetic Vector Components",
            fontsize=_FONT_TITLE - 1, fontweight="bold", loc="left",
        )
        ax_a.legend(fontsize=7, ncol=3, loc="upper right")
        ax_a.grid(True, alpha=0.25, ls=":")
        ax_a.tick_params(labelsize=_FONT_TICK)

        # ==== Panel B  (top-right) — LOS Doppler Velocity ====================
        ax_b = fig_dash.add_subplot(gs[0, 1])
        _vlos = pd.to_numeric(
            df_results["LOS_Velocity_COG"], errors="coerce"
        ).values.astype(float)
        _ok_v = np.isfinite(_vlos)

        if _ok_v.sum() >= 2:
            _vm = max(np.nanmax(np.abs(_vlos[_ok_v])), 0.05)
            _nv = mcolors.TwoSlopeNorm(vmin=-_vm, vcenter=0.0, vmax=_vm)
            sc_b = ax_b.scatter(
                slit_axis[_ok_v], _vlos[_ok_v],
                c=_vlos[_ok_v], cmap="coolwarm", norm=_nv,
                s=12, edgecolors="k", linewidths=0.2, zorder=3,
            )
            ax_b.axhline(0, color="grey", ls="--", lw=0.5)
            ax_b.plot(
                slit_axis[_ok_v], _vlos[_ok_v],
                color="silver", lw=0.35, zorder=2,
            )
            _cb_b = fig_dash.colorbar(sc_b, ax=ax_b, pad=0.01, fraction=0.04)
            _cb_b.set_label("km/s", fontsize=8)
            _cb_b.ax.tick_params(labelsize=7)
        ax_b.set_ylabel(
            "V$_{\\mathrm{LOS}}$ (km s$^{-1}$)", fontsize=_FONT_LABEL
        )
        ax_b.set_title(
            "B · LOS Doppler Velocity  (COG)",
            fontsize=_FONT_TITLE - 1, fontweight="bold", loc="left",
        )
        ax_b.grid(True, alpha=0.25, ls=":")
        ax_b.tick_params(labelsize=_FONT_TICK)

        # ==== Panel C  (mid-left) — Turbulence Profile ========================
        ax_c = fig_dash.add_subplot(gs[1, 0])
        _vt = pd.to_numeric(
            df_results["V_turb_km_s"], errors="coerce"
        ).values.astype(float)
        _ok_vt = np.isfinite(_vt)

        if _ok_vt.sum() >= 2:
            ax_c.fill_between(
                slit_axis[_ok_vt], 0, _vt[_ok_vt],
                color=_PALETTE["turb_fill"], alpha=0.22,
            )
            ax_c.plot(
                slit_axis[_ok_vt], _vt[_ok_vt],
                color=_PALETTE["turb_line"], lw=0.8, marker=".", ms=2,
            )
        ax_c.set_ylabel(
            "V$_{\\mathrm{turb}}$ (km s$^{-1}$)", fontsize=_FONT_LABEL
        )
        ax_c.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
        ax_c.set_title(
            "C · Plasma Turbulence  (Non-thermal broadening)",
            fontsize=_FONT_TITLE - 1, fontweight="bold", loc="left",
        )
        ax_c.set_ylim(bottom=0)
        ax_c.grid(True, alpha=0.25, ls=":")
        ax_c.tick_params(labelsize=_FONT_TICK)

        # ==== Panel D  (mid-right) — AI Routing Pie + Stats ==================
        ax_d = fig_dash.add_subplot(gs[1, 1])
        if "AI_Route" in df_results.columns:
            _rc = df_results["AI_Route"].value_counts()
            _pie_colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0", "#607D8B"]
            _colors_used = _pie_colors[: len(_rc)]
            wedges, texts, autotexts = ax_d.pie(
                _rc.values,
                labels=_rc.index,
                autopct="%1.1f%%",
                colors=_colors_used,
                startangle=140,
                textprops={"fontsize": 8},
                wedgeprops={"edgecolor": "white", "linewidth": 1.2},
            )
            for at in autotexts:
                at.set_fontsize(7)
                at.set_fontweight("bold")
            ax_d.set_title(
                "D · SIS AI Routing Distribution",
                fontsize=_FONT_TITLE - 1, fontweight="bold", loc="left",
            )

            # Compact stats text box
            _n_valid_b = int(df_results["B_G"].notna().sum())
            _mean_b_str = (
                f"{df_results['B_G'].dropna().mean():.0f}"
                if _n_valid_b > 0 else "—"
            )
            _n_v = int(
                pd.to_numeric(
                    df_results["LOS_Velocity_COG"], errors="coerce"
                ).notna().sum()
            )
            _n_bt = int(
                pd.to_numeric(
                    df_results["B_transverse"], errors="coerce"
                ).notna().sum()
            )
            _stats_txt = (
                f"B$_{{LOS}}$ detections: {_n_valid_b} / {n_slits}\n"
                f"Mean |B$_{{LOS}}$|: {_mean_b_str} G\n"
                f"B$_{{\\perp}}$ detections: {_n_bt}\n"
                f"V$_{{LOS}}$ detections: {_n_v}"
            )
            ax_d.text(
                -0.15, -0.08, _stats_txt,
                transform=ax_d.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#ecf0f1",
                    edgecolor="#bdc3c7",
                    alpha=0.9,
                ),
            )
        else:
            ax_d.text(
                0.5, 0.5, "AI routing data\nnot available",
                ha="center", va="center", fontsize=12, color="grey",
                transform=ax_d.transAxes,
            )
            ax_d.set_title("D · SIS AI Routing", fontsize=_FONT_TITLE - 1)

        # ==== Panel E  (bottom, full width) — Temperature Stratification ======
        ax_e = fig_dash.add_subplot(gs[2, :])
        _tcols = [
            "Temp_wing_blue", "Temp_mid_blue",
            "Temp_core",
            "Temp_mid_red", "Temp_wing_red",
        ]
        _d_labels = [
            "Wing (blue)\nDeep",
            "Mid (blue)",
            "Core\nUpper",
            "Mid (red)",
            "Wing (red)\nDeep",
        ]
        _tcols_ok = [c for c in _tcols if c in df_results.columns]

        if len(_tcols_ok) == 5:
            _tm = (
                df_results[_tcols_ok]
                .apply(pd.to_numeric, errors="coerce")
                .values
            )
            if np.isfinite(_tm).sum() > 10:
                _t_lo = np.nanpercentile(_tm[np.isfinite(_tm)], 2)
                _t_hi = np.nanpercentile(_tm[np.isfinite(_tm)], 98)
                _im_e = ax_e.imshow(
                    _tm.T,
                    aspect="auto",
                    cmap="inferno",
                    origin="lower",
                    extent=[slit_axis[0], slit_axis[-1], -0.5, 4.5],
                    vmin=_t_lo,
                    vmax=_t_hi,
                    interpolation="bilinear",
                )
                ax_e.set_yticks(range(5))
                ax_e.set_yticklabels(_d_labels, fontsize=_FONT_TICK)
                _cb_e = fig_dash.colorbar(
                    _im_e, ax=ax_e, pad=0.01, fraction=0.025
                )
                _cb_e.set_label("T$_b$ (K)", fontsize=_FONT_LABEL)
                _cb_e.ax.tick_params(labelsize=7)
            else:
                ax_e.text(
                    0.5, 0.5, "Insufficient temperature data",
                    ha="center", va="center", transform=ax_e.transAxes,
                    fontsize=12, color="grey",
                )
        else:
            ax_e.text(
                0.5, 0.5, "Temperature columns not available",
                ha="center", va="center", transform=ax_e.transAxes,
                fontsize=12, color="grey",
            )

        ax_e.set_xlabel("Slit Index", fontsize=_FONT_LABEL)
        ax_e.set_ylabel("Atmospheric Depth", fontsize=_FONT_LABEL)
        ax_e.set_title(
            "E · 2D Temperature Stratification  "
            "(Eddington-Barbier Tomography)",
            fontsize=_FONT_TITLE - 1, fontweight="bold", loc="left",
        )
        ax_e.tick_params(axis="x", labelsize=_FONT_TICK)

        # ---- Footer ----------------------------------------------------------
        fig_dash.text(
            0.5, 0.005,
            "Generated by SIS v3.0 — Solar Intelligence System  |  "
            "Hinode/SP Spectropolarimetric Analysis Pipeline  |  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center", va="bottom",
            fontsize=7.5, color="#7f8c8d", style="italic",
        )

        dash_path = f"{output_prefix}_SIS_v3_Master_Dashboard.png"
        fig_dash.savefig(
            dash_path, dpi=_DASH_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig_dash)
        print(f"[VIZ-v3] Master Dashboard   -> {dash_path}")
        print("[VIZ-v3] Advanced Scientific Visualization Suite complete.")

    except Exception as exc:
        print(f"[VIZ-v3] Master Dashboard generation failed: {exc}")
        import traceback
        traceback.print_exc()

    # Phase 3: Gemini AI scientific report (single API call, end of pipeline)
    report_path = f"{output_prefix}_{AI_REPORT_FILENAME}"
    generate_gemini_report(df_results, os.path.basename(fits_path), report_path)

    # ME solver preparation (optional)
    if run_me_prep and me_cmd_template is not None:
        me_input_dir = f"{output_prefix}_ME_input"
        os.makedirs(me_input_dir, exist_ok=True)
        top_slits = (
            df_results[df_results["B_G"].notna()]
            .sort_values("B_G", ascending=False)["slit"]
            .tolist()[:20]
        )
        for s in top_slits:
            s_idx = int(s)
            s_lo = max(0, s_idx - (SPATIAL_AVERAGING_WINDOW // 2))
            s_hi = min(n_slits, s_lo + SPATIAL_AVERAGING_WINDOW)
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
                calibrated_wavelengths, i_s, q_s, u_s, v_s, me_input_dir, s_idx
            )
            print(f"[ME] Prepared input -> {infile}")

    print(f"[INFO] Anomalous profiles archived -> {anomaly_dir}/")
    print(f"[INFO] Diagnostic plots             -> {examples_dir}/")

    return df_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main_cli() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Hinode SP Sigma-V Analyzer - SIS v3.0 (Fantastic Four)",
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
    "Clear"   -> Sigma-V (Zeeman)
    "Noisy"   -> WFA fallback or Monte Carlo brute-force (500 iterations)
    "Anomaly" -> Flagged; archived for manual scientific review

  There is no --no-ai mode.  The AI IS the pipeline.
        """,
    )
    parser.add_argument(
        "--fits", default=DEFAULT_FITS_FILE, help="Path to input FITS file"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUTPUT_PREFIX, help="Output filename prefix"
    )
    parser.add_argument(
        "--mc",
        type=int,
        default=DEFAULT_MC_ITERATIONS,
        help="Monte Carlo iterations for error estimation",
    )
    parser.add_argument(
        "--atlas",
        action="store_true",
        help="Enable Atlas-based wavelength calibration",
    )
    parser.add_argument(
        "--run_me",
        action="store_true",
        help="Prepare Milne-Eddington inversion input files",
    )

    args = parser.parse_args()

    separator = "=" * 80
    print(separator)
    print("Hinode SP Sigma-V Analyzer - SIS v3.0 (Fantastic Four)")
    print(separator)
    print(f"Input FITS       : {args.fits}")
    print(f"Output prefix    : {args.out}")
    print(f"MC iterations    : {args.mc}")
    print(f"SIS dispatcher   : ENABLED (mandatory)")
    print(separator)

    start_time = time.time()

    atlas_wav = atlas_int = None
    if args.atlas and REF_ATLAS_WAV_PATH is not None:
        try:
            atlas_data = np.loadtxt(REF_ATLAS_WAV_PATH)
            atlas_wav = atlas_data[:, 0]
            atlas_int = atlas_data[:, 1]
            print(f"[ATLAS] Loaded -> {REF_ATLAS_WAV_PATH}")
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
        print(separator)
        print(f"Analysis complete in {elapsed:.1f} s")
        print(separator)

        if "AI_Signal_Class" in df_result.columns:
            print("\n-- SIS Signal Classification --")
            print(df_result["AI_Signal_Class"].value_counts(dropna=False).to_string())

        if "AI_Route" in df_result.columns:
            print("\n-- AI Routing Distribution --")
            print(df_result["AI_Route"].value_counts(dropna=False).to_string())

        if "used" in df_result.columns:
            print("\n-- Physics Method Usage --")
            print(df_result["used"].value_counts(dropna=False).to_string())

        if "suspect" in df_result.columns:
            print(f"\nSuspect measurements : {int(df_result['suspect'].sum())}")

        print("\n-- Output files --")
        print(f"  {args.out}_results.csv")
        print(f"  {args.out}_B_profile.png")
        print(f"  {args.out}_B_sigma_profile.png")
        print(f"  {args.out}_routing_distribution.png")
        print(f"  {args.out}_examples/   (diagnostic plots)")
        print(f"  {args.out}_anomalies/  (flagged Stokes V archives)")
        print(f"  {args.out}_{AI_REPORT_FILENAME}  (if Gemini API key set)")
        print(separator)

    except Exception as error:
        print(f"[ERROR] Analysis failed: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main_cli()
