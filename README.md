# ☀️ SIS v8: Solar Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v8.0_Standalone-success.svg)](https://github.com/spectral-hooper/Solar-Intelligence-System-SIS/releases/latest)
[![XGBoost](https://img.shields.io/badge/AI-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Gemini AI](https://img.shields.io/badge/AI-Gemini-blueviolet.svg)](https://ai.google.dev/)

**An AI-first pipeline for real-time solar magnetic field analysis and space weather prediction.**
   
## 🚀 Overview
**SIS (Solar Intelligence System)** is a B2B software solution designed for aerospace corporations and observatories. It processes raw spectropolarimetric data from the Hinode satellite to monitor solar magnetic anomalies. 

Unlike traditional methods that take weeks to process data, SIS uses a **Hybrid AI Architecture** to reduce processing time to **milliseconds**. This allows operators to predict solar flares and put satellites into safe mode before geomagnetic storms hit.

---

## 🛠️ How to Run the Application
We provide two ways to run the Solar Intelligence System.

### Option 1: Standalone Application (Recommended)
For the easiest experience, we have compiled the entire system (including the AI models and physics engines) into a ready-to-use package.

1. Go to the **Releases** tab and download the `SIS_v8_Windows_x64.zip` archive.
2. **Extract the entire folder** to your PC. *(Important: Do not separate the `.exe` file from the `_internal` folder).*
3. Double-click the `.exe` file. **No Python installation is needed.** The XGBoost model and all libraries are safely bundled inside.

### Option 2: Run from Source (For developers)
If you want to run the raw Python code and review the architecture, the source files are located in the `main_files_for_manual_start` directory.

1. Clone the repository and navigate to the source folder:
```bash
git clone [https://github.com/spectral-hooper/Solar-Intelligence-System-SIS.git](https://github.com/spectral-hooper/Solar-Intelligence-System-SIS.git)
cd Solar-Intelligence-System-SIS/main_files_for_manual_start
```
Install requirements:

```bash
pip install -r requirements.txt
```
(Required packages: PySide6, numpy, pandas, astropy, scipy, xgboost, matplotlib, google-genai)

Launch the GUI:

```bash
python sis_gui_app_v2.py
```
🧠 Core Architecture: How it Works
Our system does not use "Black Box" AI to guess physical laws. Instead, we use a hybrid approach: AI for speed, Physics for accuracy, LLMs for interpretation.

1. The AI Router (XGBoost)
Before any heavy math starts, our pre-trained XGBoost machine learning model scans every pixel of the solar spectrum. It acts as an intelligent traffic cop, routing the data into 3 categories:

🟢 Clear: Strong signal. Sent to fast Zeeman splitting algorithms.

🟡 Noisy: Weak signal. Sent to WFA (Weak Field Approximation) algorithms.

🔴 Anomaly: Strange or dangerous magnetic behavior. Saved immediately for deep analysis.

Why this is innovative: By filtering out empty space and routing data dynamically, XGBoost saves 85% of computational time.

2. The Physics Engine
After the AI sorts the data, our deterministic physics engine extracts four critical parameters to predict solar flares:

Magnetic Field (Zeeman & WFA): Measures the tension of solar magnetic lines (the "size" of the threat).

Plasma LOS Velocity (Centre-of-Gravity): Tracks how fast the plasma is moving and twisting the magnetic fields.

Plasma Turbulence (Non-thermal Broadening): Measures the internal chaos and vibration of the plasma right before a flare explodes.

3D Temperature Stratification (Eddington-Barbier): Acts like an MRI for the Sun, showing how heat rises from deep layers to the surface.

3. Reliability (Monte-Carlo Simulation)
To guarantee scientific accuracy, the system uses stochastic Monte-Carlo simulations. It artificially adds noise to the data and runs 100+ iterations to calculate exact error margins.

4. Automated LLM Reporting (Powered by Gemini)
Finally, the system compiles all the raw numbers and uses an integrated LLM Agent (Google Gemini 2.5 Flash). The AI analyzes the data and generates a readable, human-friendly scientific report for the operator, summarizing the magnetic topology, temperature gradients, and potential flare risks.

📁 Repository Structure
`main_files_for_manual_start/` — Source code directory for manual execution.

`sis_gui_app_v2.py` — The front-end desktop application. Built with PySide6, featuring a mission-control dashboard and live console.

`NEW_solar_analyzer_SIS_v8_fixed.py` — The heavy backend containing the XGBoost router, physical formulas, and Gemini API integration.

`xgb_model.pkl` — The pre-trained weights for our AI Router.

`SIS_Runs/` — The automated workspace folder where all generated dashboards, CSVs, and AI reports are saved.

`LICENSE` — Proprietary licensing document.

⚖️ License & Copyright
© 2026 SpectrumTeam. All rights reserved.

This is a proprietary closed-source B2B product. The source code is published strictly for architectural review by the AEROO Space AI Competition jury. Protected by Kazpatent (Registration Number: 69497).

You may not copy, modify, distribute, or reverse-engineer this software. For full details, see the LICENSE file in the root directory.
