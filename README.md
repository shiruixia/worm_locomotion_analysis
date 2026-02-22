# 🧬 Worm Locomotion Analysis  
### A Computational Workflow for Quantifying Body-Bending Amplitude and Period in *C. elegans*

---

## 📌 Overview

This repository provides the complete Python implementation of the computational workflow described in:

**Shi R., Sun Y., Xiao J., Di Z., Liu H.**  
*A Computational Protocol for Quantifying Body-Bending Amplitude and Period in C. elegans*

This pipeline extracts **body-bending amplitude and period** from centroid trajectory data without full-body posture reconstruction.

### Method Overview

1. Smooth centroid trajectories to obtain a locomotion baseline  
2. Construct a one-dimensional displacement signal  
3. Apply Savitzky–Golay filtering  
4. Detect extrema using peak detection  
5. Quantify bending amplitude and period  

The workflow supports:

- Low-resolution imaging systems  
- Batch processing  
- Automated quality control  
- Publication-ready visualization  

---

## 📂 Repository Structure
```bash
worm_locomotion_analysis/
│
├── worm_locomotion_analysis.py
├── environment.yml
├── data/
├── results/
└── README.md
```

---

## 💻 Requirements

Tested on:

- Windows 10 / 11  
- macOS 10.14+  
- Ubuntu 18.04+  

Minimum:
- 4-core CPU  
- 16 GB RAM  

Recommended:
- 8-core CPU  
- 32 GB RAM  

---

## 🧪 Installation

### Step 1 — Install Anaconda

Download from:  
https://www.anaconda.com/

### Step 2 — Create Environment

```bash
conda env create -f environment.yml
conda activate worm_analysis
```


### Step 3 — Run Analysis
python worm_locomotion_analysis.py

## 📥 Input Data Format

Each .xlsx file must contain:

Column	Description
x	Centroid X coordinate (mm)
y	Centroid Y coordinate (mm)

Each row corresponds to one video frame.

Centroid coordinates should be exported from:

Nematode Trajectory Analysis.exe

## 🔬 Algorithm Workflow
### 1️⃣ Quality Control

Automatic detection of:

Abnormal displacement jumps

IQR-based outliers

Near-static trajectories

Fragmented recordings

Key parameters:

QC_MIN_MEDIAN_STEP = 1e-6
QC_JUMP_FACTOR = 8.0
QC_IQR_FACTOR = 4.0
QC_MIN_VALID_FRACTION = 0.6
PAD_SEC = 1
### 2️⃣ Baseline Extraction

Moving average smoothing:

BASELINE_SMOOTH_SEC = 0.5
### 3️⃣ Distance Signal Construction

Distance from raw trajectory to smoothed baseline:

Distance(t) = sqrt[(x - x_baseline)^2 + (y - y_baseline)^2]
### 4️⃣ Savitzky–Golay Filtering
SG_WINDOW_SEC = 1.0
SG_POLYORDER = 3
### 5️⃣ Peak Detection

Using:

scipy.signal.find_peaks()

Constraints:

MIN_PEAK_WIDTH_SEC = 0.1
MAX_PEAK_WIDTH_SEC = 3.5
### 6️⃣ Feature Extraction

Amplitude = Peak − adjacent Valley

Period = Time between consecutive maxima

## 📊 Output

### 📄 Excel Summary

Saved to:

results/analysis_results.xlsx

Includes:

- `avg_velocity`
- `period_mean`
- `period_median`
- `amp_mean`
- `amp_median`
- `num_cycles`
- `qc_bad_fraction`

---

### 📈 Diagnostic Figures

Saved to:

results/figures/

Includes:

- Trajectory vs baseline  
- Distance function  
- Peak detection visualization  
- Zoomed trajectory segments  

**Resolution:** 300 dpi

## ⚠️ Limitations

Does not reconstruct full-body posture

Not optimized for omega turns

Does not distinguish forward vs reverse cycles

Requires sufficient frame rate

## 👩‍🔬 Contacts

Lead Contact
He Liu
heliu@bnu.edu.cn

Technical Contact
Ruixia Shi
ruixiashi@mail.bnu.edu.cn
