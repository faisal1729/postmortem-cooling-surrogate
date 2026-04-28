# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository contains the implementation accompanying the master's 
thesis:

> **Faisal Hussain Shah** (2026). *Machine Learning in Forensic Medicine - 
> Time of Death (ToD) Estimation*. 
> Master's thesis, Universität Potsdam / Zuse Institute Berlin.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

Post-mortem body cooling follows the heat equation over a 
three-dimensional anatomical domain. Direct finite element (FE) 
simulation of this process is accurate but computationally expensive.
This repository implements a surrogate 
modelling pipeline that replaces the FE solver with a Gaussian process 
(GP) regression model trained on a small adaptive dataset of FE 
simulations, reducing prediction time to milliseconds while maintaining 
accuracy within 0.1°C.

The pipeline proceeds in three stages:

```
Physical parameters x
        ↓
FE simulation (Kaskade)  →  cooling curve T(t)
        ↓
Marshall-Hoare NLS fitting  →  parameters (A, B)
        ↓
GP surrogate prediction  →  (Â, B̂) for new configurations
        ↓
Cooling curve reconstruction via MH model
```

---

## Repository Structure

```
├── README.md
├── LICENSE
├── config.py              # Parameter ranges, paths, GP settings
├── core_functions.py      # MH model, NLS fitting, GP model,
│                          # acquisition function
├── adaptive_loop.py       # Adaptive simulation design and GP training
├── kaskadeio.py           # Subprocess wrapper for Kaskade FE solver
└── data/
    ├── training_data.csv          # Physical parameters + fitted (A,B)
    │                              # for 100 training simulations
    ├── test_data.csv              # Physical parameters + fitted (A,B)
    │                              # for 16 test simulations
    ├── adaptive_metrics.csv       # GP uncertainty and prediction error
    │                              # across 80 adaptive iterations
    └── coolingCurves/
        ├── coolingCurve1.gnu      # Raw FE cooling curves
        ├── ...                    # (indices 1-100: training)
        └── coolingCurve116.gnu    # (indices 101-116: test)
```

---

## Dependencies

### Python packages

```
numpy
pandas
scipy
gpflow >= 2.5
tensorflow >= 2.10
scikit-learn
matplotlib
seaborn
```

Install via:

```bash
pip install numpy pandas scipy gpflow tensorflow scikit-learn 
            matplotlib seaborn
```

### External tools

- **Kaskade** — finite element solver for 3D heat transfer. 
  Available at the Zuse Institute Berlin. Required only for 
  running new simulations (`adaptive_loop.py`). Not required 
  for analysis and GP training (`analysis.py`).

---

## Usage

### Running the adaptive simulation loop

This requires access to the Kaskade FE solver and the corpse 
mesh files. Update the paths in `config.py` to match your 
local installation:

```python
RUNPATH = "/path/to/kaskade/simulation"
EXE     = "./run.sh"
```

Then run:

```bash
python adaptive_loop.py
```

This will:
1. Generate 16 test simulations (indices 101--116)
2. Generate 20 initial Sobol training simulations (indices 1--20)
3. Run 80 adaptive iterations, selecting each new simulation 
   point by maximizing GP predictive variance
4. Save results to `data/AdaptiveData/`

### Running the analysis (Google Colab)

The exploratory data analysis, GP surrogate training, evaluation, 
and figure generation were performed interactively in Google Colab. 
To reproduce the analysis:

1. Upload the `data/` folder contents to your Google Drive
2. Open a new Colab notebook
3. Mount your Google Drive:
```python
from google.collab import drive
drive.mount('/content/drive')
```
4. Update the directory path to point to your uploaded data:
```python
directory = '/content/drive/MyDrive/path/to/data/coolingCurves'
```
5. The analysis covers:
   - Exploratory data analysis of training and test datasets
   - Marshall-Hoare goodness-of-fit evaluation
   - GP surrogate training and hyperparameter analysis
   - Surrogate evaluation and cooling curve reconstruction
   - All figures included in Chapter 4 of the thesis

## Data

The simulation outputs and fitted parameter CSVs are included 
in this repository:

| File | Description |
|---|---|
| `training_data.csv` | Physical parameters + fitted $(A, B)$ for 100 training simulations |
| `test_data.csv` | Physical parameters + fitted $(A, B)$ for 16 test simulations |
| `adaptive_metrics.csv` | GP uncertainty and prediction error across 80 adaptive iterations |
| `coolingCurve1.gnu` -- `coolingCurve116.gnu` | Raw FE cooling curves |

---

## Results Summary

| Metric | Value |
|---|---|
| MH model RMSE (vs FE simulation) | $0.095 \pm 0.026$°C |
| GP surrogate RMSE on $A$ (test set) | $0.0047$ |
| GP surrogate RMSE on $B$ (test set) | $0.000784$ hr⁻¹ |
| Curve reconstruction RMSE (surrogate vs MH reference) | $0.023 \pm 0.020$°C |
| Combined pipeline RMSE | $\approx 0.098$°C |
| Speedup vs FE simulation | $> 10{,}000\times$ |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{shah2026surrogate,
  author  = {Shah, Faisal Hussain},
  title   = {Gaussian Process Surrogate for Post-Mortem Cooling:
             Simulation Pipeline and Adaptive Design Implementation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## Acknowledgements

This work was carried out at the Zuse Institute Berlin (ZIB) 
within the Computational Anatomy and Physiology group, as part 
of the Time-of-Death Estimation project. The finite element 
simulations were performed using the Kaskade finite element 
toolkit developed at ZIB.

---

## License

MIT License. See `LICENSE` for details.