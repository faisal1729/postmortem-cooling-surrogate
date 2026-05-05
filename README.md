# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository contains the implementation accompanying the master's thesis:

> **Faisal Hussain Shah** (2026). *Machine Learning in Forensic Medicine – Time of Death (ToD) Estimation*.  
> Master's thesis, Universität Potsdam / Zuse Institute Berlin.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Overview

Post-mortem body cooling follows the heat equation over a three-dimensional anatomical domain.  
Direct finite element (FE) simulation of this process is accurate but computationally expensive.

This repository implements a surrogate modelling pipeline that replaces the FE solver with a Gaussian process (GP) regression model trained on an adaptive dataset of FE simulations, reducing prediction time to milliseconds while maintaining high accuracy.

---

## Repository Structure

```
├── src/
│   ├── chapter2/
│   ├── chapter3/
│   ├── chapter4/
│   └── chapter5/
│
├── data/
│   ├── training_data.csv
│   ├── test_data.csv
│   ├── coolingCurves/
│   └── sample_curve/
│
├── README.md
├── LICENSE
```

---

## Reproducibility

### Fully reproducible components

- Chapter 2 – Marshall-Hoare Post-Mortem Cooling Model 
- Chapter 3 – Gaussian Process Regression  
- Chapter 4 – Surrogate Modelling of Postmortem Cooling Dynamics  

Run:

```
python src/chapter4/analysis.py
```

---

### Chapter 5 (restricted data)

Chapter 5 uses real forensic data that cannot be shared due to privacy constraints.

- Code is included  
- Data is NOT included  

Place data in:

```
data/real_data_placeholder/
```

---

## Dependencies

```
numpy
pandas
scipy
gpflow
tensorflow
scikit-learn
matplotlib
seaborn
```

Install:

```
pip install numpy pandas scipy gpflow tensorflow scikit-learn matplotlib seaborn
```

---

## Citation

```
@software{shah2026surrogate,
  author  = {Shah, Faisal Hussain},
  title   = {Gaussian Process Surrogate for Post-Mortem Cooling Dynamics},
  year    = {2026},
  version = {v2.0.0},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

MIT License.
