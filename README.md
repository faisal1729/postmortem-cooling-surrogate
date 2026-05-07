# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository contains the implementation accompanying the master's thesis:

> **Faisal Hussain Shah (2026)**  
> *Machine Learning in Forensic Medicine – Time of Death (ToD) Estimation*  
> Master's thesis, Universität Potsdam / Zuse Institute Berlin (ZIB).

The project investigates Gaussian process (GP) surrogate modelling for post-mortem body cooling dynamics with applications to forensic time-of-death estimation. The repository combines finite element (FE) simulation outputs, Marshall–Hoare (MH) model analysis, Gaussian process regression, adaptive simulation design, and uncertainty quantification.

---

# Research Motivation

Post-mortem cooling is governed by heat transfer processes in a three-dimensional anatomical domain. Accurate simulation of this process using finite element methods is computationally expensive and therefore difficult to use repeatedly in uncertainty analyses or inverse estimation workflows.

This project develops Gaussian process surrogate models that approximate the behaviour of the FE simulations while preserving predictive accuracy. The surrogate enables:

- rapid prediction of cooling dynamics,
- efficient adaptive simulation design,
- uncertainty-aware parameter estimation,
- reduced computational cost for repeated inference tasks.

The repository contains both methodological investigations and the implementation used in the thesis experiments.

---

# Repository Structure

```text
postmortem-cooling-surrogate/
│
├── data/
│   ├── coolingCurves/          # 116 FE-generated cooling curves
│   ├── sample_curve.gnu        # Representative curve for Chapter 3
│   ├── training_data.csv
│   ├── test_data.csv
│   └── adaptive_metrics.csv
│
├── src/
│   ├── chapter2/
│   │   └── behaviour_of_mh.py
│   │
│   ├── chapter3/
│   │   └── kernel_analysis.py
│   │
│   ├── chapter4/
│   │   ├── analysis.py
│   │   ├── adaptive_loop.py
│   │   ├── config.py
│   │   ├── core_functions.py
│   │   └── kaskadeio.py
│   │
│   └── chapter5/
│
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

---

# Chapter Overview

## Chapter 2 — Marshall–Hoare Model Behaviour

Location:

```text
src/chapter2/
```

This chapter investigates the mathematical and physical behaviour of the Marshall–Hoare cooling model under varying parameter regimes. The analysis explores:

- parameter sensitivity,
- cooling-rate behaviour,
- asymptotic behaviour,
- effects of different MH parameter choices.

The implementation is intended primarily for theoretical and exploratory analysis.

---

## Chapter 3 — Gaussian Process Kernel Analysis

Location:

```text
src/chapter3/
```

This chapter studies Gaussian process regression behaviour using a representative FE-generated cooling curve. Different covariance kernels are compared, including:

- Squared Exponential kernel,
- Matérn 3/2 kernel,
- Matérn 5/2 kernel.

The implementation includes:

- prior sample visualization,
- GP regression fits,
- uncertainty visualization,
- kernel smoothness comparison.

Only a single representative cooling curve is required for this analysis.

---

## Chapter 4 — Adaptive Surrogate Modelling

Location:

```text
src/chapter4/
```

This chapter contains the main surrogate modelling framework developed in the thesis.

Key components include:

- adaptive FE simulation selection,
- Gaussian process surrogate training,
- Marshall–Hoare parameter estimation,
- prediction uncertainty quantification,
- train/test evaluation workflows.

The implementation uses 116 FE-generated cooling curves together with adaptive sampling strategies to reduce simulation requirements while maintaining predictive accuracy.

---

## Chapter 5 — Real Forensic Data Analysis

Location:

```text
src/chapter5/
```

This chapter concerns exploratory analysis on real forensic measurement data.

Due to privacy and data protection restrictions, the underlying forensic datasets are not publicly distributed in this repository.

The repository may contain:

- analysis scripts,
- preprocessing utilities,
- methodological components,
- placeholder examples.

No identifiable forensic case data are included.

---

# Dataset Description

## FE Cooling Curve Dataset

The repository contains:

- 116 finite element generated cooling curves,
- derived training and testing datasets,
- adaptive simulation metrics.

The FE simulations model post-mortem cooling under varying physical and environmental conditions.

Cooling curves are stored as `.gnu` files and are used throughout the surrogate modelling pipeline.

---

# Methodological Components

The repository combines methods from:

- Gaussian process regression,
- surrogate modelling,
- finite element simulation,
- uncertainty quantification,
- adaptive experimental design,
- inverse parameter estimation,
- forensic thermodynamics.

The work primarily relies on:

- TensorFlow,
- GPflow,
- NumPy,
- Pandas,
- Matplotlib,
- Scikit-learn.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/faisal1729/postmortem-cooling-surrogate.git
cd postmortem-cooling-surrogate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Example Usage

## Run Chapter 3 kernel analysis

```bash
python src/chapter3/kernel_analysis.py
```

## Run Chapter 4 surrogate analysis

```bash
python src/chapter4/analysis.py
```

---

# Reproducibility

The repository is structured to support reproducible computational experiments.

Features supporting reproducibility include:

- repository-relative data loading,
- deterministic random seeds,
- structured dataset organization,
- versioned releases,
- DOI-based archival through Zenodo.

---

# Citation

If you use this repository in academic work, please cite:

```text
Faisal Hussain Shah (2026).
Gaussian Process Surrogate for Post-Mortem Cooling Dynamics.
GitHub repository.
```

See `CITATION.cff` for citation metadata.

---

# DOI and Archival

This repository is archived through Zenodo and versioned using GitHub releases.

The DOI corresponding to the thesis release should be cited when referencing the repository in academic work.

---

# License

This project is distributed under the MIT License.

See `LICENSE` for details.

---

# Acknowledgements

This work was carried out at:

- Universität Potsdam
- Zuse Institute Berlin (ZIB)

within the Computational Anatomy and Physiology Group.

The author acknowledges the support and supervision provided throughout the research project.

