# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository accompanies the master's thesis:

> **Faisal Hussain Shah (2026)**  
> *Machine Learning in Forensic Medicine - Time of Death Estimation*  
> Master's thesis, Universität Potsdam, in cooperation with the Computational Anatomy and Physiology group, Zuse Institute Berlin (ZIB).  
> Supervisor: Dr. Martin Weiser.

The code reproduces the analyses, models, and figures of Chapters 2 through 4.

---

## Overview

Estimating the post-mortem interval (PMI) from body cooling is one of the oldest problems in forensic medicine, and one of the few in which physical modelling can in principle replace rules of thumb. The thesis takes a different approach from the standard one: rather than forcing a simple analytical model onto a complex physical process, it builds a computational pipeline that respects the underlying heat transfer physics while treating the unavoidable uncertainties honestly.

The core idea is to pair a high-fidelity finite element (FE) simulation of postmortem cooling with a Gaussian process surrogate that learns the relationship between an individual's physical characteristics and their cooling behaviour — rapidly, and with calibrated uncertainty. Once trained, the surrogate removes the need for the full FE pipeline at inference time: no CT segmentation, no mesh generation, no simulation software. A process that currently demands specialist infrastructure and hours of preparation is reduced to a function evaluation, with a credible interval rather than a precise-sounding point estimate.

---

## Repository structure

```text
postmortem-cooling-surrogate/
│
├── data/
│   ├── coolingCurves/          # 116 FE-generated cooling curves (100 training + 16 test)
│   ├── sample_curve.gnu        # Representative curve used in Chapter 3
│   ├── training_data.csv
│   ├── test_data.csv
│   └── adaptive_metrics.csv
│
├── src/
│   ├── chapter2/               # Marshall–Hoare model behaviour
│   ├── chapter3/               # Gaussian process kernel analysis
│   ├── chapter4/               # Adaptive surrogate on simulation data
│                  
│
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

Chapter 1 (Introduction) and Chapter 6 (Discussion and Conclusions) are expository and contain no code.

---

## Chapter contents

**Chapter 2 — Marshall–Hoare model.** A detailed study of the two-exponential cooling model: parameter sensitivity analysis of its parameters: $A$ and $B$, 
connection to Newtonian cooling in the appropriate limit, and the Henssge reparametrisation used in forensic practice. `behaviour_of_mh.py` reproduces the illustrative figures.

**Chapter 3 — Gaussian process regression.** Priors, posteriors, kernel choice, and hyperparameter learning, at the level needed for the surrogate work. `kernel_analysis.py` compares the squared exponential, Matérn-3/2, and Matérn-5/2 kernels on a single representative cooling curve from `sample_curve.gnu`.

**Chapter 4 — Adaptive surrogate on simulation data.** The methodological core of the thesis. The FE solver Kaskade generates 116 cooling curves under varying physical inputs; each is reduced to Marshall–Hoare parameters $(A, B)$ by non-linear least squares, and a Gaussian process is trained to predict $(A, B)$ directly. Simulations are placed by an adaptive design that targets points of maximum posterior variance $\sigma^2_A(\mathbf{x}) + \sigma^2_B(\mathbf{x})$. ARD length scales recover the expected physical hierarchy: heat transfer coefficient dominates $A$, convection dominates $B$, density is effectively irrelevant. Entry point: `analysis.py`.

**Chapter 5 — Forensic identifiability study.** Marshall–Hoare parameters are fitted per case on 80 real forensic cases from an institute of forensic medicine, and a Gaussian process is trained to predict them from baseline body covariates $(m_c, h, T_a)$, then augmented with rectal probe insertion depth $d$. The chapter asks whether $(A, B)$ are identifiable from these covariates and whether $d$ helps. Under leave-one-out cross-validation, $B$ is moderately identifiable ($R^2 \approx 0.57\text{--}0.59$) while $A$ is not ($R^2 < 0.20$). The asymmetry is interpreted physically: $B$ inherits the $1/m_c$ scaling of Newtonian cooling, while $A$ depends on probe–tissue geometry that case metadata does not capture. The forensic dataset is not redistributed.

---

## Datasets

`data/coolingCurves/` contains 116 `.gnu` files: 100 form the training set used by the adaptive design loop in Chapter 4, 16 form an independent test set. `sample_curve.gnu` is the single curve used throughout Chapter 3.

`training_data.csv`, `test_data.csv`, and `adaptive_metrics.csv` are logs produced during the adaptive loop in Chapter 4. They record, at each iteration, the mean integrated posterior variance, the maximum acquisition score, and the prediction error on the held-out test set, and are used to generate the convergence diagnostics shown in the chapter.

The forensic dataset of Chapter 5 is not included.

---

## Installation

```bash
git clone https://github.com/faisal1729/postmortem-cooling-surrogate.git
cd postmortem-cooling-surrogate
pip install -r requirements.txt
```

Primary dependencies: NumPy, SciPy, pandas, Matplotlib, scikit-learn, TensorFlow, GPflow. Kaskade is not required; its outputs are shipped in `data/`.

---

## Reproducing the figures

```bash
python src/chapter2/behaviour_of_mh.py    # MH model behaviour
python src/chapter3/kernel_analysis.py    # kernel comparison and GP fits
python src/chapter4/analysis.py           # adaptive surrogate pipeline
```

Random seeds are fixed throughout.

---

## Citation

```bibtex
@mastersthesis{shah2026surrogate,
  author  = {Shah, Faisal Hussain},
  title   = {Machine Learning in Forensic Medicine -
             Time of Death Estimation},
  school  = {Universit\"at Potsdam},
  year    = {2026},
  type    = {Master's thesis},
  address = {Potsdam, Germany},
  note    = {In cooperation with Zuse Institute Berlin (ZIB)}
}
```

A `CITATION.cff` file is provided for GitHub's citation widget. The most recent Zenodo DOI is linked from the GitHub release page.

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

This work was carried out within the Computational Anatomy and Physiology group at the Zuse Institute Berlin (ZIB), under the supervision of Dr. Martin Weiser. I am grateful for his time, guidance, and access to the Kaskade FE pipeline, and to UniversitätKlinikum Jena for the forensic data shared under the corresponding data-protection agreement.
