# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository accompanies the master's thesis:

> **Faisal Hussain Shah (2026)**
> *Machine Learning in Forensic Medicine - Time of Death Estimation*
> Master's thesis, Universität Potsdam, in cooperation with the Computational Anatomy and Physiology group, Zuse Institute Berlin (ZIB).
> Supervisor: Dr. Martin Weiser.

The code reproduces the analyses, models, and figures developed in Chapters 2 through 5 of the thesis.

---

## Overview

Estimating the post-mortem interval (PMI) from body cooling is one of the oldest problems in forensic medicine, and one of the few in which physical modelling can in principle replace rules of thumb. The thesis brings together three threads that are usually treated in isolation: finite element (FE) simulations of three-dimensional heat transfer in an anatomical domain, the Marshall–Hoare (MH) double-exponential cooling law that forensic practitioners actually use, and Gaussian process regression (GPR) as a surrogate that bridges the two.

The pipeline is straightforward to describe and the contribution of the thesis is to make each step rigorous:

1. FE simulations under varying physical and environmental parameters generate cooling curves at the rectal probe location.
2. Each curve is reduced to a pair of Marshall–Hoare parameters $(A, B)$ by non-linear least squares.
3. A Gaussian process is trained to predict $(A, B)$ from the underlying physical inputs, giving a fast and uncertainty-aware surrogate for the FE solver.
4. The same surrogate framework is then applied to a real forensic dataset, where the inputs are case-level covariates rather than simulation parameters.

The surrogate replaces a computation that takes hours with one that takes milliseconds, while propagating uncertainty in a principled way. Whether the same idea transfers to real casework — where the physical assumptions of the FE model no longer hold exactly — is the question Chapter 5 sets out to answer.

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

Chapter 1 (Introduction) and Chapter 6 (Discussion and Conclusions) are expository and contain no code of their own.

---

## Chapter 2 — The Marshall–Hoare Cooling Model

`src/chapter2/`

Chapter 2 develops the Marshall–Hoare model from first principles, starting from Newton's law of cooling and arriving at the two-exponential form

$$\frac{T(t) - T_a}{T_0 - T_a} \;=\; (1 + A)\, e^{Bt} \;-\; A\, e^{(1 + 1/A)\,Bt},$$

with the eigenmode interpretation following Carslaw and Jaeger. The accompanying script `behaviour_of_mh.py` produces the illustrative figures used in the chapter: the canonical plateau-and-decay shape, the limiting behaviours as $A \to 0$ and $A \to \infty$, and the comparisons against pure Newtonian cooling that motivate the two-parameter form.

The chapter is methodological rather than computational. The script exists so that every figure in the chapter is reproducible from a single source.

---

## Chapter 3 — Gaussian Process Regression

`src/chapter3/`

Chapter 3 introduces Gaussian process regression at the level needed for the thesis: priors, posteriors, kernel choice, hyperparameter learning by marginal likelihood, and the role of observation noise. The script `kernel_analysis.py` reproduces the visual material that supports the exposition, including prior draws from the squared exponential, Matérn-3/2, and Matérn-5/2 kernels and posterior fits to a single representative cooling curve from `data/sample_curve.gnu`.

The chapter is deliberately self-contained. The kernels are compared on smoothness and on their behaviour under noisy observations, with the goal of justifying the kernel choices that recur in the application chapters rather than producing a benchmark.

Prior and kernel illustrations use GPflow; posterior fits use scikit-learn. The split is historical and the two libraries agree to numerical precision on the examples shown.

---

## Chapter 4 — Surrogate Modelling on Simulation Data

`src/chapter4/`

Chapter 4 is the methodological core of the thesis. The FE solver Kaskade (developed at ZIB) generates 116 cooling curves under varying physical inputs — heat transfer coefficient, ambient temperature, body mass, specific heat, density, and thermal conductivity — of which 100 are used for surrogate training and 16 are held out for testing. Each curve is fitted to the Marshall–Hoare form by non-linear least squares, yielding a clean $(A, B)$ pair per simulation. The Gaussian process is then trained to map the physical inputs to $(A, B)$ directly, sidestepping the FE solve at inference time.

Two design choices are central:

- **Adaptive simulation selection.** Rather than placing simulations on a fixed grid, an acquisition criterion based on the combined posterior variance $\sigma^2_A(\mathbf{x}) + \sigma^2_B(\mathbf{x})$ is used to choose the next simulation point. The loop concentrates compute on regions where the surrogate is most uncertain.
- **Kernel and prior structure.** A Matérn-5/2 kernel captures the smooth dependence on most inputs, with log-normal priors on length scales for numerical stability. ARD length scales recover the expected physical hierarchy: the heat transfer coefficient dominates the plateau parameter $A$, convective input dominates the decay parameter $B$, and density turns out to be effectively irrelevant.

The script `analysis.py` is the entry point for the chapter; `adaptive_loop.py` runs the design loop, `core_functions.py` contains the MH fitting and GP utilities, `kaskadeio.py` handles I/O with the Kaskade solver, and `config.py` collects the run-time parameters. On the 100/16 split, the surrogate reaches a test RMSE of 0.092 °C against the FE solver with $R^2 = 0.9995$.

---

## Chapter 5 — Application to Real Forensic Data

`src/chapter5/`

Chapter 5 takes the surrogate framework out of the simulation regime and onto a real forensic dataset of 80 cases from an Insitute of Forensic Meidicine, in which rectal temperature, ambient temperature, and case-level covariates (body mass, rectal probe depth, time of first measurement) are recorded along with a verified PMI. The inputs are no longer FE parameters and the outputs $(A, B)$ are no longer ground truth, but the structure of the pipeline carries over: fit MH parameters per case, train a GP to predict them from covariates, evaluate by leave-one-out cross-validation.

Two findings shape the chapter. The plateau parameter $A$ turns out to be **not reliably identifiable** from the available covariates: across all kernel and covariate choices the cross-validated $R^2$ remains below $0.20$, and on the death-anchored fits it is essentially zero. The decay parameter $B$ is **moderately identifiable**, with $R^2 \approx 0.57$–$0.59$ depending on whether rectal probe depth is included. Adding probe depth helps both parameters slightly but does not rescue $A$.

The chapter argues that this asymmetry is a substantive forensic finding rather than a modelling failure. $B$ inherits the $1/m_c$ scaling of Newtonian cooling and is therefore well-anchored by body mass; $A$ depends on local probe–tissue geometry that case metadata does not capture. The chapter also documents the move from chamber-anchored to death-anchored MH fits, which removes a systematic bias in $T_0$ but tightens the identifiability of $A$ rather than loosening it.

The forensic dataset itself is not redistributed for data-protection reasons. The repository contains the analysis scripts and a placeholder structure that would reproduce the chapter on any dataset with the same schema.

---

## Datasets

The FE cooling curve dataset (`data/coolingCurves/`) contains 116 `.gnu` files, one per simulation. Of these, 100 form the training set used by the adaptive design loop in Chapter 4 and 16 form an independent test set used for the held-out evaluation reported in the same chapter. The derived files `training_data.csv`, `test_data.csv`, and `adaptive_metrics.csv` are produced by `analysis.py` and are included so that the figures of Chapter 4 can be reproduced without re-running the FE solver. `sample_curve.gnu` is the single curve used throughout Chapter 3 to illustrate kernel behaviour.

The forensic dataset of Chapter 5 is not included.

---

## Installation

```bash
git clone https://github.com/faisal1729/postmortem-cooling-surrogate.git
cd postmortem-cooling-surrogate
pip install -r requirements.txt
```

The implementation depends primarily on NumPy, SciPy, pandas, Matplotlib, scikit-learn, TensorFlow, and GPflow. Kaskade itself is not required for any of the analysis scripts; the FE outputs are shipped as cooling curves in `data/`.

---

## Reproducing the figures

```bash
# Chapter 2: MH model behaviour
python src/chapter2/behaviour_of_mh.py

# Chapter 3: kernel comparison and GP fits
python src/chapter3/kernel_analysis.py

# Chapter 4: full surrogate pipeline on simulation data
python src/chapter4/analysis.py
```

Random seeds are fixed throughout. The figures produced match the ones in the thesis up to negligible numerical noise.

---

## Citation

If you use this repository in academic work, please cite both the thesis and the software release:

```bibtex
@mastersthesis{shah2026surrogate,
  author  = {Shah, Faisal Hussain},
  title   = {Gaussian Process Surrogate for Post-Mortem Cooling Dynamics,
             with Application to Forensic Time-of-Death Estimation},
  school  = {Universit\"at Potsdam},
  year    = {2026},
  type    = {Master's thesis},
  address = {Potsdam, Germany},
  note    = {In cooperation with Zuse Institute Berlin (ZIB)}
}
```

A `CITATION.cff` file is provided for GitHub's citation widget.

---

## DOI and archival

The repository is archived through Zenodo. The DOI corresponding to the thesis release should be cited when referencing the software in academic work; the most recent version is linked from the GitHub release page.

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

This work was carried out within the Computational Anatomy and Physiology group at the Zuse Institute Berlin (ZIB), under the supervision of Dr. Martin Weiser. I am grateful for the time, guidance, and access to the Kaskade FE pipeline that made the surrogate study possible, and for the forensic data shared by UniversitätKlinikum Jena under the corresponding data-protection agreement.
