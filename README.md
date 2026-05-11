# Gaussian Process Surrogate for Post-Mortem Cooling Dynamics

This repository accompanies the master's thesis:

> **Faisal Hussain Shah (2026)**
> *Gaussian Process Surrogate for Post-Mortem Cooling Dynamics, with Application to Forensic Time-of-Death Estimation*
> Master's thesis, Universität Potsdam, in cooperation with the Computational Anatomy and Physiology group, Zuse Institute Berlin (ZIB).
> Supervisor: Dr. Martin Weiser.

The code reproduces the analyses, models, and figures developed in Chapters 2 through 5 of the thesis.

---

## Overview

Estimating the post-mortem interval (PMI) from body cooling is one of the oldest problems in forensic medicine, and one of the few in which physical modelling can, in principle, replace rules of thumb. The thesis takes a different approach from the standard one. Rather than forcing a simple analytical model onto a complex physical process, it builds a computational pipeline that respects the underlying heat transfer physics while treating the unavoidable uncertainties honestly.

The core idea is to pair a high-fidelity finite element (FE) simulation of postmortem cooling with a Gaussian process surrogate that learns the relationship between an individual's physical characteristics and their cooling behaviour — rapidly, and with calibrated uncertainty. The practical benefit is twofold. First, predictions become near-instantaneous at query time. Second, and arguably more important, once the surrogate is trained the forensic practitioner no longer needs access to the full FE pipeline at all: no CT segmentation, no mesh generation, no simulation software. A process that currently demands specialist infrastructure and hours of manual preparation is reduced to a function evaluation. The result is a tool that can give a forensic practitioner not just a PMI estimate but a credible interval around it — an honest answer rather than a precise-sounding one.

Concretely, the pipeline runs in four steps:

1. FE simulations under varying physical and environmental parameters generate cooling curves at the rectal probe location.
2. Each curve is reduced to a pair of Marshall–Hoare parameters $(A, B)$ by non-linear least squares.
3. A Gaussian process is trained to predict $(A, B)$ from the underlying physical inputs, giving a fast and uncertainty-aware surrogate for the FE solver.
4. A real forensic dataset is then examined on its own terms — fitting Marshall–Hoare per case and asking whether $(A, B)$ can be recovered from the body covariates that are actually available in casework. This is an identifiability study rather than an application of the simulation-trained surrogate, and the asymmetry between $A$ and $B$ that emerges is one of the substantive findings of the thesis.

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

Chapter 5 turns to a real forensic dataset of 80 cases from Berlin Charité in which rectal temperature, ambient temperature, and case-level covariates — corrected body mass $m_c$, height $h$, ambient temperature $T_a$, and (for a subset) rectal probe insertion depth $d$ — are recorded alongside a verified PMI. The chapter is not an attempt to deploy the simulation-trained surrogate on casework; the inputs available in the field are case covariates, not the physical parameters that drive the FE solver, so the question has to be posed differently.

The chapter pursues a twofold question. First, whether the Marshall–Hoare parameters $(A, B)$ are identifiable from the baseline body covariates $(m_c, h, T_a)$. Second, whether augmenting this baseline with the rectal probe insertion depth $d$ meaningfully improves that identifiability. The two questions are addressed in sequence, and the answers turn out to differ between the two parameters.

The structure of the analysis mirrors Chapter 4: Marshall–Hoare parameters are fitted per case, a Gaussian process is trained to predict them from covariates, and predictive quality is assessed by leave-one-out cross-validation. The decay parameter $B$ is **moderately identifiable**, with $R^2 \approx 0.57$–$0.59$ depending on whether probe depth is included. The plateau parameter $A$ is **not reliably identifiable** under any covariate set tested, with cross-validated $R^2$ below $0.20$ throughout. Adding probe depth helps both parameters, but does not rescue $A$.

The chapter argues that this asymmetry is a substantive forensic finding rather than a modelling failure. $B$ inherits the $1/m_c$ scaling of Newtonian cooling and is well-anchored by body mass; $A$ depends on local probe–tissue geometry that the available case metadata does not capture. The chapter also documents the move from chamber-anchored to death-anchored MH fits, which removes a systematic bias in $T_0$.

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

This work was carried out within the Computational Anatomy and Physiology group at the Zuse Institute Berlin (ZIB), under the supervision of Dr. Martin Weiser. I am grateful for the time, guidance, and access to the Kaskade FE pipeline that made the surrogate study possible, and for the forensic data shared by Berlin Charité under the corresponding data-protection agreement.
