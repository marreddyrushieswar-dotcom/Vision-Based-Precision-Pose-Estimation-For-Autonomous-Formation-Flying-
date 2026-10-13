# Vision-Based Pose Estimation for Autonomous Formation Flying
### Machine Learning Mini-Project | ANN + Particle Filter

This repository contains a re-implementation of **“Vision-Based Precision Pose Estimation for Autonomous Formation Flying”** by R. Punnoose (Stanford), prepared for the UE24CS352A Machine Learning mini-project.

**Project origin and attribution:** This implementation is based on the reference project by https://github.com/marreddyrushieswar-dotcom/Vision-Based-Precision-Pose-Estimation-For-Autonomous-Formation-Flying-/edit/main/README.md and the research paper credited above. The implementation and reported results should be treated as a reproduction/extension of that work, not as an entirely original algorithm. Any future changes or experiments should be documented in the sections below.

A follower aircraft estimates the relative pose of a leader from 14 structural features supplied by a simulated detector.

## Overview

- **Artificial neural network (ANN):** Maps 42 input values to a discrete pose label using two 100-unit ReLU hidden layers and a 4,800-class softmax output.
- **Particle filter (PF):** Estimates the six-degree-of-freedom relative state using a motion model, synthetic feature observations, and ANN-informed particle proposals.
- **Refinement and experiments:** Includes Gaussian observation likelihoods, Levenberg–Marquardt reprojection refinement, an optional known line-of-sight input (`--known-los`), and a softmax-based particle proposal.

The project generates its data synthetically with `data_gen.py`; there is no public dataset supplied with the paper.

## Requirements and setup

Python 3.10 or newer is recommended. Install the dependencies from the project directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The core dependencies are NumPy, SciPy, and Matplotlib. PyTorch is not required for the standard NumPy ANN implementation.

## Quick start

The trained model files (`models/ann.npz` and `models/error_pool.npy`) are included in the reference project, so a short demo can be run without retraining:

```bash
python simulate.py --seeds 1
```

Run the commands from the directory containing `simulate.py`.

## Reproduce the experiments

Use the following sequence when generating data and training a model from scratch:

```bash
# 1. Generate synthetic training and validation data
python data_gen.py --n-train 600000 --n-val 40000

# 2. Train the ANN
python train_ann.py --epochs 20

# 3. Run the baseline approach scenario
python simulate.py

# 4. Try refinement and line-of-sight information
python simulate.py --likelihood gauss --refine
python simulate.py --likelihood gauss --refine --known-los

# 5. Try the softmax particle proposal
python simulate.py --likelihood gauss --proposal softmax
python simulate.py --likelihood gauss --proposal softmax --sigma-px 1 --known-los

# 6. Collect summaries and compare against the digitised paper plots
python summarize.py
python compare_paper.py

# 7. Visualise detected and occluded features
python visualize.py
```

Training may take a while on a CPU. For a smaller test run, use `--n-train 100000 --epochs 10`. To evaluate an already saved model without training, use `python train_ann.py --eval-only`.

The paper's comparison curves are stored in `paper_data/`. The digitisation utility can be run with `python digitize_paper.py ../Punnoose.pdf` when the original PDF is available; it requires `pypdf` and `pillow`.

## Regression tests

A small test suite checks that pose labels map back to their bin centres, out-of-range poses are clipped to edge bins, expected array shapes are retained, and soft estimates behave correctly. These checks do not require the trained model files.

Run from the directory containing `simulate.py`:

```bash
python -m unittest discover -s tests -v
```

## Project structure

| File or directory | Purpose |
|---|---|
| `config.py` | Camera parameters, pose bins, and motion constants |
| `aircraft.py` | Aircraft feature geometry, camera projection, and occlusion |
| `discretize.py` | Converts poses to and from discrete labels |
| `data_gen.py` | Builds the synthetic data set |
| `ann.py`, `train_ann.py` | ANN prediction, training, and validation |
| `dynamics.py` | Simplified relative-motion model |
| `particle_filter.py` | ANN-informed particle filter |
| `refine.py` | Levenberg–Marquardt reprojection refinement |
| `simulate.py` | Runs scenarios, calculates errors, and creates plots |
| `summarize.py` | Combines output summaries into a Markdown table |
| `digitize_paper.py`, `paper_data/` | Digitised reference curves from the paper |
| `compare_paper.py` | Compares the simulation with the reference curves |
| `visualize.py` | Displays detected and occluded features |
| `tests/test_discretize.py` | Regression checks for pose conversions and soft estimates |
| `models/` | Saved ANN and error-distribution files |
| `results/` | Generated plots and summaries |

## Results

### Paper vs. simulation

![paper vs ours](Autonomous-Formation-Flying-master/results/paper_vs_ours_summary.png)

The paper's digitised curves report mean position errors of 10.2 m for N=1000 and 8.5 m for N=5000. Compare these cautiously: the implementation measures attitude error as the full 3-D rotation angle, including leader pitch and yaw, while the paper does not clearly define an equivalent attitude metric.

The reference experiment reports the following simulation metrics:

| Method (10-second approach) | Without known LOS | With known LOS |
|---|---:|---:|
| ANN, arg-max bin | 6.02 m / 5.04° | 4.12 m / 5.04° |
| ANN, softmax mean | 3.09 m / 2.85° | 1.62 m / 2.85° |
| PF, inverse-MSE likelihood (N=5000) | 3.87 ± 1.42 m / 3.65° | — |
| PF, Gaussian likelihood (N=5000) | 2.05 ± 0.54 m / 3.67° | 1.50 ± 0.18 m / 3.21° |
| Gaussian likelihood + softmax proposal (N=5000) | 2.07 ± 0.08 m / 3.51° | **0.32 ± 0.07 m / 1.43°** |
| Levenberg–Marquardt refinement | 2.54 m / 1.98° | **0.31 m / 0.26°** |

The reported ANN validation results are 33.6% top-1 accuracy and 81.1% top-5 accuracy. Position RMS decreases from 8.25 m using the arg-max bin centre to 6.10 m using the softmax mean. These figures are inherited reference results; rerun the experiment before presenting them as results from a modified version.

## Model conventions and limitations

- The state is `[x, y, z, phi, theta, psi]`; position is follower-relative-to-leader. For example, `x = -90` means the follower is 90 m behind the leader.
- The camera model uses a 1000 px focal length and a 1280 × 960 image, with 0.5 px detector noise.
- Features can be missing because of occlusion or because they fall outside the image.
- Leader pitch and yaw vary by ±5° but are not included in the discrete ANN label, creating a source of classification error.
- The motion mapping from pitch to vertical velocity and roll to lateral velocity is a modelling interpretation of the paper's brief dynamics description.
- Without the gimbal line-of-sight input, bearing changes can be confused with changes in leader attitude. The `--known-los` option is designed to reduce this ambiguity.

## Reproducibility checklist

Before comparing results, record the Python version, installed package versions, command-line arguments, random seed(s), and whether the model was retrained. Keep generated summaries and plots separate from the reference figures so that results can be traced to the exact run.

## Changes made in this repository

- Improved setup, execution, results, and reproducibility documentation.
- Added regression tests for pose discretisation and soft pose estimation.
- Preserved existing project files and the current result-image reference path.

For any further changes, record **what changed, why it changed, how it was tested, and the measured result**. Do not claim an accuracy improvement or a new algorithm unless it has been measured.

## Reference

- Research paper: R. Punnoose, “Vision-Based Precision Pose Estimation for Autonomous Formation Flying.”
- Reference implementation: [siddhardha-06/Autonomous-Formation-Flying](https://github.com/siddhardha-06/Autonomous-Formation-Flying).
