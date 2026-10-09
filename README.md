# robust-ggm-code
# Robust Bayesian Gaussian Graphical Models

## Overview

This repository contains the main implementation of a robust Bayesian framework for precision-matrix estimation in Gaussian graphical models. The method combines γ-divergence with a regularized horseshoe prior to improve robustness to extreme observations while encouraging sparse dependence structures.
The repository includes code for:
- simulation studies under different data-generating settings;
- comparison with Laplace-prior approaches;
- evaluation of estimation accuracy and sparsity recovery;
- uncertainty quantification and computational assessment; and
- real-data analysis.
The implementation is developed in Python/PyTorch and supports reproducible evaluation of the proposed methodology.


## Requirements

Install required packages:

pip install -r requirements.txt

## Run Simulation Study

python run_all.py

This runs:

* Frobenius error
* RMSE
* Coverage Probability (CP)
* Average Interval Length (AL)

## Run Real Data Analysis

python run_real_data.py

This generates:

* Network estimation
* Uncertainty quantification
* Figures (saved in /figures)

## Files

* robust_ggm_model.py → Core model
* frobenius_comparison.py → Simulation (accuracy)
* uncertainty_metrics.py → Simulation (uncertainty)
* real_data_main.py → Real data (your method)
* real_data_comparison.py → Comparison method

## Notes

* All results are reproducible using the provided scripts
* Random seeds are fixed
