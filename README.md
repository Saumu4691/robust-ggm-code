# robust-ggm-code
# Robust Bayesian Gaussian Graphical Models

## Overview

This repository contains code for simulation studies and real data analysis comparing the regularized horseshoe prior and Laplace prior in Bayesian Gaussian graphical models.


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
