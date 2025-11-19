# Thermophysical Property: Melting Point
UMC301(DSAI) - Kaggle project


This notebook is my submission for the UMC301(DSAI) - Kaggle project.
Before running the code, please adjust the file paths as this notebook was originally executed on ubuntu (Vscode).


This repository contains a complete end-to-end machine learning pipeline for predicting molecular melting points (Tm) using:

RDKit molecular descriptors

Morgan count fingerprints

MACCS keys

Per-fold feature selection

Optuna hyperparameter tuning

XGBoost, LightGBM, CatBoost, and HistGradientBoosting

Meta-model stacking with Ridge regression

Final Kaggle-style submission CSV

The full workflow is implemented in a single Python script (full_pipeline.py).

🚀 Key Features
🔬 1. RDKit Feature Engineering

The pipeline automatically generates:

9 basic RDKit descriptors (MolWt, LogP, TPSA, etc.)

2048-bit Morgan count fingerprints

167-bit MACCS keys

🧹 2. Intelligent Feature Selection

Per-fold LightGBM SelectFromModel chooses the most predictive features, improving model speed and accuracy.

🎛 3. Auto Hyperparameter Optimization

Includes optional Optuna tuning for LightGBM parameters (tune_lgb_params()).

🤖 4. Strong Ensemble Models

Each fold trains:

LightGBM

XGBoost

HistGradientBoosting

(Optional) CatBoost

With target Yeo-Johnson transformation and per-fold early stopping.

🧠 5. Stack Ensemble Meta-Learner

Final predictions are produced by a Ridge regression meta-model trained on out-of-fold predictions.

📤 6. Submission File Generation
## OUTPUT
submission_full_pipeline.csv

📁 Repository Structure

├── full_pipeline.py      
├── train.csv                
├── test.csv                 
├── sample_submission.csv    
├── README.md

