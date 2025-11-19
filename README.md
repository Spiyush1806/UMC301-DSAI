# Thermophysical Property Prediction: Melting Point (Tm)
UMC301 – Applied Data Science & Artificial Intelligence (ADSAI) – Kaggle Project

## Project Overview

This repository contains my complete solution for the UMC301 Kaggle project, where the goal is to accurately predict the melting point (Tm) of molecules.

The project combines the power of cheminformatics and machine learning, using RDKit to extract meaningful chemical features and advanced ensemble models to make final predictions. The entire workflow is designed to be easy to run, reproducible, and adaptable for further experimentation.

## Note: The code was originally executed in an Ubuntu + VSCode environment. Please adjust file paths based on your setup.

# What This Project Aims to Teach

This project isn’t just about generating predictions — it helps reinforce many important concepts in applied data science and molecular property modeling. Through the workflow, you learn how to:

Use RDKit to extract descriptors and fingerprints from molecules

Apply feature selection to reduce noise and focus on informative variables

Train multiple ML models efficiently using cross-validation

Use Optuna to automate hyperparameter tuning (when needed)

Build a stacked ensemble to combine strengths of different models

Produce a polished, Kaggle-ready output file

## How the Pipeline Works
** 1. Feature Engineering with RDKit

** To understand each molecule in depth, the pipeline creates a rich set of chemical features:

** 9 essential RDKit descriptors (like MolWt, LogP, TPSA)

** 2048-bit Morgan count fingerprints

** 167-bit MACCS keys

** These features help the models capture molecular size, shape, structure, and chemical behavior.

2. Feature Selection (Per Fold)

Every fold of cross-validation performs its own feature selection using LightGBM’s SelectFromModel.
This helps the model focus only on the features that truly matter, making training faster and predictions more stable.

3. Hyperparameter Tuning (Optional)

If you want to squeeze out extra accuracy, you can run automated tuning with Optuna using:

tune_lgb_params()


This step searches for the best LightGBM settings using cross-validated scores.

4. Training the Ensemble Models

Each fold trains multiple strong gradient boosting models:

LightGBM

XGBoost

HistGradientBoosting

CatBoost (optional)

To keep everything consistent, the target variable is transformed using Yeo-Johnson and early stopping is used to avoid overfitting.

5. Meta-Model Stacking

To get the best final prediction, a Ridge Regression meta-model is trained on the out-of-fold predictions from all the base models.
This stacking approach usually performs better than any single model alone.

6. Final Submission

Once the pipeline finishes, it creates a clean Kaggle submission file:

submission_full_pipeline.csv


Upload this to Kaggle to see your score!

Repository Structure
├── full_pipeline.py            # Complete pipeline script
├── train.csv                   # Training dataset
├── test.csv                    # Test dataset for Kaggle
├── sample_submission.csv       
├── README.md                   
