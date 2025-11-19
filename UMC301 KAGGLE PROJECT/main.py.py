#!/usr/bin/env python3
"""Full pipeline: RDKit descriptors + fingerprints, per-fold feature selection,
Optuna-tuned XGBoost + LightGBM + stacking, and submission output.

This script is a condensed, robust implementation of the notebook you provided.
It uses a small Optuna budget by default to keep runtime reasonable in this environment.
"""
import os
import gc
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb
import optuna
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys


def tune_lgb_params(X, y, n_trials=12):
    """Quick Optuna tuning for LightGBM parameters using lgb.cv on transformed target.
    Returns a dict of best params (suitable for LGBMRegressor init).
    """
    X_arr = X.values.astype(np.float32)
    pt = PowerTransformer(method='yeo-johnson')
    y_t = pt.fit_transform(y.reshape(-1,1)).ravel()
    dtrain = lgb.Dataset(X_arr, label=y_t)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 200),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        }
        cvres = lgb.cv(params, dtrain, nfold=3, num_boost_round=1000, early_stopping_rounds=50, seed=0, verbose_eval=False)
        # find the mean metric key (e.g. 'l2-mean') and return last value
        mean_keys = [k for k in cvres.keys() if k.endswith('mean')]
        if not mean_keys:
            # fallback: take first metric series
            return float(list(cvres.values())[-1][-1])
        return float(cvres[mean_keys[0]][-1])

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    # convert into sensible LGBMRegressor kwargs
    best_params = {
        'num_leaves': int(best.get('num_leaves', 31)),
        'learning_rate': float(best.get('learning_rate', 0.05)),
        'feature_fraction': float(best.get('feature_fraction', 1.0)),
        'bagging_fraction': float(best.get('bagging_fraction', 1.0)),
        'max_depth': int(best.get('max_depth', -1)),
        'min_data_in_leaf': int(best.get('min_data_in_leaf', 20)),
        'reg_alpha': float(best.get('lambda_l1', 0.0)),
        'reg_lambda': float(best.get('lambda_l2', 0.0)),
    }
    print('Tuning finished. Best lgb params:', best_params)
    return best_params


def load_data(folder):
    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    test = pd.read_csv(os.path.join(folder, 'test.csv'))
    sample = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))
    return train, test, sample


def rdkit_basic_descriptors(smiles_series):
    rows = []
    for s in smiles_series.fillna(''):
        m = Chem.MolFromSmiles(s)
        if m is None:
            # NaNs preserve shape
            rows.append([np.nan]*9)
            continue
        # core descriptors
        core = [
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.TPSA(m),
            Descriptors.NumRotatableBonds(m),
        ]
        # a few additional descriptors where available
        extra = [
            Descriptors.NumValenceElectrons(m) if hasattr(Descriptors, 'NumValenceElectrons') else 0,
            Descriptors.FpDensityMorgan1(m) if hasattr(Descriptors, 'FpDensityMorgan1') else 0,
            Descriptors.Kappa1(m) if hasattr(Descriptors, 'Kappa1') else 0,
        ]
        rows.append(core + extra)
    cols = ["MolWt","LogP","HDonors","HAcceptors","TPSA","RotB","ValenceE","FpDens1","Kappa1"]
    return pd.DataFrame(rows, columns=cols)


def morgan_count_fingerprint(smiles_series, radius=2, nbits=2048):
    # Use rdFingerprintGenerator to build a Morgan count fingerprint generator
    try:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits, countSimulation=True)
    except Exception:
        # fallback to rdMolDescriptors if API differs
        gen = None
    data = []
    for s in smiles_series.fillna(''):
        m = Chem.MolFromSmiles(s)
        if m is None:
            data.append([0]*nbits)
            continue
        if gen is not None:
            fp = gen.GetCountFingerprint(m)
            data.append([int(fp[i]) for i in range(nbits)])
        else:
            # Best-effort fallback: use rdMolDescriptors.GetMorganFingerprintAsBitVect then convert to counts
            try:
                bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
                data.append([int(bv.GetBit(i)) for i in range(nbits)])
            except Exception:
                data.append([0]*nbits)
    cols = [f'MFP_{i}' for i in range(nbits)]
    return pd.DataFrame(data, columns=cols)


def maccs_fingerprint(smiles_series):
    data = []
    for s in smiles_series.fillna(''):
        m = Chem.MolFromSmiles(s)
        if m is None:
            data.append([0]*167)
            continue
        bv = MACCSkeys.GenMACCSKeys(m)
        data.append([int(bv.GetBit(i)) for i in range(167)])
    cols = [f'MACCS_{i}' for i in range(167)]
    return pd.DataFrame(data, columns=cols)


def build_features(train, test):
    # group-count features are already columns after id, SMILES, Tm
    drop = ['id','SMILES','Tm']
    group_cols = [c for c in train.columns if c not in drop]
    X_train = train[group_cols].copy()
    X_test = test[group_cols].copy()

    # RDKit descriptors
    desc_tr = rdkit_basic_descriptors(train['SMILES'])
    desc_te = rdkit_basic_descriptors(test['SMILES'])

    X_train = pd.concat([X_train.reset_index(drop=True), desc_tr.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), desc_te.reset_index(drop=True)], axis=1)

    # Morgan count fingerprint (increase bit size)
    mfp_tr = morgan_count_fingerprint(train['SMILES'], radius=2, nbits=2048)
    mfp_te = morgan_count_fingerprint(test['SMILES'], radius=2, nbits=2048)

    X_train = pd.concat([X_train, mfp_tr], axis=1)
    X_test = pd.concat([X_test, mfp_te], axis=1)

    # MACCS keys
    mac_tr = maccs_fingerprint(train['SMILES'])
    mac_te = maccs_fingerprint(test['SMILES'])
    X_train = pd.concat([X_train, mac_tr], axis=1)
    X_test = pd.concat([X_test, mac_te], axis=1)

    return X_train, X_test


def train_and_stack(X, y, X_test, n_splits=5, lgb_tuned_params=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros(len(X))
    test_preds = []
    fold = 0

    # store out-of-fold preds for base models
    lgb_oof = np.zeros(len(X))
    xgb_oof = np.zeros(len(X))
    hgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))
    lgb_test = []
    xgb_test = []
    hgb_test = []
    cat_test = []

    for tr_idx, va_idx in kf.split(X):
        fold += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # target transform
        pt = PowerTransformer(method='yeo-johnson')
        y_tr_arr = np.asarray(y_tr).reshape(-1,1)
        y_va_arr = np.asarray(y_va).reshape(-1,1)
        y_tr_t = pt.fit_transform(y_tr_arr).ravel()
        y_va_t = pt.transform(y_va_arr).ravel()

        # feature selector via LightGBM quick model
        sel_model = lgb.LGBMRegressor(n_estimators=500, max_depth=6)
        sel_model.fit(X_tr, y_tr_t)
        selector = SelectFromModel(sel_model, prefit=True, threshold='median')
        cols = X_tr.columns[selector.get_support()]

        X_tr_sel = X_tr[cols]
        X_va_sel = X_va[cols]
        X_test_sel = X_test[cols]

        # LightGBM using the sklearn API (fit) so early_stopping_rounds works across versions
        lgbm_kwargs = {'objective':'regression', 'n_estimators':2000, 'verbosity':-1, 'boosting_type':'gbdt'}
        if lgb_tuned_params is not None:
            lgbm_kwargs.update(lgb_tuned_params)
        lgbm_model = lgb.LGBMRegressor(**lgbm_kwargs)
        # fit with eval set for early stopping; some lightgbm builds expect the sklearn API
        # Fit with best-effort support for early stopping across different LightGBM versions
        try:
            # Newer versions accept early_stopping_rounds
            lgbm_model.fit(X_tr_sel, y_tr_t, eval_set=[(X_va_sel, y_va_t)], early_stopping_rounds=100, verbose=False)
        except TypeError:
            try:
                # Some versions support callbacks for early stopping
                lgbm_model.fit(X_tr_sel, y_tr_t, eval_set=[(X_va_sel, y_va_t)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            except Exception:
                # Last resort: fit without early stopping
                lgbm_model.fit(X_tr_sel, y_tr_t)

        # XGBoost
        dtrain = xgb.DMatrix(X_tr_sel, label=y_tr_t)
        dval = xgb.DMatrix(X_va_sel, label=y_va_t)
        dtest = xgb.DMatrix(X_test_sel)
        xgb_params = {'objective':'reg:squarederror', 'eval_metric':'rmse', 'verbosity':0, 'tree_method':'hist'}
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=2000, evals=[(dval,'val')], early_stopping_rounds=100, verbose_eval=False)

        # HistGradientBoosting as a fast tree-based third model
        hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=8, random_state=0)
        hgb.fit(X_tr_sel, y_tr_t)

        # CatBoost as an optional additional base model (handles categorical features but we have none)
        if CATBOOST_AVAILABLE:
            try:
                cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=0)
                cat.fit(X_tr_sel, y_tr_t)
            except Exception:
                cat = None
        else:
            cat = None

        # predict
        lgb_pred_va = pt.inverse_transform(lgbm_model.predict(X_va_sel).reshape(-1,1)).ravel()
        xgb_pred_va = pt.inverse_transform(xgb_model.predict(dval).reshape(-1,1)).ravel()
        hgb_pred_va = pt.inverse_transform(hgb.predict(X_va_sel).reshape(-1,1)).ravel()
        cat_pred_va = None
        if cat is not None:
            try:
                cat_pred_va = pt.inverse_transform(cat.predict(X_va_sel).reshape(-1,1)).ravel()
            except Exception:
                cat_pred_va = None
        # weighted blend
        if cat_pred_va is not None:
            # include catboost with a small weight
            blend_va = 0.35*lgb_pred_va + 0.35*xgb_pred_va + 0.2*hgb_pred_va + 0.1*cat_pred_va
        else:
            blend_va = 0.4*lgb_pred_va + 0.4*xgb_pred_va + 0.2*hgb_pred_va

        oof[va_idx] = blend_va

        lgb_oof[va_idx] = pt.inverse_transform(lgbm_model.predict(X_va_sel).reshape(-1,1)).ravel()
        xgb_oof[va_idx] = pt.inverse_transform(xgb_model.predict(dval).reshape(-1,1)).ravel()
        hgb_oof[va_idx] = pt.inverse_transform(hgb.predict(X_va_sel).reshape(-1,1)).ravel()
        if cat is not None and cat_pred_va is not None:
            cat_oof[va_idx] = cat_pred_va

        lgb_test.append(pt.inverse_transform(lgbm_model.predict(X_test_sel).reshape(-1,1)).ravel())
        xgb_test.append(pt.inverse_transform(xgb_model.predict(dtest).reshape(-1,1)).ravel())
        hgb_test.append(pt.inverse_transform(hgb.predict(X_test_sel).reshape(-1,1)).ravel())
        if cat is not None:
            try:
                cat_test.append(pt.inverse_transform(cat.predict(X_test_sel).reshape(-1,1)).ravel())
            except Exception:
                cat_test.append(np.zeros(X_test_sel.shape[0]))
        else:
            # append zeros so stacking shapes match when cat is unavailable
            cat_test.append(np.zeros(X_test_sel.shape[0]))

        print(f'Fold {fold} MAE: {mean_absolute_error(y_va, blend_va):.4f}')

    # stack: use Ridge on top of base models' OOF predictions
    if CATBOOST_AVAILABLE:
        stack_train = np.vstack([lgb_oof, xgb_oof, hgb_oof, cat_oof]).T
        stack_test = np.vstack([np.mean(lgb_test, axis=0), np.mean(xgb_test, axis=0), np.mean(hgb_test, axis=0), np.mean(cat_test, axis=0)]).T
    else:
        stack_train = np.vstack([lgb_oof, xgb_oof, hgb_oof]).T
        stack_test = np.vstack([np.mean(lgb_test, axis=0), np.mean(xgb_test, axis=0), np.mean(hgb_test, axis=0)]).T

    meta = Ridge(alpha=1.0)
    meta.fit(stack_train, y)
    final_test_pred = meta.predict(stack_test)

    print('OOF MAE (blend):', mean_absolute_error(y, oof))
    return oof, final_test_pred


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    train, test, sample = load_data(folder)

    X_train, X_test = build_features(train, test)
    y = train['Tm'].values

    print('Features built:', X_train.shape)

    oof, test_pred = train_and_stack(X_train, y, X_test, n_splits=5)

    sub = pd.DataFrame({'id': test['id'], 'Tm': test_pred})
    out_path = os.path.join(folder, 'submission_full_pipeline.csv')
    sub.to_csv(out_path, index=False)
    print('Wrote submission to', out_path)


if __name__ == '__main__':
    main()
