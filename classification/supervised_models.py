#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
supervised_models.py
====================
Supervised lead-scoring pipeline for an Ooredoo Business B2B thesis.

Trains three classifiers (Logistic Regression, Random Forest, LightGBM) to predict
the column `predicted_customer`, compares them, selects the best by validation PR-AUC,
calibrates it, evaluates on a held-out test set, produces thesis-ready plots, and
saves deployment artifacts.

IMPORTANT MODELLING NOTE
------------------------
The training label `predicted_customer` is defined as
    predicted_customer = 1  if  is_known_customer == 1  OR  pu_score_prior_corr >= 0.5
                         else 0
So `pu_score_prior_corr` and `is_known_customer` *define* the label and MUST NOT be
used as features (data leakage). `is_known_customer` is kept aside only for the
sanity-anchor evaluation in step 5, because those ~625 rows are the only truly
verified positives; every other positive is a pseudo-label produced by a
Positive-Unlabeled (PU) model.

Run:
    python supervised_models.py --input ranked_prospects_v3.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
import json
import numpy as np
import pandas as pd

# ----- scikit-learn -----
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, roc_curve, precision_recall_curve,
)

import lightgbm as lgb
from lightgbm import LGBMClassifier

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------------
SEED = 42
THRESHOLD = 0.5
np.random.seed(SEED)

MODEL_ORDER = ["LogisticRegression", "RandomForest", "LightGBM"]
MODEL_COLORS = {
    "LogisticRegression": "#1f77b4",
    "RandomForest":       "#2ca02c",
    "LightGBM":           "#d62728",
}

# Raw identifier / free-text columns that are never features
ID_TEXT_COLS = [
    "name", "address", "phone", "website", "email", "place_url",
    "plus_code", "scraped_at", "hours",
]
# Columns that are the three named categoricals (encoded specially)
KNOWN_CATEGORICAL = ["category", "search_category", "governorate"]

# The target and the columns that define / leak it
TARGET_COL = "predicted_customer"
LABEL_HELPER_COL = "is_known_customer"   # kept for evaluation only, never a feature


# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[FATAL] Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"[load] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {path}")
    return df


def engineer_presence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a few binary presence flags from raw text columns *before* those raw
    columns are dropped as identifiers. These are legitimate business signals
    (not derived from the label) and are commonly predictive of being a B2B
    customer. Kept minimal and clearly logged so they can be removed if desired.
    """
    added = []
    for raw, flag in [("website", "has_website"), ("email", "has_email"),
                      ("phone", "has_phone")]:
        if raw in df.columns and flag not in df.columns:
            df[flag] = df[raw].notna().astype(int)
            added.append(flag)
    if added:
        print(f"[feateng] Added presence flags: {added}")
    return df


def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create `predicted_customer` with the exact rule if it is absent."""
    if TARGET_COL in df.columns:
        print(f"[target] '{TARGET_COL}' already present.")
        return df

    print(f"[target] '{TARGET_COL}' absent -> creating it with the specified rule.")
    if LABEL_HELPER_COL not in df.columns and "pu_score_prior_corr" not in df.columns:
        sys.exit("[FATAL] Cannot build target: need is_known_customer and/or "
                 "pu_score_prior_corr, neither is present.")
    known = df[LABEL_HELPER_COL].fillna(0).astype(int) if LABEL_HELPER_COL in df.columns \
        else pd.Series(0, index=df.index)
    pu = df["pu_score_prior_corr"] if "pu_score_prior_corr" in df.columns \
        else pd.Series(0.0, index=df.index)
    df[TARGET_COL] = ((known == 1) | (pu >= 0.5)).astype(int)
    return df


def identify_columns(df: pd.DataFrame):
    """
    Decide which columns are leakage (dropped), identifiers (dropped),
    features (kept), and split features into numeric vs categorical.
    Returns a dict describing the schema. Never crashes on missing columns.
    """
    cols = set(df.columns)

    # ----- leakage columns: anything from the PU model / ranking / label -----
    leakage_patterns = [
        re.compile(r"pu_score", re.I),
        re.compile(r"^prob_", re.I),
        re.compile(r"^rank$", re.I),
        re.compile(r"^rank_", re.I),
        re.compile(r"^pred_proba", re.I),
        re.compile(r"^propensity", re.I),
        # `match_*` columns are populated only for confirmed known customers
        # (derived from the label) -> hard leakage.
        re.compile(r"^match_", re.I),
    ]
    leakage = set()
    for c in df.columns:
        if c in (TARGET_COL, LABEL_HELPER_COL):
            continue
        if any(p.search(c) for p in leakage_patterns):
            leakage.add(c)

    # ----- identifier / free-text columns -----
    identifiers = {c for c in ID_TEXT_COLS if c in cols}

    # ----- start from everything, remove target, helper, leakage, ids -----
    drop_from_features = leakage | identifiers | {TARGET_COL, LABEL_HELPER_COL}
    feature_cols = [c for c in df.columns if c not in drop_from_features]

    # ----- drop constant / all-missing columns (no information) -----
    constant = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"[schema] Constant/all-missing cols dropped: {constant}")
    feature_cols = [c for c in feature_cols if c not in constant]

    # ----- correlation-based leakage guard against is_known_customer ---------
    # Any feature whose *presence pattern* (or value) is almost perfectly
    # aligned with the verified label is treated as leakage. This catches
    # label-derived columns even if they don't match a name pattern.
    corr_leak = []
    if LABEL_HELPER_COL in cols:
        known = df[LABEL_HELPER_COL].fillna(0).astype(int).values
        kstd = known.std()
        for c in feature_cols:
            presence = df[c].notna().astype(int).values
            if presence.std() > 0 and kstd > 0:
                r = abs(np.corrcoef(presence, known)[0, 1])
                if r > 0.98:
                    corr_leak.append((c, round(float(r), 3)))
    if corr_leak:
        print(f"[schema] Leakage guard dropped (presence ~ label): {corr_leak}")
        leakage |= {c for c, _ in corr_leak}
        drop_cols = {c for c, _ in corr_leak}
        feature_cols = [c for c in feature_cols if c not in drop_cols]

    # ----- split numeric vs categorical -----
    categorical = []
    numeric = []
    for c in feature_cols:
        if c in KNOWN_CATEGORICAL:
            categorical.append(c)
        elif df[c].dtype == object or str(df[c].dtype) == "category" or df[c].dtype == bool:
            categorical.append(c)
        else:
            numeric.append(c)

    # ----- warnings for things the user mentioned but that are missing -----
    for expected in KNOWN_CATEGORICAL:
        if expected not in cols:
            warnings.warn(f"Expected categorical column '{expected}' not found - skipping.")
    if LABEL_HELPER_COL not in cols:
        warnings.warn(f"'{LABEL_HELPER_COL}' not found - the known-customer sanity "
                      f"check in step 5 will be skipped.")

    schema = dict(
        feature_cols=feature_cols,
        numeric=numeric,
        categorical=categorical,
        leakage=sorted(leakage),
        identifiers=sorted(identifiers),
        has_known=LABEL_HELPER_COL in cols,
    )

    print(f"[schema] Features kept        : {len(feature_cols)} "
          f"({len(numeric)} numeric, {len(categorical)} categorical)")
    print(f"[schema] Leakage cols dropped : {schema['leakage']}")
    print(f"[schema] Identifier cols drop : {schema['identifiers']}")
    print(f"[schema] Categorical features : {categorical}")
    return schema


def split_data(df: pd.DataFrame, schema: dict):
    """Stratified 60/20/20 train/val/test split with fixed seed."""
    X = df[schema["feature_cols"]].copy()
    y = df[TARGET_COL].astype(int).copy()
    known = (df[LABEL_HELPER_COL].fillna(0).astype(int)
             if schema["has_known"] else pd.Series(0, index=df.index))

    if y.nunique() < 2:
        sys.exit("[FATAL] Target has a single class; cannot train a classifier.")

    # 60 / 40 first
    X_tr, X_tmp, y_tr, y_tmp, k_tr, k_tmp = train_test_split(
        X, y, known, test_size=0.40, stratify=y, random_state=SEED)
    # 40 -> 20 / 20
    X_val, X_te, y_val, y_te, k_val, k_te = train_test_split(
        X_tmp, y_tmp, k_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    def bal(name, yy):
        pos = int(yy.sum()); n = len(yy)
        print(f"    {name:5s}: n={n:6,d}  positives={pos:5,d} ({pos/n:6.2%})")

    print("[split] Stratified 60/20/20 class balance:")
    bal("train", y_tr); bal("val", y_val); bal("test", y_te)

    return dict(
        X_tr=X_tr, y_tr=y_tr, k_tr=k_tr,
        X_val=X_val, y_val=y_val, k_val=k_val,
        X_te=X_te, y_te=y_te, k_te=k_te,
    )


# ----------------------------------------------------------------------------
# Preprocessors
# ----------------------------------------------------------------------------
def build_column_transformer(numeric, categorical, scale: bool) -> ColumnTransformer:
    """
    ColumnTransformer for Logistic Regression / Random Forest.
    Numeric: median impute (+ missing indicator), optional standard scaling.
    Categorical: most-frequent impute + one-hot (rare categories collapsed).
    """
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=30, sparse_output=False)),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))

    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=False)


# The custom LightGBM preprocessor lives in a separate importable module so the
# saved model can be unpickled from predict.py (a class defined in __main__
# cannot be). See pipeline_utils.py.
from pipeline_utils import LGBMFramePrep


# ============================================================================
# 2. MODELS + light tuning on the validation set
# ============================================================================
def build_model_specs(numeric, categorical, y_tr):
    """Return {name: (make_pipeline_fn(**params), param_grid)}."""
    pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
    spw = max(neg / max(pos, 1), 1.0)   # scale_pos_weight for LightGBM

    # ---- Logistic Regression ----
    def make_lr(C=1.0):
        pre = build_column_transformer(numeric, categorical, scale=True)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                 C=C, solver="lbfgs", random_state=SEED)
        return Pipeline([("prep", pre), ("model", clf)])
    lr_grid = [{"C": c} for c in (0.1, 1.0, 3.0)]

    # ---- Random Forest ----
    def make_rf(n_estimators=400, max_depth=None, min_samples_leaf=1):
        pre = build_column_transformer(numeric, categorical, scale=False)
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight="balanced",
            n_jobs=-1, random_state=SEED)
        return Pipeline([("prep", pre), ("model", clf)])
    rf_grid = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": 16,   "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
    ]

    # ---- LightGBM ----
    def make_lgbm(n_estimators=600, learning_rate=0.05, num_leaves=31):
        pre = LGBMFramePrep(numeric, categorical)
        clf = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1)
        return Pipeline([("prep", pre), ("model", clf)])
    lgbm_grid = [
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 31},
        {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63},
    ]

    return {
        "LogisticRegression": (make_lr, lr_grid),
        "RandomForest":       (make_rf, rf_grid),
        "LightGBM":           (make_lgbm, lgbm_grid),
    }


def tune_on_validation(make_fn, grid, X_tr, y_tr, X_val, y_val):
    """
    Light tuning: fit each candidate on TRAIN, score PR-AUC (average precision)
    on VALIDATION, keep the best. The test set is never touched here.
    """
    best_ap, best_est, best_params = -1.0, None, None
    for params in grid:
        est = make_fn(**params)
        est.fit(X_tr, y_tr)
        ap = average_precision_score(y_val, est.predict_proba(X_val)[:, 1])
        if ap > best_ap:
            best_ap, best_est, best_params = ap, est, params
    return best_est, best_params, best_ap


# ============================================================================
# 3. EVALUATION
# ============================================================================
def metrics_at_threshold(y_true, proba, threshold=THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy":          accuracy_score(y_true, pred),
        "precision":         precision_score(y_true, pred, zero_division=0),
        "recall":            recall_score(y_true, pred, zero_division=0),
        "f1":                f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc":           roc_auc_score(y_true, proba),
        "pr_auc":            average_precision_score(y_true, proba),
        "brier":             brier_score_loss(y_true, proba),
        "log_loss":          log_loss(y_true, proba, labels=[0, 1]),
    }


def known_customer_check(y_pred_bin, proba, known_mask):
    """Recall on verified positives (is_known_customer==1) and their mean proba."""
    if known_mask.sum() == 0:
        return {"known_recall": np.nan, "known_mean_proba": np.nan, "n_known": 0}
    km = known_mask.values if hasattr(known_mask, "values") else known_mask
    return {
        "known_recall":     float(y_pred_bin[km].mean()),      # all are true positives
        "known_mean_proba": float(proba[km].mean()),
        "n_known":          int(km.sum()),
    }


# ============================================================================
# 4/…  CALIBRATION helper (works across sklearn versions)
# ============================================================================
def prefit_calibrator(fitted_estimator, X_val, y_val, method="isotonic"):
    """Calibrate an already-fitted estimator on the validation set."""
    try:
        from sklearn.frozen import FrozenEstimator      # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method=method)
        cal.fit(X_val, y_val)
    except Exception:
        cal = CalibratedClassifierCV(fitted_estimator, method=method, cv="prefit")
        cal.fit(X_val, y_val)
    return cal


# ============================================================================
# 6. PLOTS
# ============================================================================
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [fig] {path}")


def plot_roc(results, y_te, figdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name in MODEL_ORDER:
        proba = results[name]["proba_te"]
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc = results[name]["test"]["roc_auc"]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set"); ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(figdir, "roc_curves.png"))


def plot_pr(results, y_te, figdir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    base = float(np.mean(y_te))
    for name in MODEL_ORDER:
        proba = results[name]["proba_te"]
        prec, rec, _ = precision_recall_curve(y_te, proba)
        ap = results[name]["test"]["pr_auc"]
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
                label=f"{name} (AP={ap:.3f})")
    ax.axhline(base, ls="--", color="k", alpha=0.6, label=f"Baseline ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — Test Set"); ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(figdir, "pr_curves.png"))


def plot_confusion(results, y_te, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, name in zip(axes, MODEL_ORDER):
        pred = (results[name]["proba_te"] >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_te, pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices — Test Set (threshold = 0.5)", y=1.03, fontsize=13)
    _save(fig, os.path.join(figdir, "confusion_matrices.png"))


def plot_calibration(y_te, proba_uncal, proba_cal, best_name, figdir):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for proba, lab, col in [(proba_uncal, "Before calibration", "#ff7f0e"),
                            (proba_cal,   "After calibration",  "#1f77b4")]:
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=col, label=lab)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram — {best_name}"); ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(figdir, "calibration_best_model.png"))


def plot_proba_hist(y_te, proba, best_name, figdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(proba[y_te == 0], bins=40, alpha=0.6, color="#1f77b4",
            label="True label 0", density=True)
    ax.hist(proba[y_te == 1], bins=40, alpha=0.6, color="#d62728",
            label="True label 1", density=True)
    ax.axvline(THRESHOLD, ls="--", color="k", alpha=0.7, label="Threshold 0.5")
    ax.set_xlabel("Predicted probability of client"); ax.set_ylabel("Density")
    ax.set_title(f"Predicted-Probability Distribution — {best_name} (Test)")
    ax.legend()
    _save(fig, os.path.join(figdir, "proba_histogram_best_model.png"))


def plot_feature_importance(best_name, best_pipe, X_te, figdir, top_n=20):
    """SHAP summary bar for tree models; |coef| for Logistic Regression."""
    prep = best_pipe.named_steps["prep"]
    model = best_pipe.named_steps["model"]
    feat_names = list(prep.get_feature_names_out())

    if best_name == "LogisticRegression":
        coefs = model.coef_.ravel()
        imp = pd.Series(np.abs(coefs), index=feat_names).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 7))
        imp[::-1].plot.barh(ax=ax, color="#1f77b4")
        ax.set_xlabel("|coefficient| (standardized features)")
        ax.set_title(f"Top {top_n} Features — {best_name} (|coef|)")
        _save(fig, os.path.join(figdir, "feature_importance_best_model.png"))
        return

    # tree models -> SHAP
    try:
        import shap
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="shap")
        Xt = prep.transform(X_te)
        # sample for speed
        if hasattr(Xt, "iloc") and Xt.shape[0] > 800:
            Xt = Xt.iloc[:800]
        elif not hasattr(Xt, "iloc") and Xt.shape[0] > 800:
            Xt = Xt[:800]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):          # older API: [class0, class1]
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        fig = plt.figure()
        shap.summary_plot(sv, Xt, feature_names=feat_names, plot_type="bar",
                          max_display=top_n, show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Feature Importance — {best_name} (top {top_n})", y=1.02)
        _save(fig, os.path.join(figdir, "feature_importance_best_model.png"))
    except Exception as e:
        warnings.warn(f"SHAP failed ({e}); falling back to native importance.")
        imp = pd.Series(model.feature_importances_, index=feat_names)
        imp = imp.sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 7))
        imp[::-1].plot.barh(ax=ax, color=MODEL_COLORS.get(best_name, "#333"))
        ax.set_xlabel("importance"); ax.set_title(f"Top {top_n} Features — {best_name}")
        _save(fig, os.path.join(figdir, "feature_importance_best_model.png"))


def plot_metric_bars(comparison: pd.DataFrame, figdir):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(metrics)); w = 0.25
    for i, name in enumerate(MODEL_ORDER):
        vals = [comparison.loc[name, m] for m in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=MODEL_COLORS[name])
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Test Metrics (threshold = 0.5)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(figdir, "metric_comparison_bars.png"))


# ============================================================================
# 7. DEPLOYMENT
# ============================================================================
def load_bundle(path="best_model.joblib"):
    return joblib.load(path)


def predict_proba_new(df_new: pd.DataFrame, bundle=None,
                      bundle_path="best_model.joblib") -> np.ndarray:
    """
    Score new raw businesses (same raw columns as training).
    Returns a 1-D array of P(client) in [0, 1]. Missing feature columns are
    added as NaN and imputed by the pipeline; extra columns are ignored.
    """
    if bundle is None:
        bundle = load_bundle(bundle_path)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    X = df_new.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    return model.predict_proba(X)[:, 1]


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Ooredoo B2B supervised lead-scoring pipeline")
    parser.add_argument("--input", default="ranked_prospects_v3.csv", help="input CSV path")
    parser.add_argument("--outdir", default=".", help="output directory")
    parser.add_argument("--calibration", default="isotonic",
                        choices=["isotonic", "sigmoid"], help="calibration method")
    args = parser.parse_args()

    outdir = args.outdir
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 78)
    print("OOREDOO B2B LEAD SCORING — SUPERVISED PIPELINE")
    print("=" * 78)

    # ---- 1. data prep ----
    df = load_data(args.input)
    df = engineer_presence_flags(df)
    df = ensure_target(df)
    schema = identify_columns(df)
    S = split_data(df, schema)

    # ---- 2. models + tuning ----
    specs = build_model_specs(schema["numeric"], schema["categorical"], S["y_tr"])
    print("\n[train] Tuning each model on the validation set (criterion: PR-AUC)…")
    fitted, results = {}, {}
    for name in MODEL_ORDER:
        make_fn, grid = specs[name]
        est, params, val_ap = tune_on_validation(
            make_fn, grid, S["X_tr"], S["y_tr"], S["X_val"], S["y_val"])
        fitted[name] = est
        proba_val = est.predict_proba(S["X_val"])[:, 1]
        proba_te = est.predict_proba(S["X_te"])[:, 1]
        results[name] = dict(
            params=params, val_ap=val_ap,
            proba_val=proba_val, proba_te=proba_te,
            test=metrics_at_threshold(S["y_te"], proba_te),
        )
        print(f"    {name:18s} best params={params}  val PR-AUC={val_ap:.4f}")

    # ---- 3. evaluation table ----
    metric_cols = ["accuracy", "precision", "recall", "f1", "balanced_accuracy",
                   "roc_auc", "pr_auc", "brier", "log_loss"]
    comparison = pd.DataFrame(
        {name: {m: results[name]["test"][m] for m in metric_cols} for name in MODEL_ORDER}
    ).T[metric_cols]
    comparison.index.name = "model"
    comp_path = os.path.join(outdir, "model_comparison.csv")
    comparison.to_csv(comp_path)
    print("\n[eval] Test-set metrics (threshold = 0.5):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(comparison.round(4).to_string())
    print(f"[eval] saved -> {comp_path}")

    # ---- 5. known-customer sanity anchor ----
    known_rows = []
    for name in MODEL_ORDER:
        pred_bin = (results[name]["proba_te"] >= THRESHOLD).astype(int)
        chk = known_customer_check(pred_bin, results[name]["proba_te"], S["k_te"] == 1)
        chk["model"] = name
        known_rows.append(chk)
    known_df = pd.DataFrame(known_rows).set_index("model")[["n_known", "known_recall", "known_mean_proba"]]
    known_path = os.path.join(outdir, "known_customer_check.csv")
    known_df.to_csv(known_path)
    print("\n[anchor] Known-customer check on the TEST split "
          "(only is_known_customer==1 are verified positives):")
    print(known_df.round(4).to_string())
    print(f"[anchor] saved -> {known_path}")

    # ---- 4. model selection by VALIDATION PR-AUC ----
    best_name = max(MODEL_ORDER, key=lambda n: results[n]["val_ap"])
    best_pipe = fitted[best_name]
    print(f"\n[select] Best model by validation PR-AUC: {best_name} "
          f"(val PR-AUC={results[best_name]['val_ap']:.4f})")

    # ---- calibration of the best model on the validation set ----
    proba_te_uncal = results[best_name]["proba_te"]
    brier_before = brier_score_loss(S["y_te"], proba_te_uncal)
    calibrated = prefit_calibrator(best_pipe, S["X_val"], S["y_val"], method=args.calibration)
    proba_te_cal = calibrated.predict_proba(S["X_te"])[:, 1]
    brier_after = brier_score_loss(S["y_te"], proba_te_cal)
    print(f"[calib] {args.calibration} calibration  Brier: "
          f"{brier_before:.4f} (before) -> {brier_after:.4f} (after)")

    cal_summary = pd.DataFrame({
        "metric": ["brier_before", "brier_after"],
        "value":  [brier_before, brier_after],
    })
    cal_summary.to_csv(os.path.join(outdir, "calibration_summary.csv"), index=False)

    # ---- 6. plots ----
    print("\n[plots] Writing thesis-ready figures to figures/ …")
    plot_roc(results, S["y_te"].values, figdir)
    plot_pr(results, S["y_te"].values, figdir)
    plot_confusion(results, S["y_te"].values, figdir)
    plot_calibration(S["y_te"].values, proba_te_uncal, proba_te_cal, best_name, figdir)
    plot_proba_hist(S["y_te"].values, proba_te_cal, best_name, figdir)
    plot_feature_importance(best_name, best_pipe, S["X_te"], figdir)
    plot_metric_bars(comparison, figdir)

    # ---- 7. deployment artifacts ----
    print("\n[deploy] Saving model bundle + scoring artifacts …")
    bundle = dict(
        model=calibrated,                       # calibrated pipeline (preprocessing inside)
        uncalibrated_model=best_pipe,
        feature_columns=schema["feature_cols"],
        numeric=schema["numeric"],
        categorical=schema["categorical"],
        best_model_name=best_name,
        threshold=THRESHOLD,
        calibration=args.calibration,
        sklearn_note="predict_proba(...)[:,1] -> P(client); binary at threshold 0.5",
    )
    model_path = os.path.join(outdir, "best_model.joblib")
    joblib.dump(bundle, model_path)
    print(f"    [model] {model_path}")

    # score the full dataset with the calibrated best model
    full_proba = predict_proba_new(df[schema["feature_cols"]], bundle=bundle)
    scored = df.copy()
    scored["prob_client"] = full_proba
    scored["predicted_client"] = (full_proba >= THRESHOLD).astype(int)
    scored_path = os.path.join(outdir, "scored_businesses.csv")
    scored.to_csv(scored_path, index=False)
    print(f"    [data ] {scored_path}  (prob_client for all {len(scored):,} rows)")

    # ---- 8. final summary ----
    bt = comparison.loc[best_name]
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"Split sizes      : train={len(S['y_tr']):,}  "
          f"val={len(S['y_val']):,}  test={len(S['y_te']):,}")
    print(f"Best model       : {best_name}  "
          f"(selected by highest validation PR-AUC = {results[best_name]['val_ap']:.4f})")
    print(f"Test ROC-AUC     : {bt['roc_auc']:.4f}")
    print(f"Test PR-AUC      : {bt['pr_auc']:.4f}")
    print(f"Test F1 / recall : {bt['f1']:.4f} / {bt['recall']:.4f}")
    print(f"Brier (cal.)     : {brier_before:.4f} -> {brier_after:.4f}")
    kc = known_df.loc[best_name]
    print(f"Known-customer   : recall={kc['known_recall']:.4f}  "
          f"mean_proba={kc['known_mean_proba']:.4f}  (n={int(kc['n_known'])})")
    print("Artifacts        : model_comparison.csv, known_customer_check.csv,")
    print("                   calibration_summary.csv, scored_businesses.csv,")
    print("                   best_model.joblib, figures/*.png")
    print("=" * 78)

    # machine-readable run summary for the thesis
    run_summary = dict(
        split_sizes=dict(train=len(S["y_tr"]), val=len(S["y_val"]), test=len(S["y_te"])),
        best_model=best_name,
        selection_criterion="validation PR-AUC (average precision)",
        val_pr_auc={n: float(results[n]["val_ap"]) for n in MODEL_ORDER},
        test_metrics={n: {m: float(results[n]["test"][m]) for m in metric_cols}
                      for n in MODEL_ORDER},
        brier_before=float(brier_before), brier_after=float(brier_after),
        known_customer_check=known_df.reset_index().to_dict(orient="records"),
    )
    with open(os.path.join(outdir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)


if __name__ == "__main__":
    main()
