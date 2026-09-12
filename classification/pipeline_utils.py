#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_utils.py
=================
Custom, picklable preprocessing components shared by the training script
(`supervised_models.py`) and the scoring script (`predict.py`).

This lives in its own importable module (not inside a script run as __main__)
so that a model saved with joblib can be reloaded from any process/script:
a custom transformer defined in __main__ cannot be unpickled elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LGBMFramePrep(BaseEstimator, TransformerMixin):
    """
    Preprocessor that returns a pandas DataFrame so LightGBM can use its native
    categorical handling. Numeric cols get median imputation + a missing
    indicator; categorical cols get a sentinel fill and `category` dtype.
    """
    def __init__(self, numeric, categorical):
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.medians_ = {c: X[c].median() for c in self.numeric}
        # only add an indicator for numeric columns that actually have NaNs in train
        self.indicator_cols_ = [c for c in self.numeric if X[c].isna().any()]
        # remember training categories so unseen test categories are handled cleanly
        self.categories_ = {
            c: pd.Index(X[c].astype("object").fillna("__NA__").unique())
            for c in self.categorical
        }
        self.feature_names_out_ = (
            self.numeric
            + [f"{c}_missing" for c in self.indicator_cols_]
            + self.categorical
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)
        # numeric: coerce to numeric, median-impute
        for c in self.numeric:
            col = pd.to_numeric(X[c], errors="coerce") if c in X else np.nan
            out[c] = pd.Series(col, index=X.index).fillna(self.medians_[c])
        # missing indicators (only for columns that had NaNs at fit time)
        for c in self.indicator_cols_:
            src = pd.to_numeric(X[c], errors="coerce") if c in X \
                else pd.Series(np.nan, index=X.index)
            out[f"{c}_missing"] = src.isna().astype(int)
        # categorical -> category dtype (LightGBM auto-detects these)
        for c in self.categorical:
            vals = X[c].astype("object").fillna("__NA__") if c in X \
                else pd.Series("__NA__", index=X.index)
            cats = self.categories_[c]
            if "__NA__" not in cats:
                cats = cats.append(pd.Index(["__NA__"]))
            # categories unseen during fit -> NaN (LightGBM treats as missing)
            vals = vals.where(vals.isin(cats), np.nan)
            out[c] = pd.Categorical(vals, categories=cats)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
