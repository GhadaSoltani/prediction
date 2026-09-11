"""
prior_estimation.py -- Robust class prior estimators for PU learning.

Replaces the spy-based estimator in pu_methods.py (which gave the
implausible pi_hat = 0.77 on the Ooredoo data) with two better options:

  A. estimate_c_holdout    -- Elkan-Noto 'e1' estimator using held-out
                              positives. Simplest, most robust when you
                              have a validation set of positives.
  B. estimate_c_tice       -- TIcE (Bekker & Davis 2018): find subsets
                              of feature space with high labeling ratio;
                              max ratio is a lower-bound on c.

Both return the same triple (pi_hat, c_hat, threshold) as the spy method,
so they drop into lead_scoring_v2.py without other code changes.

Notation reminder:
  s = observed label  (1 = "known positive",  0 = "unlabeled")
  y = true label      (1 = positive,          0 = negative)  -- unobserved
  c = P(s = 1 | y = 1)                    -- labeling probability
  pi = P(y = 1)                            -- true prior of positives
  Under SCAR: P(s=1|x) = c * P(y=1|x)      -- Elkan-Noto 2008
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


def _make_lgb(random_state=42):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=random_state,
        n_jobs=-1, verbose=-1,
    )


# ============================================================
# A. HELD-OUT POSITIVE ESTIMATOR  (Elkan-Noto e1)
# ============================================================
def estimate_c_holdout(X, y_train, holdout_pos_idx, random_state=42,
                       reliable_neg_quantile=0.05):
    """
    Elkan-Noto e1 estimator.

    Idea: Under SCAR, a naive classifier trained on labeled-vs-unlabeled
    outputs P(labeled=1 | x). For a truly positive x that happens to be
    unlabeled, this expected output equals c. So:

        c_hat  =  mean( f(x) )  over held-out positives

    We use held-out positives that the caller has already reserved for
    evaluation. They were NOT used in training, so their scores are unbiased.

    Returns
    -------
    pi_hat    : estimated prior P(y=1) in the unlabeled pool
    c_hat     : estimated P(labeled | positive)
    threshold : score threshold for reliable-negative extraction
    """
    # CRITICAL: exclude holdout positives from training, otherwise the
    # classifier overfits them to label 0 (they appear as unlabeled) and
    # their scores collapse to ~0, biasing c_hat toward 0.
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[holdout_pos_idx] = False

    clf = _make_lgb(random_state)
    clf.fit(X.iloc[train_mask], y_train[train_mask])
    scores = clf.predict_proba(X)[:, 1]

    holdout_scores = scores[holdout_pos_idx]
    # Mean is the standard e1 estimator. Trim outliers for robustness.
    c_hat = float(np.mean(np.clip(holdout_scores, 1e-6, 1.0)))

    # Convert c_hat to pi_hat.
    #   Let n_L = number of labeled points, n_U = number of unlabeled.
    #   P(labeled) = n_L / n  =  P(y=1) * c   =>   pi = P(labeled) / c
    #   That gives pi over the WHOLE data. We want pi over the UNLABELED pool.
    #   pi_u = P(y=1 | s=0) = P(y=1) * (1 - c) / P(s=0)
    n = len(y_train)
    n_labeled = int(y_train.sum())
    p_labeled = n_labeled / n
    if c_hat > 1e-6:
        pi_total = min(p_labeled / c_hat, 0.99)
    else:
        pi_total = p_labeled

    p_s0 = 1.0 - p_labeled
    pi_hat_unlabeled = float(pi_total * (1.0 - c_hat) / p_s0) if p_s0 > 0 else 0.0
    pi_hat_unlabeled = max(0.0, min(pi_hat_unlabeled, 0.99))

    # Reliable-negative threshold: unlabeled points scoring far below
    # the held-out positives are likely truly negative.
    threshold = float(np.quantile(holdout_scores, reliable_neg_quantile))
    threshold = max(threshold, 1e-4)   # avoid zero-threshold pathology

    return pi_hat_unlabeled, c_hat, threshold


# ============================================================
# B. TIcE ESTIMATOR  (Bekker & Davis 2018, simplified)
# ============================================================
def estimate_c_tice(X, y_train, n_folds=5, delta=0.5, max_depth=8,
                     min_samples_leaf=30, random_state=42):
    """
    TIcE: Tree Induction for c Estimation. Simplified single-tree version.

    Idea: Under SCAR, in any subset A of the feature space:
        labeled_fraction(A) = P(s=1 | x in A) = c * P(y=1 | x in A) <= c
    So the maximum labeled_fraction across many subsets is a LOWER BOUND
    on c. Bekker-Davis prove that with a concentration correction:

        c_hat  =  max_{leaf L in tree}  BEPP(L)

    where BEPP is a lower-confidence-bound on the labeled fraction in L.
    Fitting a tree to predict s=labeled naturally finds subsets with the
    highest labeling ratios (that is the tree's objective).

    We do k-fold splitting: fit the tree on fold train, then evaluate BEPP
    on fold test. Averaging c_hat across folds reduces variance.

    Parameters
    ----------
    delta : confidence level for the lower bound (smaller = more conservative)
    max_depth, min_samples_leaf : tree regularization to avoid tiny leaves
                                    that would inflate BEPP by chance
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    c_estimates = []

    # If X has categorical dtypes (LightGBM style), one-hot them for the tree
    if any(str(X[c].dtype) == "category" for c in X.columns):
        X_tree = pd.get_dummies(X, dummy_na=False)
    else:
        X_tree = X.copy()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tree)):
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold,
        )
        tree.fit(X_tree.iloc[train_idx].values, y_train[train_idx])

        # For each test point, find its leaf id
        leaf_ids = tree.apply(X_tree.iloc[test_idx].values)
        y_test = y_train[test_idx]

        # Per-leaf BEPP: (n_labeled - correction) / n_total
        best_bepp = 0.0
        for leaf in np.unique(leaf_ids):
            mask = (leaf_ids == leaf)
            n_leaf = int(mask.sum())
            if n_leaf < min_samples_leaf:
                continue
            n_lab = int(y_test[mask].sum())
            frac = n_lab / n_leaf
            # Bekker-Davis lower-confidence-bound:
            #   BEPP = frac - sqrt( (1 - frac) * ln(1/delta) / n_leaf )
            correction = np.sqrt(max(0.0, (1 - frac) * np.log(1.0 / delta) / n_leaf))
            bepp = frac - correction
            if bepp > best_bepp:
                best_bepp = bepp
        c_estimates.append(best_bepp)

    c_hat = float(np.mean(c_estimates))
    c_hat = max(c_hat, 1e-3)   # avoid zero

    # Convert to pi_hat (same conversion as holdout estimator)
    n = len(y_train)
    p_labeled = int(y_train.sum()) / n
    pi_total = min(p_labeled / c_hat, 0.99)
    p_s0 = 1.0 - p_labeled
    pi_hat_unlabeled = float(pi_total * (1.0 - c_hat) / p_s0) if p_s0 > 0 else 0.0
    pi_hat_unlabeled = max(0.0, min(pi_hat_unlabeled, 0.99))

    # Reliable-neg threshold: retrain naive classifier once, pick low quantile
    clf = _make_lgb(random_state)
    clf.fit(X, y_train)
    scores = clf.predict_proba(X)[:, 1]
    threshold = float(np.quantile(scores[y_train == 1], 0.05))
    threshold = max(threshold, 1e-4)

    return pi_hat_unlabeled, c_hat, threshold


# ============================================================
# C. ENSEMBLE OF ESTIMATORS
# ============================================================
def estimate_c_robust(X, y_train, holdout_pos_idx, random_state=42, verbose=True):
    """
    Runs both estimators and returns:
      - median c_hat  (robust central tendency)
      - min pi_hat    (conservative -- prefer under- to over-correcting)
      - median threshold
      - per-method dict for the report

    Recommended: this is what lead_scoring_v3 should call.
    """
    results = {}

    pi_h, c_h, thr_h = estimate_c_holdout(X, y_train, holdout_pos_idx,
                                           random_state=random_state)
    results["holdout"] = {"pi": pi_h, "c": c_h, "threshold": thr_h}
    if verbose:
        print(f"  Holdout estimator: pi_hat={pi_h:.4f}  c_hat={c_h:.4f}  thr={thr_h:.6f}")

    pi_t, c_t, thr_t = estimate_c_tice(X, y_train, random_state=random_state)
    results["tice"] = {"pi": pi_t, "c": c_t, "threshold": thr_t}
    if verbose:
        print(f"  TIcE estimator:    pi_hat={pi_t:.4f}  c_hat={c_t:.4f}  thr={thr_t:.6f}")

    c_final  = float(np.median([c_h, c_t]))
    pi_final = float(min(pi_h, pi_t))
    thr_final = float(np.median([thr_h, thr_t]))
    if verbose:
        print(f"  --> Final (median c, min pi): pi_hat={pi_final:.4f}  c_hat={c_final:.4f}")

    return pi_final, c_final, thr_final, results
