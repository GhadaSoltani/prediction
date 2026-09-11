"""
pu_methods.py -- Advanced PU learning methods for Ooredoo lead scoring.

Provides:
  - estimate_class_prior_spy: estimate pi = P(y=1) in unlabeled via spy technique
  - bagging_pu: refactored Bagging-PU with OOB scoring
  - prior_corrected_bagging_pu: same, but with class-prior sample weighting
      (nnPU spirit: down-weight unlabeled-as-negative by (1 - pi))
  - extract_reliable_negatives: two-step PU -- find unlabeled points that
      are very likely truly negative, for a standard retrain
  - retrain_with_reliable_negatives: standard LightGBM on P + reliable N
  - ensemble_scores: rank-based ensemble across multiple ranker outputs
  - calibrate_scores_isotonic: convert scores to probabilities using held-out
      positives + reliable negatives
  - stability_across_seeds: run any pu method with k seeds, return mean & std

All rankers return an array of length n with scores in [0, 1]-ish (higher = more
customer-like). None of them are calibrated by default; use calibrate_ for that.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")


def _make_lgb(random_state=42, n_estimators=200):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=random_state,
        n_jobs=-1, verbose=-1,
    )


# ============================================================
# 1. CLASS PRIOR ESTIMATION (spy technique)
# ============================================================
def estimate_class_prior_spy(X, y_train, spy_ratio=0.15,
                              threshold_quantile=0.05, random_state=42):
    """
    Estimate pi = P(y=1 | unlabeled) via the spy technique (Liu et al. 2003).

    Idea: Hide `spy_ratio` of the known positives inside the unlabeled pool.
    Train a naive classifier. Spies should score high because they're really
    positive. Pick the score at the 5th percentile of spy scores -- unlabeled
    points below this threshold are "reliable negatives"; unlabeled above
    are the estimated positive contamination.

    Returns
    -------
    pi_hat    : float, estimated P(y=1) in the unlabeled pool
    c_hat     : float, estimated P(labeled | positive) -- Elkan-Noto's c
    threshold : score threshold below which points are reliable negatives
    """
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y_train == 1)[0]
    unl_idx = np.where(y_train == 0)[0]

    n_spies = max(int(spy_ratio * len(pos_idx)), 20)
    spy_idx = rng.choice(pos_idx, size=n_spies, replace=False)

    y_spy = y_train.copy()
    y_spy[spy_idx] = 0

    clf = _make_lgb(random_state)
    clf.fit(X, y_spy)
    scores = clf.predict_proba(X)[:, 1]

    spy_scores = scores[spy_idx]
    threshold  = float(np.quantile(spy_scores, threshold_quantile))

    unl_scores = scores[unl_idx]
    pi_hat = float((unl_scores > threshold).mean())
    c_hat  = float(spy_scores.mean())

    return pi_hat, c_hat, threshold


# ============================================================
# 2. BAGGING-PU (refactored from your notebook)
# ============================================================
def bagging_pu(X, y_train, n_iterations=50, negative_multiple=3,
               positive_weight=None, random_state=42, verbose=True):
    """
    Mordelet & Vert 2014, with OOB scoring. `positive_weight` optionally
    lets us down-weight borderline entity-linkage matches.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    pos_idx = np.where(y_train == 1)[0]
    unl_idx = np.where(y_train == 0)[0]
    neg_sample_size = negative_multiple * len(pos_idx)

    score_sum = np.zeros(n)
    score_count = np.zeros(n, dtype=int)

    if positive_weight is None:
        pos_w = np.ones(len(pos_idx))
    else:
        pos_w = np.asarray(positive_weight)[pos_idx]

    for i in range(n_iterations):
        neg_sample = rng.choice(unl_idx, size=neg_sample_size, replace=False)
        train_idx  = np.concatenate([pos_idx, neg_sample])
        y_round    = np.concatenate([np.ones(len(pos_idx)),
                                     np.zeros(neg_sample_size)]).astype(int)
        w_round    = np.concatenate([pos_w, np.ones(neg_sample_size)])

        clf = _make_lgb(random_state + i)
        clf.fit(X.iloc[train_idx], y_round, sample_weight=w_round)

        oob_mask = np.ones(n, dtype=bool)
        oob_mask[neg_sample] = False
        oob_scores = clf.predict_proba(X[oob_mask])[:, 1]

        score_sum[oob_mask]   += oob_scores
        score_count[oob_mask] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Bagging-PU iter {i+1}/{n_iterations}")

    assert score_count.min() > 0, "Some rows never OOB. Increase n_iterations."
    return score_sum / score_count


# ============================================================
# 3. PRIOR-CORRECTED BAGGING-PU  (nnPU-spirit for trees)
# ============================================================
def prior_corrected_bagging_pu(X, y_train, pi_hat, n_iterations=50,
                                negative_multiple=3, positive_weight=None,
                                random_state=42, verbose=True):
    """
    Same as bagging_pu but pseudo-negatives are weighted by (1 - pi_hat).
    Rationale: since pi_hat of unlabeled are actually positives, treating
    them all as fully negative is systematically wrong. Down-weighting
    corrects this bias -- similar in spirit to nnPU's risk correction.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    pos_idx = np.where(y_train == 1)[0]
    unl_idx = np.where(y_train == 0)[0]
    neg_sample_size = negative_multiple * len(pos_idx)
    neg_weight = max(0.05, 1.0 - pi_hat)   # floor to avoid instability

    score_sum = np.zeros(n)
    score_count = np.zeros(n, dtype=int)

    if positive_weight is None:
        pos_w = np.ones(len(pos_idx))
    else:
        pos_w = np.asarray(positive_weight)[pos_idx]

    for i in range(n_iterations):
        neg_sample = rng.choice(unl_idx, size=neg_sample_size, replace=False)
        train_idx  = np.concatenate([pos_idx, neg_sample])
        y_round    = np.concatenate([np.ones(len(pos_idx)),
                                     np.zeros(neg_sample_size)]).astype(int)
        w_round    = np.concatenate([pos_w,
                                     np.full(neg_sample_size, neg_weight)])

        clf = _make_lgb(random_state + i)
        clf.fit(X.iloc[train_idx], y_round, sample_weight=w_round)

        oob_mask = np.ones(n, dtype=bool)
        oob_mask[neg_sample] = False
        oob_scores = clf.predict_proba(X[oob_mask])[:, 1]

        score_sum[oob_mask]   += oob_scores
        score_count[oob_mask] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  prior-corrected PU iter {i+1}/{n_iterations}  (neg_w={neg_weight:.3f})")

    return score_sum / score_count


# ============================================================
# 4. TWO-STEP PU: reliable negatives + retrain
# ============================================================
def extract_reliable_negatives(X, y_train, threshold, base_scores=None,
                               random_state=42):
    """
    Given a spy-derived score threshold, return the indices of unlabeled
    points scoring below it -- these are the "reliable negatives" for a
    standard retrain.

    If base_scores is None, retrains a naive classifier on (P vs full U)
    to get scores.
    """
    if base_scores is None:
        clf = _make_lgb(random_state)
        clf.fit(X, y_train)
        base_scores = clf.predict_proba(X)[:, 1]

    unl_idx = np.where(y_train == 0)[0]
    reliable_neg = unl_idx[base_scores[unl_idx] < threshold]
    return reliable_neg, base_scores


def retrain_with_reliable_negatives(X, y_train, reliable_neg_idx,
                                     positive_weight=None, random_state=42):
    """
    Standard classifier on: known positives (label 1) vs reliable negatives
    (label 0). Returns scores for ALL rows in X.
    """
    pos_idx = np.where(y_train == 1)[0]
    train_idx = np.concatenate([pos_idx, reliable_neg_idx])
    y_round   = np.concatenate([np.ones(len(pos_idx)),
                                np.zeros(len(reliable_neg_idx))]).astype(int)
    if positive_weight is None:
        w_round = None
    else:
        w_round = np.concatenate([np.asarray(positive_weight)[pos_idx],
                                  np.ones(len(reliable_neg_idx))])
    clf = _make_lgb(random_state)
    clf.fit(X.iloc[train_idx], y_round, sample_weight=w_round)
    return clf.predict_proba(X)[:, 1]


# ============================================================
# 5. ENSEMBLE (rank-based)
# ============================================================
def ensemble_scores(score_arrays, weights=None):
    """
    Rank-based average of multiple score arrays. More robust than raw score
    averaging when different methods produce different score scales.

    Returns normalized average rank in [0, 1].
    """
    ranks = np.stack([rankdata(s) for s in score_arrays])  # (n_methods, n_rows)
    if weights is None:
        avg_rank = ranks.mean(axis=0)
    else:
        w = np.array(weights, dtype=float) / np.sum(weights)
        avg_rank = (ranks * w[:, None]).sum(axis=0)
    return avg_rank / ranks.shape[1]


# ============================================================
# 6. SCORE CALIBRATION (isotonic on held-out P + reliable N)
# ============================================================
def calibrate_scores_isotonic(scores, holdout_positive_idx, reliable_neg_idx):
    """
    Fit isotonic regression on:
      - held-out positives -> label 1
      - reliable negatives -> label 0
    Then transform ALL scores to calibrated probabilities.

    Held-out positives are known positives that were NOT used in training
    (see lead_scoring_v2). Reliable negatives come from the spy step.
    """
    labels = np.concatenate([np.ones(len(holdout_positive_idx)),
                              np.zeros(len(reliable_neg_idx))])
    fit_scores = np.concatenate([scores[holdout_positive_idx],
                                  scores[reliable_neg_idx]])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(fit_scores, labels)
    return iso.transform(scores), iso


def elkan_noto_correction(scores, c_hat):
    """
    Divide raw PU scores by estimated c = P(labeled | positive) to recover
    posterior P(y=1 | x). Elkan & Noto 2008. Clip to [0, 1].
    """
    if c_hat <= 0:
        return scores
    return np.clip(scores / c_hat, 0.0, 1.0)


# ============================================================
# 7. STABILITY ACROSS SEEDS
# ============================================================
def stability_across_seeds(score_fn, seeds, **kwargs):
    """
    Run `score_fn(random_state=seed, **kwargs)` for each seed and return:
      mean_scores : (n,) mean score per row
      std_scores  : (n,) std across seeds
      all_scores  : (n_seeds, n) raw
    """
    all_scores = []
    for s in seeds:
        print(f"\n--- seed {s} ---")
        all_scores.append(score_fn(random_state=s, **kwargs))
    A = np.stack(all_scores)
    return A.mean(0), A.std(0), A


# ============================================================
# 8. EVALUATION HELPERS
# ============================================================
def precision_recall_at_k(scores, is_target, ks):
    order = np.argsort(-scores)
    hits_cum = np.cumsum(is_target[order])
    total = is_target.sum()
    rows = []
    for k in ks:
        hits = int(hits_cum[k - 1])
        rows.append({"k": k, "hits": hits,
                     "precision@k": hits / k,
                     "recall@k": hits / total if total > 0 else 0.0})
    return pd.DataFrame(rows)
