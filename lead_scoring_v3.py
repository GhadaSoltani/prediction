"""
lead_scoring_v3.py -- Step 3 of the pipeline, v3.

Same structure as v2, but the class prior estimation step (which produced
the implausible pi_hat = 0.77 on the Ooredoo data) is replaced with:
  - Elkan-Noto e1 estimator using held-out positives
  - TIcE (Bekker-Davis 2018)
  - Median-of-both, used to drive Method B and Method C
Both robust estimators are reported side-by-side with the old spy method
for the thesis comparison table.

Everything else -- Bagging-PU, prior-corrected Bagging-PU, two-step PU,
ensemble, calibration, stability -- is identical to v2.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pu_methods import (
    estimate_class_prior_spy,        # kept for comparison
    bagging_pu,
    prior_corrected_bagging_pu,
    extract_reliable_negatives,
    retrain_with_reliable_negatives,
    ensemble_scores,
    calibrate_scores_isotonic,
    precision_recall_at_k,
    stability_across_seeds,
)
from prior_estimation import (
    estimate_c_holdout,
    estimate_c_tice,
    estimate_c_robust,
)

warnings.filterwarnings("ignore")

# ============================ CONFIG ====================================
FEATURES_FILE = "features_v2.csv"
MAPS_FILE     = "maps_labeled_v3.csv" if os.path.exists("maps_labeled_v3.csv") else (
                "maps_labeled_v2.csv" if os.path.exists("maps_labeled_v2.csv") else
                "maps_labeled.csv")
N_ITERATIONS  = 50
NEG_MULTIPLE  = 3
HOLDOUT_FRAC  = 0.20
K_VALUES      = [100, 500, 1000, 2000, 3000, 5000]
SEEDS         = [42, 43, 44, 45, 46]
TOP_N_PER_SEGMENT = 50
RANDOM_SEED   = 42

# ============================ LOAD ======================================
print("=" * 70)
print("Ooredoo Lead-Scoring v3 (robust prior estimation)")
print("=" * 70)

features = pd.read_csv(FEATURES_FILE)
for c in ("category", "search_category", "governorate"):
    features[c] = features[c].astype("category")

y_all = features["is_customer"].values
positive_weight_full = features["positive_weight"].values if "positive_weight" in features.columns else np.ones(len(features))
X = features.drop(columns=["is_customer", "positive_weight"], errors="ignore")

print(f"Rows: {len(X):,}  Features: {X.shape[1]}  Positives: {int(y_all.sum()):,}")

pos_idx_all = np.where(y_all == 1)[0]
train_pos, holdout_pos = train_test_split(
    pos_idx_all, test_size=HOLDOUT_FRAC, random_state=RANDOM_SEED)

y_train = np.zeros(len(y_all), dtype=int)
y_train[train_pos] = 1
is_holdout = np.zeros(len(y_all), dtype=bool)
is_holdout[holdout_pos] = True

print(f"Training positives: {len(train_pos):,}  Held-out: {len(holdout_pos):,}")

# =================== STEP 1: PRIOR ESTIMATION (compare 3 methods) =======
print("\n[1/6] Estimating class prior -- comparing spy vs holdout vs TIcE")

pi_spy, c_spy, thr_spy = estimate_class_prior_spy(
    X, y_train, spy_ratio=0.15, threshold_quantile=0.05, random_state=RANDOM_SEED)
print(f"  Spy   (unreliable): pi_hat={pi_spy:.4f}  c_hat={c_spy:.4f}")

pi_hat, c_hat, spy_threshold, all_estimators = estimate_c_robust(
    X, y_train, holdout_pos_idx=holdout_pos, random_state=RANDOM_SEED, verbose=True)

# Save the comparison for the thesis
prior_comparison = pd.DataFrame({
    "estimator": ["spy", "holdout", "tice", "final (robust)"],
    "pi_hat":    [pi_spy, all_estimators["holdout"]["pi"],
                  all_estimators["tice"]["pi"], pi_hat],
    "c_hat":     [c_spy, all_estimators["holdout"]["c"],
                  all_estimators["tice"]["c"], c_hat],
    "threshold": [thr_spy, all_estimators["holdout"]["threshold"],
                  all_estimators["tice"]["threshold"], spy_threshold],
}).round(4)
prior_comparison.to_csv("prior_comparison.csv", index=False)
print("\nPrior estimator comparison (saved to prior_comparison.csv):")
print(prior_comparison.to_string(index=False))

# =================== STEP 2-4: THREE PU METHODS =========================
print(f"\n[2/6] Method A: Bagging-PU (baseline, ignores pi)")
scores_bagging = bagging_pu(X, y_train, n_iterations=N_ITERATIONS,
                             negative_multiple=NEG_MULTIPLE,
                             positive_weight=positive_weight_full,
                             random_state=RANDOM_SEED)

print(f"\n[3/6] Method B: Prior-corrected Bagging-PU (pi_hat={pi_hat:.4f})")
scores_prior = prior_corrected_bagging_pu(
    X, y_train, pi_hat=pi_hat, n_iterations=N_ITERATIONS,
    negative_multiple=NEG_MULTIPLE, positive_weight=positive_weight_full,
    random_state=RANDOM_SEED)

print(f"\n[4/6] Method C: Two-step PU (threshold={spy_threshold:.6f})")
reliable_neg, base_scores = extract_reliable_negatives(
    X, y_train, threshold=spy_threshold, random_state=RANDOM_SEED)
print(f"  Reliable negatives extracted: {len(reliable_neg):,}")
scores_twostep = retrain_with_reliable_negatives(
    X, y_train, reliable_neg, positive_weight=positive_weight_full,
    random_state=RANDOM_SEED)

# =================== STEP 5: ENSEMBLE + STEP 6: CALIBRATION =============
print("\n[5/6] Rank-based ensemble of A, B, C")
scores_ensemble = ensemble_scores([scores_bagging, scores_prior, scores_twostep])

print("[6/6] Calibrating via isotonic regression")
calibrated, iso = calibrate_scores_isotonic(scores_ensemble, holdout_pos, reliable_neg)

# ================ EVALUATION ============================================
print("\n" + "=" * 70)
print("Evaluation: Recall@k on held-out positives")
print("=" * 70)

methods = {
    "A_bagging_pu":      scores_bagging,
    "B_prior_corrected": scores_prior,
    "C_two_step":        scores_twostep,
    "D_ensemble":        scores_ensemble,
    "E_calibrated":      calibrated,
}
comparison = pd.concat([
    precision_recall_at_k(s, is_holdout, K_VALUES).assign(method=name)
    for name, s in methods.items()
], ignore_index=True)
pivot = comparison.pivot(index="k", columns="method", values="recall@k").round(3)
print("\nRecall@k per method:")
print(pivot.to_string())
comparison.to_csv("method_comparison_v3.csv", index=False)

# Plots
ks_smooth = list(range(50, 6001, 50))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, s in methods.items():
    curve = precision_recall_at_k(s, is_holdout, ks_smooth)
    axes[0].plot(ks_smooth, curve["recall@k"], label=name)
axes[0].set_xlabel("k"); axes[0].set_ylabel("Recall@k")
axes[0].set_title("v3 methods (with robust prior)")
axes[0].legend(); axes[0].grid(alpha=0.3)

rng = np.random.RandomState(RANDOM_SEED)
sample_unl = rng.choice(np.where(~is_holdout & (y_train == 0))[0], size=5000, replace=False)
axes[1].hist(calibrated[sample_unl], bins=40, alpha=0.5, label="unlabeled sample", density=True)
axes[1].hist(calibrated[is_holdout], bins=40, alpha=0.7, label="held-out positives", density=True)
axes[1].set_xlabel("Calibrated probability"); axes[1].set_ylabel("density")
axes[1].set_title("Calibrated score distribution"); axes[1].legend()
plt.tight_layout(); plt.savefig("evaluation_v3.png", dpi=120, bbox_inches="tight")

# ================ STABILITY =============================================
print(f"\nStability check: {SEEDS}")

def _one_run(random_state, X, y_train, holdout_pos_idx, positive_weight_full):
    pi, c, thr, _ = estimate_c_robust(X, y_train, holdout_pos_idx,
                                       random_state=random_state, verbose=False)
    s_bag = bagging_pu(X, y_train, n_iterations=N_ITERATIONS,
                       negative_multiple=NEG_MULTIPLE,
                       positive_weight=positive_weight_full,
                       random_state=random_state, verbose=False)
    s_pri = prior_corrected_bagging_pu(
        X, y_train, pi_hat=pi, n_iterations=N_ITERATIONS,
        negative_multiple=NEG_MULTIPLE, positive_weight=positive_weight_full,
        random_state=random_state, verbose=False)
    rn, _ = extract_reliable_negatives(X, y_train, threshold=thr,
                                        random_state=random_state)
    s_two = retrain_with_reliable_negatives(X, y_train, rn,
                                             positive_weight=positive_weight_full,
                                             random_state=random_state)
    return ensemble_scores([s_bag, s_pri, s_two])

mean_scores, std_scores, all_scores = stability_across_seeds(
    _one_run, seeds=SEEDS,
    X=X, y_train=y_train, holdout_pos_idx=holdout_pos,
    positive_weight_full=positive_weight_full)

stab = pd.DataFrame([{"k": k,
                       "recall_mean": np.mean([precision_recall_at_k(all_scores[i], is_holdout, [k]).iloc[0]["recall@k"]
                                                for i in range(len(SEEDS))]),
                       "recall_std":  np.std([precision_recall_at_k(all_scores[i], is_holdout, [k]).iloc[0]["recall@k"]
                                                for i in range(len(SEEDS))])}
                      for k in K_VALUES])
print("\nStability (mean +/- std across seeds):")
print(stab.round(4).to_string(index=False))
stab.to_csv("stability_v3.csv", index=False)

# ================ FINAL DELIVERABLE =====================================
maps = pd.read_csv(MAPS_FILE)
output = maps.copy()
output["pu_score_bagging"]    = scores_bagging
output["pu_score_prior_corr"] = scores_prior
output["pu_score_twostep"]    = scores_twostep
output["pu_score_ensemble"]   = scores_ensemble
output["prob_calibrated"]     = calibrated
output["prob_stability_std"]  = std_scores
output["is_known_customer"]   = output["is_customer"]
output = output.drop(columns=["is_customer"])
output = output.sort_values("prob_calibrated", ascending=False).reset_index(drop=True)
output.insert(0, "rank", np.arange(1, len(output) + 1))
output.to_csv("ranked_prospects_v3.csv", index=False)

non_customers = output[output["is_known_customer"] == 0].copy()
non_customers.sort_values("prob_calibrated", ascending=False)\
    .groupby("governorate", observed=True).head(TOP_N_PER_SEGMENT)\
    .to_csv("top_by_governorate_v3.csv", index=False)
non_customers.sort_values("prob_calibrated", ascending=False)\
    .groupby("search_category", observed=True).head(TOP_N_PER_SEGMENT)\
    .to_csv("top_by_search_category_v3.csv", index=False)

print("\nDone. Outputs: ranked_prospects_v3.csv, method_comparison_v3.csv,")
print("               stability_v3.csv, prior_comparison.csv, evaluation_v3.png,")
print("               top_by_governorate_v3.csv, top_by_search_category_v3.csv")
