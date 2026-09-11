"""
lead_scoring_v2.py -- Step 3 of the pipeline, v2.

Pipeline:
  1. Load features_v2.csv + maps_labeled_v2.csv
  2. Split known positives 80/20 (train / held-out for validation)
  3. Estimate class prior pi_hat via spy technique
  4. Run three PU rankers:
       (a) Bagging-PU (original)
       (b) Prior-corrected Bagging-PU (nnPU spirit)
       (c) Two-step: reliable negatives + standard classifier retrain
  5. Rank-based ensemble of the three
  6. Calibrate ensemble scores to probabilities via isotonic regression
  7. Multi-seed stability run for confidence intervals
  8. Deliverables:
       - ranked_prospects_v2.csv     : all businesses, ranked, with calibrated prob
       - top_by_governorate.csv      : top-N per governorate
       - top_by_search_category.csv  : top-N per search_category
       - method_comparison_v2.csv    : recall@k across methods
       - evaluation_v2.png           : recall curve

Run: python lead_scoring_v2.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pu_methods import (
    estimate_class_prior_spy,
    bagging_pu,
    prior_corrected_bagging_pu,
    extract_reliable_negatives,
    retrain_with_reliable_negatives,
    ensemble_scores,
    calibrate_scores_isotonic,
    elkan_noto_correction,
    precision_recall_at_k,
    stability_across_seeds,
)

warnings.filterwarnings("ignore")

# ============================ CONFIG ====================================
FEATURES_FILE = "features_v2.csv"
MAPS_FILE     = "maps_labeled_v2.csv" if os.path.exists("maps_labeled_v2.csv") else "maps_labeled.csv"
N_ITERATIONS  = 50
NEG_MULTIPLE  = 3
HOLDOUT_FRAC  = 0.20
K_VALUES      = [100, 500, 1000, 2000, 3000, 5000]
SEEDS         = [42, 43, 44, 45, 46]
TOP_N_PER_SEGMENT = 50
RANDOM_SEED   = 42

# ============================ LOAD ======================================
print("=" * 70)
print("Ooredoo Lead-Scoring v2")
print("=" * 70)

features = pd.read_csv(FEATURES_FILE)
for c in ("category", "search_category", "governorate"):
    features[c] = features[c].astype("category")

y_all = features["is_customer"].values
positive_weight_full = features["positive_weight"].values if "positive_weight" in features.columns else np.ones(len(features))
X = features.drop(columns=["is_customer", "positive_weight"], errors="ignore")

print(f"Rows           : {len(X):,}")
print(f"Features       : {X.shape[1]}")
print(f"Known positives: {int(y_all.sum()):,}  ({y_all.mean()*100:.2f}%)")

# ============================ SPLIT =====================================
pos_idx_all = np.where(y_all == 1)[0]
train_pos, holdout_pos = train_test_split(
    pos_idx_all, test_size=HOLDOUT_FRAC, random_state=RANDOM_SEED)

y_train = np.zeros(len(y_all), dtype=int)
y_train[train_pos] = 1
is_holdout = np.zeros(len(y_all), dtype=bool)
is_holdout[holdout_pos] = True

print(f"Training positives: {len(train_pos):,}")
print(f"Held-out positives: {len(holdout_pos):,}")

# =================== STEP 1: ESTIMATE CLASS PRIOR =======================
print("\n[1/6] Estimating class prior via spy technique...")
pi_hat, c_hat, spy_threshold = estimate_class_prior_spy(
    X, y_train, spy_ratio=0.15, threshold_quantile=0.05, random_state=RANDOM_SEED)
print(f"  Estimated pi_hat (fraction of unlabeled that are actually positive): {pi_hat:.4f}")
print(f"  Estimated c_hat  (P(labeled | positive)):                            {c_hat:.4f}")
print(f"  Spy-derived threshold for reliable negatives:                        {spy_threshold:.4f}")

# =================== STEP 2: BAGGING-PU (original) ======================
print("\n[2/6] Method A: Bagging-PU (original)")
scores_bagging = bagging_pu(X, y_train, n_iterations=N_ITERATIONS,
                             negative_multiple=NEG_MULTIPLE,
                             positive_weight=positive_weight_full,
                             random_state=RANDOM_SEED)

# =================== STEP 3: PRIOR-CORRECTED BAGGING-PU =================
print("\n[3/6] Method B: Prior-corrected Bagging-PU (nnPU spirit)")
scores_prior = prior_corrected_bagging_pu(
    X, y_train, pi_hat=pi_hat, n_iterations=N_ITERATIONS,
    negative_multiple=NEG_MULTIPLE, positive_weight=positive_weight_full,
    random_state=RANDOM_SEED)

# =================== STEP 4: TWO-STEP PU ================================
print("\n[4/6] Method C: Two-step PU (reliable negatives + retrain)")
reliable_neg, base_scores = extract_reliable_negatives(
    X, y_train, threshold=spy_threshold, random_state=RANDOM_SEED)
print(f"  Reliable negatives extracted: {len(reliable_neg):,}")
scores_twostep = retrain_with_reliable_negatives(
    X, y_train, reliable_neg, positive_weight=positive_weight_full,
    random_state=RANDOM_SEED)

# =================== STEP 5: ENSEMBLE ===================================
print("\n[5/6] Rank-based ensemble of methods A, B, C")
scores_ensemble = ensemble_scores(
    [scores_bagging, scores_prior, scores_twostep], weights=[1.0, 1.0, 1.0])

# =================== STEP 6: CALIBRATION ================================
print("\n[6/6] Calibrating ensemble scores via isotonic regression")
calibrated, iso = calibrate_scores_isotonic(
    scores_ensemble, holdout_positive_idx=holdout_pos, reliable_neg_idx=reliable_neg)

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
comparison_rows = []
for name, s in methods.items():
    df_m = precision_recall_at_k(s, is_holdout, K_VALUES)
    df_m["method"] = name
    comparison_rows.append(df_m)
comparison = pd.concat(comparison_rows, ignore_index=True)
pivot = comparison.pivot(index="k", columns="method", values="recall@k").round(3)
print("\nRecall@k per method:")
print(pivot.to_string())
comparison.to_csv("method_comparison_v2.csv", index=False)

# ================ PLOTS =================================================
print("\nPlotting recall curve...")
ks_smooth = list(range(50, 6001, 50))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, s in methods.items():
    curve = precision_recall_at_k(s, is_holdout, ks_smooth)
    axes[0].plot(ks_smooth, curve["recall@k"], label=name)
axes[0].set_xlabel("k (top-ranked prospects)")
axes[0].set_ylabel("Recall@k on held-out positives")
axes[0].set_title("v2 method comparison")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Score distribution
rng = np.random.RandomState(RANDOM_SEED)
sample_unl = rng.choice(np.where(~is_holdout & (y_train == 0))[0], size=5000, replace=False)
axes[1].hist(calibrated[sample_unl], bins=40, alpha=0.5, label="unlabeled sample", density=True)
axes[1].hist(calibrated[is_holdout], bins=40, alpha=0.7, label="held-out positives", density=True)
axes[1].set_xlabel("Calibrated probability")
axes[1].set_ylabel("density")
axes[1].set_title("Calibrated score distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_v2.png", dpi=120, bbox_inches="tight")
print("Saved: evaluation_v2.png")

# ================ STABILITY (multi-seed) ================================
print("\n" + "=" * 70)
print(f"Stability check: running ensemble across seeds {SEEDS}")
print("=" * 70)

def _one_run(random_state, X, y_train, pi_hat, spy_threshold, positive_weight_full):
    s_bag = bagging_pu(X, y_train, n_iterations=N_ITERATIONS,
                       negative_multiple=NEG_MULTIPLE,
                       positive_weight=positive_weight_full,
                       random_state=random_state, verbose=False)
    s_pri = prior_corrected_bagging_pu(
        X, y_train, pi_hat=pi_hat, n_iterations=N_ITERATIONS,
        negative_multiple=NEG_MULTIPLE, positive_weight=positive_weight_full,
        random_state=random_state, verbose=False)
    rn, _ = extract_reliable_negatives(X, y_train, threshold=spy_threshold,
                                        random_state=random_state)
    s_two = retrain_with_reliable_negatives(X, y_train, rn,
                                             positive_weight=positive_weight_full,
                                             random_state=random_state)
    return ensemble_scores([s_bag, s_pri, s_two])

mean_scores, std_scores, all_scores = stability_across_seeds(
    _one_run, seeds=SEEDS,
    X=X, y_train=y_train, pi_hat=pi_hat, spy_threshold=spy_threshold,
    positive_weight_full=positive_weight_full,
)

# Recall@k mean & std over seeds
stab_rows = []
for k in K_VALUES:
    recalls = [precision_recall_at_k(all_scores[i], is_holdout, [k]).iloc[0]["recall@k"]
               for i in range(len(SEEDS))]
    stab_rows.append({"k": k, "recall_mean": np.mean(recalls),
                      "recall_std": np.std(recalls)})
stability = pd.DataFrame(stab_rows)
print("\nRecall@k stability (mean +/- std across seeds):")
print(stability.round(4).to_string(index=False))
stability.to_csv("stability_v2.csv", index=False)

# ================ FINAL DELIVERABLE =====================================
print("\n" + "=" * 70)
print("Final deliverable: ranked_prospects_v2.csv")
print("=" * 70)

maps = pd.read_csv(MAPS_FILE)
assert len(maps) == len(calibrated), "row count mismatch"

output = maps.copy()
output["pu_score_bagging"]      = scores_bagging
output["pu_score_prior_corr"]   = scores_prior
output["pu_score_twostep"]      = scores_twostep
output["pu_score_ensemble"]     = scores_ensemble
output["prob_calibrated"]       = calibrated
output["prob_stability_std"]    = std_scores
output["is_known_customer"]     = output["is_customer"]
output = output.drop(columns=["is_customer"])
output = output.sort_values("prob_calibrated", ascending=False).reset_index(drop=True)
output.insert(0, "rank", np.arange(1, len(output) + 1))
output.to_csv("ranked_prospects_v2.csv", index=False)

# ---- Segmented output: top-N per governorate, per search_category ------
non_customers = output[output["is_known_customer"] == 0].copy()

top_gov = (non_customers.sort_values("prob_calibrated", ascending=False)
           .groupby("governorate", observed=True).head(TOP_N_PER_SEGMENT))
top_gov.to_csv("top_by_governorate.csv", index=False)

top_scat = (non_customers.sort_values("prob_calibrated", ascending=False)
            .groupby("search_category", observed=True).head(TOP_N_PER_SEGMENT))
top_scat.to_csv("top_by_search_category.csv", index=False)

print(f"\nSaved:")
print(f"  ranked_prospects_v2.csv    ({len(output):,} rows)")
print(f"  top_by_governorate.csv     (top {TOP_N_PER_SEGMENT} per governorate)")
print(f"  top_by_search_category.csv (top {TOP_N_PER_SEGMENT} per search_category)")
print(f"  method_comparison_v2.csv   (recall@k for all methods)")
print(f"  stability_v2.csv           (multi-seed recall stability)")
print(f"  evaluation_v2.png          (curves + score histograms)")

print("\n=== Top 10 non-customer prospects overall ===")
top10 = non_customers.head(10)
print(top10[["rank", "name", "governorate", "category",
             "prob_calibrated", "prob_stability_std"]].to_string(index=False))

print("\nDone.")
