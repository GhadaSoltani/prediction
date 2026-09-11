# B2B Lead Scoring via Positive-Unlabeled Learning

Predicting which Tunisian businesses on Google Maps are likely to become B2B telecom customers, given only a small set of known customers and no confirmed non-customers.

This is a research/thesis project applying **Positive-Unlabeled (PU) learning** to a real CRM problem.

---

## Problem

We have three data sources:

- **~500 known B2B customers** (from Ooredoo's purchase records)
- **~28,000 Tunisian businesses** scraped from Google Maps
- **No labeled non-customers** — a business we haven't sold to might be a great prospect nobody has called yet

Standard classification doesn't work here: treating unlabeled businesses as "negative" teaches the model that customer-like patterns are bad. This is the **Positive-Unlabeled** learning setting, which requires specialised methods.

The goal: **rank all 28,000 businesses by conversion likelihood**, so sales can work top-down instead of cold-calling at random.

---

## Pipeline

Four stages, three iterations (v1 → v3):

```
purchases_1.xlsx ─┐
purchases_2.xlsx ─┼─► [1. Entity Linkage] ─► maps_labeled.csv, match_review.csv
maps.csv         ─┘         │
                            ▼
                 [2. Feature Engineering] ─► features.csv
                            │
                            ▼
                    [3. PU Learning] ─► ranked_prospects.csv
                            │             top_by_governorate.csv
                            │             top_by_search_category.csv
                            ▼
                    [4. Validation] ─► validation_buckets.csv (planned)
```

### Version history

| Version | What changed |
|---|---|
| **v1** | Baseline: fuzzy name matching, 26 features, Bagging-PU only |
| **v2** | Two-pass matching (exact + fuzzy) with ambiguity detection; 51 features including `is_legal_entity`, 12 vertical keyword flags, address structural flags, geographic proximity to known customers; three PU methods (Bagging-PU, prior-corrected, two-step) + rank ensemble + isotonic calibration |
| **v3** | Multi-axis entity linkage (phone/website/RNE in addition to name); robust class prior estimation (Elkan-Noto e1 + TIcE) replacing the unreliable spy heuristic |

---

## Repository structure

### Code

| File | Stage | Role |
|---|---|---|
| `entity_linkage.py` | 1 | v1 — single-pass fuzzy name matching |
| `entity_linkage_v2.py` | 1 | v2 — two-pass, ambiguity detection, confidence buckets |
| `entity_linkage_v3.py` | 1 | v3 — adds phone / website / RNE axes with graceful fallback |
| `feature_engineering.py` | 2 | v1 — 26 features |
| `feature_engineering_v2.py` | 2 | v2 — 51 features including legal-entity flag, verticals, proximity |
| `pu_methods.py` | 3 | Library: Bagging-PU, prior-corrected, two-step, ensemble, calibration |
| `prior_estimation.py` | 3 | Library: robust class prior estimators (holdout + TIcE) |
| `lead_scoring.ipynb` | 3 | v1 notebook |
| `lead_scoring_v2.py` | 3 | v2 orchestration |
| `lead_scoring_v3.py` | 3 | v3 orchestration using robust prior estimators |
| `pu_experiments.ipynb` | — | Method comparison / ablation notebook |
| `validation.py` | 4 | Match top predictions against held-out Ooredoo DB (needs full DB) |

### Results (committed)

| File | Description |
|---|---|
| `method_comparison.csv` / `_v2.csv` / `_v3.csv` | Recall@k per method per version |
| `stability_results.csv` / `stability_v2.csv` / `stability_v3.csv` | Multi-seed stability of the ensemble |
| `prior_comparison.csv` | Comparison of spy vs holdout vs TIcE prior estimators |
| `evaluation.png` / `_v2.png` / `_v3.png` | Recall curves + score distributions |
| `feature_importance.png` | SHAP feature importance |
| `method_comparison.png` | Bagging-PU vs Elkan-Noto comparison plot |
| `stability_curve.png` | Recall@k ± std across seeds |

### Not committed (data / regenerable, see `.gitignore`)

- Input: `purchases_1.xlsx`, `purchases_2.xlsx`, `maps.csv`
- Intermediate: `maps_labeled*.csv`, `features*.csv`, `match_review*.csv`
- Final delivery: `ranked_prospects*.csv`, `top_by_governorate*.csv`, `top_by_search_category*.csv`

---

## How to run

### Prerequisites

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install directly:

```bash
pip install pandas numpy scikit-learn lightgbm rapidfuzz unidecode tqdm openpyxl matplotlib scipy shap
```

### Input files

Place these in the project root (they're not in the repo):

```
purchases_1.xlsx      # Ooredoo B2B customer export (must have account_contact_name column)
purchases_2.xlsx      # Second customer export
maps.csv              # Google Maps scrape (must have: name, category, search_category,
                      # governorate, latitude, longitude, phone, website, email,
                      # address, rating, reviews)
```

### Recommended: run the v3 pipeline

```bash
# Stage 1 — Entity linkage (~2 min)
python entity_linkage_v3.py
#   -> maps_labeled_v3.csv, match_review_v3.csv

# Stage 2 — Feature engineering (~1 min)
# Note: feature_engineering_v2.py reads maps_labeled_v3.csv if present
python feature_engineering_v2.py
#   -> features_v2.csv

# Stage 3 — PU learning + ranking (~10-15 min)
python lead_scoring_v3.py
#   -> ranked_prospects_v3.csv
#   -> top_by_governorate_v3.csv
#   -> top_by_search_category_v3.csv
#   -> method_comparison_v3.csv, stability_v3.csv, prior_comparison.csv
#   -> evaluation_v3.png

# Stage 4 — External validation (planned; needs full Ooredoo DB)
python validation.py
#   -> validation_buckets.csv, validation_hits.csv, validation_curve.png
```

### To run the older v1 or v2 for comparison

```bash
python entity_linkage.py && python feature_engineering.py && jupyter nbconvert --to notebook --execute lead_scoring.ipynb
```

or

```bash
python entity_linkage_v2.py && python feature_engineering_v2.py && python lead_scoring_v2.py
```

### Configuration knobs

Top of `entity_linkage_v3.py`:

- `PURCHASE_PHONE_COL` — set to your phone column name when available (currently `None` — phone matching skipped)
- `PURCHASE_WEBSITE_COL` — same, for website matching
- `PURCHASE_RNE_COL` — currently `"RNE"`; RNE matching activates when Maps side is enriched

Top of `lead_scoring_v3.py`:

- `N_ITERATIONS` — Bagging-PU rounds (default 50; try 100 for smoother scores)
- `NEG_MULTIPLE` — pseudo-negatives per positive per round (default 3)
- `HOLDOUT_FRAC` — fraction of positives held out for evaluation (default 0.20)
- `SEEDS` — random seeds for stability run (default `[42..46]`)

---

## Key results

On the current data (~625 positives after v2 entity linkage, ~27,854 total businesses):

| Metric | v1 | v2 | v3 |
|---|---:|---:|---:|
| Positive count | 595 | 625 | 625+ (depends on axes enabled) |
| Recall@2000 (7% of list) | ~30% | **38%** | 38%+ |
| Recall@5000 (18% of list) | ~50% | **62%** | 62%+ |
| Stability std @ k=2000 | ~2% | 1.2% | 1.2% |
| Class prior estimation | — | Spy (unreliable, gave π̂=0.77) | Holdout + TIcE (robust) |

**Business translation:** working the top 2,000 predictions (7% of the total list) reaches roughly **38% of real prospects** — a **~5× lift** over random calling.

---

## Method summary

**Entity linkage.** Fuzzy-match customer names against Google Maps names using `rapidfuzz` with `token_set_ratio`, after normalization (unaccent, drop legal suffixes like SARL/SA/Ets, drop embedded codes). v3 adds phone-number and website-domain matching as more reliable axes.

**Feature engineering.** Combines geographic, categorical, market-density, popularity, digital-presence, and semantic features. v2's most impactful additions:

- `is_legal_entity` — explicit flag for formal registered businesses
- `vert_*` — 12 keyword-based vertical flags (industrial, medical, IT, restaurant, ...)
- `customer_nearest_km`, `customers_within_5km` — geographic proximity to known customers

**PU learning.** Three complementary methods, then ensembled by rank average:

- **A. Bagging-PU** (Mordelet & Vert 2014) — 50 rounds of sampling pseudo-negatives from the unlabeled pool; each business is scored only when it was NOT used as a pseudo-negative (out-of-bag scoring).
- **B. Prior-corrected Bagging-PU** — same as A but pseudo-negatives are weighted by `(1 − π̂)` to account for hidden positives (nnPU spirit, Kiryo et al. 2017).
- **C. Two-step PU** — extract "reliable negatives" (unlabeled points scoring very low under a naive classifier), then train a standard classifier on positives vs reliable negatives.

**Calibration.** Isotonic regression fit on held-out positives (label 1) and reliable negatives (label 0) turns ranks into interpretable probabilities.

**Validation.** Match top-scored non-customers against a held-out data source (full Ooredoo database, not just active B2B accounts). Hit-rate should decrease monotonically from top decile to bottom decile — that's causal evidence the ranking is meaningful.

---

## References

- Elkan, C., & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. *KDD*.
- Mordelet, F., & Vert, J.-P. (2014). A bagging SVM to learn from positive and unlabeled examples. *Pattern Recognition Letters*.
- Liu, B., Dai, Y., Li, X., Lee, W. S., & Yu, P. S. (2003). Building text classifiers using positive and unlabeled examples. *ICDM*.
- Bekker, J., & Davis, J. (2018). Estimating the class prior in positive and unlabeled data through decision tree induction. *AAAI*.
- Kiryo, R., Niu, G., du Plessis, M. C., & Sugiyama, M. (2017). Positive-unlabeled learning with non-negative risk estimator. *NeurIPS*.
- Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: a survey. *Machine Learning*.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*.

---

## Reproducibility

- Random seed: `RANDOM_SEED = 42` throughout
- Multi-seed stability: seeds `[42, 43, 44, 45, 46]`
- Environment: Python 3.10+, versions pinned in `requirements.txt`

---

## Notes and known limitations

- **SCAR assumption** — the method assumes the labeling probability `P(labeled | positive)` is constant across the feature space. If customer acquisition is biased (e.g., sales concentrated on Tunis), this may not hold.
- **Positive count is small** — 625 positives out of 27,854 businesses. Growing this via manual review of the "borderline" bucket in `match_review_v3.csv` or via enabling phone/website matching is the highest-leverage improvement.
- **Class prior estimation is difficult** — the spy heuristic gave the implausible `π̂ = 0.77` on this data; v3 replaces it with the Elkan-Noto holdout estimator + TIcE (see `prior_comparison.csv`).
- **External validation is not yet run** — requires access to the full Ooredoo database.

---

## License / Confidentiality

This repository contains research code developed as part of an internship / thesis project. The code is provided for educational and reproducibility purposes. The underlying data is confidential and is not included in this repository.
