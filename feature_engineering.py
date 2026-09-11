"""
feature_engineering.py -- Step 2 of the Ooredoo PU-learning lead-scoring pipeline.

Reads maps_labeled.csv (from entity_linkage.py) and produces features.csv:
a model-ready feature table for the downstream Bagging-PU stage.

Enhancements over the first version:
  - Competitor density  (same-category businesses within 2 km, via BallTree)
  - Governorate-level and category-level frequency features
  - Website / phone / email quality signals (e.g. .tn domain, +216 prefix)
  - Rating kept as NaN + rating_missing flag  (not filled with 0)
  - Categoricals cast to pandas 'category' dtype (LightGBM handles natively)

Run:
    pip install pandas numpy scikit-learn
    python feature_engineering.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

INPUT_FILE  = "maps_labeled.csv"
OUTPUT_FILE = "features.csv"

# competitor-density radius (metres). 2 km is a walking-distance-ish local market.
COMPETITOR_RADIUS_M = 2000
EARTH_RADIUS_M      = 6_371_000


# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Positives (is_customer=1): {int(df['is_customer'].sum()):,}  "
      f"({df['is_customer'].mean()*100:.2f}% base rate)\n")


# ============================================================
# 2. NUMERIC + POPULARITY
# ============================================================
for c in ("rating", "reviews", "latitude", "longitude"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Rating: keep NaN (a missing rating is meaningful) + add a flag.
df["rating_missing"] = df["rating"].isna().astype(int)

# Reviews: 0 IS a natural value (no reviews yet), so filling with 0 is fine.
df["reviews"]     = df["reviews"].fillna(0)
df["log_reviews"] = np.log1p(df["reviews"])
df["has_reviews"] = (df["reviews"] > 0).astype(int)


# ============================================================
# 3. DIGITAL PRESENCE (with quality signals)
# ============================================================
df["phone"]   = df["phone"].fillna("").astype(str)
df["website"] = df["website"].fillna("").astype(str)
df["email"]   = df["email"].fillna("").astype(str)

df["has_phone"]   = (df["phone"]   != "").astype(int)
df["has_website"] = (df["website"] != "").astype(int)
df["has_email"]   = (df["email"]   != "").astype(int)

# Quality signals: not just presence, but what KIND of presence.
df["phone_is_tn"]        = df["phone"].str.contains(r"\+?216",   regex=True).astype(int)
df["website_is_tn"]      = df["website"].str.contains(r"\.tn(?:/|$)", regex=True).astype(int)
df["website_is_social"]  = df["website"].str.contains(
    r"facebook\.com|instagram\.com|linkedin\.com", regex=True, case=False
).astype(int)
df["email_count"]        = df["email"].apply(
    lambda s: 0 if not s else len([e for e in s.split(";") if e.strip()])
)

df["digital_presence_count"] = (
    df["has_phone"] + df["has_website"] + df["has_email"]
)


# ============================================================
# 4. TEXT SHAPE
# ============================================================
df["name"]    = df["name"].fillna("").astype(str)
df["address"] = df["address"].fillna("").astype(str)

df["name_length"]        = df["name"].str.len()
df["address_length"]     = df["address"].str.len()
df["name_word_count"]    = df["name"].str.split().str.len().fillna(0).astype(int)
df["address_word_count"] = df["address"].str.split().str.len().fillna(0).astype(int)


# ============================================================
# 5. CATEGORICALS
# ============================================================
for c in ("category", "search_category", "governorate"):
    df[c] = df[c].fillna("Unknown").astype(str)


# ============================================================
# 6. FREQUENCY FEATURES  (governorate and category density)
# ============================================================
# How many businesses share this row's governorate / category?
# A crude but useful "market thickness" signal, computed on the whole scrape.
gov_counts  = df["governorate"].value_counts()
cat_counts  = df["category"].value_counts()
scat_counts = df["search_category"].value_counts()
df["gov_size"]              = df["governorate"].map(gov_counts).astype(int)
df["category_size"]         = df["category"].map(cat_counts).astype(int)
df["search_category_size"]  = df["search_category"].map(scat_counts).astype(int)


# ============================================================
# 7. LOCAL COMPETITOR DENSITY  (spatial, via BallTree + haversine)
# ============================================================
# For each business with coordinates, count how many businesses of the SAME
# search_category sit within COMPETITOR_RADIUS_M. Uses a BallTree with the
# haversine metric -- O(n log n), so ~27k rows finishes in seconds.

df["competitor_density"] = 0
has_coords = df["latitude"].notna() & df["longitude"].notna()
print(f"Computing competitor density for {has_coords.sum():,} rows with coordinates...")

for scat, sub in df[has_coords].groupby("search_category"):
    if len(sub) < 2:
        continue
    coords_rad = np.radians(sub[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = COMPETITOR_RADIUS_M / EARTH_RADIUS_M
    # query_radius returns arrays of neighbor indices per point (self included)
    neigh = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
    df.loc[sub.index, "competitor_density"] = neigh - 1   # exclude self


# ============================================================
# 8. TARGET
# ============================================================
df["is_customer"] = pd.to_numeric(
    df["is_customer"], errors="coerce"
).fillna(0).astype(int)


# ============================================================
# 9. SELECT + CAST + SAVE
# ============================================================
feature_columns = [
    # Geographic
    "latitude", "longitude", "governorate",
    # Business classification
    "category", "search_category",
    # Density / market context
    "gov_size", "category_size", "search_category_size", "competitor_density",
    # Popularity
    "rating", "rating_missing",
    "reviews", "log_reviews", "has_reviews",
    # Digital presence
    "has_phone", "has_website", "has_email", "digital_presence_count",
    "phone_is_tn", "website_is_tn", "website_is_social", "email_count",
    # Text shape
    "name_length", "address_length", "name_word_count", "address_word_count",
    # Target
    "is_customer",
]
features = df[feature_columns].copy()

# Cast categoricals -- LightGBM reads this dtype natively (no encoding needed).
for c in ("category", "search_category", "governorate"):
    features[c] = features[c].astype("category")

features.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n================ FEATURE ENGINEERING ================")
print(f"Output shape : {features.shape}")
print(f"Features     : {len(feature_columns) - 1}")
print(f"Positive     : {int(features['is_customer'].sum()):,}")
print(f"Negative     : {int((features['is_customer'] == 0).sum()):,}")
print(f"\nCompetitor density: min={features['competitor_density'].min()}, "
      f"median={int(features['competitor_density'].median())}, "
      f"max={features['competitor_density'].max()}")
print(f"Rating missing rate: {features['rating_missing'].mean()*100:.1f}%")

# Quick sanity check: are positives distributed differently from negatives on
# a couple of features? If yes, there IS signal in the data for PU to pick up.
print("\n=== Sanity check: positive vs negative means ===")
comp = features.groupby("is_customer")[
    ["rating", "log_reviews", "digital_presence_count",
     "competitor_density", "has_website"]
].mean().round(3)
print(comp.T)

print(f"\nSaved to: {OUTPUT_FILE}")
