"""
feature_engineering_v2.py -- Enhanced Step 2 of the pipeline.

Adds on top of v1:
  - is_legal_entity: name contains SARL / SA / Ets / Groupe / etc.
    (Directly replaces the 'name_length as formal-business proxy' pattern.)
  - Vertical keyword flags (bilingual FR/EN): medical, industrial, transport,
    IT/informatique, restaurant/cafe, hotel, retail, education, construction,
    services -- cheap semantic features that don't need embeddings
  - Address structural flags: zone_industrielle, has_immeuble, has_avenue_rue
  - Character composition of name: has_arabic, has_digits, is_upper
  - Geographic proximity to KNOWN CUSTOMERS: nearest-customer distance +
    count within 5 km.  This is the strongest new signal.
  - Rating x reviews interaction (established business)
  - Match-confidence-aware positive weighting (if match_confidence column
    present, we can down-weight borderline matches later)

Reads:  maps_labeled_v2.csv (or falls back to maps_labeled.csv)
Writes: features_v2.csv
"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

INPUT_FILE  = "maps_labeled_v2.csv" if os.path.exists("maps_labeled_v2.csv") else "maps_labeled.csv"
OUTPUT_FILE = "features_v2.csv"

COMPETITOR_RADIUS_M = 2000
CUSTOMER_RADIUS_M   = 5000
EARTH_RADIUS_M      = 6_371_000

# ---- vocabularies --------------------------------------------------------
LEGAL_FORMS = [
    r"\bsarl\b", r"\bsuarl\b", r"\bs\.?a\.?r\.?l\.?\b",
    r"\bs\.?a\.?\b", r"\bspa\b", r"\bsnc\b", r"\bscs\b",
    r"\bets\b", r"\betablissement", r"\bentreprise",
    r"\bgroupe\b", r"\bsociete\b", r"\bste\b", r"\bcompany\b",
    r"\bllc\b", r"\bltd\b", r"\bgmbh\b", r"\binc\b",
]
LEGAL_RE = re.compile("|".join(LEGAL_FORMS), re.IGNORECASE)

VERTICAL_KEYWORDS = {
    "medical":       r"clinic|clinique|medic|pharmac|dentaire|dentist|hospital|hopital|labo|laborat|imagerie|radiolog|cabinet.*medical",
    "industrial":    r"industri|usine|fabric|manufactur|metallurg|plastique|chimique|textile",
    "transport":     r"transport|logistic|logisti|cargo|freight|shipping|expedition|livraison|taxi",
    "it":            r"informatique|software|technolog|digital|numerique|systeme|reseau|network|it\s|solution",
    "restaurant":    r"restaurant|cafe|coffee|resto|pizza|snack|patisser|boulanger",
    "hotel":         r"hotel|motel|hostel|residence|dar\b|auberge|villa\b|resort",
    "retail":        r"boutique|magasin|shop|store|supermarch|epicer|commerce",
    "education":     r"ecole|school|institut|universit|college|formation|academy|academie",
    "construction":  r"construction|batiment|immobilier|architec|travaux|entreprise.*btp|beton",
    "auto":          r"auto|garage|vehicule|mecanique|carrosser|pieces.*auto",
    "finance":       r"banque|bank|assurance|insurance|finance|comptable|expert",
    "services":      r"service|consulting|conseil|agence|bureau",
}
VERTICAL_RES = {k: re.compile(v, re.IGNORECASE) for k, v in VERTICAL_KEYWORDS.items()}

ADDRESS_FLAGS = {
    "zone_industrielle": r"zone\s+industriel|z\.i\.|parc\s+industriel|parc\s+d.?activit",
    "has_immeuble":      r"immeuble|residence|complexe|centre\s+commercial",
    "has_avenue_rue":    r"\bavenue\b|\bav\.?\b|\brue\b|\bboulevard\b|\bbd\.?\b|\broute\b",
    "has_floor":         r"etage|floor|\bappart|\bapt\b",
}
ADDRESS_RES = {k: re.compile(v, re.IGNORECASE) for k, v in ADDRESS_FLAGS.items()}

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
DIGIT_RE  = re.compile(r"\d")


# ============================================================
# 1. LOAD
# ============================================================
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {INPUT_FILE}: {df.shape[0]:,} rows, {df.shape[1]} cols")
print(f"Positives: {int(df['is_customer'].sum()):,} "
      f"({df['is_customer'].mean()*100:.2f}% base rate)\n")

# ============================================================
# 2. NUMERIC + POPULARITY (unchanged from v1)
# ============================================================
for c in ("rating", "reviews", "latitude", "longitude"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["rating_missing"] = df["rating"].isna().astype(int)
df["reviews"]        = df["reviews"].fillna(0)
df["log_reviews"]    = np.log1p(df["reviews"])
df["has_reviews"]    = (df["reviews"] > 0).astype(int)

# Interaction: established = high rating AND many reviews
df["rating_x_logreviews"] = df["rating"].fillna(0) * df["log_reviews"]

# ============================================================
# 3. DIGITAL PRESENCE (unchanged from v1)
# ============================================================
df["phone"]   = df["phone"].fillna("").astype(str)
df["website"] = df["website"].fillna("").astype(str)
df["email"]   = df["email"].fillna("").astype(str)

df["has_phone"]   = (df["phone"]   != "").astype(int)
df["has_website"] = (df["website"] != "").astype(int)
df["has_email"]   = (df["email"]   != "").astype(int)
df["phone_is_tn"]       = df["phone"].str.contains(r"\+?216",   regex=True).astype(int)
df["website_is_tn"]     = df["website"].str.contains(r"\.tn(?:/|$)", regex=True).astype(int)
df["website_is_social"] = df["website"].str.contains(
    r"facebook\.com|instagram\.com|linkedin\.com", regex=True, case=False).astype(int)
df["email_count"] = df["email"].apply(
    lambda s: 0 if not s else len([e for e in s.split(";") if e.strip()]))
df["digital_presence_count"] = df["has_phone"] + df["has_website"] + df["has_email"]

# ============================================================
# 4. NAME + ADDRESS SEMANTICS  (this is the big v2 addition)
# ============================================================
df["name"]    = df["name"].fillna("").astype(str)
df["address"] = df["address"].fillna("").astype(str)

# --- Formal entity flag: DIRECTLY replaces the name_length proxy ---
df["is_legal_entity"] = df["name"].apply(lambda s: int(bool(LEGAL_RE.search(s))))

# --- Character composition ---
df["name_has_arabic"] = df["name"].apply(lambda s: int(bool(ARABIC_RE.search(s))))
df["name_has_digits"] = df["name"].apply(lambda s: int(bool(DIGIT_RE.search(s))))
df["name_is_upper"]   = df["name"].apply(
    lambda s: int(len(s) > 3 and s.upper() == s and any(c.isalpha() for c in s)))

# --- Shape (kept from v1, but they should matter less now) ---
df["name_length"]        = df["name"].str.len()
df["address_length"]     = df["address"].str.len()
df["name_word_count"]    = df["name"].str.split().str.len().fillna(0).astype(int)
df["address_word_count"] = df["address"].str.split().str.len().fillna(0).astype(int)

# --- Vertical keyword flags on name+category (concatenated) ---
combined_text = (df["name"] + " " + df["category"].fillna("") + " " +
                 df["search_category"].fillna(""))
for vert, regex in VERTICAL_RES.items():
    df[f"vert_{vert}"] = combined_text.apply(lambda s: int(bool(regex.search(s))))

# --- Address structural flags ---
for flag, regex in ADDRESS_RES.items():
    df[f"addr_{flag}"] = df["address"].apply(lambda s: int(bool(regex.search(s))))

# ============================================================
# 5. CATEGORICALS + FREQUENCY (unchanged from v1)
# ============================================================
for c in ("category", "search_category", "governorate"):
    df[c] = df[c].fillna("Unknown").astype(str)

df["gov_size"]             = df["governorate"].map(df["governorate"].value_counts()).astype(int)
df["category_size"]        = df["category"].map(df["category"].value_counts()).astype(int)
df["search_category_size"] = df["search_category"].map(df["search_category"].value_counts()).astype(int)

# ============================================================
# 6. COMPETITOR DENSITY (same-category within 2 km) -- from v1
# ============================================================
df["competitor_density"] = 0
has_coords = df["latitude"].notna() & df["longitude"].notna()
print(f"Computing competitor density for {has_coords.sum():,} rows with coords...")
for scat, sub in df[has_coords].groupby("search_category"):
    if len(sub) < 2:
        continue
    coords = np.radians(sub[["latitude", "longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    r = COMPETITOR_RADIUS_M / EARTH_RADIUS_M
    df.loc[sub.index, "competitor_density"] = tree.query_radius(coords, r=r, count_only=True) - 1

# ============================================================
# 7. GEOGRAPHIC PROXIMITY TO KNOWN CUSTOMERS  (new, strong signal)
# ============================================================
# Intuition: prospects near existing customers are typically in the same
# business cluster (zones industrielles, business districts) and share the
# environmental signals that drove existing customers to us.
#
# We compute two features:
#   - customer_nearest_km : distance to closest known customer (log-transformed)
#   - customers_within_5km: count of known customers within 5 km
#
# CAUTION: This feature LEAKS if we're not careful. For a known customer,
# their nearest-customer distance is trivially small (they're a customer
# themselves). We must exclude self-matches. Done below with query_radius
# then filtering; and for nearest we use k=2 and take the second-nearest
# ONLY for known customers.
df["customer_nearest_km"]  = np.nan
df["customers_within_5km"] = 0

pos_mask = (df["is_customer"] == 1) & has_coords
if pos_mask.sum() >= 2:
    pos_coords = np.radians(df.loc[pos_mask, ["latitude", "longitude"]].values)
    tree = BallTree(pos_coords, metric="haversine")
    all_coords = np.radians(df.loc[has_coords, ["latitude", "longitude"]].values)

    # Nearest-neighbor distance
    # For positives: use k=2 and take second (to skip self-match).
    # For unlabeled: use k=1.
    dist_k2, idx_k2 = tree.query(all_coords, k=min(2, pos_mask.sum()))
    nearest = np.where(
        df.loc[has_coords, "is_customer"].values == 1,
        dist_k2[:, 1] if dist_k2.shape[1] > 1 else dist_k2[:, 0],
        dist_k2[:, 0]
    )
    df.loc[has_coords, "customer_nearest_km"] = nearest * EARTH_RADIUS_M / 1000

    # Within-5km count (subtract 1 for self if positive)
    r = CUSTOMER_RADIUS_M / EARTH_RADIUS_M
    counts = tree.query_radius(all_coords, r=r, count_only=True)
    is_pos_arr = (df.loc[has_coords, "is_customer"] == 1).values
    df.loc[has_coords, "customers_within_5km"] = counts - is_pos_arr.astype(int)

df["customer_nearest_km"] = df["customer_nearest_km"].fillna(df["customer_nearest_km"].median())
df["log_customer_nearest_km"] = np.log1p(df["customer_nearest_km"])
df["has_customer_within_5km"] = (df["customers_within_5km"] > 0).astype(int)

# ============================================================
# 8. TARGET + MATCH CONFIDENCE
# ============================================================
df["is_customer"] = pd.to_numeric(df["is_customer"], errors="coerce").fillna(0).astype(int)

# Optional: sample weight for training. Exact/high matches trusted, borderline
# matches down-weighted. Used later by pu_methods.
if "match_confidence" in df.columns:
    conf_weight = {"exact": 1.0, "high": 1.0, "borderline": 0.5, "": 1.0}
    df["positive_weight"] = df["match_confidence"].fillna("").map(conf_weight).fillna(1.0)
else:
    df["positive_weight"] = 1.0

# ============================================================
# 9. SELECT + CAST + SAVE
# ============================================================
feature_columns = [
    # Geographic
    "latitude", "longitude", "governorate",
    # Business classification
    "category", "search_category",
    # Density / market context
    "gov_size", "category_size", "search_category_size",
    "competitor_density",
    # Proximity to customers  (NEW)
    "customer_nearest_km", "log_customer_nearest_km",
    "customers_within_5km", "has_customer_within_5km",
    # Popularity
    "rating", "rating_missing", "reviews", "log_reviews", "has_reviews",
    "rating_x_logreviews",
    # Digital presence
    "has_phone", "has_website", "has_email", "digital_presence_count",
    "phone_is_tn", "website_is_tn", "website_is_social", "email_count",
    # Formality / name shape  (NEW)
    "is_legal_entity", "name_has_arabic", "name_has_digits", "name_is_upper",
    "name_length", "address_length", "name_word_count", "address_word_count",
    # Vertical flags  (NEW)
    *[f"vert_{v}" for v in VERTICAL_KEYWORDS],
    # Address structure  (NEW)
    *[f"addr_{a}" for a in ADDRESS_FLAGS],
    # Weight + target
    "positive_weight", "is_customer",
]

features = df[feature_columns].copy()
for c in ("category", "search_category", "governorate"):
    features[c] = features[c].astype("category")

features.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n================ FEATURE ENGINEERING v2 ================")
print(f"Output shape: {features.shape}")
print(f"Features: {len(feature_columns) - 2}   (excludes weight & target)")
print(f"Positives: {int(features['is_customer'].sum()):,}\n")

print("=== Positive vs negative means on key v2 features ===")
key = ["is_legal_entity", "customer_nearest_km", "customers_within_5km",
       "has_customer_within_5km", "vert_industrial", "vert_medical",
       "vert_it", "addr_zone_industrielle", "rating_x_logreviews"]
print(features.groupby("is_customer")[key].mean().round(3).T)

print(f"\nSaved to: {OUTPUT_FILE}")
